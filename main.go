package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Config: всё настраивается из JSON-файла (пути, бинарь, таймаут и список
// моделей). Модель — это имя, алиасы (значения поля "model" в запросе),
// переменные окружения и argv llama-server.
// ---------------------------------------------------------------------------

type ModelDef struct {
	Name    string            `json:"name"`
	Aliases []string          `json:"aliases"`
	Env     map[string]string `json:"env"`
	Args    []string          `json:"args"`
	Bin     string            `json:"bin"` // опц.: бинарник этой модели (переопределяет глобальный llama_bin)
}

// bin возвращает бинарник модели: свой, если задан, иначе глобальный llama_bin.
func (m *ModelDef) bin(c *Config) string {
	if b := strings.TrimSpace(m.Bin); b != "" {
		return b
	}
	return c.LlamaBin
}

func (m *ModelDef) allNames() []string {
	out := []string{m.Name}
	for _, a := range m.Aliases {
		a = strings.TrimSpace(a)
		if a != "" && a != m.Name {
			out = append(out, a)
		}
	}
	return out
}

type Config struct {
	ListenAddr   string      `json:"listen_addr"`
	BackendAddr  string      `json:"backend_addr"`
	LlamaBin     string      `json:"llama_bin"`
	Cwd          string      `json:"cwd"`
	LoadTimeout  string      `json:"load_timeout"`
	DefaultModel string      `json:"default_model"` // модель для старта: не задано -> первая в списке, "none" -> не грузить
	NvidiaSmi    string      `json:"nvidia_smi"`    // дашборд: путь к nvidia-smi ("" -> из PATH, "none" -> не опрашивать GPU)
	BackendUI    string      `json:"backend_ui"`    // дашборд: ссылка на web-UI llama-server ("" -> http://<хост>:<порт backend_addr>/)
	Models       []*ModelDef `json:"models"`

	// resolved
	table map[string]*ModelDef
	order []*ModelDef
	loadT time.Duration
}

// newClosedChan создаёт закрытый канал: начальное значение server.idle
// («in-flight запросов нет»).
func newClosedChan() chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}

func loadConfig(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var c Config
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}
	// default-значения
	if c.ListenAddr == "" {
		c.ListenAddr = ":8080"
	}
	if c.BackendAddr == "" {
		c.BackendAddr = "127.0.0.1:8081"
	}
	if c.LlamaBin == "" {
		c.LlamaBin = "./llama.cpp/build/bin/llama-server"
	}
	if c.Cwd == "" {
		c.Cwd = "."
	}
	if c.LoadTimeout == "" {
		c.LoadTimeout = "15m"
	}
	c.loadT, err = time.ParseDuration(c.LoadTimeout)
	if err != nil {
		return nil, fmt.Errorf("parse load_timeout %q: %w", c.LoadTimeout, err)
	}
	if len(c.Models) == 0 {
		return nil, fmt.Errorf("config %s: no models defined", path)
	}
	if err := c.buildTable(); err != nil {
		return nil, err
	}
	return &c, nil
}

// buildTable строит (или перестраивает) таблицу алиасов для моделей.
func (c *Config) buildTable() error {
	c.table = make(map[string]*ModelDef, len(c.Models)*2)
	c.order = c.order[:0]
	seen := map[string]string{} // alias -> model name (для контроля дублей)
	for _, m := range c.Models {
		m.Name = strings.TrimSpace(m.Name)
		if m.Name == "" {
			return fmt.Errorf("config: model has empty name")
		}
		for _, n := range m.allNames() {
			key := strings.ToLower(n)
			if owner, dup := seen[key]; dup {
				return fmt.Errorf("config: alias %q is used by both %q and %q", n, owner, m.Name)
			}
			seen[key] = m.Name
			c.table[key] = m
		}
		c.order = append(c.order, m)
	}
	d := strings.TrimSpace(c.DefaultModel)
	if d != "" && !strings.EqualFold(d, "none") {
		if _, ok := c.table[strings.ToLower(d)]; !ok {
			return fmt.Errorf("config: default_model %q not found", c.DefaultModel)
		}
	}
	return nil
}

// startupModel возвращает модель, которую прокси грузит при старте:
// default_model, если задан; иначе первая модель из списка;
// "none" (регистр не важен) — автозагрузки нет вообще.
func (c *Config) startupModel() *ModelDef {
	d := strings.TrimSpace(c.DefaultModel)
	if strings.EqualFold(d, "none") {
		return nil
	}
	if d != "" {
		return c.resolve(d)
	}
	return c.order[0]
}

func (c *Config) resolve(name string) *ModelDef {
	return c.table[strings.ToLower(strings.TrimSpace(name))]
}

func (c *Config) knownNames() string {
	var out []string
	for _, m := range c.order {
		out = append(out, m.Name)
	}
	return strings.Join(out, ", ")
}

// backendArgs = argv модели, где любой --port/<value> из конфига заменён
// нашим (порт берётся из backend_addr).
func backendArgs(m *ModelDef, port string) []string {
	args := make([]string, 0, len(m.Args)+2)
	for i := 0; i < len(m.Args); i++ {
		if m.Args[i] == "--port" || m.Args[i] == "-port" {
			i++ // заодно выкидываем значение порта
			continue
		}
		args = append(args, m.Args[i])
	}
	args = append(args, "--port", port)
	return args
}

func (c *Config) backendPort() string {
	parts := strings.Split(c.BackendAddr, ":")
	if len(parts) == 2 && parts[1] != "" {
		return parts[1]
	}
	return "8081"
}

// ---------------------------------------------------------------------------
// running = процесс бэкенда, поднятый прокси.
// ---------------------------------------------------------------------------

type running struct {
	cmd  *exec.Cmd
	done chan struct{}
}

// server — HTTP-прокси, притворяющийся llama-server. Держит ровно один
// реальный llama-server на backend-порту.
type server struct {
	cfg       *Config
	proxy     *httputil.ReverseProxy
	startTime time.Time

	mu        sync.Mutex
	switchMu  sync.Mutex
	current   string    // ModelDef.Name, "external" или ""
	loadedDef *ModelDef // конфиг-модель, по которой поднят бэкенд
	backendID string    // id модели, который сообщает сам бэкенд
	run       *running

	// inflight = сколько запросов прямо сейчас прокидывается бэкенду
	// (идущие генерации/стримы, прокси сам эти коннекты держит и знает,
	// когда они закончились). idle замыкается, когда inflight становится 0.
	inflight int
	idle     chan struct{}

	// Состояние для дашборда (под mu).
	phase       string    // "", "draining", "loading", "unloading"
	phaseTarget string    // модель, на которую идёт свитч
	phaseSince  time.Time //
	loadedAt    time.Time // когда текущая модель стала готова
	lastErr     string
	lastErrAt   time.Time
	queued      map[string]int // модель -> запросов, ждущих в прокси (свитч/очередь)
	forceCh     chan struct{}  // закрытие = перестать ждать in-flight при свитче
	adminBusy   bool           // идёт свитч/выгрузка, запущенные с дашборда

	mon  *monitor
	logs *logTail
	reqs *reqLog
}

var (
	configFlag = flag.String("config", "config.json", "path to the JSON config file")
	checkFlag  = flag.Bool("check", false, "validate the config and print resolved models, then exit")
)

// ---------------------------------------------------------------------------
// Управление бэкендом.
// ---------------------------------------------------------------------------

// cfgSnapshot атомарно снимает ссылку на текущий конфиг (для SIGHUP-reload).
func (s *server) cfgSnapshot() *Config {
	s.mu.Lock()
	c := s.cfg
	s.mu.Unlock()
	return c
}

func (s *server) stopCurrent() {
	s.mu.Lock()
	cr := s.run
	s.run = nil
	s.current = ""
	s.loadedDef = nil
	s.backendID = ""
	s.loadedAt = time.Time{}
	s.mu.Unlock()

	if cr == nil {
		return
	}
	// SIGINT — это то же самое, что Ctrl+C в терминале: llama-server
	// корректно завершается сам (чистит память/GPU, пишет лог выхода).
	log.Printf("stopping backend: pid=%d (SIGINT, grace 15s)", cr.cmd.Process.Pid)
	_ = cr.cmd.Process.Signal(os.Interrupt)
	select {
	case <-cr.done:
	case <-time.After(15 * time.Second):
		log.Printf("backend pid=%d did not exit in 15s, sending SIGKILL", cr.cmd.Process.Pid)
		_ = cr.cmd.Process.Kill()
		<-cr.done
	}
	log.Printf("backend stopped")
}

// start поднимает llama-server для модели m и ждёт готовности.
func (s *server) start(m *ModelDef) error {
	c := s.cfgSnapshot()

	env := make([]string, 0, len(os.Environ())+len(m.Env))
	managed := make(map[string]bool, len(m.Env))
	for k := range m.Env {
		managed[k] = true
	}
	for _, e := range os.Environ() {
		var k string
		if i := strings.IndexByte(e, '='); i > 0 {
			k = e[:i]
		}
		if managed[k] {
			continue
		}
		env = append(env, e)
	}
	for k, v := range m.Env {
		env = append(env, k+"="+v)
	}

	args := backendArgs(m, c.backendPort())

	bin, _ := filepath.Abs(m.bin(c))
	cwd, _ := filepath.Abs(c.Cwd)

	log.Printf("starting %s: %s %s (cwd=%s)", m.Name, bin, strings.Join(args, " "), cwd)
	rc := exec.Command(bin, args...)
	rc.Dir = cwd
	rc.Env = env
	rc.Stdout = os.Stdout
	rc.Stderr = os.Stderr
	if s.logs != nil {
		// Хвост лога + скорость последнего запроса для дашборда.
		rc.Stdout = io.MultiWriter(os.Stdout, s.logs)
		rc.Stderr = io.MultiWriter(os.Stderr, s.logs)
	}
	if err := rc.Start(); err != nil {
		return fmt.Errorf("start backend: %w", err)
	}

	done := make(chan struct{})
	s.mu.Lock()
	s.run = &running{cmd: rc, done: done}
	s.mu.Unlock()

	go func() {
		waitErr := rc.Wait()
		close(done)
		s.mu.Lock()
		isCurrent := s.run != nil && s.run.cmd == rc
		code := -1
		if rc.ProcessState != nil {
			code = rc.ProcessState.ExitCode()
		}
		if isCurrent {
			s.run = nil
			s.current = ""
			s.backendID = ""
			s.loadedAt = time.Time{}
			s.lastErr = fmt.Sprintf("llama-server (%s) завершился сам: code=%d", m.Name, code)
			s.lastErrAt = time.Now()
		}
		s.mu.Unlock()
		if isCurrent {
			log.Printf("backend (%s) exited: code=%d err=%v", m.Name, code, waitErr)
		}
	}()
	return nil
}

// waitReady паллит /v1/models до готовности и возвращает id модели,
// который бэкенд сам сообщает (на него и подставляется в поле "model").
func (s *server) waitReady() (string, error) {
	c := s.cfgSnapshot()
	url := "http://" + c.BackendAddr + "/v1/models"
	deadline := time.Now().Add(c.loadT)
	client := &http.Client{Timeout: 3 * time.Second}
	for time.Now().Before(deadline) {
		s.mu.Lock()
		cr := s.run
		s.mu.Unlock()
		if cr != nil {
			select {
			case <-cr.done:
				return "", fmt.Errorf("backend exited during model load (see its log above)")
			default:
			}
		}
		resp, err := client.Get(url)
		if err == nil {
			ok := resp.StatusCode == http.StatusOK
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if ok {
				var doc struct {
					Data []struct {
						ID string `json:"id"`
					} `json:"data"`
				}
				_ = json.Unmarshal(body, &doc)
				if len(doc.Data) > 0 {
					return doc.Data[0].ID, nil
				}
				return "", nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return "", fmt.Errorf("timed out after %s waiting for backend to become ready", c.loadT)
}

// loadedIs возвращает реальный id модели и загрузлена ли модель m
// (сравнение по указателю: после SIGHUP модель с изменёнными
// параметрами получит новый указатель и будет переподнята).
func (s *server) loadedIs(m *ModelDef) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.current == m.Name && s.loadedDef == m {
		return s.backendID, true
	}
	return "", false
}

// doEnsureModel блокирует до тех пор, пока не загрузится модель m, и
// возвращает id модели, который сообщает бэкенд. Свитч не прерывает
// идущие запросы: сначала дожидаются завершения всех in-flight
// запросов к текущему бэкенду (мягкая остановка), остальные запросы
// к другой модели копятся в очереди (switchMu) до готового свитча.
// Если клиент отвалился во время ожидания (ctx canceled), запрос
// отбрасывается БЕЗ переключения модели — иначе мёртвые запросы
// гоняли бы модели туда-сюда по кругу вместе с ретраями.
//
// CRITICAL: при count=true запрос учитывается в in-flight счётчике,
// ПОКА ещё не отпущен switchMu. Если посчитать позже,
// следующий запрос из очереди увидит нулевой счётчик и отрежет
// только что поднятую модель, пока этот запрос ещё не начал
// стримиться (микросекундная щель, а клиент получает 502 + ретраи
// и вечный пинг-понг моделей).
func (s *server) doEnsureModel(m *ModelDef, ctx context.Context, count bool) (string, error) {
	if count {
		s.addQueued(m.Name, 1)
		defer s.addQueued(m.Name, -1)
	}
	s.switchMu.Lock()
	defer s.switchMu.Unlock()

	// Мягкая остановка: не прерываем идущую генерацию — ждём, пока
	// закончатся все активные запросы к текущей модели.
	id, ok := s.loadedIs(m)
	if !ok {
		s.setPhase("draining", m.Name)
		defer s.setPhase("", "")
		if err := s.waitIdle(ctx); err != nil {
			log.Printf("request for model %s canceled while waiting for in-flight to drain; skipping switch", m.Name)
			return "", err
		}
		// Пока ждали, модель могла стать запрошенной (запрос, стоявший
		// раньше в очереди, уже сделал свитч).
		id, ok = s.loadedIs(m)
		if !ok {
			// Клиент ушёл, пока запрос стоял в очереди — модель под
			// него поднимать бессмысленно.
			if err := ctx.Err(); err != nil {
				log.Printf("request for model %s canceled while queued; skipping switch", m.Name)
				return "", err
			}
			s.mu.Lock()
			old := s.current
			s.mu.Unlock()

			log.Printf("switching model: %s -> %s", old, m.Name)
			s.stopCurrent()
			var err error
			id, err = s.startAndWait(m)
			if err != nil {
				return "", err
			}
		}
	}
	if ok {
		log.Printf("model %s already loaded, reuse", m.Name)
	}

	if count {
		// Клиент мог уйти прямо перед стартом стрима — тогда и
		// считать незачем (иначе в счётчике останется "призрак",
		// который заблокирует следующие свитчи навсегда).
		if err := ctx.Err(); err != nil {
			log.Printf("request for model %s canceled; skipping generate", m.Name)
			return "", err
		}
		s.countInflight()
	}
	return id, nil
}

// ensureModel — doEnsureModel для обычного запроса: модель готова
// И запрос уже учтён в in-flight (свитч больше не дотянется до него).
func (s *server) ensureModel(m *ModelDef, ctx context.Context) (string, error) {
	return s.doEnsureModel(m, ctx, true)
}

// countInflight/uncountInflight ведут счётчик запросов, которые прямо
// сейчас прокидываются бэкенду (генерация/стрим в работе).
func (s *server) countInflight() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.inflight == 0 {
		s.idle = make(chan struct{})
	}
	s.inflight++
}

func (s *server) uncountInflight() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inflight--
	if s.inflight == 0 {
		close(s.idle)
	}
}

// waitIdle ждёт, пока не завершатся все in-flight запросы к бэкенду
// (свитч модели не должен обрывать стримы). Если запрос, который
// ждёт очереди, отозвали (клиент отвалился) — возвращает ctx-ошибку.
// Каскадные пробуждения (новый in-flight сразу после дрена)
// обрабатываются повторной проверкой.
func (s *server) waitIdle(ctx context.Context) error {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	lastN := -1
	for {
		s.mu.Lock()
		if s.inflight == 0 {
			s.mu.Unlock()
			return nil
		}
		n := s.inflight
		ch := s.idle
		if s.forceCh == nil {
			s.forceCh = make(chan struct{})
		}
		fc := s.forceCh
		s.mu.Unlock()
		if n != lastN {
			log.Printf("model switch deferred: %d in-flight request(s), waiting for them to finish", n)
			lastN = n
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ch:
		case <-fc:
			log.Printf("forced from dashboard: not waiting for %d in-flight request(s)", n)
			return nil
		case <-ticker.C:
			log.Printf("still waiting for in-flight requests to finish")
		}
	}
}

// startAndWait поднимает бэкенд для модели и ждёт готовности (вызывать со
// захваченным switchMu, но не mu).
func (s *server) startAndWait(m *ModelDef) (string, error) {
	s.setPhase("loading", m.Name)
	if err := s.start(m); err != nil {
		s.setError(err.Error())
		return "", err
	}
	id, err := s.waitReady()
	if err != nil {
		s.stopCurrent()
		err = fmt.Errorf("failed to load %s: %w", m.Name, err)
		s.setError(err.Error())
		return "", err
	}
	s.mu.Lock()
	s.current = m.Name
	s.loadedDef = m
	s.backendID = id
	s.loadedAt = time.Now()
	s.lastErr = ""
	s.mu.Unlock()
	log.Printf("model %s ready (backend id %q)", m.Name, id)
	return id, nil
}

// reload перечитывает конфиг-файл и подменяет таблицу моделей на лету
// (по SIGHUP). Изменения env/args/aliases применяются с момента перечита,
// listen_addr/backend_addr — при следующем рестарте.
func (s *server) reload(path string) error {
	c, err := loadConfig(path)
	if err != nil {
		return err
	}
	s.mu.Lock()
	old := s.cfg
	// Модели без изменений оставляем теми же указателями: тогда уже
	// загруженная модель после обычного reload переподниматься не будет.
	for i, m := range c.Models {
		for _, o := range old.Models {
			if o.Name == m.Name && reflect.DeepEqual(*o, *m) {
				c.Models[i] = o
				break
			}
		}
	}
	if err := c.buildTable(); err != nil {
		s.mu.Unlock()
		return err
	}
	s.cfg = c
	s.mu.Unlock()
	log.Printf("config reloaded from %s: %d model(s): %s", path, len(c.Models), c.knownNames())
	return nil
}

// ---------------------------------------------------------------------------
// HTTP-обработка.
// ---------------------------------------------------------------------------

func rewriteModelField(body []byte, id string) []byte {
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(body, &doc); err != nil {
		return body
	}
	raw, err := json.Marshal(id)
	if err != nil {
		return body
	}
	doc["model"] = raw
	out, err := json.Marshal(doc)
	if err != nil {
		return body
	}
	return out
}

func looksLikeJSON(b []byte) bool {
	for _, c := range b {
		switch c {
		case ' ', '\t', '\r', '\n':
			continue
		case '{':
			return true
		}
		return false
	}
	return false
}

func aiError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    "server_error",
			"code":    strconv.Itoa(code),
		},
	})
}

func (s *server) writeModels(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	c := s.cfgSnapshot()
	data := make([]map[string]string, 0, len(c.order))
	for _, m := range c.order {
		data = append(data, map[string]string{
			"id": m.Name, "object": "model", "owned_by": "llama-proxy",
		})
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
}

func (s *server) writeStatus(w http.ResponseWriter) {
	s.mu.Lock()
	cur := s.current
	pid := 0
	if s.run != nil && s.run.cmd.Process != nil {
		pid = s.run.cmd.Process.Pid
	}
	s.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"current":     cur,
		"backend_pid": pid,
		"up_seconds":  int(time.Since(s.startTime).Seconds()),
	})
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	c := s.cfgSnapshot()
	switch r.URL.Path {
	case "/":
		if r.Method == http.MethodGet || r.Method == http.MethodHead {
			s.serveDashboard(w, r)
			return
		}
		_, _ = io.WriteString(w, "llama-proxy ok\n")
		return
	case "/health":
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "llama-proxy ok\n")
		return
	case "/_admin/status":
		s.writeStatus(w)
		return
	case "/v1/models", "/models":
		if r.Method == http.MethodGet {
			s.writeModels(w)
			return
		}
	}
	if strings.HasPrefix(r.URL.Path, "/_admin/api/") {
		s.serveAdminAPI(w, r)
		return
	}

	// Журнал запросов для дашборда (статус, длительность, ожидание).
	sw := &statusWriter{ResponseWriter: w}
	w = sw
	var entry *reqEntry
	if logRequest(r) {
		entry = s.reqs.begin(r.Method, r.URL.Path)
		defer func() { s.reqs.end(entry, sw.status, sw.n) }()
	}

	var body []byte
	if r.Body != nil {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			aiError(w, http.StatusBadRequest, "failed to read request body: "+err.Error())
			return
		}
		body = b
	}

	var m *ModelDef
	unknownModel := ""
	if len(body) > 0 && looksLikeJSON(body) {
		var doc map[string]json.RawMessage
		if json.Unmarshal(body, &doc) == nil {
			if raw, ok := doc["model"]; ok {
				var name string
				if json.Unmarshal(raw, &name) == nil {
					if strings.TrimSpace(name) != "" {
						m = c.resolve(name)
						if m == nil {
							unknownModel = name
						}
					}
				}
			}
		}
	}

	counted := false
	switch {
	case m != nil:
		// ensureModel учёл запрос в in-flight ещё до освобождения
		// lock на свитч — следующий свитч не порежет модель у него
		// из-под носа.
		s.reqs.update(entry, func(e *reqEntry) { e.Model = m.Name })
		t0 := time.Now()
		id, err := s.ensureModel(m, r.Context())
		s.reqs.update(entry, func(e *reqEntry) { e.WaitMs = time.Since(t0).Milliseconds() })
		if err != nil {
			// Клиент отвалился, пока запрос стоял в очереди:
			// отвечать неоткуда и некому.
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				sw.status = 499
				return
			}
			aiError(w, http.StatusServiceUnavailable, "model switch failed: "+err.Error())
			return
		}
		counted = true
		if id != "" {
			body = rewriteModelField(body, id)
		}
	case unknownModel != "":
		aiError(w, http.StatusNotFound, "unknown model "+strconv.Quote(unknownModel)+", available: "+c.knownNames())
		return
	default:
		s.mu.Lock()
		cur := s.current
		s.mu.Unlock()
		if cur == "" {
			aiError(w, http.StatusServiceUnavailable, "no model loaded, specify \"model\" in the request (available: "+c.knownNames()+")")
			return
		}
		s.countInflight()
		counted = true
	}

	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.Header.Del("Content-Length")
	r.Header.Del("Transfer-Encoding")

	s.mu.Lock()
	fwdModel := s.current
	s.mu.Unlock()
	s.reqs.update(entry, func(e *reqEntry) { e.Model = fwdModel })
	log.Printf("%s %s -> backend (model=%s)", r.Method, r.URL.RequestURI(), fwdModel)
	if counted {
		defer s.uncountInflight()
	}
	s.proxy.ServeHTTP(w, r)
}

func newProxy(s *server) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		FlushInterval: -1,
		Director: func(req *http.Request) {
			backend := s.cfgSnapshot().BackendAddr
			req.URL.Scheme = "http"
			req.URL.Host = backend
			req.Host = backend
		},
		Transport: &http.Transport{
			MaxIdleConns:        64,
			MaxIdleConnsPerHost: 64,
			IdleConnTimeout:     90 * time.Second,
			DisableCompression:  true,
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			log.Printf("proxy error %s %s: %v", r.Method, r.URL.Path, err)
			aiError(w, http.StatusBadGateway, "backend error: "+err.Error())
		},
	}
}

func (s *server) adoptExisting() {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get("http://" + s.cfgSnapshot().BackendAddr + "/v1/models")
	if err != nil {
		return
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return
	}
	var doc struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	_ = json.Unmarshal(body, &doc)
	id := ""
	if len(doc.Data) > 0 {
		id = doc.Data[0].ID
	}
	s.mu.Lock()
	s.current = "external"
	s.backendID = id
	s.mu.Unlock()
	log.Printf("WARNING: llama-server already running at %s (model %q). Proxy will adopt it, but cannot kill it on switch — stop that process manually if a switch fails.", s.cfgSnapshot().BackendAddr, id)
}

func printResolved(c *Config) {
	for _, m := range c.order {
		fmt.Printf("model %q (aliases: %s)\n", m.Name, strings.Join(m.allNames(), ", "))
		fmt.Printf("  bin : %s\n", m.bin(c))
		fmt.Printf("  env : %v\n", m.Env)
		fmt.Printf("  args: %s\n", strings.Join(m.Args, " "))
	}
	if dm := c.startupModel(); dm != nil {
		fmt.Printf("startup model: %s\n", dm.Name)
	} else {
		fmt.Println("startup model: none (auto-load disabled)")
	}
}

func main() {
	flag.Parse()
	log.SetPrefix("llama-proxy: ")
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	c, err := loadConfig(*configFlag)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("loaded config %s: %d model(s), backend %s, timeout %s",
		*configFlag, len(c.Models), c.BackendAddr, c.loadT)

	if *checkFlag {
		printResolved(c)
		return
	}

	s := &server{cfg: c, startTime: time.Now(), idle: newClosedChan(), reqs: newReqLog()}
	s.logs = newLogTail(func() string {
		s.mu.Lock()
		defer s.mu.Unlock()
		return s.current
	})
	s.mon = newMonitor(s)
	s.proxy = newProxy(s)
	s.adoptExisting()
	if dm := c.startupModel(); dm != nil {
		// count=false: стартап-загрузка не связана с реальным
		// запросом, а "призрачный" in-flight навсегда заблокировал
		// бы первый свитч.
		if _, err := s.doEnsureModel(dm, context.Background(), false); err != nil {
			log.Printf("initial load of default model %s failed: %v (next request will retry)", dm.Name, err)
		}
	}

	log.Printf("proxy listening on %s", c.ListenAddr)

	srv := &http.Server{
		Addr:              c.ListenAddr,
		Handler:           s,
		ReadHeaderTimeout: 30 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, notifySignals()...)
		for sgn := range sig {
			if isReload(sgn) {
				if err := s.reload(*configFlag); err != nil {
					log.Printf("config reload failed, keeping old config: %v", err)
				}
				continue
			}
			log.Printf("signal %v received, shutting down", sgn)
			s.stopCurrent()
			os.Exit(0)
		}
	}()

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
