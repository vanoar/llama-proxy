package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// monitor — сбор метрик для дашборда. Главное правило: НИЧЕГО не
// делается в фоне, пока дашборд никто не смотрит. Сэмпл снимается
// только по запросу /_admin/api/stats и не чаще minSampleInterval —
// сколько бы вкладок ни было открыто, все получают один кэшированный
// снапшот. Источники выбраны самые дешёвые:
//   - CPU/RAM/сеть/процессы — /proc (чтение текстовых файлов ядра);
//   - температуры — /sys/class/hwmon (список файлов кэшируется);
//   - GPU — один долгоживущий `nvidia-smi -lms`, который гасится сам
//     после gpuIdleStop без запросов (NVML инициализируется один раз,
//     а не fork+init на каждый опрос);
//   - llama-server — /slots и /metrics (лёгкие задачи вне очереди
//     генерации), /props — один раз на процесс бэкенда.
// ---------------------------------------------------------------------------

const (
	minSampleInterval = 900 * time.Millisecond
	coldAfter         = 15 * time.Second // сэмпл старше — дельты не считаем по нему
	gpuIdleStop       = 45 * time.Second
	hwmonRescan       = 5 * time.Minute
)

type cpuTimes struct{ idle, total uint64 }

type slotPrev struct {
	task     int64
	nDecoded int64
	at       time.Time
	seen     time.Time // когда впервые увидели эту задачу в слоте
}

type monitor struct {
	s *server

	mu       sync.Mutex // сериализует сэмплы
	cached   []byte
	cachedAt time.Time

	lastAt   time.Time
	prevCPU  []cpuTimes
	prevNet  [2]uint64
	prevProc struct {
		pid   int
		ticks uint64
	}
	slots    map[int]slotPrev
	propsPID int
	props    *backendProps

	cpuModel string
	hw       hwmonCache
	gpu      *gpuPoller
	client   *http.Client
}

func newMonitor(s *server) *monitor {
	m := &monitor{
		s:     s,
		slots: map[int]slotPrev{},
		client: &http.Client{
			Timeout: 2 * time.Second,
			Transport: &http.Transport{
				MaxIdleConnsPerHost: 2,
				IdleConnTimeout:     30 * time.Second,
				DisableCompression:  true,
			},
		},
	}
	m.cpuModel = readCPUModel()
	m.gpu = newGPUPoller(s.cfgSnapshot().NvidiaSmi)
	return m
}

// --- JSON-документ дашборда ------------------------------------------------

type statsDoc struct {
	Now      int64         `json:"now"`
	Proxy    proxyStats    `json:"proxy"`
	Models   []modelInfo   `json:"models"`
	Backend  *backendStats `json:"backend,omitempty"`
	System   sysStats      `json:"system"`
	GPUs     []gpuStats    `json:"gpus"`
	GPUNote  string        `json:"gpu_note,omitempty"`
	Requests reqStats      `json:"requests"`
}

type modelInfo struct {
	Name    string   `json:"name"`
	Aliases []string `json:"aliases"`
	Quant   string   `json:"source,omitempty"` // значение -hf / -m из args
	Ctx     string   `json:"ctx,omitempty"`
}

type proxyStats struct {
	Current      string         `json:"current"`
	BackendID    string         `json:"backend_id"`
	PID          int            `json:"pid"`
	Phase        string         `json:"phase"` // "", draining, loading, unloading
	PhaseTarget  string         `json:"phase_target,omitempty"`
	PhaseSeconds float64        `json:"phase_seconds,omitempty"`
	LoadedSec    float64        `json:"loaded_seconds,omitempty"`
	UpSeconds    float64        `json:"up_seconds"`
	Inflight     int            `json:"inflight"`
	Queued       int            `json:"queued"`
	QueuedBy     map[string]int `json:"queued_by,omitempty"`
	LastError    string         `json:"last_error,omitempty"`
	LastErrorAgo float64        `json:"last_error_ago,omitempty"`
	AdminBusy    bool           `json:"admin_busy"`
	BackendPort  string         `json:"backend_port"`
	BackendUI    string         `json:"backend_ui,omitempty"`
	Last         *reqTimings    `json:"last,omitempty"`
}

type backendStats struct {
	Error        string        `json:"error,omitempty"`
	Metrics      bool          `json:"metrics"` // --metrics включён
	SlotsAPI     bool          `json:"slots_api"`
	Processing   int           `json:"processing"`
	Prompting    int           `json:"prompting"`
	Generating   int           `json:"generating"`
	Deferred     int           `json:"deferred"`
	PromptTotal  float64       `json:"prompt_tokens_total"`
	GenTotal     float64       `json:"gen_tokens_total"`
	PromptAvgTPS float64       `json:"prompt_avg_tps"`
	GenAvgTPS    float64       `json:"gen_avg_tps"`
	GenLiveTPS   *float64      `json:"gen_live_tps"`
	Slots        []slotStats   `json:"slots"`
	Props        *backendProps `json:"props,omitempty"`
	CPUPct       *float64      `json:"cpu_pct"` // % одного ядра
	RSS          uint64        `json:"rss"`
}

type slotStats struct {
	ID       int      `json:"id"`
	Task     int64    `json:"task"`
	NCtx     int      `json:"n_ctx"`
	State    string   `json:"state"` // idle | prompt | gen
	NDecoded int64    `json:"n_decoded"`
	NRemain  int64    `json:"n_remain"`
	TPS      *float64 `json:"tps"`
	Elapsed  float64  `json:"elapsed"`
}

type backendProps struct {
	NCtx       int    `json:"n_ctx"`
	TotalSlots int    `json:"total_slots"`
	ModelPath  string `json:"model_path"`
	Build      string `json:"build"`
}

type sysStats struct {
	Hostname  string     `json:"hostname"`
	Uptime    float64    `json:"uptime"`
	CPUModel  string     `json:"cpu_model"`
	Cores     int        `json:"cores"`
	CPUPct    *float64   `json:"cpu_pct"`
	PerCore   []float64  `json:"per_core"`
	Load      []float64  `json:"load"`
	MemTotal  uint64     `json:"mem_total"`
	MemAvail  uint64     `json:"mem_avail"`
	SwapTotal uint64     `json:"swap_total"`
	SwapFree  uint64     `json:"swap_free"`
	NetRx     *float64   `json:"net_rx"` // байт/с
	NetTx     *float64   `json:"net_tx"`
	Temps     []tempChip `json:"temps"`
	ProxyRSS  uint64     `json:"proxy_rss"`
}

// --- сэмплинг ---------------------------------------------------------------

func (m *monitor) stats() []byte {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cached != nil && time.Since(m.cachedAt) < minSampleInterval {
		return m.cached
	}
	doc := m.sample()
	b, _ := json.Marshal(doc)
	m.cached, m.cachedAt = b, time.Now()
	return b
}

func (m *monitor) sample() *statsDoc {
	s := m.s
	c := s.cfgSnapshot()
	now := time.Now()

	// GPU первым: на холодном старте nvidia-smi поднимается параллельно.
	m.gpu.want()

	// Холодный старт (дашборд только что открыли): чтобы CPU% не
	// оказался средним за последний час, делаем короткий замер.
	cold := m.lastAt.IsZero() || now.Sub(m.lastAt) > coldAfter
	if cold {
		m.prevCPU = readCPUTimes()
		m.prevNet = readNet()
		if pid := s.backendPID(); pid > 0 {
			m.prevProc.pid = pid
			m.prevProc.ticks, _ = readProcTicks(pid)
		}
		m.lastAt = now
		time.Sleep(250 * time.Millisecond)
		now = time.Now()
	}
	dt := now.Sub(m.lastAt).Seconds()
	m.lastAt = now

	doc := &statsDoc{Now: now.UnixMilli()}
	doc.Proxy = s.proxySnapshot(c)
	doc.Proxy.Last = s.logs.lastTimings()
	for _, md := range c.order {
		doc.Models = append(doc.Models, describeModel(md))
	}

	// --- система
	sys := &doc.System
	sys.Hostname, _ = os.Hostname()
	sys.CPUModel = m.cpuModel
	sys.Cores = runtime.NumCPU()
	if cur := readCPUTimes(); len(cur) > 0 {
		if len(m.prevCPU) == len(cur) {
			for i := range cur {
				p := cpuPct(m.prevCPU[i], cur[i])
				if i == 0 {
					sys.CPUPct = &p
				} else {
					sys.PerCore = append(sys.PerCore, round1(p))
				}
			}
		}
		m.prevCPU = cur
	}
	sys.Load = readLoadAvg()
	sys.Uptime = readUptime()
	mi := readMeminfo()
	sys.MemTotal, sys.MemAvail = mi["MemTotal"], mi["MemAvailable"]
	sys.SwapTotal, sys.SwapFree = mi["SwapTotal"], mi["SwapFree"]
	if net := readNet(); net != [2]uint64{} {
		if m.prevNet != [2]uint64{} && dt > 0 && net[0] >= m.prevNet[0] && net[1] >= m.prevNet[1] {
			rx, tx := float64(net[0]-m.prevNet[0])/dt, float64(net[1]-m.prevNet[1])/dt
			sys.NetRx, sys.NetTx = &rx, &tx
		}
		m.prevNet = net
	}
	sys.Temps = m.hw.read()
	sys.ProxyRSS = readProcRSS("self")

	// --- llama-server
	pid := doc.Proxy.PID
	if doc.Proxy.Current != "" && doc.Proxy.Phase != "loading" {
		doc.Backend = m.sampleBackend(c, pid, now)
	} else {
		m.slots = map[int]slotPrev{}
	}
	if pid > 0 {
		if doc.Backend == nil {
			doc.Backend = &backendStats{}
		}
		ticks, ok := readProcTicks(pid)
		if ok && m.prevProc.pid == pid && ticks >= m.prevProc.ticks && dt > 0 {
			p := round1(float64(ticks-m.prevProc.ticks) / clkTck / dt * 100)
			doc.Backend.CPUPct = &p
		}
		m.prevProc.pid, m.prevProc.ticks = pid, ticks
		doc.Backend.RSS = readProcRSS(strconv.Itoa(pid))
	}

	// --- GPU
	doc.GPUs, doc.GPUNote = m.gpu.get()
	doc.Requests = s.reqs.snapshot()
	return doc
}

func describeModel(md *ModelDef) modelInfo {
	mi := modelInfo{Name: md.Name, Aliases: md.allNames()[1:]}
	for i := 0; i+1 < len(md.Args); i++ {
		switch md.Args[i] {
		case "-hf", "--hf-repo", "-m", "--model":
			mi.Quant = md.Args[i+1]
		case "-c", "--ctx-size":
			mi.Ctx = md.Args[i+1]
		}
	}
	if mi.Aliases == nil {
		mi.Aliases = []string{}
	}
	return mi
}

func (m *monitor) sampleBackend(c *Config, pid int, now time.Time) *backendStats {
	base := "http://" + c.BackendAddr
	b := &backendStats{}

	var wg sync.WaitGroup
	var slotsBody, metricsBody []byte
	var slotsCode, metricsCode int
	var slotsErr error
	wg.Add(2)
	go func() { defer wg.Done(); slotsBody, slotsCode, slotsErr = m.get(base + "/slots") }()
	go func() { defer wg.Done(); metricsBody, metricsCode, _ = m.get(base + "/metrics") }()
	if pid != m.propsPID || m.props == nil {
		if body, code, err := m.get(base + "/props"); err == nil && code == 200 {
			m.props, m.propsPID = parseProps(body), pid
		}
	}
	wg.Wait()
	b.Props = m.props

	if metricsCode == 200 {
		b.Metrics = true
		pm := parseProm(metricsBody)
		b.Processing = int(pm["llamacpp:requests_processing"])
		b.Deferred = int(pm["llamacpp:requests_deferred"])
		b.PromptTotal = pm["llamacpp:prompt_tokens_total"]
		b.GenTotal = pm["llamacpp:tokens_predicted_total"]
		if s := pm["llamacpp:prompt_seconds_total"]; s > 0 {
			b.PromptAvgTPS = round1(b.PromptTotal / s)
		}
		if s := pm["llamacpp:tokens_predicted_seconds_total"]; s > 0 {
			b.GenAvgTPS = round1(b.GenTotal / s)
		}
	}

	switch {
	case slotsErr != nil:
		b.Error = slotsErr.Error()
	case slotsCode == 200:
		b.SlotsAPI = true
		m.applySlots(b, parseSlots(slotsBody), now, m.s.logs.liveNow(4*time.Second))
	case slotsCode == 503:
		b.Error = "бэкенд ещё не готов"
	}
	if !b.Metrics && b.SlotsAPI {
		b.Processing = 0
		for _, sl := range b.Slots {
			if sl.State != "idle" {
				b.Processing++
			}
		}
	}
	return b
}

func (m *monitor) applySlots(b *backendStats, raw []rawSlot, now time.Time, hint liveHint) {
	next := make(map[int]slotPrev, len(raw))
	var live float64
	haveLive := false
	for _, r := range raw {
		st := slotStats{ID: r.ID, Task: r.Task, NCtx: r.NCtx, NDecoded: r.NDecoded, NRemain: r.NRemain, State: "idle"}
		p, had := m.slots[r.ID]
		cur := slotPrev{task: r.Task, nDecoded: r.NDecoded, at: now, seen: now}
		if had && p.task == r.Task {
			cur.seen = p.seen
		}
		if r.Processing {
			switch {
			case !r.HasPhase && hint.Phase == "":
				st.State = "busy" // фаза неизвестна
			case !r.HasPhase && hint.Phase == "gen", r.HasPhase && r.NDecoded > 0:
				st.State = "gen"
				b.Generating++
			default:
				st.State = "prompt"
				b.Prompting++
			}
			if !r.HasPhase && hint.Phase == "gen" && hint.TPS > 0 {
				v := hint.TPS
				st.TPS = &v
				live += v
				haveLive = true
			}
			if had {
				st.Elapsed = round1(now.Sub(cur.seen).Seconds())
			}
			if had && p.task == r.Task && r.NDecoded >= p.nDecoded && p.nDecoded > 0 {
				if dt := now.Sub(p.at).Seconds(); dt > 0 {
					v := round1(float64(r.NDecoded-p.nDecoded) / dt)
					st.TPS = &v
					live += v
					haveLive = true
				}
			}
		}
		next[r.ID] = cur
		b.Slots = append(b.Slots, st)
	}
	m.slots = next
	if haveLive {
		v := round1(live)
		b.GenLiveTPS = &v
	} else if b.Generating == 0 {
		z := 0.0
		b.GenLiveTPS = &z
	}
}

func (m *monitor) get(url string) ([]byte, int, error) {
	resp, err := m.client.Get(url)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	return body, resp.StatusCode, err
}

// --- парсеры ответов llama-server -------------------------------------------

func parseProm(b []byte) map[string]float64 {
	out := map[string]float64{}
	for _, line := range strings.Split(string(b), "\n") {
		if line == "" || line[0] == '#' {
			continue
		}
		f := strings.Fields(line)
		if len(f) < 2 {
			continue
		}
		name := f[0]
		if i := strings.IndexByte(name, '{'); i >= 0 {
			name = name[:i]
		}
		if v, err := strconv.ParseFloat(f[1], 64); err == nil {
			out[name] = v
		}
	}
	return out
}

type rawSlot struct {
	ID         int
	Task       int64
	NCtx       int
	Processing bool
	HasPhase   bool // есть next_token (llama-server); у Strata нет — фазу берём из лога
	NDecoded   int64
	NRemain    int64
}

func parseSlots(b []byte) []rawSlot {
	var doc []struct {
		ID         int             `json:"id"`
		IDTask     int64           `json:"id_task"`
		NCtx       int             `json:"n_ctx"`
		Processing bool            `json:"is_processing"`
		NextToken  json.RawMessage `json:"next_token"`
	}
	if json.Unmarshal(b, &doc) != nil {
		return nil
	}
	out := make([]rawSlot, 0, len(doc))
	for _, d := range doc {
		type nt struct {
			NDecoded int64 `json:"n_decoded"`
			NRemain  int64 `json:"n_remain"`
		}
		// next_token в разных версиях llama.cpp — объект или массив из одного объекта.
		var t nt
		if json.Unmarshal(d.NextToken, &t) != nil {
			var arr []nt
			if json.Unmarshal(d.NextToken, &arr) == nil && len(arr) > 0 {
				t = arr[0]
			}
		}
		out = append(out, rawSlot{ID: d.ID, Task: d.IDTask, NCtx: d.NCtx, Processing: d.Processing, HasPhase: len(d.NextToken) > 0,
			NDecoded: t.NDecoded, NRemain: t.NRemain})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func parseProps(b []byte) *backendProps {
	var doc struct {
		TotalSlots int    `json:"total_slots"`
		ModelPath  string `json:"model_path"`
		Build      string `json:"build_info"`
		Gen        struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
	}
	if json.Unmarshal(b, &doc) != nil {
		return nil
	}
	return &backendProps{NCtx: doc.Gen.NCtx, TotalSlots: doc.TotalSlots, ModelPath: filepath.Base(doc.ModelPath), Build: doc.Build}
}

// --- /proc ------------------------------------------------------------------

const clkTck = 100 // USER_HZ на Linux практически всегда 100

func readCPUTimes() []cpuTimes {
	b, err := os.ReadFile("/proc/stat")
	if err != nil {
		return nil
	}
	return parseCPUTimes(b)
}

func parseCPUTimes(b []byte) []cpuTimes {
	var out []cpuTimes
	for _, line := range strings.Split(string(b), "\n") {
		if !strings.HasPrefix(line, "cpu") {
			if len(out) > 0 {
				break
			}
			continue
		}
		f := strings.Fields(line)
		if len(f) < 5 {
			continue
		}
		var t cpuTimes
		// user nice system idle iowait irq softirq steal (guest уже в user)
		for i := 1; i < len(f) && i <= 8; i++ {
			v, _ := strconv.ParseUint(f[i], 10, 64)
			t.total += v
			if i == 4 || i == 5 {
				t.idle += v
			}
		}
		out = append(out, t)
	}
	return out
}

func cpuPct(a, b cpuTimes) float64 {
	dt := float64(b.total - a.total)
	if b.total <= a.total {
		return 0
	}
	return round1((1 - float64(b.idle-a.idle)/dt) * 100)
}

func readLoadAvg() []float64 {
	b, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return nil
	}
	f := strings.Fields(string(b))
	var out []float64
	for i := 0; i < 3 && i < len(f); i++ {
		v, _ := strconv.ParseFloat(f[i], 64)
		out = append(out, v)
	}
	return out
}

func readUptime() float64 {
	b, err := os.ReadFile("/proc/uptime")
	if err != nil {
		return 0
	}
	f := strings.Fields(string(b))
	if len(f) == 0 {
		return 0
	}
	v, _ := strconv.ParseFloat(f[0], 64)
	return v
}

// readMeminfo возвращает значения в байтах.
func readMeminfo() map[string]uint64 {
	out := map[string]uint64{}
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return out
	}
	for _, line := range strings.Split(string(b), "\n") {
		k, rest, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		f := strings.Fields(rest)
		if len(f) == 0 {
			continue
		}
		v, _ := strconv.ParseUint(f[0], 10, 64)
		out[k] = v * 1024
	}
	return out
}

// readNet — суммарные rx/tx байты по всем интерфейсам, кроме lo.
func readNet() [2]uint64 {
	b, err := os.ReadFile("/proc/net/dev")
	if err != nil {
		return [2]uint64{}
	}
	var rx, tx uint64
	for _, line := range strings.Split(string(b), "\n") {
		name, rest, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		name = strings.TrimSpace(name)
		if name == "lo" {
			continue
		}
		f := strings.Fields(rest)
		if len(f) < 9 {
			continue
		}
		r, _ := strconv.ParseUint(f[0], 10, 64)
		t, _ := strconv.ParseUint(f[8], 10, 64)
		rx += r
		tx += t
	}
	return [2]uint64{rx, tx}
}

func readProcTicks(pid int) (uint64, bool) {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return 0, false
	}
	// comm в скобках может содержать пробелы — режем по последней ')'.
	s := string(b)
	i := strings.LastIndexByte(s, ')')
	if i < 0 {
		return 0, false
	}
	f := strings.Fields(s[i+1:])
	// после ')' : state(0) ... utime(11) stime(12)
	if len(f) < 13 {
		return 0, false
	}
	u, _ := strconv.ParseUint(f[11], 10, 64)
	st, _ := strconv.ParseUint(f[12], 10, 64)
	return u + st, true
}

func readProcRSS(pid string) uint64 {
	b, err := os.ReadFile("/proc/" + pid + "/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "VmRSS:") {
			f := strings.Fields(line[6:])
			if len(f) > 0 {
				v, _ := strconv.ParseUint(f[0], 10, 64)
				return v * 1024
			}
		}
	}
	return 0
}

// readCPUModel читается один раз при старте: /proc/cpuinfo на x86
// опрашивает частоту каждого ядра (IPI), в цикле его читать не стоит.
func readCPUModel() string {
	f, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if ok && strings.TrimSpace(k) == "model name" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func round1(v float64) float64 {
	return float64(int64(v*10+0.5)) / 10
}

// --- hwmon (температуры) -------------------------------------------------------

type tempSensor struct {
	Label string   `json:"label"`
	C     float64  `json:"c"`
	Max   *float64 `json:"max,omitempty"`
	Crit  *float64 `json:"crit,omitempty"`
}

type tempChip struct {
	Name    string       `json:"name"`
	Kind    string       `json:"kind"` // cpu | nvme | gpu | other
	Sensors []tempSensor `json:"sensors"`
}

type hwmonFile struct {
	chip  int
	input string
	label string
	max   *float64
	crit  *float64
}

type hwmonCache struct {
	scanned time.Time
	chips   []tempChip // шаблоны (без значений)
	files   []hwmonFile
}

// skipChips — сенсоры, чтение которых не бесплатно: drivetemp шлёт
// SMART-команду диску (может будить спящий HDD).
var skipChips = map[string]bool{"drivetemp": true}

func (h *hwmonCache) rescan() {
	h.scanned = time.Now()
	h.chips, h.files = nil, nil
	dirs, _ := filepath.Glob("/sys/class/hwmon/hwmon*")
	sort.Strings(dirs)
	for _, d := range dirs {
		name := readTrim(filepath.Join(d, "name"))
		if name == "" || skipChips[name] {
			continue
		}
		inputs, _ := filepath.Glob(filepath.Join(d, "temp*_input"))
		if len(inputs) == 0 {
			continue
		}
		sort.Slice(inputs, func(i, j int) bool { return sensorNum(inputs[i]) < sensorNum(inputs[j]) })
		chip := len(h.chips)
		h.chips = append(h.chips, tempChip{Name: name, Kind: chipKind(name)})
		for _, in := range inputs {
			pfx := strings.TrimSuffix(in, "_input")
			f := hwmonFile{chip: chip, input: in, label: readTrim(pfx + "_label")}
			if f.label == "" {
				f.label = filepath.Base(pfx)
			}
			f.max = readLimit(pfx + "_max")
			f.crit = readLimit(pfx + "_crit")
			h.files = append(h.files, f)
		}
	}
}

func (h *hwmonCache) read() []tempChip {
	if time.Since(h.scanned) > hwmonRescan {
		h.rescan()
	}
	out := make([]tempChip, len(h.chips))
	copy(out, h.chips)
	for i := range out {
		out[i].Sensors = nil
	}
	for _, f := range h.files {
		v := readMilli(f.input)
		if v == nil || !plausibleTemp(out[f.chip].Kind, *v) {
			continue
		}
		out[f.chip].Sensors = append(out[f.chip].Sensors, tempSensor{Label: f.label, C: *v, Max: f.max, Crit: f.crit})
	}
	res := out[:0]
	for _, c := range out {
		if len(c.Sensors) > 0 {
			res = append(res, c)
		}
	}
	return res
}

// plausibleTemp отсекает неподключённые входы: Super-IO (nct67xx, it87…)
// на свободных пинах отдаёт 0, 127, -59.5, 109 и т.п.
func plausibleTemp(kind string, c float64) bool {
	if kind == "other" {
		return c > 5 && c < 100
	}
	return c > 0 && c < 127
}

func chipKind(name string) string {
	switch {
	case name == "coretemp" || name == "k10temp" || name == "zenpower" || name == "cpu_thermal" || strings.HasPrefix(name, "x86_pkg"):
		return "cpu"
	case name == "nvme":
		return "nvme"
	case name == "amdgpu" || name == "nouveau" || name == "radeon":
		return "gpu"
	}
	return "other"
}

func sensorNum(path string) int {
	b := filepath.Base(path)
	b = strings.TrimPrefix(b, "temp")
	b = strings.TrimSuffix(b, "_input")
	n, _ := strconv.Atoi(b)
	return n
}

func readTrim(p string) string {
	b, err := os.ReadFile(p)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

func readMilli(p string) *float64 {
	s := readTrim(p)
	if s == "" {
		return nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return nil
	}
	v = round1(v / 1000)
	return &v
}

// readLimit — порог (max/crit); 0 и меньше означает «не задан».
func readLimit(p string) *float64 {
	v := readMilli(p)
	if v == nil || *v <= 0 {
		return nil
	}
	return v
}

// --- NVIDIA ------------------------------------------------------------------

const gpuQuery = "index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,fan.speed,clocks.sm,clocks.max.sm,pstate"

type gpuStats struct {
	Index    int      `json:"index"`
	Name     string   `json:"name"`
	Temp     *float64 `json:"temp"`
	Util     *float64 `json:"util"`
	MemUtil  *float64 `json:"mem_util"`
	MemUsed  *float64 `json:"mem_used"` // MiB
	MemTotal *float64 `json:"mem_total"`
	Power    *float64 `json:"power"`
	PowerLim *float64 `json:"power_limit"`
	Fan      *float64 `json:"fan"`
	ClockSM  *float64 `json:"clock_sm"`
	ClockMax *float64 `json:"clock_max"`
	PState   string   `json:"pstate"`
	at       time.Time
}

// gpuPoller держит `nvidia-smi --query-gpu ... -lms N` запущенным,
// пока дашборд опрашивается, и гасит его после gpuIdleStop тишины.
type gpuPoller struct {
	bin string

	mu       sync.Mutex
	cmd      *exec.Cmd
	lastWant time.Time
	gpus     map[int]gpuStats
	note     string
	nextTry  time.Time
	first    chan struct{}
}

func newGPUPoller(cfgBin string) *gpuPoller {
	p := &gpuPoller{gpus: map[int]gpuStats{}}
	switch strings.ToLower(strings.TrimSpace(cfgBin)) {
	case "none", "off":
		p.note = "мониторинг GPU отключён в конфиге (nvidia_smi: none)"
		return p
	case "":
		if bin, err := exec.LookPath("nvidia-smi"); err == nil {
			p.bin = bin
		} else {
			p.note = "nvidia-smi не найден в PATH"
		}
	default:
		p.bin = cfgBin
	}
	return p
}

// want отмечает интерес к GPU и при необходимости поднимает nvidia-smi.
func (p *gpuPoller) want() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.lastWant = time.Now()
	if p.bin == "" || p.cmd != nil || time.Now().Before(p.nextTry) {
		return
	}
	cmd := exec.Command(p.bin, "--query-gpu="+gpuQuery, "--format=csv,noheader,nounits", "-lms", "2000")
	setChildDeath(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		p.note = err.Error()
		p.nextTry = time.Now().Add(time.Minute)
		return
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		p.note = "nvidia-smi: " + err.Error()
		p.nextTry = time.Now().Add(time.Minute)
		return
	}
	p.cmd = cmd
	p.first = make(chan struct{})
	go p.read(cmd, stdout, &stderr, p.first)
	go p.reaper(cmd)
}

func (p *gpuPoller) read(cmd *exec.Cmd, r io.Reader, stderr *bytes.Buffer, first chan struct{}) {
	sc := bufio.NewScanner(r)
	gotFirst := false
	for sc.Scan() {
		g, ok := parseGPULine(sc.Text())
		if !ok {
			continue
		}
		g.at = time.Now()
		p.mu.Lock()
		p.gpus[g.Index] = g
		p.note = ""
		p.mu.Unlock()
		if !gotFirst {
			gotFirst = true
			close(first)
		}
	}
	err := cmd.Wait()
	if !gotFirst {
		close(first)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cmd == cmd {
		p.cmd = nil
	}
	// Вышел сам (а не погашен нами по простою) — ошибка драйвера и т.п.
	if time.Since(p.lastWant) < gpuIdleStop {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" && err != nil {
			msg = err.Error()
		}
		if msg == "" {
			msg = "nvidia-smi завершился"
		}
		p.note = "nvidia-smi: " + msg
		p.nextTry = time.Now().Add(30 * time.Second)
	}
}

func (p *gpuPoller) reaper(cmd *exec.Cmd) {
	t := time.NewTicker(10 * time.Second)
	defer t.Stop()
	for range t.C {
		p.mu.Lock()
		if p.cmd != cmd {
			p.mu.Unlock()
			return
		}
		if time.Since(p.lastWant) > gpuIdleStop {
			p.cmd = nil
			p.gpus = map[int]gpuStats{}
			p.mu.Unlock()
			_ = cmd.Process.Kill()
			return
		}
		p.mu.Unlock()
	}
}

func (p *gpuPoller) get() ([]gpuStats, string) {
	p.mu.Lock()
	first := p.first
	p.mu.Unlock()
	if first != nil {
		// Холодный старт nvidia-smi: подождём первую порцию, но недолго.
		select {
		case <-first:
		case <-time.After(2 * time.Second):
		}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]gpuStats, 0, len(p.gpus))
	for _, g := range p.gpus {
		if time.Since(g.at) < 10*time.Second {
			out = append(out, g)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Index < out[j].Index })
	return out, p.note
}

func parseGPULine(line string) (gpuStats, bool) {
	f := strings.Split(line, ",")
	if len(f) < 13 {
		return gpuStats{}, false
	}
	for i := range f {
		f[i] = strings.TrimSpace(f[i])
	}
	// имя GPU теоретически может содержать запятую — склеиваем обратно.
	if n := len(f) - 13; n > 0 {
		f = append([]string{f[0], strings.Join(f[1:2+n], ",")}, f[2+n:]...)
	}
	idx, err := strconv.Atoi(f[0])
	if err != nil {
		return gpuStats{}, false
	}
	num := func(s string) *float64 {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil // [N/A], [Not Supported]
		}
		return &v
	}
	return gpuStats{
		Index: idx, Name: f[1],
		Temp: num(f[2]), Util: num(f[3]), MemUtil: num(f[4]),
		MemUsed: num(f[5]), MemTotal: num(f[6]),
		Power: num(f[7]), PowerLim: num(f[8]), Fan: num(f[9]),
		ClockSM: num(f[10]), ClockMax: num(f[11]), PState: f[12],
	}, true
}
