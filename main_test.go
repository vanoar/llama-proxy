package main

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func writeTmpConfig(t *testing.T, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

const baseConfig = `{
  "listen_addr": ":8080",
  "backend_addr": "127.0.0.1:8081",
  "llama_bin": "/x/llama-server",
  "cwd": "/work",
  "load_timeout": "15m",
  "models": [
    {"name": "Muse-Glimmer-30B", "aliases": ["Muse-Glimmer-30B", "muse"],
     "env": {"GGML_OP_OFFLOAD_MIN_BATCH": "2048"},
     "args": ["-hf", "repo-a/Q5.gguf", "-c", "500000"]},
    {"name": "Qwen3.8-27b", "aliases": ["Qwen3.8-27b"],
     "env": {"MTMD_BACKEND_DEVICE": "none"},
     "args": ["-hf", "repo-b:Q8", "-c", "262144", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]}
  ]
}`

func TestLoadConfigOK(t *testing.T) {
	c, err := loadConfig(writeTmpConfig(t, baseConfig))
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if c.ListenAddr != ":8080" || c.BackendAddr != "127.0.0.1:8081" {
		t.Fatalf("unexpected addr fields: %q %q", c.ListenAddr, c.BackendAddr)
	}
	if c.backendPort() != "8081" {
		t.Fatalf("backendPort = %q", c.backendPort())
	}
	if len(c.order) != 2 {
		t.Fatalf("expected 2 models, got %d", len(c.order))
	}
	if got := c.resolve("MUSE"); got == nil || got.Name != "Muse-Glimmer-30B" {
		t.Fatalf("resolve(MUSE) = %v, want Muse-Glimmer-30B", got)
	}
	if got := c.resolve("qwen3.8-27b"); got == nil || got.Name != "Qwen3.8-27b" {
		t.Fatalf("resolve(qwen3.8-27b) = %v", got)
	}
	if got := c.resolve("unknown-x"); got != nil {
		t.Fatalf("resolve(unknown-x) = %v, want nil", got)
	}
	if got := c.knownNames(); got != "Muse-Glimmer-30B, Qwen3.8-27b" {
		t.Fatalf("knownNames = %q", got)
	}
}

func TestLoadConfigDefaults(t *testing.T) {
	p := writeTmpConfig(t, `{"models":[{"name":"a","args":["-c","1"]}]}`)
	c, err := loadConfig(p)
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if c.ListenAddr != ":8080" || c.BackendAddr != "127.0.0.1:8081" {
		t.Fatalf("defaults not applied: %q %q", c.ListenAddr, c.BackendAddr)
	}
	if c.LlamaBin == "" || c.Cwd == "" || c.loadT <= 0 {
		t.Fatalf("defaults not applied: bin=%q cwd=%q loadT=%v", c.LlamaBin, c.Cwd, c.loadT)
	}
}

func TestLoadConfigErrors(t *testing.T) {
	cases := map[string]string{
		"no models":  `{"models": []}`,
		"empty name": `{"models": [{"name": "  ", "args": []}]}`,
		"dup alias": `{"models": [
			{"name": "a", "aliases": ["shared"], "args": []},
			{"name": "b", "aliases": ["Shared"], "args": []}]}`,
		"bad timeout":   `{"load_timeout": "soon", "models": [{"name": "a", "args": []}]}`,
		"unknown field": `{"listen_addrr": ":1", "models": [{"name": "a", "args": []}]}`,
	}
	for name, content := range cases {
		if _, err := loadConfig(writeTmpConfig(t, content)); err == nil {
			t.Errorf("case %q: expected error, got nil", name)
		}
	}
	if _, err := loadConfig(filepath.Join(t.TempDir(), "nope.json")); err == nil {
		t.Errorf("missing file: expected error, got nil")
	}
}

func TestDefaultModelValidation(t *testing.T) {
	if _, err := loadConfig(writeTmpConfig(t, `{"default_model": "ghost", "models": [{"name": "a", "args": []}]}`)); err == nil {
		t.Errorf("unknown default_model: expected error")
	}
	c, err := loadConfig(writeTmpConfig(t, `{"default_model": "a", "models": [{"name": "a", "aliases": ["A2"], "args": []}]}`))
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if c.resolve(c.DefaultModel).Name != "a" {
		t.Errorf("default_model should resolve to model a")
	}
}

func TestStartupModel(t *testing.T) {
	// без default_model -> первая модель из списка
	c, err := loadConfig(writeTmpConfig(t, baseConfig))
	if err != nil {
		t.Fatal(err)
	}
	if got := c.startupModel(); got == nil || got.Name != "Muse-Glimmer-30B" {
		t.Fatalf("startupModel without default_model = %v, want Muse-Glimmer-30B (first in list)", got)
	}
	// явный default_model (по алиасу) -> эта модель
	c2, err := loadConfig(writeTmpConfig(t, `{"default_model": "qwen", "models": [
		{"name": "a", "args": []},
		{"name": "q", "aliases": ["QWEN"], "args": []}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := c2.startupModel(); got == nil || got.Name != "q" {
		t.Fatalf("startupModel with default_model = %v, want q", got)
	}
	// "none" (регистр не важен, можно с пробелами) -> автозагрузки нет
	c3, err := loadConfig(writeTmpConfig(t, `{"default_model": " NONE ", "models": [{"name": "a", "args": []}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := c3.startupModel(); got != nil {
		t.Fatalf("startupModel with default_model=none = %v, want nil", got)
	}
}

func TestReloadPointerSharing(t *testing.T) {
	base := writeTmpConfig(t, baseConfig)
	c, err := loadConfig(base)
	if err != nil {
		t.Fatal(err)
	}
	s := &server{cfg: c}
	oldMuse := c.table["muse"]
	oldQwen := c.table["qwen3.8-27b"]

	// reload без изменений -> те же указатели (не форсирует переподъём)
	if err := s.reload(base); err != nil {
		t.Fatal(err)
	}
	got := s.cfgSnapshot()
	if got.table["muse"] != oldMuse || got.table["qwen3.8-27b"] != oldQwen {
		t.Fatal("unchanged models must keep old pointers")
	}

	// смена env у qwen -> у неё новый указатель, у muse тот же
	modPath := writeTmpConfig(t, strings.Replace(baseConfig, `"MTMD_BACKEND_DEVICE": "none"`, `"MTMD_BACKEND_DEVICE": "cuda"`, 1))
	if err := s.reload(modPath); err != nil {
		t.Fatal(err)
	}
	nc := s.cfgSnapshot()
	if nc.table["muse"] != oldMuse {
		t.Fatal("muse should keep the old pointer")
	}
	if nc.table["qwen3.8-27b"] == oldQwen {
		t.Fatal("qwen should get a new pointer after env change")
	}

	// битый конфиг -> старый остаётся
	if err := s.reload(writeTmpConfig(t, `{"models": []}`)); err == nil {
		t.Fatal("want error for empty models")
	}
	if s.cfgSnapshot() != nc {
		t.Fatal("config must be preserved after a failed reload")
	}
}

func TestBackendArgs(t *testing.T) {
	// --port из конфига не должен дублироваться при старте
	m := &ModelDef{
		Name: "x",
		Args: []string{"-c", "1", "--port", "9999", "--metrics"},
	}
	got := strings.Join(backendArgs(m, "8081"), " ")
	if strings.Contains(got, "9999") || !strings.HasSuffix(got, "--port 8081") {
		t.Fatalf("bad args: %q", got)
	}
	// без --port в конфиге — просто дописывается
	m2 := &ModelDef{Name: "y", Args: []string{"-c", "2"}}
	if got := strings.Join(backendArgs(m2, "1234"), " "); got != "-c 2 --port 1234" {
		t.Fatalf("bad args: %q", got)
	}
	// --port в самом конце не должен убить значение следующей итерации
	m3 := &ModelDef{Name: "z", Args: []string{"--port", "7777"}}
	if got := strings.Join(backendArgs(m3, "8081"), " "); got != "--port 8081" {
		t.Fatalf("bad args: %q", got)
	}
}

// Свитч не должен прерывать идущую генерацию: пока в бэкенде висит
// запрос к текущей модели, запрос к другой модели ждёт его завершения,
// а бэкенд при этом не убивается.
func TestSwitchWaitsForInflightGeneration(t *testing.T) {
	// fake llama-server для модели "a": "генерация" завершается только
	// после закрытия канала genDone (имитация долгого стрима).
	genDone := make(chan struct{})
	var hitA int32
	fakeA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			atomic.AddInt32(&hitA, 1)
			<-genDone
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"model":"fake-a"}`))
			return
		}
		http.NotFound(w, r)
	}))
	defer fakeA.Close()

	host := strings.TrimPrefix(fakeA.URL, "http://")
	c, err := loadConfig(writeTmpConfig(t, fmt.Sprintf(`{
	  "backend_addr": %q,
	  "llama_bin": "/x/llama-server-does-not-exist",
	  "models": [
	    {"name": "a", "aliases": ["a"], "args": ["-c", "1"]},
	    {"name": "b", "aliases": ["b"], "args": ["-c", "1"]}
	  ]
	}`, host)))
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}

	s := &server{cfg: c, startTime: time.Now(), idle: newClosedChan()}
	s.proxy = newProxy(s)
	s.mu.Lock()
	s.current = "a"
	s.loadedDef = c.resolve("a")
	s.backendID = "fake-a"
	s.mu.Unlock()

	ls, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	proxySrv := &http.Server{Handler: s}
	go proxySrv.Serve(ls)
	defer proxySrv.Close()
	base := "http://" + ls.Addr().String()

	client := &http.Client{Timeout: 15 * time.Second}
	post := func(model string) chan *http.Response {
		ch := make(chan *http.Response, 1)
		go func() {
			req, err := http.NewRequest(http.MethodPost, base+"/v1/chat/completions",
				strings.NewReader(`{"model":`+model+`}`))
			if err != nil {
				t.Errorf("new request %s: %v", model, err)
				ch <- nil
				return
			}
			resp, err := client.Do(req)
			if err != nil {
				t.Errorf("post %s: %v", model, err)
				ch <- nil
				return
			}
			ch <- resp
		}()
		return ch
	}

	// 1. Долгий запрос к "a" — генерация в работе.
	respACh := post(`"a"`)
	deadline := time.Now().Add(5 * time.Second)
	for atomic.LoadInt32(&hitA) == 0 {
		if time.Now().After(deadline) {
			t.Fatal("backend never received the request for model a")
		}
		time.Sleep(10 * time.Millisecond)
	}
	time.Sleep(50 * time.Millisecond) // запрос "a" точно учтён в in-flight

	// 2. Запрос к "b" должен зависнуть, пока генерация "a" не закончилась.
	respBCh := post(`"b"`)
	select {
	case r := <-respBCh:
		if r != nil {
			r.Body.Close()
		}
		t.Fatal("request for model b completed before generation for a finished")
	case <-time.After(300 * time.Millisecond):
	}

	// 3. Генерация "a" закончилась — свитч на "b" может идти.
	close(genDone)

	respA := <-respACh
	if respA == nil {
		t.Fatal("no response for model a")
	}
	bodyA, _ := io.ReadAll(respA.Body)
	respA.Body.Close()
	if respA.StatusCode != 200 || !strings.Contains(string(bodyA), "fake-a") {
		t.Fatalf("model a: status=%d body=%s", respA.StatusCode, bodyA)
	}

	respB := <-respBCh
	if respB == nil {
		t.Fatal("no response for model b")
	}
	bodyB, _ := io.ReadAll(respB.Body)
	respB.Body.Close()
	// Сам свитч на "b" не удался (фейковый llama_bin) — но главное:
	// он затеялся только ПОСЛЕ завершения генерации "a".
	if respB.StatusCode != 503 || !strings.Contains(string(bodyB), "model switch failed") {
		t.Fatalf("model b: status=%d body=%s", respB.StatusCode, bodyB)
	}
}

// Отменённый запрос (клиент отвалился, пока стоял в очереди) не должен
// запускать свитч модели — иначе мёртвые запросы + ретраи гоняют
// модели туда-сюда бесконечно.
func TestCanceledQueuedRequestSkipsSwitch(t *testing.T) {
	genDone := make(chan struct{})
	var hitA int32
	fakeA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			atomic.AddInt32(&hitA, 1)
			<-genDone
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"model":"fake-a"}`))
			return
		}
		http.NotFound(w, r)
	}))
	defer fakeA.Close()

	host := strings.TrimPrefix(fakeA.URL, "http://")
	c, err := loadConfig(writeTmpConfig(t, fmt.Sprintf(`{
	  "backend_addr": %q,
	  "llama_bin": "/x/llama-server-does-not-exist",
	  "models": [
	    {"name": "a", "aliases": ["a"], "args": ["-c", "1"]},
	    {"name": "b", "aliases": ["b"], "args": ["-c", "1"]}
	  ]
	}`, host)))
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}

	s := &server{cfg: c, startTime: time.Now(), idle: newClosedChan()}
	s.proxy = newProxy(s)
	s.mu.Lock()
	s.current = "a"
	s.loadedDef = c.resolve("a")
	s.backendID = "fake-a"
	s.mu.Unlock()

	ls, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	proxySrv := &http.Server{Handler: s}
	go proxySrv.Serve(ls)
	defer proxySrv.Close()
	base := "http://" + ls.Addr().String()
	client := &http.Client{Timeout: 15 * time.Second}

	// 1. В работе генерация "a".
	reqA, err := http.NewRequest(http.MethodPost, base+"/v1/chat/completions",
		strings.NewReader(`{"model":"a"}`))
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		resp, err := client.Do(reqA)
		if err != nil {
			t.Errorf("post a: %v", err)
			return
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}()
	deadline := time.Now().Add(5 * time.Second)
	for atomic.LoadInt32(&hitA) == 0 {
		if time.Now().After(deadline) {
			t.Fatal("backend never received the request for model a")
		}
		time.Sleep(10 * time.Millisecond)
	}
	time.Sleep(50 * time.Millisecond) // запрос "a" точно учтён в in-flight

	// 2. Запрос "b" висит в очереди, затем клиент отваливается.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	reqB, err := http.NewRequest(http.MethodPost, base+"/v1/chat/completions",
		strings.NewReader(`{"model":"b"}`))
	if err != nil {
		t.Fatal(err)
	}
	reqB = reqB.WithContext(ctx)
	go func() {
		resp, err := client.Do(reqB)
		if err == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
	}()
	time.Sleep(300 * time.Millisecond)
	cancel() // клиент "b" отвалился
	time.Sleep(300 * time.Millisecond) // обработчик успел выйти из очереди

	// 3. Генерация закончилась — модель всё ещё "a", никакого свитча.
	close(genDone)
	time.Sleep(500 * time.Millisecond)

	resp, err := client.Get(base + "/_admin/status")
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(body), `"current":"a"`) {
		t.Fatalf("model was switched despite canceled queued request: %s", body)
	}
}

// fakeBackendSource — фейковый llama-server: отвечает на /v1/models
// и /v1/chat/completions (задержка 150 мс = "генерация"), чистый
// stdlib, корректно завершается по SIGINT.
const fakeBackendSource = `package main

import (
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	name, port := "fake", "8081"
	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--name":
			if i+1 < len(args) { name = args[i+1]; i++ }
		case "--port":
			if i+1 < len(args) { port = args[i+1]; i++ }
		}
	}
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		os.Exit(0)
	}()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, "{\"data\":[{\"id\":\"%s\"}]}", name)
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(150 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, "{\"model\":\"%s\",\"choices\":[{\"message\":{\"content\":\"ok from %s\"}}]}", name, name)
	})
	if err := http.ListenAndServe("127.0.0.1:"+port, mux); err != nil {
		fmt.Fprintf(os.Stderr, "fake-server %s: %v\n", name, err)
		os.Exit(1)
	}
}
`

// Два параллельных запроса к РАЗНЫМ моделям, оба клиента живые:
// первый делает свитч и обязан ПОЛНОСТЬЮ получить свой ответ; второй
// ждёт его, затем свитчит обратно. В старой версии счётчик in-flight
// увеличивался только ПОСЛЕ освобождения lock свитча, и второй запрос
// успевал "свободным" бэкендом и резал модель первому ровно тогда,
// когда тот только начинал стримить (502 connection refused + ретраи
// = бесконечный пинг-понг моделей).
func TestSwitchQueueServesBothRequests(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("graceful backend stop (SIGINT) is only available on Linux")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("`go` toolchain not available to build the fake backend")
	}

	dir := t.TempDir()
	fakeSrc := filepath.Join(dir, "fakebackend.go")
	if err := os.WriteFile(fakeSrc, []byte(fakeBackendSource), 0o600); err != nil {
		t.Fatal(err)
	}
	fakeBin := filepath.Join(dir, "fakebackend")
	if out, err := exec.Command("go", "build", "-o", fakeBin, fakeSrc).CombinedOutput(); err != nil {
		t.Fatalf("build fake backend: %v\n%s", err, out)
	}

	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := strconv.Itoa(ln.Addr().(*net.TCPAddr).Port)
	ln.Close()

	c, err := loadConfig(writeTmpConfig(t, fmt.Sprintf(`{
	  "backend_addr": "127.0.0.1:%s",
	  "llama_bin": %q,
	  "cwd": %q,
	  "load_timeout": "60s",
	  "models": [
	    {"name": "a", "aliases": ["a"], "args": ["--name", "fake-a"]},
	    {"name": "b", "aliases": ["b"], "args": ["--name", "fake-b"]}
	  ]
	}`, port, fakeBin, dir)))
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}

	s := &server{cfg: c, startTime: time.Now(), idle: newClosedChan()}
	s.proxy = newProxy(s)

	// Модель "a" "уже загружена": фейковый процесс запускаем руками.
	cmdA := exec.Command(fakeBin, "--name", "fake-a", "--port", port)
	cmdA.Stderr = os.Stderr
	if err := cmdA.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cmdA.Process.Kill() })
	doneA := make(chan struct{})
	go func() { _ = cmdA.Wait(); close(doneA) }()
	waitURL := "http://127.0.0.1:" + port + "/v1/models"
	deadline := time.Now().Add(10 * time.Second)
	for {
		if time.Now().After(deadline) {
			t.Fatal("fake backend a never became ready")
		}
		if resp, err := http.Get(waitURL); err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				break
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	s.mu.Lock()
	s.run = &running{cmd: cmdA, done: doneA}
	s.current = "a"
	s.loadedDef = c.resolve("a")
	s.backendID = "fake-a"
	s.mu.Unlock()

	ls, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	proxySrv := &http.Server{Handler: s}
	go proxySrv.Serve(ls)
	defer proxySrv.Close()
	base := "http://" + ls.Addr().String()
	client := &http.Client{Timeout: 30 * time.Second}

	post := func(model string) chan *http.Response {
		ch := make(chan *http.Response, 1)
		go func() {
			req, err := http.NewRequest(http.MethodPost, base+"/v1/chat/completions",
				strings.NewReader(`{"model":`+model+`}`))
			if err != nil {
				t.Errorf("new request %s: %v", model, err)
				ch <- nil
				return
			}
			resp, err := client.Do(req)
			if err != nil {
				t.Errorf("post %s: %v", model, err)
				ch <- nil
				return
			}
			ch <- resp
		}()
		return ch
	}

	// Первый запрос — на свитч ("b"); через 30 мс второй — назад на "a".
	chR1 := post(`"b"`)
	time.Sleep(30 * time.Millisecond)
	chR2 := post(`"a"`)

	resp1 := <-chR1
	if resp1 == nil {
		t.Fatal("no response for request b")
	}
	body1, _ := io.ReadAll(resp1.Body)
	resp1.Body.Close()
	if resp1.StatusCode != 200 || !strings.Contains(string(body1), "ok from fake-b") {
		t.Fatalf("request b: status=%d body=%s (модель b срезали, пока шёл ответ)", resp1.StatusCode, body1)
	}

	resp2 := <-chR2
	if resp2 == nil {
		t.Fatal("no response for request a")
	}
	body2, _ := io.ReadAll(resp2.Body)
	resp2.Body.Close()
	if resp2.StatusCode != 200 || !strings.Contains(string(body2), "ok from fake-a") {
		t.Fatalf("request a: status=%d body=%s", resp2.StatusCode, body2)
	}

	// Последний фейковый бэкенд наследовал stdout теста — останавливаем,
	// чтобы тест-процесс мог выйти (иначе WaitDelay висит 60 с).
	s.stopCurrent()
}
