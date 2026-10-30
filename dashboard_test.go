package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func TestParseCPUTimes(t *testing.T) {
	in := "cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 60 0 20 400 20 0 0 0 0 0\ncpu1 40 0 30 400 30 0 0 0 0 0\nintr 123\n"
	ts := parseCPUTimes([]byte(in))
	if len(ts) != 3 || ts[0].total != 1000 || ts[0].idle != 850 {
		t.Fatalf("got %+v", ts)
	}
	a := cpuTimes{idle: 850, total: 1000}
	b := cpuTimes{idle: 900, total: 1100} // 100 тиков, из них 50 простоя
	if p := cpuPct(a, b); p != 50 {
		t.Fatalf("cpuPct = %v", p)
	}
}

func TestParseProm(t *testing.T) {
	in := "# HELP llamacpp:prompt_tokens_total x\n# TYPE llamacpp:prompt_tokens_total counter\nllamacpp:prompt_tokens_total 1234\nllamacpp:requests_deferred{a=\"b\"} 2\n"
	m := parseProm([]byte(in))
	if m["llamacpp:prompt_tokens_total"] != 1234 || m["llamacpp:requests_deferred"] != 2 {
		t.Fatalf("got %v", m)
	}
}

func TestParseSlotsBothShapes(t *testing.T) {
	in := `[{"id":1,"id_task":7,"n_ctx":4096,"is_processing":true,"next_token":[{"n_decoded":5,"n_remain":-1}]},
	        {"id":0,"id_task":3,"n_ctx":4096,"is_processing":false,"next_token":{"n_decoded":9,"n_remain":0}}]`
	s := parseSlots([]byte(in))
	if len(s) != 2 || s[0].ID != 0 || s[0].NDecoded != 9 || s[1].NDecoded != 5 || !s[1].Processing {
		t.Fatalf("got %+v", s)
	}
}

func TestApplySlotsLiveSpeed(t *testing.T) {
	m := &monitor{slots: map[int]slotPrev{}}
	now := timeNowForTest()
	b := &backendStats{}
	m.applySlots(b, []rawSlot{{ID: 0, Task: 1, Processing: true, HasPhase: true, NDecoded: 10}}, now, liveHint{})
	b = &backendStats{}
	m.applySlots(b, []rawSlot{{ID: 0, Task: 1, Processing: true, HasPhase: true, NDecoded: 110}}, now.Add(2e9), liveHint{})
	if b.GenLiveTPS == nil || *b.GenLiveTPS != 50 || b.Generating != 1 {
		t.Fatalf("live tps = %v, gen=%d", b.GenLiveTPS, b.Generating)
	}
	// новая задача в слоте — скорость не считаем по чужому baseline
	b = &backendStats{}
	m.applySlots(b, []rawSlot{{ID: 0, Task: 2, Processing: true, HasPhase: true, NDecoded: 0}}, now.Add(4e9), liveHint{})
	if b.Prompting != 1 || b.Slots[0].TPS != nil {
		t.Fatalf("prompt phase: %+v", b)
	}
}

func TestParseGPULine(t *testing.T) {
	g, ok := parseGPULine("0, NVIDIA GeForce RTX 4090, 65, 97, 41, 21800, 24564, 351.20, 450.00, [N/A], 2520, 3105, P2")
	if !ok || g.Name != "NVIDIA GeForce RTX 4090" || *g.Util != 97 || g.Fan != nil || g.PState != "P2" {
		t.Fatalf("got %+v ok=%v", g, ok)
	}
	g, ok = parseGPULine("1, Weird, Name, 50, 1, 2, 3, 4, 5, 6, 7, 8, 9, P0")
	if !ok || g.Name != "Weird,Name" || *g.Temp != 50 {
		t.Fatalf("comma name: %+v", g)
	}
	if _, ok := parseGPULine("garbage"); ok {
		t.Fatal("garbage parsed")
	}
}

func TestLogTailTimings(t *testing.T) {
	lt := newLogTail(func() string { return "m" })
	_, _ = lt.Write([]byte("slot print_timing: id  0 | task 0 | \nprompt eval time =      54.06 ms /    13 tokens (    4.16 ms per token,   240.47 tokens per second)\n       eval time =     623.35 ms /    50 to"))
	_, _ = lt.Write([]byte("kens (   12.47 ms per token,    80.21 tokens per second)\n      total time =     677.41 ms /    63 tokens\ndraft acceptance rate = 0.81250 (   26 accepted /    32 generated)\ndownloading 10%\rdownloading 20%\r"))
	r := lt.lastTimings()
	if r == nil || r.PromptN != 13 || r.PromptTPS != 240.47 || r.GenN != 50 || r.GenTPS != 80.21 || r.Model != "m" || r.DraftRate == nil || *r.DraftRate != 0.8125 {
		t.Fatalf("got %+v", r)
	}
	lines, seq := lt.tail(0)
	if seq != 7 || len(lines) != 7 || lines[6] != "downloading 20%" {
		t.Fatalf("tail: seq=%d %q", seq, lines)
	}
	if more, _ := lt.tail(seq); len(more) != 0 {
		t.Fatalf("incremental tail returned %q", more)
	}
}

func TestReqLogRing(t *testing.T) {
	l := newReqLog()
	for i := 0; i < reqLogSize+5; i++ {
		e := l.begin("POST", "/x")
		l.end(e, 200, 1)
	}
	e := l.begin("POST", "/active")
	st := l.snapshot()
	if len(st.Recent) != reqLogSize || st.Recent[0].ID != uint64(reqLogSize+5) || len(st.Active) != 1 || st.Active[0].ID != e.ID {
		t.Fatalf("recent=%d first=%d active=%v", len(st.Recent), st.Recent[0].ID, st.Active)
	}
}

func TestDashboardRoutes(t *testing.T) {
	c := &Config{BackendAddr: "127.0.0.1:1", Models: []*ModelDef{{Name: "a"}}}
	if err := c.buildTable(); err != nil {
		t.Fatal(err)
	}
	s := &server{cfg: c, idle: newClosedChan(), reqs: newReqLog()}
	s.logs = newLogTail(nil)
	s.mon = newMonitor(s)

	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "<title>llama-proxy</title>") {
		t.Fatalf("GET / = %d", rec.Code)
	}
	rec = httptest.NewRecorder()
	s.ServeHTTP(rec, httptest.NewRequest("GET", "/health", nil))
	if rec.Body.String() != "llama-proxy ok\n" {
		t.Fatalf("/health = %q", rec.Body.String())
	}
	rec = httptest.NewRecorder()
	s.ServeHTTP(rec, httptest.NewRequest("GET", "/_admin/api/stats", nil))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"models":[{"name":"a"`) {
		t.Fatalf("stats = %d %s", rec.Code, rec.Body.String())
	}
	// без JSON Content-Type действие не выполняется (защита от CSRF)
	rec = httptest.NewRecorder()
	s.ServeHTTP(rec, httptest.NewRequest("POST", "/_admin/api/switch", strings.NewReader(`{"model":"a"}`)))
	if rec.Code != http.StatusUnsupportedMediaType {
		t.Fatalf("switch without json ct = %d", rec.Code)
	}
	req := httptest.NewRequest("POST", "/_admin/api/switch", strings.NewReader(`{"model":"nope"}`))
	req.Header.Set("Content-Type", "application/json")
	rec = httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("switch unknown = %d", rec.Code)
	}
}

func timeNowForTest() time.Time { return time.Unix(1_700_000_000, 0) }

// SSE должен стримиться через statusWriter (Flush через Unwrap), а не
// копиться до конца ответа.
func TestStreamingThroughStatusWriter(t *testing.T) {
	release := make(chan struct{})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: first\n\n"))
		w.(http.Flusher).Flush()
		<-release
		_, _ = w.Write([]byte("data: done\n\n"))
	}))
	defer backend.Close()
	defer close(release)

	c := &Config{BackendAddr: strings.TrimPrefix(backend.URL, "http://"), Models: []*ModelDef{{Name: "a"}}}
	if err := c.buildTable(); err != nil {
		t.Fatal(err)
	}
	s := &server{cfg: c, idle: newClosedChan(), reqs: newReqLog(), current: "a", loadedDef: c.Models[0]}
	s.proxy = newProxy(s)
	front := httptest.NewServer(s)
	defer front.Close()

	resp, err := http.Post(front.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"a","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got := make(chan string, 1)
	go func() {
		buf := make([]byte, 64)
		n, _ := resp.Body.Read(buf)
		got <- string(buf[:n])
	}()
	select {
	case s := <-got:
		if !strings.Contains(s, "first") {
			t.Fatalf("first chunk = %q", s)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("first SSE chunk was not flushed to the client")
	}
	if st := s.reqs.snapshot(); len(st.Active) != 1 || st.Active[0].Model != "a" {
		t.Fatalf("active requests = %+v", st.Active)
	}
}

func TestStrataLogAndSlots(t *testing.T) {
	lt := newLogTail(func() string { return "flash" })
	_, _ = lt.Write([]byte("[strata] reading the prompt: 48,078 of 48,083 tokens, 6 s so far\n"))
	if h := lt.liveNow(time.Minute); h.Phase != "prompt" {
		t.Fatalf("hint after read = %+v", h)
	}
	_, _ = lt.Write([]byte("[strata] thinking: 96 of budget 160540 (max 214053) tokens, 72.5 tok/s, 8 s\n"))
	h := lt.liveNow(time.Minute)
	if h.Phase != "gen" || h.TPS != 72.5 {
		t.Fatalf("hint after thinking = %+v", h)
	}
	// Strata /slots: без next_token — фаза и скорость из лога.
	m := &monitor{slots: map[int]slotPrev{}}
	b := &backendStats{}
	m.applySlots(b, parseSlots([]byte(`[{"id":0,"n_ctx":262144,"is_processing":true,"n_prompt_tokens":48083}]`)), timeNowForTest(), h)
	if b.Generating != 1 || b.GenLiveTPS == nil || *b.GenLiveTPS != 72.5 {
		t.Fatalf("strata slot: gen=%d live=%v", b.Generating, b.GenLiveTPS)
	}
	_, _ = lt.Write([]byte("[strata] done: 203 tokens in 9 s (78.1 tok/s) (stop, cancel=False), expert cache 98.0% hit\n"))
	r := lt.lastTimings()
	if r == nil || r.PromptN != 48078 || r.PromptTPS != 8013 || r.GenN != 203 || r.GenTPS != 78.1 || r.Model != "flash" {
		t.Fatalf("strata timings = %+v", r)
	}
	if h := lt.liveNow(time.Minute); h.Phase != "" {
		t.Fatalf("hint after done = %+v", h)
	}
}

func TestPlausibleTemp(t *testing.T) {
	for _, c := range []struct {
		kind string
		v    float64
		ok   bool
	}{{"other", 45, true}, {"other", 127, false}, {"other", -59.5, false}, {"other", 109, false}, {"other", 0, false},
		{"cpu", 40, true}, {"cpu", 105, true}, {"cpu", 0, false}, {"nvme", 127, false}} {
		if got := plausibleTemp(c.kind, c.v); got != c.ok {
			t.Errorf("plausibleTemp(%s, %v) = %v", c.kind, c.v, got)
		}
	}
}

func TestEstimatePower(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	gpus := []gpuStats{{Power: f(250)}, {Power: f(100)}, {Power: nil}}
	base := 40.0
	c := &Config{PowerBaseWatts: &base, PSUEfficiency: 0.9}
	p := estimatePower(c, gpus, f(80), f(10), 135, true, f(50))
	// DC = 350 + 80 + 10 + 40 = 480; от розетки = 480/0.9 = 533.3
	if p.CPUSource != "rapl" || *p.GPUW != 350 || p.DCW != 480 || p.TotalW != 533.3 || p.LossW != 53.3 {
		t.Fatalf("rapl: %+v", p)
	}
	// без прав на RAPL — CPU по загрузке: 135 * (0.2 + 0.8*0.5) = 81
	p = estimatePower(&Config{}, gpus, nil, nil, 135, false, f(50))
	if p.CPUSource != "load" || *p.CPUW != 81 || p.BaseW != defaultPowerBase || p.PSUEff != defaultPSUEff || p.Note == "" {
		t.Fatalf("load: %+v", p)
	}
	if estimatePower(&Config{}, nil, nil, nil, 0, false, nil) != nil {
		t.Fatal("expected nil without any source")
	}
}

func TestRAPLWrap(t *testing.T) {
	dir := t.TempDir()
	file := dir + "/energy_uj"
	write := func(v string) {
		if err := os.WriteFile(file, []byte(v+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	z := &raplZone{path: file, name: "package-0", max: 1_000_000_000}
	r := &raplReader{scanned: time.Now(), zones: []*raplZone{z}}
	write("999000000")
	r.read()
	z.prevAt = z.prevAt.Add(-time.Second) // как будто прошла секунда
	write("99000000")                     // переполнение: +1 Дж до max, +99 Дж после
	cpu, _, _, ok := r.read()
	if !ok || cpu == nil || *cpu < 99 || *cpu > 101 {
		t.Fatalf("cpu = %v ok=%v", cpu, ok)
	}
}

// Формат JSON /metrics Strata: одиночный запрос идёт вне батч-слотов.
const strataMetricsSample = `{"engine": {"version": "0.1.52", "batch_slots": 2},
 "live": {"state": "generating", "queued": 0, "phase": "answering", "prompt_tokens": 2836, "prompt_read": null, "prompt_total": null,
  "generated": 1200, "elapsed_s": 16.2, "tok_s": 78.4, "parallel": 2, "running": 1,
  "slots": [{"slot": 0, "state": "idle", "held_tokens": 0}, {"slot": 1, "state": "reading", "prompt_tokens": 40000, "prompt_read": 10000, "generated": 0, "elapsed_s": 3.0, "tok_s": null}],
  "outside_slots": 1, "waiting": 2, "admitting": 0},
 "requests": [{"time": 1793349315.5, "prompt_tokens": 2836, "reused": 2768, "output_tokens": 4640, "prompt_read": 68, "prompt_ms": 762.3, "decode_ms": 59498.3, "decode_tok_s": 78.0, "drafts_offered": 3896, "drafts_accepted": 2284}],
 "totals": {"prompt_tokens": 5609, "reused": 2768, "output_tokens": 4791, "prompt_ms": 7175.2, "decode_ms": 61933.8}}`

func TestApplyStrata(t *testing.T) {
	b := &backendStats{Props: &backendProps{NCtx: 262144}}
	if !applyStrata(b, []byte(strataMetricsSample)) {
		t.Fatal("not detected as strata")
	}
	if b.Engine != "strata" || len(b.Slots) != 3 || b.Generating != 1 || b.Prompting != 1 || b.Deferred != 2 {
		t.Fatalf("live: %+v", b)
	}
	solo := b.Slots[2]
	if !solo.Solo || solo.State != "gen" || solo.NDecoded != 1200 || solo.TPS == nil || *solo.TPS != 78.4 || solo.NCtx != 262144 {
		t.Fatalf("solo row: %+v", solo)
	}
	if s := b.Slots[1]; s.State != "prompt" || s.PromptRead != 10000 || s.PromptTotal != 40000 {
		t.Fatalf("reading slot: %+v", s)
	}
	if *b.GenLiveTPS != 78.4 || b.PromptTotal != 2841 || b.GenAvgTPS != 77.4 || b.PromptAvgTPS != 395.9 {
		t.Fatalf("totals: live=%v prompt=%v gen_avg=%v prompt_avg=%v", *b.GenLiveTPS, b.PromptTotal, b.GenAvgTPS, b.PromptAvgTPS)
	}
	l := b.last
	if l == nil || l.PromptN != 68 || l.CachedN != 2768 || l.GenN != 4640 || l.GenTPS != 78 || l.DraftRate == nil || l.PromptTPS != 89.2 {
		t.Fatalf("last: %+v", l)
	}
	// llama-server Prometheus — не Strata
	if applyStrata(&backendStats{}, []byte("llamacpp:requests_processing 1\n")) {
		t.Fatal("prometheus text detected as strata")
	}
}
