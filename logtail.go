package main

import (
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// logTail — хвост stdout/stderr llama-server для дашборда. Пишется как
// io.Writer между бэкендом и нашим stdout: держит последние N строк
// (кольцевой буфер) и попутно вытаскивает из строк print_timings
// скорость последнего запроса — бесплатно, без опроса бэкенда.
// ---------------------------------------------------------------------------

const logTailLines = 300

type reqTimings struct {
	Model     string   `json:"model"`
	At        int64    `json:"at"` // unix ms
	PromptN   int      `json:"prompt_n"`
	PromptMs  float64  `json:"prompt_ms"`
	PromptTPS float64  `json:"prompt_tps"`
	GenN      int      `json:"gen_n"`
	GenMs     float64  `json:"gen_ms"`
	GenTPS    float64  `json:"gen_tps"`
	DraftRate *float64 `json:"draft_rate,omitempty"` // доля принятых draft-токенов (speculative / MTP)
	CachedN   int      `json:"cached_n,omitempty"`   // токены промпта из кэша (Strata)
}

type logTail struct {
	mu      sync.Mutex
	lines   []string // кольцо
	next    int
	full    bool
	seq     uint64 // номер последней записанной строки (для инкрементального чтения)
	partial []byte
	model   func() string

	pending reqTimings // собирается из нескольких строк
	last    *reqTimings
	live    liveHint // текущая фаза из лога (Strata: в её /slots фазы нет)
}

// liveHint — что бэкенд пишет в лог прямо сейчас: читает промпт или
// генерирует и с какой скоростью. Нужен для бэкендов без next_token в /slots.
type liveHint struct {
	Phase string // "prompt" | "gen" | ""
	TPS   float64
	At    time.Time
}

func newLogTail(model func() string) *logTail {
	return &logTail{lines: make([]string, logTailLines), model: model}
}

var (
	// prompt eval time =     54.06 ms /    13 tokens (    4.16 ms per token,   240.47 tokens per second)
	//        eval time =    623.35 ms /    50 tokens (   12.47 ms per token,    80.21 tokens per second)
	reTiming = regexp.MustCompile(`(prompt eval|eval) time =\s*([\d.]+) ms /\s*(\d+) tokens .*?([\d.]+) tokens per second`)
	// draft acceptance rate = 0.81250 (   26 accepted /    32 generated)
	reDraft = regexp.MustCompile(`draft acceptance rate = ([\d.]+)`)

	// Strata (serve/server.py):
	// [strata] reading the prompt: 48,078 of 48,083 tokens, 6 s so far
	// [strata] thinking: 96 of budget 160540 (max 214053) tokens, 72.5 tok/s, 8 s
	// [strata] answering: 178 of max 214053 tokens, 76.3 tok/s, 9 s
	// [strata] done: 203 tokens in 9 s (78.1 tok/s) (stop, cancel=False), ...
	reStrataRead = regexp.MustCompile(`\[strata\] reading the prompt: ([\d,]+) of [\d,]+ tokens, (\d+) s`)
	reStrataGen  = regexp.MustCompile(`\[strata\] (?:thinking|answering): .*?([\d.]+) tok/s`)
	reStrataDone = regexp.MustCompile(`\[strata\] done: ([\d,]+) tokens in [\d.]+ s \(([\d.]+) tok/s\)`)
)

func atoiComma(s string) int {
	n, _ := strconv.Atoi(strings.ReplaceAll(s, ",", ""))
	return n
}

func (t *logTail) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	start := 0
	for i, c := range p {
		// \r — прогресс-бары скачивания (-hf): каждую перерисовку
		// считаем отдельной строкой, иначе "строка" растёт бесконечно.
		if c != '\n' && c != '\r' {
			continue
		}
		t.partial = append(t.partial, p[start:i]...)
		start = i + 1
		if len(t.partial) > 0 {
			t.addLine(string(t.partial))
		}
		t.partial = t.partial[:0]
	}
	t.partial = append(t.partial, p[start:]...)
	if len(t.partial) > 4096 {
		t.addLine(string(t.partial))
		t.partial = t.partial[:0]
	}
	return len(p), nil
}

func (t *logTail) addLine(s string) {
	t.lines[t.next] = s
	t.next = (t.next + 1) % len(t.lines)
	if t.next == 0 {
		t.full = true
	}
	t.seq++
	t.parse(s)
}

func (t *logTail) parse(s string) {
	if strings.Contains(s, "[strata]") {
		t.parseStrata(s)
		return
	}
	if m := reTiming.FindStringSubmatch(s); m != nil {
		ms, _ := strconv.ParseFloat(m[2], 64)
		n, _ := strconv.Atoi(m[3])
		tps, _ := strconv.ParseFloat(m[4], 64)
		if m[1] == "prompt eval" {
			t.pending = reqTimings{PromptN: n, PromptMs: ms, PromptTPS: tps}
			return
		}
		t.pending.GenN, t.pending.GenMs, t.pending.GenTPS = n, ms, tps
		done := t.pending
		done.At = time.Now().UnixMilli()
		if t.model != nil {
			done.Model = t.model()
		}
		t.last = &done
		return
	}
	// draft acceptance печатается после eval time того же запроса.
	if m := reDraft.FindStringSubmatch(s); m != nil && t.last != nil {
		if v, err := strconv.ParseFloat(m[1], 64); err == nil {
			t.last.DraftRate = &v
		}
	}
}

func (t *logTail) parseStrata(s string) {
	now := time.Now()
	if m := reStrataRead.FindStringSubmatch(s); m != nil {
		n := atoiComma(m[1])
		sec, _ := strconv.ParseFloat(m[2], 64)
		t.pending = reqTimings{PromptN: n, PromptMs: sec * 1000}
		if sec > 0 {
			t.pending.PromptTPS = round1(float64(n) / sec)
		}
		t.live = liveHint{Phase: "prompt", At: now}
		return
	}
	if m := reStrataGen.FindStringSubmatch(s); m != nil {
		tps, _ := strconv.ParseFloat(m[1], 64)
		t.live = liveHint{Phase: "gen", TPS: tps, At: now}
		return
	}
	if m := reStrataDone.FindStringSubmatch(s); m != nil {
		done := t.pending
		t.pending = reqTimings{}
		done.GenN = atoiComma(m[1])
		done.GenTPS, _ = strconv.ParseFloat(m[2], 64)
		if done.GenTPS > 0 {
			done.GenMs = round1(float64(done.GenN) / done.GenTPS * 1000)
		}
		done.At = now.UnixMilli()
		if t.model != nil {
			done.Model = t.model()
		}
		t.last = &done
		t.live = liveHint{}
	}
}

// liveNow возвращает фазу из лога, если она свежая.
func (t *logTail) liveNow(maxAge time.Duration) liveHint {
	if t == nil {
		return liveHint{}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.live.Phase == "" || time.Since(t.live.At) > maxAge {
		return liveHint{}
	}
	return t.live
}

// tail возвращает строки с номером > after (не больше лимита буфера)
// и номер последней строки.
func (t *logTail) tail(after uint64) ([]string, uint64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	n := t.next
	if t.full {
		n = len(t.lines)
	}
	want := t.seq - after
	if after > t.seq || want > uint64(n) {
		want = uint64(n)
	}
	out := make([]string, 0, want)
	for i := uint64(0); i < want; i++ {
		idx := (t.next - int(want) + int(i) + len(t.lines)) % len(t.lines)
		out = append(out, t.lines[idx])
	}
	return out, t.seq
}

func (t *logTail) lastTimings() *reqTimings {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.last == nil {
		return nil
	}
	c := *t.last
	return &c
}

// ---------------------------------------------------------------------------
// reqLog — активные и последние запросы, прошедшие через прокси.
// ---------------------------------------------------------------------------

const reqLogSize = 40

type reqEntry struct {
	ID     uint64 `json:"id"`
	Start  int64  `json:"start"` // unix ms
	Method string `json:"method"`
	Path   string `json:"path"`
	Model  string `json:"model"`
	Status int    `json:"status"`
	WaitMs int64  `json:"wait_ms"` // ожидание в очереди прокси (свитч модели)
	DurMs  int64  `json:"dur_ms"`
	Bytes  int64  `json:"bytes"`
}

type reqLog struct {
	mu     sync.Mutex
	nextID uint64
	active map[uint64]*reqEntry
	recent []reqEntry // кольцо
	pos    int
	total  uint64
	errors uint64
}

func newReqLog() *reqLog {
	return &reqLog{active: map[uint64]*reqEntry{}, recent: make([]reqEntry, 0, reqLogSize)}
}

// Методы reqLog/logTail допускают nil (сервер без дашборда — в тестах).

func (l *reqLog) begin(method, path string) *reqEntry {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.nextID++
	e := &reqEntry{ID: l.nextID, Start: time.Now().UnixMilli(), Method: method, Path: path}
	l.active[e.ID] = e
	return e
}

// update меняет поля активной записи под локом (её читает снапшот).
func (l *reqLog) update(e *reqEntry, f func(*reqEntry)) {
	if l == nil || e == nil {
		return
	}
	l.mu.Lock()
	f(e)
	l.mu.Unlock()
}

func (l *reqLog) end(e *reqEntry, status int, bytes int64) {
	if l == nil || e == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.active, e.ID)
	e.Status = status
	e.Bytes = bytes
	e.DurMs = time.Now().UnixMilli() - e.Start
	l.total++
	if status >= 500 {
		l.errors++
	}
	if len(l.recent) < reqLogSize {
		l.recent = append(l.recent, *e)
	} else {
		l.recent[l.pos] = *e
	}
	l.pos = (l.pos + 1) % reqLogSize
}

type reqStats struct {
	Total  uint64     `json:"total"`
	Errors uint64     `json:"errors"`
	Active []reqEntry `json:"active"`
	Recent []reqEntry `json:"recent"` // новые первыми
}

func (l *reqLog) snapshot() reqStats {
	if l == nil {
		return reqStats{}
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now().UnixMilli()
	st := reqStats{Total: l.total, Errors: l.errors, Active: make([]reqEntry, 0, len(l.active))}
	for _, e := range l.active {
		c := *e
		c.DurMs = now - c.Start
		st.Active = append(st.Active, c)
	}
	sortEntries(st.Active)
	n := len(l.recent)
	st.Recent = make([]reqEntry, 0, n)
	for i := 1; i <= n; i++ {
		st.Recent = append(st.Recent, l.recent[(l.pos-i+n)%n])
	}
	return st
}

func sortEntries(es []reqEntry) {
	for i := 1; i < len(es); i++ {
		for j := i; j > 0 && es[j].ID < es[j-1].ID; j-- {
			es[j], es[j-1] = es[j-1], es[j]
		}
	}
}
