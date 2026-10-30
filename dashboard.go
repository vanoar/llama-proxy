package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"log"
	"mime"
	"net/http"
	"strconv"
	"time"
)

// ---------------------------------------------------------------------------
// Дашборд на "/": статическая страница (встроена в бинарь) + JSON API
// под /_admin/api/. Страница сама опрашивает /_admin/api/stats и
// перестаёт это делать, когда вкладка скрыта.
// ---------------------------------------------------------------------------

//go:embed dashboard.html
var dashboardHTML []byte

func (s *server) serveDashboard(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	if r.Method == http.MethodHead {
		return
	}
	_, _ = w.Write(dashboardHTML)
}

func (s *server) serveAdminAPI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	path := r.URL.Path[len("/_admin/api/"):]
	switch {
	case path == "stats" && r.Method == http.MethodGet:
		if s.mon == nil {
			aiError(w, http.StatusServiceUnavailable, "monitor disabled")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(s.mon.stats())
	case path == "logs" && r.Method == http.MethodGet:
		after, _ := strconv.ParseUint(r.URL.Query().Get("after"), 10, 64)
		var lines []string
		var seq uint64
		if s.logs != nil {
			lines, seq = s.logs.tail(after)
		}
		writeJSON(w, http.StatusOK, map[string]any{"lines": lines, "seq": seq})
	case r.Method == http.MethodPost && (path == "switch" || path == "unload" || path == "force"):
		// Требуем JSON: кросс-доменная форма/fetch с другого сайта не
		// сможет отправить такой запрос без CORS-preflight (а его мы не
		// разрешаем) — простая защита от CSRF из браузера.
		if ct, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type")); ct != "application/json" {
			aiError(w, http.StatusUnsupportedMediaType, "Content-Type must be application/json")
			return
		}
		s.serveAdminAction(w, r, path)
	default:
		aiError(w, http.StatusNotFound, "unknown admin endpoint")
	}
}

func (s *server) serveAdminAction(w http.ResponseWriter, r *http.Request, action string) {
	switch action {
	case "force":
		s.mu.Lock()
		if s.forceCh != nil {
			close(s.forceCh)
			s.forceCh = nil
		}
		s.mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]string{"status": "forced"})
	case "unload":
		if !s.tryAdmin() {
			aiError(w, http.StatusConflict, "уже идёт переключение, запущенное с дашборда")
			return
		}
		go func() {
			defer s.doneAdmin()
			s.unload()
		}()
		writeJSON(w, http.StatusAccepted, map[string]string{"status": "unloading"})
	case "switch":
		var req struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			aiError(w, http.StatusBadRequest, "bad json: "+err.Error())
			return
		}
		c := s.cfgSnapshot()
		m := c.resolve(req.Model)
		if m == nil {
			aiError(w, http.StatusNotFound, "unknown model "+strconv.Quote(req.Model)+", available: "+c.knownNames())
			return
		}
		if _, ok := s.loadedIs(m); ok {
			writeJSON(w, http.StatusOK, map[string]string{"status": "already loaded"})
			return
		}
		if !s.tryAdmin() {
			aiError(w, http.StatusConflict, "уже идёт переключение, запущенное с дашборда")
			return
		}
		log.Printf("dashboard: switch to %s requested", m.Name)
		go func() {
			defer s.doneAdmin()
			// count=false: это не запрос генерации, держать in-flight незачем.
			if _, err := s.doEnsureModel(m, context.Background(), false); err != nil {
				log.Printf("dashboard switch to %s failed: %v", m.Name, err)
			}
		}()
		writeJSON(w, http.StatusAccepted, map[string]string{"status": "switching", "model": m.Name})
	}
}

func (s *server) tryAdmin() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.adminBusy {
		return false
	}
	s.adminBusy = true
	return true
}

func (s *server) doneAdmin() {
	s.mu.Lock()
	s.adminBusy = false
	s.mu.Unlock()
}

// unload выгружает модель (освобождает VRAM) с той же мягкой
// остановкой, что и свитч: сначала дожидаемся in-flight запросов.
func (s *server) unload() {
	s.switchMu.Lock()
	defer s.switchMu.Unlock()
	s.setPhase("unloading", "")
	defer s.setPhase("", "")
	_ = s.waitIdle(context.Background())
	log.Printf("dashboard: unloading model")
	s.stopCurrent()
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

// --- состояние для дашборда ---------------------------------------------------

func (s *server) setPhase(phase, target string) {
	s.mu.Lock()
	s.phase, s.phaseTarget, s.phaseSince = phase, target, time.Now()
	if phase == "" && s.forceCh != nil {
		s.forceCh = nil
	}
	s.mu.Unlock()
}

func (s *server) setError(msg string) {
	s.mu.Lock()
	s.lastErr, s.lastErrAt = msg, time.Now()
	s.mu.Unlock()
}

func (s *server) addQueued(model string, d int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.queued == nil {
		s.queued = map[string]int{}
	}
	s.queued[model] += d
	if s.queued[model] <= 0 {
		delete(s.queued, model)
	}
}

func (s *server) backendPID() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.run != nil && s.run.cmd.Process != nil {
		return s.run.cmd.Process.Pid
	}
	return 0
}

func (s *server) proxySnapshot(c *Config) proxyStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	p := proxyStats{
		Current:     s.current,
		BackendID:   s.backendID,
		Phase:       s.phase,
		PhaseTarget: s.phaseTarget,
		UpSeconds:   round1(now.Sub(s.startTime).Seconds()),
		Inflight:    s.inflight,
		AdminBusy:   s.adminBusy,
		BackendPort: c.backendPort(),
		BackendUI:   c.BackendUI,
		LastError:   s.lastErr,
	}
	if s.run != nil && s.run.cmd.Process != nil {
		p.PID = s.run.cmd.Process.Pid
	}
	if s.phase != "" {
		p.PhaseSeconds = round1(now.Sub(s.phaseSince).Seconds())
	}
	if !s.loadedAt.IsZero() {
		p.LoadedSec = round1(now.Sub(s.loadedAt).Seconds())
	}
	if s.lastErr != "" {
		p.LastErrorAgo = round1(now.Sub(s.lastErrAt).Seconds())
	}
	if len(s.queued) > 0 {
		p.QueuedBy = make(map[string]int, len(s.queued))
		for k, v := range s.queued {
			p.QueuedBy[k] = v
			p.Queued += v
		}
	}
	return p
}

// --- журнал запросов -----------------------------------------------------------

// logRequest: служебные GET'ы (скрейп метрик и т.п.) в журнал не пишем.
func logRequest(r *http.Request) bool {
	if r.Method != http.MethodGet {
		return true
	}
	switch r.URL.Path {
	case "/metrics", "/slots", "/props", "/health", "/v1/health":
		return false
	}
	return true
}

// statusWriter запоминает код ответа и объём тела. Unwrap нужен, чтобы
// ReverseProxy (через http.ResponseController) мог делать Flush для SSE.
type statusWriter struct {
	http.ResponseWriter
	status int
	n      int64
}

func (w *statusWriter) WriteHeader(code int) {
	if w.status == 0 && code >= 200 {
		w.status = code
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	n, err := w.ResponseWriter.Write(b)
	w.n += int64(n)
	return n, err
}

func (w *statusWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }
