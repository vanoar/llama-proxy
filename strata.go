package main

import (
	"encoding/json"
	"strings"
)

// ---------------------------------------------------------------------------
// Strata (serve/server.py) вместо llama-server: её GET /metrics без
// Accept: text/plain отдаёт JSON монитора — живое состояние (в т.ч.
// запрос, идущий вне батч-слотов), очередь, итоги и последние запросы.
// ---------------------------------------------------------------------------

type strataDoc struct {
	Engine *struct {
		Version string `json:"version"`
	} `json:"engine"`
	Live *struct {
		State        string   `json:"state"` // idle | reading | generating | unloaded
		Queued       int      `json:"queued"`
		PromptRead   *int     `json:"prompt_read"`
		PromptTotal  *int     `json:"prompt_total"`
		PromptTokens *int     `json:"prompt_tokens"`
		Generated    *int64   `json:"generated"`
		ElapsedS     *float64 `json:"elapsed_s"`
		TokS         *float64 `json:"tok_s"`
		Parallel     int      `json:"parallel"`
		Running      int      `json:"running"`
		OutsideSlots int      `json:"outside_slots"`
		Waiting      int      `json:"waiting"`
		Slots        []struct {
			Slot         int      `json:"slot"`
			State        string   `json:"state"` // idle | reading | decoding
			PromptTokens int      `json:"prompt_tokens"`
			PromptRead   *int     `json:"prompt_read"`
			Generated    int64    `json:"generated"`
			ElapsedS     float64  `json:"elapsed_s"`
			TokS         *float64 `json:"tok_s"`
		} `json:"slots"`
	} `json:"live"`
	Requests []struct {
		Time           float64 `json:"time"`
		PromptRead     int     `json:"prompt_read"`
		Reused         int     `json:"reused"`
		OutputTokens   int     `json:"output_tokens"`
		PromptMs       float64 `json:"prompt_ms"`
		DecodeMs       float64 `json:"decode_ms"`
		DecodeTokS     float64 `json:"decode_tok_s"`
		DraftsOffered  int     `json:"drafts_offered"`
		DraftsAccepted int     `json:"drafts_accepted"`
	} `json:"requests"`
	Totals struct {
		PromptTokens float64 `json:"prompt_tokens"`
		Reused       float64 `json:"reused"`
		OutputTokens float64 `json:"output_tokens"`
		PromptMs     float64 `json:"prompt_ms"`
		DecodeMs     float64 `json:"decode_ms"`
	} `json:"totals"`
}

// applyStrata заполняет b из JSON-метрик Strata; false — это не Strata.
func applyStrata(b *backendStats, body []byte) bool {
	var d strataDoc
	if json.Unmarshal(body, &d) != nil || d.Live == nil || d.Engine == nil {
		return false
	}
	lv := d.Live
	b.Engine, b.Metrics, b.SlotsAPI = "strata", true, true
	nctx := 0
	if b.Props != nil {
		nctx = b.Props.NCtx
	}
	phase := func(s string) string {
		switch s {
		case "idle", "":
			return "idle"
		case "reading":
			return "prompt"
		}
		return "gen" // decoding / generating
	}
	slotGen := false
	for _, s := range lv.Slots {
		st := slotStats{ID: s.Slot, NCtx: nctx, State: phase(s.State), NDecoded: s.Generated, TPS: s.TokS, Elapsed: s.ElapsedS}
		if st.State == "prompt" {
			st.PromptTotal = s.PromptTokens
			if s.PromptRead != nil {
				st.PromptRead = *s.PromptRead
			}
		}
		if st.State == "gen" {
			slotGen = true
		}
		b.Slots = append(b.Slots, st)
	}
	// Запрос вне слотов (одиночный «solo path» или без --batch вообще):
	// его состояние — это live.* верхнего уровня.
	if lv.State != "idle" && lv.State != "unloaded" && (lv.Parallel == 0 || lv.OutsideSlots > 0) {
		st := slotStats{ID: -1, Solo: true, NCtx: nctx, State: phase(lv.State)}
		if lv.Generated != nil {
			st.NDecoded = *lv.Generated
		}
		if lv.ElapsedS != nil {
			st.Elapsed = *lv.ElapsedS
		}
		if st.State == "gen" && !slotGen {
			st.TPS = lv.TokS // tok_s — сумма по всем запросам; делить не на что
		}
		if lv.PromptRead != nil && lv.PromptTotal != nil {
			st.PromptRead, st.PromptTotal = *lv.PromptRead, *lv.PromptTotal
		} else if lv.PromptTokens != nil {
			st.PromptTotal = *lv.PromptTokens
		}
		b.Slots = append(b.Slots, st)
	}
	for _, s := range b.Slots {
		switch s.State {
		case "prompt":
			b.Prompting++
		case "gen":
			b.Generating++
		}
	}
	b.Processing = b.Prompting + b.Generating
	if lv.Running > b.Processing {
		b.Processing = lv.Running
	}
	b.Deferred = max(lv.Queued, lv.Waiting)
	if lv.TokS != nil && b.Generating > 0 {
		v := round1(*lv.TokS)
		b.GenLiveTPS = &v
	} else {
		z := 0.0
		b.GenLiveTPS = &z
	}

	t := d.Totals
	b.PromptTotal = t.PromptTokens - t.Reused // прочитано (без токенов из кэша)
	b.GenTotal = t.OutputTokens
	if t.PromptMs > 0 {
		b.PromptAvgTPS = round1(b.PromptTotal / (t.PromptMs / 1000))
	}
	if t.DecodeMs > 0 {
		b.GenAvgTPS = round1(b.GenTotal / (t.DecodeMs / 1000))
	}

	if len(d.Requests) > 0 { // новые первыми
		r := d.Requests[0]
		lt := &reqTimings{
			At:      int64(r.Time * 1000),
			PromptN: r.PromptRead, PromptMs: r.PromptMs, CachedN: r.Reused,
			GenN: r.OutputTokens, GenMs: r.DecodeMs, GenTPS: r.DecodeTokS,
		}
		if r.PromptMs > 0 {
			lt.PromptTPS = round1(float64(r.PromptRead) / (r.PromptMs / 1000))
		}
		if r.DraftsOffered > 0 {
			v := float64(r.DraftsAccepted) / float64(r.DraftsOffered)
			lt.DraftRate = &v
		}
		b.last = lt
	}
	if d.Engine.Version != "" && b.Props != nil && b.Props.Build == "" {
		b.Props.Build = "strata " + strings.TrimSpace(d.Engine.Version)
	}
	return true
}
