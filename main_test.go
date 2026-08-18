package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
