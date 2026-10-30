package main

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"
)

// ---------------------------------------------------------------------------
// Энергопотребление сервера. Прямого счётчика «от розетки» на обычной
// плате нет (ни BMC/IPMI, ни ACPI power meter), поэтому считаем оценку:
//   GPU  — nvidia-smi power.draw (точно, по каждой карте);
//   CPU  — Intel RAPL energy_uj (точно; файл по умолчанию читает только
//          root — см. README), иначе оценка по загрузке от TDP;
//   DRAM — RAPL, если у платформы есть такая зона;
//   остальное (плата, диски, вентиляторы) — константа power_base_watts;
//   потери БП — по psu_efficiency.
// ---------------------------------------------------------------------------

const (
	defaultPowerBase = 50.0
	defaultPSUEff    = 0.9
)

type raplZone struct {
	path   string // .../energy_uj
	name   string // package-0, dram
	max    uint64 // max_energy_range_uj (счётчик переполняется)
	tdpW   float64
	prevUJ uint64
	prevAt time.Time
	ok     bool // предыдущее значение валидно
}

type raplReader struct {
	scanned time.Time
	zones   []*raplZone
}

type powerStats struct {
	TotalW    float64  `json:"total_w"` // оценка от розетки
	DCW       float64  `json:"dc_w"`    // на выходе БП
	GPUW      *float64 `json:"gpu_w"`
	CPUW      *float64 `json:"cpu_w"`
	CPUSource string   `json:"cpu_source"` // rapl | load
	DRAMW     *float64 `json:"dram_w,omitempty"`
	BaseW     float64  `json:"base_w"`
	LossW     float64  `json:"loss_w"`
	PSUEff    float64  `json:"psu_eff"`
	Note      string   `json:"note,omitempty"`
}

func (r *raplReader) rescan() {
	r.scanned = time.Now()
	r.zones = nil
	// intel-rapl:N — пакеты, intel-rapl:N:M — подзоны (core/uncore/dram).
	// core/uncore входят в package, поэтому берём только package и dram.
	paths, _ := filepath.Glob("/sys/class/powercap/intel-rapl:*")
	sort.Strings(paths)
	for _, p := range paths {
		name := readTrim(filepath.Join(p, "name"))
		if name != "dram" && !(len(name) > 8 && name[:8] == "package-") {
			continue
		}
		z := &raplZone{path: filepath.Join(p, "energy_uj"), name: name}
		z.max, _ = strconv.ParseUint(readTrim(filepath.Join(p, "max_energy_range_uj")), 10, 64)
		if uw, err := strconv.ParseFloat(readTrim(filepath.Join(p, "constraint_0_power_limit_uw")), 64); err == nil {
			z.tdpW = uw / 1e6
		}
		r.zones = append(r.zones, z)
	}
}

// prime запоминает начальные значения счётчиков (холодный старт дашборда).
func (r *raplReader) prime() {
	if r.scanned.IsZero() || time.Since(r.scanned) > hwmonRescan {
		r.rescan()
	}
	now := time.Now()
	for _, z := range r.zones {
		if v, err := readUint(z.path); err == nil {
			z.prevUJ, z.prevAt, z.ok = v, now, true
		}
	}
}

// read возвращает мощность пакетов CPU и DRAM в ваттах; readable=false,
// если счётчики есть, но прочитать их нельзя (нет прав).
func (r *raplReader) read() (cpuW, dramW *float64, tdp float64, readable bool) {
	if r.scanned.IsZero() || time.Since(r.scanned) > hwmonRescan {
		r.rescan()
	}
	now := time.Now()
	var cpu, dram float64
	var haveCPU, haveDRAM bool
	readable = len(r.zones) > 0
	for _, z := range r.zones {
		if z.name != "dram" {
			tdp += z.tdpW
		}
		v, err := readUint(z.path)
		if err != nil {
			readable = false
			z.ok = false
			continue
		}
		if z.ok {
			dt := now.Sub(z.prevAt).Seconds()
			d := float64(v) - float64(z.prevUJ)
			if d < 0 && z.max > 0 { // переполнение счётчика
				d += float64(z.max)
			}
			if dt > 0 && d >= 0 {
				w := d / 1e6 / dt
				if z.name == "dram" {
					dram += w
					haveDRAM = true
				} else {
					cpu += w
					haveCPU = true
				}
			}
		}
		z.prevUJ, z.prevAt, z.ok = v, now, true
	}
	if haveCPU {
		cpu = round1(cpu)
		cpuW = &cpu
	}
	if haveDRAM {
		dram = round1(dram)
		dramW = &dram
	}
	return cpuW, dramW, tdp, readable
}

func readUint(p string) (uint64, error) {
	b, err := os.ReadFile(p)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(string(trimSpaceBytes(b)), 10, 64)
}

func trimSpaceBytes(b []byte) []byte {
	for len(b) > 0 && (b[len(b)-1] == '\n' || b[len(b)-1] == ' ') {
		b = b[:len(b)-1]
	}
	return b
}

// estimatePower собирает оценку по компонентам.
func estimatePower(c *Config, gpus []gpuStats, cpuW, dramW *float64, tdp float64, raplReadable bool, cpuPct *float64) *powerStats {
	p := &powerStats{BaseW: defaultPowerBase, PSUEff: defaultPSUEff}
	if c.PowerBaseWatts != nil {
		p.BaseW = *c.PowerBaseWatts
	}
	if c.PSUEfficiency > 0 && c.PSUEfficiency <= 1 {
		p.PSUEff = c.PSUEfficiency
	}
	var gpu float64
	have := false
	for _, g := range gpus {
		if g.Power != nil {
			gpu += *g.Power
			have = true
		}
	}
	if have {
		gpu = round1(gpu)
		p.GPUW = &gpu
	}
	switch {
	case cpuW != nil:
		p.CPUW, p.CPUSource = cpuW, "rapl"
		p.DRAMW = dramW
	case tdp > 0 && cpuPct != nil:
		// Грубо: ~20% TDP в простое + остальное пропорционально загрузке.
		v := round1(tdp * (0.2 + 0.8**cpuPct/100))
		p.CPUW, p.CPUSource = &v, "load"
		if !raplReadable {
			p.Note = "нет прав на RAPL: CPU оценён по загрузке"
		}
	}
	if p.GPUW == nil && p.CPUW == nil {
		return nil
	}
	p.DCW = p.BaseW
	for _, v := range []*float64{p.GPUW, p.CPUW, p.DRAMW} {
		if v != nil {
			p.DCW += *v
		}
	}
	p.DCW = round1(p.DCW)
	p.TotalW = round1(p.DCW / p.PSUEff)
	p.LossW = round1(p.TotalW - p.DCW)
	return p
}
