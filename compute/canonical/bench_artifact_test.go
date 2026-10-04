package canonical

// Phase F benchmark artifact: times the reference block execution on this
// machine and writes docs/phase-f-benchmark-results.json. Gated behind
// PRISMA_PHASE_F_BENCH=1 so the default test suite stays fast; the file is
// regenerated, never hand-edited:
//
//   PRISMA_PHASE_F_BENCH=1 go test ./compute/canonical -run TestPhaseFBenchmarkArtifact -count=1 -v

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestPhaseFBenchmarkArtifact(t *testing.T) {
	if os.Getenv("PRISMA_PHASE_F_BENCH") != "1" {
		t.Skip("set PRISMA_PHASE_F_BENCH=1 to regenerate docs/phase-f-benchmark-results.json")
	}
	type blockResult struct {
		Name          string         `json:"name"`
		Config        map[string]int `json:"config"`
		Nodes         int            `json:"nodes"`
		GEMMMACs      int64          `json:"gemm_macs"`
		Runs          int            `json:"runs"`
		SecondsAvg    float64        `json:"seconds_avg"`
		MACsPerSecond float64        `json:"macs_per_second"`
	}
	var blocks []blockResult
	for _, tc := range []struct {
		name string
		cfg  BlockConfig
		runs int
	}{
		{"mini", miniBlockConfig(), 4},
		{"medium", mediumBlockConfig(), 3},
	} {
		fix := buildBlockFixture(t, tc.cfg)
		var total time.Duration
		for i := 0; i < tc.runs; i++ {
			start := time.Now()
			if _, err := ExecuteGraph(fix.graph, fix.inputs, fix.tables); err != nil {
				t.Fatal(err)
			}
			total += time.Since(start)
		}
		avg := total.Seconds() / float64(tc.runs)
		macs := fix.exec.WorkVector.Get("GEMM_MAC", 0)
		blocks = append(blocks, blockResult{
			Name: tc.name,
			Config: map[string]int{
				"seq": tc.cfg.Seq, "d_model": tc.cfg.DModel, "heads": tc.cfg.Heads,
				"head_dim": tc.cfg.HeadDim, "mlp_hidden": tc.cfg.MLPHidden,
			},
			Nodes:         len(fix.graph.Nodes),
			GEMMMACs:      macs,
			Runs:          tc.runs,
			SecondsAvg:    avg,
			MACsPerSecond: float64(macs) / avg,
		})
		t.Logf("%s: %.3fs/run, %.1fM MAC/s", tc.name, avg, float64(macs)/avg/1e6)
	}

	artifact := map[string]any{
		"phase": "F",
		"date":  time.Now().UTC().Format("2006-01-02"),
		"environment": map[string]any{
			"go":        runtime.Version(),
			"os":        runtime.GOOS,
			"arch":      runtime.GOARCH,
			"cpu_count": runtime.NumCPU(),
			"note":      "single-threaded scalar reference path (the protocol reference; no GPU, no threads)",
		},
		"blocks": blocks,
		"gpu": map[string]any{
			"status": "NOT TESTED",
			"reason": "no GPU backend is part of CANONICAL_GRAPH_V1 v1; the Python/Go references define bit-exactness and a GPU backend must reproduce them before it can be admitted",
		},
		"watcher": map[string]any{
			"status": "NOT TESTED",
			"reason": "graph watcher fast-verification used the Phase E GEMM watcher path only; the block-level watcher E2E (Phase F DoD) was not run in this milestone",
		},
		"real_model_block": map[string]any{
			"status": "NOT TESTED",
			"reason": "no pinned open-model block was converted; see docs/transformer-block-v1.md for the conversion plan",
		},
	}

	path := filepath.Join("..", "..", "docs", "phase-f-benchmark-results.json")
	raw, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("written: %s", path)
}
