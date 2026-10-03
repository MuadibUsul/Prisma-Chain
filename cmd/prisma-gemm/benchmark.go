package main

// Benchmark for GEMM_INT8_V1: native (CPU reference) GEMM time, commitment
// times, challenge detection, on-demand tile trace generation, bisection
// rounds and micro-step arbitration. Full-trace construction is NOT part of
// the normal path in v0.1.1 and is therefore not benchmarked as one.

import (
	"crypto/ed25519"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"prismachain/compute/gemmv1"
)

type benchResult struct {
	M                                uint64  `json:"m"`
	N                                uint64  `json:"n"`
	K                                uint64  `json:"k"`
	MacCount                         uint64  `json:"canonical_mac_count"`
	CWU                              float64 `json:"cwu"`
	InputGenerationMs                float64 `json:"input_generation_ms"`
	MatrixCommitmentMs               float64 `json:"matrix_commitment_ms"`
	ReferenceGEMMMs                  float64 `json:"reference_gemm_ms"`
	OutputCommitmentMs               float64 `json:"output_commitment_ms"`
	NormalPathOverheadRatio          float64 `json:"normal_path_overhead_ratio"`
	ChallengeDetectionMs             float64 `json:"challenge_detection_ms"`
	FabricatedWorkerTraceMs          float64 `json:"fabricated_worker_trace_ms"`
	HonestTraceMs                    float64 `json:"honest_tile_trace_ms"`
	OnDemandTraceBytes               int     `json:"on_demand_trace_bytes"`
	BisectionRounds                  int     `json:"bisection_rounds"`
	BisectionMs                      float64 `json:"bisection_ms"`
	ArbitrationMs                    float64 `json:"arbitration_ms"`
	ArbitrationMACs                  int     `json:"arbitration_macs"`
	ChallengeOverheadVsFullRecompute float64 `json:"challenge_overhead_vs_full_recompute"`
	PeakHeapMB                       float64 `json:"peak_heap_mb"`
	Outcome                          string  `json:"outcome"`
}

func cmdBenchmark(args []string) error {
	fs := flag.NewFlagSet("benchmark", flag.ExitOnError)
	var m, n, k uint64
	var sizes, out string
	fs.Uint64Var(&m, "m", 0, "rows of A (single-size mode)")
	fs.Uint64Var(&n, "n", 0, "cols of B")
	fs.Uint64Var(&k, "k", 0, "inner dimension")
	fs.StringVar(&sizes, "sizes", "", "comma-separated cubic sizes, e.g. 128,256,512")
	fs.StringVar(&out, "out", "", "optional path for benchmark-results.json")
	if err := fs.Parse(args); err != nil {
		return err
	}

	var results []benchResult
	switch {
	case sizes != "":
		for _, part := range strings.Split(sizes, ",") {
			var s uint64
			if _, err := fmt.Sscanf(strings.TrimSpace(part), "%d", &s); err != nil {
				return fmt.Errorf("bad size %q", part)
			}
			r, err := runBench(s, s, s)
			if err != nil {
				return err
			}
			results = append(results, *r)
		}
	case m != 0 && n != 0 && k != 0:
		r, err := runBench(m, n, k)
		if err != nil {
			return err
		}
		results = append(results, *r)
	default:
		return fmt.Errorf("benchmark needs --m/--n/--k or --sizes")
	}

	for _, r := range results {
		fmt.Printf("%dx%dx%d | mac=%d (%.1f CWU) | gemm %.0fms | matrix commit %.0fms | output commit %.0fms | overhead %.2fx | detect %.0fms | trace %.0fms (%dB) | bisection %d rounds %.1fms | arbitration %.1fms | peak heap %.0fMB | %s\n",
			r.M, r.N, r.K, r.MacCount, r.CWU,
			r.ReferenceGEMMMs, r.MatrixCommitmentMs, r.OutputCommitmentMs, r.NormalPathOverheadRatio,
			r.ChallengeDetectionMs, r.HonestTraceMs, r.OnDemandTraceBytes,
			r.BisectionRounds, r.BisectionMs, r.ArbitrationMs, r.PeakHeapMB, r.Outcome)
	}
	if out != "" {
		data, err := json.MarshalIndent(results, "", "  ")
		if err != nil {
			return err
		}
		if err := writeFile(out, append(data, '\n')); err != nil {
			return err
		}
		fmt.Printf("written: %s\n", out)
	}
	return nil
}

func writeFile(path string, data []byte) error {
	return os.WriteFile(path, data, 0o600)
}

func runBench(m, n, k uint64) (*benchResult, error) {
	r := &benchResult{M: m, N: n, K: k, ArbitrationMACs: 8 * 8 * 8}
	const seed = 1234
	workerPriv := devKey(devWorkerSeedDomain, seed)

	var ms runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&ms)
	heapStart := ms.HeapAlloc

	t0 := time.Now()
	matrixA := gemmv1.GenTestMatrix('A', seed, m*k)
	matrixB := gemmv1.GenTestMatrix('B', seed, k*n)
	r.InputGenerationMs = fms(t0)

	t0 = time.Now()
	rootA, rootB, err := gemmv1.BuildMatrixRoots(matrixA, matrixB, m, n, k)
	if err != nil {
		return nil, err
	}
	r.MatrixCommitmentMs = fms(t0)

	task, err := gemmv1.NewTaskDescriptor(workerPriv.Public().(ed25519.PublicKey), gemmv1.Int8ToBytes(gemmv1.GenTestMatrix('N', seed, 16)), 1000, m, n, k, rootA[:], rootB[:], 100, 1000)
	if err != nil {
		return nil, err
	}
	taskID := mustTaskID(task)
	assignment := &gemmv1.Assignment{
		TaskID:          taskID,
		WorkerPubKey:    workerPriv.Public().(ed25519.PublicKey),
		AssignmentNonce: gemmv1.Int8ToBytes(gemmv1.GenTestMatrix('M', seed, 16)),
		AcceptedEpoch:   1001,
	}
	assignmentID := mustAssignmentID(assignment)

	// Normal path: GEMM + output tiles + root.
	t0 = time.Now()
	cvals := gemmv1.ReferenceGEMM(matrixA, matrixB, m, n, k)
	gemmMs := fms(t0)
	r.ReferenceGEMMMs = gemmMs
	t0 = time.Now()
	tiles := gemmv1.OutputTiles(cvals, m, n)
	colsC := uint32((n + gemmv1.TileSize - 1) / gemmv1.TileSize)
	leaves := gemmv1.OutputLeaves(taskID, assignmentID, tiles, colsC)
	outputRoot, err := gemmv1.MerkleRoot(leaves)
	if err != nil {
		return nil, err
	}
	_ = outputRoot
	r.OutputCommitmentMs = fms(t0)
	// Development fraud injection: the worker corrupts one output tile so
	// that the benchmark also exercises detection, tracing and bisection.
	lastTileIdx := len(tiles) - 1
	tiles[lastTileIdx][0] += 1
	r.NormalPathOverheadRatio = (r.MatrixCommitmentMs + r.OutputCommitmentMs) / gemmMs
	r.MacCount = task.CanonicalMACCount()
	r.CWU = gemmv1.CWUnits(r.MacCount)

	// Challenge detection: challenger recomputes everything.
	t0 = time.Now()
	challC := gemmv1.ReferenceGEMM(matrixA, matrixB, m, n, k)
	challTiles := gemmv1.OutputTiles(challC, m, n)
	tileI, tileJ, found, err := gemmv1.FindDisputedTile(tiles, challTiles, colsC)
	if err != nil {
		return nil, err
	}
	detectionMs := fms(t0)
	r.ChallengeDetectionMs = detectionMs
	if !found {
		r.Outcome = "optimistic_unchallenged"
	}

	// On-demand tile traces for the LAST tile (worst-case locality).
	lastI := tileI
	lastJ := tileJ
	t0 = time.Now()
	challArt, err := gemmv1.BuildTileTrace(matrixA, matrixB, m, n, k, taskID, assignmentID, lastI, lastJ)
	if err != nil {
		return nil, err
	}
	r.HonestTraceMs = fms(t0)
	r.OnDemandTraceBytes = (int(gemmv1.RSteps(k)) + 1) * 256
	workerArt := challArt
	if tiles[tileI*colsC+tileJ] != challArt.States[gemmv1.RSteps(k)] {
		workerArt = devFabricateTrace(task, matrixA, matrixB, taskID, assignmentID, lastI, lastJ, tiles[tileI*colsC+tileJ])
		r.FabricatedWorkerTraceMs = 0 // included in honest trace time budget
	}

	// Dispute: locks + bisection + arbitration.
	workerCommit, err := gemmv1.BuildTraceCommit(task, assignmentID, gemmv1.Worker, lastI, lastJ, workerArt, 1011)
	if err != nil {
		return nil, err
	}
	challCommit, err := gemmv1.BuildTraceCommit(task, assignmentID, gemmv1.Challenger, lastI, lastJ, challArt, 1011)
	if err != nil {
		return nil, err
	}
	cfg := gemmv1.DisputeConfig{
		Task: task, Assignment: assignment, TaskID: taskID,
		TileI: lastI, TileJ: lastJ,
		WorkerTile:     tiles[tileI*colsC+tileJ].CanonicalBytes(),
		ChallengerTile: challTiles[tileI*colsC+tileJ].CanonicalBytes(),
		MatrixARoot:    rootA, MatrixBRoot: rootB, RoundPeriod: 50,
	}
	workerClaim := gemmv1.TraceClaim{Party: gemmv1.Worker, TraceRoot: workerArt.Root,
		InitialState: workerCommit.InitialState, InitialProof: workerCommit.InitialProof,
		FinalState: workerCommit.FinalState, FinalProof: workerCommit.FinalProof}
	challClaim := gemmv1.TraceClaim{Party: gemmv1.Challenger, TraceRoot: challArt.Root,
		InitialState: challCommit.InitialState, InitialProof: challCommit.InitialProof,
		FinalState: challCommit.FinalState, FinalProof: challCommit.FinalProof}
	dispute, err := gemmv1.NewGEMMDispute(cfg, workerClaim, challClaim, 1010)
	if err != nil {
		return nil, err
	}
	t0 = time.Now()
	rounds := 0
	for {
		s := dispute.Status()
		if s.Outcome != gemmv1.Pending || s.ArbitrationReady {
			break
		}
		if _, err := gemmv1.SubmitMidFromTrace(dispute, gemmv1.Worker, workerArt, uint64(1012+2*rounds)); err != nil {
			return nil, err
		}
		if _, err := gemmv1.SubmitMidFromTrace(dispute, gemmv1.Challenger, challArt, uint64(1013+2*rounds)); err != nil {
			return nil, err
		}
		rounds++
		if rounds > 64 {
			return nil, fmt.Errorf("bisection did not converge")
		}
	}
	r.BisectionRounds = rounds
	r.BisectionMs = fms(t0)

	step := dispute.Status().Low
	t0 = time.Now()
	aTile := gemmv1.ExtractATile(matrixA, m, k, uint64(lastI), uint64(step))
	bTile := gemmv1.ExtractBTile(matrixB, k, n, uint64(step), uint64(lastJ))
	aProof, err := gemmv1.ProveInputTile(gemmv1.MatrixIDA, matrixA, matrixB, m, n, k, uint64(lastI), uint64(step))
	if err != nil {
		return nil, err
	}
	bProof, err := gemmv1.ProveInputTile(gemmv1.MatrixIDB, matrixA, matrixB, m, n, k, uint64(step), uint64(lastJ))
	if err != nil {
		return nil, err
	}
	outcome, err := dispute.Arbitrate(aTile, bTile, aProof, bProof, dispute.Status().Deadline)
	if err != nil {
		return nil, err
	}
	r.ArbitrationMs = fms(t0)
	r.Outcome = outcomeName(outcome)

	// Challenge overhead against full recomputation.
	totalChallenge := detectionMs + r.HonestTraceMs + r.BisectionMs + r.ArbitrationMs
	if gemmMs > 0 {
		r.ChallengeOverheadVsFullRecompute = totalChallenge / gemmMs
	}

	runtime.ReadMemStats(&ms)
	r.PeakHeapMB = float64(ms.TotalAlloc-heapStart) / (1 << 20)
	return r, nil
}

func fms(t time.Time) float64 {
	return float64(time.Since(t).Microseconds()) / 1000.0
}
