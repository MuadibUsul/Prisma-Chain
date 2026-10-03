package main

// bench-verify: the v0.1.2 benchmark ladder. Measures, per shape, the
// honest full recomputation (denominator), the Freivalds detection time
// for each round count, the exact one-row localization, and the existing
// v0.1.1 dispute cost on an injected bad tile. Detection cost and
// deterministic arbitration cost are reported separately (invariant 10).

import (
	"crypto/ed25519"
	"encoding/json"
	"flag"
	"fmt"
	"strings"
	"time"

	"prismachain/compute/gemmv1"
	"prismachain/compute/gemmv1/verify"
)

type verifyBenchRow struct {
	Size                int     `json:"size"`
	MacCount            uint64  `json:"canonical_mac_count"`
	FullRecomputeMs     float64 `json:"full_recompute_ms"`
	OutputCommitMs      float64 `json:"output_commitment_ms"`
	Rounds              int     `json:"rounds"`
	FreivaldsMs         float64 `json:"freivalds_ms"`
	FreivaldsMACs       uint64  `json:"freivalds_macs"`
	DetectionRatio      float64 `json:"detection_ratio"`
	RowLocalizationMs   float64 `json:"bad_row_localization_ms"`
	DisputeMs           float64 `json:"existing_dispute_ms"`
	TotalFraudPathMs    float64 `json:"total_fraud_path_ms"`
	TotalFraudPathRatio float64 `json:"total_fraud_path_ratio"`
	BisectionRounds     int     `json:"bisection_rounds"`
	ArbitrationStep     uint32  `json:"arbitration_step"`
	Outcome             string  `json:"outcome"`
	OnDemandTraceBytes  int     `json:"on_demand_trace_bytes"`
}

type verifyBenchSize struct {
	Size                int              `json:"size"`
	MacCount            uint64           `json:"canonical_mac_count"`
	FullRecomputeMs     float64          `json:"full_recompute_ms"`
	OutputCommitMs      float64          `json:"output_commitment_ms"`
	RoundResults        []verifyBenchRow `json:"rounds"`
	RowLocalizationMs   float64          `json:"bad_row_localization_ms"`
	DisputeMs           float64          `json:"existing_dispute_ms"`
	BisectionRounds     int              `json:"bisection_rounds"`
	ArbitrationStep     uint32           `json:"arbitration_step"`
	Outcome             string           `json:"outcome"`
	TotalFraudPathMs    float64          `json:"total_fraud_path_ms"`
	TotalFraudPathRatio float64          `json:"total_fraud_path_ratio"`
	OnDemandTraceBytes  int              `json:"on_demand_trace_bytes"`
}

func cmdBenchVerify(args []string) error {
	fs := flag.NewFlagSet("bench-verify", flag.ExitOnError)
	var sizes, roundsCSV, out string
	fs.StringVar(&sizes, "sizes", "512,1024,2048,4096", "comma-separated cubic sizes")
	fs.StringVar(&roundsCSV, "rounds", "8,16,32,40,64", "comma-separated Freivalds round counts")
	fs.StringVar(&out, "out", "", "output JSON path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	var sizeList, roundList []int
	for _, p := range strings.Split(sizes, ",") {
		var v int
		if _, err := fmt.Sscanf(strings.TrimSpace(p), "%d", &v); err != nil {
			return fmt.Errorf("bad size %q", p)
		}
		sizeList = append(sizeList, v)
	}
	for _, p := range strings.Split(roundsCSV, ",") {
		var v int
		if _, err := fmt.Sscanf(strings.TrimSpace(p), "%d", &v); err != nil {
			return fmt.Errorf("bad rounds %q", p)
		}
		roundList = append(roundList, v)
	}

	results := make([]verifyBenchSize, 0, len(sizeList))
	for _, s := range sizeList {
		row, err := benchVerifySize(s, roundList)
		if err != nil {
			return err
		}
		results = append(results, *row)
		fmt.Printf("%d³ | full recompute %.0fms | freivalds %s | row local %.1fms | dispute %.1fms (%d rounds) | fraud path ratio %.4f\n",
			s, row.FullRecomputeMs, roundsSummary(row.RoundResults), row.RowLocalizationMs,
			row.DisputeMs, row.BisectionRounds, row.TotalFraudPathRatio)
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

func roundsSummary(rows []verifyBenchRow) string {
	parts := make([]string, 0, len(rows))
	for _, r := range rows {
		parts = append(parts, fmt.Sprintf("%dr=%.0fms(%.4f)", r.Rounds, r.FreivaldsMs, r.DetectionRatio))
	}
	return strings.Join(parts, " ")
}

func benchVerifySize(size int, roundList []int) (*verifyBenchSize, error) {
	m := uint64(size)
	n, k := m, m
	const seed = 1234
	workerPriv := devKey(devWorkerSeedDomain, seed)

	a := gemmv1.GenTestMatrix('A', seed, m*k)
	b := gemmv1.GenTestMatrix('B', seed, k*n)
	rootA, rootB, err := gemmv1.BuildMatrixRoots(a, b, m, n, k)
	if err != nil {
		return nil, err
	}
	task, err := gemmv1.NewTaskDescriptor(workerPriv.Public().(ed25519.PublicKey), gemmv1.Int8ToBytes(gemmv1.GenTestMatrix('N', seed, 16)), 1000, m, n, k, rootA[:], rootB[:], 100, 1000)
	if err != nil {
		return nil, err
	}
	taskID := mustTaskID(task)
	assignment := &gemmv1.Assignment{
		TaskID: taskID, WorkerPubKey: workerPriv.Public().(ed25519.PublicKey),
		AssignmentNonce: gemmv1.Int8ToBytes(gemmv1.GenTestMatrix('M', seed, 16)), AcceptedEpoch: 1001,
	}
	assignmentID := mustAssignmentID(assignment)

	out := &verifyBenchSize{Size: size, MacCount: task.CanonicalMACCount()}

	t0 := time.Now()
	c := gemmv1.ReferenceGEMM(a, b, m, n, k)
	out.FullRecomputeMs = ms(t0)

	t0 = time.Now()
	tiles := gemmv1.OutputTiles(c, m, n)
	colsC := uint32((n + gemmv1.TileSize - 1) / gemmv1.TileSize)
	leaves := gemmv1.OutputLeaves(taskID, assignmentID, tiles, colsC)
	if _, err := gemmv1.MerkleRoot(leaves); err != nil {
		return nil, err
	}
	out.OutputCommitMs = ms(t0)

	// Fraudulent C: one element flipped inside the middle tile.
	cFraud := append([]int32(nil), c...)
	badElem := (m/2)*n + n/2
	cFraud[badElem]++

	// Dispute cost (round-count independent): localization + trace +
	// bisection + arbitration on the injected bad tile, mirroring the
	// v0.1.1 flow.
	disputeMs, rounds, step, outcome, traceBytes, err := runDisputeForBench(task, taskID, assignment, assignmentID, a, b, cFraud, tiles, colsC)
	if err != nil {
		return nil, err
	}
	out.DisputeMs = disputeMs
	out.BisectionRounds = rounds
	out.ArbitrationStep = step
	out.Outcome = outcome
	out.OnDemandTraceBytes = traceBytes

	locStart := time.Now()
	loc, found, err := verify.LocalizeFromRows(a, b, cFraud, m, n, k, []uint64{m / 2})
	if err != nil || !found {
		return nil, fmt.Errorf("localization failed during benchmark: found=%v err=%v", found, err)
	}
	// The challenger tile for ChallengeOpen is built from 8 exact row
	// recomputations (the E2E fast path does the same); include them in the
	// localization cost so the fraud-path ratio covers the whole detection
	// -> tile-evidence sequence.
	tileI8 := loc.TileI * 8
	for r8 := uint64(0); r8 < 8; r8++ {
		if tileI8+r8 < m {
			if _, err := verify.ReferenceRowGEMM(a, tileI8+r8, b, m, n, k); err != nil {
				return nil, err
			}
		}
	}
	out.RowLocalizationMs = ms(locStart)

	for _, q := range roundList {
		prof := verify.VerificationProfile{Algorithm: verify.AlgorithmFreivaldsBinaryV1, Rounds: uint16(q)}
		src, err := verify.NewChallengeRandomness(assignmentID)
		if err != nil {
			return nil, err
		}
		t0 := time.Now()
		res, err := verify.VerifyFreivaldsBatch(a, b, cFraud, m, n, k, prof, src)
		if err != nil {
			return nil, err
		}
		fms := ms(t0)
		if res.Passed {
			return nil, fmt.Errorf("benchmark fraud was not detected at size %d rounds %d", size, q)
		}
		row := verifyBenchRow{
			Size: size, MacCount: out.MacCount, FullRecomputeMs: out.FullRecomputeMs,
			OutputCommitMs: out.OutputCommitMs, Rounds: q,
			FreivaldsMs: fms, FreivaldsMACs: res.VerificationMACs,
			DetectionRatio:    fms / out.FullRecomputeMs,
			RowLocalizationMs: out.RowLocalizationMs, DisputeMs: out.DisputeMs,
			BisectionRounds: rounds, ArbitrationStep: step, Outcome: outcome,
			OnDemandTraceBytes: traceBytes,
		}
		row.TotalFraudPathMs = fms + out.RowLocalizationMs + out.DisputeMs
		row.TotalFraudPathRatio = row.TotalFraudPathMs / out.FullRecomputeMs
		out.RoundResults = append(out.RoundResults, row)
	}
	// Size-level aggregates use the 40-round profile for the headline ratio.
	for _, r := range out.RoundResults {
		if r.Rounds == 40 {
			out.TotalFraudPathMs = r.TotalFraudPathMs
			out.TotalFraudPathRatio = r.TotalFraudPathRatio
		}
	}
	if out.TotalFraudPathRatio == 0 && len(out.RoundResults) > 0 {
		last := out.RoundResults[len(out.RoundResults)-1]
		out.TotalFraudPathMs = last.TotalFraudPathMs
		out.TotalFraudPathRatio = last.TotalFraudPathRatio
	}
	return out, nil
}

func runDisputeForBench(task *gemmv1.TaskDescriptor, taskID []byte, assignment *gemmv1.Assignment, assignmentID []byte, a, b []int8, cFraud []int32, honestTiles []gemmv1.State, colsC uint32) (float64, int, uint32, string, int, error) {
	fraudTiles := gemmv1.OutputTiles(cFraud, task.M, task.N)
	tileI := uint32(task.M / 2 / gemmv1.TileSize)
	tileJ := uint32(task.N / 2 / gemmv1.TileSize)
	idx := int(tileI)*int(colsC) + int(tileJ)

	leaves := gemmv1.OutputLeaves(taskID, assignmentID, fraudTiles, colsC)
	levels, err := gemmv1.BuildLevels(leaves)
	if err != nil {
		return 0, 0, 0, "", 0, err
	}
	workerProof, err := gemmv1.ProveLeaf(levels, uint32(idx))
	if err != nil {
		return 0, 0, 0, "", 0, err
	}
	co := &gemmv1.ChallengeOpen{
		ProtocolVersion:      gemmv1.ProtocolVersion,
		TaskID:               taskID,
		AssignmentID:         assignmentID,
		ChallengerPubKey:     make([]byte, 32),
		WorkerPubKey:         assignment.WorkerPubKey,
		DisputedTileI:        uint64(tileI),
		DisputedTileJ:        uint64(tileJ),
		WorkerOutputTile:     fraudTiles[idx].CanonicalBytes(),
		WorkerOutputProof:    workerProof,
		ChallengerOutputTile: honestTiles[idx].CanonicalBytes(),
		ChallengeBond:        1000,
		OpenedEpoch:          1010,
	}
	rSteps := uint32(gemmv1.RSteps(task.K))
	challArt, err := gemmv1.BuildTileTrace(a, b, task.M, task.N, task.K, taskID, assignmentID, tileI, tileJ)
	if err != nil {
		return 0, 0, 0, "", 0, err
	}
	states := append([]gemmv1.State(nil), challArt.States...)
	states[rSteps] = fraudTiles[idx]
	fLeaves := make([]gemmv1.Hash, len(states))
	for step, s := range states {
		fLeaves[step] = gemmv1.LeafTraceState(taskID, assignmentID, tileI, tileJ, uint32(step), s.CanonicalBytes())
	}
	fLevels, err := gemmv1.BuildLevels(fLeaves)
	if err != nil {
		return 0, 0, 0, "", 0, err
	}
	workerArt := &gemmv1.TileTraceArtifacts{States: states, Levels: fLevels, Root: fLevels[len(fLevels)-1][0]}

	cfg := gemmv1.DisputeConfig{
		Task: task, Assignment: assignment, TaskID: taskID,
		TileI: tileI, TileJ: tileJ,
		WorkerTile: co.WorkerOutputTile, ChallengerTile: co.ChallengerOutputTile,
		MatrixARoot: mustMatrixRoot(a, b, task, gemmv1.MatrixIDA),
		MatrixBRoot: mustMatrixRoot(a, b, task, gemmv1.MatrixIDB),
		RoundPeriod: 50,
	}
	workerCommit, err := gemmv1.BuildTraceCommit(task, assignmentID, gemmv1.Worker, tileI, tileJ, workerArt, 1011)
	if err != nil {
		return 0, 0, 0, "", 0, err
	}
	challCommit, err := gemmv1.BuildTraceCommit(task, assignmentID, gemmv1.Challenger, tileI, tileJ, challArt, 1011)
	if err != nil {
		return 0, 0, 0, "", 0, err
	}
	d, err := gemmv1.NewGEMMDispute(cfg,
		gemmv1.TraceClaim{Party: gemmv1.Worker, TraceRoot: workerArt.Root, InitialState: workerCommit.InitialState, InitialProof: workerCommit.InitialProof, FinalState: workerCommit.FinalState, FinalProof: workerCommit.FinalProof},
		gemmv1.TraceClaim{Party: gemmv1.Challenger, TraceRoot: challArt.Root, InitialState: challCommit.InitialState, InitialProof: challCommit.InitialProof, FinalState: challCommit.FinalState, FinalProof: challCommit.FinalProof},
		1010)
	if err != nil {
		return 0, 0, 0, "", 0, err
	}
	t0 := time.Now()
	bisectionRounds := 0
	for {
		s := d.Status()
		if s.Outcome != gemmv1.Pending || s.ArbitrationReady {
			break
		}
		if _, err := gemmv1.SubmitMidFromTrace(d, gemmv1.Worker, workerArt, uint64(1012+2*bisectionRounds)); err != nil {
			return 0, 0, 0, "", 0, err
		}
		if _, err := gemmv1.SubmitMidFromTrace(d, gemmv1.Challenger, challArt, uint64(1013+2*bisectionRounds)); err != nil {
			return 0, 0, 0, "", 0, err
		}
		bisectionRounds++
	}
	step := d.Status().Low
	aTile := gemmv1.ExtractATile(a, task.M, task.K, uint64(tileI), uint64(step))
	bTile := gemmv1.ExtractBTile(b, task.K, task.N, uint64(step), uint64(tileJ))
	aProof, err := gemmv1.ProveInputTile(gemmv1.MatrixIDA, a, b, task.M, task.N, task.K, uint64(tileI), uint64(step))
	if err != nil {
		return 0, 0, 0, "", 0, err
	}
	bProof, err := gemmv1.ProveInputTile(gemmv1.MatrixIDB, a, b, task.M, task.N, task.K, uint64(step), uint64(tileJ))
	if err != nil {
		return 0, 0, 0, "", 0, err
	}
	outcome, err := d.Arbitrate(aTile, bTile, aProof, bProof, d.Status().Deadline)
	if err != nil {
		return 0, 0, 0, "", 0, err
	}
	return ms(t0), bisectionRounds, step, outcomeName(outcome), (int(rSteps) + 1) * 256 * 2, nil
}

func mustMatrixRoot(a, b []int8, task *gemmv1.TaskDescriptor, id byte) gemmv1.Hash {
	ra, rb, err := gemmv1.BuildMatrixRoots(a, b, task.M, task.N, task.K)
	if err != nil {
		panic(err)
	}
	if id == gemmv1.MatrixIDA {
		return ra
	}
	return rb
}

func ms(t time.Time) float64 {
	return float64(time.Since(t).Microseconds()) / 1000.0
}
