package main

// verify-fast: the v0.1.2 cheap verification path. Freivalds detection
// over exact int64 arithmetic plus (on mismatch) exact one-row
// recomputation and tile localization. The full GEMM is never recomputed
// on this path; the optional full-recompute estimate is measured
// separately and reported only as a comparison denominator.

import (
	"encoding/json"
	"flag"
	"fmt"
	"time"

	"prismachain/compute/gemmv1"
	"prismachain/compute/gemmv1/verify"
)

func cmdVerifyFast(args []string) error {
	fs := flag.NewFlagSet("verify-fast", flag.ExitOnError)
	var algorithm string
	var rounds int
	var measureFull bool
	fs.StringVar(&algorithm, "algorithm", "freivalds-binary-v1", "verification algorithm")
	fs.IntVar(&rounds, "rounds", 40, "Freivalds rounds")
	fs.BoolVar(&measureFull, "measure-full-recompute", true, "time a full ReferenceGEMM as the comparison denominator")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("verify-fast needs task.json and result.json")
	}
	if algorithm != "freivalds-binary-v1" {
		return fmt.Errorf("unsupported algorithm %q", algorithm)
	}
	task, spec, err := loadTask(fs.Arg(0))
	if err != nil {
		return err
	}
	rf, err := loadResult(fs.Arg(1))
	if err != nil {
		return err
	}
	matrixA, matrixB, err := matricesFor(task, spec)
	if err != nil {
		return err
	}

	// Invariant 7: the received C must hash to the committed output_root
	// before it may be verified.
	c := make([]int32, task.M*task.N)
	workerTiles, err := workerTilesOf(rf)
	if err != nil {
		return err
	}
	for idx, tile := range workerTiles {
		ti, tj := idx/int(rf.OutputTiles.ColsC), idx%int(rf.OutputTiles.ColsC)
		for row := 0; row < gemmv1.TileSize; row++ {
			gi := uint64(ti)*gemmv1.TileSize + uint64(row)
			if gi >= task.M {
				break
			}
			for col := 0; col < gemmv1.TileSize; col++ {
				gj := uint64(tj)*gemmv1.TileSize + uint64(col)
				if gj >= task.N {
					break
				}
				c[gi*task.N+gj] = tile[row*gemmv1.TileSize+col]
			}
		}
	}
	taskID := mustTaskID(task)
	assignmentID := rf.ResultCommit.AssignmentID
	counts := gemmv1.TileCountsFor(task.M, task.N, task.K)
	leaves := gemmv1.OutputLeaves(taskID, assignmentID, workerTiles, uint32(counts.ColsC))
	root, err := gemmv1.MerkleRoot(leaves)
	if err != nil {
		return err
	}
	if root != (gemmv1.Hash)(arrayOf(rf.ResultCommit.OutputRoot)) {
		return fmt.Errorf("OUTPUT_DATA_COMMITMENT_MISMATCH: served C does not hash to output_root")
	}

	prof := verify.VerificationProfile{Algorithm: verify.AlgorithmFreivaldsBinaryV1, Rounds: uint16(rounds)}
	src, err := verify.NewChallengeRandomness(assignmentID)
	if err != nil {
		return err
	}
	started := time.Now()
	res, err := verify.VerifyFreivaldsBatch(matrixA, matrixB, c, task.M, task.N, task.K, prof, src)
	if err != nil {
		return err
	}
	verificationMs := time.Since(started).Seconds() * 1000

	var fullMs float64
	if measureFull {
		t0 := time.Now()
		gemmv1.ReferenceGEMM(matrixA, matrixB, task.M, task.N, task.K)
		fullMs = time.Since(t0).Seconds() * 1000
	}
	report := map[string]any{
		"verification_algorithm":   prof.Algorithm,
		"rounds":                   res.RoundsExecuted,
		"passed":                   res.Passed,
		"detected_bad_rows":        res.ResidualRows,
		"verification_time_ms":     roundMs(verificationMs),
		"verification_macs":        res.VerificationMACs,
		"canonical_mac_count":      task.CanonicalMACCount(),
		"full_recompute_ms":        roundMs(fullMs),
		"full_recompute_measured":  measureFull,
		"theoretical_false_accept": prof.FalseAcceptUpperBound(),
	}
	if !res.Passed && fullMs > 0 {
		report["detection_ratio"] = verificationMs / fullMs
	}
	if !res.Passed {
		loc, found, err := verify.LocalizeFromRows(matrixA, matrixB, c, task.M, task.N, task.K, res.ResidualRows)
		if err != nil {
			return err
		}
		if found {
			report["bad_row"] = loc.Row
			report["bad_col"] = loc.Columns[0]
			report["bad_cols_found"] = len(loc.Columns)
			report["tile_i"] = loc.TileI
			report["tile_j"] = loc.TileJ
			report["next_step"] = "open the v0.1.1 ChallengeOpen for this tile; Freivalds mismatch is not a slashing proof"
		} else {
			report["next_step"] = "residual rows did not localize; rerun with more rounds"
		}
	}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(data))
	if !res.Passed {
		return fmt.Errorf("FRAUD_SUSPECTED: run `prisma-gemm challenge` for the deterministic dispute")
	}
	fmt.Println("PASS: Freivalds detection passed; no challenge")
	return nil
}

func roundMs(v float64) float64 {
	return float64(int64(v*1000)) / 1000
}
