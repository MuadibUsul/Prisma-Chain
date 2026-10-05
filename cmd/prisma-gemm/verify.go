package main

// Independent verification and the local dispute run: the challenger
// recomputes the task from the descriptor inputs, compares output tiles,
// and on a mismatch drives the full on-demand trace dispute.

import (
	"crypto/ed25519"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"prismachain/compute/gemmv1"
)

func readFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}

func unmarshal(data []byte, v any) error {
	return json.Unmarshal(data, v)
}

type disputeReport struct {
	TaskID              string             `json:"task_id"`
	DisputedTile        [2]int             `json:"disputed_tile"`
	BisectionRounds     int                `json:"bisection_rounds"`
	ArbitrationStep     uint32             `json:"arbitration_step"`
	Outcome             string             `json:"outcome"`
	OnDemandTraceBytes  int                `json:"on_demand_trace_bytes"`
	WorkerTraceRoot     string             `json:"worker_trace_root"`
	ChallengerTraceRoot string             `json:"challenger_trace_root"`
	TranscriptDigest    string             `json:"transcript_digest"`
	WorkerGotVWR        bool               `json:"worker_got_vwr"`
	ReceiptID           string             `json:"receipt_id,omitempty"`
	Detail              *arbitrationDetail `json:"arbitration_detail,omitempty"`
}

type arbitrationDetail struct {
	Step           uint32  `json:"step"`
	ATile          []int32 `json:"a_tile"`
	BTile          []int32 `json:"b_tile"`
	LowState       []int32 `json:"low_state"`
	WorkerHigh     []int32 `json:"worker_high"`
	ChallengerHigh []int32 `json:"challenger_high"`
	Expected       []int32 `json:"expected"`
	ArbitrationMAC int     `json:"arbitration_mac"`
}

func loadResult(path string) (*resultJSON, error) {
	data, err := readFile(path)
	if err != nil {
		return nil, err
	}
	var rf resultJSON
	if err := unmarshal(data, &rf); err != nil {
		return nil, err
	}
	return &rf, nil
}

func workerTilesOf(rf *resultJSON) ([]gemmv1.State, error) {
	tiles := make([]gemmv1.State, len(rf.OutputTiles.Tiles))
	for i, flat := range rf.OutputTiles.Tiles {
		if len(flat) != len(gemmv1.State{}) {
			return nil, fmt.Errorf("tile %d has %d values, want 64", i, len(flat))
		}
		copy(tiles[i][:], flat)
	}
	return tiles, nil
}

// runVerification is the shared challenger path of verify and challenge.
func runVerification(task *gemmv1.TaskDescriptor, spec matrixSpec, rf *resultJSON, verbose bool) (*disputeReport, error) {
	started := time.Now()
	matrixA, matrixB, err := matricesFor(task, spec)
	if err != nil {
		return nil, err
	}

	// The coordinator binds the commit to task and assignment and checks
	// the worker signature before anything else.
	workerPriv := devKey(devWorkerSeedDomain, spec.Seed)
	assignment := &gemmv1.Assignment{
		TaskID:          mustTaskID(task),
		WorkerPubKey:    workerPriv.Public().(ed25519.PublicKey),
		AssignmentNonce: gemmv1.Int8ToBytes(gemmv1.GenTestMatrix('M', spec.Seed, 16)),
		AcceptedEpoch:   task.IssuedEpoch + 1,
	}
	rc := rf.ResultCommit
	if err := gemmv1.ValidateResultCommit(task, assignment, &rc); err != nil {
		return nil, err
	}
	if !gemmv1.VerifyResultCommitSignature(&rc) {
		return nil, fmt.Errorf("ResultCommit signature does not verify")
	}

	// The committed output root must be the root over the submitted tiles.
	workerTiles, err := workerTilesOf(rf)
	if err != nil {
		return nil, err
	}
	taskID := mustTaskID(task)
	assignmentID := mustAssignmentID(assignment)
	levels, workerRoot := mustTree(taskID, assignmentID, workerTiles, rf.OutputTiles.ColsC)
	if workerRoot != (gemmv1.Hash)(arrayOf(rc.OutputRoot)) {
		return nil, fmt.Errorf("submitted tiles do not hash to the committed output_root")
	}

	// Independent recomputation (full verification is allowed in v0.1.1).
	challengerC := gemmv1.ReferenceGEMM(matrixA, matrixB, task.M, task.N, task.K)
	challengerTiles := gemmv1.OutputTiles(challengerC, task.M, task.N)
	detectionTime := time.Since(started)

	tileI, tileJ, found, err := gemmv1.FindDisputedTile(workerTiles, challengerTiles, rf.OutputTiles.ColsC)
	if err != nil {
		return nil, err
	}
	if !found {
		return &disputeReport{
			TaskID:       fmt.Sprintf("%x", taskID),
			Outcome:      "optimistic_unchallenged",
			WorkerGotVWR: true,
		}, nil
	}
	fmt.Printf("fraud detected at output tile (%d,%d) after %s\n", tileI, tileJ, detectionTime.Round(time.Millisecond))

	idx := int(tileI)*int(rf.OutputTiles.ColsC) + int(tileJ)
	workerProof, err := gemmv1.ProveLeaf(levels, uint32(idx))
	if err != nil {
		return nil, err
	}
	co := &gemmv1.ChallengeOpen{
		ProtocolVersion:      gemmv1.ProtocolVersion,
		TaskID:               taskID,
		AssignmentID:         assignmentID,
		ChallengerPubKey:     devKey(devChallengerSeedDomain, spec.Seed).Public().(ed25519.PublicKey),
		WorkerPubKey:         assignment.WorkerPubKey,
		DisputedTileI:        uint64(tileI),
		DisputedTileJ:        uint64(tileJ),
		WorkerOutputTile:     workerTiles[idx].CanonicalBytes(),
		WorkerOutputProof:    workerProof,
		ChallengerOutputTile: challengerTiles[idx].CanonicalBytes(),
		ChallengeBond:        1000,
		OpenedEpoch:          task.IssuedEpoch + 10,
	}
	if err := gemmv1.ValidateChallengeOpen(task, co); err != nil {
		return nil, err
	}
	leaf := gemmv1.LeafOutputTile(taskID, assignmentID, tileI, tileJ, co.WorkerOutputTile)
	if !gemmv1.VerifyLeafInclusion(workerRoot, leaf, workerProof) {
		return nil, fmt.Errorf("worker tile is not a member of the committed output_root")
	}

	// Traces: the challenger's is honest; the worker trace is rebuilt to
	// end at its committed tile (development simulation of a lying
	// worker; on a real network the worker itself would lock its trace).
	challArt, err := gemmv1.BuildTileTrace(matrixA, matrixB, task.M, task.N, task.K, taskID, assignmentID, tileI, tileJ)
	if err != nil {
		return nil, err
	}
	rSteps := uint32(gemmv1.RSteps(task.K))
	workerArt := challArt
	if workerTiles[idx] != challArt.States[rSteps] {
		workerArt = devFabricateTrace(task, matrixA, matrixB, taskID, assignmentID, tileI, tileJ, workerTiles[idx])
	}

	openedEpoch := task.IssuedEpoch + 10
	workerCommit, err := gemmv1.BuildTraceCommit(task, assignmentID, gemmv1.Worker, tileI, tileJ, workerArt, openedEpoch+1)
	if err != nil {
		return nil, err
	}
	challCommit, err := gemmv1.BuildTraceCommit(task, assignmentID, gemmv1.Challenger, tileI, tileJ, challArt, openedEpoch+1)
	if err != nil {
		return nil, err
	}
	cfg := gemmv1.DisputeConfig{
		Task:           task,
		Assignment:     assignment,
		TaskID:         taskID,
		TileI:          tileI,
		TileJ:          tileJ,
		WorkerTile:     co.WorkerOutputTile,
		ChallengerTile: co.ChallengerOutputTile,
		MatrixARoot:    arrayOf(task.MatrixARoot[:32]),
		MatrixBRoot:    arrayOf(task.MatrixBRoot[:32]),
		RoundPeriod:    50,
	}
	workerClaim := gemmv1.TraceClaim{Party: gemmv1.Worker, TraceRoot: workerArt.Root,
		InitialState: workerCommit.InitialState, InitialProof: workerCommit.InitialProof,
		FinalState: workerCommit.FinalState, FinalProof: workerCommit.FinalProof}
	challClaim := gemmv1.TraceClaim{Party: gemmv1.Challenger, TraceRoot: challArt.Root,
		InitialState: challCommit.InitialState, InitialProof: challCommit.InitialProof,
		FinalState: challCommit.FinalState, FinalProof: challCommit.FinalProof}
	dispute, err := gemmv1.NewGEMMDispute(cfg, workerClaim, challClaim, openedEpoch)
	if err != nil {
		return nil, err
	}

	rounds := 0
	for {
		s := dispute.Status()
		if s.Outcome != gemmv1.Pending || s.ArbitrationReady {
			break
		}
		if _, err := gemmv1.SubmitMidFromTrace(dispute, gemmv1.Worker, workerArt, uint64(openedEpoch+2+2*uint64(rounds))); err != nil {
			return nil, err
		}
		if _, err := gemmv1.SubmitMidFromTrace(dispute, gemmv1.Challenger, challArt, uint64(openedEpoch+3+2*uint64(rounds))); err != nil {
			return nil, err
		}
		rounds++
		if rounds > 64 {
			return nil, fmt.Errorf("bisection did not converge")
		}
	}

	step := dispute.Status().Low
	aTile := gemmv1.ExtractATile(matrixA, task.M, task.K, uint64(tileI), uint64(step))
	bTile := gemmv1.ExtractBTile(matrixB, task.K, task.N, uint64(step), uint64(tileJ))
	aProof, err := gemmv1.ProveInputTile(gemmv1.MatrixIDA, matrixA, matrixB, task.M, task.N, task.K, uint64(tileI), uint64(step))
	if err != nil {
		return nil, err
	}
	bProof, err := gemmv1.ProveInputTile(gemmv1.MatrixIDB, matrixA, matrixB, task.M, task.N, task.K, uint64(step), uint64(tileJ))
	if err != nil {
		return nil, err
	}
	outcome, err := dispute.Arbitrate(aTile, bTile, aProof, bProof, dispute.Status().Deadline)
	if err != nil {
		return nil, err
	}

	report := &disputeReport{
		TaskID:              fmt.Sprintf("%x", taskID),
		DisputedTile:        [2]int{int(tileI), int(tileJ)},
		BisectionRounds:     rounds,
		ArbitrationStep:     step,
		Outcome:             outcomeName(outcome),
		OnDemandTraceBytes:  (int(rSteps) + 1) * 256,
		WorkerTraceRoot:     fmt.Sprintf("%x", workerArt.Root),
		ChallengerTraceRoot: fmt.Sprintf("%x", challArt.Root),
		TranscriptDigest:    fmt.Sprintf("%x", dispute.TranscriptDigest()),
	}
	if outcome == gemmv1.WorkerWins {
		receipt, err := gemmv1.BuildVerifiedWorkReceipt(task, assignment, &rc, gemmv1.ModeChallengedWorkerWon, task.IssuedEpoch+100, nil, dispute.TranscriptDigest())
		if err != nil {
			return nil, err
		}
		report.WorkerGotVWR = true
		id, err := receipt.ReceiptID()
		if err != nil {
			return nil, err
		}
		report.ReceiptID = fmt.Sprintf("%x", id)
	}
	if verbose {
		expected := gemmv1.MicroStep(&challArt.States[step], aTile, bTile)
		report.Detail = &arbitrationDetail{
			Step:           step,
			ATile:          tileToInt32s(gemmv1.Int8TileBytes(aTile), false),
			BTile:          tileToInt32s(gemmv1.Int8TileBytes(bTile), false),
			LowState:       tileToInt32s(challArt.States[step].CanonicalBytes(), true),
			WorkerHigh:     tileToInt32s(workerArt.States[rSteps].CanonicalBytes(), true),
			ChallengerHigh: tileToInt32s(challArt.States[rSteps].CanonicalBytes(), true),
			Expected:       tileToInt32s(expected.CanonicalBytes(), true),
			ArbitrationMAC: 8 * 8 * 8,
		}
	}
	return report, nil
}

// devFabricateTrace rebuilds the honest trace for a tile and then shifts
// element (0,0) of the final state so it ends exactly at the worker's
// committed tile. DEVELOPMENT SIMULATION ONLY: on a real network the lying
// worker would produce this trace itself and lock it.
func devFabricateTrace(task *gemmv1.TaskDescriptor, matrixA, matrixB []int8, taskID, assignmentID []byte, tileI, tileJ uint32, committedTile gemmv1.State) *gemmv1.TileTraceArtifacts {
	honest, err := gemmv1.BuildTileTrace(matrixA, matrixB, task.M, task.N, task.K, taskID, assignmentID, tileI, tileJ)
	if err != nil {
		panic(err)
	}
	rSteps := uint32(gemmv1.RSteps(task.K))
	delta := committedTile[0] - honest.States[rSteps][0]
	states := append([]gemmv1.State(nil), honest.States...)
	for r := int(rSteps); r < len(states); r++ {
		states[r][0] += delta
	}
	leaves := make([]gemmv1.Hash, len(states))
	for step, s := range states {
		leaves[step] = gemmv1.LeafTraceState(taskID, assignmentID, tileI, tileJ, uint32(step), s.CanonicalBytes())
	}
	levels, err := gemmv1.BuildLevels(leaves)
	if err != nil {
		panic(err)
	}
	return &gemmv1.TileTraceArtifacts{States: states, Levels: levels, Root: levels[len(levels)-1][0]}
}

func outcomeName(o gemmv1.Outcome) string {
	switch o {
	case gemmv1.WorkerWins:
		return "worker_wins"
	case gemmv1.ChallengerWins:
		return "challenger_wins"
	case gemmv1.BothInvalid:
		return "both_invalid"
	default:
		return "pending"
	}
}

func mustTree(taskID, assignmentID []byte, tiles []gemmv1.State, colsC uint32) ([][]gemmv1.Hash, gemmv1.Hash) {
	leaves := gemmv1.OutputLeaves(taskID, assignmentID, tiles, colsC)
	root, err := gemmv1.MerkleRoot(leaves)
	if err != nil {
		panic(err)
	}
	levels, err := gemmv1.BuildLevels(leaves)
	if err != nil {
		panic(err)
	}
	return levels, root
}

func arrayOf(b []byte) [32]byte {
	var out [32]byte
	copy(out[:], b)
	return out
}

func tileToInt32s(raw []byte, isInt32 bool) []int32 {
	if isInt32 {
		vals, err := gemmv1.CanonicalToInt32s(raw)
		if err != nil {
			panic(err)
		}
		return vals
	}
	out := make([]int32, len(raw))
	for i, b := range raw {
		out[i] = int32(int8(b))
	}
	return out
}

func cmdVerify(args []string) error {
	fs := flag.NewFlagSet("verify", flag.ExitOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("verify needs task.json and result.json")
	}
	task, spec, err := loadTask(fs.Arg(0))
	if err != nil {
		return err
	}
	rf, err := loadResult(fs.Arg(1))
	if err != nil {
		return err
	}
	report, err := runVerification(task, spec, rf, false)
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(data))
	if report.Outcome == "optimistic_unchallenged" {
		fmt.Println("PASS: challenger output identical; result may finalize as optimistic_unchallenged")
		return nil
	}
	return fmt.Errorf("output mismatch at tile (%d,%d); run `prisma-gemm challenge` to start the dispute",
		report.DisputedTile[0], report.DisputedTile[1])
}

func cmdChallenge(args []string, verbose bool) error {
	fs := flag.NewFlagSet("challenge", flag.ExitOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("challenge needs task.json and result.json")
	}
	task, spec, err := loadTask(fs.Arg(0))
	if err != nil {
		return err
	}
	rf, err := loadResult(fs.Arg(1))
	if err != nil {
		return err
	}
	report, err := runVerification(task, spec, rf, verbose)
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(data))
	switch report.Outcome {
	case "optimistic_unchallenged":
		fmt.Println("PASS: no dispute needed; outputs identical")
	case "challenger_wins":
		fmt.Println("PASS: fraud confirmed; ChallengerWins; the worker receives NO Verified Work Receipt")
	case "worker_wins":
		fmt.Println("PASS: challenge rejected; WorkerWins; Verified Work Receipt issued as challenged_worker_won")
	default:
		return fmt.Errorf("dispute ended with %s", report.Outcome)
	}
	return nil
}

func cmdTraceTile(args []string) error {
	fs := flag.NewFlagSet("trace-tile", flag.ExitOnError)
	var i, j int
	fs.IntVar(&i, "i", 0, "tile row i")
	fs.IntVar(&j, "j", 0, "tile col j")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("trace-tile needs task.json and result.json")
	}
	task, spec, err := loadTask(fs.Arg(0))
	if err != nil {
		return err
	}
	matrixA, matrixB, err := matricesFor(task, spec)
	if err != nil {
		return err
	}
	taskID := mustTaskID(task)
	assignment := &gemmv1.Assignment{
		TaskID:          taskID,
		WorkerPubKey:    devKey(devWorkerSeedDomain, spec.Seed).Public().(ed25519.PublicKey),
		AssignmentNonce: gemmv1.Int8ToBytes(gemmv1.GenTestMatrix('M', spec.Seed, 16)),
		AcceptedEpoch:   task.IssuedEpoch + 1,
	}
	assignmentID := mustAssignmentID(assignment)
	started := time.Now()
	art, err := gemmv1.BuildTileTrace(matrixA, matrixB, task.M, task.N, task.K, taskID, assignmentID, uint32(i), uint32(j))
	if err != nil {
		return err
	}
	rSteps := gemmv1.RSteps(task.K)
	fmt.Printf("tile: (%d,%d)\nsteps: R+1 = %d\ntrace_root: %x\non_demand_trace_bytes: %d\ngeneration_time: %s\n",
		i, j, rSteps+1, art.Root, (int(rSteps)+1)*256, time.Since(started).Round(time.Millisecond))
	return nil
}
