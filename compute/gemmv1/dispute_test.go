package gemmv1

// Dispute scenarios: optimistic path, output fraud (C), false challenge
// (D), invalid proofs (E), timeouts (G) and bisection round counts.

import (
	"bytes"
	"testing"
)

// buildTreeFor builds levels and root over given tiles.
func buildTreeFor(t *testing.T, taskID, assignmentID []byte, tiles []State, colsC uint32) ([][]Hash, Hash) {
	t.Helper()
	leaves := OutputLeaves(taskID, assignmentID, tiles, colsC)
	levels, err := buildLevels(leaves)
	if err != nil {
		t.Fatal(err)
	}
	return levels, levels[len(levels)-1][0]
}

// fabricateTrace builds a trace that follows the honest states and then
// injects delta into element 0 from transition divergeStep+1 onward, ending
// exactly at the party's claimed tile. This models a participant that
// fabricated a trace to match its committed output.
func fabricateTrace(t *testing.T, f *fixture, honest *TileTraceArtifacts, tileI, tileJ, divergeStep uint32, delta int32) *TileTraceArtifacts {
	t.Helper()
	states := append([]State(nil), honest.States...)
	for r := int(divergeStep) + 1; r < len(states); r++ {
		states[r][0] += delta
	}
	leaves := make([]Hash, len(states))
	for step, s := range states {
		leaves[step] = LeafTraceState(f.taskID, f.assignmentID, tileI, tileJ, uint32(step), s.CanonicalBytes())
	}
	levels, err := buildLevels(leaves)
	if err != nil {
		t.Fatal(err)
	}
	return &TileTraceArtifacts{States: states, Levels: levels, Root: levels[len(levels)-1][0]}
}

// openChallenge runs the challenger and coordinator validation for one
// disputed tile and returns the dispute-ready context.
func openChallenge(t *testing.T, f *fixture, workerTiles, challengerTiles []State, openedEpoch uint64) (*GEMMDispute, *TileTraceArtifacts, *TileTraceArtifacts) {
	t.Helper()
	taskID, assignmentID := f.taskID, f.assignmentID

	// Challenger detects the differing tile.
	tileI, tileJ, found, err := FindDisputedTile(workerTiles, challengerTiles, f.colsC)
	if err != nil || !found {
		t.Fatalf("challenger must detect fraud: found=%v err=%v", found, err)
	}

	// Worker output tree over the (fraudulent) tiles; membership proof.
	workerLevels, workerRoot := buildTreeFor(t, taskID, assignmentID, workerTiles, f.colsC)
	idx := int(tileI)*int(f.colsC) + int(tileJ)
	workerProof, err := ProveLeaf(workerLevels, uint32(idx))
	if err != nil {
		t.Fatal(err)
	}

	co := &ChallengeOpen{
		ProtocolVersion:      ProtocolVersion,
		TaskID:               append([]byte(nil), taskID...),
		AssignmentID:         append([]byte(nil), assignmentID...),
		ChallengerPubKey:     append([]byte(nil), f.challPub...),
		WorkerPubKey:         append([]byte(nil), f.workerPub...),
		DisputedTileI:        uint64(tileI),
		DisputedTileJ:        uint64(tileJ),
		WorkerOutputTile:     workerTiles[idx].CanonicalBytes(),
		WorkerOutputProof:    workerProof,
		ChallengerOutputTile: challengerTiles[idx].CanonicalBytes(),
		ChallengeBond:        1000,
		OpenedEpoch:          openedEpoch,
	}
	if err := ValidateChallengeOpen(f.task, co); err != nil {
		t.Fatalf("challenge rejected: %v", err)
	}
	// Coordinator verifies the worker tile belongs to the committed root.
	leaf := LeafOutputTile(taskID, assignmentID, tileI, tileJ, co.WorkerOutputTile)
	if !VerifyLeafInclusion(workerRoot, leaf, workerProof) {
		t.Fatal("worker tile not in committed output root")
	}

	// Honest trace for the disputed tile; each party whose claimed tile
	// differs from the honest one fabricates a matching trace.
	honestArt, err := BuildTileTrace(f.matrixA, f.matrixB, f.task.M, f.task.N, f.task.K, taskID, assignmentID, tileI, tileJ)
	if err != nil {
		t.Fatal(err)
	}
	rSteps := uint32(RSteps(f.task.K))
	honestFinal := honestArt.States[rSteps].CanonicalBytes()

	workerArt := honestArt
	if !bytes.Equal(workerTiles[idx].CanonicalBytes(), honestFinal) {
		delta := int32(workerTiles[idx][0]) - int32(honestArt.States[rSteps][0])
		workerArt = fabricateTrace(t, f, honestArt, tileI, tileJ, rSteps-1, delta)
	}
	challArt := honestArt
	if !bytes.Equal(challengerTiles[idx].CanonicalBytes(), honestFinal) {
		delta := int32(challengerTiles[idx][0]) - int32(honestArt.States[rSteps][0])
		challArt = fabricateTrace(t, f, honestArt, tileI, tileJ, rSteps-1, delta)
	}
	if !bytes.Equal(workerArt.States[rSteps].CanonicalBytes(), co.WorkerOutputTile) {
		t.Fatal("worker trace does not end at the committed tile")
	}
	if !bytes.Equal(challArt.States[rSteps].CanonicalBytes(), co.ChallengerOutputTile) {
		t.Fatal("challenger trace does not end at the claimed tile")
	}

	workerCommit, err := BuildTraceCommit(f.task, assignmentID, Worker, tileI, tileJ, workerArt, openedEpoch+1)
	if err != nil {
		t.Fatal(err)
	}
	challCommit, err := BuildTraceCommit(f.task, assignmentID, Challenger, tileI, tileJ, challArt, openedEpoch+1)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DisputeConfig{
		Task:           f.task,
		Assignment:     f.assignment,
		TaskID:         taskID,
		TileI:          tileI,
		TileJ:          tileJ,
		WorkerTile:     co.WorkerOutputTile,
		ChallengerTile: co.ChallengerOutputTile,
		MatrixARoot:    f.rootA,
		MatrixBRoot:    f.rootB,
		RoundPeriod:    50,
	}
	workerClaim := TraceClaim{
		Party: Worker, TraceRoot: workerArt.Root,
		InitialState: workerCommit.InitialState, InitialProof: workerCommit.InitialProof,
		FinalState: workerCommit.FinalState, FinalProof: workerCommit.FinalProof,
	}
	challClaim := TraceClaim{
		Party: Challenger, TraceRoot: challArt.Root,
		InitialState: challCommit.InitialState, InitialProof: challCommit.InitialProof,
		FinalState: challCommit.FinalState, FinalProof: challCommit.FinalProof,
	}
	d, err := NewGEMMDispute(cfg, workerClaim, challClaim, openedEpoch)
	if err != nil {
		t.Fatalf("dispute open: %v", err)
	}
	return d, workerArt, challArt
}

func runBisection(t *testing.T, d *GEMMDispute, workerArt, challArt *TileTraceArtifacts) int {
	t.Helper()
	rounds := 0
	for {
		s := d.Status()
		if s.Outcome != Pending || s.ArbitrationReady {
			break
		}
		if _, err := SubmitMidFromTrace(d, Worker, workerArt, uint64(1020+2*rounds)); err != nil {
			t.Fatalf("worker mid: %v", err)
		}
		if _, err := SubmitMidFromTrace(d, Challenger, challArt, uint64(1021+2*rounds)); err != nil {
			t.Fatalf("challenger mid: %v", err)
		}
		rounds++
		if rounds > 64 {
			t.Fatal("bisection did not converge")
		}
	}
	return rounds
}

// C: a modified output tile is detected, disputed, bisected and arbitrated.
func TestFraudDetectedAndChallengerWins(t *testing.T) {
	f := newFixture(t, 8, 8, 16, 42, true) // R = 2
	honest := f.challengerHonestTiles()
	fraudTiles := append([]State(nil), honest...)
	fraudTiles[0][0] += 1

	d, workerArt, challArt := openChallenge(t, f, fraudTiles, honest, 1010)
	rounds := runBisection(t, d, workerArt, challArt)
	if rounds != 1 {
		t.Fatalf("expected 1 bisection round for R=2, got %d", rounds)
	}
	step := d.Status().Low
	aTile := ExtractATile(f.matrixA, f.task.M, f.task.K, 0, uint64(step))
	bTile := ExtractBTile(f.matrixB, f.task.K, f.task.N, uint64(step), 0)
	aProof, err := ProveInputTile(MatrixIDA, f.matrixA, f.matrixB, f.task.M, f.task.N, f.task.K, 0, uint64(step))
	if err != nil {
		t.Fatal(err)
	}
	bProof, err := ProveInputTile(MatrixIDB, f.matrixA, f.matrixB, f.task.M, f.task.N, f.task.K, uint64(step), 0)
	if err != nil {
		t.Fatal(err)
	}
	outcome, err := d.Arbitrate(aTile, bTile, aProof, bProof, 1030)
	if err != nil {
		t.Fatalf("arbitration: %v", err)
	}
	if outcome != ChallengerWins {
		t.Fatalf("outcome = %d, want ChallengerWins", outcome)
	}
	if len(d.TranscriptDigest()) != 32 {
		t.Fatal("transcript digest missing")
	}
}

// D: worker is correct, malicious challenger must lose.
func TestFalseChallengeWorkerWins(t *testing.T) {
	f := newFixture(t, 8, 8, 16, 42, true)
	honest := f.challengerHonestTiles()
	// Worker honest; challenger claims a wrong tile.
	wrongTiles := append([]State(nil), honest...)
	wrongTiles[0][0] -= 5

	d, workerArt, challArt := openChallenge(t, f, honest, wrongTiles, 1010)
	runBisection(t, d, workerArt, challArt)
	step := d.Status().Low
	aTile := ExtractATile(f.matrixA, f.task.M, f.task.K, 0, uint64(step))
	bTile := ExtractBTile(f.matrixB, f.task.K, f.task.N, uint64(step), 0)
	aProof, _ := ProveInputTile(MatrixIDA, f.matrixA, f.matrixB, f.task.M, f.task.N, f.task.K, 0, uint64(step))
	bProof, _ := ProveInputTile(MatrixIDB, f.matrixA, f.matrixB, f.task.M, f.task.N, f.task.K, uint64(step), 0)
	outcome, err := d.Arbitrate(aTile, bTile, aProof, bProof, 1030)
	if err != nil {
		t.Fatal(err)
	}
	if outcome != WorkerWins {
		t.Fatalf("outcome = %d, want WorkerWins", outcome)
	}
}

// E: every class of invalid proof must be rejected.
func TestInvalidProofsRejected(t *testing.T) {
	f := newFixture(t, 16, 16, 16, 42, true) // 2x2 output tiles so proofs have siblings
	honest := f.challengerHonestTiles()
	fraudTiles := append([]State(nil), honest...)
	fraudTiles[0][0] += 1

	// ChallengeOpen with a tampered worker output proof.
	tileI, tileJ := uint32(0), uint32(0)
	workerLevels, _ := buildTreeFor(t, f.taskID, f.assignmentID, fraudTiles, f.colsC)
	workerProof, _ := ProveLeaf(workerLevels, 0)
	workerProof.Siblings[0][0] ^= 0xff
	co := &ChallengeOpen{
		ProtocolVersion:      ProtocolVersion,
		TaskID:               f.taskID,
		AssignmentID:         f.assignmentID,
		ChallengerPubKey:     f.challPub,
		WorkerPubKey:         f.workerPub,
		DisputedTileI:        uint64(tileI),
		DisputedTileJ:        uint64(tileJ),
		WorkerOutputTile:     fraudTiles[0].CanonicalBytes(),
		WorkerOutputProof:    workerProof,
		ChallengerOutputTile: honest[0].CanonicalBytes(),
		ChallengeBond:        1000,
	}
	if err := ValidateChallengeOpen(f.task, co); err != nil {
		t.Fatal(err)
	}
	levels, root := buildTreeFor(t, f.taskID, f.assignmentID, fraudTiles, f.colsC)
	goodProof, _ := ProveLeaf(levels, 0)
	leaf := LeafOutputTile(f.taskID, f.assignmentID, 0, 0, co.WorkerOutputTile)
	if VerifyLeafInclusion(root, leaf, workerProof) {
		t.Fatal("tampered output proof accepted")
	}
	if !VerifyLeafInclusion(root, leaf, goodProof) {
		t.Fatal("honest output proof rejected")
	}

	// Midpoint submission with an invalid trace proof.
	d, workerArt, challArt := openChallenge(t, f, fraudTiles, honest, 1010)
	badProof, _ := ProveLeaf(challArt.Levels, d.Status().Mid)
	badProof.Siblings[0][0] ^= 0xff
	if _, err := d.SubmitMid(Challenger, challArt.States[d.Status().Mid].CanonicalBytes(), badProof, 1011); err == nil {
		t.Fatal("invalid midpoint proof accepted")
	}
	// The party can still submit the honest proof afterwards.
	if _, err := SubmitMidFromTrace(d, Challenger, challArt, 1011); err != nil {
		t.Fatalf("honest resubmission rejected: %v", err)
	}
	if _, err := SubmitMidFromTrace(d, Worker, workerArt, 1012); err != nil {
		t.Fatalf("worker mid rejected: %v", err)
	}
	// Arbitration with a wrong A tile proof must be rejected.
	step := d.Status().Low
	aTile := ExtractATile(f.matrixA, f.task.M, f.task.K, 0, uint64(step))
	bTile := ExtractBTile(f.matrixB, f.task.K, f.task.N, uint64(step), 0)
	aProof, _ := ProveInputTile(MatrixIDA, f.matrixA, f.matrixB, f.task.M, f.task.N, f.task.K, 0, uint64(step))
	bProof, _ := ProveInputTile(MatrixIDB, f.matrixA, f.matrixB, f.task.M, f.task.N, f.task.K, uint64(step), 0)
	aProof.Siblings[0][0] ^= 0xff
	if _, err := d.Arbitrate(aTile, bTile, aProof, bProof, 1030); err == nil {
		t.Fatal("arbitration accepted an invalid input tile proof")
	}
}

// G: timeouts on both sides and by both.
func TestTimeoutScenarios(t *testing.T) {
	build := func(t *testing.T) (*GEMMDispute, *TileTraceArtifacts, *TileTraceArtifacts) {
		f := newFixture(t, 8, 8, 16, 42, true)
		honest := f.challengerHonestTiles()
		fraudTiles := append([]State(nil), honest...)
		fraudTiles[0][0] += 1
		return openChallenge(t, f, fraudTiles, honest, 1010)
	}

	// Neither party answers: BothInvalid (refund path).
	d, workerArt, challArt := build(t)
	if _, err := d.Timeout(1061); err != nil {
		t.Fatalf("timeout: %v", err)
	}
	if o, _ := d.Timeout(1061); o != BothInvalid {
		t.Fatalf("outcome = %d, want BothInvalid", o)
	}
	_ = workerArt
	_ = challArt

	// Challenger silent: worker wins the round.
	d, workerArt, challArt = build(t)
	if _, err := SubmitMidFromTrace(d, Worker, workerArt, 1011); err != nil {
		t.Fatal(err)
	}
	if _, err := SubmitMidFromTrace(d, Challenger, challArt, 1012); err != nil {
		t.Fatal(err)
	}
	// Both answered the last needed round -> arbitration ready, no timeout.
	if !d.Status().ArbitrationReady {
		// With R=2 one round completes the bisection; if not ready, time out.
		if _, err := d.Timeout(1061); err != nil {
			t.Fatalf("timeout after worker-only round: %v", err)
		}
		if o, _ := d.Timeout(1061); o != WorkerWins {
			t.Fatalf("outcome = %d, want WorkerWins", o)
		}
		return
	}
	// Arbitration-ready dispute must refuse a timeout.
	if _, err := d.Timeout(1061); err == nil {
		t.Fatal("timeout accepted while arbitration is pending")
	}

	// Worker silent: challenger wins the round.
	d, workerArt, challArt = build(t)
	if _, err := SubmitMidFromTrace(d, Challenger, challArt, 1011); err != nil {
		t.Fatal(err)
	}
	if d.Status().ArbitrationReady {
		t.Fatal("single submission must not complete the bisection")
	}
	if _, err := d.Timeout(1061); err != nil {
		t.Fatal(err)
	}
	if o, _ := d.Timeout(1061); o != ChallengerWins {
		t.Fatalf("outcome = %d, want ChallengerWins", o)
	}
	_ = workerArt
}

// Bisection complexity is O(log2 R) rounds.
func TestBisectionRoundsLogarithmic(t *testing.T) {
	const k = 4096 // R = 512 -> 10 rounds over 513 states
	f := newFixture(t, 8, 8, k, 77, true)
	honest := f.challengerHonestTiles()
	fraudTiles := append([]State(nil), honest...)
	fraudTiles[0][0] += 1
	d, workerArt, challArt := openChallenge(t, f, fraudTiles, honest, 1010)
	rounds := runBisection(t, d, workerArt, challArt)
	// Interval width is R = 512, so halving takes ceil(log2(512)) = 9 rounds.
	if rounds != 9 {
		t.Fatalf("expected 9 bisection rounds for R=512, got %d", rounds)
	}
	step := d.Status().Low
	aTile := ExtractATile(f.matrixA, f.task.M, f.task.K, 0, uint64(step))
	bTile := ExtractBTile(f.matrixB, f.task.K, f.task.N, uint64(step), 0)
	aProof, _ := ProveInputTile(MatrixIDA, f.matrixA, f.matrixB, f.task.M, f.task.N, f.task.K, 0, uint64(step))
	bProof, _ := ProveInputTile(MatrixIDB, f.matrixA, f.matrixB, f.task.M, f.task.N, f.task.K, uint64(step), 0)
	outcome, err := d.Arbitrate(aTile, bTile, aProof, bProof, 1040)
	if err != nil {
		t.Fatal(err)
	}
	if outcome != ChallengerWins {
		t.Fatalf("outcome = %d, want ChallengerWins", outcome)
	}
}
