package verify

// Adversarial tests for the v0.1.2 fast verification layer. The full-recompute
// oracle (gemmv1.ReferenceGEMM) is used ONLY to construct fixtures and
// honest baselines here; the fast path itself never calls it.

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"testing"

	"prismachain/compute/gemmv1"
)

type vfFixture struct {
	a, b         []int8
	cHonest      []int32
	task         *gemmv1.TaskDescriptor
	taskID       []byte
	assignment   *gemmv1.Assignment
	assignmentID []byte
	outputRoot   gemmv1.Hash
	colsC        uint32
	tiles        []gemmv1.State
}

func newVF(t *testing.T, m, n, k uint64, seed uint32) *vfFixture {
	t.Helper()
	a := gemmv1.GenTestMatrix('A', seed, m*k)
	b := gemmv1.GenTestMatrix('B', seed, k*n)
	rootA, rootB, err := gemmv1.BuildMatrixRoots(a, b, m, n, k)
	if err != nil {
		t.Fatal(err)
	}
	reqKey := sha256.Sum256([]byte("req"))
	task, err := gemmv1.NewTaskDescriptor(reqKey[:], []byte("nonce"), 1000, m, n, k, rootA[:], rootB[:], 100, 1000)
	if err != nil {
		t.Fatal(err)
	}
	taskID, _ := task.TaskID()
	wKey := sha256.Sum256([]byte("w"))
	assignment := &gemmv1.Assignment{TaskID: taskID, WorkerPubKey: wKey[:], AssignmentNonce: []byte("an"), AcceptedEpoch: 1001}
	assignmentID, _ := assignment.AssignmentID()
	c := gemmv1.ReferenceGEMM(a, b, m, n, k)
	tiles := gemmv1.OutputTiles(c, m, n)
	counts := gemmv1.TileCountsFor(m, n, k)
	leaves := gemmv1.OutputLeaves(taskID, assignmentID, tiles, uint32(counts.ColsC))
	root, err := gemmv1.MerkleRoot(leaves)
	if err != nil {
		t.Fatal(err)
	}
	return &vfFixture{a: a, b: b, cHonest: c, task: task, taskID: taskID, assignment: assignment, assignmentID: assignmentID, outputRoot: root, colsC: uint32(counts.ColsC), tiles: tiles}
}

func prof(rounds uint16) VerificationProfile {
	return VerificationProfile{Algorithm: AlgorithmFreivaldsBinaryV1, Rounds: rounds}
}

// 5: profile admission must reject shapes beyond the proven int64 bound.
func TestProfileAdmissionRejectsUnsafeShapes(t *testing.T) {
	p := prof(8)
	if err := p.AdmitFor(4, 4, 4); err != nil {
		t.Fatalf("small shape must pass: %v", err)
	}
	if err := p.AdmitFor(0, 4, 4); !errors.Is(err, errShape) {
		t.Fatalf("empty dimension must be rejected, got %v", err)
	}
	// Force the overflow bound: N*K beyond 2^48 with the y bound wrapping.
	huge := VerificationProfile{Algorithm: AlgorithmFreivaldsBinaryV1, Rounds: 1}
	if err := huge.AdmitFor(1, 1<<40, 1<<40); !errors.Is(err, errOverflowRisk) {
		t.Fatalf("unbounded shape must be rejected, got %v", err)
	}
	if err := p.Validate(); err != nil {
		t.Fatalf("valid profile rejected: %v", err)
	}
	bad := VerificationProfile{Algorithm: "FP32_FAST_MATH", Rounds: 8}
	if err := bad.Validate(); err == nil {
		t.Fatal("unknown algorithm must be rejected")
	}
}

// G: production challenge randomness cannot exist before a commit.
func TestRandomnessRequiresCommit(t *testing.T) {
	if _, err := NewChallengeRandomness(nil); err == nil {
		t.Fatal("empty commit must not yield challenge randomness")
	}
	if _, err := NewChallengeRandomness(make([]byte, 32)); err != nil {
		t.Fatalf("committed randomness refused: %v", err)
	}
}

// A: an honest C passes every round with CSPRNG randomness, scalar and batch.
func TestFreivaldsHonestPasses(t *testing.T) {
	f := newVF(t, 32, 48, 64, 3)
	src, err := NewChallengeRandomness(f.assignmentID)
	if err != nil {
		t.Fatal(err)
	}
	for _, impl := range []struct {
		name string
		fn   func() (*VerificationResult, error)
	}{
		{"scalar", func() (*VerificationResult, error) {
			return VerifyFreivalds(f.a, f.b, f.cHonest, 32, 48, 64, prof(16), src)
		}},
		{"batch", func() (*VerificationResult, error) {
			return VerifyFreivaldsBatch(f.a, f.b, f.cHonest, 32, 48, 64, prof(16), src)
		}},
	} {
		res, err := impl.fn()
		if err != nil {
			t.Fatal(err)
		}
		if !res.Passed || res.RoundsExecuted != 16 {
			t.Fatalf("%s: honest C did not pass: %+v", impl.name, res)
		}
		if res.VerificationMACs != 16*(48*64+32*64+32*48) {
			t.Fatalf("%s: unexpected MAC count %d", impl.name, res.VerificationMACs)
		}
	}
}

// detectSeeds finds a development-only seed whose fixed R stream detects
// the injected fraud within the given rounds. Deterministic and stable.
func detectSeeds(t *testing.T, f *vfFixture, c []int32, m, n, k uint64, rounds uint16, maxSeed int) *VerificationResult {
	t.Helper()
	for s := 1; s <= maxSeed; s++ {
		src := NewDeterministicSource([]byte{byte(s)})
		res, err := VerifyFreivalds(f.a, f.b, c, m, n, k, prof(rounds), src)
		if err != nil {
			t.Fatal(err)
		}
		if !res.Passed {
			return res
		}
	}
	t.Fatalf("no seed detected the fraud within %d rounds", rounds)
	return nil
}

// B: a single corrupted element is detected, localized to row/col/tile and
// the existing v0.1.1 dispute ends with ChallengerWins.
func TestSingleElementFraudDetectedAndProven(t *testing.T) {
	f := newVF(t, 16, 16, 24, 5)
	c := append([]int32(nil), f.cHonest...)
	c[5*16+9]++ // one element in tile (0,1)
	res := detectSeeds(t, f, c, 16, 16, 24, 4, 8)
	if len(res.ResidualRows) == 0 {
		t.Fatal("no residual rows")
	}
	loc, found, err := LocalizeFromRows(f.a, f.b, c, 16, 16, 24, res.ResidualRows)
	if err != nil || !found {
		t.Fatalf("localization failed: found=%v err=%v", found, err)
	}
	if loc.Row != 5 || loc.TileI != 0 || loc.TileJ != 1 {
		t.Fatalf("localized to row %d tile (%d,%d), want row 5 tile (0,1)", loc.Row, loc.TileI, loc.TileJ)
	}
	proveFraudViaDispute(t, f, c, loc)
}

// C: a whole 8x8 tile corrupted takes the full fast path to ChallengerWins.
func TestWholeTileFraudChain(t *testing.T) {
	f := newVF(t, 24, 24, 32, 6)
	c := append([]int32(nil), f.cHonest...)
	for r := 0; r < 8; r++ {
		for col := 0; col < 8; col++ {
			c[(16+r)*24+8+col] += 7
		}
	}
	res := detectSeeds(t, f, c, 24, 24, 32, 4, 8)
	loc, found, err := LocalizeFromRows(f.a, f.b, c, 24, 24, 32, res.ResidualRows)
	if err != nil || !found {
		t.Fatalf("localization failed: %v %v", found, err)
	}
	if loc.TileI != 2 || loc.TileJ != 1 {
		t.Fatalf("localized tile (%d,%d), want (2,1)", loc.TileI, loc.TileJ)
	}
	proveFraudViaDispute(t, f, c, loc)
}

// I: errors across multiple rows and columns still yield one fraud path.
func TestMultipleErrorsOneFraudPath(t *testing.T) {
	f := newVF(t, 32, 32, 40, 7)
	c := append([]int32(nil), f.cHonest...)
	c[3*32+3] += 11
	c[17*32+22] -= 13
	c[29*32+30] += 17
	res := detectSeeds(t, f, c, 32, 32, 40, 4, 8)
	loc, found, err := LocalizeFromRows(f.a, f.b, c, 32, 32, 40, res.ResidualRows)
	if err != nil || !found {
		t.Fatalf("localization failed: %v %v", found, err)
	}
	proveFraudViaDispute(t, f, c, loc)
}

// D: a challenger facing an honest C cannot open a challenge: the exact
// row recomputation matches the worker output, so no differing tile exists.
func TestFalseChallengeRejected(t *testing.T) {
	f := newVF(t, 16, 16, 24, 8)
	// An honest residual is empty; a forced "bad row" claim localizes to
	// nothing.
	loc, found, err := LocalizeFromRows(f.a, f.b, f.cHonest, 16, 16, 24, []uint64{2})
	if err != nil {
		t.Fatal(err)
	}
	if found {
		t.Fatal("false claim produced a differing tile on honest C")
	}
	_ = loc
	// And an honest C passes Freivalds, so no mismatch -> no challenge at all.
	src := NewDeterministicSource([]byte{9})
	res, err := VerifyFreivalds(f.a, f.b, f.cHonest, 16, 16, 24, prof(8), src)
	if err != nil || !res.Passed {
		t.Fatalf("honest C flagged: %+v err=%v", res, err)
	}
}

// proveFraudViaDispute drives the v0.1.1 deterministic dispute from a
// localized tile: ChallengeOpen -> traces -> bisection -> 512-MAC
// arbitration -> ChallengerWins.
func proveFraudViaDispute(t *testing.T, f *vfFixture, c []int32, loc *Localization) {
	t.Helper()
	taskID, assignmentID := f.taskID, f.assignmentID
	workerTiles := gemmv1.OutputTiles(c, f.task.M, f.task.N)
	challengerTiles := f.tiles // exact recomputation
	chalKey := sha256.Sum256([]byte("chal"))
	levels, workerRoot := mustTree(t, taskID, assignmentID, workerTiles, f.colsC)
	idx := int(loc.TileI)*int(f.colsC) + int(loc.TileJ)
	workerProof, err := gemmv1.ProveLeaf(levels, uint32(idx))
	if err != nil {
		t.Fatal(err)
	}
	co := &gemmv1.ChallengeOpen{
		ProtocolVersion:      gemmv1.ProtocolVersion,
		TaskID:               taskID,
		AssignmentID:         assignmentID,
		ChallengerPubKey:     chalKey[:],
		WorkerPubKey:         f.assignment.WorkerPubKey,
		DisputedTileI:        uint64(loc.TileI),
		DisputedTileJ:        uint64(loc.TileJ),
		WorkerOutputTile:     workerTiles[idx].CanonicalBytes(),
		WorkerOutputProof:    workerProof,
		ChallengerOutputTile: challengerTiles[idx].CanonicalBytes(),
		ChallengeBond:        1000,
		OpenedEpoch:          1010,
	}
	if err := gemmv1.ValidateChallengeOpen(f.task, co); err != nil {
		t.Fatalf("challenge rejected: %v", err)
	}
	leaf := gemmv1.LeafOutputTile(taskID, assignmentID, uint32(loc.TileI), uint32(loc.TileJ), co.WorkerOutputTile)
	if !gemmv1.VerifyLeafInclusion(workerRoot, leaf, workerProof) {
		t.Fatal("worker tile not in committed output root")
	}

	challArt, err := gemmv1.BuildTileTrace(f.a, f.b, f.task.M, f.task.N, f.task.K, taskID, assignmentID, uint32(loc.TileI), uint32(loc.TileJ))
	if err != nil {
		t.Fatal(err)
	}
	workerArt := challArt
	if workerTiles[idx] != challengerTiles[idx] {
		workerArt = fabricateTraceFor(t, f, challArt, uint32(loc.TileI), uint32(loc.TileJ), workerTiles[idx])
	}
	d := openDispute(t, f, co, workerArt, challArt, workerRoot)
	for {
		s := d.Status()
		if s.Outcome != gemmv1.Pending || s.ArbitrationReady {
			break
		}
		if _, err := gemmv1.SubmitMidFromTrace(d, gemmv1.Worker, workerArt, uint64(1020+2*s.Mid)); err != nil {
			t.Fatal(err)
		}
		if _, err := gemmv1.SubmitMidFromTrace(d, gemmv1.Challenger, challArt, uint64(1021+2*s.Mid)); err != nil {
			t.Fatal(err)
		}
	}
	step := d.Status().Low
	aTile := gemmv1.ExtractATile(f.a, f.task.M, f.task.K, uint64(loc.TileI), uint64(step))
	bTile := gemmv1.ExtractBTile(f.b, f.task.K, f.task.N, uint64(step), uint64(loc.TileJ))
	aProof, err := gemmv1.ProveInputTile(gemmv1.MatrixIDA, f.a, f.b, f.task.M, f.task.N, f.task.K, uint64(loc.TileI), uint64(step))
	if err != nil {
		t.Fatal(err)
	}
	bProof, err := gemmv1.ProveInputTile(gemmv1.MatrixIDB, f.a, f.b, f.task.M, f.task.N, f.task.K, uint64(step), uint64(loc.TileJ))
	if err != nil {
		t.Fatal(err)
	}
	outcome, err := d.Arbitrate(aTile, bTile, aProof, bProof, d.Status().Deadline)
	if err != nil {
		t.Fatal(err)
	}
	if outcome != gemmv1.ChallengerWins {
		t.Fatalf("dispute outcome = %d, want ChallengerWins", outcome)
	}
}

func mustTree(t *testing.T, taskID, assignmentID []byte, tiles []gemmv1.State, colsC uint32) ([][]gemmv1.Hash, gemmv1.Hash) {
	t.Helper()
	leaves := gemmv1.OutputLeaves(taskID, assignmentID, tiles, colsC)
	levels, err := gemmv1.BuildLevels(leaves)
	if err != nil {
		t.Fatal(err)
	}
	return levels, levels[len(levels)-1][0]
}

func fabricateTraceFor(t *testing.T, f *vfFixture, honest *gemmv1.TileTraceArtifacts, tileI, tileJ uint32, committed gemmv1.State) *gemmv1.TileTraceArtifacts {
	t.Helper()
	rSteps := uint32(gemmv1.RSteps(f.task.K))
	// Fabrication: keep the honest prefix and end exactly at the committed
	// (fraudulent) tile. Any lying worker can construct such a trace.
	states := append([]gemmv1.State(nil), honest.States...)
	states[rSteps] = committed
	leaves := make([]gemmv1.Hash, len(states))
	for step, s := range states {
		leaves[step] = gemmv1.LeafTraceState(f.taskID, f.assignmentID, tileI, tileJ, uint32(step), s.CanonicalBytes())
	}
	levels, err := gemmv1.BuildLevels(leaves)
	if err != nil {
		t.Fatal(err)
	}
	return &gemmv1.TileTraceArtifacts{States: states, Levels: levels, Root: levels[len(levels)-1][0]}
}

func openDispute(t *testing.T, f *vfFixture, co *gemmv1.ChallengeOpen, workerArt, challArt *gemmv1.TileTraceArtifacts, workerRoot gemmv1.Hash) *gemmv1.GEMMDispute {
	t.Helper()
	cfg := gemmv1.DisputeConfig{
		Task:           f.task,
		Assignment:     f.assignment,
		TaskID:         f.taskID,
		TileI:          uint32(co.DisputedTileI),
		TileJ:          uint32(co.DisputedTileJ),
		WorkerTile:     co.WorkerOutputTile,
		ChallengerTile: co.ChallengerOutputTile,
		MatrixARoot:    mustRoot(t, f, gemmv1.MatrixIDA),
		MatrixBRoot:    mustRoot(t, f, gemmv1.MatrixIDB),
		RoundPeriod:    50,
	}
	assignmentID, err := cfg.Assignment.AssignmentID()
	if err != nil {
		t.Fatal(err)
	}
	wc, err := gemmv1.BuildTraceCommit(f.task, assignmentID, gemmv1.Worker, uint32(co.DisputedTileI), uint32(co.DisputedTileJ), workerArt, 1011)
	if err != nil {
		t.Fatal(err)
	}
	cc, err := gemmv1.BuildTraceCommit(f.task, assignmentID, gemmv1.Challenger, uint32(co.DisputedTileI), uint32(co.DisputedTileJ), challArt, 1011)
	if err != nil {
		t.Fatal(err)
	}
	d, err := gemmv1.NewGEMMDispute(cfg,
		gemmv1.TraceClaim{Party: gemmv1.Worker, TraceRoot: workerArt.Root, InitialState: wc.InitialState, InitialProof: wc.InitialProof, FinalState: wc.FinalState, FinalProof: wc.FinalProof},
		gemmv1.TraceClaim{Party: gemmv1.Challenger, TraceRoot: challArt.Root, InitialState: cc.InitialState, InitialProof: cc.InitialProof, FinalState: cc.FinalState, FinalProof: cc.FinalProof},
		1010)
	if err != nil {
		t.Fatal(err)
	}
	_ = workerRoot
	return d
}

func mustRoot(t *testing.T, f *vfFixture, id byte) gemmv1.Hash {
	t.Helper()
	ra, rb, err := gemmv1.BuildMatrixRoots(f.a, f.b, f.task.M, f.task.N, f.task.K)
	if err != nil {
		t.Fatal(err)
	}
	if id == gemmv1.MatrixIDA {
		return ra
	}
	return rb
}

// E: output data unavailable blocks optimistic finalization.
func TestDataUnavailable(t *testing.T) {
	f := newVF(t, 8, 8, 16, 11)
	desc, err := NewOutputAvailabilityDescriptor(f.task, f.assignmentID, f.outputRoot[:], "dev://missing")
	if err != nil {
		t.Fatal(err)
	}
	failing := FetcherFunc(func(*OutputAvailabilityDescriptor) ([]int32, error) {
		return nil, fmt.Errorf("storage 404")
	})
	ok, err := CanFinalizeOptimistic(f.task, f.assignmentID, desc, failing)
	if ok || !errors.Is(err, ErrDataUnavailable) {
		t.Fatalf("data-unavailable must block finalize: ok=%v err=%v", ok, err)
	}
}

// F: served C that does not hash to output_root is rejected outright.
func TestCommitmentMismatchRejected(t *testing.T) {
	f := newVF(t, 8, 8, 16, 12)
	desc, err := NewOutputAvailabilityDescriptor(f.task, f.assignmentID, f.outputRoot[:], "dev://c")
	if err != nil {
		t.Fatal(err)
	}
	corrupted := append([]int32(nil), f.cHonest...)
	corrupted[0]++
	lying := FetcherFunc(func(*OutputAvailabilityDescriptor) ([]int32, error) { return corrupted, nil })
	ok, err := CanFinalizeOptimistic(f.task, f.assignmentID, desc, lying)
	if ok || !errors.Is(err, ErrCommitmentMismatch) {
		t.Fatalf("commitment mismatch must be rejected: ok=%v err=%v", ok, err)
	}
	honest := FetcherFunc(func(*OutputAvailabilityDescriptor) ([]int32, error) { return f.cHonest, nil })
	ok, err = CanFinalizeOptimistic(f.task, f.assignmentID, desc, honest)
	if !ok || err != nil {
		t.Fatalf("honest C must finalize: ok=%v err=%v", ok, err)
	}
}

// 24: empirical false-accept simulation with fixed deterministic trial
// seeds (stable, reproducible). The observed rates never replace the
// 2^-rounds theoretical bound.
func TestFalseAcceptSimulation(t *testing.T) {
	f := newVF(t, 16, 16, 24, 13)
	c := append([]int32(nil), f.cHonest...)
	c[7*16+7]++
	cases := []struct {
		rounds uint16
		trials int
	}{
		{1, 4000}, {2, 4000}, {4, 4000}, {8, 20000}, {16, 50000},
	}
	for _, tc := range cases {
		accepts := 0
		for trial := 0; trial < tc.trials; trial++ {
			seed := sha256.Sum256([]byte(fmt.Sprintf("false-accept-%d-%d", tc.rounds, trial)))
			src := NewDeterministicSource(seed[:])
			res, err := VerifyFreivalds(f.a, f.b, c, 16, 16, 24, prof(tc.rounds), src)
			if err != nil {
				t.Fatal(err)
			}
			if res.Passed {
				accepts++
			}
		}
		rate := float64(accepts) / float64(tc.trials)
		bound := (&VerificationProfile{Algorithm: AlgorithmFreivaldsBinaryV1, Rounds: tc.rounds}).FalseAcceptUpperBound()
		t.Logf("rounds=%2d trials=%d accepts=%d empirical=%.5f theoretical_bound=%.3g",
			tc.rounds, tc.trials, accepts, rate, bound)
		if rate > bound*4+0.01 {
			t.Fatalf("rounds=%d empirical false accept %.5f far above the theoretical bound", tc.rounds, rate)
		}
	}
}

// Batch and scalar must agree exactly on the same deterministic stream.
func TestBatchMatchesScalar(t *testing.T) {
	f := newVF(t, 16, 16, 24, 14)
	c := append([]int32(nil), f.cHonest...)
	c[9*16+2]++
	srcS := NewDeterministicSource([]byte{42})
	srcB := NewDeterministicSource([]byte{42})
	rs, err := VerifyFreivalds(f.a, f.b, c, 16, 16, 24, prof(8), srcS)
	if err != nil {
		t.Fatal(err)
	}
	rb, err := VerifyFreivaldsBatch(f.a, f.b, c, 16, 16, 24, prof(8), srcB)
	if err != nil {
		t.Fatal(err)
	}
	if rs.Passed != rb.Passed || !equalU64(rs.ResidualRows, rb.ResidualRows) {
		t.Fatalf("batch disagrees with scalar: %+v vs %+v", rs, rb)
	}
}

func equalU64(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

var _ = bytes.Equal
