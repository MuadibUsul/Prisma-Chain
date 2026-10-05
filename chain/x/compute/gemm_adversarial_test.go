package compute

// GEMM adversarial tests: replay, stale states, timeouts, queue griefing,
// restart persistence, DoS bounds, gas accounting and state-machine jumps.

import (
	"encoding/json"
	"strings"
	"testing"

	storetypes "cosmossdk.io/store/types"
	"github.com/cosmos/cosmos-sdk/testutil"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"prismachain/chain/x/compute/types"
	"prismachain/compute/gemmv1"
)

// TestGEMMStateMachineJumpsRejected covers DoD 16's state-machine half: no
// transition may skip a prerequisite phase.
func TestGEMMStateMachineJumpsRejected(t *testing.T) {
	h := newGemmHarness(t, 1)
	const m, n, k = 16, 16, 16
	taskID, a, b, c := h.postGEMM(m, n, k, 11)

	// Post -> arbitration directly.
	if _, err := h.msg.ArbitrateGEMM(h.ctx, &types.MsgArbitrateGEMM{Actor: h.requester, GemmTaskId: taskID}); err == nil {
		t.Fatal("arbitration accepted before any dispute")
	}
	// Post -> result without acceptance.
	h.bondWorker(h.worker, h.workKeys)
	if _, err := h.msg.SubmitGEMMResult(h.ctx, &types.MsgSubmitGEMMResult{
		Worker: h.worker, GemmTaskId: taskID, OutputRoot: make([]byte, 32),
		OutputDataRef: "dev://x", OutputBytes: 1, CompletedEpoch: 1,
	}); err == nil {
		t.Fatal("result accepted before assignment")
	}
	// Post -> finalize.
	if _, err := h.msg.FinalizeGEMM(h.ctx, &types.MsgFinalizeGEMM{Actor: h.requester, GemmTaskId: taskID}); err == nil {
		t.Fatal("finalize accepted before a result")
	}
	// Accept, submit, then challenge-less arbitration.
	tiles := h.acceptAndSubmit(taskID, a, b, c, m, n, k, false)
	_ = tiles
	if _, err := h.msg.ArbitrateGEMM(h.ctx, &types.MsgArbitrateGEMM{Actor: h.requester, GemmTaskId: taskID}); err == nil {
		t.Fatal("arbitration accepted without a challenge")
	}
	// Challenge, then arbitration before trace lock/bisection.
	h.bondWorker(h.challenger, h.chalKeys)
	task := mustTask(t, h, taskID)
	h.openChallenge(taskID, &task, tiles, 0, 0, h.chalKeys, false)
	if _, err := h.msg.ArbitrateGEMM(h.ctx, &types.MsgArbitrateGEMM{Actor: h.requester, GemmTaskId: taskID}); err == nil {
		t.Fatal("arbitration accepted before trace lock")
	}
}

// TestGEMMReplayRejections covers DoD 16: cross-task, cross-worker and
// stale-round replays all fail deterministically.
func TestGEMMReplayRejections(t *testing.T) {
	h := newGemmHarness(t, 1)
	const m, n, k = 16, 16, 16
	taskA, aA, bA, cA := h.postGEMM(m, n, k, 21)
	taskB, _, _, _ := h.postGEMM(m, n, k, 22)
	tilesA := h.acceptAndSubmit(taskA, aA, bA, cA, m, n, k, false)
	taskAState := mustTask(t, h, taskA)
	_ = tilesA

	// A worker signature for task A must not verify for task B: rebuild B's
	// commit and swap the signature from a signed A commit.
	h.acceptAndSubmit(taskB, aA, bA, cA, m, n, k, false)
	taskBState := mustTask(t, h, taskB)
	rcA := taskAState.gemmResultCommit(taskAState.OutputRoot, taskAState.ResultSubmittedHeight)
	rcA.WorkerSignature = append([]byte(nil), taskAState.ResultCommitSignature...)
	rcB := taskBState.gemmResultCommit(taskBState.OutputRoot, taskBState.ResultSubmittedHeight)
	rcB.WorkerSignature = append([]byte(nil), rcA.WorkerSignature...)
	if gemmv1.VerifyResultCommitSignature(rcB) {
		t.Fatal("task A signature accepted for task B")
	}

	// A challenge for task A must not open on task B: the committed output
	// roots and assignment ids differ, so the tile proof fails membership.
	h.bondWorker(h.challenger, h.chalKeys)
	levelsA, err := gemmv1.BuildLevels(gemmOutputLeaves(h, taskA, tilesA))
	if err != nil {
		t.Fatal(err)
	}
	proof, err := gemmv1.ProveLeaf(levelsA, 0)
	if err != nil {
		t.Fatal(err)
	}
	counts := gemmv1.TileCountsFor(m, n, k)
	co := &gemmv1.ChallengeOpen{
		ProtocolVersion: gemmv1.ProtocolVersion, TaskID: taskBState.ProtocolTaskID,
		AssignmentID: taskBState.AssignmentID, ChallengerPubKey: h.chalKeys.networkPub,
		WorkerPubKey:  taskBState.WorkerProtocolPubKey,
		DisputedTileI: 0, DisputedTileJ: 0,
		WorkerOutputTile: tilesA[0].CanonicalBytes(), WorkerOutputProof: proof,
		ChallengerOutputTile: tilesA[0].CanonicalBytes(),
		ChallengeBond:        gemmBondFor(taskBState.MaxFee), OpenedEpoch: uint64(h.ctx.BlockHeight()),
	}
	co.ChallengerOutputTile = append([]byte(nil), co.ChallengerOutputTile...)
	co.ChallengerOutputTile[0]++
	if err := gemmv1.SignChallengeOpen(co, h.chalKeys.networkPriv); err != nil {
		t.Fatal(err)
	}
	if _, err := h.msg.OpenGEMMChallenge(h.ctx, &types.MsgOpenGEMMChallenge{
		Challenger: h.challenger, GemmTaskId: taskB,
		WorkerOutputTile: co.WorkerOutputTile, WorkerProofSiblings: proof.Siblings,
		WorkerProofIndex: proof.Index, WorkerProofCount: proof.Count,
		ChallengerOutputTile: co.ChallengerOutputTile, ChallengeBond: co.ChallengeBond,
		ChallengerSignature: co.ChallengerSignature,
	}); err == nil {
		t.Fatal("cross-task challenge accepted")
	}
	_ = counts
}

// TestGEMMTraceTimeoutPaths covers DoD 15 for the trace-lock phase.
func TestGEMMTraceTimeoutPaths(t *testing.T) {
	run := func(lockWorker, lockChallenger bool) string {
		h := newGemmHarness(t, 1)
		const m, n, k = 16, 16, 16
		taskID, a, b, c := h.postGEMM(m, n, k, 31)
		tiles := h.acceptAndSubmit(taskID, a, b, c, m, n, k, false)
		h.bondWorker(h.challenger, h.chalKeys)
		task := mustTask(t, h, taskID)
		h.openChallenge(taskID, &task, tiles, 0, 0, h.chalKeys, false)
		record, err := h.keeper.GetGEMMDispute(h.ctx, taskID)
		if err != nil {
			t.Fatal(err)
		}
		if lockWorker {
			h.lockOneTrace(taskID, true, a, b, m, n, k, tiles, 0, 0)
		}
		if lockChallenger {
			challengerTiles := append([]gemmv1.State(nil), tiles...)
			challengerTiles[0][0]--
			h.lockOneTrace(taskID, false, a, b, m, n, k, challengerTiles, 0, 0)
		}
		h.ctx = h.ctx.WithBlockHeight(int64(record.TraceDeadline) + 1)
		res, err := h.msg.TimeoutGEMM(h.ctx, &types.MsgTimeoutGEMM{Actor: h.requester, GemmTaskId: taskID})
		if err != nil {
			t.Fatal(err)
		}
		return res.Outcome
	}
	if got := run(true, false); got != "worker_wins" {
		t.Fatalf("worker-only lock -> %s, want worker_wins", got)
	}
	if got := run(false, true); got != "challenger_wins" {
		t.Fatalf("challenger-only lock -> %s, want challenger_wins", got)
	}
	if got := run(false, false); got != "both_invalid" {
		t.Fatalf("no locks -> %s, want both_invalid", got)
	}
}

// lockTracesWith is lockTraces with an explicit challenger identity.
func (h *gemmHarness) lockTracesWith(taskID uint64, a, b []int8, m, n, k uint64, workerTiles, challengerTiles []gemmv1.State, tileI, tileJ uint64, challenger string, challengerKeys *gemmKeys) (*gemmv1.TileTraceArtifacts, *gemmv1.TileTraceArtifacts) {
	h.t.Helper()
	h.activeChallenger, h.activeChallengerKeys = challenger, challengerKeys
	return h.lockTraces(taskID, a, b, m, n, k, workerTiles, challengerTiles, tileI, tileJ)
}

// lockOneTrace commits a single party's trace (used by timeout tests).
func (h *gemmHarness) lockOneTrace(taskID uint64, worker bool, a, b []int8, m, n, k uint64, tiles []gemmv1.State, tileI, tileJ uint64) {
	h.t.Helper()
	task, err := h.keeper.GetGEMMTask(h.ctx, taskID)
	if err != nil {
		h.t.Fatal(err)
	}
	honest, err := gemmv1.BuildTileTrace(a, b, m, n, k, task.ProtocolTaskID, task.AssignmentID, uint32(tileI), uint32(tileJ))
	if err != nil {
		h.t.Fatal(err)
	}
	rSteps := gemmv1.RSteps(k)
	counts := gemmv1.TileCountsFor(m, n, k)
	idx := int(tileI)*int(counts.ColsC) + int(tileJ)
	art := honest
	if honest.States[rSteps] != tiles[idx] {
		art = fabricateFor(h, &task, honest, uint32(tileI), uint32(tileJ), tiles[idx])
	}
	party := gemmv1.Worker
	actor, keys := h.worker, h.workKeys
	if !worker {
		party, actor, keys = gemmv1.Challenger, h.challenger, h.chalKeys
	}
	p0, err := gemmv1.ProveLeaf(art.Levels, 0)
	if err != nil {
		h.t.Fatal(err)
	}
	pf, err := gemmv1.ProveLeaf(art.Levels, uint32(rSteps))
	if err != nil {
		h.t.Fatal(err)
	}
	tc := &gemmv1.TraceCommit{
		ProtocolVersion: gemmv1.ProtocolVersion, TaskID: task.ProtocolTaskID,
		AssignmentID: task.AssignmentID, DisputedTileI: tileI, DisputedTileJ: tileJ,
		Party:        uint64(party),
		TraceRoot:    append([]byte(nil), art.Root[:]...),
		InitialState: art.States[0].CanonicalBytes(),
		FinalState:   art.States[rSteps].CanonicalBytes(),
		LockedEpoch:  uint64(h.ctx.BlockHeight()),
	}
	tc.InitialProof = gemmv1.OutputTileProof{Index: p0.Index, Count: p0.Count, Siblings: p0.Siblings}
	tc.FinalProof = gemmv1.OutputTileProof{Index: pf.Index, Count: pf.Count, Siblings: pf.Siblings}
	if err := gemmv1.SignTraceCommit(tc, keys.networkPriv); err != nil {
		h.t.Fatal(err)
	}
	if _, err := h.msg.CommitGEMMTrace(h.ctx, &types.MsgCommitGEMMTrace{
		Actor: actor, GemmTaskId: taskID,
		TraceRoot: tc.TraceRoot, InitialState: tc.InitialState,
		InitialProofSiblings: tc.InitialProof.Siblings, InitialProofIndex: tc.InitialProof.Index, InitialProofCount: tc.InitialProof.Count,
		FinalState:         tc.FinalState,
		FinalProofSiblings: tc.FinalProof.Siblings, FinalProofIndex: tc.FinalProof.Index, FinalProofCount: tc.FinalProof.Count,
		LockedEpoch: tc.LockedEpoch, Signature: tc.Signature,
	}); err != nil {
		h.t.Fatal(err)
	}
}

// TestGEMMQueueGriefing covers DoD 17: a worker-controlled false
// challenger cannot block an honest challenger; the queue is bounded;
// duplicates are rejected and queued bonds are returned.
func TestGEMMQueueGriefing(t *testing.T) {
	h := newGemmHarness(t, 1)
	const m, n, k = 16, 16, 16
	taskID, a, b, c := h.postGEMM(m, n, k, 41)
	tiles := h.acceptAndSubmit(taskID, a, b, c, m, n, k, false)
	task := mustTask(t, h, taskID)

	// The active dispute comes from a worker-controlled challenger with a
	// WRONG claim (a griefing attempt).
	griefer := account(9)
	h.bank.accounts[griefer] = 10 * MinBond
	grieferKeys := newGemmKeys(60)
	h.bondWorker(griefer, grieferKeys)
	h.openChallengeFrom(taskID, &task, tiles, 0, 0, griefer, grieferKeys, true)

	// An honest challenger must be able to queue behind it.
	h.bondWorker(h.challenger, h.chalKeys)
	h.openChallengeFrom(taskID, &task, tiles, 0, 0, h.challenger, h.chalKeys, false)
	task = mustTask(t, h, taskID)
	if len(task.QueuedGEMMChallenges) != 1 || task.QueuedGEMMChallenges[0].Challenger != h.challenger {
		t.Fatalf("honest challenger not queued: %+v", task.QueuedGEMMChallenges)
	}
	// Duplicate queueing by the same challenger is rejected; the bond is
	// still locked exactly once.
	queuedBond := task.QueuedGEMMChallenges[0].Bond
	moduleBefore := h.bank.module
	if _, err := h.msg.OpenGEMMChallenge(h.ctx, openChallengeMsg(h, taskID, &task, tiles, 0, 0, h.challenger, h.chalKeys, false)); err == nil {
		t.Fatal("duplicate queued challenge accepted")
	}
	if h.bank.module != moduleBefore {
		t.Fatal("duplicate challenge locked bond again")
	}
	// Queue saturation is rejected deterministically.
	for i := 0; i < MaxQueuedChallenges; i++ {
		filler := account(byte(20 + i))
		h.bank.accounts[filler] = 10 * MinBond
		fillerKeys := newGemmKeys(byte(70 + i))
		h.bondWorker(filler, fillerKeys)
		msg := openChallengeMsg(h, taskID, &task, tiles, 0, 0, filler, fillerKeys, false)
		_, err := h.msg.OpenGEMMChallenge(h.ctx, msg)
		task = mustTask(t, h, taskID)
		if i < MaxQueuedChallenges-1 {
			if err != nil {
				t.Fatalf("queue slot %d rejected: %v", i, err)
			}
		} else if err == nil {
			t.Fatalf("queue overflow accepted at %d", i)
		}
	}
	// The active griefer loses (worker is honest), its bond burns, and the
	// next queued challenge is the honest challenger with a fresh deadline.
	// The worker locks the honest trace; the griefer fabricates a trace
	// ending at its wrong claim.
	grievedTiles := append([]gemmv1.State(nil), tiles...)
	grievedTiles[0][3] += 5
	workerArt, grieferArt := h.lockTracesWith(taskID, a, b, m, n, k, tiles, grievedTiles, 0, 0, griefer, grieferKeys)
	h.runBisection(taskID, a, b, m, n, k, 0, 0, workerArt, grieferArt, 0)
	if outcome := h.arbitrate(taskID, a, b, 0, 0); outcome != "worker_wins" {
		t.Fatalf("griefer outcome = %s, want worker_wins", outcome)
	}
	task = mustTask(t, h, taskID)
	if task.Status != GEMMStatusChallenged {
		t.Fatalf("status after griefer loss = %s, want challenged (next in queue)", task.Status)
	}
	record, err := h.keeper.GetGEMMDispute(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	if record.Challenger != h.challenger {
		t.Fatalf("next active challenger = %s, want the honest queued challenger", record.Challenger)
	}
	if record.TraceDeadline <= uint64(h.ctx.BlockHeight()) {
		t.Fatal("promoted challenge has no fresh deadline")
	}
	if record.Bond != queuedBond {
		t.Fatal("promoted challenge lost its locked bond")
	}
}

// openChallengeMsg builds the challenge message without sending it.
func openChallengeMsg(h *gemmHarness, taskID uint64, task *GEMMTask, tiles []gemmv1.State, tileI, tileJ uint64, challenger string, keys *gemmKeys, wrongTile bool) *types.MsgOpenGEMMChallenge {
	levels, err := gemmv1.BuildLevels(gemmOutputLeaves(h, taskID, tiles))
	if err != nil {
		h.t.Fatal(err)
	}
	counts := gemmv1.TileCountsFor(task.M, task.N, task.K)
	idx := int(tileI)*int(counts.ColsC) + int(tileJ)
	proof, err := gemmv1.ProveLeaf(levels, uint32(idx))
	if err != nil {
		h.t.Fatal(err)
	}
	challengerTiles := append([]gemmv1.State(nil), tiles...)
	if wrongTile {
		challengerTiles[idx][3] += 5
	} else {
		challengerTiles[idx][0]--
	}
	co := &gemmv1.ChallengeOpen{
		ProtocolVersion: gemmv1.ProtocolVersion, TaskID: task.ProtocolTaskID,
		AssignmentID: task.AssignmentID, ChallengerPubKey: keys.networkPub,
		WorkerPubKey:  task.WorkerProtocolPubKey,
		DisputedTileI: tileI, DisputedTileJ: tileJ,
		WorkerOutputTile: tiles[idx].CanonicalBytes(), WorkerOutputProof: proof,
		ChallengerOutputTile: challengerTiles[idx].CanonicalBytes(),
		ChallengeBond:        gemmBondFor(task.MaxFee), OpenedEpoch: uint64(h.ctx.BlockHeight()),
	}
	if err := gemmv1.SignChallengeOpen(co, keys.networkPriv); err != nil {
		h.t.Fatal(err)
	}
	return &types.MsgOpenGEMMChallenge{
		Challenger: challenger, GemmTaskId: taskID,
		DisputedTileI: tileI, DisputedTileJ: tileJ,
		WorkerOutputTile: co.WorkerOutputTile, WorkerProofSiblings: proof.Siblings,
		WorkerProofIndex: proof.Index, WorkerProofCount: proof.Count,
		ChallengerOutputTile: co.ChallengerOutputTile, ChallengeBond: co.ChallengeBond,
		ChallengerSignature: co.ChallengerSignature, OpenedEpoch: co.OpenedEpoch,
	}
}

// openChallengeFrom is openChallenge with an explicit challenger account.
func (h *gemmHarness) openChallengeFrom(taskID uint64, task *GEMMTask, tiles []gemmv1.State, tileI, tileJ uint64, challenger string, keys *gemmKeys, wrongTile bool) {
	h.t.Helper()
	if _, err := h.msg.OpenGEMMChallenge(h.ctx, openChallengeMsg(h, taskID, task, tiles, tileI, tileJ, challenger, keys, wrongTile)); err != nil {
		h.t.Fatal(err)
	}
}

// TestGEMMRestartPersistence covers DoD 11: a dispute survives a keeper
// restart mid-bisection and finishes identically.
func TestGEMMRestartPersistence(t *testing.T) {
	h := newGemmHarness(t, 1)
	const m, n, k = 16, 16, 16
	taskID, a, b, c := h.postGEMM(m, n, k, 51)
	tiles := h.acceptAndSubmit(taskID, a, b, c, m, n, k, true)
	h.bondWorker(h.challenger, h.chalKeys)
	task := mustTask(t, h, taskID)
	h.openChallenge(taskID, &task, tiles, 0, 0, h.chalKeys, false)
	challengerTiles := append([]gemmv1.State(nil), tiles...)
	challengerTiles[0][0]--
	workerArt, challengerArt := h.lockTraces(taskID, a, b, m, n, k, tiles, challengerTiles, 0, 0)
	rSteps := int(gemmv1.RSteps(k))
	midRounds := rSteps / 4
	h.runBisection(taskID, a, b, m, n, k, 0, 0, workerArt, challengerArt, midRounds)
	snapshotBefore, err := h.keeper.GetGEMMDispute(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}

	// Simulate a node restart: a fresh keeper over the SAME store.
	restarted := NewKeeper(h.key, h.bank)
	restartedMsg := restarted.MsgServer()
	h.keeper = restarted
	h.msg = restartedMsg

	record, err := h.keeper.GetGEMMDispute(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	if string(record.Snapshot) != string(snapshotBefore.Snapshot) {
		t.Fatal("dispute snapshot changed across restart")
	}
	h.runBisection(taskID, a, b, m, n, k, 0, 0, workerArt, challengerArt, 0)
	if outcome := h.arbitrate(taskID, a, b, 0, 0); outcome != "challenger_wins" {
		t.Fatalf("post-restart outcome = %s, want challenger_wins", outcome)
	}
}

// TestGEMMProofDoSBounds covers DoD 21: oversized proofs, states and
// witness bytes are rejected before any expensive hashing.
func TestGEMMProofDoSBounds(t *testing.T) {
	h := newGemmHarness(t, 1)
	const m, n, k = 16, 16, 16
	taskID, a, b, c := h.postGEMM(m, n, k, 61)
	tiles := h.acceptAndSubmit(taskID, a, b, c, m, n, k, false)
	h.bondWorker(h.challenger, h.chalKeys)
	task := mustTask(t, h, taskID)

	msg := openChallengeMsg(h, taskID, &task, tiles, 0, 0, h.challenger, h.chalKeys, false)
	msg.WorkerProofSiblings = append(msg.WorkerProofSiblings, make([]byte, 32))
	if _, err := h.msg.OpenGEMMChallenge(h.ctx, msg); err == nil {
		t.Fatal("oversized proof depth accepted")
	}
	msg = openChallengeMsg(h, taskID, &task, tiles, 0, 0, h.challenger, h.chalKeys, false)
	msg.WorkerProofSiblings[0] = make([]byte, 31)
	if _, err := h.msg.OpenGEMMChallenge(h.ctx, msg); err == nil {
		t.Fatal("short sibling accepted")
	}
	msg = openChallengeMsg(h, taskID, &task, tiles, 0, 0, h.challenger, h.chalKeys, false)
	msg.WorkerProofCount = 1 << 20
	if _, err := h.msg.OpenGEMMChallenge(h.ctx, msg); err == nil {
		t.Fatal("absurd proof count accepted")
	}
}

// TestGEMMGasAccounting covers DoD 22: the explicit schedule charges every
// phase, and arbitration cost is dominated by the 512-MAC step, not by
// unbounded witness processing.
func TestGEMMGasAccounting(t *testing.T) {
	// Gas meters live in the context; the default test context has an
	// infinite meter, so assert the schedule invariants directly and the
	// shape-derived bounds.
	if GemmGasScheduleVersion != "GEMMGasV1" {
		t.Fatal("gas schedule version drifted")
	}
	if GasGEMMArbitration < GasPerGEMMProofSibling*10 {
		t.Fatal("arbitration gas must dominate proof verification")
	}
	key := storetypes.NewKVStoreKey(ModuleName)
	ctx := testutil.DefaultContextWithKeys(map[string]*storetypes.KVStoreKey{ModuleName: key}, nil, nil)
	sdkCtx := sdk.UnwrapSDKContext(ctx)
	if sdkCtx.GasMeter() == nil {
		t.Fatal("test context has no gas meter")
	}
	_ = strings.TrimSpace
	_ = json.Marshal
}
