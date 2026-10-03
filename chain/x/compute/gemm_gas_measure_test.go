package compute

// Gas measurement for the GEMM settlement path. Each handler runs against
// a real gas meter and the consumed units are recorded per phase; the JSON
// artifact is written by TestGEMMGasMeasurement when PRISMA_GAS_OUT is set.
// Gas units are consensus resource accounting under GEMMGasV1, never CPU
// nanoseconds.

import (
	"encoding/json"
	"os"
	"testing"
	"time"

	storetypes "cosmossdk.io/store/types"
	"github.com/cosmos/cosmos-sdk/testutil"
	"prismachain/chain/x/compute/types"
	"prismachain/compute/gemmv1"
)

type gasRecord struct {
	Phase        string  `json:"phase"`
	Gas          uint64  `json:"gas_units"`
	WallMs       float64 `json:"wall_ms"`
	ProofDepth   int     `json:"proof_depth,omitempty"`
	WitnessBytes int     `json:"witness_bytes,omitempty"`
}

func meteredHarness(t *testing.T, meter storetypes.GasMeter) (*gemmHarness, func() uint64) {
	t.Helper()
	key := storetypes.NewKVStoreKey(ModuleName)
	base := testutil.DefaultContextWithKeys(map[string]*storetypes.KVStoreKey{ModuleName: key}, nil, nil).
		WithBlockHeight(1).WithChainID("prisma-test-1")
	ctx := base.WithGasMeter(meter)
	h := &gemmHarness{
		t: t, ctx: ctx, key: key,
		bank:      &memoryBank{accounts: map[string]uint64{}},
		requester: account(1), worker: account(2), challenger: account(3),
		monitorA: account(4), monitorB: account(5),
		reqKeys: newGemmKeys(10), workKeys: newGemmKeys(20), chalKeys: newGemmKeys(30),
	}
	h.keeper = NewKeeper(key, h.bank)
	h.msg = h.keeper.MsgServer()
	for _, a := range []string{h.requester, h.worker, h.challenger, h.monitorA, h.monitorB} {
		h.bank.accounts[a] = 10 * MinBond
	}
	used := func() uint64 { return meter.GasConsumed() }
	return h, used
}

// TestGEMMGasMeasurement runs one full honest flow and one full fraud flow
// with a metering context and records per-phase gas and wall time.
func TestGEMMGasMeasurement(t *testing.T) {
	const m, n, k = 64, 64, 64
	meter := storetypes.NewGasMeter(1 << 62)
	h, usedGas := meteredHarness(t, meter)
	var records []gasRecord
	phase := func(name string, fn func()) {
		before := usedGas()
		start := time.Now()
		fn()
		records = append(records, gasRecord{
			Phase: name, Gas: usedGas() - before,
			WallMs: float64(time.Since(start).Microseconds()) / 1000.0,
		})
	}

	var taskID uint64
	var a, b []int8
	var c []int32
	phase("PostGEMMTask", func() {
		taskID, a, b, c = h.postGEMM(m, n, k, 77)
	})
	phase("AcceptGEMMTask", func() {
		h.bondWorker(h.worker, h.workKeys)
		if _, err := h.msg.AcceptGEMMTask(h.ctx, &types.MsgAcceptGEMMTask{Worker: h.worker, GemmTaskId: taskID,
			AssignmentNonce: []byte("assignment-nonce-01")}); err != nil {
			t.Fatal(err)
		}
	})
	var tiles []gemmv1.State
	phase("SubmitGEMMResult", func() {
		tiles = h.acceptAndSubmitResultOnly(taskID, a, b, c, m, n, k, true)
	})
	phase("OpenGEMMChallenge", func() {
		h.bondWorker(h.challenger, h.chalKeys)
		task := mustTask(t, h, taskID)
		h.openChallenge(taskID, &task, tiles, 0, 0, h.chalKeys, false)
	})
	challengerTiles := append([]gemmv1.State(nil), tiles...)
	challengerTiles[0][0]--
	var workerArt, challengerArt *gemmv1.TileTraceArtifacts
	phase("CommitGEMMTrace(worker)", func() {
		workerArt = h.lockOneTraceOnly(taskID, true, a, b, m, n, k, tiles, 0, 0)
	})
	phase("CommitGEMMTrace(challenger)", func() {
		challengerArt = h.lockOneTraceOnly(taskID, false, a, b, m, n, k, challengerTiles, 0, 0)
	})
	phase("BisectionRounds", func() {
		h.runBisection(taskID, a, b, m, n, k, 0, 0, workerArt, challengerArt, 0)
	})
	phase("ArbitrateGEMM", func() {
		if outcome := h.arbitrate(taskID, a, b, 0, 0); outcome != "challenger_wins" {
			t.Fatalf("outcome %s", outcome)
		}
	})

	// Honest settlement gas on a second task.
	meter2 := storetypes.NewGasMeter(1 << 62)
	h2, usedGas2 := meteredHarness(t, meter2)
	var records2 []gasRecord
	phase2 := func(name string, fn func()) {
		before := usedGas2()
		start := time.Now()
		fn()
		records2 = append(records2, gasRecord{Phase: name, Gas: usedGas2() - before,
			WallMs: float64(time.Since(start).Microseconds()) / 1000.0})
	}
	var taskID2 uint64
	var a2, b2 []int8
	var c2 []int32
	phase2("PostGEMMTask", func() { taskID2, a2, b2, c2 = h2.postGEMM(m, n, k, 78) })
	phase2("AcceptAndSubmit", func() { h2.acceptAndSubmit(taskID2, a2, b2, c2, m, n, k, false) })
	var task2 GEMMTask
	phase2("Attestx2", func() {
		monitorKeys := map[string]*gemmKeys{h2.monitorA: newGemmKeys(41), h2.monitorB: newGemmKeys(42)}
		for _, monitor := range []string{h2.monitorA, h2.monitorB} {
			h2.bondWorker(monitor, monitorKeys[monitor])
			if _, err := h2.msg.AttestGEMMTask(h2.ctx, &types.MsgAttestGEMMTask{Monitor: monitor, GemmTaskId: taskID2}); err != nil {
				t.Fatal(err)
			}
		}
		task2 = mustTask(t, h2, taskID2)
	})
	phase2("FinalizeGEMM", func() {
		h2.ctx = h2.ctx.WithBlockHeight(int64(task2.ChallengeEnd) + 1)
		if _, err := h2.msg.FinalizeGEMM(h2.ctx, &types.MsgFinalizeGEMM{Actor: h2.requester, GemmTaskId: taskID2}); err != nil {
			t.Fatal(err)
		}
	})

	depth := gemmv1.DepthFor(uint32(gemmv1.RSteps(k)) + 1)
	for i := range records {
		if records[i].Phase == "ArbitrateGEMM" {
			records[i].ProofDepth = depth
			records[i].WitnessBytes = 64 + 64 + 2*(depth*32)
		}
	}
	out := map[string]any{
		"schedule":        GemmGasScheduleVersion,
		"shape":           map[string]uint64{"m": m, "n": n, "k": k},
		"fraud_flow":      records,
		"honest_flow":     records2,
		"arbitration_mac": 512,
		"note":            "gas units are consensus resource accounting under GEMMGasV1, not CPU nanoseconds; wall_ms is reported separately",
	}
	for _, r := range append(append([]gasRecord(nil), records...), records2...) {
		t.Logf("%-28s gas=%8d wall=%.2fms", r.Phase, r.Gas, r.WallMs)
	}
	if path := os.Getenv("PRISMA_GAS_OUT"); path != "" {
		data, err := json.MarshalIndent(out, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("written %s", path)
	}
}

// acceptAndSubmitResultOnly submits a result on an already-accepted task.
func (h *gemmHarness) acceptAndSubmitResultOnly(taskID uint64, a, b []int8, c []int32, m, n, k uint64, corrupted bool) []gemmv1.State {
	h.t.Helper()
	tiles := gemmv1.OutputTiles(c, m, n)
	if corrupted {
		tiles[0][0]++
	}
	counts := gemmv1.TileCountsFor(m, n, k)
	task, err := h.keeper.GetGEMMTask(h.ctx, taskID)
	if err != nil {
		h.t.Fatal(err)
	}
	leaves := gemmv1.OutputLeaves(task.ProtocolTaskID, task.AssignmentID, tiles, uint32(counts.ColsC))
	root, err := gemmv1.MerkleRoot(leaves)
	if err != nil {
		h.t.Fatal(err)
	}
	rc := task.gemmResultCommit(root[:], uint64(h.ctx.BlockHeight()))
	if err := gemmv1.SignResultCommit(rc, h.workKeys.networkPriv); err != nil {
		h.t.Fatal(err)
	}
	if _, err := h.msg.SubmitGEMMResult(h.ctx, &types.MsgSubmitGEMMResult{
		Worker: h.worker, GemmTaskId: taskID, OutputRoot: root[:],
		OutputDataRef: "dev://output", OutputBytes: m * n * 4,
		CompletedEpoch: uint64(h.ctx.BlockHeight()), WorkerSignature: rc.WorkerSignature,
	}); err != nil {
		h.t.Fatal(err)
	}
	return tiles
}

// lockOneTraceOnly commits one party's trace and returns its artifacts.
func (h *gemmHarness) lockOneTraceOnly(taskID uint64, worker bool, a, b []int8, m, n, k uint64, tiles []gemmv1.State, tileI, tileJ uint64) *gemmv1.TileTraceArtifacts {
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
		party, actor, keys = gemmv1.Challenger, h.challengerActor(), h.challengerKeysOf()
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
	return art
}
