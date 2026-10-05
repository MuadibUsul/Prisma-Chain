package compute

// A2 end-to-end: a self-consistent fraudulent wide GEMM result settles
// through graph bisection -> wide dispute trace bisection -> exactly
// 512-MAC on-chain arbitration -> ChallengerWins -> fraud, zero VWR.

import (
	"encoding/binary"
	"encoding/json"
	"testing"

	"prismachain/chain/x/compute/types"
	"prismachain/compute/canonical"
)

// buildChainGraphV2WideFixture is a K=8 two-node V2 block (wide GEMM +
// requant) suitable for the wide dispute geometry rules.
func buildChainGraphV2WideFixture(t *testing.T) *chainGraphV2Fixture {
	t.Helper()
	mk := func(desc canonical.TensorDescriptorV2, data []int64) *canonical.TensorV2 {
		tensor, err := canonical.NewTensorV2(desc, data)
		if err != nil {
			t.Fatal(err)
		}
		return tensor
	}
	aData := []int64{1, -2, 3, -4, 5, -6, 7, -8, 4095, -4096, 100, -100, 5, 6, -7, 8}
	wData := make([]int64, 64)
	for i := range wData {
		wData[i] = int64(i%17) - 8
	}
	a := mk(canonical.NewDescV2(canonical.DtypeV2A13, 2, 8), aData)
	w := mk(canonical.NewDescV2(canonical.DtypeV2W10, 8, 8), wData)
	aroot, _ := a.TensorRootV2()
	wroot, _ := w.TensorRootV2()
	g := &canonical.GraphDescriptorV2{
		ProtocolVersion: canonical.ProtocolVersionGraphV2,
		Spec:            "TEST_WIDE_DISPUTE_V1",
		Arithmetic:      canonical.A13W10I64Profile(),
		Inputs: []canonical.GraphInputV2{
			{Name: "a", Desc: a.Desc, Root: aroot[:]},
			{Name: "w", Desc: w.Desc, Root: wroot[:]},
		},
		Nodes: []canonical.GraphNodeV2{
			{NodeID: 0, OperatorID: canonical.OpGEMMWideA13W10, Version: canonical.OpGEMMVersionWide,
				Inputs: []canonical.TensorRef{{Kind: 0, Index: 0}, {Kind: 0, Index: 1}},
				Output: canonical.NewDescV2(canonical.DtypeV2Int64Accum, 2, 8),
				Params: canonical.ParamList{}},
			{NodeID: 1, OperatorID: canonical.OpRequantizeWideV1, Version: canonical.OpRequantVersionWide,
				Inputs: []canonical.TensorRef{{Kind: 1, Index: 0}},
				Output: canonical.NewDescV2(canonical.DtypeV2A13, 2, 8),
				Params: canonical.ParamList{
					{Key: "clamp_hi", Value: canonical.A13Max},
					{Key: "clamp_lo", Value: canonical.A13Min},
					{Key: "mult", Value: 1 << 20},
					{Key: "out_dtype", Value: int64(canonical.DtypeV2A13)},
					{Key: "shift", Value: 20},
				}},
		},
		Outputs: []canonical.TensorRef{{Kind: 1, Index: 1}},
	}
	exec, err := canonical.ExecuteGraphV2(g, map[uint32]*canonical.TensorV2{0: a, 1: w})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(g)
	if err != nil {
		t.Fatal(err)
	}
	return &chainGraphV2Fixture{graph: g, graphJSON: raw,
		inputs: map[uint32]*canonical.TensorV2{0: a, 1: w}, exec: exec}
}

func mustRootV2Chain(t *testing.T, tensor *canonical.TensorV2) canonical.Hash {
	t.Helper()
	root, err := tensor.TensorRootV2()
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func TestWideGEMMDisputeE2E(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphV2WideFixture(t)
	taskID, _ := h.postGraphV2(t, fx)
	assignmentRef := h.acceptGraph(t, taskID)

	node0 := fx.graph.Nodes[0]
	honest0 := fx.exec.Tensors[canonical.TensorRef{Kind: 1, Index: 0}]
	honest1 := fx.exec.Tensors[canonical.TensorRef{Kind: 1, Index: 1}]
	// fraudulent node-0 output (one element altered), node 1 recomputed
	fraud0Data := append([]int64(nil), honest0.Data...)
	fraud0Data[0]++
	fraud0, err := canonical.NewTensorV2(node0.Output, fraud0Data)
	if err != nil {
		t.Fatal(err)
	}
	fraud1Data := make([]int64, len(honest1.Data))
	for i, v := range fraud0Data {
		clamped := v
		if clamped > canonical.A13Max {
			clamped = canonical.A13Max
		}
		if clamped < canonical.A13Min {
			clamped = canonical.A13Min
		}
		fraud1Data[i] = clamped
	}
	fraud1, err := canonical.NewTensorV2(fx.graph.Nodes[1].Output, fraud1Data)
	if err != nil {
		t.Fatal(err)
	}
	fraudTensors := map[canonical.TensorRef]*canonical.TensorV2{}
	for k, v := range fx.exec.Tensors {
		fraudTensors[k] = v
	}
	fraudTensors[canonical.TensorRef{Kind: 1, Index: 0}] = fraud0
	fraudTensors[canonical.TensorRef{Kind: 1, Index: 1}] = fraud1
	fraudFinalRoot := mustRootV2Chain(t, fraud1)
	fraudExec := &canonical.GraphExecutionV2{Tensors: fraudTensors,
		Outputs: []canonical.Hash{fraudFinalRoot}}

	// worker commits the fraudulent (self-consistent) result
	var taskRef [8]byte
	binary.BigEndian.PutUint64(taskRef[:], taskID)
	rc, err := canonical.NewGraphResultCommitV3(fx.graph, taskRef[:], assignmentRef,
		h.workKeys.networkPub, fraudExec, 42)
	if err != nil {
		t.Fatal(err)
	}
	if err := canonical.SignGraphResultCommitV3(rc, h.workKeys.networkPriv); err != nil {
		t.Fatal(err)
	}
	if _, err := h.msg.SubmitGraphResultV2(h.ctx, &types.MsgSubmitGraphResultV2{
		Worker: h.worker, GraphTaskId: taskID,
		NodeOutputManifestRoot: rc.NodeOutputManifestRootV2,
		OutputRoots:            rc.OutputRoots, FinalOutputRoot: rc.FinalOutputRoot,
		CompletedEpoch: rc.CompletedEpoch, WorkerSignature: rc.Signature,
	}); err != nil {
		t.Fatal(err)
	}
	h.bondWorker(h.challenger, h.chalKeys)
	if _, err := h.msg.OpenGraphChallenge(h.ctx, &types.MsgOpenGraphChallenge{
		Challenger: h.challenger, GraphTaskId: taskID,
		ChallengerOutputRoots: [][]byte{fx.exec.Outputs[0][:]},
		ChallengeBond:         MinBond,
	}); err != nil {
		t.Fatal(err)
	}

	// graph trails diverging after node 0 (the fraudulent GEMM): the three
	// states are S0 (inputs), S1 (after the GEMM), S2 (final)
	graphID, _ := fx.graph.GraphIDV2()
	inputsOnly := map[canonical.TensorRef]canonical.Hash{}
	for i, in := range fx.graph.Inputs {
		var hh canonical.Hash
		copy(hh[:], in.Root)
		inputsOnly[canonical.TensorRef{Kind: 0, Index: uint32(i)}] = hh
	}
	afterNode0 := map[canonical.TensorRef]canonical.Hash{}
	for k, v := range inputsOnly {
		afterNode0[k] = v
	}
	afterNode0[canonical.TensorRef{Kind: 1, Index: 0}] = mustRootV2Chain(t, fraud0)
	fraudStateRoot, err := canonical.GraphStateRootV2FromRoots(afterNode0)
	if err != nil {
		t.Fatal(err)
	}
	liveFraudFinal := map[canonical.TensorRef]canonical.Hash{}
	for k, v := range afterNode0 {
		liveFraudFinal[k] = v
	}
	liveFraudFinal[canonical.TensorRef{Kind: 1, Index: 1}] = mustRootV2Chain(t, fraud1)
	fraudFinalStateRoot, err := canonical.GraphStateRootV2FromRoots(liveFraudFinal)
	if err != nil {
		t.Fatal(err)
	}
	honestTrail := fx.exec.Trail
	fraudTrail := []canonical.Hash{honestTrail[0], fraudStateRoot, fraudFinalStateRoot}
	lockTrail := func(party string, trail []canonical.Hash) {
		pi, _ := canonical.TrailProofV2(graphID, trail, 0)
		pf, _ := canonical.TrailProofV2(graphID, trail, uint32(len(trail)-1))
		root, _ := canonical.TrailRootV2(graphID, trail)
		if _, err := h.msg.GraphTrailClaim(h.ctx, &types.MsgGraphTrailClaim{
			Party: party, GraphTaskId: taskID,
			TrailRoot: root[:], InitialRoot: trail[0][:], InitialProof: siblingsOfV2(pi),
			FinalRoot: trail[len(trail)-1][:], FinalProof: siblingsOfV2(pf),
		}); err != nil {
			t.Fatal(err)
		}
	}
	lockTrail(h.worker, fraudTrail)
	lockTrail(h.challenger, honestTrail)
	for rounds := 0; rounds < 8; rounds++ {
		record, err := h.keeper.GetGraphDispute(h.ctx, taskID)
		if err != nil {
			t.Fatal(err)
		}
		if record.Status == GraphDisputeArbReady {
			break
		}
		dispute, err := canonical.RestoreGraphDisputeV2(record.Snapshot, fx.graph, ChallengeRoundBlocks)
		if err != nil {
			t.Fatal(err)
		}
		low, high := dispute.Interval()
		mid := low + (high-low)/2
		submitMid := func(party string, trail []canonical.Hash) {
			proof, _ := canonical.TrailProofV2(graphID, trail, mid)
			h.advance(h.currentHeight() + 1)
			if _, err := h.msg.GraphMidPoint(h.ctx, &types.MsgGraphMidPoint{
				Party: party, GraphTaskId: taskID,
				StateRoot: trail[mid][:], ProofSiblings: siblingsOfV2(proof), Epoch: h.currentHeight(),
			}); err != nil {
				t.Fatal(err)
			}
		}
		submitMid(h.worker, fraudTrail)
		submitMid(h.challenger, honestTrail)
	}
	record, err := h.keeper.GetGraphDispute(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := canonical.RestoreGraphDisputeV2(record.Snapshot, fx.graph, ChallengeRoundBlocks)
	if err != nil {
		t.Fatal(err)
	}
	nodeID, err := restored.FirstDivergentNode()
	if err != nil || nodeID != 0 {
		t.Fatalf("first divergent node %d (err %v), want 0 (the GEMM)", nodeID, err)
	}

	// open the wide dispute on the GEMM node, tile (0,0)
	if _, err := h.msg.OpenWideGEMMDispute(h.ctx, &types.MsgOpenWideGEMMDispute{
		Challenger: h.challenger, GraphTaskId: taskID, NodeId: 0,
		TileI: 0, TileJ: 0, ChallengeBond: MinBond,
	}); err != nil {
		t.Fatal(err)
	}
	// a non-GEMM node is refused
	if _, err := h.msg.OpenWideGEMMDispute(h.ctx, &types.MsgOpenWideGEMMDispute{
		Challenger: h.challenger, GraphTaskId: taskID, NodeId: 1,
		TileI: 0, TileJ: 0, ChallengeBond: MinBond,
	}); err == nil {
		t.Fatal("wide dispute opened on a non-GEMM node")
	}

	// partial-state traces: S1 = A_tile * W_tile over the single K step
	aTile, err := canonical.WideATile(fx.inputs[0].Data, 2, 8, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	wTile, err := canonical.WideWTile(fx.inputs[1].Data, 8, 8, 0, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	honestS1 := canonical.WideTileStep(canonical.WideTileState{}, aTile, wTile)
	fraudS1 := honestS1
	fraudS1[0]++
	claimTrace := func(party string, s1 canonical.WideTileState) {
		states := []canonical.WideTileState{{}, s1}
		root, err := canonical.WideTraceRootV1(0, 0, states)
		if err != nil {
			t.Fatal(err)
		}
		p0, _ := canonical.WideTraceProofV1(0, 0, states, 0)
		p1, _ := canonical.WideTraceProofV1(0, 0, states, 1)
		if _, err := h.msg.WideTraceClaim(h.ctx, &types.MsgWideTraceClaim{
			Party: party, GraphTaskId: taskID,
			TraceRoot: root[:], InitialState: states[0].Bytes(), InitialProof: siblingsOfV2(p0),
			FinalState: states[1].Bytes(), FinalProof: siblingsOfV2(p1),
		}); err != nil {
			t.Fatal(err)
		}
	}
	claimTrace(h.worker, fraudS1)
	claimTrace(h.challenger, honestS1)
	wd, err := h.keeper.GetWideDispute(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	if wd.Status != WideStatusArbReady {
		t.Fatalf("wide dispute status %s, want wide_arb_ready (K=8 -> single step)", wd.Status)
	}

	// 16 operand evidence chunks (8 A rows, 8 W rows)
	inputsLive := map[canonical.TensorRef]canonical.Hash{}
	for i, in := range fx.graph.Inputs {
		var hh canonical.Hash
		copy(hh[:], in.Root)
		inputsLive[canonical.TensorRef{Kind: 0, Index: uint32(i)}] = hh
	}
	evidence := make([]*types.GraphChunkEvidence, 0, 16)
	mkEvidence := func(ref canonical.TensorRef, tensor *canonical.TensorV2, root canonical.Hash,
		chunkIndex uint32) *types.GraphChunkEvidence {
		descJSON, _ := json.Marshal(tensor.Desc)
		chunk, proof, err := tensor.ChunkProofV2(int(chunkIndex))
		if err != nil {
			t.Fatal(err)
		}
		_, _, _, stateSiblings, err := canonical.StateProofV2(inputsLive, ref)
		if err != nil {
			t.Fatal(err)
		}
		return &types.GraphChunkEvidence{
			DescJson: descJSON, RefKind: uint32(ref.Kind), RefIndex: ref.Index, Root: root[:],
			ChunkIndex: chunkIndex, Count: uint32(tensor.ChunkCount()),
			Chunk: chunk, Proof: siblingsOfV2(proof), StateProof: siblingsOfV2(stateSiblings),
		}
	}
	aRoot := mustRootV2Chain(t, fx.inputs[0])
	wRoot := mustRootV2Chain(t, fx.inputs[1])
	for i := 0; i < 2; i++ { // rows 0 and 1 of A (M=2)
		evidence = append(evidence, mkEvidence(canonical.TensorRef{Kind: 0, Index: 0},
			fx.inputs[0], aRoot, uint32((i*8)/canonical.ChunkElems)))
	}
	for d := 0; d < 8; d++ { // rows 0..7 of W (K=8, non-transposed)
		evidence = append(evidence, mkEvidence(canonical.TensorRef{Kind: 0, Index: 1},
			fx.inputs[1], wRoot, uint32((d*8)/canonical.ChunkElems)))
	}
	statesW := []canonical.WideTileState{{}, fraudS1}
	workerHighProof, _ := canonical.WideTraceProofV1(0, 0, statesW, 1)
	statesC := []canonical.WideTileState{{}, honestS1}
	challengerHighProof, _ := canonical.WideTraceProofV1(0, 0, statesC, 1)
	res, err := h.msg.ArbitrateWide512(h.ctx, &types.MsgArbitrateWide512{
		Actor: h.challenger, GraphTaskId: taskID,
		WorkerNextState:      fraudS1.Bytes(),
		WorkerNextProof:      siblingsOfV2(workerHighProof),
		ChallengerNextState:  honestS1.Bytes(),
		ChallengerNextProof:  siblingsOfV2(challengerHighProof),
		Evidence:             evidence,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Outcome != "challenger_wins" {
		t.Fatalf("outcome %q, want challenger_wins", res.Outcome)
	}
	task, err := h.keeper.GetGraphTask(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	if task.Status != GraphStatusFraud {
		t.Fatalf("task status %s, want fraud (zero VWR)", task.Status)
	}
	if len(task.ReceiptID) != 0 {
		t.Fatal("fraud task produced a receipt")
	}
}
