package compute

// CANONICAL_GRAPH_V2 on-chain honest-path tests: the same settlement
// machinery as V1 (escrow, bond, challenge window, receipt) driven by a
// typed wide-integer descriptor and a GraphResultCommitV3 signature.

import (
	"bytes"
	"crypto/ed25519"
	"encoding/binary"
	"encoding/json"
	"testing"

	"prismachain/chain/x/compute/types"
	"prismachain/compute/canonical"
)

type chainGraphV2Fixture struct {
	graph     *canonical.GraphDescriptorV2
	graphJSON []byte
	inputs    map[uint32]*canonical.TensorV2
	exec      *canonical.GraphExecutionV2
}

func buildChainGraphV2Fixture(t *testing.T) *chainGraphV2Fixture {
	t.Helper()
	mk := func(desc canonical.TensorDescriptorV2, data []int64) *canonical.TensorV2 {
		tensor, err := canonical.NewTensorV2(desc, data)
		if err != nil {
			t.Fatal(err)
		}
		return tensor
	}
	a := mk(canonical.NewDescV2(canonical.DtypeV2A13, 2, 4), []int64{1, -2, 3, -4, 4095, -4096, 7, 8})
	w := mk(canonical.NewDescV2(canonical.DtypeV2W10, 4, 3), []int64{1, -1, 2, 0, -3, 511, 2, 2, -2, -1, 0, 1})
	aroot, _ := a.TensorRootV2()
	wroot, _ := w.TensorRootV2()
	g := &canonical.GraphDescriptorV2{
		ProtocolVersion: canonical.ProtocolVersionGraphV2,
		Spec:            "TEST_WIDE_CHAIN_V1",
		Arithmetic:      canonical.A13W10I64Profile(),
		Inputs: []canonical.GraphInputV2{
			{Name: "a", Desc: a.Desc, Root: aroot[:]},
			{Name: "w", Desc: w.Desc, Root: wroot[:]},
		},
		Nodes: []canonical.GraphNodeV2{
			{NodeID: 0, OperatorID: canonical.OpGEMMWideA13W10,
				Version: canonical.OpGEMMVersionWide,
				Inputs:  []canonical.TensorRef{{Kind: 0, Index: 0}, {Kind: 0, Index: 1}},
				Output:  canonical.NewDescV2(canonical.DtypeV2Int64Accum, 2, 3),
				Params:  canonical.ParamList{}},
			{NodeID: 1, OperatorID: canonical.OpRequantizeWideV1,
				Version: canonical.OpRequantVersionWide,
				Inputs:  []canonical.TensorRef{{Kind: 1, Index: 0}},
				Output:  canonical.NewDescV2(canonical.DtypeV2A13, 2, 3),
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
	return &chainGraphV2Fixture{graph: g, graphJSON: raw, inputs: map[uint32]*canonical.TensorV2{0: a, 1: w}, exec: exec}
}

func (h *gemmHarness) postGraphV2(t *testing.T, fx *chainGraphV2Fixture) (uint64, []byte) {
	t.Helper()
	nonce := []byte("graph-v2-requester-nonce")
	binding, err := canonicalJSON(map[string]any{
		"chain_id": h.ctx.ChainID(), "requester": h.requester,
		"requester_protocol_pubkey": hexStr(h.reqKeys.protocolPub),
		"requester_nonce":           hexStr(nonce),
	})
	if err != nil {
		t.Fatal(err)
	}
	proof := ed25519.Sign(h.reqKeys.protocolPriv, append([]byte(GraphRequesterKeyBindingDomain), binding...))
	res, err := h.msg.PostGraphTask(h.ctx, &types.MsgPostGraphTask{
		Requester: h.requester, RequesterProtocolPubkey: h.reqKeys.protocolPub,
		RequesterKeyProof: proof, RequesterNonce: nonce,
		GraphJson: fx.graphJSON, InputDataRef: "dev://graph-v2",
		ChallengeWindow: ChallengeBlocks, MaxPricePerCwu: 1000, MaxFee: 2 * MinBond,
	})
	if err != nil {
		t.Fatal(err)
	}
	return res.GraphTaskId, res.GraphId
}

func TestGraphV2HonestFlow(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphV2Fixture(t)
	taskID, postedID := h.postGraphV2(t, fx)
	graphID, err := fx.graph.GraphIDV2()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(postedID, graphID[:]) {
		t.Fatal("posted V2 graph id mismatch")
	}
	task, err := h.keeper.GetGraphTask(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	if task.ProtocolVersion != canonical.ProtocolVersionGraphV2 {
		t.Fatalf("task protocol version = %q", task.ProtocolVersion)
	}
	assignmentRef := h.acceptGraph(t, taskID)

	// Build and sign the V3 commitment (worker protocol key is the bonded
	// network key, same as V1/V2 submissions).
	var taskRef [8]byte
	binary.BigEndian.PutUint64(taskRef[:], taskID)
	rc, err := canonical.NewGraphResultCommitV3(fx.graph, taskRef[:], assignmentRef,
		h.workKeys.networkPub, fx.exec, 42)
	if err != nil {
		t.Fatal(err)
	}
	if err := canonical.SignGraphResultCommitV3(rc, h.workKeys.networkPriv); err != nil {
		t.Fatal(err)
	}

	// Tampering with the manifest root must be refused.
	badRoot := append([]byte(nil), rc.NodeOutputManifestRootV2...)
	badRoot[0] ^= 1
	if _, err := h.msg.SubmitGraphResultV2(h.ctx, &types.MsgSubmitGraphResultV2{
		Worker: h.worker, GraphTaskId: taskID, NodeOutputManifestRoot: badRoot,
		OutputRoots: rc.OutputRoots, FinalOutputRoot: rc.FinalOutputRoot,
		CompletedEpoch: rc.CompletedEpoch, WorkerSignature: rc.Signature,
	}); err == nil {
		t.Fatal("tampered V3 manifest root accepted")
	}

	if _, err := h.msg.SubmitGraphResultV2(h.ctx, &types.MsgSubmitGraphResultV2{
		Worker: h.worker, GraphTaskId: taskID,
		NodeOutputManifestRoot: rc.NodeOutputManifestRootV2,
		OutputRoots:            rc.OutputRoots, FinalOutputRoot: rc.FinalOutputRoot,
		CompletedEpoch: rc.CompletedEpoch, WorkerSignature: rc.Signature,
	}); err != nil {
		t.Fatal(err)
	}
	task, err = h.keeper.GetGraphTask(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	if task.CommitVersion != "3" {
		t.Fatalf("commit version = %q, want 3", task.CommitVersion)
	}
	h.advance(task.ChallengeEnd + 1)
	res, err := h.msg.FinalizeGraphTask(h.ctx, &types.MsgFinalizeGraphTask{Actor: h.requester, GraphTaskId: taskID})
	if err != nil {
		t.Fatal(err)
	}
	receiptRaw := h.keeper.store(h.ctx).Get(graphReceiptKey(res.ReceiptId))
	if len(receiptRaw) == 0 {
		t.Fatal("no V3 receipt stored")
	}
	var receipt canonical.VerifiedGraphWorkReceiptV3
	if err := json.Unmarshal(receiptRaw, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.ProtocolVersion != "3.0.0" || receipt.ArithmeticID != canonical.ArithmeticProfileA13W10 {
		t.Fatalf("receipt identity wrong: %+v", receipt)
	}
	if receipt.WorkVector.Get("GEMM_A13W10_MAC", 0) != 2*4*3 {
		t.Fatalf("receipt work vector wrong: %v", receipt.WorkVector)
	}
	task, err = h.keeper.GetGraphTask(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	if task.Status != GraphStatusFinalized {
		t.Fatalf("status = %s", task.Status)
	}
	// double finalization refused (exactly one receipt)
	if _, err := h.msg.FinalizeGraphTask(h.ctx, &types.MsgFinalizeGraphTask{Actor: h.requester, GraphTaskId: taskID}); err == nil {
		t.Fatal("double finalization accepted")
	}
}

func TestGraphV2RejectsV1SignatureOnV3(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphV2Fixture(t)
	taskID, _ := h.postGraphV2(t, fx)
	assignmentRef := h.acceptGraph(t, taskID)
	var taskRef [8]byte
	binary.BigEndian.PutUint64(taskRef[:], taskID)
	rc, err := canonical.NewGraphResultCommitV3(fx.graph, taskRef[:], assignmentRef,
		h.workKeys.networkPub, fx.exec, 42)
	if err != nil {
		t.Fatal(err)
	}
	// Sign the V3 bytes with the V2 signing domain: must be rejected.
	unsigned := *rc
	unsigned.Signature = nil
	enc, err := canonical.EncodeCanonical(&unsigned)
	if err != nil {
		t.Fatal(err)
	}
	wrongSig := ed25519.Sign(h.workKeys.networkPriv, append([]byte(canonical.DomainReceiptV2), enc...))
	if _, err := h.msg.SubmitGraphResultV2(h.ctx, &types.MsgSubmitGraphResultV2{
		Worker: h.worker, GraphTaskId: taskID,
		NodeOutputManifestRoot: rc.NodeOutputManifestRootV2,
		OutputRoots:            rc.OutputRoots, FinalOutputRoot: rc.FinalOutputRoot,
		CompletedEpoch: rc.CompletedEpoch, WorkerSignature: wrongSig,
	}); err == nil {
		t.Fatal("wrong-domain signature accepted for V3")
	}
}

// TestGraphV2DisputeBisection drives the full V2 challenge path through
// the message server: challenge open (V2 dispatch), both trail claims
// (V2 trail domain), bisection midpoints until arb_ready, and the
// canonical layer localizes the injected first-divergent node.
func TestGraphV2DisputeBisection(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphV2Fixture(t)
	taskID, _ := h.postGraphV2(t, fx)
	assignmentRef := h.acceptGraph(t, taskID)

	// honest V3 submission
	var taskRef [8]byte
	binary.BigEndian.PutUint64(taskRef[:], taskID)
	rc, err := canonical.NewGraphResultCommitV3(fx.graph, taskRef[:], assignmentRef,
		h.workKeys.networkPub, fx.exec, 42)
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

	// challenge with a different final root (V2 output count dispatch)
	h.bondWorker(h.challenger, h.chalKeys)
	fakeFinal := make([]byte, 32)
	for i := range fakeFinal {
		fakeFinal[i] = 0xAA
	}
	if _, err := h.msg.OpenGraphChallenge(h.ctx, &types.MsgOpenGraphChallenge{
		Challenger: h.challenger, GraphTaskId: taskID,
		ChallengerOutputRoots: [][]byte{fakeFinal}, ChallengeBond: MinBond,
	}); err != nil {
		t.Fatal(err)
	}

	// both parties lock V2 trails; the challenger's trail diverges from
	// node 0 onward while sharing the committed initial state
	graphID, err := fx.graph.GraphIDV2()
	if err != nil {
		t.Fatal(err)
	}
	honest := fx.exec.Trail
	bad := append([]canonical.Hash(nil), honest...)
	bad[1] = canonical.Hash{0xF1}
	bad[2] = canonical.Hash{0xF2}
	lock := func(party string, trail []canonical.Hash) {
		pi, err := canonical.TrailProofV2(graphID, trail, 0)
		if err != nil {
			t.Fatal(err)
		}
		pf, err := canonical.TrailProofV2(graphID, trail, uint32(len(trail)-1))
		if err != nil {
			t.Fatal(err)
		}
		root, err := canonical.TrailRootV2(graphID, trail)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := h.msg.GraphTrailClaim(h.ctx, &types.MsgGraphTrailClaim{
			Party: party, GraphTaskId: taskID,
			TrailRoot: root[:], InitialRoot: trail[0][:], InitialProof: siblingsOfV2(pi),
			FinalRoot: trail[len(trail)-1][:], FinalProof: siblingsOfV2(pf),
		}); err != nil {
			t.Fatal(err)
		}
	}
	// a claim whose initial root is not the committed input state is
	// refused before anything is stored
	if _, err := h.msg.GraphTrailClaim(h.ctx, &types.MsgGraphTrailClaim{
		Party: h.worker, GraphTaskId: taskID,
		TrailRoot: make([]byte, 32), InitialRoot: make([]byte, 32),
		InitialProof: [][]byte{}, FinalRoot: make([]byte, 32), FinalProof: [][]byte{},
	}); err == nil {
		t.Fatal("wrong initial state accepted")
	}
	lock(h.worker, honest)
	lock(h.challenger, bad)

	// bisection
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
			proof, err := canonical.TrailProofV2(graphID, trail, mid)
			if err != nil {
				t.Fatal(err)
			}
			h.advance(h.currentHeight() + 1)
			if _, err := h.msg.GraphMidPoint(h.ctx, &types.MsgGraphMidPoint{
				Party: party, GraphTaskId: taskID,
				StateRoot: trail[mid][:], ProofSiblings: siblingsOfV2(proof), Epoch: h.currentHeight(),
			}); err != nil {
				t.Fatal(err)
			}
		}
		submitMid(h.worker, honest)
		submitMid(h.challenger, bad)
	}
	record, err := h.keeper.GetGraphDispute(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != GraphDisputeArbReady {
		t.Fatalf("dispute status %s, want arb_ready", record.Status)
	}
	restored, err := canonical.RestoreGraphDisputeV2(record.Snapshot, fx.graph, ChallengeRoundBlocks)
	if err != nil {
		t.Fatal(err)
	}
	node, err := restored.FirstDivergentNode()
	if err != nil {
		t.Fatal(err)
	}
	if node != 0 {
		t.Fatalf("first divergent node %d, want 0 (the injected GEMM node)", node)
	}
}

func siblingsOfV2(sibs []canonical.Hash) [][]byte {
	out := make([][]byte, len(sibs))
	for i, s := range sibs {
		out[i] = append([]byte(nil), s[:]...)
	}
	return out
}

// TestGraphV2RequantArbitration drives the complete V2 fraud settlement:
// a self-consistent fraudulent result (one requant output element altered,
// final output recomputed downstream), the honest challenger's trail, the
// bisection to node 1 (the requant), and the typed 512-element-window
// arbitration -> ChallengerWins -> fraud status -> zero VWR.
func TestGraphV2RequantArbitration(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphV2Fixture(t)
	taskID, _ := h.postGraphV2(t, fx)
	assignmentRef := h.acceptGraph(t, taskID)

	node1 := fx.graph.Nodes[1]
	honestNode1 := fx.exec.Tensors[canonical.TensorRef{Kind: 1, Index: 1}]
	honestRoot1, _ := honestNode1.TensorRootV2()
	honestChunk, honestProof, err := honestNode1.ChunkProofV2(0)
	if err != nil {
		t.Fatal(err)
	}
	// fraudulent output: one element changed in range; the final output is
	// recomputed from it, so every commitment is self-consistent
	fraudData := append([]int64(nil), honestNode1.Data...)
	fraudData[0]--
	fraudTensor, err := canonical.NewTensorV2(node1.Output, fraudData)
	if err != nil {
		t.Fatal(err)
	}
	fraudRoot1, _ := fraudTensor.TensorRootV2()
	fraudChunk, fraudProof, _ := fraudTensor.ChunkProofV2(0)
	fraudTensors := map[canonical.TensorRef]*canonical.TensorV2{}
	for k, v := range fx.exec.Tensors {
		fraudTensors[k] = v
	}
	fraudTensors[canonical.TensorRef{Kind: 1, Index: 1}] = fraudTensor
	fraudRootFinal, err := fraudTensor.TensorRootV2()
	if err != nil {
		t.Fatal(err)
	}
	fraudExec := &canonical.GraphExecutionV2{Tensors: fraudTensors,
		Outputs: []canonical.Hash{fraudRootFinal}}

	// worker submits the fraudulent V3 result
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
	// the honest challenger counter-claims the honest final root
	h.bondWorker(h.challenger, h.chalKeys)
	if _, err := h.msg.OpenGraphChallenge(h.ctx, &types.MsgOpenGraphChallenge{
		Challenger: h.challenger, GraphTaskId: taskID,
		ChallengerOutputRoots: [][]byte{fx.exec.Outputs[0][:]},
		ChallengeBond:         MinBond,
	}); err != nil {
		t.Fatal(err)
	}

	// trails: shared through node 0, divergent from node 1
	graphID, _ := fx.graph.GraphIDV2()
	honestTrail := fx.exec.Trail
	liveFraud := map[canonical.TensorRef]canonical.Hash{}
	for ref, tensor := range fraudTensors {
		root, err := tensor.TensorRootV2()
		if err != nil {
			t.Fatal(err)
		}
		liveFraud[ref] = root
	}
	fraudStateRoot, err := canonical.GraphStateRootV2FromRoots(liveFraud)
	if err != nil {
		t.Fatal(err)
	}
	fraudTrail := []canonical.Hash{honestTrail[0], honestTrail[1], fraudStateRoot}
	lock := func(party string, trail []canonical.Hash) {
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
	lock(h.worker, fraudTrail)
	lock(h.challenger, honestTrail)

	// bisection to the requant node
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
	if record.Status != GraphDisputeArbReady {
		t.Fatalf("dispute status %s, want arb_ready", record.Status)
	}

	// evidence: node 1's input (node 0's INT64_ACCUM output) with a state
	// membership proof against the agreed low state (after node 0)
	node0 := fx.exec.Tensors[canonical.TensorRef{Kind: 1, Index: 0}]
	node0Root, _ := node0.TensorRootV2()
	liveLow := map[canonical.TensorRef]canonical.Hash{}
	for i, in := range fx.graph.Inputs {
		var hh canonical.Hash
		copy(hh[:], in.Root)
		liveLow[canonical.TensorRef{Kind: 0, Index: uint32(i)}] = hh
	}
	liveLow[canonical.TensorRef{Kind: 1, Index: 0}] = node0Root
	_, _, _, siblings, err := canonical.StateProofV2(liveLow, canonical.TensorRef{Kind: 1, Index: 0})
	if err != nil {
		t.Fatal(err)
	}
	inChunk, inProof, err := node0.ChunkProofV2(0)
	if err != nil {
		t.Fatal(err)
	}
	descJSON, _ := json.Marshal(node0.Desc)
	evidence := []*types.GraphChunkEvidence{{
		DescJson: descJSON, RefKind: 1, RefIndex: 0, Root: node0Root[:],
		StateProof: siblingsOfV2(siblings), ChunkIndex: 0, Count: 1,
		Chunk: inChunk, Proof: siblingsOfV2(inProof),
	}}
	res, err := h.msg.ArbitrateGraphNode(h.ctx, &types.MsgArbitrateGraphNode{
		Actor: h.challenger, GraphTaskId: taskID,
		WorkerOutRoot:        fraudRoot1[:],
		WorkerChunkIndex:     0,
		WorkerChunk:          fraudChunk,
		WorkerChunkProof:     siblingsOfV2(fraudProof),
		ChallengerOutRoot:    honestRoot1[:],
		ChallengerChunkIndex: 0,
		ChallengerChunk:      honestChunk,
		ChallengerChunkProof: siblingsOfV2(honestProof),
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
