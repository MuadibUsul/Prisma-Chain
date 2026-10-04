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
