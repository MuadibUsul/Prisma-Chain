package compute

// A4 tests: graph DA attestations, quorum gate, typed chunk challenge,
// objective timeout and the availability-failure refund.

import (
	"encoding/binary"
	"encoding/json"
	"testing"

	"prismachain/chain/x/compute/types"
	"prismachain/compute/canonical"
)

func (h *gemmHarness) graphDARaw(t *testing.T, taskID uint64, provider string,
	keys *gemmKeys, mutate func(*canonical.GraphDAAttestationV2)) []byte {
	t.Helper()
	task, err := h.keeper.GetGraphTask(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	att := canonical.GraphDAAttestationV2{
		ProtocolVersion:      canonical.ProtocolVersionGraphDA,
		GraphTaskID:          taskID,
		GraphID:              task.GraphID,
		ManifestRootV2:       task.NodeOutputManifestRoot,
		FinalOutputRoot:      task.FinalOutputRoot,
		ProviderAccount:      []byte(provider),
		ProviderPubKey:       keys.networkPub,
		BundleBytes:          4096,
		AvailableUntilHeight: uint64(h.ctx.BlockHeight()) + 200,
		AttestedHeight:       uint64(h.ctx.BlockHeight()),
	}
	if mutate != nil {
		mutate(&att)
	}
	if err := canonical.SignGraphDAAttestationV2(&att, keys.networkPriv); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(&att)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func (h *gemmHarness) submitGraphDA(taskID uint64, provider string, raw []byte) error {
	_, err := h.msg.SubmitGraphDAAttestation(h.ctx, &types.MsgSubmitGraphDAAttestation{
		Provider: provider, GraphTaskId: taskID, AttestationJson: raw,
	})
	return err
}

// submitV2HonestResult posts, accepts and submits the honest result for the
// given V2 fixture; returns taskID and the execution.
func (h *gemmHarness) submitV2HonestResult(t *testing.T, fx *chainGraphV2Fixture) (uint64, *canonical.GraphExecutionV2) {
	t.Helper()
	taskID, _ := h.postGraphV2(t, fx)
	assignmentRef := h.acceptGraph(t, taskID)
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
	return taskID, fx.exec
}

func TestGraphV2QuorumGate(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphV2Fixture(t)
	taskID, _ := h.submitV2HonestResult(t, fx)
	task, err := h.keeper.GetGraphTask(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	h.advance(task.ChallengeEnd + 1)
	// no attestations: finalize refused (quorum gate)
	if _, err := h.msg.FinalizeGraphTask(h.ctx, &types.MsgFinalizeGraphTask{
		Actor: h.requester, GraphTaskId: taskID}); err == nil {
		t.Fatal("finalize without DA quorum accepted")
	}
	// register two providers; one attestation still refuses
	prov1, prov2 := newGemmKeys(41), newGemmKeys(42)
	h.bondWorker(h.monitorA, prov1)
	h.bondWorker(h.monitorB, prov2)
	if _, err := h.msg.RegisterDAProvider(h.ctx, &types.MsgRegisterDAProvider{Provider: h.monitorA}); err != nil {
		t.Fatal(err)
	}
	if _, err := h.msg.RegisterDAProvider(h.ctx, &types.MsgRegisterDAProvider{Provider: h.monitorB}); err != nil {
		t.Fatal(err)
	}
	if err := h.submitGraphDA(taskID, h.monitorA, h.graphDARaw(t, taskID, h.monitorA, prov1, nil)); err != nil {
		t.Fatal(err)
	}
	if _, err := h.msg.FinalizeGraphTask(h.ctx, &types.MsgFinalizeGraphTask{
		Actor: h.requester, GraphTaskId: taskID}); err == nil {
		t.Fatal("finalize with a single attestation accepted (1-of-2)")
	}
	// tampered binding refused
	bad := h.graphDARaw(t, taskID, h.monitorB, prov2, func(a *canonical.GraphDAAttestationV2) {
		a.ManifestRootV2 = make([]byte, 32)
	})
	if err := h.submitGraphDA(taskID, h.monitorB, bad); err == nil {
		t.Fatal("mismatched manifest binding accepted")
	}
	// the worker cannot attest itself
	if err := h.submitGraphDA(taskID, h.worker, h.graphDARaw(t, taskID, h.worker, h.workKeys, nil)); err == nil {
		t.Fatal("worker self-attestation accepted")
	}
	// second valid provider: quorum reached, finalize succeeds with one VWR
	if err := h.submitGraphDA(taskID, h.monitorB, h.graphDARaw(t, taskID, h.monitorB, prov2, nil)); err != nil {
		t.Fatal(err)
	}
	res, err := h.msg.FinalizeGraphTask(h.ctx, &types.MsgFinalizeGraphTask{
		Actor: h.requester, GraphTaskId: taskID})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ReceiptId) != 32 {
		t.Fatal("finalize produced no receipt")
	}
}

func TestGraphV2DAChallengeAndTimeout(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphV2Fixture(t)
	taskID, _ := h.submitV2HonestResult(t, fx)
	prov, keys := h.monitorA, newGemmKeys(41)
	h.bondWorker(prov, keys)
	if _, err := h.msg.RegisterDAProvider(h.ctx, &types.MsgRegisterDAProvider{Provider: prov}); err != nil {
		t.Fatal(err)
	}
	if err := h.submitGraphDA(taskID, prov, h.graphDARaw(t, taskID, prov, keys, nil)); err != nil {
		t.Fatal(err)
	}
	resp, err := h.msg.OpenGraphDAChallenge(h.ctx, &types.MsgOpenGraphDAChallenge{
		Challenger: h.challenger, GraphTaskId: taskID, Provider: prov,
		NodeId: 0, ChunkIndex: 0, Nonce: []byte("n"), Bond: MinBond,
	})
	if err != nil {
		t.Fatal(err)
	}
	node0 := fx.exec.Tensors[canonical.TensorRef{Kind: 1, Index: 0}]
	nodeRoot, err := node0.TensorRootV2()
	if err != nil {
		t.Fatal(err)
	}
	nodeRoots := make([]canonical.Hash, len(fx.graph.Nodes))
	for i := range fx.graph.Nodes {
		root, err := fx.exec.Tensors[canonical.TensorRef{Kind: 1, Index: uint32(i)}].TensorRootV2()
		if err != nil {
			t.Fatal(err)
		}
		nodeRoots[i] = root
	}
	_, manifestProof, err := canonical.NodeOutputManifestProofV2(fx.graph, nodeRoots, 0)
	if err != nil {
		t.Fatal(err)
	}
	chunk, chunkProof, err := node0.ChunkProofV2(0)
	if err != nil {
		t.Fatal(err)
	}
	descJSON, _ := json.Marshal(node0.Desc)
	// a short (wrong) chunk first: refused
	if _, err := h.msg.RespondGraphDAChallenge(h.ctx, &types.MsgRespondGraphDAChallenge{
		Provider: prov, ChallengeId: resp.ChallengeId, NodeRoot: nodeRoot[:],
		NodeDescJson: descJSON, ManifestProof: siblingsOfV2(manifestProof),
		Chunk: append([]byte(nil), chunk[:len(chunk)-1]...), ChunkIndex: 0, ChunkCount: 1,
		ChunkProof: siblingsOfV2(chunkProof),
	}); err == nil {
		t.Fatal("short chunk accepted")
	}
	// honesty answers
	if _, err := h.msg.RespondGraphDAChallenge(h.ctx, &types.MsgRespondGraphDAChallenge{
		Provider: prov, ChallengeId: resp.ChallengeId, NodeRoot: nodeRoot[:],
		NodeDescJson: descJSON, ManifestProof: siblingsOfV2(manifestProof),
		Chunk: chunk, ChunkIndex: 0, ChunkCount: 1, ChunkProof: siblingsOfV2(chunkProof),
	}); err != nil {
		t.Fatal(err)
	}
	// second challenge: no response; timeout slashes the provider bond
	resp2, err := h.msg.OpenGraphDAChallenge(h.ctx, &types.MsgOpenGraphDAChallenge{
		Challenger: h.challenger, GraphTaskId: taskID, Provider: prov,
		NodeId: 0, ChunkIndex: 0, Nonce: []byte("n2"), Bond: MinBond,
	})
	if err != nil {
		t.Fatal(err)
	}
	bondBefore := h.keeper.GetBond(h.ctx, prov)
	h.advance(resp2.Deadline + 1)
	if _, err := h.msg.TimeoutGraphDAChallenge(h.ctx, &types.MsgTimeoutGraphDAChallenge{
		Actor: h.requester, ChallengeId: resp2.ChallengeId}); err != nil {
		t.Fatal(err)
	}
	if h.keeper.GetBond(h.ctx, prov) >= bondBefore {
		t.Fatal("provider bond was not slashed on timeout")
	}
}

func TestGraphV2AvailabilityFailure(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphV2Fixture(t)
	taskID, _ := h.submitV2HonestResult(t, fx)
	task, err := h.keeper.GetGraphTask(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	requiredUntil := task.ResultSubmittedHeight + task.ChallengeWindow + DAWindowBlocks
	if _, err := h.msg.FailGraphAvailability(h.ctx, &types.MsgFailGraphAvailability{
		Actor: h.requester, GraphTaskId: taskID}); err == nil {
		t.Fatal("availability failure before the DA window closed")
	}
	h.advance(requiredUntil + 1)
	if _, err := h.msg.FailGraphAvailability(h.ctx, &types.MsgFailGraphAvailability{
		Actor: h.requester, GraphTaskId: taskID}); err != nil {
		t.Fatal(err)
	}
	final, err := h.keeper.GetGraphTask(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	if final.Status != GraphStatusAvailabilityFailed {
		t.Fatalf("status %s, want availability_failed", final.Status)
	}
	if len(final.ReceiptID) != 0 {
		t.Fatal("availability failure produced a receipt")
	}
}
