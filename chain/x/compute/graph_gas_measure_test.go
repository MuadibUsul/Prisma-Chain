package compute

// F.5C A5-06/A5-07: metered V2 flows - per-transaction gas on the frozen
// bounded schedule plus the authoritative 512-MAC arbitration witness
// measurements (bytes, proof depth, gas).

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	storetypes "cosmossdk.io/store/types"

	"prismachain/chain/x/compute/types"
	"prismachain/compute/canonical"
)

type graphGasRecord struct {
	Step  string `json:"step"`
	Gas   uint64 `json:"gas"`
	Bytes int    `json:"witness_bytes,omitempty"`
}

func TestGraphV2GasMeasure(t *testing.T) {
	meter := storetypes.NewGasMeter(1 << 62)
	h, usedGas := meteredHarness(t, meter)
	h.bondWorker(h.challenger, h.chalKeys)
	var records []graphGasRecord
	step := func(name string, fn func()) {
		before := usedGas()
		fn()
		records = append(records, graphGasRecord{Step: name, Gas: usedGas() - before})
	}

	fx := buildChainGraphV2WideFixture(t)
	var taskID uint64
	step("post_graph_v2", func() { taskID, _ = h.postGraphV2(t, fx) })
	var assignmentRef []byte
	step("accept", func() { assignmentRef = h.acceptGraph(t, taskID) })
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
	step("submit_result_v3", func() {
		if _, err := h.msg.SubmitGraphResultV2(h.ctx, &types.MsgSubmitGraphResultV2{
			Worker: h.worker, GraphTaskId: taskID,
			NodeOutputManifestRoot: rc.NodeOutputManifestRootV2,
			OutputRoots:            rc.OutputRoots, FinalOutputRoot: rc.FinalOutputRoot,
			CompletedEpoch: rc.CompletedEpoch, WorkerSignature: rc.Signature,
		}); err != nil {
			t.Fatal(err)
		}
	})
	prov1, prov2 := newGemmKeys(45), newGemmKeys(46)
	h.bondWorker(h.monitorA, prov1)
	h.bondWorker(h.monitorB, prov2)
	if _, err := h.msg.RegisterDAProvider(h.ctx, &types.MsgRegisterDAProvider{Provider: h.monitorA}); err != nil {
		t.Fatal(err)
	}
	if _, err := h.msg.RegisterDAProvider(h.ctx, &types.MsgRegisterDAProvider{Provider: h.monitorB}); err != nil {
		t.Fatal(err)
	}
	step("da_attestation_1", func() {
		if err := h.submitGraphDA(taskID, h.monitorA, h.graphDARaw(t, taskID, h.monitorA, prov1, nil)); err != nil {
			t.Fatal(err)
		}
	})
	step("da_attestation_2", func() {
		if err := h.submitGraphDA(taskID, h.monitorB, h.graphDARaw(t, taskID, h.monitorB, prov2, nil)); err != nil {
			t.Fatal(err)
		}
	})
	task, err := h.keeper.GetGraphTask(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	h.advance(task.ChallengeEnd + 1)
	step("finalize_with_quorum", func() {
		if _, err := h.msg.FinalizeGraphTask(h.ctx, &types.MsgFinalizeGraphTask{
			Actor: h.requester, GraphTaskId: taskID}); err != nil {
			t.Fatal(err)
		}
	})

	// 512-MAC arbitration throughput: honest task 2 with an injected fraud
	// at the GEMM node, driven through graph bisection -> wide dispute.
	fx2 := buildChainGraphV2WideFixture(t)
	var task2 uint64
	step("post_graph_v2_fraud_case", func() { task2, _ = h.postGraphV2(t, fx2) })
	var assign2 []byte
	step("accept_fraud_case", func() { assign2 = h.acceptGraph(t, task2) })
	fraud0Data := append([]int64(nil), fx2.exec.Tensors[canonical.TensorRef{Kind: 1, Index: 0}].Data...)
	fraud0Data[0]++
	fraud0, err := canonical.NewTensorV2(fx2.graph.Nodes[0].Output, fraud0Data)
	if err != nil {
		t.Fatal(err)
	}
	fraud1Data := append([]int64(nil), fx2.exec.Tensors[canonical.TensorRef{Kind: 1, Index: 1}].Data...)
	fraud1Data[0]++
	fraud1, err := canonical.NewTensorV2(fx2.graph.Nodes[1].Output, fraud1Data)
	if err != nil {
		t.Fatal(err)
	}
	fraudTensors := map[canonical.TensorRef]*canonical.TensorV2{}
	for k, v := range fx2.exec.Tensors {
		fraudTensors[k] = v
	}
	fraudTensors[canonical.TensorRef{Kind: 1, Index: 0}] = fraud0
	fraudTensors[canonical.TensorRef{Kind: 1, Index: 1}] = fraud1
	fraudRoot, _ := fraud1.TensorRootV2()
	fraudExec := &canonical.GraphExecutionV2{Tensors: fraudTensors,
		Outputs: []canonical.Hash{fraudRoot}}
	var taskRef2 [8]byte
	binary.BigEndian.PutUint64(taskRef2[:], task2)
	rc2, err := canonical.NewGraphResultCommitV3(fx2.graph, taskRef2[:], assign2,
		h.workKeys.networkPub, fraudExec, 42)
	if err != nil {
		t.Fatal(err)
	}
	if err := canonical.SignGraphResultCommitV3(rc2, h.workKeys.networkPriv); err != nil {
		t.Fatal(err)
	}
	step("submit_fraud_v3", func() {
		if _, err := h.msg.SubmitGraphResultV2(h.ctx, &types.MsgSubmitGraphResultV2{
			Worker: h.worker, GraphTaskId: task2,
			NodeOutputManifestRoot: rc2.NodeOutputManifestRootV2,
			OutputRoots:            rc2.OutputRoots, FinalOutputRoot: rc2.FinalOutputRoot,
			CompletedEpoch: rc2.CompletedEpoch, WorkerSignature: rc2.Signature,
		}); err != nil {
			t.Fatal(err)
		}
	})
	step("open_graph_challenge", func() {
		if _, err := h.msg.OpenGraphChallenge(h.ctx, &types.MsgOpenGraphChallenge{
			Challenger: h.challenger, GraphTaskId: task2,
			ChallengerOutputRoots: [][]byte{fx2.exec.Outputs[0][:]},
			ChallengeBond:         MinBond,
		}); err != nil {
			t.Fatal(err)
		}
	})
	graphID, _ := fx2.graph.GraphIDV2()
	inputsOnly2 := map[canonical.TensorRef]canonical.Hash{}
	for i, in := range fx2.graph.Inputs {
		var hh canonical.Hash
		copy(hh[:], in.Root)
		inputsOnly2[canonical.TensorRef{Kind: 0, Index: uint32(i)}] = hh
	}
	afterNode0 := map[canonical.TensorRef]canonical.Hash{}
	for k, v := range inputsOnly2 {
		afterNode0[k] = v
	}
	root0, _ := fraud0.TensorRootV2()
	afterNode0[canonical.TensorRef{Kind: 1, Index: 0}] = root0
	fraudStateRoot, _ := canonical.GraphStateRootV2FromRoots(afterNode0)
	finalLive := map[canonical.TensorRef]canonical.Hash{}
	for k, v := range afterNode0 {
		finalLive[k] = v
	}
	finalLive[canonical.TensorRef{Kind: 1, Index: 1}] = fraudRoot
	fraudFinalStateRoot, _ := canonical.GraphStateRootV2FromRoots(finalLive)
	honestTrail := fx2.exec.Trail
	fraudTrail := []canonical.Hash{honestTrail[0], fraudStateRoot, fraudFinalStateRoot}
	lockTrail := func(party string, trail []canonical.Hash) {
		pi, _ := canonical.TrailProofV2(graphID, trail, 0)
		pf, _ := canonical.TrailProofV2(graphID, trail, uint32(len(trail)-1))
		root, _ := canonical.TrailRootV2(graphID, trail)
		if _, err := h.msg.GraphTrailClaim(h.ctx, &types.MsgGraphTrailClaim{
			Party: party, GraphTaskId: task2,
			TrailRoot: root[:], InitialRoot: trail[0][:], InitialProof: siblingsOfV2(pi),
			FinalRoot: trail[len(trail)-1][:], FinalProof: siblingsOfV2(pf),
		}); err != nil {
			t.Fatal(err)
		}
	}
	step("graph_trail_claims", func() {
		lockTrail(h.worker, fraudTrail)
		lockTrail(h.challenger, honestTrail)
	})
	step("graph_midpoint_round", func() {
		record, err := h.keeper.GetGraphDispute(h.ctx, task2)
		if err != nil {
			t.Fatal(err)
		}
		dispute, err := canonical.RestoreGraphDisputeV2(record.Snapshot, fx2.graph, ChallengeRoundBlocks)
		if err != nil {
			t.Fatal(err)
		}
		low, high := dispute.Interval()
		mid := low + (high-low)/2
		submitMid := func(party string, trail []canonical.Hash) {
			proof, _ := canonical.TrailProofV2(graphID, trail, mid)
			h.advance(h.currentHeight() + 1)
			if _, err := h.msg.GraphMidPoint(h.ctx, &types.MsgGraphMidPoint{
				Party: party, GraphTaskId: task2,
				StateRoot: trail[mid][:], ProofSiblings: siblingsOfV2(proof), Epoch: h.currentHeight(),
			}); err != nil {
				t.Fatal(err)
			}
		}
		submitMid(h.worker, fraudTrail)
		submitMid(h.challenger, honestTrail)
	})
	step("open_wide_dispute", func() {
		if _, err := h.msg.OpenWideGEMMDispute(h.ctx, &types.MsgOpenWideGEMMDispute{
			Challenger: h.challenger, GraphTaskId: task2, NodeId: 0,
			TileI: 0, TileJ: 0, ChallengeBond: MinBond,
		}); err != nil {
			t.Fatal(err)
		}
	})
	// partial-state traces (K=8: single step -> arb_ready immediately)
	aTile, _ := canonical.WideATile(fx2.inputs[0].Data, 2, 8, 0, 0)
	wTile, _ := canonical.WideWTile(fx2.inputs[1].Data, 8, 8, 0, 0, false)
	honestS1 := canonical.WideTileStep(canonical.WideTileState{}, aTile, wTile)
	fraudS1 := honestS1
	fraudS1[0]++
	var wProofs, cProofs []canonical.Hash
	claimTrace := func(party string, s1 canonical.WideTileState) []canonical.Hash {
		states := []canonical.WideTileState{{}, s1}
		root, _ := canonical.WideTraceRootV1(0, 0, states)
		p0, _ := canonical.WideTraceProofV1(0, 0, states, 0)
		p1, _ := canonical.WideTraceProofV1(0, 0, states, 1)
		if _, err := h.msg.WideTraceClaim(h.ctx, &types.MsgWideTraceClaim{
			Party: party, GraphTaskId: task2,
			TraceRoot: root[:], InitialState: states[0].Bytes(), InitialProof: siblingsOfV2(p0),
			FinalState: states[1].Bytes(), FinalProof: siblingsOfV2(p1),
		}); err != nil {
			t.Fatal(err)
		}
		return p1
	}
	step("wide_trace_claims", func() {
		wProofs = claimTrace(h.worker, fraudS1)
		cProofs = claimTrace(h.challenger, honestS1)
	})
	inputsLive := map[canonical.TensorRef]canonical.Hash{}
	for i, in := range fx2.graph.Inputs {
		var hh canonical.Hash
		copy(hh[:], in.Root)
		inputsLive[canonical.TensorRef{Kind: 0, Index: uint32(i)}] = hh
	}
	mkEvidence := func(ref canonical.TensorRef, tensor *canonical.TensorV2,
		root canonical.Hash, chunkIndex uint32) *types.GraphChunkEvidence {
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
	aRoot, _ := fx2.inputs[0].TensorRootV2()
	wRoot, _ := fx2.inputs[1].TensorRootV2()
	var evidence []*types.GraphChunkEvidence
	for i := 0; i < 2; i++ {
		evidence = append(evidence, mkEvidence(canonical.TensorRef{Kind: 0, Index: 0},
			fx2.inputs[0], aRoot, uint32((i*8)/canonical.ChunkElems)))
	}
	for d := 0; d < 8; d++ {
		evidence = append(evidence, mkEvidence(canonical.TensorRef{Kind: 0, Index: 1},
			fx2.inputs[1], wRoot, uint32((d*8)/canonical.ChunkElems)))
	}
	msg512 := &types.MsgArbitrateWide512{
		Actor: h.challenger, GraphTaskId: task2,
		WorkerNextState:     fraudS1.Bytes(),
		WorkerNextProof:     siblingsOfV2(wProofs),
		ChallengerNextState: honestS1.Bytes(),
		ChallengerNextProof: siblingsOfV2(cProofs),
		Evidence:            evidence,
	}
	witnessBytes := 0
	if raw, err := json.Marshal(msg512); err == nil {
		witnessBytes = len(raw)
	}
	var outcome string
	step("arbitrate_wide_512", func() {
		res, err := h.msg.ArbitrateWide512(h.ctx, msg512)
		if err != nil {
			t.Fatal(err)
		}
		outcome = res.Outcome
	})
	if outcome != "challenger_wins" {
		t.Fatalf("outcome %q", outcome)
	}

	doc := map[string]any{
		"version": "f5c-gas-results/1.0.0",
		"schedule": map[string]any{
			"version":   GraphGasScheduleVersion,
			"tx_base":   GasGraphTxBase,
			"arb_unit":  GasGraphArbiterUnit,
			"evidence":  GasGraphEvidence,
			"store":     GasGraphStore,
			"hash_base": GasGraphHash,
		},
		"records": records,
		"wide512_witness": map[string]any{
			"mac": canonical.Wide512MAC,
			"witness_bytes": witnessBytes,
			"manifest_proof_depth_expected": 0,
			"proof_siblings_worker":         len(wProofs),
			"proof_siblings_challenger":     len(cProofs),
			"evidence_chunks":               len(evidence),
			"chunk_payload_bytes":           len(evidence[0].Chunk),
			"note": "the chain recomputes exactly 512 logical MAC in exact int64; the " +
				"witness is bounded state bytes plus typed chunk proofs",
		},
		"outcome": outcome,
	}
	out := filepath.Join("..", "..", "..", "docs", "phase-f5c-gas-results.json")
	body, err := json.MarshalIndent(doc, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if err := writeFileAtomic(out, append(body, '\n')); err != nil {
		t.Fatal(err)
	}
	t.Logf("gas records: %d entries", len(records))
}

func writeFileAtomic(path string, data []byte) error {
	return os.WriteFile(path, data, 0o644)
}
