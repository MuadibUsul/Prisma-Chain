package compute

// CANONICAL_GRAPH_V1 settlement tests: honest flow, fraud dispute with
// bounded node arbitration, false challenge refusal, trail replay
// rejection, restart persistence and DoS bounds. Fixtures are produced by
// the real compute/canonical library — the chain never sees hand-written
// hashes.

import (
	"bytes"
	"crypto/ed25519"
	"encoding/binary"
	"encoding/json"
	"testing"

	"prismachain/chain/x/compute/types"
	"prismachain/compute/canonical"
)

// --- fixture ---------------------------------------------------------------

type chainLCG struct{ s uint64 }

func (r *chainLCG) next() uint64 {
	r.s = r.s*6364136223846793005 + 1442695040888963407
	return r.s
}

func (r *chainLCG) inRange(lo, hi int64) int64 {
	return lo + int64(r.next()>>33)%(hi-lo+1)
}

type chainGraphFixture struct {
	cfg       canonical.BlockConfig
	graph     canonical.GraphDescriptor
	graphJSON []byte
	inputs    map[uint32]canonical.Tensor
	tables    map[uint32]*canonical.RopeConstants
	exec      *canonical.GraphExecution
	weights   canonical.BlockInputSet
}

func chainMiniConfig() canonical.BlockConfig {
	return canonical.BlockConfig{Seq: 16, DModel: 128, Heads: 4, HeadDim: 32, MLPHidden: 256}
}

func chainQuant(t *testing.T, cfg canonical.BlockConfig) canonical.BlockQuant {
	t.Helper()
	scale, err := canonical.AttentionScaleFx(cfg.HeadDim)
	if err != nil {
		t.Fatal(err)
	}
	narrow := canonical.RequantSpec{Mult: 1, Shift: 15, Lo: -127, Hi: 127, OutInt8: true}
	fx := func(mult, shift int64) canonical.RequantSpec {
		return canonical.RequantSpec{Mult: mult, Shift: shift, Lo: canonical.MinFx, Hi: canonical.MaxFx}
	}
	return canonical.BlockQuant{
		NormEpsFx:       10,
		Norm1ToInt8:     narrow,
		Norm2ToInt8:     narrow,
		QKAccumToFx:     fx(1024, 0),
		QKRopeToInt8:    narrow,
		VAccumToInt8:    canonical.RequantSpec{Mult: 1, Shift: 5, Lo: -127, Hi: 127, OutInt8: true},
		ScoresAccumToFx: fx(1024*scale, 20),
		SoftmaxToInt8:   canonical.RequantSpec{Mult: 1, Shift: 15, Lo: 0, Hi: 127, OutInt8: true},
		CtxAccumToInt8:  canonical.RequantSpec{Mult: 1, Shift: 5, Lo: -127, Hi: 127, OutInt8: true},
		ProjAccumToFx:   fx(1024, 0),
		GateAccumToFx:   fx(1024, 0),
		UpAccumToFx:     fx(1024, 0),
		MlpMulToInt8:    narrow,
		DownAccumToFx:   fx(1024, 0),
	}
}

func buildChainGraphFixture(t *testing.T) *chainGraphFixture {
	t.Helper()
	cfg := chainMiniConfig()
	r := &chainLCG{s: 0xFEEDFACE12345}
	d, hd, mlp := int64(cfg.DModel), int64(cfg.HeadDim), int64(cfg.MLPHidden)
	randTensor := func(desc canonical.TensorDescriptor, lo, hi int64) canonical.Tensor {
		elems, _ := desc.Elems()
		data := make([]int32, elems)
		for i := range data {
			data[i] = int32(r.inRange(lo, hi))
		}
		return canonical.Tensor{Desc: desc, Data: data}
	}
	rootOf := func(tens canonical.Tensor) []byte {
		root, err := tens.TensorRoot()
		if err != nil {
			t.Fatal(err)
		}
		return root[:]
	}

	x := randTensor(canonical.NewDesc(canonical.DtypeQ12_20, int64(cfg.Seq), d), -(1 << 18), 1<<18)
	n1 := randTensor(canonical.NewDesc(canonical.DtypeQ12_20, d), 1<<20, 1<<20)
	n2 := randTensor(canonical.NewDesc(canonical.DtypeQ12_20, d), 1<<20, 1<<20)
	table := make([]int32, cfg.Seq*(cfg.HeadDim/2)*2)
	for i := range table {
		table[i] = int32(r.inRange(-(1 << 20), 1<<20))
	}
	constants, err := canonical.NewRopeConstants(table, cfg.HeadDim/2)
	if err != nil {
		t.Fatal(err)
	}
	tableTensor := canonical.Tensor{Desc: canonical.NewDesc(canonical.DtypeQ12_20, int64(len(table))), Data: table}

	inputs := map[uint32]canonical.Tensor{0: x, 1: n1, 2: n2, 3: tableTensor}
	var set canonical.BlockInputSet
	set.X = canonical.GraphInput{Desc: x.Desc, Root: rootOf(x)}
	set.Norm1 = canonical.GraphInput{Desc: n1.Desc, Root: rootOf(n1)}
	set.Norm2 = canonical.GraphInput{Desc: n2.Desc, Root: rootOf(n2)}
	set.RopeTable = canonical.GraphInput{Desc: tableTensor.Desc, Root: rootOf(tableTensor)}
	idx := uint32(4)
	addWeight := func(shape ...int64) canonical.GraphInput {
		w := randTensor(canonical.NewDesc(canonical.DtypeInt8, shape...), -16, 16)
		inputs[idx] = w
		idx++
		return canonical.GraphInput{Desc: w.Desc, Root: rootOf(w)}
	}
	for h := 0; h < cfg.Heads; h++ {
		set.WQ = append(set.WQ, addWeight(d, hd))
		set.WK = append(set.WK, addWeight(d, hd))
		set.WV = append(set.WV, addWeight(d, hd))
		set.WO = append(set.WO, addWeight(hd, d))
	}
	set.WGate = addWeight(d, mlp)
	set.WUp = addWeight(d, mlp)
	set.WDown = addWeight(mlp, d)

	graph, err := canonical.BuildTransformerBlockV1(cfg, set, chainQuant(t, cfg))
	if err != nil {
		t.Fatal(err)
	}
	graphJSON, err := json.Marshal(graph)
	if err != nil {
		t.Fatal(err)
	}
	tables := map[uint32]*canonical.RopeConstants{3: constants}
	exec, err := canonical.ExecuteGraph(graph, inputs, tables)
	if err != nil {
		t.Fatal(err)
	}
	return &chainGraphFixture{
		cfg: cfg, graph: *graph, graphJSON: graphJSON,
		inputs: inputs, tables: tables, exec: exec, weights: set,
	}
}

func (fx *chainGraphFixture) graphID(t *testing.T) canonical.Hash {
	t.Helper()
	id, err := fx.graph.GraphID()
	if err != nil {
		t.Fatal(err)
	}
	return id
}

// --- harness helpers -------------------------------------------------------

func (h *gemmHarness) postGraph(t *testing.T, fx *chainGraphFixture) (uint64, []byte) {
	t.Helper()
	nonce := []byte("graph-requester-nonce")
	binding, err := canonicalJSON(map[string]any{
		"chain_id": h.ctx.ChainID(), "requester": h.requester,
		"requester_protocol_pubkey": hexStr(h.reqKeys.protocolPub),
		"requester_nonce":           hexStr(nonce),
	})
	if err != nil {
		h.t.Fatal(err)
	}
	proof := ed25519.Sign(h.reqKeys.protocolPriv, append([]byte(GraphRequesterKeyBindingDomain), binding...))
	res, err := h.msg.PostGraphTask(h.ctx, &types.MsgPostGraphTask{
		Requester: h.requester, RequesterProtocolPubkey: h.reqKeys.protocolPub,
		RequesterKeyProof: proof, RequesterNonce: nonce,
		GraphJson: fx.graphJSON, InputDataRef: "dev://graph",
		ChallengeWindow: ChallengeBlocks, MaxPricePerCwu: 1000, MaxFee: 2 * MinBond,
	})
	if err != nil {
		h.t.Fatal(err)
	}
	return res.GraphTaskId, res.GraphId
}

func (h *gemmHarness) acceptGraph(t *testing.T, taskID uint64) []byte {
	t.Helper()
	h.bondWorker(h.worker, h.workKeys)
	res, err := h.msg.AcceptGraphTask(h.ctx, &types.MsgAcceptGraphTask{
		Worker: h.worker, GraphTaskId: taskID, AssignmentNonce: []byte("graph-assignment-nonce"),
	})
	if err != nil {
		h.t.Fatal(err)
	}
	return res.AssignmentRef
}

func (h *gemmHarness) submitGraphResult(t *testing.T, taskID uint64, fx *chainGraphFixture, assignmentRef []byte, outputs []canonical.Hash) {
	t.Helper()
	var taskRef [8]byte
	binary.BigEndian.PutUint64(taskRef[:], taskID)
	rc, err := canonical.NewGraphResultCommit(&fx.graph, taskRef[:], assignmentRef, h.workKeys.networkPub, outputs, 42)
	if err != nil {
		h.t.Fatal(err)
	}
	if err := canonical.SignGraphResultCommit(rc, h.workKeys.networkPriv); err != nil {
		h.t.Fatal(err)
	}
	if _, err := h.msg.SubmitGraphResult(h.ctx, &types.MsgSubmitGraphResult{
		Worker: h.worker, GraphTaskId: taskID,
		OutputRoots: rc.OutputRoots, FinalOutputRoot: rc.FinalOutputRoot,
		CompletedEpoch: rc.CompletedEpoch, WorkerSignature: rc.Signature,
	}); err != nil {
		h.t.Fatal(err)
	}
}

func (h *gemmHarness) advance(height uint64) {
	h.ctx = h.ctx.WithBlockHeight(int64(height))
}

func (h *gemmHarness) currentHeight() uint64 {
	return uint64(h.ctx.BlockHeight())
}

func hashesOf(t *testing.T, outputs []canonical.Hash) [][]byte {
	t.Helper()
	out := make([][]byte, len(outputs))
	for i, o := range outputs {
		out[i] = append([]byte(nil), o[:]...)
	}
	return out
}

func assertReserved(t *testing.T, h *gemmHarness, worker string, want uint64) {
	t.Helper()
	if got := h.keeper.getReserved(h.ctx, worker); got != want {
		t.Fatalf("reserved for %s = %d, want %d", worker, got, want)
	}
}

// runGraphDispute drives claims + bisection + arbitration for a fraud at
// the given node and returns the arbitration outcome.
func (h *gemmHarness) runGraphDispute(t *testing.T, taskID uint64, fx *chainGraphFixture, corruptionNode uint32) string {
	t.Helper()
	graphID := fx.graphID(t)
	honest := fx.exec
	workerTrail, err := canonical.FraudTrail(&fx.graph, honest, corruptionNode, func(tens *canonical.Tensor) {
		tens.Data[0]++
	})
	if err != nil {
		h.t.Fatal(err)
	}

	// Both parties lock their trails.
	claim := func(party string, trail []canonical.Hash) {
		pi, err := canonical.TrailProof(graphID, trail, 0)
		if err != nil {
			h.t.Fatal(err)
		}
		pf, err := canonical.TrailProof(graphID, trail, uint32(len(trail)-1))
		if err != nil {
			h.t.Fatal(err)
		}
		if _, err := h.msg.GraphTrailClaim(h.ctx, &types.MsgGraphTrailClaim{
			Party: party, GraphTaskId: taskID,
			TrailRoot: trailRootOf(t, graphID, trail), InitialRoot: trail[0][:], InitialProof: siblingsOf(pi),
			FinalRoot: trail[len(trail)-1][:], FinalProof: siblingsOf(pf),
		}); err != nil {
			h.t.Fatal(err)
		}
	}
	claim(h.worker, workerTrail)
	claim(h.challenger, honest.Trail)

	// Bisection rounds: epochs are block heights.
	for rounds := 0; rounds < 24; rounds++ {
		record, err := h.keeper.GetGraphDispute(h.ctx, taskID)
		if err != nil {
			h.t.Fatal(err)
		}
		if record.Status == GraphDisputeArbReady {
			break
		}
		dispute, err := canonical.RestoreGraphDispute(record.Snapshot, &fx.graph, ChallengeRoundBlocks)
		if err != nil {
			h.t.Fatal(err)
		}
		low, high := dispute.Interval()
		mid := low + (high-low)/2
		submitMid := func(party string, trail []canonical.Hash) {
			proof, err := canonical.TrailProof(graphID, trail, mid)
			if err != nil {
				h.t.Fatal(err)
			}
			h.advance(h.currentHeight() + 1)
			if _, err := h.msg.GraphMidPoint(h.ctx, &types.MsgGraphMidPoint{
				Party: party, GraphTaskId: taskID,
				StateRoot: trail[mid][:], ProofSiblings: siblingsOf(proof), Epoch: h.currentHeight(),
			}); err != nil {
				h.t.Fatal(err)
			}
		}
		submitMid(h.worker, workerTrail)
		submitMid(h.challenger, honest.Trail)
	}

	// Arbitration by a permissionless actor.
	record, err := h.keeper.GetGraphDispute(h.ctx, taskID)
	if err != nil {
		h.t.Fatal(err)
	}
	if record.Status != GraphDisputeArbReady {
		h.t.Fatal("dispute did not reach arbitration readiness")
	}
	dispute, err := canonical.RestoreGraphDispute(record.Snapshot, &fx.graph, ChallengeRoundBlocks)
	if err != nil {
		h.t.Fatal(err)
	}
	nodeID, err := dispute.FirstDivergentNode()
	if err != nil {
		h.t.Fatal(err)
	}
	node := fx.graph.Nodes[nodeID]

	corruptedTensors := map[canonical.TensorRef]canonical.Tensor{}
	for ref, tens := range honest.Tensors {
		corruptedTensors[ref] = tens
	}
	workerRef := canonical.TensorRef{Kind: 1, Index: nodeID}
	corrupted := corruptedTensors[workerRef]
	corrupted.Data = append([]int32(nil), corrupted.Data...)
	corrupted.Data[0]++
	corruptedTensors[workerRef] = corrupted

	chunkClaim := func(tens canonical.Tensor, index uint32) ([]byte, []byte, [][]byte) {
		root, err := tens.TensorRoot()
		if err != nil {
			h.t.Fatal(err)
		}
		chunk, err := tens.ChunkBytes(int(index))
		if err != nil {
			h.t.Fatal(err)
		}
		proof, err := tens.ChunkProof(int(index))
		if err != nil {
			h.t.Fatal(err)
		}
		return root[:], chunk, hashListBytes(proof)
	}
	workerRoot, workerChunk, workerProof := chunkClaim(corrupted, 0)
	chRoot, chChunk, chProof := chunkClaim(honest.Tensors[workerRef], 0)

	// Evidence: one chunk per node input at the disputed chunk index,
	// with state-inclusion proofs against the committed input state.
	live := map[canonical.TensorRef]canonical.Tensor{}
	for ref, tens := range honest.Tensors {
		if ref.Kind == 0 || ref.Index < nodeID {
			live[ref] = tens
		}
	}
	var evidence []*types.GraphChunkEvidence
	for _, ref := range node.Inputs {
		tens := honest.Tensors[ref]
		root, err := tens.TensorRoot()
		if err != nil {
			h.t.Fatal(err)
		}
		chunk, err := tens.ChunkBytes(0)
		if err != nil {
			h.t.Fatal(err)
		}
		proof, err := tens.ChunkProof(0)
		if err != nil {
			h.t.Fatal(err)
		}
		stateProof, err := canonical.StateInclusionProof(live, ref)
		if err != nil {
			h.t.Fatal(err)
		}
		descJSON, err := json.Marshal(tens.Desc)
		if err != nil {
			h.t.Fatal(err)
		}
		count := uint32(tens.ChunkCount())
		evidence = append(evidence, &types.GraphChunkEvidence{
			RefKind: uint32(ref.Kind), RefIndex: ref.Index, DescJson: descJSON,
			Root: root[:], ChunkIndex: 0, Count: count,
			Chunk: chunk, Proof: hashListBytes(proof), StateProof: siblingsOf(stateProof),
		})
	}
	var rope []int64
	if node.OperatorID == canonical.OpRoPEFixedV1 {
		for _, v := range fx.inputs[node.Inputs[len(node.Inputs)-1].Index].Data {
			rope = append(rope, int64(v))
		}
	}
	res, err := h.msg.ArbitrateGraphNode(h.ctx, &types.MsgArbitrateGraphNode{
		Actor: h.challenger, GraphTaskId: taskID,
		WorkerOutRoot: workerRoot, WorkerChunkIndex: 0, WorkerChunk: workerChunk, WorkerChunkProof: workerProof,
		ChallengerOutRoot: chRoot, ChallengerChunkIndex: 0, ChallengerChunk: chChunk, ChallengerChunkProof: chProof,
		Evidence: evidence, RopeTable: rope,
	})
	if err != nil {
		h.t.Fatal(err)
	}
	if nodeID != corruptionNode {
		t.Fatalf("dispute localized to node %d, want %d", nodeID, corruptionNode)
	}
	return res.Outcome
}

// trailRootOf is the root over the hashed trail leaves (TrailLevels),
// which is what a party's trail claim locks.
func trailRootOf(t *testing.T, graphID canonical.Hash, trail []canonical.Hash) []byte {
	t.Helper()
	levels, err := canonical.TrailLevels(graphID, trail)
	if err != nil {
		t.Fatal(err)
	}
	root := levels[len(levels)-1][0]
	return root[:]
}

func hashListBytes(hashes []canonical.Hash) [][]byte {
	out := make([][]byte, len(hashes))
	for i, h := range hashes {
		out[i] = append([]byte(nil), h[:]...)
	}
	return out
}

func siblingsOf(proof canonical.MerkleProof) [][]byte {
	out := make([][]byte, len(proof.Siblings))
	for i, s := range proof.Siblings {
		out[i] = append([]byte(nil), s[:]...)
	}
	return out
}

// --- tests -----------------------------------------------------------------

func TestGraphHonestFlow(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphFixture(t)
	taskID, postGraphID := h.postGraph(t, fx)
	graphID := fx.graphID(t)
	if !bytes.Equal(postGraphID, graphID[:]) {
		t.Fatal("posted graph id mismatch")
	}
	assignmentRef := h.acceptGraph(t, taskID)
	assertReserved(t, h, h.worker, 2*MinBond)
	h.submitGraphResult(t, taskID, fx, assignmentRef, fx.exec.Outputs)

	// Finalization before the window closes must be refused.
	if _, err := h.msg.FinalizeGraphTask(h.ctx, &types.MsgFinalizeGraphTask{Actor: h.requester, GraphTaskId: taskID}); err == nil {
		t.Fatal("finalized before the challenge window closed")
	}
	task, err := h.keeper.GetGraphTask(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	h.advance(task.ChallengeEnd + 1)
	res, err := h.msg.FinalizeGraphTask(h.ctx, &types.MsgFinalizeGraphTask{Actor: h.requester, GraphTaskId: taskID})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ReceiptId) != 32 {
		t.Fatal("finalization produced no receipt")
	}
	task, err = h.keeper.GetGraphTask(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	if task.Status != GraphStatusFinalized {
		t.Fatalf("status = %s", task.Status)
	}
	assertReserved(t, h, h.worker, 0)
	// Replay: finalizing again must be refused.
	if _, err := h.msg.FinalizeGraphTask(h.ctx, &types.MsgFinalizeGraphTask{Actor: h.requester, GraphTaskId: taskID}); err == nil {
		t.Fatal("double finalization accepted")
	}
}

func TestGraphFraudDisputeAndArbitration(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphFixture(t)
	taskID, _ := h.postGraph(t, fx)
	assignmentRef := h.acceptGraph(t, taskID)

	// The lying worker commits the output of a corrupted execution: the
	// last node (the block output ADD) was mutated.
	fraudNode := uint32(len(fx.graph.Nodes) - 1)
	workerTrail, err := canonical.FraudTrail(&fx.graph, fx.exec, fraudNode, func(tens *canonical.Tensor) {
		tens.Data[0]++
	})
	if err != nil {
		t.Fatal(err)
	}
	var taskRef [8]byte
	binary.BigEndian.PutUint64(taskRef[:], taskID)
	rc, err := canonical.NewGraphResultCommit(&fx.graph, taskRef[:], assignmentRef, h.workKeys.networkPub,
		[]canonical.Hash{workerTrail[len(workerTrail)-1]}, 42)
	if err != nil {
		t.Fatal(err)
	}
	if err := canonical.SignGraphResultCommit(rc, h.workKeys.networkPriv); err != nil {
		t.Fatal(err)
	}
	if _, err := h.msg.SubmitGraphResult(h.ctx, &types.MsgSubmitGraphResult{
		Worker: h.worker, GraphTaskId: taskID,
		OutputRoots: rc.OutputRoots, FinalOutputRoot: rc.FinalOutputRoot,
		CompletedEpoch: rc.CompletedEpoch, WorkerSignature: rc.Signature,
	}); err != nil {
		t.Fatal(err)
	}

	// The honest challenger counter-claims the honest final root.
	h.bondWorker(h.challenger, h.chalKeys)
	if _, err := h.msg.OpenGraphChallenge(h.ctx, &types.MsgOpenGraphChallenge{
		Challenger: h.challenger, GraphTaskId: taskID,
		ChallengerOutputRoots: hashesOf(t, fx.exec.Outputs), ChallengeBond: MinBond,
	}); err != nil {
		t.Fatal(err)
	}
	outcome := h.runGraphDispute(t, taskID, fx, fraudNode)
	if outcome != "challenger_wins" {
		t.Fatalf("fraud verdict = %s, want challenger_wins", outcome)
	}
	task, err := h.keeper.GetGraphTask(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	if task.Status != GraphStatusFraud {
		t.Fatalf("task status = %s, want fraud", task.Status)
	}
	if len(task.ReceiptID) != 0 {
		t.Fatal("a fraudulent task must not receive a receipt")
	}
	assertReserved(t, h, h.worker, 0)
}

func TestGraphFalseChallengeRefused(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphFixture(t)
	taskID, _ := h.postGraph(t, fx)
	assignmentRef := h.acceptGraph(t, taskID)
	h.submitGraphResult(t, taskID, fx, assignmentRef, fx.exec.Outputs)
	h.bondWorker(h.challenger, h.chalKeys)
	if _, err := h.msg.OpenGraphChallenge(h.ctx, &types.MsgOpenGraphChallenge{
		Challenger: h.challenger, GraphTaskId: taskID,
		ChallengerOutputRoots: hashesOf(t, fx.exec.Outputs), ChallengeBond: MinBond,
	}); err == nil {
		t.Fatal("a challenge asserting the committed result was accepted")
	}
}

func TestGraphReplayAndPhases(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphFixture(t)
	taskID, _ := h.postGraph(t, fx)
	assignmentRef := h.acceptGraph(t, taskID)
	h.submitGraphResult(t, taskID, fx, assignmentRef, fx.exec.Outputs)
	// Replay the same result.
	var taskRef [8]byte
	binary.BigEndian.PutUint64(taskRef[:], taskID)
	rc, err := canonical.NewGraphResultCommit(&fx.graph, taskRef[:], assignmentRef, h.workKeys.networkPub, fx.exec.Outputs, 42)
	if err != nil {
		t.Fatal(err)
	}
	if err := canonical.SignGraphResultCommit(rc, h.workKeys.networkPriv); err != nil {
		t.Fatal(err)
	}
	if _, err := h.msg.SubmitGraphResult(h.ctx, &types.MsgSubmitGraphResult{
		Worker: h.worker, GraphTaskId: taskID,
		OutputRoots: rc.OutputRoots, FinalOutputRoot: rc.FinalOutputRoot,
		CompletedEpoch: rc.CompletedEpoch, WorkerSignature: rc.Signature,
	}); err == nil {
		t.Fatal("replayed result accepted")
	}
	// A second challenge while one is active is refused.
	h.bondWorker(h.challenger, h.chalKeys)
	alt := hashesOf(t, fx.exec.Outputs)
	alt[0][0] ^= 0x01
	if _, err := h.msg.OpenGraphChallenge(h.ctx, &types.MsgOpenGraphChallenge{
		Challenger: h.challenger, GraphTaskId: taskID,
		ChallengerOutputRoots: alt, ChallengeBond: MinBond,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := h.msg.OpenGraphChallenge(h.ctx, &types.MsgOpenGraphChallenge{
		Challenger: h.worker, GraphTaskId: taskID,
		ChallengerOutputRoots: alt, ChallengeBond: MinBond,
	}); err == nil {
		t.Fatal("the worker challenged its own result")
	}
}

func TestGraphMislabeledTrailRejected(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphFixture(t)
	taskID, _ := h.postGraph(t, fx)
	assignmentRef := h.acceptGraph(t, taskID)
	h.submitGraphResult(t, taskID, fx, assignmentRef, fx.exec.Outputs)
	h.bondWorker(h.challenger, h.chalKeys)
	alt := hashesOf(t, fx.exec.Outputs)
	alt[0][0] ^= 0x01
	if _, err := h.msg.OpenGraphChallenge(h.ctx, &types.MsgOpenGraphChallenge{
		Challenger: h.challenger, GraphTaskId: taskID,
		ChallengerOutputRoots: alt, ChallengeBond: MinBond,
	}); err != nil {
		t.Fatal(err)
	}
	// A claim whose initial state is not the committed input state.
	graphID := fx.graphID(t)
	proof, err := canonical.TrailProof(graphID, fx.exec.Trail, 0)
	if err != nil {
		t.Fatal(err)
	}
	var bogusInitial canonical.Hash
	bogusInitial[0] = 0xAA
	if _, err := h.msg.GraphTrailClaim(h.ctx, &types.MsgGraphTrailClaim{
		Party: h.worker, GraphTaskId: taskID,
		TrailRoot: trailRootOf(t, graphID, fx.exec.Trail), InitialRoot: bogusInitial[:], InitialProof: siblingsOf(proof),
		FinalRoot: fx.exec.Trail[len(fx.exec.Trail)-1][:], FinalProof: siblingsOf(mustProof(t, graphID, fx.exec.Trail, uint32(len(fx.exec.Trail)-1))),
	}); err == nil {
		t.Fatal("mislabeled initial state accepted")
	}
}

func mustProof(t *testing.T, graphID canonical.Hash, trail []canonical.Hash, index uint32) canonical.MerkleProof {
	t.Helper()
	proof, err := canonical.TrailProof(graphID, trail, index)
	if err != nil {
		t.Fatal(err)
	}
	return proof
}

func TestGraphRestartPersistence(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphFixture(t)
	taskID, _ := h.postGraph(t, fx)
	assignmentRef := h.acceptGraph(t, taskID)
	h.submitGraphResult(t, taskID, fx, assignmentRef, fx.exec.Outputs)
	h.bondWorker(h.challenger, h.chalKeys)
	alt := hashesOf(t, fx.exec.Outputs)
	alt[0][0] ^= 0x01
	if _, err := h.msg.OpenGraphChallenge(h.ctx, &types.MsgOpenGraphChallenge{
		Challenger: h.challenger, GraphTaskId: taskID,
		ChallengerOutputRoots: alt, ChallengeBond: MinBond,
	}); err != nil {
		t.Fatal(err)
	}
	// Claims lock the dispute; then a "restart" must not lose it.
	graphID := fx.graphID(t)
	claim := func(party string, trail []canonical.Hash) {
		pi := mustProof(t, graphID, trail, 0)
		pf := mustProof(t, graphID, trail, uint32(len(trail)-1))
		if _, err := h.msg.GraphTrailClaim(h.ctx, &types.MsgGraphTrailClaim{
			Party: party, GraphTaskId: taskID,
			TrailRoot: trailRootOf(t, graphID, trail), InitialRoot: trail[0][:], InitialProof: siblingsOf(pi),
			FinalRoot: trail[len(trail)-1][:], FinalProof: siblingsOf(pf),
		}); err != nil {
			t.Fatal(err)
		}
	}
	workerTrail, err := canonical.FraudTrail(&fx.graph, fx.exec, 0, func(tens *canonical.Tensor) { tens.Data[0]++ })
	if err != nil {
		t.Fatal(err)
	}
	claim(h.worker, workerTrail)
	claim(h.challenger, fx.exec.Trail)
	record, err := h.keeper.GetGraphDispute(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != GraphDisputeBisection || len(record.Snapshot) == 0 {
		t.Fatalf("dispute not persisted: %s", record.Status)
	}
	// Rebuild a fresh message server over the same store: state survives
	// and the bisection continues from the persisted snapshot.
	fresh := h.keeper.MsgServer()
	dispute, err := canonical.RestoreGraphDispute(record.Snapshot, &fx.graph, ChallengeRoundBlocks)
	if err != nil {
		t.Fatal(err)
	}
	low, high := dispute.Interval()
	mid := low + (high-low)/2
	if _, err := fresh.GraphMidPoint(h.ctx, &types.MsgGraphMidPoint{
		Party: h.worker, GraphTaskId: taskID, StateRoot: workerTrail[mid][:],
		ProofSiblings: siblingsOf(mustProof(t, graphID, workerTrail, mid)), Epoch: h.currentHeight(),
	}); err != nil {
		t.Fatal(err)
	}
}

func TestGraphDoSAdmissionBounds(t *testing.T) {
	h := newGemmHarness(t, 100)
	fx := buildChainGraphFixture(t)

	// Oversized descriptor.
	big := make([]byte, MaxGraphJSONBytes+1)
	if _, err := h.msg.PostGraphTask(h.ctx, &types.MsgPostGraphTask{
		Requester: h.requester, RequesterProtocolPubkey: h.reqKeys.protocolPub, RequesterKeyProof: make([]byte, 64),
		RequesterNonce: []byte("n"), GraphJson: big, ChallengeWindow: ChallengeBlocks, MaxFee: MinBond, InputDataRef: "x",
	}); err == nil {
		t.Fatal("oversized descriptor accepted")
	}

	// A structurally valid JSON with too many nodes is refused by the
	// admission bounds before any expensive validation.
	nodes := make([]map[string]any, 0, MaxGraphNodes+2)
	for i := 0; i <= MaxGraphNodes+1; i++ {
		nodes = append(nodes, map[string]any{"node_id": i})
	}
	tooMany, err := json.Marshal(map[string]any{
		"protocol_version": canonical.GraphProtocolVersion,
		"spec":             "DOS",
		"nodes":            nodes,
	})
	if err != nil {
		t.Fatal(err)
	}
	tooManyHuge := append(tooMany, bytes.Repeat([]byte{' '}, MaxGraphJSONBytes-len(tooMany)-1)...)
	nonce := []byte("graph-requester-nonce")
	binding, err := canonicalJSON(map[string]any{
		"chain_id": h.ctx.ChainID(), "requester": h.requester,
		"requester_protocol_pubkey": hexStr(h.reqKeys.protocolPub),
		"requester_nonce":           hexStr(nonce),
	})
	if err != nil {
		t.Fatal(err)
	}
	proof := ed25519.Sign(h.reqKeys.protocolPriv, append([]byte(GraphRequesterKeyBindingDomain), binding...))
	if _, err := h.msg.PostGraphTask(h.ctx, &types.MsgPostGraphTask{
		Requester: h.requester, RequesterProtocolPubkey: h.reqKeys.protocolPub, RequesterKeyProof: proof,
		RequesterNonce: nonce, GraphJson: tooManyHuge, ChallengeWindow: ChallengeBlocks, MaxFee: MinBond, InputDataRef: "x",
	}); err == nil {
		t.Fatal("over-limit graph accepted")
	}
	_ = fx
}

func TestGraphGasScheduleFrozen(t *testing.T) {
	schedule := GraphGasScheduleJSON()
	if schedule["version"] != GraphGasScheduleVersion {
		t.Fatal("gas schedule version drifted")
	}
	work := canonical.WorkVector{{Key: "GEMM_MAC", Value: 1_000_000}, {Key: "ADD_ELEMENT", Value: 10}}
	gas := graphWorkGas(work)
	want := (uint64(1_000_010) - GasGraphFreeWorkUnits) * GasGraphWorkUnit
	if gas != want {
		t.Fatalf("work gas = %d, want %d", gas, want)
	}
}

func TestGraphAssignmentRefDeterministic(t *testing.T) {
	a := graphAssignmentRef([]byte("g"), "worker", []byte("nonce"), 7)
	b := graphAssignmentRef([]byte("g"), "worker", []byte("nonce"), 7)
	if !bytes.Equal(a, b) || len(a) != 32 {
		t.Fatal("assignment ref is not deterministic")
	}
	c := graphAssignmentRef([]byte("g"), "worker", []byte("nonce"), 8)
	if bytes.Equal(a, c) {
		t.Fatal("assignment ref ignores the height")
	}
}
