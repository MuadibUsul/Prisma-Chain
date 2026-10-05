package compute

// CANONICAL_GRAPH_V2 on-chain path (Phase F.5C).  Additive: the task
// envelope (`MsgPostGraphTask.graph_json`) carries its own explicit
// `protocol_version`, and the chain dispatches on it.  V1 transactions
// keep their exact behaviour; nothing is silently reinterpreted.
//
// Scope of the V2 route in this file: post (V2 admission with typed
// descriptor validation and arithmetic/policy binding), submit
// (GraphResultCommitV3 validation), finalize (V3 receipt).  The dispute
// and DA extensions build on the same dispatch.

import (
	"context"
	"crypto/ed25519"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"prismachain/chain/x/compute/types"
	"prismachain/compute/canonical"
)

// graphProtocolVersionOf extracts the descriptor's own protocol_version.
func graphProtocolVersionOf(graphJSON []byte) (string, error) {
	var probe struct {
		ProtocolVersion string `json:"protocol_version"`
	}
	if err := json.Unmarshal(graphJSON, &probe); err != nil {
		return "", err
	}
	return probe.ProtocolVersion, nil
}

// graphTaskProtocolV2 reports whether a stored task is a V2 task.
func graphTaskProtocolV2(task *GraphTask) bool {
	return task.ProtocolVersion == canonical.ProtocolVersionGraphV2
}

// graphDescriptorV2 decodes, re-validates and re-hashes the stored V2
// descriptor; the stored graph id must match the derived GraphIDV2.
func graphDescriptorV2(task *GraphTask) (*canonical.GraphDescriptorV2, error) {
	var g canonical.GraphDescriptorV2
	if err := json.Unmarshal(task.GraphJSON, &g); err != nil {
		return nil, fmt.Errorf("graph V2 descriptor decode: %w", err)
	}
	if err := g.ValidateV2(); err != nil {
		return nil, err
	}
	graphID, err := g.GraphIDV2()
	if err != nil {
		return nil, err
	}
	if !equalBytes32(graphID[:], task.GraphID) {
		return nil, errors.New("graph V2 descriptor does not match the committed graph id")
	}
	return &g, nil
}

// postGraphTaskV2 is the admission path for a canonical GRAPH_V2 task:
// bounds first, then full typed validation (dtypes, ranges, operator
// versions, arithmetic profile, MaxSafeK64 per GEMM), then escrow and
// storage.  The task records its protocol version so every later phase
// dispatches deterministically.
func (s msgServer) postGraphTaskV2(ctx context.Context, msg *types.MsgPostGraphTask,
	height uint64, requester sdk.AccAddress) (*types.MsgPostGraphTaskResponse, error) {
	var graph canonical.GraphDescriptorV2
	if err := json.Unmarshal(msg.GraphJson, &graph); err != nil {
		return nil, fmt.Errorf("graph V2 descriptor decode: %w", err)
	}
	if len(graph.Nodes) > MaxGraphNodes || len(graph.Inputs) > MaxGraphInputs ||
		len(graph.Outputs) > MaxGraphOutputs {
		return nil, errors.New("graph V2 exceeds the admission bounds")
	}
	if err := graph.ValidateV2(); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "graph V2 decode",
		GasGraphNodeDecode*uint64(len(graph.Nodes))+GasGraphInputDecode*uint64(len(graph.Inputs)))
	work, err := canonical.GraphWorkVectorV2(&graph)
	if err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "graph V2 work", graphWorkGas(work))
	graphID, err := graph.GraphIDV2()
	if err != nil {
		return nil, err
	}
	if err := s.bank.SendCoinsFromAccountToModule(ctx, requester, ModuleName, amount(msg.MaxFee)); err != nil {
		return nil, err
	}
	store := s.store(ctx)
	idBytes := store.Get([]byte("graph:next"))
	var id uint64 = 1
	if len(idBytes) == 8 {
		id = binary.BigEndian.Uint64(idBytes)
	}
	if id == ^uint64(0) {
		return nil, errors.New("graph task id exhausted")
	}
	var next [8]byte
	binary.BigEndian.PutUint64(next[:], id+1)
	store.Set([]byte("graph:next"), next[:])
	task := GraphTask{
		ID: id, GraphID: graphID[:], Spec: graph.Spec, GraphJSON: append([]byte(nil), msg.GraphJson...),
		ProtocolVersion:        canonical.ProtocolVersionGraphV2,
		Requester:              msg.Requester,
		RequesterProtocolPubKey: append([]byte(nil), msg.RequesterProtocolPubkey...),
		InputDataRef:           msg.InputDataRef, ChallengeWindow: msg.ChallengeWindow,
		MaxPricePerCWU: msg.MaxPricePerCwu, MaxFee: msg.MaxFee, IssuedHeight: height,
		Status: GraphStatusPosted,
	}
	if err := s.setGraphTask(ctx, task); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "graph V2 store",
		GasGraphStore+GasGraphHash*uint64(len(msg.GraphJson)/256+1))
	return &types.MsgPostGraphTaskResponse{GraphTaskId: id, GraphId: graphID[:]}, nil
}

// submitGraphResultV3 validates a GraphResultCommitV3 submission for a V2
// task and locks the manifest root before any verification randomness.
func (s msgServer) submitGraphResultV3(ctx context.Context, task *GraphTask,
	graph *canonical.GraphDescriptorV2, msg *types.MsgSubmitGraphResultV2,
	height uint64) (*types.MsgSubmitGraphResultV2Response, error) {
	rc, err := buildAndValidateV3(task, graph, msg)
	if err != nil {
		return nil, err
	}
	task.CommitVersion = "3"
	task.NodeOutputManifestRoot = append([]byte(nil), rc.NodeOutputManifestRootV2...)
	task.FinalOutputRoot = append([]byte(nil), rc.FinalOutputRoot...)
	task.OutputRoots = cloneBytes2D(rc.OutputRoots)
	task.ResultSignature = append([]byte(nil), rc.Signature...)
	task.CompletedEpoch = msg.CompletedEpoch
	task.ResultSubmittedHeight = height
	task.ChallengeEnd = height + task.ChallengeWindow
	task.Status = GraphStatusResultSubmitted
	if err := s.setGraphTask(ctx, *task); err != nil {
		return nil, err
	}
	return &types.MsgSubmitGraphResultV2Response{ChallengeEnd: task.ChallengeEnd}, nil
}

func buildAndValidateV3(task *GraphTask, graph *canonical.GraphDescriptorV2,
	msg *types.MsgSubmitGraphResultV2) (*canonical.GraphResultCommitV3, error) {
	if len(msg.NodeOutputManifestRoot) != 32 {
		return nil, errors.New("graph V3 manifest root must be 32 bytes")
	}
	if len(msg.OutputRoots) != len(graph.Outputs) || len(msg.WorkerSignature) != ed25519.SignatureSize {
		return nil, errors.New("graph V3 commitment shape is invalid")
	}
	var taskRef [8]byte
	binary.BigEndian.PutUint64(taskRef[:], task.ID)
	rc := &canonical.GraphResultCommitV3{
		ProtocolVersion:          canonical.GraphResultCommitV3Version,
		GraphID:                  append([]byte(nil), task.GraphID...),
		ArithmeticID:             graph.Arithmetic.ID,
		PolicyID:                 graph.Arithmetic.PolicyID,
		TaskRef:                  taskRef[:],
		AssignmentRef:            append([]byte(nil), task.AssignmentRef...),
		WorkerPubKey:             append([]byte(nil), task.WorkerProtocolPubKey...),
		NodeOutputManifestRootV2: append([]byte(nil), msg.NodeOutputManifestRoot...),
		FinalOutputRoot:          append([]byte(nil), msg.FinalOutputRoot...),
		OutputRoots:              cloneBytes2D(msg.OutputRoots),
		CompletedEpoch:           msg.CompletedEpoch,
		Signature:                append([]byte(nil), msg.WorkerSignature...),
	}
	if err := canonical.ValidateGraphResultCommitV3(graph, rc, nil); err != nil {
		return nil, err
	}
	return rc, nil
}

// --- A1-04/A1-05: V2 trail claim + midpoint ----------------------------------

// graphV2InitialStateRootV2 derives the V2 state root of the committed
// graph inputs (kind=0 leaves over the descriptor's input roots).
func graphV2InitialStateRootV2(g *canonical.GraphDescriptorV2) (canonical.Hash, error) {
	live := map[canonical.TensorRef]canonical.Hash{}
	for i, in := range g.Inputs {
		var h canonical.Hash
		if len(in.Root) != 32 {
			return canonical.Hash{}, errors.New("graph V2 input root malformed")
		}
		copy(h[:], in.Root)
		live[canonical.TensorRef{Kind: 0, Index: uint32(i)}] = h
	}
	return canonical.GraphStateRootV2FromRoots(live)
}

// buildTrailClaimV2 mirrors buildTrailClaim over the V2 trail domain: the
// claimed initial state must equal the descriptor-derived input state and
// both endpoint proofs must verify under VerifyTrailProofV2.
func (s msgServer) buildTrailClaimV2(graph *canonical.GraphDescriptorV2, task *GraphTask,
	msg *types.MsgGraphTrailClaim) (GraphTrailClaimData, error) {
	graphID, err := root32Of(task.GraphID)
	if err != nil {
		return GraphTrailClaimData{}, err
	}
	expectedInitial, err := graphV2InitialStateRootV2(graph)
	if err != nil {
		return GraphTrailClaimData{}, err
	}
	initialRoot, err := root32Of(msg.InitialRoot)
	if err != nil {
		return GraphTrailClaimData{}, err
	}
	if initialRoot != expectedInitial {
		return GraphTrailClaimData{}, errors.New("claimed initial state is not the committed input state")
	}
	finalRoot, err := root32Of(msg.FinalRoot)
	if err != nil {
		return GraphTrailClaimData{}, err
	}
	trailRoot, err := root32Of(msg.TrailRoot)
	if err != nil {
		return GraphTrailClaimData{}, err
	}
	count := uint32(len(graph.Nodes) + 1)
	initialProof, err := hashesFrom(msg.InitialProof)
	if err != nil {
		return GraphTrailClaimData{}, err
	}
	finalProof, err := hashesFrom(msg.FinalProof)
	if err != nil {
		return GraphTrailClaimData{}, err
	}
	if !canonical.VerifyTrailProofV2(trailRoot, graphID, 0, count, initialRoot, initialProof) {
		return GraphTrailClaimData{}, errors.New("initial state proof does not match the claimed trail root")
	}
	if !canonical.VerifyTrailProofV2(trailRoot, graphID, count-1, count, finalRoot, finalProof) {
		return GraphTrailClaimData{}, errors.New("final state proof does not match the claimed trail root")
	}
	return GraphTrailClaimData{
		Root: trailRoot[:], InitialRoot: initialRoot[:], InitialProof: cloneBytes2D(msg.InitialProof),
		FinalRoot: finalRoot[:], FinalProof: cloneBytes2D(msg.FinalProof),
	}, nil
}

// lockTrailClaimV2 stores one party's claim and, once both are locked,
// opens the V2 bisection (or short-circuits when the endpoints agree).
func (s msgServer) lockTrailClaimV2(ctx context.Context, task *GraphTask,
	graph *canonical.GraphDescriptorV2, record *GraphDisputeRecord,
	msg *types.MsgGraphTrailClaim, claim GraphTrailClaimData, height uint64) (*types.MsgGraphTrailClaimResponse, error) {
	worker := msg.Party == task.Worker
	if worker {
		if len(record.WorkerClaim.Root) != 0 {
			return nil, errors.New("worker trail already claimed")
		}
		record.WorkerClaim = claim
	} else {
		if len(record.ChallengerClaim.Root) != 0 {
			return nil, errors.New("challenger trail already claimed")
		}
		record.ChallengerClaim = claim
	}
	if len(record.WorkerClaim.Root) != 0 && len(record.ChallengerClaim.Root) != 0 {
		if equalBytes32(record.WorkerClaim.FinalRoot, record.ChallengerClaim.FinalRoot) {
			if err := s.bank.BurnCoins(ctx, ModuleName, amount(record.Bond)); err != nil {
				return nil, err
			}
			task.SurvivedChallenge = true
			task.Status = GraphStatusResultSubmitted
			task.ChallengeEnd = height + task.ChallengeWindow
			task.DisputeTranscriptDigest = record.TranscriptDigest
			s.deleteGraphDispute(ctx, task.ID)
			if err := s.setGraphTask(ctx, *task); err != nil {
				return nil, err
			}
			return &types.MsgGraphTrailClaimResponse{}, nil
		}
		graphID, err := root32Of(task.GraphID)
		if err != nil {
			return nil, err
		}
		dispute, err := canonical.NewGraphDisputeV2(canonical.GraphDisputeConfigV2{
			Graph: graph, GraphID: graphID, RoundPeriod: ChallengeRoundBlocks,
		}, toCanonicalClaim(record.WorkerClaim), toCanonicalClaim(record.ChallengerClaim), height)
		if err != nil {
			return nil, err
		}
		snapshot, err := dispute.SnapshotV2()
		if err != nil {
			return nil, err
		}
		record.Snapshot = snapshot
		record.Status = GraphDisputeBisection
		if dispute.ArbReady() {
			// A single-node graph converges the moment both trails lock:
			// no midpoint round can be submitted, so promote the record
			// immediately or the arbitration would be unreachable.
			record.Status = GraphDisputeArbReady
		}
		record.TranscriptDigest, err = graphTranscriptStep(record.TranscriptDigest, "claims_locked_v2",
			map[string]any{"worker_root": fmt.Sprintf("%x", record.WorkerClaim.Root),
				"challenger_root": fmt.Sprintf("%x", record.ChallengerClaim.Root)})
		if err != nil {
			return nil, err
		}
	}
	if err := s.setGraphDispute(ctx, *record); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "claim hashing",
		GasGraphHash*uint64(len(claim.InitialProof)+len(claim.FinalProof)+4))
	return &types.MsgGraphTrailClaimResponse{}, nil
}

// graphMidPointV2 mirrors GraphMidPoint over the restored V2 dispute.
func (s msgServer) graphMidPointV2(ctx context.Context, task *GraphTask,
	record *GraphDisputeRecord, msg *types.MsgGraphMidPoint, height uint64) (*types.MsgGraphMidPointResponse, error) {
	graph, err := graphDescriptorV2(task)
	if err != nil {
		return nil, err
	}
	var party canonical.Party
	switch msg.Party {
	case task.Worker:
		party = canonical.Worker
	case record.Challenger:
		party = canonical.Challenger
	default:
		return nil, errors.New("only the assigned worker or the challenger may submit midpoints")
	}
	dispute, err := canonical.RestoreGraphDisputeV2(record.Snapshot, graph, ChallengeRoundBlocks)
	if err != nil {
		return nil, err
	}
	stateRoot, err := root32Of(msg.StateRoot)
	if err != nil {
		return nil, err
	}
	proof, err := hashesFrom(msg.ProofSiblings)
	if err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "midpoint proofs",
		GasGraphProofSibling*uint64(len(proof))+GasGraphHash)
	if _, err := dispute.SubmitMid(party, stateRoot, proof, msg.Epoch); err != nil {
		return nil, err
	}
	snapshot, err := dispute.SnapshotV2()
	if err != nil {
		return nil, err
	}
	record.Snapshot = snapshot
	if dispute.ArbReady() {
		record.Status = GraphDisputeArbReady
	}
	if err := s.setGraphDispute(ctx, *record); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "dispute store", GasGraphStore+GasGraphHash*uint64(len(snapshot)/256+1))
	return &types.MsgGraphMidPointResponse{}, nil
}

// --- A1-06/A1-07: V2 node arbitration (cheap ops; wide GEMM -> A2) -----------

// graphStateGeometryV2 mirrors graphStateGeometry over the V2 descriptor.
func graphStateGeometryV2(g *canonical.GraphDescriptorV2, beforeNode uint32,
	ref canonical.TensorRef) (index, count uint32, err error) {
	k := uint32(len(g.Inputs))
	if ref.Kind == 0 {
		if ref.Index >= k {
			return 0, 0, errors.New("evidence references an unknown input")
		}
		return ref.Index, k + beforeNode, nil
	}
	if ref.Kind == 1 {
		if ref.Index >= beforeNode {
			return 0, 0, errors.New("evidence references a tensor that does not exist in the committed state")
		}
		return k + ref.Index, k + beforeNode, nil
	}
	return 0, 0, errors.New("unknown tensor ref kind")
}

// verifyGraphEvidenceV2State checks one evidence tensor root against the
// committed V2 state tree (V2 leaf domain).
func verifyGraphEvidenceV2State(g *canonical.GraphDescriptorV2, beforeNode uint32,
	stateRoot canonical.Hash, ref canonical.TensorRef, root canonical.Hash,
	stateProof [][]byte) error {
	index, count, err := graphStateGeometryV2(g, beforeNode, ref)
	if err != nil {
		return err
	}
	siblings := make([]canonical.Hash, len(stateProof))
	for i, raw := range stateProof {
		h, err := root32Of(raw)
		if err != nil {
			return err
		}
		siblings[i] = h
	}
	leaf := canonical.StateLeafV2(ref.Kind, ref.Index, root)
	if !canonical.VerifyLeafInclusion(stateRoot, leaf, canonical.MerkleProof{Index: index, Count: count, Siblings: siblings}) {
		return errors.New("evidence tensor is not part of the committed input state")
	}
	return nil
}

// arbitrateGraphNodeV2 adjudicates the first divergent node of a V2
// dispute with typed evidence.  Wide GEMM nodes are refused here and must
// use the wide dispute path (A2).
func (s msgServer) arbitrateGraphNodeV2(ctx context.Context, task *GraphTask,
	record *GraphDisputeRecord, msg *types.MsgArbitrateGraphNode) (*types.MsgArbitrateGraphNodeResponse, error) {
	graph, err := graphDescriptorV2(task)
	if err != nil {
		return nil, err
	}
	dispute, err := canonical.RestoreGraphDisputeV2(record.Snapshot, graph, ChallengeRoundBlocks)
	if err != nil {
		return nil, err
	}
	nodeID, err := dispute.FirstDivergentNode()
	if err != nil {
		return nil, err
	}
	node := graph.Nodes[nodeID]
	if node.OperatorID == canonical.OpGEMMWideA13W10 {
		return nil, errors.New("wide GEMM nodes are adjudicated by the wide dispute path (A2)")
	}
	if len(msg.Evidence) == 0 || len(msg.Evidence) > MaxGraphEvidenceChunks {
		return nil, errors.New("graph evidence count out of bounds")
	}
	s.consumeGraphGas(ctx, "arbitration evidence", GasGraphEvidence*uint64(len(msg.Evidence)))
	evidence := make([]canonical.ChunkEvidenceV2, 0, len(msg.Evidence))
	inputStateRoot := dispute.LowStateRoot()
	for _, ev := range msg.Evidence {
		var desc canonical.TensorDescriptorV2
		if err := json.Unmarshal(ev.DescJson, &desc); err != nil {
			return nil, errors.New("malformed evidence descriptor")
		}
		root, err := root32Of(ev.Root)
		if err != nil {
			return nil, err
		}
		proof, err := hashesFrom(ev.Proof)
		if err != nil {
			return nil, err
		}
		ref := canonical.TensorRef{Kind: uint8(ev.RefKind), Index: ev.RefIndex}
		if err := verifyGraphEvidenceV2State(graph, nodeID, inputStateRoot, ref, root, ev.StateProof); err != nil {
			return nil, err
		}
		s.consumeGraphGas(ctx, "evidence state leaf", GasGraphStateLeaf+GasGraphProofSibling*uint64(len(ev.StateProof)))
		evidence = append(evidence, canonical.ChunkEvidenceV2{
			Ref: ref, Desc: desc, Root: root,
			ChunkIndex: ev.ChunkIndex, Count: ev.Count, Bytes: ev.Chunk, Proof: proof,
		})
	}
	var ropeTable *canonical.RopeConstants
	if node.OperatorID == canonical.OpRoPEFixedV1 {
		tableRef := node.Inputs[len(node.Inputs)-1]
		if len(msg.RopeTable) == 0 {
			return nil, errors.New("rope node arbitration requires the pinned table")
		}
		table := make([]int32, len(msg.RopeTable))
		for i, v := range msg.RopeTable {
			table[i] = int32(v)
		}
		tableDesc := canonical.NewDescV2(canonical.DtypeV2Q12_20, int64(len(table)))
		tableData := make([]int64, len(table))
		for i, v := range table {
			tableData[i] = int64(v)
		}
		tableRoot, err := canonical.TensorRootV2Of(tableDesc, tableData)
		if err != nil {
			return nil, err
		}
		if int(tableRef.Index) >= len(graph.Inputs) || !equalBytes32(tableRoot[:], graph.Inputs[tableRef.Index].Root) {
			return nil, errors.New("rope table does not match the committed input")
		}
		pairs := int(node.Output.Shape[len(node.Output.Shape)-1]) / 2
		ropeTable, err = canonical.NewRopeConstants(table, pairs)
		if err != nil {
			return nil, err
		}
	}
	workerRoot, err := root32Of(msg.WorkerOutRoot)
	if err != nil {
		return nil, err
	}
	workerProof, err := hashesFrom(msg.WorkerChunkProof)
	if err != nil {
		return nil, err
	}
	challengerRoot, err := root32Of(msg.ChallengerOutRoot)
	if err != nil {
		return nil, err
	}
	challengerProof, err := hashesFrom(msg.ChallengerChunkProof)
	if err != nil {
		return nil, err
	}
	outcome, err := canonical.ArbitrateNodeChunkV2(graph, node, inputStateRoot,
		canonical.NodeChunkClaim{OutRoot: workerRoot, ChunkIndex: msg.WorkerChunkIndex,
			ChunkBytes: msg.WorkerChunk, Proof: workerProof},
		canonical.NodeChunkClaim{OutRoot: challengerRoot, ChunkIndex: msg.ChallengerChunkIndex,
			ChunkBytes: msg.ChallengerChunk, Proof: challengerProof},
		evidence, ropeTable)
	if err != nil {
		return nil, err
	}
	record.TranscriptDigest, err = graphTranscriptStep(record.TranscriptDigest, "node_arbitration_v2",
		map[string]any{"node": nodeID, "operator": node.OperatorID, "outcome": uint64(outcome)})
	if err != nil {
		return nil, err
	}
	if _, err := s.resolveGraphOutcome(ctx, task, record, outcome); err != nil {
		return nil, err
	}
	return &types.MsgArbitrateGraphNodeResponse{Outcome: canonicalOutcomeName(outcome)}, nil
}
