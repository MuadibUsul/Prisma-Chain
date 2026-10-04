package compute

// CANONICAL_GRAPH_V1 message handling: task admission with escrow, bonded
// assignment, signed result commitment, challenge admission, trail claims,
// persisted bisection, permissionless bounded node arbitration and
// settlement. Identity is double-bound exactly like the GEMM path: the
// Cosmos tx signer must be the on-chain party AND the protocol signature
// must verify under the bonded network Ed25519 key.

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"prismachain/chain/x/compute/types"
	"prismachain/compute/canonical"
)

// verifyGraphRequesterKeyProof checks the requester's protocol key proof.
func verifyGraphRequesterKeyProof(ctx context.Context, requester string, pub, proof, nonce []byte) error {
	if len(pub) != 32 || len(proof) != ed25519.SignatureSize {
		return errors.New("requester protocol key binding is invalid")
	}
	chainID := sdk.UnwrapSDKContext(ctx).ChainID()
	payload, err := canonicalJSON(map[string]any{
		"chain_id":                  chainID,
		"requester":                 requester,
		"requester_protocol_pubkey": hex.EncodeToString(pub),
		"requester_nonce":           hex.EncodeToString(nonce),
	})
	if err != nil || chainID == "" {
		return errors.New("requester protocol key binding is invalid")
	}
	if !ed25519.Verify(pub, append([]byte(GraphRequesterKeyBindingDomain), payload...), proof) {
		return errors.New("requester protocol key possession proof is invalid")
	}
	return nil
}

// PostGraphTask validates, escrows and stores one graph task.
func (s msgServer) PostGraphTask(ctx context.Context, msg *types.MsgPostGraphTask) (*types.MsgPostGraphTaskResponse, error) {
	s.consumeGraphGas(ctx, "post base", GasGraphTxBase)
	requester, err := addr(msg.Requester)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if err := verifyGraphRequesterKeyProof(ctx, msg.Requester, msg.RequesterProtocolPubkey,
		msg.RequesterKeyProof, msg.RequesterNonce); err != nil {
		return nil, err
	}
	if msg.MaxFee == 0 || msg.ChallengeWindow < ChallengeBlocks || len(msg.InputDataRef) > 512 {
		return nil, errors.New("invalid graph task envelope")
	}
	if len(msg.GraphJson) == 0 || len(msg.GraphJson) > MaxGraphJSONBytes {
		return nil, errors.New("graph descriptor size out of bounds")
	}
	var graph canonical.GraphDescriptor
	if err := json.Unmarshal(msg.GraphJson, &graph); err != nil {
		return nil, fmt.Errorf("graph descriptor decode: %w", err)
	}
	// Bounds first: a hostile descriptor must be rejected before any
	// expensive structural validation runs.
	if len(graph.Nodes) > MaxGraphNodes || len(graph.Inputs) > MaxGraphInputs || len(graph.Outputs) > MaxGraphOutputs {
		return nil, errors.New("graph exceeds the admission bounds")
	}
	if err := graph.Validate(); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "graph decode", GasGraphNodeDecode*uint64(len(graph.Nodes))+GasGraphInputDecode*uint64(len(graph.Inputs)))
	work, err := canonical.GraphWorkVector(&graph)
	if err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "graph work", graphWorkGas(work))
	graphID, err := graph.GraphID()
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
		Requester: msg.Requester, RequesterProtocolPubKey: append([]byte(nil), msg.RequesterProtocolPubkey...),
		InputDataRef: msg.InputDataRef, ChallengeWindow: msg.ChallengeWindow,
		MaxPricePerCWU: msg.MaxPricePerCwu, MaxFee: msg.MaxFee, IssuedHeight: height,
		Status: GraphStatusPosted,
	}
	if err := s.setGraphTask(ctx, task); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "graph store", GasGraphStore+GasGraphHash*uint64(len(msg.GraphJson)/256+1))
	return &types.MsgPostGraphTaskResponse{GraphTaskId: id, GraphId: graphID[:]}, nil
}

// AcceptGraphTask binds a bonded worker to the task and derives the
// canonical assignment reference.
func (s msgServer) AcceptGraphTask(ctx context.Context, msg *types.MsgAcceptGraphTask) (*types.MsgAcceptGraphTaskResponse, error) {
	s.consumeGraphGas(ctx, "accept base", GasGraphTxBase)
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if task.Status != GraphStatusPosted {
		return nil, errGraphWrongPhase
	}
	if height > task.IssuedHeight+task.ChallengeWindow {
		return nil, errors.New("graph task acceptance window closed")
	}
	networkKey, err := s.bondedNetworkKey(ctx, msg.Worker)
	if err != nil {
		return nil, err
	}
	if len(msg.AssignmentNonce) == 0 || len(msg.AssignmentNonce) > 64 {
		return nil, errors.New("assignment nonce must be 1..64 bytes")
	}
	requiredBond := task.MaxFee
	if requiredBond < MinBond {
		requiredBond = MinBond
	}
	bond, reserved := s.GetBond(ctx, msg.Worker), s.getReserved(ctx, msg.Worker)
	if bond < reserved || bond-reserved < requiredBond {
		return nil, errors.New("worker bond below graph task collateral")
	}
	s.setReserved(ctx, msg.Worker, reserved+requiredBond)
	task.Worker = msg.Worker
	task.WorkerProtocolPubKey = append([]byte(nil), networkKey...)
	task.AssignmentNonce = append([]byte(nil), msg.AssignmentNonce...)
	task.AcceptedHeight = height
	task.ReservedBond = requiredBond
	task.AssignmentRef = graphAssignmentRef(task.GraphID, msg.Worker, msg.AssignmentNonce, height)
	task.Status = GraphStatusAssigned
	if err := s.setGraphTask(ctx, task); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "graph store", GasGraphStore)
	return &types.MsgAcceptGraphTaskResponse{AssignmentRef: task.AssignmentRef}, nil
}

// graphAssignmentRef derives the deterministic on-chain assignment
// reference swept into the worker's signed result commitment.
func graphAssignmentRef(graphID []byte, worker string, nonce []byte, acceptedHeight uint64) []byte {
	h := sha256.New()
	h.Write([]byte("prisma:graph-assignment:v1\n"))
	h.Write(graphID)
	h.Write([]byte(worker))
	h.Write(nonce)
	var height [8]byte
	binary.BigEndian.PutUint64(height[:], acceptedHeight)
	h.Write(height[:])
	return h.Sum(nil)
}

// SubmitGraphResult validates the worker's canonical GraphResultCommit:
// identity double binding, graph identity, output roots, final output root
// and the network-key signature. No execution trace is accepted.
func (s msgServer) SubmitGraphResult(ctx context.Context, msg *types.MsgSubmitGraphResult) (*types.MsgSubmitGraphResultResponse, error) {
	s.consumeGraphGas(ctx, "submit base", GasGraphTxBase)
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if task.Status != GraphStatusAssigned {
		return nil, errGraphWrongPhase
	}
	if msg.Worker != task.Worker {
		return nil, errors.New("graph result is not from the assigned worker")
	}
	graph, err := graphDescriptor(&task)
	if err != nil {
		return nil, err
	}
	if len(msg.OutputRoots) != len(graph.Outputs) || len(msg.WorkerSignature) != ed25519.SignatureSize {
		return nil, errors.New("graph result commitment shape is invalid")
	}
	var taskRef [8]byte
	binary.BigEndian.PutUint64(taskRef[:], task.ID)
	rc := &canonical.GraphResultCommit{
		ProtocolVersion: canonical.GraphResultCommitVersion,
		GraphID:         append([]byte(nil), task.GraphID...),
		TaskRef:         taskRef[:],
		AssignmentRef:   append([]byte(nil), task.AssignmentRef...),
		WorkerPubKey:    append([]byte(nil), task.WorkerProtocolPubKey...),
		FinalOutputRoot: append([]byte(nil), msg.FinalOutputRoot...),
		OutputRoots:     cloneBytes2D(msg.OutputRoots),
		CompletedEpoch:  msg.CompletedEpoch,
		Signature:       append([]byte(nil), msg.WorkerSignature...),
	}
	s.consumeGraphGas(ctx, "result hashing", GasGraphHash*uint64(len(graph.Outputs)+len(msg.OutputRoots)+2))
	if err := canonical.ValidateGraphResultCommit(graph, rc); err != nil {
		return nil, err
	}
	task.FinalOutputRoot = append([]byte(nil), rc.FinalOutputRoot...)
	task.OutputRoots = cloneBytes2D(rc.OutputRoots)
	task.ResultSignature = append([]byte(nil), rc.Signature...)
	task.CompletedEpoch = msg.CompletedEpoch
	task.ResultSubmittedHeight = height
	task.ChallengeEnd = height + task.ChallengeWindow
	task.Status = GraphStatusResultSubmitted
	if err := s.setGraphTask(ctx, task); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "graph store", GasGraphStore+GasGraphHash*uint64(len(task.GraphJSON)/256+1))
	return &types.MsgSubmitGraphResultResponse{ChallengeEnd: task.ChallengeEnd}, nil
}

// OpenGraphChallenge admits one bonded challenge that counter-claims a
// different final output root. A challenge that asserts the committed
// result is refused up front, and only one challenge may be active.
func (s msgServer) OpenGraphChallenge(ctx context.Context, msg *types.MsgOpenGraphChallenge) (*types.MsgOpenGraphChallengeResponse, error) {
	s.consumeGraphGas(ctx, "challenge base", GasGraphTxBase)
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if task.Status != GraphStatusResultSubmitted {
		return nil, errGraphWrongPhase
	}
	if height > task.ChallengeEnd {
		return nil, errors.New("graph challenge window has closed")
	}
	if msg.Challenger == task.Worker {
		return nil, errors.New("the worker cannot challenge its own result")
	}
	challengerKey, err := s.bondedNetworkKey(ctx, msg.Challenger)
	if err != nil {
		return nil, err
	}
	if msg.ChallengeBond < MinBond || msg.ChallengeBond > MaxGraphChallengeBondMultiple*MinBond {
		return nil, errors.New("graph challenge bond out of bounds")
	}
	graph, err := graphDescriptor(&task)
	if err != nil {
		return nil, err
	}
	if len(msg.ChallengerOutputRoots) != len(graph.Outputs) {
		return nil, errors.New("challenger output root count mismatch")
	}
	outputs := make([]canonical.Hash, len(msg.ChallengerOutputRoots))
	for i, raw := range msg.ChallengerOutputRoots {
		root, err := root32Of(raw)
		if err != nil {
			return nil, err
		}
		outputs[i] = root
	}
	challengerFinal := canonical.FinalOutputRoot(outputs)
	if equalBytes32(challengerFinal[:], task.FinalOutputRoot) {
		return nil, errors.New("challenge asserts the committed result; nothing to dispute")
	}
	if _, err := s.GetGraphDispute(ctx, task.ID); err == nil {
		return nil, errors.New("a graph challenge is already active for this task")
	}
	challengerAddr, err := addr(msg.Challenger)
	if err != nil {
		return nil, err
	}
	if err := s.bank.SendCoinsFromAccountToModule(ctx, challengerAddr, ModuleName, amount(msg.ChallengeBond)); err != nil {
		return nil, err
	}
	record := GraphDisputeRecord{
		TaskID: task.ID, Challenger: msg.Challenger,
		ChallengerPubKey:  append([]byte(nil), challengerKey...),
		Bond:              msg.ChallengeBond,
		ChallengerOutputs: cloneBytes2D(msg.ChallengerOutputRoots),
		Status:            GraphDisputeClaiming,
		ClaimDeadline:     height + ChallengeRoundBlocks,
	}
	s.consumeGraphGas(ctx, "challenge hashing", GasGraphHash*uint64(len(outputs)+2))
	if err := s.setGraphDispute(ctx, record); err != nil {
		return nil, err
	}
	task.Status = GraphStatusChallenged
	if err := s.setGraphTask(ctx, task); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "graph store", GasGraphStore)
	return &types.MsgOpenGraphChallengeResponse{ChallengeEnd: task.ChallengeEnd}, nil
}

// GraphTrailClaim locks one party's on-demand trail. The claimed initial
// state must equal the descriptor-derived committed input state, and both
// endpoint proofs must verify against the claimed trail root.
func (s msgServer) GraphTrailClaim(ctx context.Context, msg *types.MsgGraphTrailClaim) (*types.MsgGraphTrailClaimResponse, error) {
	s.consumeGraphGas(ctx, "claim base", GasGraphTxBase)
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != GraphStatusChallenged {
		return nil, errGraphWrongPhase
	}
	record, err := s.GetGraphDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	if record.Status != GraphDisputeClaiming {
		return nil, errGraphWrongPhase
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if height > record.ClaimDeadline {
		return nil, errors.New("graph trail claim window has closed")
	}
	if msg.Party != task.Worker && msg.Party != record.Challenger {
		return nil, errors.New("only the assigned worker or the challenger may claim")
	}
	graph, err := graphDescriptor(&task)
	if err != nil {
		return nil, err
	}
	claim, err := s.buildTrailClaim(graph, &task, msg)
	if err != nil {
		return nil, err
	}
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
			// Both trails end in the same state: there is no fraud to
			// adjudicate. The failed challenge forfeits its bond to the
			// worker (incentive against spam challenges).
			if err := s.bank.BurnCoins(ctx, ModuleName, amount(record.Bond)); err != nil {
				return nil, err
			}
			// The worker keeps its bond reservation until finalization.
			task.SurvivedChallenge = true
			task.Status = GraphStatusResultSubmitted
			task.ChallengeEnd = height + task.ChallengeWindow
			task.DisputeTranscriptDigest = record.TranscriptDigest
			s.deleteGraphDispute(ctx, task.ID)
			if err := s.setGraphTask(ctx, task); err != nil {
				return nil, err
			}
			return &types.MsgGraphTrailClaimResponse{}, nil
		}
		graphID, err := root32Of(task.GraphID)
		if err != nil {
			return nil, err
		}
		dispute, err := canonical.NewGraphDispute(canonical.GraphDisputeConfig{
			Graph: graph, GraphID: graphID, RoundPeriod: ChallengeRoundBlocks,
		}, toCanonicalClaim(record.WorkerClaim), toCanonicalClaim(record.ChallengerClaim), height)
		if err != nil {
			return nil, err
		}
		snapshot, err := dispute.SnapshotV1()
		if err != nil {
			return nil, err
		}
		record.Snapshot = snapshot
		record.Status = GraphDisputeBisection
		record.TranscriptDigest, err = graphTranscriptStep(record.TranscriptDigest, "claims_locked",
			map[string]any{"worker_root": hex.EncodeToString(record.WorkerClaim.Root), "challenger_root": hex.EncodeToString(record.ChallengerClaim.Root)})
		if err != nil {
			return nil, err
		}
	}
	if err := s.setGraphDispute(ctx, record); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "claim hashing", GasGraphHash*uint64(len(claim.InitialProof)+len(claim.FinalProof)+4))
	return &types.MsgGraphTrailClaimResponse{}, nil
}

// buildTrailClaim validates one party's claim against the descriptor.
func (s msgServer) buildTrailClaim(graph *canonical.GraphDescriptor, task *GraphTask, msg *types.MsgGraphTrailClaim) (GraphTrailClaimData, error) {
	graphID, err := root32Of(task.GraphID)
	if err != nil {
		return GraphTrailClaimData{}, err
	}
	expectedInitial, err := graphInitialStateRoot(graph)
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
	if !canonical.VerifyLeafInclusion(trailRoot, canonical.TrailLeaf(graphID, 0, initialRoot),
		canonical.MerkleProof{Index: 0, Count: count, Siblings: initialProof}) {
		return GraphTrailClaimData{}, errors.New("initial state proof does not match the claimed trail root")
	}
	if !canonical.VerifyLeafInclusion(trailRoot, canonical.TrailLeaf(graphID, count-1, finalRoot),
		canonical.MerkleProof{Index: count - 1, Count: count, Siblings: finalProof}) {
		return GraphTrailClaimData{}, errors.New("final state proof does not match the claimed trail root")
	}
	return GraphTrailClaimData{
		Root: trailRoot[:], InitialRoot: initialRoot[:], InitialProof: cloneBytes2D(msg.InitialProof),
		FinalRoot: finalRoot[:], FinalProof: cloneBytes2D(msg.FinalProof),
	}, nil
}

// Queue the bisection midpoint: the epoch must equal the block height so
// the dispute clock is chain-derived.
func (s msgServer) GraphMidPoint(ctx context.Context, msg *types.MsgGraphMidPoint) (*types.MsgGraphMidPointResponse, error) {
	s.consumeGraphGas(ctx, "midpoint base", GasGraphTxBase)
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != GraphStatusChallenged {
		return nil, errGraphWrongPhase
	}
	record, err := s.GetGraphDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	if record.Status != GraphDisputeBisection || len(record.Snapshot) == 0 {
		return nil, errGraphWrongPhase
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	// The epoch is the round clock, not the execution height: mempool
	// timing makes exact-height equality unusable. It may not claim the
	// distant future, and the canonical layer enforces the round window.
	if msg.Epoch > height+2 {
		return nil, errors.New("midpoint epoch is ahead of the chain clock")
	}
	graph, err := graphDescriptor(&task)
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
	dispute, err := canonical.RestoreGraphDispute(record.Snapshot, graph, ChallengeRoundBlocks)
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
	s.consumeGraphGas(ctx, "midpoint proofs", GasGraphProofSibling*uint64(len(proof))+GasGraphHash)
	if _, err := dispute.SubmitMid(party, stateRoot, proof, msg.Epoch); err != nil {
		return nil, err
	}
	snapshot, err := dispute.SnapshotV1()
	if err != nil {
		return nil, err
	}
	record.Snapshot = snapshot
	if dispute.ArbReady() {
		record.Status = GraphDisputeArbReady
	}
	if err := s.setGraphDispute(ctx, record); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "dispute store", GasGraphStore+GasGraphHash*uint64(len(snapshot)/256+1))
	return &types.MsgGraphMidPointResponse{}, nil
}

// ArbitrateGraphNode is permissionless: it adjudicates the disputed chunk
// of the first divergent node from committed evidence and settles.
func (s msgServer) ArbitrateGraphNode(ctx context.Context, msg *types.MsgArbitrateGraphNode) (*types.MsgArbitrateGraphNodeResponse, error) {
	s.consumeGraphGas(ctx, "arbitration base", GasGraphTxBase)
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != GraphStatusChallenged {
		return nil, errGraphWrongPhase
	}
	record, err := s.GetGraphDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	if record.Status != GraphDisputeArbReady || len(record.Snapshot) == 0 {
		return nil, errGraphWrongPhase
	}
	graph, err := graphDescriptor(&task)
	if err != nil {
		return nil, err
	}
	dispute, err := canonical.RestoreGraphDispute(record.Snapshot, graph, ChallengeRoundBlocks)
	if err != nil {
		return nil, err
	}
	nodeID, err := dispute.FirstDivergentNode()
	if err != nil {
		return nil, err
	}
	node := graph.Nodes[nodeID]

	if len(msg.Evidence) == 0 || len(msg.Evidence) > MaxGraphEvidenceChunks {
		return nil, errors.New("graph evidence count out of bounds")
	}
	s.consumeGraphGas(ctx, "arbitration evidence", GasGraphEvidence*uint64(len(msg.Evidence)))
	evidence := make([]canonical.ChunkEvidence, 0, len(msg.Evidence))
	inputStateRoot := dispute.LowStateRoot()
	for _, ev := range msg.Evidence {
		var desc canonical.TensorDescriptor
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
		if err := verifyGraphEvidenceState(graph, nodeID, inputStateRoot, ref, root, ev.StateProof); err != nil {
			return nil, err
		}
		s.consumeGraphGas(ctx, "evidence state leaf", GasGraphStateLeaf+GasGraphProofSibling*uint64(len(ev.StateProof)))
		evidence = append(evidence, canonical.ChunkEvidence{
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
		tableDesc := canonical.NewDesc(canonical.DtypeQ12_20, int64(len(table)))
		tableRoot, err := canonical.TensorRootOf(tableDesc, table)
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
	outcome, err := canonical.ArbitrateNodeChunk(graph, node, inputStateRoot,
		canonical.NodeChunkClaim{OutRoot: workerRoot, ChunkIndex: msg.WorkerChunkIndex, ChunkBytes: msg.WorkerChunk, Proof: workerProof},
		canonical.NodeChunkClaim{OutRoot: challengerRoot, ChunkIndex: msg.ChallengerChunkIndex, ChunkBytes: msg.ChallengerChunk, Proof: challengerProof},
		evidence, ropeTable)
	if err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "arbiter witness", GasGraphArbiterUnit*uint64(canonicalExpectedWitness(node)))
	record.TranscriptDigest, err = graphTranscriptStep(record.TranscriptDigest, "node_arbitration",
		map[string]any{"node": nodeID, "operator": node.OperatorID, "outcome": uint64(outcome)})
	if err != nil {
		return nil, err
	}
	if _, err := s.resolveGraphOutcome(ctx, &task, &record, outcome); err != nil {
		return nil, err
	}
	return &types.MsgArbitrateGraphNodeResponse{Outcome: canonicalOutcomeName(outcome)}, nil
}

// canonicalExpectedWitness bounds the charged arbiter witness.
func canonicalExpectedWitness(node canonical.GraphNode) int {
	units, err := canonical.ExpectedArbiterUnits(node)
	if err != nil || units < 0 {
		return 0
	}
	return units
}

func canonicalOutcomeName(o canonical.Outcome) string {
	switch o {
	case canonical.WorkerWins:
		return "worker_wins"
	case canonical.ChallengerWins:
		return "challenger_wins"
	case canonical.BothInvalid:
		return "both_invalid"
	default:
		return "pending"
	}
}

// FinalizeGraphTask settles a task: the optimistic path after an
// unchallenged window, the refund path for expired unaccepted tasks, and
// the timeout path for stalled disputes.
func (s msgServer) FinalizeGraphTask(ctx context.Context, msg *types.MsgFinalizeGraphTask) (*types.MsgFinalizeGraphTaskResponse, error) {
	s.consumeGraphGas(ctx, "finalize base", GasGraphTxBase)
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	switch task.Status {
	case GraphStatusPosted, GraphStatusAssigned:
		if height <= task.IssuedHeight+task.ChallengeWindow {
			return nil, errors.New("graph task has not expired")
		}
		if err := s.releaseGraphReservation(ctx, &task); err != nil {
			return nil, err
		}
		if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, mustAddr(task.Requester), amount(task.MaxFee)); err != nil {
			return nil, err
		}
		task.Status = GraphStatusRefunded
		if err := s.setGraphTask(ctx, task); err != nil {
			return nil, err
		}
		return &types.MsgFinalizeGraphTaskResponse{}, nil
	case GraphStatusChallenged:
		record, err := s.GetGraphDispute(ctx, task.ID)
		if err != nil {
			return nil, err
		}
		if record.Status == GraphDisputeArbReady {
			return nil, errors.New("the dispute is ready for arbitration, not finalization")
		}
		if len(record.Snapshot) == 0 {
			if height <= record.ClaimDeadline {
				return nil, errors.New("the claim window has not closed")
			}
			switch {
			case len(record.WorkerClaim.Root) == 0 && len(record.ChallengerClaim.Root) == 0:
				// Nobody substantiated anything: the challenge fails.
				if err := s.bank.BurnCoins(ctx, ModuleName, amount(record.Bond)); err != nil {
					return nil, err
				}
				task.Status = GraphStatusResultSubmitted
				task.ChallengeEnd = height + task.ChallengeWindow
				s.deleteGraphDispute(ctx, task.ID)
				if err := s.setGraphTask(ctx, task); err != nil {
					return nil, err
				}
				return &types.MsgFinalizeGraphTaskResponse{}, nil
			case len(record.WorkerClaim.Root) == 0:
				// The worker defaulted on its trail: the challenger wins.
				return s.resolveGraphOutcome(ctx, &task, &record, canonical.ChallengerWins)
			default:
				// The challenger defaulted: the challenge fails.
				if err := s.bank.BurnCoins(ctx, ModuleName, amount(record.Bond)); err != nil {
					return nil, err
				}
				task.SurvivedChallenge = true
				task.Status = GraphStatusResultSubmitted
				task.ChallengeEnd = height + task.ChallengeWindow
				s.deleteGraphDispute(ctx, task.ID)
				if err := s.setGraphTask(ctx, task); err != nil {
					return nil, err
				}
				return &types.MsgFinalizeGraphTaskResponse{}, nil
			}
		}
		graph, err := graphDescriptor(&task)
		if err != nil {
			return nil, err
		}
		dispute, err := canonical.RestoreGraphDispute(record.Snapshot, graph, ChallengeRoundBlocks)
		if err != nil {
			return nil, err
		}
		if height <= dispute.DisputeDeadline() {
			return nil, errors.New("the dispute round has not timed out")
		}
		outcome, err := dispute.Timeout(height)
		if err != nil {
			return nil, err
		}
		record.TranscriptDigest, err = graphTranscriptStep(record.TranscriptDigest, "dispute_timeout",
			map[string]any{"outcome": uint64(outcome), "height": height})
		if err != nil {
			return nil, err
		}
		return s.resolveGraphOutcome(ctx, &task, &record, outcome)
	case GraphStatusResultSubmitted:
		if height <= task.ChallengeEnd {
			return nil, errors.New("the challenge window has not closed")
		}
		return s.finalizeGraphOptimistic(ctx, &task)
	default:
		return nil, errGraphWrongPhase
	}
}

// finalizeGraphOptimistic settles an unchallenged (or survived) result,
// derives the receipt and pays the feeSplit.
func (s msgServer) finalizeGraphOptimistic(ctx context.Context, task *GraphTask) (*types.MsgFinalizeGraphTaskResponse, error) {
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	graph, err := graphDescriptor(task)
	if err != nil {
		return nil, err
	}
	work, err := canonical.GraphWorkVector(graph)
	if err != nil {
		return nil, err
	}
	outputs := make([]canonical.Hash, len(task.OutputRoots))
	for i, raw := range task.OutputRoots {
		root, err := root32Of(raw)
		if err != nil {
			return nil, err
		}
		outputs[i] = root
	}
	mode := canonical.ModeOptimisticUnchallenged
	if task.SurvivedChallenge {
		mode = canonical.ModeChallengedWorkerWon
	}
	var taskRef [8]byte
	binary.BigEndian.PutUint64(taskRef[:], task.ID)
	receipt, err := canonical.BuildVerifiedGraphWorkReceiptV1(graph,
		taskRef[:], task.AssignmentRef, task.WorkerProtocolPubKey, outputs, work,
		mode, height, []byte("graph:"+fmt.Sprint(task.ID)), task.DisputeTranscriptDigest)
	if err != nil {
		return nil, err
	}
	receiptID, err := receipt.ReceiptID()
	if err != nil {
		return nil, err
	}
	if len(s.store(ctx).Get(graphReceiptKey(receiptID[:]))) != 0 {
		return nil, errors.New("graph receipt already exists")
	}
	burn, _, workerShare := feeSplit(task.MaxFee)
	if burn > 0 {
		if err := s.bank.BurnCoins(ctx, ModuleName, amount(burn)); err != nil {
			return nil, err
		}
	}
	if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, mustAddr(task.Worker), amount(workerShare)); err != nil {
		return nil, err
	}
	if err := s.releaseGraphReservation(ctx, task); err != nil {
		return nil, err
	}
	receiptJSON, err := json.Marshal(receipt)
	if err != nil {
		return nil, err
	}
	s.store(ctx).Set(graphReceiptKey(receiptID[:]), receiptJSON)
	task.ReceiptID = receiptID[:]
	task.Status = GraphStatusFinalized
	if err := s.setGraphTask(ctx, *task); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "receipt", GasGraphReceipt+GasGraphHash*uint64(len(receiptJSON)/256+1))
	return &types.MsgFinalizeGraphTaskResponse{ReceiptId: receiptID[:]}, nil
}

// resolveGraphOutcome applies a finished dispute outcome to the task and
// its escrow/bond economics, mirroring the bounded-VM challenge rules.
func (s msgServer) resolveGraphOutcome(ctx context.Context, task *GraphTask, record *GraphDisputeRecord, outcome canonical.Outcome) (*types.MsgFinalizeGraphTaskResponse, error) {
	if outcome == canonical.Pending {
		return nil, errors.New("graph dispute is still pending")
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	challenger := mustAddr(record.Challenger)
	record.Outcome = canonicalOutcomeName(outcome)
	if outcome == canonical.WorkerWins {
		// Burning the losing challenger's bond prevents a worker-controlled
		// challenger from recovering the cost through the worker account.
		if err := s.bank.BurnCoins(ctx, ModuleName, amount(record.Bond)); err != nil {
			return nil, err
		}
		task.SurvivedChallenge = true
		task.DisputeTranscriptDigest = append([]byte(nil), record.TranscriptDigest...)
		task.Status = GraphStatusResultSubmitted
		task.ChallengeEnd = height + task.ChallengeWindow
		s.deleteGraphDispute(ctx, task.ID)
		if err := s.setGraphTask(ctx, *task); err != nil {
			return nil, err
		}
		return &types.MsgFinalizeGraphTaskResponse{}, nil
	}
	// ChallengerWins or BothInvalid: refund the requester escrow, release
	// the worker reservation and settle bonds.
	requester := mustAddr(task.Requester)
	if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, requester, amount(task.MaxFee)); err != nil {
		return nil, err
	}
	penaltyCap := task.ReservedBond
	if err := s.releaseGraphReservation(ctx, task); err != nil {
		return nil, err
	}
	if outcome == canonical.ChallengerWins {
		if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, challenger, amount(record.Bond)); err != nil {
			return nil, err
		}
	} else if err := s.bank.BurnCoins(ctx, ModuleName, amount(record.Bond)); err != nil {
		return nil, err
	}
	bond := s.GetBond(ctx, task.Worker)
	slash := bond / 10
	if slash > penaltyCap {
		slash = penaltyCap
	}
	if slash > 0 {
		if outcome == canonical.ChallengerWins {
			if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, challenger, amount(slash)); err != nil {
				return nil, err
			}
		} else if err := s.bank.BurnCoins(ctx, ModuleName, amount(slash)); err != nil {
			return nil, err
		}
		s.setBond(ctx, task.Worker, bond-slash)
	}
	task.Status = GraphStatusFraud
	task.DisputeTranscriptDigest = append([]byte(nil), record.TranscriptDigest...)
	if err := s.setGraphTask(ctx, *task); err != nil {
		return nil, err
	}
	s.deleteGraphDispute(ctx, task.ID)
	return &types.MsgFinalizeGraphTaskResponse{}, nil
}

func (s msgServer) releaseGraphReservation(ctx context.Context, task *GraphTask) error {
	if task.ReservedBond == 0 {
		return nil
	}
	reserved := s.getReserved(ctx, task.Worker)
	if reserved < task.ReservedBond {
		return errors.New("worker bond reservation invariant broken")
	}
	s.setReserved(ctx, task.Worker, reserved-task.ReservedBond)
	task.ReservedBond = 0
	return nil
}

// --- helpers ---------------------------------------------------------------

func cloneBytes2D(in [][]byte) [][]byte {
	out := make([][]byte, len(in))
	for i, raw := range in {
		out[i] = append([]byte(nil), raw...)
	}
	return out
}

func hashesFrom(raw [][]byte) ([]canonical.Hash, error) {
	out := make([]canonical.Hash, len(raw))
	for i, b := range raw {
		h, err := root32Of(b)
		if err != nil {
			return nil, err
		}
		out[i] = h
	}
	return out, nil
}

func toCanonicalClaim(claim GraphTrailClaimData) canonical.TrailClaim {
	root, _ := root32Of(claim.Root)
	initial, _ := root32Of(claim.InitialRoot)
	final, _ := root32Of(claim.FinalRoot)
	initialProof, _ := hashesFrom(claim.InitialProof)
	finalProof, _ := hashesFrom(claim.FinalProof)
	return canonical.TrailClaim{
		Root: root, InitialRoot: initial, InitialProof: initialProof,
		FinalRoot: final, FinalProof: finalProof,
	}
}
