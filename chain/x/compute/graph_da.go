package compute

// GRAPH_VERIFICATION_BUNDLE_V2 availability on chain (roadmap A4): typed
// provider attestations, a 2-of-3-style quorum gate on finalization, an
// objective typed-chunk challenge path, an objective timeout penalty and
// the availability-failed refund.  Providers only ever prove storage; they
// never execute the Transformer.

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"prismachain/chain/x/compute/types"
	"prismachain/compute/canonical"
)

const (
	GraphDAStatusPending = "pending"
	GraphDAStatusReady   = "ready"
	GraphDAStatusFailed  = "availability_failed"

	GraphDAChallengeOpen     = "open"
	GraphDAChallengeAnswered = "answered"
	GraphDAChallengeTimedOut = "timed_out"

	// GraphStatusAvailabilityFailed is the terminal state of a task whose
	// DA quorum was never restored.
	GraphStatusAvailabilityFailed = "availability_failed"
)

type GraphDARecord struct {
	TaskID         uint64                     `json:"task_id"`
	Attesters      map[string]json.RawMessage `json:"attesters"`
	Status         string                     `json:"status"`
	OpenChallenges uint32                     `json:"open_challenges"`
}

type GraphDAChallenge struct {
	ID         uint64 `json:"id"`
	TaskID     uint64 `json:"task_id"`
	Provider   string `json:"provider"`
	NodeID     uint32 `json:"node_id"`
	ChunkIndex uint32 `json:"chunk_index"`
	Challenger string `json:"challenger"`
	Bond       uint64 `json:"bond"`
	Deadline   uint64 `json:"deadline"`
	Status     string `json:"status"`
}

func graphDAKey(taskID uint64) []byte {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], taskID)
	return append([]byte("graphda:"), b[:]...)
}

func graphDAChallengeKey(id uint64) []byte {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], id)
	return append([]byte("graphdachallenge:"), b[:]...)
}

func (k Keeper) GetGraphDARecord(ctx context.Context, taskID uint64) (GraphDARecord, error) {
	raw := k.store(ctx).Get(graphDAKey(taskID))
	if len(raw) == 0 {
		return GraphDARecord{}, errors.New("graph DA record not found")
	}
	var rec GraphDARecord
	if err := json.Unmarshal(raw, &rec); err != nil {
		return GraphDARecord{}, err
	}
	return rec, nil
}

func (s msgServer) setGraphDARecord(ctx context.Context, rec GraphDARecord) error {
	raw, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	s.store(ctx).Set(graphDAKey(rec.TaskID), raw)
	return nil
}

func (k Keeper) GetGraphDAChallenge(ctx context.Context, id uint64) (GraphDAChallenge, error) {
	raw := k.store(ctx).Get(graphDAChallengeKey(id))
	if len(raw) == 0 {
		return GraphDAChallenge{}, errors.New("graph DA challenge not found")
	}
	var rec GraphDAChallenge
	if err := json.Unmarshal(raw, &rec); err != nil {
		return GraphDAChallenge{}, err
	}
	return rec, nil
}

func (s msgServer) setGraphDAChallenge(ctx context.Context, rec GraphDAChallenge) error {
	raw, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	s.store(ctx).Set(graphDAChallengeKey(rec.ID), raw)
	return nil
}

func (s msgServer) nextGraphDAChallengeID(ctx context.Context) (uint64, error) {
	store := s.store(ctx)
	idBytes := store.Get([]byte("graphdachallenge:next"))
	var id uint64 = 1
	if len(idBytes) == 8 {
		id = binary.BigEndian.Uint64(idBytes)
	}
	if id == ^uint64(0) {
		return 0, errors.New("graph DA challenge id exhausted")
	}
	var next [8]byte
	binary.BigEndian.PutUint64(next[:], id+1)
	store.Set([]byte("graphdachallenge:next"), next[:])
	return id, nil
}

// validGraphDAAttesters collects the attesters whose signed promises bind
// the committed graph identity, are backed by their bonded network key,
// and cover the requested height.
func (s msgServer) validGraphDAAttesters(ctx context.Context, task *GraphTask,
	requiredUntil uint64, height uint64) map[string]*canonical.GraphDAAttestationV2 {
	out := map[string]*canonical.GraphDAAttestationV2{}
	record, err := s.GetGraphDARecord(ctx, task.ID)
	if err != nil {
		return out
	}
	for provider, raw := range record.Attesters {
		var att canonical.GraphDAAttestationV2
		if err := json.Unmarshal(raw, &att); err != nil {
			continue
		}
		if err := canonical.ValidateGraphDAAttestationV2(&att); err != nil {
			continue
		}
		if !equalBytesMsg(att.GraphID, task.GraphID) ||
			!equalBytesMsg(att.ManifestRootV2, task.NodeOutputManifestRoot) ||
			!equalBytesMsg(att.FinalOutputRoot, task.FinalOutputRoot) {
			continue
		}
		daProvider, err := s.GetDAProvider(ctx, provider)
		if err != nil || !daProvider.Active {
			continue
		}
		if !equalBytesMsg(att.ProviderPubKey, daProvider.NetworkPubKey) {
			continue
		}
		if !canonical.VerifyGraphDAAttestationV2Signature(&att) {
			continue
		}
		if att.AvailableUntilHeight < requiredUntil || att.AttestedHeight > height {
			continue
		}
		out[provider] = &att
	}
	return out
}

// SubmitGraphDAAttestation (A4-01/A4-02): validates the typed attestation
// against chain state and the worker's locked commitments.
func (s msgServer) SubmitGraphDAAttestation(ctx context.Context, msg *types.MsgSubmitGraphDAAttestation) (*types.MsgSubmitGraphDAAttestationResponse, error) {
	s.consumeGEMMGas(ctx, "graph da attestation", GasDAAttestation)
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	if !graphTaskProtocolV2(&task) {
		return nil, errors.New("graph DA attestation is only available for CANONICAL_GRAPH_V2 tasks")
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if task.Status != GraphStatusResultSubmitted && task.Status != GraphStatusChallenged {
		return nil, errors.New("graph DA attestation not available for this task state")
	}
	provider, err := s.GetDAProvider(ctx, msg.Provider)
	if err != nil {
		return nil, err
	}
	if !provider.Active {
		return nil, errors.New("da provider is inactive")
	}
	if msg.Provider == task.Worker {
		return nil, errors.New("the worker cannot be its own DA provider")
	}
	var att canonical.GraphDAAttestationV2
	if err := json.Unmarshal(msg.AttestationJson, &att); err != nil {
		return nil, errors.New("graph DA attestation decode failed")
	}
	if err := canonical.ValidateGraphDAAttestationV2(&att); err != nil {
		return nil, err
	}
	if att.GraphTaskID != task.ID {
		return nil, errors.New("graph DA attestation binds a different task")
	}
	if !equalBytesMsg(att.GraphID, task.GraphID) ||
		!equalBytesMsg(att.ManifestRootV2, task.NodeOutputManifestRoot) ||
		!equalBytesMsg(att.FinalOutputRoot, task.FinalOutputRoot) {
		return nil, errors.New("graph DA attestation does not match the committed result")
	}
	if !equalBytesMsg(att.ProviderPubKey, provider.NetworkPubKey) {
		return nil, errGEMMIdentity
	}
	if att.AttestedHeight > height {
		return nil, errors.New("graph DA attestation height is in the future")
	}
	if !canonical.VerifyGraphDAAttestationV2Signature(&att) {
		return nil, errors.New("graph DA attestation signature is invalid")
	}
	record, err := s.GetGraphDARecord(ctx, task.ID)
	if err != nil {
		record = GraphDARecord{TaskID: task.ID, Attesters: map[string]json.RawMessage{},
			Status: GraphDAStatusPending}
	}
	if record.Attesters == nil {
		record.Attesters = map[string]json.RawMessage{}
	}
	if prior, ok := record.Attesters[msg.Provider]; ok {
		var previous canonical.GraphDAAttestationV2
		if err := json.Unmarshal(prior, &previous); err == nil &&
			att.AvailableUntilHeight < previous.AvailableUntilHeight {
			return nil, errors.New("graph DA availability window must not shrink")
		}
	}
	record.Attesters[msg.Provider] = json.RawMessage(append([]byte(nil), msg.AttestationJson...))
	requiredUntil := task.ResultSubmittedHeight + task.ChallengeWindow + DAWindowBlocks
	if len(s.validGraphDAAttesters(ctx, &task, requiredUntil, height)) >= canonical.GraphDARequiredReplicas {
		record.Status = GraphDAStatusReady
	}
	if err := s.setGraphDARecord(ctx, record); err != nil {
		return nil, err
	}
	return &types.MsgSubmitGraphDAAttestationResponse{}, nil
}

// FailGraphAvailability (A4-06): after the DA window closed with the
// quorum still unmet, refund the requester; never a receipt, never pay the
// worker.
func (s msgServer) FailGraphAvailability(ctx context.Context, msg *types.MsgFailGraphAvailability) (*types.MsgFailGraphAvailabilityResponse, error) {
	if _, err := addr(msg.Actor); err != nil {
		return nil, err
	}
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	if !graphTaskProtocolV2(&task) {
		return nil, errors.New("availability failure is only available for CANONICAL_GRAPH_V2 tasks")
	}
	if task.Status != GraphStatusResultSubmitted {
		return nil, errGraphWrongPhase
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	requiredUntil := task.ResultSubmittedHeight + task.ChallengeWindow + DAWindowBlocks
	if height <= requiredUntil {
		return nil, errors.New("graph DA window has not closed")
	}
	if len(s.validGraphDAAttesters(ctx, &task, requiredUntil, height)) >= canonical.GraphDARequiredReplicas {
		return nil, errors.New("graph DA quorum is available; availability failure refused")
	}
	if err := s.releaseGraphReservation(ctx, &task); err != nil {
		return nil, err
	}
	if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, mustAddr(task.Requester), amount(task.MaxFee)); err != nil {
		return nil, err
	}
	if record, err := s.GetGraphDARecord(ctx, task.ID); err == nil {
		record.Status = GraphDAStatusFailed
		if err := s.setGraphDARecord(ctx, record); err != nil {
			return nil, err
		}
	}
	task.Status = GraphStatusAvailabilityFailed
	if err := s.setGraphTask(ctx, task); err != nil {
		return nil, err
	}
	return &types.MsgFailGraphAvailabilityResponse{}, nil
}

// OpenGraphDAChallenge (A4-04): objective on-chain availability challenge
// against one registered provider and one committed node chunk.
func (s msgServer) OpenGraphDAChallenge(ctx context.Context, msg *types.MsgOpenGraphDAChallenge) (*types.MsgOpenGraphDAChallengeResponse, error) {
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	if !graphTaskProtocolV2(&task) {
		return nil, errors.New("graph DA challenge is only available for CANONICAL_GRAPH_V2 tasks")
	}
	if task.Status != GraphStatusResultSubmitted {
		return nil, errGraphWrongPhase
	}
	if msg.Challenger == msg.Provider {
		return nil, errors.New("a provider cannot challenge itself")
	}
	provider, err := s.GetDAProvider(ctx, msg.Provider)
	if err != nil || !provider.Active {
		return nil, errors.New("da provider is not registered and active")
	}
	graph, err := graphDescriptorV2(&task)
	if err != nil {
		return nil, err
	}
	if int(msg.NodeId) >= len(graph.Nodes) {
		return nil, errors.New("graph DA challenge node id out of range")
	}
	node := graph.Nodes[msg.NodeId]
	elems := int64(1)
	for _, d := range node.Output.Shape {
		elems *= d
	}
	count := uint32((elems + canonical.ChunkElems - 1) / canonical.ChunkElems)
	if count == 0 {
		count = 1
	}
	if msg.ChunkIndex >= count {
		return nil, errors.New("graph DA challenge chunk index out of range")
	}
	if msg.Bond < MinBond || msg.Bond > MaxGraphChallengeBondMultiple*MinBond {
		return nil, errors.New("graph DA challenge bond out of bounds")
	}
	record, err := s.GetGraphDARecord(ctx, task.ID)
	if err != nil || record.OpenChallenges >= 1 {
		return nil, errors.New("a graph DA challenge is already open for this task")
	}
	challengerAddr, err := addr(msg.Challenger)
	if err != nil {
		return nil, err
	}
	if err := s.bank.SendCoinsFromAccountToModule(ctx, challengerAddr, ModuleName, amount(msg.Bond)); err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	id, err := s.nextGraphDAChallengeID(ctx)
	if err != nil {
		return nil, err
	}
	challenge := GraphDAChallenge{
		ID: id, TaskID: task.ID, Provider: msg.Provider, NodeID: msg.NodeId,
		ChunkIndex: msg.ChunkIndex, Challenger: msg.Challenger, Bond: msg.Bond,
		Deadline: height + ChallengeRoundBlocks, Status: GraphDAChallengeOpen,
	}
	if err := s.setGraphDAChallenge(ctx, challenge); err != nil {
		return nil, err
	}
	record.OpenChallenges++
	if err := s.setGraphDARecord(ctx, record); err != nil {
		return nil, err
	}
	return &types.MsgOpenGraphDAChallengeResponse{ChallengeId: id, Deadline: challenge.Deadline}, nil
}

// RespondGraphDAChallenge (A4-04): the provider answers with the typed
// chunk, its TensorRootV2 proof and the node's manifest inclusion proof.
func (s msgServer) RespondGraphDAChallenge(ctx context.Context, msg *types.MsgRespondGraphDAChallenge) (*types.MsgRespondGraphDAChallengeResponse, error) {
	challenge, err := s.GetGraphDAChallenge(ctx, msg.ChallengeId)
	if err != nil {
		return nil, err
	}
	if challenge.Status != GraphDAChallengeOpen {
		return nil, errors.New("graph DA challenge is already resolved")
	}
	if msg.Provider != challenge.Provider {
		return nil, errors.New("only the challenged provider may respond")
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if height > challenge.Deadline {
		return nil, errors.New("graph DA challenge deadline has passed")
	}
	task, err := s.GetGraphTask(ctx, challenge.TaskID)
	if err != nil {
		return nil, err
	}
	graph, err := graphDescriptorV2(&task)
	if err != nil {
		return nil, err
	}
	node := graph.Nodes[challenge.NodeID]
	var desc canonical.TensorDescriptorV2
	if err := json.Unmarshal(msg.NodeDescJson, &desc); err != nil {
		return nil, errors.New("malformed node descriptor evidence")
	}
	if !sameDescV2Local(desc, node.Output) {
		return nil, errors.New("node descriptor evidence does not match the committed descriptor")
	}
	if msg.ChunkIndex != challenge.ChunkIndex {
		return nil, errors.New("response chunk index does not match the challenge")
	}
	graphID, err := root32Of(task.GraphID)
	if err != nil {
		return nil, err
	}
	nodeRoot, err := root32Of(msg.NodeRoot)
	if err != nil {
		return nil, err
	}
	manifestProof, err := hashesFrom(msg.ManifestProof)
	if err != nil {
		return nil, err
	}
	if !canonical.VerifyNodeOutputManifestLeafV2(root32Must(task.NodeOutputManifestRoot),
		graphID, node, nodeRoot, canonical.MerkleProof{
			Index: node.NodeID, Count: uint32(len(graph.Nodes)), Siblings: manifestProof}) {
		return nil, errors.New("node output root is not part of the committed manifest")
	}
	chunkProof, err := hashesFrom(msg.ChunkProof)
	if err != nil {
		return nil, err
	}
	if !canonical.VerifyChunkV2(nodeRoot, desc, msg.ChunkIndex, msg.ChunkCount, msg.Chunk, chunkProof) {
		return nil, errors.New("chunk does not match the committed node output root")
	}
	challenge.Status = GraphDAChallengeAnswered
	if err := s.setGraphDAChallenge(ctx, challenge); err != nil {
		return nil, err
	}
	// return the challenger's bond and clear the open-challenge counter
	if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, mustAddr(challenge.Challenger), amount(challenge.Bond)); err != nil {
		return nil, err
	}
	if record, err := s.GetGraphDARecord(ctx, challenge.TaskID); err == nil && record.OpenChallenges > 0 {
		record.OpenChallenges--
		if err := s.setGraphDARecord(ctx, record); err != nil {
			return nil, err
		}
	}
	return &types.MsgRespondGraphDAChallengeResponse{}, nil
}

// TimeoutGraphDAChallenge (A4-05): only a missed on-chain deadline is an
// objective provider failure.  HTTP failures are never fraud proofs.
func (s msgServer) TimeoutGraphDAChallenge(ctx context.Context, msg *types.MsgTimeoutGraphDAChallenge) (*types.MsgTimeoutGraphDAChallengeResponse, error) {
	if _, err := addr(msg.Actor); err != nil {
		return nil, err
	}
	challenge, err := s.GetGraphDAChallenge(ctx, msg.ChallengeId)
	if err != nil {
		return nil, err
	}
	if challenge.Status != GraphDAChallengeOpen {
		return nil, errors.New("graph DA challenge is already resolved")
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if height <= challenge.Deadline {
		return nil, errors.New("graph DA challenge deadline has not passed")
	}
	challenge.Status = GraphDAChallengeTimedOut
	if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, mustAddr(challenge.Challenger), amount(challenge.Bond)); err != nil {
		return nil, err
	}
	bond := s.GetBond(ctx, challenge.Provider)
	penalty := uint64(DAPenaltyUprsm)
	if penalty > bond {
		penalty = bond
	}
	if penalty > 0 {
		if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, mustAddr(challenge.Challenger), amount(penalty)); err != nil {
			return nil, err
		}
		s.setBond(ctx, challenge.Provider, bond-penalty)
		if provider, err := s.GetDAProvider(ctx, challenge.Provider); err == nil {
			provider.SlashedUprsm += penalty
			s.setDAProvider(ctx, provider)
		}
	}
	if record, err := s.GetGraphDARecord(ctx, challenge.TaskID); err == nil && record.OpenChallenges > 0 {
		record.OpenChallenges--
		if err := s.setGraphDARecord(ctx, record); err != nil {
			return nil, err
		}
	}
	return &types.MsgTimeoutGraphDAChallengeResponse{}, s.setGraphDAChallenge(ctx, challenge)
}

func root32Must(raw []byte) canonical.Hash {
	var h canonical.Hash
	copy(h[:], raw)
	return h
}
