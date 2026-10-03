package compute

// DA_REPLICA_V1 chain logic: provider registry, availability attestations,
// quorum accounting, on-chain sampling challenges with objective timeout
// determination, and the economics. Consensus handlers verify signatures,
// bounded bytes, Merkle proofs and heights only: no network I/O, no
// Freivalds, no output reconstruction, no floats.

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"prismachain/chain/x/compute/types"
	"prismachain/compute/gemmv1"
)

// DA availability statuses of a GEMM task.
const (
	DAStatusPending     = "availability_pending"
	DAStatusReady       = "challenge_window_ready"
	DAStatusFailed      = "availability_failed"
	DAStatusNotRequired = "not_required"
)

// DA challenge statuses.
const (
	DAChallengeOpen     = "open"
	DAChallengePassed   = "passed"
	DAChallengeFailed   = "failed"
	DAChallengeTimedOut = "timed_out"
)

// Gas units for the DA schedule (GEMMGasV1 extension).
const (
	GasDARegister         uint64 = 10_000
	GasDAAttestation      uint64 = 60_000
	GasDAOpenChallenge    uint64 = 40_000
	GasDARespond          uint64 = 45_000
	GasDATimeout          uint64 = 25_000
	GasDAFailAvailability uint64 = 25_000
)

// DAPenaltyUprsm is the fixed, versioned testnet penalty for a DA provider
// that fails an on-chain challenge by timing out.
const DAPenaltyUprsm = gemmv1.DAPenaltyUprsm

// DAWindowBlocks is the extra window after the challenge window during
// which the DA quorum must be reached or the availability challenge
// resolved, before anyone may fail the task and refund the requester.
const DAWindowBlocks uint64 = 30

// DAProvider is one permissionless, bonded availability provider.
type DAProvider struct {
	Account          string `json:"account"`
	NetworkPubKey    []byte `json:"network_pub_key"`
	RegisteredHeight uint64 `json:"registered_height"`
	Active           bool   `json:"active"`
	// SlashedUprsm accumulates DA penalties against the provider's bond.
	SlashedUprsm uint64 `json:"slashed_uprsm"`
}

// DARecord is the per-task availability accounting.
type DARecord struct {
	TaskID    uint64                     `json:"task_id"`
	Attesters map[string]json.RawMessage `json:"attesters"`
	Status    string                     `json:"status"`
	// OpenChallenges counts sampling challenges that have not resolved;
	// tasks cannot finalize while any DA challenge is open.
	OpenChallenges uint32 `json:"open_challenges"`
}

// DAChallenge is one on-chain sampling challenge against one provider.
type DAChallenge struct {
	ID           uint64 `json:"id"`
	TaskID       uint64 `json:"task_id"`
	Provider     string `json:"provider"`
	Challenger   string `json:"challenger"`
	TileI        uint32 `json:"tile_i"`
	TileJ        uint32 `json:"tile_j"`
	OpenedHeight uint64 `json:"opened_height"`
	Deadline     uint64 `json:"deadline"`
	Bond         uint64 `json:"bond"`
	Status       string `json:"status"`
}

func daProviderKey(account string) []byte { return append([]byte{'D'}, []byte(account)...) }
func daRecordKey(taskID uint64) []byte {
	var key [9]byte
	key[0] = 'A'
	binary.BigEndian.PutUint64(key[1:], taskID)
	return key[:]
}
func daChallengeKey(id uint64) []byte {
	var key [9]byte
	key[0] = 'C'
	binary.BigEndian.PutUint64(key[1:], id)
	return key[:]
}
func daChallengeTaskKey(taskID uint64, id uint64) []byte {
	var key [17]byte
	key[0] = 'c'
	binary.BigEndian.PutUint64(key[1:], taskID)
	binary.BigEndian.PutUint64(key[9:], id)
	return key[:]
}
func daOpenCountKey(provider string) []byte {
	return append([]byte{'O'}, []byte(provider)...)
}

func (k Keeper) GetDAProvider(ctx context.Context, account string) (DAProvider, error) {
	data := k.store(ctx).Get(daProviderKey(account))
	if len(data) == 0 {
		return DAProvider{}, errors.New("da provider not registered")
	}
	var provider DAProvider
	if err := json.Unmarshal(data, &provider); err != nil {
		return DAProvider{}, err
	}
	return provider, nil
}

func (k Keeper) setDAProvider(ctx context.Context, provider DAProvider) error {
	data, err := json.Marshal(provider)
	if err != nil {
		return err
	}
	k.store(ctx).Set(daProviderKey(provider.Account), data)
	return nil
}

func (k Keeper) GetDARecord(ctx context.Context, taskID uint64) (DARecord, error) {
	data := k.store(ctx).Get(daRecordKey(taskID))
	if len(data) == 0 {
		return DARecord{}, errors.New("da record not found")
	}
	var record DARecord
	if err := json.Unmarshal(data, &record); err != nil {
		return DARecord{}, err
	}
	return record, nil
}

func (k Keeper) setDARecord(ctx context.Context, record DARecord) error {
	data, err := json.Marshal(record)
	if err != nil {
		return err
	}
	k.store(ctx).Set(daRecordKey(record.TaskID), data)
	return nil
}

func (k Keeper) GetDAChallenge(ctx context.Context, id uint64) (DAChallenge, error) {
	data := k.store(ctx).Get(daChallengeKey(id))
	if len(data) == 0 {
		return DAChallenge{}, errors.New("da challenge not found")
	}
	var challenge DAChallenge
	if err := json.Unmarshal(data, &challenge); err != nil {
		return DAChallenge{}, err
	}
	return challenge, nil
}

func (k Keeper) setDAChallenge(ctx context.Context, challenge DAChallenge) error {
	data, err := json.Marshal(challenge)
	if err != nil {
		return err
	}
	store := k.store(ctx)
	store.Set(daChallengeKey(challenge.ID), data)
	store.Set(daChallengeTaskKey(challenge.TaskID, challenge.ID), data)
	return nil
}

// daOpenChallenges counts challenges against one provider that have not
// reached a terminal state (DoS bound: one open challenge per provider).
func (k Keeper) daOpenChallenges(ctx context.Context, provider string) uint32 {
	data := k.store(ctx).Get(daOpenCountKey(provider))
	if len(data) != 4 {
		return 0
	}
	return binary.BigEndian.Uint32(data)
}

func (k Keeper) setDAOpenChallenges(ctx context.Context, provider string, count uint32) {
	var data [4]byte
	binary.BigEndian.PutUint32(data[:], count)
	k.store(ctx).Set(daOpenCountKey(provider), data[:])
}

// validDAAttesters returns the distinct, signature-valid attestations for
// a task whose available_until covers the required height.
func (k Keeper) validDAAttesters(ctx context.Context, task GEMMTask, requiredUntil uint64) map[string]*gemmv1.DAAttestation {
	result := map[string]*gemmv1.DAAttestation{}
	record, err := k.GetDARecord(ctx, task.ID)
	if err != nil {
		return result
	}
	for account, raw := range record.Attesters {
		var attestation gemmv1.DAAttestation
		if err := json.Unmarshal(raw, &attestation); err != nil {
			continue
		}
		if !gemmv1.VerifyDAAttestationSignature(&attestation) {
			continue
		}
		if attestation.AvailableUntilHeight < requiredUntil {
			continue
		}
		result[account] = &attestation
	}
	return result
}

// RegisterDAProvider admits a permissionless provider that already holds a
// bonded network Ed25519 key (no second identity system).
func (s msgServer) RegisterDAProvider(ctx context.Context, msg *types.MsgRegisterDAProvider) (*types.MsgRegisterDAProviderResponse, error) {
	s.consumeGEMMGas(ctx, "da register", GasDARegister)
	networkKey, err := s.bondedNetworkKey(ctx, msg.Provider)
	if err != nil {
		return nil, err
	}
	provider := DAProvider{
		Account: msg.Provider, NetworkPubKey: append([]byte(nil), networkKey...),
		RegisteredHeight: uint64(sdk.UnwrapSDKContext(ctx).BlockHeight()), Active: true,
	}
	return &types.MsgRegisterDAProviderResponse{}, s.setDAProvider(ctx, provider)
}

// SubmitDAAttestation verifies the canonical signed attestation against
// chain state: registered provider, not the worker, key == bonded key,
// task/assignment/output_root/bytes match, no duplicates, non-decreasing
// available_until, and the provider is not currently penalized.
func (s msgServer) SubmitDAAttestation(ctx context.Context, msg *types.MsgSubmitDAAttestation) (*types.MsgSubmitDAAttestationResponse, error) {
	s.consumeGEMMGas(ctx, "da attestation", GasDAAttestation)
	task, err := s.GetGEMMTask(ctx, msg.GemmTaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if task.Status != GEMMStatusResultSubmitted && task.Status != GEMMStatusChallenged {
		return nil, errors.New("da attestation not available for this task state")
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
	if task.OutputRoot == nil {
		return nil, errors.New("task has no committed output to attest")
	}
	// The attestation CBOR is JSON-persisted chain-internally after being
	// re-encoded: decode with the protocol object shape and re-derive the
	// signature preimage through the gemmv1 helpers.
	var attestation gemmv1.DAAttestation
	if err := decodeDAAttestation(msg.AttestationJson, &attestation); err != nil {
		return nil, err
	}
	if err := gemmv1.ValidateDAAttestation(&attestation); err != nil {
		preview := msg.AttestationJson
		if len(preview) > 80 {
			preview = preview[:80]
		}
		return nil, fmt.Errorf("%w [raw_bytes=%d preview=%q]", err, len(msg.AttestationJson), string(preview))
	}
	if !equalBytesMsg(attestation.TaskID, task.ProtocolTaskID) ||
		!equalBytesMsg(attestation.AssignmentID, task.AssignmentID) ||
		!equalBytesMsg(attestation.OutputRoot, task.OutputRoot) {
		return nil, errors.New("da attestation does not match the committed task")
	}
	if !equalBytesMsg(attestation.ProviderPubKey, provider.NetworkPubKey) {
		return nil, errGEMMIdentity
	}
	if attestation.OutputBytes != task.M*task.N*4 {
		return nil, fmt.Errorf("da attestation output_bytes %d does not match M*N*4", attestation.OutputBytes)
	}
	if attestation.AttestedHeight > height {
		return nil, errors.New("da attestation height is in the future")
	}
	if !gemmv1.VerifyDAAttestationSignature(&attestation) {
		return nil, errors.New("da attestation signature is invalid")
	}
	record, err := s.GetDARecord(ctx, task.ID)
	if err != nil {
		record = DARecord{TaskID: task.ID, Attesters: map[string]json.RawMessage{}, Status: DAStatusPending}
	}
	if record.Attesters == nil {
		record.Attesters = map[string]json.RawMessage{}
	}
	if prior, ok := record.Attesters[msg.Provider]; ok {
		var previous gemmv1.DAAttestation
		if err := json.Unmarshal(prior, &previous); err == nil && attestation.AvailableUntilHeight < previous.AvailableUntilHeight {
			return nil, errors.New("da attestation cannot reduce available_until")
		}
	}
	raw, err := json.Marshal(attestation)
	if err != nil {
		return nil, err
	}
	record.Attesters[msg.Provider] = raw
	// Availability is ready once the quorum covers the mandatory retention
	// horizon (the challenge window plus the DA window).
	requiredUntil := task.ResultSubmittedHeight + task.ChallengeWindow + DAWindowBlocks
	if len(s.validDAAttesters(ctx, task, requiredUntil)) >= gemmv1.DARequiredReplicas {
		record.Status = DAStatusReady
	} else {
		record.Status = DAStatusPending
	}
	return &types.MsgSubmitDAAttestationResponse{}, s.setDARecord(ctx, record)
}

// decodeDAAttestationCBOR decodes the signed canonical object. The chain
// stores it as JSON internally (like the locked traces) and always
// re-verifies through the protocol signature helper.
func decodeDAAttestation(raw []byte, out *gemmv1.DAAttestation) error {
	if len(raw) == 0 || len(raw) > 4096 {
		return errors.New("da attestation must be 1..4096 bytes")
	}
	var asMap map[string]any
	if err := json.Unmarshal(raw, &asMap); err == nil {
		// Tolerate a JSON rendering of the same object (REST clients), but
		// the signature is still checked over the canonical preimage.
		encoded, err := json.Marshal(asMap)
		if err != nil {
			return err
		}
		return json.Unmarshal(encoded, out)
	}
	return errors.New("da attestation must be a JSON map of the canonical object")
}

// OpenGEMMDAChallenge opens one objective sampling challenge against one
// provider. The tile is derived from chain facts; the challenger cannot
// choose it, and no HTTP-failure claim is involved anywhere.
func (s msgServer) OpenGEMMDAChallenge(ctx context.Context, msg *types.MsgOpenGEMMDAChallenge) (*types.MsgOpenGEMMDAChallengeResponse, error) {
	s.consumeGEMMGas(ctx, "da open challenge", GasDAOpenChallenge)
	if _, err := addr(msg.Challenger); err != nil {
		return nil, err
	}
	task, err := s.GetGEMMTask(ctx, msg.GemmTaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if task.Status != GEMMStatusResultSubmitted && task.Status != GEMMStatusChallenged {
		return nil, errors.New("da challenge not available for this task state")
	}
	if _, err := s.GetDAProvider(ctx, msg.Provider); err != nil {
		return nil, err
	}
	record, err := s.GetDARecord(ctx, task.ID)
	if err != nil {
		return nil, errors.New("task has no DA attestations to challenge")
	}
	if _, ok := record.Attesters[msg.Provider]; !ok {
		return nil, errors.New("provider has not attested this task")
	}
	if msg.Provider == task.Worker || msg.Provider == msg.Challenger {
		return nil, errors.New("challenger must be independent of the provider")
	}
	if len(msg.Nonce) == 0 || len(msg.Nonce) > 64 {
		return nil, errors.New("da challenge nonce must be 1..64 bytes")
	}
	if msg.Bond != DAPenaltyUprsm {
		return nil, errors.New("da challenge bond must equal the versioned DA penalty")
	}
	if s.daOpenChallenges(ctx, msg.Provider) >= 1 {
		return nil, errors.New("provider already has an open DA challenge")
	}
	counts := gemmv1.TileCountsFor(task.M, task.N, task.K)
	tileI, tileJ, err := gemmv1.DAChallengeTile(task.ProtocolTaskID, []byte(msg.Provider),
		[]byte(msg.Challenger), msg.Nonce, height, uint32(counts.RowsC), uint32(counts.ColsC))
	if err != nil {
		return nil, err
	}
	if err := s.bank.SendCoinsFromAccountToModule(ctx, mustAddr(msg.Challenger), ModuleName, amount(msg.Bond)); err != nil {
		return nil, err
	}
	store := s.store(ctx)
	idBytes := store.Get([]byte("da:next"))
	var id uint64 = 1
	if len(idBytes) == 8 {
		id = binary.BigEndian.Uint64(idBytes)
	}
	var next [8]byte
	binary.BigEndian.PutUint64(next[:], id+1)
	store.Set([]byte("da:next"), next[:])
	challenge := DAChallenge{
		ID: id, TaskID: task.ID, Provider: msg.Provider, Challenger: msg.Challenger,
		TileI: tileI, TileJ: tileJ, OpenedHeight: height,
		Deadline: height + DAChallengeBlocks, Bond: msg.Bond, Status: DAChallengeOpen,
	}
	record.OpenChallenges++
	if err := s.setDARecord(ctx, record); err != nil {
		return nil, err
	}
	s.setDAOpenChallenges(ctx, msg.Provider, s.daOpenChallenges(ctx, msg.Provider)+1)
	if err := s.setDAChallenge(ctx, challenge); err != nil {
		return nil, err
	}
	s.consumeGEMMGas(ctx, "da challenge write", GasPerGEMMStoreWrite)
	return &types.MsgOpenGEMMDAChallengeResponse{
		ChallengeId: id, TileI: tileI, TileJ: tileJ, Deadline: challenge.Deadline,
	}, nil
}

// DAChallengeBlocks is the response deadline window of a DA challenge.
const DAChallengeBlocks uint64 = 20

// RespondGEMMDAChallenge verifies a 256-byte canonical tile plus an
// OutputTileProof against the worker's committed output_root.
func (s msgServer) RespondGEMMDAChallenge(ctx context.Context, msg *types.MsgRespondGEMMDAChallenge) (*types.MsgRespondGEMMDAChallengeResponse, error) {
	s.consumeGEMMGas(ctx, "da respond", GasDARespond)
	challenge, err := s.GetDAChallenge(ctx, msg.ChallengeId)
	if err != nil {
		return nil, err
	}
	if challenge.Status != DAChallengeOpen {
		return nil, errors.New("da challenge is already resolved")
	}
	if challenge.Provider != msg.Provider {
		return nil, errors.New("only the challenged provider may respond")
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if height > challenge.Deadline {
		return nil, errors.New("da challenge response deadline has passed; use the timeout path")
	}
	task, err := s.GetGEMMTask(ctx, challenge.TaskID)
	if err != nil {
		return nil, err
	}
	if len(msg.Tile) != 256 {
		return nil, errors.New("da challenge tile must be 256 canonical bytes")
	}
	leafCount, depth := gemmLeafCountFor(&task)
	if msg.ProofCount != leafCount {
		return nil, errGEMMProofShape
	}
	proof, err := gemmProofFromMsg(msg.ProofSiblings, int(msg.ProofIndex), int(msg.ProofCount), depth)
	if err != nil {
		return nil, err
	}
	s.gasForProofs(ctx, "da tile proof", 1, len(proof.Siblings))
	s.gasForStates(ctx, "da tile", 256)
	leaf := gemmv1.LeafOutputTile(task.ProtocolTaskID, task.AssignmentID, challenge.TileI, challenge.TileJ, msg.Tile)
	if !gemmv1.VerifyLeafInclusion(root32(task.OutputRoot), leaf, proof) {
		return nil, errors.New("da challenge tile is not in the committed output_root")
	}
	challenge.Status = DAChallengePassed
	s.setDAOpenChallenges(ctx, msg.Provider, s.daOpenChallenges(ctx, msg.Provider)-1)
	if record, err := s.GetDARecord(ctx, challenge.TaskID); err == nil && record.OpenChallenges > 0 {
		record.OpenChallenges--
		if err := s.setDARecord(ctx, record); err != nil {
			return nil, err
		}
	}
	// Anti-grief: a passing provider burns the challenger's bond.
	if err := s.bank.BurnCoins(ctx, ModuleName, amount(challenge.Bond)); err != nil {
		return nil, err
	}
	return &types.MsgRespondGEMMDAChallengeResponse{}, s.setDAChallenge(ctx, challenge)
}

// TimeoutGEMMDAChallenge is the objective, chain-verified failure path: a
// provider that missed its deadline loses a fixed penalty from its bond and
// the challenger's bond is returned. No HTTP failure is ever involved.
func (s msgServer) TimeoutGEMMDAChallenge(ctx context.Context, msg *types.MsgTimeoutGEMMDAChallenge) (*types.MsgTimeoutGEMMDAChallengeResponse, error) {
	s.consumeGEMMGas(ctx, "da timeout", GasDATimeout)
	if _, err := addr(msg.Actor); err != nil {
		return nil, err
	}
	challenge, err := s.GetDAChallenge(ctx, msg.ChallengeId)
	if err != nil {
		return nil, err
	}
	if challenge.Status != DAChallengeOpen {
		return nil, errors.New("da challenge is already resolved")
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if height <= challenge.Deadline {
		return nil, errors.New("da challenge deadline has not passed")
	}
	challenge.Status = DAChallengeTimedOut
	s.setDAOpenChallenges(ctx, challenge.Provider, s.daOpenChallenges(ctx, challenge.Provider)-1)
	if record, err := s.GetDARecord(ctx, challenge.TaskID); err == nil && record.OpenChallenges > 0 {
		record.OpenChallenges--
		if err := s.setDARecord(ctx, record); err != nil {
			return nil, err
		}
	}
	// Return the challenger's bond, slash the provider's bond by the fixed
	// penalty and pay it to the challenger as the objective reward.
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
	// DA quorum may have dropped: recompute availability status.
	task, err := s.GetGEMMTask(ctx, challenge.TaskID)
	if err == nil {
		if record, err := s.GetDARecord(ctx, task.ID); err == nil {
			requiredUntil := task.ResultSubmittedHeight + task.ChallengeWindow + DAWindowBlocks
			if len(s.validDAAttesters(ctx, task, requiredUntil)) >= gemmv1.DARequiredReplicas {
				record.Status = DAStatusReady
			} else {
				record.Status = DAStatusPending
			}
			s.setDARecord(ctx, record)
		}
	}
	return &types.MsgTimeoutGEMMDAChallengeResponse{}, s.setDAChallenge(ctx, challenge)
}

// FailGEMMAvailability routes a task whose DA quorum was never restored to
// the requester refund path. It can only run after the DA window closed,
// can never mint a receipt, and never pays the worker.
func (s msgServer) FailGEMMAvailability(ctx context.Context, msg *types.MsgFailGEMMAvailability) (*types.MsgFailGEMMAvailabilityResponse, error) {
	s.consumeGEMMGas(ctx, "da fail availability", GasDAFailAvailability)
	if _, err := addr(msg.Actor); err != nil {
		return nil, err
	}
	task, err := s.GetGEMMTask(ctx, msg.GemmTaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if task.Status != GEMMStatusResultSubmitted {
		return nil, errors.New("task is not in the availability window")
	}
	if height <= task.ResultSubmittedHeight+task.ChallengeWindow+DAWindowBlocks {
		return nil, errors.New("DA window has not closed yet")
	}
	requiredUntil := task.ResultSubmittedHeight + task.ChallengeWindow + DAWindowBlocks
	if len(s.validDAAttesters(ctx, task, requiredUntil)) >= gemmv1.DARequiredReplicas {
		return nil, errors.New("DA quorum is satisfied; use finalize")
	}
	if record, err := s.GetDARecord(ctx, task.ID); err == nil {
		record.Status = DAStatusFailed
		s.setDARecord(ctx, record)
	}
	if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, mustAddr(task.Requester), amount(task.MaxFee)); err != nil {
		return nil, err
	}
	if err := s.releaseGEMMReservation(ctx, &task); err != nil {
		return nil, err
	}
	task.Status = GEMMStatusRefunded
	return &types.MsgFailGEMMAvailabilityResponse{}, s.setGEMMTask(ctx, task)
}

// gemmDAStatus summarizes availability for queries and finalize checks.
func (s msgServer) gemmDAStatus(ctx context.Context, task GEMMTask) (string, uint32, uint32) {
	if task.OutputRoot == nil {
		return DAStatusNotRequired, 0, 0
	}
	requiredUntil := task.ResultSubmittedHeight + task.ChallengeWindow + DAWindowBlocks
	valid := uint32(len(s.validDAAttesters(ctx, task, requiredUntil)))
	open := uint32(0)
	if record, err := s.GetDARecord(ctx, task.ID); err == nil {
		if record.Status == DAStatusFailed {
			return DAStatusFailed, valid, open
		}
	}
	if valid >= gemmv1.DARequiredReplicas {
		return DAStatusReady, valid, open
	}
	return DAStatusPending, valid, open
}

// daFinalizeReady is the DA precondition of FinalizeGEMM: quorum reached,
// no unresolved sampling challenge and no prior availability failure.
func (s msgServer) daFinalizeReady(ctx context.Context, task GEMMTask) error {
	status, valid, _ := s.gemmDAStatus(ctx, task)
	if valid < gemmv1.DARequiredReplicas {
		return fmt.Errorf("da quorum not reached: %d/%d valid attestations", valid, gemmv1.DARequiredReplicas)
	}
	if status == DAStatusFailed {
		return errors.New("task availability already failed")
	}
	if record, err := s.GetDARecord(ctx, task.ID); err == nil && record.OpenChallenges > 0 {
		return errors.New("unresolved DA challenge blocks finalization")
	}
	return nil
}
