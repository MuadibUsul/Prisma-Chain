package compute

// GEMM task admission, escrow, assignment and result submission. Sender
// and protocol identity are double-bound (invariant): the Cosmos tx signer
// must be the on-chain party AND the protocol signature must verify under
// the bonded network Ed25519 key.

import (
	"context"
	"crypto/ed25519"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"prismachain/chain/x/compute/types"
	"prismachain/compute/gemmv1"
)

var (
	errGEMMTaskNotFound    = errors.New("gemm task not found")
	errGEMMDisputeNotFound = errors.New("gemm dispute not found")
	errGEMMIdentity        = errors.New("gemm: tx sender and protocol identity are not bound")
)

// verifyGEMMRequesterKeyProof checks that the requester possesses the
// protocol Ed25519 key it wants written into the TaskDescriptor.
func verifyGEMMRequesterKeyProof(ctx context.Context, requester string, pub, proof, nonce []byte) error {
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
	if !ed25519.Verify(pub, append([]byte(GemmRequesterKeyBindingDomain), payload...), proof) {
		return errors.New("requester protocol key possession proof is invalid")
	}
	return nil
}

// bondedNetworkKey returns the bonded network Ed25519 key of an account.
func (s msgServer) bondedNetworkKey(ctx context.Context, account string) ([]byte, error) {
	if _, err := addr(account); err != nil {
		return nil, err
	}
	key := s.store(ctx).Get(networkKey(account))
	if len(key) != 32 || len(s.store(ctx).Get(networkProofKey(account))) == 0 {
		return nil, fmt.Errorf("account %s has no bonded network Ed25519 key", account)
	}
	return key, nil
}

// PostGEMMTask validates and escrows one GEMM_INT8_V1 task. The protocol
// task id is derived by the library from the canonical descriptor.
func (s msgServer) PostGEMMTask(ctx context.Context, msg *types.MsgPostGEMMTask) (*types.MsgPostGEMMTaskResponse, error) {
	s.consumeGEMMGas(ctx, "post base", GasGEMMTxBase)
	requester, err := addr(msg.Requester)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if err := verifyGEMMRequesterKeyProof(ctx, msg.Requester, msg.RequesterProtocolPubkey,
		msg.RequesterKeyProof, msg.RequesterNonce); err != nil {
		return nil, err
	}
	if msg.MaxFee == 0 || msg.ChallengeWindow < ChallengeBlocks || len(msg.InputDataRef) > 512 {
		return nil, errors.New("invalid gemm task envelope")
	}
	descriptor, err := gemmv1.NewTaskDescriptor(msg.RequesterProtocolPubkey, msg.RequesterNonce,
		uint64(height), msg.M, msg.N, msg.K, msg.MatrixARoot, msg.MatrixBRoot,
		msg.ChallengeWindow, msg.MaxPricePerCwu)
	if err != nil {
		return nil, fmt.Errorf("gemm task admission failed: %w", err)
	}
	protocolTaskID, err := descriptor.TaskID()
	if err != nil {
		return nil, err
	}
	if err := s.bank.SendCoinsFromAccountToModule(ctx, requester, ModuleName, amount(msg.MaxFee)); err != nil {
		return nil, err
	}
	store := s.store(ctx)
	idBytes := store.Get([]byte("gemm:next"))
	var id uint64 = 1
	if len(idBytes) == 8 {
		id = binary.BigEndian.Uint64(idBytes)
	}
	if id == ^uint64(0) {
		return nil, errors.New("gemm task id exhausted")
	}
	var next [8]byte
	binary.BigEndian.PutUint64(next[:], id+1)
	store.Set([]byte("gemm:next"), next[:])
	task := GEMMTask{
		ID: id, ProtocolTaskID: protocolTaskID,
		Requester: msg.Requester, RequesterProtocolPubKey: append([]byte(nil), msg.RequesterProtocolPubkey...),
		M: msg.M, N: msg.N, K: msg.K,
		MatrixARoot: append([]byte(nil), msg.MatrixARoot...), MatrixBRoot: append([]byte(nil), msg.MatrixBRoot...),
		RequesterNonce:  append([]byte(nil), msg.RequesterNonce...),
		IssuedHeight:    uint64(height),
		ChallengeWindow: msg.ChallengeWindow,
		MaxPricePerCWU:  msg.MaxPricePerCwu, MaxFee: msg.MaxFee,
		InputDataRef: msg.InputDataRef, Status: GEMMStatusPosted,
	}
	if err := s.setGEMMTask(ctx, task); err != nil {
		return nil, err
	}
	s.consumeGEMMGas(ctx, "gemm task write", GasPerGEMMStoreWrite)
	return &types.MsgPostGEMMTaskResponse{GemmTaskId: id, ProtocolTaskId: protocolTaskID}, nil
}

// AcceptGEMMTask binds a bonded worker (with a network Ed25519 key) to the
// task and derives the canonical assignment id.
func (s msgServer) AcceptGEMMTask(ctx context.Context, msg *types.MsgAcceptGEMMTask) (*types.MsgAcceptGEMMTaskResponse, error) {
	s.consumeGEMMGas(ctx, "accept base", GasGEMMTxBase)
	task, err := s.GetGEMMTask(ctx, msg.GemmTaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if task.Status != GEMMStatusPosted {
		return nil, errors.New("gemm task unavailable")
	}
	if height > task.IssuedHeight+task.ChallengeWindow {
		return nil, errors.New("gemm task acceptance window closed")
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
		return nil, errors.New("worker bond below gemm task collateral")
	}
	s.setReserved(ctx, msg.Worker, reserved+requiredBond)
	task.Worker = msg.Worker
	task.WorkerProtocolPubKey = append([]byte(nil), networkKey...)
	task.AssignmentNonce = append([]byte(nil), msg.AssignmentNonce...)
	task.AcceptedHeight = height
	task.ReservedBond = requiredBond
	task.Status = GEMMStatusAssigned
	assignmentID, err := task.gemmAssignment().AssignmentID()
	if err != nil {
		return nil, err
	}
	task.AssignmentID = assignmentID
	if err := s.setGEMMTask(ctx, task); err != nil {
		return nil, err
	}
	s.consumeGEMMGas(ctx, "gemm task write", GasPerGEMMStoreWrite)
	return &types.MsgAcceptGEMMTaskResponse{AssignmentId: assignmentID}, nil
}

// SubmitGEMMResult validates the canonical ResultCommit: identity double
// binding, chain-derived canonical MAC count, worker signature, and moves
// the task into its challenge window. No execution trace is accepted.
func (s msgServer) SubmitGEMMResult(ctx context.Context, msg *types.MsgSubmitGEMMResult) (*types.MsgSubmitGEMMResultResponse, error) {
	s.consumeGEMMGas(ctx, "submit base", GasGEMMTxBase)
	task, err := s.GetGEMMTask(ctx, msg.GemmTaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if task.Status != GEMMStatusAssigned || task.Worker != msg.Worker {
		return nil, errors.New("gemm result rejected")
	}
	if !digest32(msg.OutputRoot) || len(msg.OutputDataRef) == 0 || len(msg.OutputDataRef) > 512 {
		return nil, errors.New("invalid gemm result envelope")
	}
	if msg.CompletedEpoch < task.AcceptedHeight {
		return nil, errors.New("gemm completed epoch precedes assignment")
	}
	networkKey, err := s.bondedNetworkKey(ctx, msg.Worker)
	if err != nil {
		return nil, err
	}
	// Identity double binding: the bonded network key must be the key the
	// ResultCommit was signed with (§42).
	rc := task.gemmResultCommit(msg.OutputRoot, msg.CompletedEpoch)
	if !equalBytesMsg(networkKey, rc.WorkerPubKey) {
		return nil, errGEMMIdentity
	}
	if err := gemmv1.ValidateResultCommit(task.mustDescriptor(), task.gemmAssignment(), rc); err != nil {
		return nil, err
	}
	if !gemmv1.VerifyResultCommitSignature(rc) {
		return nil, errors.New("gemm ResultCommit signature is invalid")
	}
	task.OutputRoot = append([]byte(nil), msg.OutputRoot...)
	task.OutputDataRef = msg.OutputDataRef
	task.OutputBytes = task.M * task.N * 4
	task.ResultSubmittedHeight = height
	task.Status = GEMMStatusResultSubmitted
	task.ChallengeEnd = height + task.ChallengeWindow
	if err := s.setGEMMTask(ctx, task); err != nil {
		return nil, err
	}
	s.consumeGEMMGas(ctx, "gemm task write", GasPerGEMMStoreWrite)
	return &types.MsgSubmitGEMMResultResponse{ChallengeEnd: task.ChallengeEnd}, nil
}

// AttestGEMMTask mirrors the verifiable-task monitor attestation rules so
// settlement economics stay identical.
func (s msgServer) AttestGEMMTask(ctx context.Context, msg *types.MsgAttestGEMMTask) (*types.MsgAttestGEMMTaskResponse, error) {
	s.consumeGEMMGas(ctx, "attest base", GasGEMMTxBase)
	if _, err := addr(msg.Monitor); err != nil {
		return nil, err
	}
	task, err := s.GetGEMMTask(ctx, msg.GemmTaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != GEMMStatusResultSubmitted && task.Status != GEMMStatusChallenged {
		return nil, errors.New("gemm task is not attesting")
	}
	if msg.Monitor == task.Worker || msg.Monitor == task.Requester ||
		s.GetBond(ctx, msg.Monitor) < MinBond {
		return nil, errors.New("monitor is not independent and bonded")
	}
	if record, err := s.GetGEMMDispute(ctx, msg.GemmTaskId); err == nil {
		if msg.Monitor == record.Challenger {
			return nil, errors.New("monitor is not independent and bonded")
		}
	}
	for _, a := range task.Attesters {
		if a == msg.Monitor {
			return nil, errors.New("duplicate monitor")
		}
	}
	if len(task.Attesters) >= 2 {
		return nil, errors.New("two monitors already attested")
	}
	task.Attesters = append(task.Attesters, msg.Monitor)
	return &types.MsgAttestGEMMTaskResponse{}, s.setGEMMTask(ctx, task)
}

func equalBytesMsg(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
