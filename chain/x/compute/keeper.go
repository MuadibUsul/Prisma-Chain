package compute

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	mathsdk "cosmossdk.io/math"
	storetypes "cosmossdk.io/store/types"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"prismachain/chain/x/compute/types"
	"prismachain/vm"
)

const (
	ModuleName                  = "compute"
	Denom                       = "uprsm"
	MinBond              uint64 = 1_000_000
	ChallengeBlocks      uint64 = 20
	ChallengeRoundBlocks uint64 = 5
)

type Model struct {
	ID              string  `json:"id"`
	Version         string  `json:"version"`
	Mode            string  `json:"mode"`
	Owner           string  `json:"owner"`
	ImageDigest     []byte  `json:"image_digest"`
	TokenizerDigest []byte  `json:"tokenizer_digest"`
	WeightsDigest   []byte  `json:"weights_digest"`
	ProgramDigest   vm.Hash `json:"program_digest"`
	Program         []byte  `json:"program,omitempty"`
}

type Task struct {
	ID              uint64   `json:"id"`
	Mode            string   `json:"mode"`
	ModelID         string   `json:"model_id"`
	SpecVersion     string   `json:"spec_version"`
	Requester       string   `json:"requester"`
	Worker          string   `json:"worker,omitempty"`
	ReservedBond    uint64   `json:"reserved_bond,omitempty"`
	InputCommitment []byte   `json:"input_commitment"`
	DataRef         string   `json:"data_ref"`
	MaxFee          uint64   `json:"max_fee"`
	Deadline        uint64   `json:"deadline"`
	PrivacyTier     string   `json:"privacy_tier"`
	PublicInput     []int64  `json:"public_input,omitempty"`
	Status          string   `json:"status"`
	OutputDigest    []byte   `json:"output_digest,omitempty"`
	TraceRoot       []byte   `json:"trace_root,omitempty"`
	ReceiptDigest   []byte   `json:"receipt_digest,omitempty"`
	OutputTokens    uint64   `json:"output_tokens,omitempty"`
	TraceClaimJSON  []byte   `json:"trace_claim_json,omitempty"`
	ChallengeEnd    uint64   `json:"challenge_end,omitempty"`
	Attesters       []string `json:"attesters,omitempty"`
	Challenger      string   `json:"challenger,omitempty"`
	ChallengeBond   uint64   `json:"challenge_bond,omitempty"`
	DisputeSnapshot []byte   `json:"dispute_snapshot,omitempty"`
}

type Keeper struct {
	key  *storetypes.KVStoreKey
	bank BankKeeper
}

type BankKeeper interface {
	SendCoinsFromAccountToModule(context.Context, sdk.AccAddress, string, sdk.Coins) error
	SendCoinsFromModuleToAccount(context.Context, string, sdk.AccAddress, sdk.Coins) error
	BurnCoins(context.Context, string, sdk.Coins) error
}

func NewKeeper(key *storetypes.KVStoreKey, bank BankKeeper) Keeper {
	return Keeper{key: key, bank: bank}
}

func (k Keeper) store(ctx context.Context) storetypes.KVStore {
	return sdk.UnwrapSDKContext(ctx).KVStore(k.key)
}

func taskKey(id uint64) []byte {
	var key [9]byte
	key[0] = 't'
	binary.BigEndian.PutUint64(key[1:], id)
	return key[:]
}

func modelKey(id string, version string) []byte {
	key := append([]byte{'m'}, []byte(id)...)
	key = append(key, 0)
	return append(key, []byte(version)...)
}

func ownerKey(id string) []byte { return append([]byte{'o'}, []byte(id)...) }

func bondKey(worker string) []byte    { return append([]byte{'b'}, []byte(worker)...) }
func reserveKey(worker string) []byte { return append([]byte{'r'}, []byte(worker)...) }
func networkKey(worker string) []byte { return append([]byte{'k'}, []byte(worker)...) }
func nodeKey(digest [32]byte) []byte  { return append([]byte{'n'}, digest[:]...) }

func (k Keeper) GetTask(ctx context.Context, id uint64) (Task, error) {
	data := k.store(ctx).Get(taskKey(id))
	if len(data) == 0 {
		return Task{}, fmt.Errorf("task %d not found", id)
	}
	var task Task
	if err := json.Unmarshal(data, &task); err != nil {
		return Task{}, err
	}
	return task, nil
}

func (k Keeper) setTask(ctx context.Context, task Task) error {
	data, err := json.Marshal(task)
	if err != nil {
		return err
	}
	k.store(ctx).Set(taskKey(task.ID), data)
	return nil
}

func (k Keeper) GetModel(ctx context.Context, id string, version string) (Model, error) {
	data := k.store(ctx).Get(modelKey(id, version))
	if len(data) == 0 {
		return Model{}, fmt.Errorf("model %q v%q not found", id, version)
	}
	var model Model
	if err := json.Unmarshal(data, &model); err != nil {
		return Model{}, err
	}
	return model, nil
}

func (k Keeper) GetBond(ctx context.Context, worker string) uint64 {
	data := k.store(ctx).Get(bondKey(worker))
	if len(data) != 8 {
		return 0
	}
	return binary.BigEndian.Uint64(data)
}

func (k Keeper) setBond(ctx context.Context, worker string, amount uint64) {
	var data [8]byte
	binary.BigEndian.PutUint64(data[:], amount)
	k.store(ctx).Set(bondKey(worker), data[:])
}

func (k Keeper) getReserved(ctx context.Context, worker string) uint64 {
	data := k.store(ctx).Get(reserveKey(worker))
	if len(data) != 8 {
		return 0
	}
	return binary.BigEndian.Uint64(data)
}

func (k Keeper) setReserved(ctx context.Context, worker string, amount uint64) {
	var data [8]byte
	binary.BigEndian.PutUint64(data[:], amount)
	k.store(ctx).Set(reserveKey(worker), data[:])
}

func (k Keeper) releaseReservation(ctx context.Context, task *Task) error {
	if task.ReservedBond == 0 {
		return nil
	}
	reserved := k.getReserved(ctx, task.Worker)
	if reserved < task.ReservedBond {
		return errors.New("worker bond reservation invariant broken")
	}
	k.setReserved(ctx, task.Worker, reserved-task.ReservedBond)
	task.ReservedBond = 0
	return nil
}

func amount(n uint64) sdk.Coins { return sdk.NewCoins(sdk.NewCoin(Denom, mathsdk.NewIntFromUint64(n))) }

func addr(raw string) (sdk.AccAddress, error) {
	a, err := sdk.AccAddressFromBech32(raw)
	if err != nil {
		return nil, fmt.Errorf("invalid prsm address: %w", err)
	}
	return a, nil
}

func digest32(b []byte) bool { return len(b) == sha256.Size }

func validLabel(s string, allowSlash bool) bool {
	if len(s) == 0 || len(s) > 128 {
		return false
	}
	for _, c := range s {
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' ||
			c == '_' || c == '-' || c == '.' || allowSlash && c == '/' {
			continue
		}
		return false
	}
	return true
}

func decodeProgram(model Model) (vm.Program, error) {
	var code []vm.Instruction
	if err := json.Unmarshal(model.Program, &code); err != nil {
		return vm.Program{}, err
	}
	p, err := vm.NewProgram(code)
	if err != nil {
		return vm.Program{}, err
	}
	if p.Digest() != model.ProgramDigest {
		return vm.Program{}, errors.New("registered program digest mismatch")
	}
	return p, nil
}

// OutputDigest is the v1 commitment to a bounded VM's signed integer output.
func OutputDigest(value int64) [32]byte {
	var input [27]byte
	copy(input[:19], []byte("prismavm:output:v1\x00"))
	binary.BigEndian.PutUint64(input[19:], uint64(value))
	return sha256.Sum256(input[:])
}

type msgServer struct{ Keeper }

func (k Keeper) MsgServer() types.MsgServer     { return msgServer{k} }
func (k Keeper) QueryServer() types.QueryServer { return queryServer{k} }

func (s msgServer) RegisterModel(ctx context.Context, msg *types.MsgRegisterModel) (*types.MsgRegisterModelResponse, error) {
	if _, err := addr(msg.Owner); err != nil {
		return nil, err
	}
	if !validLabel(msg.ModelId, true) || !validLabel(msg.SpecVersion, false) || len(msg.SpecVersion) > 64 || (msg.Mode != "verifiable" && msg.Mode != "lightweight") {
		return nil, errors.New("invalid model identity or mode")
	}
	for _, digest := range [][]byte{msg.ImageDigest, msg.TokenizerDigest, msg.WeightsDigest} {
		if !digest32(digest) {
			return nil, errors.New("image, tokenizer and weights digests must be 32 bytes")
		}
	}
	if owner := s.store(ctx).Get(ownerKey(msg.ModelId)); len(owner) != 0 && string(owner) != msg.Owner {
		return nil, errors.New("model ID belongs to another owner")
	}
	if s.store(ctx).Has(modelKey(msg.ModelId, msg.SpecVersion)) {
		return nil, errors.New("model version already exists")
	}
	model := Model{ID: msg.ModelId, Version: msg.SpecVersion, Mode: msg.Mode, Owner: msg.Owner,
		ImageDigest: msg.ImageDigest, TokenizerDigest: msg.TokenizerDigest, WeightsDigest: msg.WeightsDigest}
	if msg.Mode == "verifiable" {
		if len(msg.Program) == 0 || len(msg.Program) > 256_000 {
			return nil, errors.New("invalid bounded program size")
		}
		model.Program = msg.Program
		program, err := decodeUnregisteredProgram(msg.Program)
		if err != nil {
			return nil, err
		}
		model.ProgramDigest = program.Digest()
	} else if len(msg.Program) != 0 {
		return nil, errors.New("lightweight model must not contain a VM program")
	}
	data, err := json.Marshal(model)
	if err != nil {
		return nil, err
	}
	s.store(ctx).Set(modelKey(msg.ModelId, msg.SpecVersion), data)
	s.store(ctx).Set(ownerKey(msg.ModelId), []byte(msg.Owner))
	return &types.MsgRegisterModelResponse{}, nil
}

func decodeUnregisteredProgram(data []byte) (vm.Program, error) {
	var code []vm.Instruction
	if err := json.Unmarshal(data, &code); err != nil {
		return vm.Program{}, err
	}
	return vm.NewProgram(code)
}

func (s msgServer) BondWorker(ctx context.Context, msg *types.MsgBondWorker) (*types.MsgBondWorkerResponse, error) {
	worker, err := addr(msg.Worker)
	if err != nil {
		return nil, err
	}
	if msg.Amount == 0 || msg.Amount > math.MaxUint64-s.GetBond(ctx, msg.Worker) {
		return nil, errors.New("invalid bond amount")
	}
	if len(msg.NetworkPublicKey) != 0 && len(msg.NetworkPublicKey) != 32 {
		return nil, errors.New("network Ed25519 public key must be 32 bytes")
	}
	if len(msg.NetworkPublicKey) == 32 {
		if prior := s.store(ctx).Get(networkKey(msg.Worker)); len(prior) != 0 && string(prior) != string(msg.NetworkPublicKey) {
			return nil, errors.New("bonded network key cannot be replaced")
		}
		node := sha256.Sum256(msg.NetworkPublicKey)
		if prior := s.store(ctx).Get(nodeKey(node)); len(prior) != 0 && string(prior) != msg.Worker {
			return nil, errors.New("network key belongs to another bonded account")
		}
		s.store(ctx).Set(networkKey(msg.Worker), msg.NetworkPublicKey)
		s.store(ctx).Set(nodeKey(node), []byte(msg.Worker))
	}
	if err := s.bank.SendCoinsFromAccountToModule(ctx, worker, ModuleName, amount(msg.Amount)); err != nil {
		return nil, err
	}
	s.setBond(ctx, msg.Worker, s.GetBond(ctx, msg.Worker)+msg.Amount)
	return &types.MsgBondWorkerResponse{}, nil
}

func (s msgServer) PostTask(ctx context.Context, msg *types.MsgPostTask) (*types.MsgPostTaskResponse, error) {
	requester, err := addr(msg.Requester)
	if err != nil {
		return nil, err
	}
	model, err := s.GetModel(ctx, msg.ModelId, msg.SpecVersion)
	if err != nil {
		return nil, err
	}
	height := sdk.UnwrapSDKContext(ctx).BlockHeight()
	if height < 0 || msg.Deadline < uint64(height)+ChallengeBlocks+1 || msg.MaxFee == 0 ||
		msg.Mode != model.Mode || !digest32(msg.InputCommitment) || len(msg.DataRef) > 512 {
		return nil, errors.New("invalid task envelope")
	}
	if msg.Mode == "verifiable" {
		if msg.PrivacyTier != "public" {
			return nil, errors.New("verifiable VM input is public")
		}
		program, err := decodeProgram(model)
		if err != nil {
			return nil, err
		}
		if len(msg.PublicInput) != program.RequiredInputs() || len(msg.PublicInput) > vm.MaxInputValues {
			return nil, errors.New("wrong public input length")
		}
		digest := vm.InputDigest(msg.PublicInput)
		if string(digest[:]) != string(msg.InputCommitment) {
			return nil, errors.New("input commitment mismatch")
		}
	} else if msg.PrivacyTier != "tier0_relative" || len(msg.PublicInput) != 0 || msg.DataRef == "" {
		return nil, errors.New("lightweight task requires tier0_relative and an encrypted data reference")
	}
	if err := s.bank.SendCoinsFromAccountToModule(ctx, requester, ModuleName, amount(msg.MaxFee)); err != nil {
		return nil, err
	}
	store := s.store(ctx)
	idBytes := store.Get([]byte("next"))
	var id uint64 = 1
	if len(idBytes) == 8 {
		id = binary.BigEndian.Uint64(idBytes)
	}
	if id == math.MaxUint64 {
		return nil, errors.New("task ID exhausted")
	}
	var next [8]byte
	binary.BigEndian.PutUint64(next[:], id+1)
	store.Set([]byte("next"), next[:])
	task := Task{ID: id, Mode: msg.Mode, ModelID: msg.ModelId, SpecVersion: msg.SpecVersion,
		Requester: msg.Requester, InputCommitment: msg.InputCommitment, DataRef: msg.DataRef,
		MaxFee: msg.MaxFee, Deadline: msg.Deadline, PrivacyTier: msg.PrivacyTier, PublicInput: msg.PublicInput, Status: "posted"}
	if err := s.setTask(ctx, task); err != nil {
		return nil, err
	}
	return &types.MsgPostTaskResponse{TaskId: id}, nil
}

func (s msgServer) AcceptTask(ctx context.Context, msg *types.MsgAcceptTask) (*types.MsgAcceptTaskResponse, error) {
	if _, err := addr(msg.Worker); err != nil {
		return nil, err
	}
	task, err := s.GetTask(ctx, msg.TaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != "posted" || uint64(sdk.UnwrapSDKContext(ctx).BlockHeight()) > task.Deadline {
		return nil, errors.New("task unavailable")
	}
	requiredBond := task.MaxFee
	if requiredBond < MinBond {
		requiredBond = MinBond
	}
	bond, reserved := s.GetBond(ctx, msg.Worker), s.getReserved(ctx, msg.Worker)
	if bond < reserved || bond-reserved < requiredBond {
		return nil, errors.New("worker bond below task collateral")
	}
	if task.Mode == "lightweight" && len(s.store(ctx).Get(networkKey(msg.Worker))) != 32 {
		return nil, errors.New("lightweight worker must register a network Ed25519 key")
	}
	s.setReserved(ctx, msg.Worker, reserved+requiredBond)
	task.Worker, task.Status, task.ReservedBond = msg.Worker, "accepted", requiredBond
	return &types.MsgAcceptTaskResponse{}, s.setTask(ctx, task)
}

func (s msgServer) SubmitResult(ctx context.Context, msg *types.MsgSubmitResult) (*types.MsgSubmitResultResponse, error) {
	task, err := s.GetTask(ctx, msg.TaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != "accepted" || task.Worker != msg.Worker || uint64(sdk.UnwrapSDKContext(ctx).BlockHeight()) > task.Deadline || !digest32(msg.OutputDigest) {
		return nil, errors.New("result rejected")
	}
	if task.Mode == "verifiable" {
		if len(msg.TraceClaimJson) == 0 || len(msg.TraceClaimJson) > 8192 || len(msg.ReceiptDigest) != 0 {
			return nil, errors.New("invalid verifiable result evidence")
		}
		model, err := s.GetModel(ctx, task.ModelID, task.SpecVersion)
		if err != nil {
			return nil, err
		}
		program, err := decodeProgram(model)
		if err != nil {
			return nil, err
		}
		var claim vm.TraceClaim
		if err := json.Unmarshal(msg.TraceClaimJson, &claim); err != nil {
			return nil, err
		}
		if !digest32(msg.TraceRoot) || string(claim.Root[:]) != string(msg.TraceRoot) || claim.Final.PC != uint32(program.Len()) {
			return nil, errors.New("invalid trace endpoint")
		}
		if !vm.VerifyProof(claim.Root, vm.State{}, 0, uint32(program.Len()+1), claim.InitialProof) ||
			!vm.VerifyProof(claim.Root, claim.Final, uint32(program.Len()), uint32(program.Len()+1), claim.FinalProof) {
			return nil, errors.New("invalid trace endpoint proofs")
		}
		out := OutputDigest(claim.Final.Reg[0])
		if string(out[:]) != string(msg.OutputDigest) {
			return nil, errors.New("output does not match trace endpoint")
		}
	} else if !digest32(msg.ReceiptDigest) || msg.OutputTokens == 0 || len(msg.TraceClaimJson) != 0 || len(msg.TraceRoot) != 0 {
		return nil, errors.New("lightweight result requires receipt digest and billed output tokens")
	}
	task.OutputDigest, task.TraceRoot, task.ReceiptDigest = msg.OutputDigest, msg.TraceRoot, msg.ReceiptDigest
	task.OutputTokens, task.TraceClaimJSON = msg.OutputTokens, msg.TraceClaimJson
	task.Status = "pending"
	task.ChallengeEnd = uint64(sdk.UnwrapSDKContext(ctx).BlockHeight()) + ChallengeBlocks
	return &types.MsgSubmitResultResponse{}, s.setTask(ctx, task)
}

func (s msgServer) AttestResult(ctx context.Context, msg *types.MsgAttestResult) (*types.MsgAttestResultResponse, error) {
	if _, err := addr(msg.Monitor); err != nil {
		return nil, err
	}
	task, err := s.GetTask(ctx, msg.TaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != "pending" || msg.Monitor == task.Worker || msg.Monitor == task.Requester || s.GetBond(ctx, msg.Monitor) < MinBond {
		return nil, errors.New("monitor is not independent and bonded")
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
	return &types.MsgAttestResultResponse{}, s.setTask(ctx, task)
}

func (s msgServer) StartChallenge(ctx context.Context, msg *types.MsgStartChallenge) (*types.MsgStartChallengeResponse, error) {
	challenger, err := addr(msg.Challenger)
	if err != nil {
		return nil, err
	}
	task, err := s.GetTask(ctx, msg.TaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if task.Mode != "verifiable" || task.Status != "pending" || height > task.ChallengeEnd || msg.Challenger == task.Worker {
		return nil, errors.New("challenge unavailable")
	}
	if len(msg.TraceClaimJson) == 0 || len(msg.TraceClaimJson) > 8192 {
		return nil, errors.New("invalid challenger trace claim size")
	}
	model, err := s.GetModel(ctx, task.ModelID, task.SpecVersion)
	if err != nil {
		return nil, err
	}
	program, err := decodeProgram(model)
	if err != nil {
		return nil, err
	}
	var worker, claimant vm.TraceClaim
	if err := json.Unmarshal(task.TraceClaimJSON, &worker); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(msg.TraceClaimJson, &claimant); err != nil {
		return nil, err
	}
	dispute, err := vm.NewDispute(program, task.PublicInput, worker, claimant, height, ChallengeRoundBlocks)
	if err != nil {
		return nil, err
	}
	bond := task.MaxFee / 100
	if bond == 0 {
		bond = 1
	}
	if err := s.bank.SendCoinsFromAccountToModule(ctx, challenger, ModuleName, amount(bond)); err != nil {
		return nil, err
	}
	snapshot, err := dispute.Snapshot()
	if err != nil {
		return nil, err
	}
	task.Challenger, task.ChallengeBond = msg.Challenger, bond
	task.DisputeSnapshot, task.Status = snapshot, "challenged"
	if err := s.resolveChallenge(ctx, &task, dispute.Status().Outcome); err != nil {
		return nil, err
	}
	return &types.MsgStartChallengeResponse{}, s.setTask(ctx, task)
}

func (s msgServer) ChallengeMidpoint(ctx context.Context, msg *types.MsgChallengeMidpoint) (*types.MsgChallengeMidpointResponse, error) {
	task, err := s.GetTask(ctx, msg.TaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != "challenged" {
		return nil, errors.New("task is not challenged")
	}
	if len(msg.StateJson) == 0 || len(msg.StateJson) > 2048 || len(msg.ProofJson) == 0 || len(msg.ProofJson) > 8192 {
		return nil, errors.New("invalid bounded challenge witness")
	}
	var party vm.Party
	switch msg.Actor {
	case task.Worker:
		party = vm.Worker
	case task.Challenger:
		party = vm.Challenger
	default:
		return nil, errors.New("actor is not a dispute party")
	}
	model, err := s.GetModel(ctx, task.ModelID, task.SpecVersion)
	if err != nil {
		return nil, err
	}
	program, err := decodeProgram(model)
	if err != nil {
		return nil, err
	}
	dispute, err := vm.RestoreDispute(task.DisputeSnapshot, program, task.PublicInput)
	if err != nil {
		return nil, err
	}
	var state vm.State
	var proof vm.Proof
	if err := json.Unmarshal(msg.StateJson, &state); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(msg.ProofJson, &proof); err != nil {
		return nil, err
	}
	outcome, err := dispute.SubmitMid(party, state, proof, uint64(sdk.UnwrapSDKContext(ctx).BlockHeight()))
	if err != nil {
		return nil, err
	}
	task.DisputeSnapshot, err = dispute.Snapshot()
	if err != nil {
		return nil, err
	}
	if err := s.resolveChallenge(ctx, &task, outcome); err != nil {
		return nil, err
	}
	return &types.MsgChallengeMidpointResponse{}, s.setTask(ctx, task)
}

func (s msgServer) TimeoutChallenge(ctx context.Context, msg *types.MsgTimeoutChallenge) (*types.MsgTimeoutChallengeResponse, error) {
	if _, err := addr(msg.Actor); err != nil {
		return nil, err
	}
	task, err := s.GetTask(ctx, msg.TaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != "challenged" {
		return nil, errors.New("task is not challenged")
	}
	model, err := s.GetModel(ctx, task.ModelID, task.SpecVersion)
	if err != nil {
		return nil, err
	}
	program, err := decodeProgram(model)
	if err != nil {
		return nil, err
	}
	dispute, err := vm.RestoreDispute(task.DisputeSnapshot, program, task.PublicInput)
	if err != nil {
		return nil, err
	}
	outcome, err := dispute.Timeout(uint64(sdk.UnwrapSDKContext(ctx).BlockHeight()))
	if err != nil {
		return nil, err
	}
	task.DisputeSnapshot, err = dispute.Snapshot()
	if err != nil {
		return nil, err
	}
	if err := s.resolveChallenge(ctx, &task, outcome); err != nil {
		return nil, err
	}
	return &types.MsgTimeoutChallengeResponse{}, s.setTask(ctx, task)
}

func (s msgServer) resolveChallenge(ctx context.Context, task *Task, outcome vm.Outcome) error {
	if outcome == vm.Pending {
		return nil
	}
	worker, _ := addr(task.Worker)
	challenger, _ := addr(task.Challenger)
	if outcome == vm.WorkerWins {
		if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, worker, amount(task.ChallengeBond)); err != nil {
			return err
		}
		task.Status = "pending"
		task.ChallengeEnd = uint64(sdk.UnwrapSDKContext(ctx).BlockHeight()) + ChallengeRoundBlocks
	} else {
		requester, _ := addr(task.Requester)
		if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, requester, amount(task.MaxFee)); err != nil {
			return err
		}
		penaltyCap := task.ReservedBond
		if err := s.releaseReservation(ctx, task); err != nil {
			return err
		}
		if outcome == vm.ChallengerWins {
			if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, challenger, amount(task.ChallengeBond)); err != nil {
				return err
			}
		} else if err := s.bank.BurnCoins(ctx, ModuleName, amount(task.ChallengeBond)); err != nil {
			return err
		}
		bond := s.GetBond(ctx, task.Worker)
		slash := bond / 10
		if slash > penaltyCap {
			slash = penaltyCap
		}
		if slash > 0 {
			if outcome == vm.ChallengerWins {
				if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, challenger, amount(slash)); err != nil {
					return err
				}
			} else if err := s.bank.BurnCoins(ctx, ModuleName, amount(slash)); err != nil {
				return err
			}
			s.setBond(ctx, task.Worker, bond-slash)
		}
		task.Status = "refunded"
	}
	task.DisputeSnapshot = nil
	return nil
}

func (s msgServer) FinalizeTask(ctx context.Context, msg *types.MsgFinalizeTask) (*types.MsgFinalizeTaskResponse, error) {
	if _, err := addr(msg.Actor); err != nil {
		return nil, err
	}
	task, err := s.GetTask(ctx, msg.TaskId)
	if err != nil {
		return nil, err
	}
	if !payable(task, uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())) {
		return nil, errors.New("task not finalizable")
	}
	burn, monitorShare, workerShare := feeSplit(task.MaxFee)
	if burn > 0 {
		if err := s.bank.BurnCoins(ctx, ModuleName, amount(burn)); err != nil {
			return nil, err
		}
	}
	for _, a := range task.Attesters {
		if monitorShare > 0 {
			monitor, _ := addr(a)
			if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, monitor, amount(monitorShare)); err != nil {
				return nil, err
			}
		}
	}
	worker, _ := addr(task.Worker)
	if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, worker, amount(workerShare)); err != nil {
		return nil, err
	}
	task.Status = "settled"
	if err := s.releaseReservation(ctx, &task); err != nil {
		return nil, err
	}
	return &types.MsgFinalizeTaskResponse{}, s.setTask(ctx, task)
}

func (s msgServer) RefundTask(ctx context.Context, msg *types.MsgRefundTask) (*types.MsgRefundTaskResponse, error) {
	if _, err := addr(msg.Actor); err != nil {
		return nil, err
	}
	task, err := s.GetTask(ctx, msg.TaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if !refundable(task, height) {
		return nil, errors.New("task not refundable")
	}
	requester, _ := addr(task.Requester)
	if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, requester, amount(task.MaxFee)); err != nil {
		return nil, err
	}
	if err := s.releaseReservation(ctx, &task); err != nil {
		return nil, err
	}
	task.Status = "refunded"
	return &types.MsgRefundTaskResponse{}, s.setTask(ctx, task)
}

func finalizable(task Task, height uint64) bool {
	return task.Status == "pending" && height > task.ChallengeEnd && len(task.Attesters) == 2
}

// Lightweight settlement stays closed until signed receipts and usage pricing
// are verified by the module. A receipt hash alone never authorizes payment.
func payable(task Task, height uint64) bool {
	return task.Mode == "verifiable" && finalizable(task, height)
}

func refundable(task Task, height uint64) bool {
	if (task.Status == "posted" || task.Status == "accepted") && height > task.Deadline {
		return true
	}
	return task.Status == "pending" && height > task.ChallengeEnd+ChallengeBlocks &&
		(task.Mode == "lightweight" || len(task.Attesters) < 2)
}

func feeSplit(fee uint64) (burn, eachMonitor, worker uint64) {
	burn = fee / 5
	eachMonitor = fee / 20
	worker = fee - burn - 2*eachMonitor
	return
}

type queryServer struct{ Keeper }

func (q queryServer) Task(ctx context.Context, req *types.QueryTaskRequest) (*types.QueryTaskResponse, error) {
	task, err := q.GetTask(ctx, req.TaskId)
	if err != nil {
		return nil, err
	}
	data, err := json.Marshal(task)
	return &types.QueryTaskResponse{TaskJson: data}, err
}

func (q queryServer) Model(ctx context.Context, req *types.QueryModelRequest) (*types.QueryModelResponse, error) {
	model, err := q.GetModel(ctx, req.ModelId, req.SpecVersion)
	if err != nil {
		return nil, err
	}
	data, err := json.Marshal(model)
	return &types.QueryModelResponse{ModelJson: data}, err
}

func (q queryServer) Worker(ctx context.Context, req *types.QueryWorkerRequest) (*types.QueryWorkerResponse, error) {
	if _, err := addr(req.Worker); err != nil {
		return nil, err
	}
	return &types.QueryWorkerResponse{BondedUprsm: q.GetBond(ctx, req.Worker),
		NetworkPublicKey: q.store(ctx).Get(networkKey(req.Worker)), ReservedUprsm: q.getReserved(ctx, req.Worker)}, nil
}
