package compute

// GEMM_INT8_V1 on-chain settlement state. The GEMM path is independent of
// the bounded-VM and lightweight task paths: separate storage namespace,
// separate status machine, separate dispute records. Canonical protocol
// values (task id, assignment id, canonical MAC count) are always derived
// through the prismachain/compute/gemmv1 library, never accepted from
// clients.

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"

	storetypes "cosmossdk.io/store/types"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"prismachain/compute/gemmv1"
)

// GEMMProtocolVersion is the frozen wire protocol of the settlement path.
// The v0.1.2 watcher layer is off-chain and must never bump this.
const GEMMProtocolVersion = gemmv1.ProtocolVersion

// GemmRequesterKeyBindingDomain separates requester key-binding signatures.
const GemmRequesterKeyBindingDomain = "prisma:gemm-requester-key-binding:v1\n"

// GEMM statuses: posted -> assigned -> result_submitted -> finalized,
// with challenged disputes in between and refunded / fraud as terminal
// failure states.
const (
	GEMMStatusPosted          = "posted"
	GEMMStatusAssigned        = "assigned"
	GEMMStatusResultSubmitted = "result_submitted"
	GEMMStatusChallenged      = "challenged"
	GEMMStatusFinalized       = "finalized"
	GEMMStatusRefunded        = "refunded"
	GEMMStatusFraud           = "fraud"
)

// GEMMQueuedChallenge stores one bonded queued challenge while another
// dispute is active. The full canonical ChallengeOpen (with signature) is
// kept so promotion is deterministic.
type GEMMQueuedChallenge struct {
	Challenger       string `json:"challenger"`
	ChallengerPubKey []byte `json:"challenger_pub_key"`
	Bond             uint64 `json:"bond"`
	TileI            uint64 `json:"tile_i"`
	TileJ            uint64 `json:"tile_j"`
	WorkerTile       []byte `json:"worker_tile"`
	ChallengerTile   []byte `json:"challenger_tile"`
}

// GEMMTask is the on-chain settlement record of one GEMM_INT8_V1 task.
type GEMMTask struct {
	ID                      uint64                `json:"id"`
	ProtocolTaskID          []byte                `json:"protocol_task_id"`
	Requester               string                `json:"requester"`
	RequesterProtocolPubKey []byte                `json:"requester_protocol_pub_key"`
	M                       uint64                `json:"m"`
	N                       uint64                `json:"n"`
	K                       uint64                `json:"k"`
	MatrixARoot             []byte                `json:"matrix_a_root"`
	MatrixBRoot             []byte                `json:"matrix_b_root"`
	RequesterNonce          []byte                `json:"requester_nonce"`
	IssuedHeight            uint64                `json:"issued_height"`
	ChallengeWindow         uint64                `json:"challenge_window"`
	MaxPricePerCWU          uint64                `json:"max_price_per_cwu"`
	MaxFee                  uint64                `json:"max_fee"`
	InputDataRef            string                `json:"input_data_ref"`
	Worker                  string                `json:"worker,omitempty"`
	WorkerProtocolPubKey    []byte                `json:"worker_protocol_pub_key,omitempty"`
	AssignmentNonce         []byte                `json:"assignment_nonce,omitempty"`
	AssignmentID            []byte                `json:"assignment_id,omitempty"`
	AcceptedHeight          uint64                `json:"accepted_height,omitempty"`
	ReservedBond            uint64                `json:"reserved_bond,omitempty"`
	OutputRoot              []byte                `json:"output_root,omitempty"`
	OutputDataRef           string                `json:"output_data_ref,omitempty"`
	OutputBytes             uint64                `json:"output_bytes,omitempty"`
	ResultSubmittedHeight   uint64                `json:"result_submitted_height,omitempty"`
	ChallengeEnd            uint64                `json:"challenge_end,omitempty"`
	Status                  string                `json:"status"`
	QueuedGEMMChallenges    []GEMMQueuedChallenge `json:"queued_gemm_challenges,omitempty"`
	ActiveTileI             uint64                `json:"active_tile_i,omitempty"`
	ActiveTileJ             uint64                `json:"active_tile_j,omitempty"`
	Attesters               []string              `json:"attesters,omitempty"`
	ReceiptID               []byte                `json:"receipt_id,omitempty"`
	SurvivedChallenge       bool                  `json:"survived_challenge,omitempty"`
	DisputeTranscriptDigest []byte                `json:"dispute_transcript_digest,omitempty"`
}

// GEMMDisputeRecord persists one challenge and its dispute. The dispute
// core lives in the canonical GMD1 snapshot; everything around it is chain
// identity and economics.
type GEMMDisputeRecord struct {
	TaskID           uint64 `json:"task_id"`
	Challenger       string `json:"challenger"`
	ChallengerPubKey []byte `json:"challenger_pub_key"`
	Bond             uint64 `json:"bond"`
	WorkerTile       []byte `json:"worker_tile"`
	ChallengerTile   []byte `json:"challenger_tile"`
	WorkerTrace      []byte `json:"worker_trace,omitempty"`
	ChallengerTrace  []byte `json:"challenger_trace,omitempty"`
	Snapshot         []byte `json:"snapshot,omitempty"`
	TranscriptDigest []byte `json:"transcript_digest"`
	Status           string `json:"status"`
	TraceDeadline    uint64 `json:"trace_deadline"`
	Outcome          string `json:"outcome,omitempty"`
}

// GEMM dispute sub-phase statuses.
const (
	GEMMDisputeOpen      = "open"
	GEMMDisputeBisection = "bisection"
	GEMMDisputeArbReady  = "arb_ready"
	GEMMDisputeResolved  = "resolved"
)

func gemmTaskKey(id uint64) []byte {
	var key [9]byte
	key[0] = 'g'
	binary.BigEndian.PutUint64(key[1:], id)
	return key[:]
}

func gemmDisputeKey(taskID uint64) []byte {
	var key [9]byte
	key[0] = 'd'
	binary.BigEndian.PutUint64(key[1:], taskID)
	return key[:]
}

func gemmReceiptKey(receiptID []byte) []byte {
	return append([]byte{'v'}, receiptID...)
}

func gemmWorkerWorkKey(worker string) []byte { return append([]byte{'w'}, []byte(worker)...) }

func (k Keeper) GetGEMMTask(ctx context.Context, id uint64) (GEMMTask, error) {
	data := k.store(ctx).Get(gemmTaskKey(id))
	if len(data) == 0 {
		return GEMMTask{}, errGEMMTaskNotFound
	}
	var task GEMMTask
	if err := json.Unmarshal(data, &task); err != nil {
		return GEMMTask{}, err
	}
	return task, nil
}

func (k Keeper) setGEMMTask(ctx context.Context, task GEMMTask) error {
	data, err := json.Marshal(task)
	if err != nil {
		return err
	}
	k.store(ctx).Set(gemmTaskKey(task.ID), data)
	return nil
}

func (k Keeper) GetGEMMDispute(ctx context.Context, taskID uint64) (GEMMDisputeRecord, error) {
	data := k.store(ctx).Get(gemmDisputeKey(taskID))
	if len(data) == 0 {
		return GEMMDisputeRecord{}, errGEMMDisputeNotFound
	}
	var record GEMMDisputeRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return GEMMDisputeRecord{}, err
	}
	return record, nil
}

func (k Keeper) setGEMMDispute(ctx context.Context, record GEMMDisputeRecord) error {
	data, err := json.Marshal(record)
	if err != nil {
		return err
	}
	k.store(ctx).Set(gemmDisputeKey(record.TaskID), data)
	return nil
}

func (k Keeper) deleteGEMMDispute(ctx context.Context, taskID uint64) {
	k.store(ctx).Delete(gemmDisputeKey(taskID))
}

// GetGEMMVerifiedWork returns the worker's total settled canonical MACs.
func (k Keeper) GetGEMMVerifiedWork(ctx context.Context, worker string) uint64 {
	data := k.store(ctx).Get(gemmWorkerWorkKey(worker))
	if len(data) != 8 {
		return 0
	}
	return binary.BigEndian.Uint64(data)
}

func (k Keeper) addGEMMVerifiedWork(ctx context.Context, worker string, macs uint64) {
	total := k.GetGEMMVerifiedWork(ctx, worker)
	var data [8]byte
	binary.BigEndian.PutUint64(data[:], total+macs)
	k.store(ctx).Set(gemmWorkerWorkKey(worker), data[:])
}

// gemmDescriptor rebuilds the canonical TaskDescriptor from chain state.
// The protocol library is the single source of truth for admission rules
// and derived values.
func (task *GEMMTask) gemmDescriptor() (*gemmv1.TaskDescriptor, error) {
	return gemmv1.NewTaskDescriptor(task.RequesterProtocolPubKey, task.RequesterNonce,
		task.IssuedHeight, task.M, task.N, task.K, task.MatrixARoot, task.MatrixBRoot,
		task.ChallengeWindow, task.MaxPricePerCWU)
}

// gemmAssignment rebuilds the canonical Assignment from chain state.
func (task *GEMMTask) gemmAssignment() *gemmv1.Assignment {
	return &gemmv1.Assignment{
		TaskID:          task.ProtocolTaskID,
		WorkerPubKey:    task.WorkerProtocolPubKey,
		AssignmentNonce: task.AssignmentNonce,
		AcceptedEpoch:   task.AcceptedHeight,
	}
}

// gemmResultCommit rebuilds the canonical ResultCommit from chain state and
// the worker's submitted fields. CanonicalMACCount is chain-derived.
func (task *GEMMTask) gemmResultCommit(outputRoot []byte, completedEpoch uint64) *gemmv1.ResultCommit {
	return &gemmv1.ResultCommit{
		ProtocolVersion:   GEMMProtocolVersion,
		TaskID:            task.ProtocolTaskID,
		AssignmentID:      task.AssignmentID,
		WorkerPubKey:      task.WorkerProtocolPubKey,
		OutputRoot:        outputRoot,
		CanonicalMACCount: task.M * task.N * task.K,
		CompletedEpoch:    completedEpoch,
	}
}

// gemmBondFor is the challenge bond: existing verifiable-task semantics.
func gemmBondFor(maxFee uint64) uint64 {
	bond := maxFee / 100
	if bond == 0 {
		bond = 1
	}
	return bond
}

var _ = storetypes.KVStoreKey{}
var _ = sdk.AccAddress{}
var _ = gemmv1.TileSize

// mustDescriptor is used on chain-state tasks that already passed
// admission; a failure here is a state-corruption bug, not a user error.
func (task *GEMMTask) mustDescriptor() *gemmv1.TaskDescriptor {
	descriptor, err := task.gemmDescriptor()
	if err != nil {
		panic(fmt.Sprintf("stored gemm task no longer validates: %v", err))
	}
	return descriptor
}
