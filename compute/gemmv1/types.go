package gemmv1

// Protocol constants and canonical message types for GEMM_INT8_V1.
//
// GEMM v1 is an independent, versioned operator next to PRISMA_BOUNDED_INT_VM_V1
// (the vm/ package). It shares no state, trace or proof formats with the VM;
// only general Merkle and hashing discipline is conceptually common.

import "math"

const (
	// ProtocolVersion is the version string embedded in every GEMM v1 object.
	ProtocolVersion = "0.1.1"
	// Operator is the fixed operator identifier of this protocol.
	Operator = "GEMM_INT8_V1"
	// ArithmeticSpec is the fixed arithmetic identifier of this protocol.
	ArithmeticSpec = "INT8_INT32_V1"
	// SettlementAssetUPRSM names the settlement unit of the local chain.
	SettlementAssetUPRSM = "uprsm"

	// TileSize is the fixed GEMM tile side. All tiles are TileSize x TileSize;
	// boundary tiles are zero padded and padding never enters
	// canonical_mac_count.
	TileSize = 8
)

// Verification modes of a VerifiedWorkReceipt.
const (
	ModeOptimisticUnchallenged = "optimistic_unchallenged"
	ModeChallengedWorkerWon    = "challenged_worker_won"
)

// Dispute parties and outcomes, mirroring the vm package vocabulary.
type Party uint8

const (
	Worker Party = iota + 1
	Challenger
)

type Outcome uint8

const (
	Pending Outcome = iota
	WorkerWins
	ChallengerWins
	BothInvalid
)

// TaskDescriptor describes one verifiable INT8 GEMM task. All fields are
// always encoded; matrix roots and identities are fixed-size byte strings.
type TaskDescriptor struct {
	ProtocolVersion string `gemm:"protocol_version"`
	Operator        string `gemm:"operator"`

	RequesterPubKey []byte `gemm:"requester_pubkey"`
	RequesterNonce  []byte `gemm:"requester_nonce"`

	IssuedEpoch uint64 `gemm:"issued_epoch"`

	M uint64 `gemm:"m"`
	N uint64 `gemm:"n"`
	K uint64 `gemm:"k"`

	MatrixARoot []byte `gemm:"matrix_a_root"`
	MatrixBRoot []byte `gemm:"matrix_b_root"`

	ArithmeticSpec string `gemm:"arithmetic_spec"`
	TileSize       uint64 `gemm:"tile_size"`

	ChallengeWindow uint64 `gemm:"challenge_window"`

	MaxPricePerCWU  uint64 `gemm:"max_price_per_cwu"`
	SettlementAsset string `gemm:"settlement_asset"`
}

// Assignment binds one worker to one task. assignment_id is derived from the
// canonical encoding of this struct and is bound into every ResultCommit,
// trace leaf and receipt to prevent cross-worker and cross-task replay.
type Assignment struct {
	TaskID          []byte `gemm:"task_id"`
	WorkerPubKey    []byte `gemm:"worker_pubkey"`
	AssignmentNonce []byte `gemm:"assignment_nonce"`
	AcceptedEpoch   uint64 `gemm:"accepted_epoch"`
}

// ResultCommit is the only normal-path submission of a worker under the
// output-first optimistic commitment. It deliberately does NOT contain a
// full execution trace root; on-demand tile traces exist only inside a
// dispute and only for the disputed tile.
type ResultCommit struct {
	ProtocolVersion   string `gemm:"protocol_version"`
	TaskID            []byte `gemm:"task_id"`
	AssignmentID      []byte `gemm:"assignment_id"`
	WorkerPubKey      []byte `gemm:"worker_pubkey"`
	OutputRoot        []byte `gemm:"output_root"`
	CanonicalMACCount uint64 `gemm:"canonical_mac_count"`
	CompletedEpoch    uint64 `gemm:"completed_epoch"`
	WorkerSignature   []byte `gemm:"worker_signature"`
}

// OutputTileProof proves that one int32 tile belongs to a committed
// output_root. Index and Count bind the tile position; Siblings is the
// bottom-up sibling path.
type OutputTileProof struct {
	Index    uint32   `gemm:"index"`
	Count    uint32   `gemm:"count"`
	Siblings [][]byte `gemm:"siblings"`
}

// InputTileProof proves one int8 tile belongs to a matrix root. TileCols is
// the tile-grid width of that matrix so the leaf position is bound.
type InputTileProof struct {
	MatrixID uint32   `gemm:"matrix_id"` // 1 = A, 2 = B
	TileRow  uint32   `gemm:"tile_row"`
	TileCol  uint32   `gemm:"tile_col"`
	TileCols uint32   `gemm:"tile_cols"`
	Count    uint32   `gemm:"count"`
	Siblings [][]byte `gemm:"siblings"`
}

// ChallengeOpen disputes one output tile. It is only valid after the
// coordinator verifies that WorkerOutputTile is a member of the worker's
// committed output_root and that it differs from ChallengerOutputTile.
type ChallengeOpen struct {
	ProtocolVersion  string `gemm:"protocol_version"`
	TaskID           []byte `gemm:"task_id"`
	AssignmentID     []byte `gemm:"assignment_id"`
	ChallengerPubKey []byte `gemm:"challenger_pubkey"`
	WorkerPubKey     []byte `gemm:"worker_pubkey"`

	DisputedTileI uint64 `gemm:"disputed_tile_i"`
	DisputedTileJ uint64 `gemm:"disputed_tile_j"`

	WorkerOutputTile     []byte          `gemm:"worker_output_tile"`
	WorkerOutputProof    OutputTileProof `gemm:"worker_output_proof"`
	ChallengerOutputTile []byte          `gemm:"challenger_output_tile"`

	ChallengeBond uint64 `gemm:"challenge_bond"`
	OpenedEpoch   uint64 `gemm:"opened_epoch"`

	ChallengerSignature []byte `gemm:"challenger_signature"`
}

// TraceClaim locks a party's tile trace: the root plus membership proofs for
// the fixed S0 (zero matrix) and S_R endpoints. After both parties lock, the
// trace is immutable for the rest of the dispute.
type TraceClaim struct {
	Party        Party           `gemm:"-"`
	TraceRoot    Hash            `gemm:"-"`
	InitialState []byte          `gemm:"-"`
	InitialProof OutputTileProof `gemm:"-"`
	FinalState   []byte          `gemm:"-"`
	FinalProof   OutputTileProof `gemm:"-"`
}

// TraceCommit is the signed wire form of a TraceClaim.
type TraceCommit struct {
	ProtocolVersion string `gemm:"protocol_version"`
	TaskID          []byte `gemm:"task_id"`
	AssignmentID    []byte `gemm:"assignment_id"`

	DisputedTileI uint64 `gemm:"disputed_tile_i"`
	DisputedTileJ uint64 `gemm:"disputed_tile_j"`

	Party uint64 `gemm:"party"`

	TraceRoot    []byte          `gemm:"trace_root"`
	InitialState []byte          `gemm:"initial_state"`
	InitialProof OutputTileProof `gemm:"initial_proof"`
	FinalState   []byte          `gemm:"final_state"`
	FinalProof   OutputTileProof `gemm:"final_proof"`

	LockedEpoch uint64 `gemm:"locked_epoch"`
	Signature   []byte `gemm:"signature"`
}

// MidStateSubmit is one bisection round submission: the claimed state at the
// current midpoint with an inclusion proof against the party's locked root.
type MidStateSubmit struct {
	Party Party
	Step  uint32
	State []byte
	Proof OutputTileProof
	Epoch uint64
}

// ArbitrationInputs carry everything the micro-step arbiter needs. The
// arbiter never re-executes the full task: it checks the two 8x8 input tiles
// against the matrix roots and recomputes exactly one 8x8x8 micro-step
// (512 canonical MACs).
type ArbitrationInputs struct {
	Step  uint32
	ATile []int8
	BTile []int8

	AProof InputTileProof
	BProof InputTileProof

	WorkerHigh     []byte
	ChallengerHigh []byte
}

// VerifiedWorkReceipt is the settlement artifact of a FINALIZED task. A full
// execution root is intentionally absent: under v0.1.1 the normal path never
// commits one. DisputeTranscriptDigest is a zero-length byte string when the
// task finished unchallenged.
type VerifiedWorkReceipt struct {
	ProtocolVersion string `gemm:"protocol_version"`

	TaskID       []byte `gemm:"task_id"`
	AssignmentID []byte `gemm:"assignment_id"`
	WorkerPubKey []byte `gemm:"worker_pubkey"`

	Operator          string `gemm:"operator"`
	CanonicalMACCount uint64 `gemm:"canonical_mac_count"`
	OutputRoot        []byte `gemm:"output_root"`

	VerificationMode string `gemm:"verification_mode"`

	FinalizedEpoch uint64 `gemm:"finalized_epoch"`

	SettlementReference     []byte `gemm:"settlement_reference"`
	DisputeTranscriptDigest []byte `gemm:"dispute_transcript_digest"`
}

// Admission limits. K is bounded so that no accumulator of up to K products
// of two int8 values can overflow a signed int32, which makes every partial
// sum in any loop order bit-exact. The MAC and output-size caps are generous
// DoS guards for task admission, not consensus limits.
const (
	// MaxSafeK bounds K so that the worst-case accumulator of K products
	// cannot overflow a signed int32. The largest |int8*int8| product is
	// (-128)*(-128) = 16384, so the bound is (2^31-1)/16384 = 131071. The
	// earlier 127*127 divisor was wrong: a task with K in (131071, 133162]
	// and adversarial -128 x -128 inputs could wrap the accumulator.
	MaxSafeK = math.MaxInt32 / (128 * 128) // 131071: 16384 * MaxSafeK <= 2^31-1
	// MaxOutputBytes caps the int32 output buffer admitted per task (4 GiB).
	MaxOutputBytes = uint64(1) << 32
	// MaxMACCount caps canonical_mac_count per task.
	MaxMACCount = uint64(1) << 48
)

// MatrixIDA and MatrixIDB are the distinct matrix identifiers mixed into
// input tile leaves.
const (
	MatrixIDA byte = 0x01
	MatrixIDB byte = 0x02
)
