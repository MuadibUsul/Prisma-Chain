package gemmv1

// Deterministic dispute transcript hash chain for on-chain use.
//
// The in-memory GEMMDispute transcript is a local transparency log, not a
// consensus structure. A chain that persists DisputeTranscriptDigest in a
// VerifiedWorkReceipt must instead maintain this versioned hash chain over
// the canonical protocol events it observes, so the digest is identical
// across node restarts:
//
//	H0     = zero digest
//	H(n+1) = SHA256(DomainDisputeTranscriptV1 || Hn || tag || canonical_event)

import "crypto/sha256"

// DomainDisputeTranscriptV1 separates transcript-chain hashes from every
// other GEMM hash space.
const DomainDisputeTranscriptV1 = "PRISMA_GEMM_TRANSCRIPT_V1\x00"

// Transcript event payloads. All fields are always encoded; parties and
// outcomes use the protocol's numeric identifiers.
type TranscriptChallengeOpened struct {
	TaskID           []byte `gemm:"task_id"`
	AssignmentID     []byte `gemm:"assignment_id"`
	TileI            uint32 `gemm:"tile_i"`
	TileJ            uint32 `gemm:"tile_j"`
	ChallengerPubKey []byte `gemm:"challenger_pub_key"`
	Bond             uint64 `gemm:"bond"`
}

type TranscriptTraceLocked struct {
	Party     uint64 `gemm:"party"`
	TraceRoot []byte `gemm:"trace_root"`
}

type TranscriptBisectionResolved struct {
	Step            uint32 `gemm:"step"`
	KeptLow         uint64 `gemm:"kept_low"` // 1 = interval moved low, 0 = moved high
	WorkerState     []byte `gemm:"worker_state"`
	ChallengerState []byte `gemm:"challenger_state"`
}

type TranscriptTimeout struct {
	Outcome uint64 `gemm:"outcome"`
}

type TranscriptArbitration struct {
	Step     uint32 `gemm:"step"`
	Outcome  uint64 `gemm:"outcome"`
	Expected []byte `gemm:"expected"`
}

// TranscriptStep advances the hash chain by one event. The event tag is a
// short ASCII label ending without separators; it is length-prefixed by the
// canonical encoding of a one-field struct to keep concatenation
// unambiguous.
func TranscriptStep(prev [32]byte, tag string, event any) ([32]byte, error) {
	tagEnc, err := EncodeCanonical(struct {
		Tag string `gemm:"tag"`
	}{Tag: tag})
	if err != nil {
		return [32]byte{}, err
	}
	eventEnc, err := EncodeCanonical(event)
	if err != nil {
		return [32]byte{}, err
	}
	h := sha256.New()
	h.Write([]byte(DomainDisputeTranscriptV1))
	h.Write(prev[:])
	h.Write(tagEnc)
	h.Write(eventEnc)
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out, nil
}

// Known transcript event tags.
const (
	TranscriptTagChallengeOpened = "challenge_opened"
	TranscriptTagTraceLocked     = "trace_locked"
	TranscriptTagBisection       = "bisection_resolved"
	TranscriptTagTimeout         = "timeout"
	TranscriptTagArbitration     = "arbitration"
)
