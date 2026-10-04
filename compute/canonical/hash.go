package canonical

// Domain-separated hashing for CANONICAL_GRAPH_V1. Independent version
// space from GEMM v1; GEMM's frozen hashes are reused only where a graph
// node literally calls GEMM_INT8_V1, whose commitments keep their own
// domains.

import (
	"crypto/sha256"
)

const (
	DomainTensor     = "PRISMA_CANONICAL_TENSOR_V1\x00"
	DomainTensorRoot = "PRISMA_CANONICAL_TENSOR_ROOT_V1\x00"
	DomainGraph      = "PRISMA_CANONICAL_GRAPH_V1\x00"
	DomainGraphState = "PRISMA_CANONICAL_GRAPH_STATE_V1\x00"
	DomainGraphTrace = "PRISMA_CANONICAL_GRAPH_TRACE_V1\x00"
	DomainStageState = "PRISMA_CANONICAL_STAGE_STATE_V1\x00"
	DomainVWR        = "PRISMA_GRAPH_VWR_V1\x00"
	DomainSig        = "PRISMA_CANONICAL_SIG_V1\x00"
	DomainWorkVector = "PRISMA_CANONICAL_WORK_VECTOR_V1\x00"
)

// Hash is a SHA-256 digest value type.
type Hash [32]byte

func hashBytes(parts ...[]byte) Hash {
	h := sha256.New()
	for _, p := range parts {
		h.Write(p)
	}
	var out Hash
	copy(out[:], h.Sum(nil))
	return out
}

func canonicalObjectHash(domain string, v any) (Hash, error) {
	enc, err := EncodeCanonical(v)
	if err != nil {
		return Hash{}, err
	}
	return hashBytes([]byte(domain), enc), nil
}

func appendUint32BE(dst []byte, v uint32) []byte {
	return append(dst, byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
}

// Dispute parties and outcomes (same vocabulary as the rest of the repo).
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

func partyIndex(p Party) (int, error) {
	switch p {
	case Worker:
		return 0, nil
	case Challenger:
		return 1, nil
	default:
		return 0, errorsNewParty(p)
	}
}

func errorsNewParty(p Party) error {
	return &partyError{p}
}

type partyError struct{ p Party }

func (e *partyError) Error() string { return "canonical: invalid party" }
