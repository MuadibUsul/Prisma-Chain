package verify

// Output data availability for v0.1.2. The v0.1.1 ResultCommit is frozen;
// availability is a separate, versioned descriptor so the v0.1.1 receipt
// hash and commit hash are untouched.
//
// A worker that commits an output_root but cannot serve the corresponding
// C during the challenge window must not finalize as
// optimistic_unchallenged: the task routes to the refund path instead.

import (
	"errors"
	"fmt"

	"prismachain/compute/gemmv1"
)

// DomainOutputAvailability is the v0.1.2 availability descriptor domain.
const DomainOutputAvailability = "PRISMA_GEMM_OUTPUT_AVAILABILITY_V1\x00"

// Sentinel verification failures. Callers must distinguish them: a data
// problem is not worker fraud evidence, it only blocks optimistic
// finalization.
var (
	// ErrDataUnavailable: the worker did not serve the committed C.
	ErrDataUnavailable = errors.New("verify: output data unavailable in the challenge window")
	// ErrCommitmentMismatch: the served C does not hash to output_root.
	ErrCommitmentMismatch = errors.New("verify: OUTPUT_DATA_COMMITMENT_MISMATCH")
)

// OutputAvailabilityDescriptor is the versioned v0.1.2 object linking a
// ResultCommit to the retrievable output bytes.
type OutputAvailabilityDescriptor struct {
	ProtocolVersion string `gemm:"protocol_version"`
	TaskID          []byte `gemm:"task_id"`
	AssignmentID    []byte `gemm:"assignment_id"`
	OutputRoot      []byte `gemm:"output_root"`
	OutputBytes     uint64 `gemm:"output_bytes"`
	OutputDataRef   string `gemm:"output_data_ref"`
}

// NewOutputAvailabilityDescriptor fills the fixed fields and validates the
// sizes against the task.
func NewOutputAvailabilityDescriptor(t *gemmv1.TaskDescriptor, assignmentID []byte, outputRoot []byte, outputDataRef string) (*OutputAvailabilityDescriptor, error) {
	taskID, err := t.TaskID()
	if err != nil {
		return nil, err
	}
	if len(assignmentID) != 32 || len(outputRoot) != 32 {
		return nil, fmt.Errorf("verify: assignment_id/output_root must be 32 bytes")
	}
	return &OutputAvailabilityDescriptor{
		ProtocolVersion: ProfileVersion,
		TaskID:          taskID,
		AssignmentID:    append([]byte(nil), assignmentID...),
		OutputRoot:      append([]byte(nil), outputRoot...),
		OutputBytes:     t.M * t.N * 4,
		OutputDataRef:   outputDataRef,
	}, nil
}

// OutputFetcher retrieves the committed C for a descriptor. Returning
// ErrDataUnavailable (or any fetch error) marks the data unavailable for
// this challenge window.
type OutputFetcher interface {
	Fetch(desc *OutputAvailabilityDescriptor) ([]int32, error)
}

// FetcherFunc adapts a function to OutputFetcher.
type FetcherFunc func(desc *OutputAvailabilityDescriptor) ([]int32, error)

// Fetch implements OutputFetcher.
func (f FetcherFunc) Fetch(desc *OutputAvailabilityDescriptor) ([]int32, error) {
	return f(desc)
}

// ResolveOutput fetches C, verifies that it matches the descriptor's
// committed output_root (invariant 7) and returns it. Any mismatch is
// ErrCommitmentMismatch; the returned C must never be trusted as the
// worker's committed output in that case.
func ResolveOutput(t *gemmv1.TaskDescriptor, assignmentID []byte, desc *OutputAvailabilityDescriptor, f OutputFetcher) ([]int32, error) {
	if f == nil {
		return nil, ErrDataUnavailable
	}
	c, err := f.Fetch(desc)
	if err != nil {
		return nil, ErrDataUnavailable
	}
	if uint64(len(c)) != t.M*t.N {
		return nil, fmt.Errorf("%w: length %d != M*N %d", ErrCommitmentMismatch, len(c), t.M*t.N)
	}
	tiles := gemmv1.OutputTiles(c, t.M, t.N)
	counts := gemmv1.TileCountsFor(t.M, t.N, t.K)
	leaves := gemmv1.OutputLeaves(desc.TaskID, desc.AssignmentID, tiles, uint32(counts.ColsC))
	root, err := gemmv1.MerkleRoot(leaves)
	if err != nil {
		return nil, err
	}
	if root != (gemmv1.Hash)(array32(desc.OutputRoot)) {
		return nil, ErrCommitmentMismatch
	}
	return c, nil
}

// CanFinalizeOptimistic reports whether the task may finalize as
// optimistic_unchallenged. Data-unavailable or commitment-mismatched
// outputs must not finalize normally (invariant 8); callers route those to
// the refund path. Bond penalties stay a Phase D decision.
func CanFinalizeOptimistic(t *gemmv1.TaskDescriptor, assignmentID []byte, desc *OutputAvailabilityDescriptor, f OutputFetcher) (bool, error) {
	if _, err := ResolveOutput(t, assignmentID, desc, f); err != nil {
		if errors.Is(err, ErrDataUnavailable) || errors.Is(err, ErrCommitmentMismatch) {
			return false, err
		}
		return false, err
	}
	return true, nil
}

func array32(b []byte) [32]byte {
	var out [32]byte
	copy(out[:], b)
	return out
}
