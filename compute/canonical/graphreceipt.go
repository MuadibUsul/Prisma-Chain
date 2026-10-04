package canonical

// VerifiedGraphWorkReceiptV1: the settlement artifact of a FINALIZED graph
// task. It is derived by the chain from the validated result commitment
// (the old gemmv1 VerifiedWorkReceipt is untouched); authenticity comes
// from the committed result signature plus the graph id, so the receipt
// carries no signature of its own.

import (
	"errors"
)

// Graph receipt verification modes.
const (
	ModeOptimisticUnchallenged = "optimistic_unchallenged"
	ModeChallengedWorkerWon    = "challenged_worker_won"
)

// VerifiedGraphWorkReceiptV1 is the canonical receipt object.
type VerifiedGraphWorkReceiptV1 struct {
	ProtocolVersion string `gemm:"protocol_version"`

	GraphID []byte `gemm:"graph_id"`
	Spec    string `gemm:"spec"`

	TaskRef       []byte `gemm:"task_ref"`
	AssignmentRef []byte `gemm:"assignment_ref"`
	WorkerPubKey  []byte `gemm:"worker_pubkey"`

	FinalOutputRoot []byte   `gemm:"final_output_root"`
	OutputRoots     [][]byte `gemm:"output_roots"`

	WorkVector WorkVector `gemm:"work_vector"`

	VerificationMode string `gemm:"verification_mode"`

	FinalizedEpoch uint64 `gemm:"finalized_epoch"`

	SettlementReference     []byte `gemm:"settlement_reference"`
	DisputeTranscriptDigest []byte `gemm:"dispute_transcript_digest"`
}

// BuildVerifiedGraphWorkReceiptV1 assembles the receipt of a finalized
// graph task. The output roots must hash to the given final output root
// and every commitment is re-derived, never trusted.
func BuildVerifiedGraphWorkReceiptV1(g *GraphDescriptor, taskRef, assignmentRef, workerPubKey []byte,
	outputs []Hash, work WorkVector, mode string, finalizedEpoch uint64,
	settlementRef, transcriptDigest []byte) (*VerifiedGraphWorkReceiptV1, error) {
	if err := g.Validate(); err != nil {
		return nil, err
	}
	if mode != ModeOptimisticUnchallenged && mode != ModeChallengedWorkerWon {
		return nil, errors.New("canonical: unknown graph receipt verification mode")
	}
	if len(workerPubKey) != 32 {
		return nil, errors.New("canonical: graph receipt needs a 32-byte worker key")
	}
	if len(outputs) == 0 {
		return nil, errors.New("canonical: graph receipt needs at least one output root")
	}
	if len(transcriptDigest) != 0 && len(transcriptDigest) != 32 {
		return nil, errors.New("canonical: dispute transcript digest must be 32 bytes")
	}
	graphID, err := g.GraphID()
	if err != nil {
		return nil, err
	}
	finalRoot := FinalOutputRoot(outputs)
	roots := make([][]byte, len(outputs))
	for i, h := range outputs {
		roots[i] = append([]byte(nil), h[:]...)
	}
	return &VerifiedGraphWorkReceiptV1{
		ProtocolVersion:         GraphProtocolVersion,
		GraphID:                 append([]byte(nil), graphID[:]...),
		Spec:                    g.Spec,
		TaskRef:                 append([]byte(nil), taskRef...),
		AssignmentRef:           append([]byte(nil), assignmentRef...),
		WorkerPubKey:            append([]byte(nil), workerPubKey...),
		FinalOutputRoot:         append([]byte(nil), finalRoot[:]...),
		OutputRoots:             roots,
		WorkVector:              work.Canonical(),
		VerificationMode:        mode,
		FinalizedEpoch:          finalizedEpoch,
		SettlementReference:     append([]byte(nil), settlementRef...),
		DisputeTranscriptDigest: append([]byte(nil), transcriptDigest...),
	}, nil
}

// ReceiptID derives the canonical receipt identifier.
func (r *VerifiedGraphWorkReceiptV1) ReceiptID() (Hash, error) {
	return canonicalObjectHash(DomainReceipt, r)
}

// VerifyReceiptBinding re-derives the receipt's commitments against a
// graph and a claimed output root set; used by any party validating a
// receipt it did not build.
func (r *VerifiedGraphWorkReceiptV1) VerifyReceiptBinding(g *GraphDescriptor, outputs []Hash) error {
	if r.ProtocolVersion != GraphProtocolVersion {
		return errors.New("canonical: receipt protocol version mismatch")
	}
	graphID, err := g.GraphID()
	if err != nil {
		return err
	}
	if !equalBytes(r.GraphID, graphID[:]) || r.Spec != g.Spec {
		return errors.New("canonical: receipt does not bind this graph")
	}
	if len(r.OutputRoots) != len(outputs) || len(outputs) == 0 {
		return errors.New("canonical: receipt output set mismatch")
	}
	for i, h := range outputs {
		if !equalBytes(r.OutputRoots[i], h[:]) {
			return errors.New("canonical: receipt output root mismatch")
		}
	}
	finalRoot := FinalOutputRoot(outputs)
	if !equalBytes(r.FinalOutputRoot, finalRoot[:]) {
		return errors.New("canonical: receipt final output root mismatch")
	}
	return nil
}
