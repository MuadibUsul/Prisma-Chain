package canonical

// VerifiedGraphWorkReceiptV2 (Phase F.1): V1 plus the node-output manifest
// root. Additive type — VerifiedGraphWorkReceiptV1, its hashing and every
// V1 receipt id remain exactly as they are.

import (
	"errors"
)

// VerifiedGraphWorkReceiptV2 is the canonical receipt object of a V2 task
// (flat field list: the canonical encoder has no embedded-struct form).
type VerifiedGraphWorkReceiptV2 struct {
	ProtocolVersion string `gemm:"protocol_version"`

	GraphID []byte `gemm:"graph_id"`
	Spec    string `gemm:"spec"`

	TaskRef       []byte `gemm:"task_ref"`
	AssignmentRef []byte `gemm:"assignment_ref"`
	WorkerPubKey  []byte `gemm:"worker_pubkey"`

	NodeOutputManifestRoot []byte `gemm:"node_output_manifest_root"`

	FinalOutputRoot []byte   `gemm:"final_output_root"`
	OutputRoots     [][]byte `gemm:"output_roots"`

	WorkVector WorkVector `gemm:"work_vector"`

	VerificationMode string `gemm:"verification_mode"`

	FinalizedEpoch uint64 `gemm:"finalized_epoch"`

	SettlementReference     []byte `gemm:"settlement_reference"`
	DisputeTranscriptDigest []byte `gemm:"dispute_transcript_digest"`
}

// BuildVerifiedGraphWorkReceiptV2 assembles the receipt of a finalized V2
// graph task. The manifest root must equal the commitment validated at
// result submission.
func BuildVerifiedGraphWorkReceiptV2(g *GraphDescriptor, taskRef, assignmentRef, workerPubKey []byte,
	outputs []Hash, work WorkVector, manifestRoot []byte, mode string, finalizedEpoch uint64,
	settlementRef, transcriptDigest []byte) (*VerifiedGraphWorkReceiptV2, error) {
	if len(manifestRoot) != 32 {
		return nil, errors.New("canonical: V2 receipt needs a 32-byte manifest root")
	}
	base, err := BuildVerifiedGraphWorkReceiptV1(g, taskRef, assignmentRef, workerPubKey,
		outputs, work, mode, finalizedEpoch, settlementRef, transcriptDigest)
	if err != nil {
		return nil, err
	}
	return &VerifiedGraphWorkReceiptV2{
		ProtocolVersion: base.ProtocolVersion,
		GraphID:         base.GraphID, Spec: base.Spec,
		TaskRef: base.TaskRef, AssignmentRef: base.AssignmentRef, WorkerPubKey: base.WorkerPubKey,
		NodeOutputManifestRoot: append([]byte(nil), manifestRoot...),
		FinalOutputRoot:        base.FinalOutputRoot, OutputRoots: base.OutputRoots,
		WorkVector:              base.WorkVector,
		VerificationMode:        base.VerificationMode,
		FinalizedEpoch:          base.FinalizedEpoch,
		SettlementReference:     base.SettlementReference,
		DisputeTranscriptDigest: base.DisputeTranscriptDigest,
	}, nil
}

// ReceiptID derives the canonical V2 receipt identifier (its own domain:
// V1 receipt ids are untouched).
func (r *VerifiedGraphWorkReceiptV2) ReceiptID() (Hash, error) {
	return canonicalObjectHash(DomainReceiptV2, r)
}
