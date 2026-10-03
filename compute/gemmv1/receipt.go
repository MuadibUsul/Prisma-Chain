package gemmv1

// Verified Work Receipt: the settlement artifact of a FINALIZED GEMM task.
// No full execution root exists in v0.1.1; only a dispute transcript digest
// is optionally attached.

import "errors"

var errBadMode = errors.New("gemmv1: unknown verification_mode")

// BuildVerifiedWorkReceipt assembles the receipt for a finalized task from
// its validated ResultCommit. verificationMode must be one of the two v0.1.1
// modes; settlementReference is chain-specific and stays empty in Phase A-C.
func BuildVerifiedWorkReceipt(t *TaskDescriptor, a *Assignment, rc *ResultCommit, verificationMode string, finalizedEpoch uint64, settlementReference, transcriptDigest []byte) (*VerifiedWorkReceipt, error) {
	if verificationMode != ModeOptimisticUnchallenged && verificationMode != ModeChallengedWorkerWon {
		return nil, errBadMode
	}
	if err := ValidateResultCommit(t, a, rc); err != nil {
		return nil, err
	}
	// A receipt may only be built over a validly signed ResultCommit; this
	// binds output_root and canonical_mac_count to the worker's signature.
	if !VerifyResultCommitSignature(rc) {
		return nil, errors.New("gemmv1: ResultCommit signature does not verify")
	}
	taskID := append([]byte(nil), rc.TaskID...)
	assignmentID := append([]byte(nil), rc.AssignmentID...)
	if transcriptDigest != nil && len(transcriptDigest) != 32 {
		return nil, errBadLength("dispute_transcript_digest", 32, len(transcriptDigest))
	}
	return &VerifiedWorkReceipt{
		ProtocolVersion:         ProtocolVersion,
		TaskID:                  taskID,
		AssignmentID:            assignmentID,
		WorkerPubKey:            append([]byte(nil), rc.WorkerPubKey...),
		Operator:                Operator,
		CanonicalMACCount:       rc.CanonicalMACCount,
		OutputRoot:              append([]byte(nil), rc.OutputRoot...),
		VerificationMode:        verificationMode,
		FinalizedEpoch:          finalizedEpoch,
		SettlementReference:     append([]byte(nil), settlementReference...),
		DisputeTranscriptDigest: append([]byte(nil), transcriptDigest...),
	}, nil
}

// ReceiptID derives the receipt identifier.
func (r *VerifiedWorkReceipt) ReceiptID() ([]byte, error) {
	h, err := canonicalObjectHash(DomainVWR, r)
	if err != nil {
		return nil, err
	}
	return h[:], nil
}
