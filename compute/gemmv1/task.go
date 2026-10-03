package gemmv1

import (
	"errors"
	"fmt"
)

// Task construction, validation and admission for GEMM_INT8_V1.

var (
	errDims             = errors.New("gemmv1: matrix dimensions must be at least 1")
	errTileSize         = errors.New("gemmv1: tile_size must be 8")
	errArithmetic       = errors.New("gemmv1: arithmetic_spec must be INT8_INT32_V1")
	errOperator         = errors.New("gemmv1: operator must be GEMM_INT8_V1")
	errVersion          = errors.New("gemmv1: protocol_version mismatch")
	errKUnsafe          = errors.New("gemmv1: K exceeds the int32-safe accumulator bound")
	errMACTooLarge      = errors.New("gemmv1: canonical_mac_count exceeds the admission cap")
	errOutputTooLarge   = errors.New("gemmv1: output buffer exceeds the admission cap")
	errWindow           = errors.New("gemmv1: challenge_window must be at least 1")
	errAsset            = errors.New("gemmv1: settlement_asset must be uprsm")
	errNonce            = errors.New("gemmv1: requester_nonce must be 1..64 bytes")
	errBadChallengeTile = errors.New("gemmv1: disputed tile coordinates out of range")
)

// NewTaskDescriptor fills the fixed fields of a TaskDescriptor and validates
// it. matrix roots are supplied by the caller after committing A and B.
func NewTaskDescriptor(requesterPubKey, requesterNonce []byte, issuedEpoch, m, n, k uint64, matrixARoot, matrixBRoot []byte, challengeWindow, maxPricePerCWU uint64) (*TaskDescriptor, error) {
	t := &TaskDescriptor{
		ProtocolVersion: ProtocolVersion,
		Operator:        Operator,
		RequesterPubKey: append([]byte(nil), requesterPubKey...),
		RequesterNonce:  append([]byte(nil), requesterNonce...),
		IssuedEpoch:     issuedEpoch,
		M:               m,
		N:               n,
		K:               k,
		MatrixARoot:     append([]byte(nil), matrixARoot...),
		MatrixBRoot:     append([]byte(nil), matrixBRoot...),
		ArithmeticSpec:  ArithmeticSpec,
		TileSize:        TileSize,
		ChallengeWindow: challengeWindow,
		MaxPricePerCWU:  maxPricePerCWU,
		SettlementAsset: SettlementAssetUPRSM,
	}
	if err := t.Validate(); err != nil {
		return nil, err
	}
	return t, nil
}

// Validate checks the internal consistency of a TaskDescriptor.
func (t *TaskDescriptor) Validate() error {
	if t.ProtocolVersion != ProtocolVersion {
		return errVersion
	}
	if t.Operator != Operator {
		return errOperator
	}
	if t.ArithmeticSpec != ArithmeticSpec {
		return errArithmetic
	}
	if t.TileSize != TileSize {
		return errTileSize
	}
	if t.SettlementAsset != SettlementAssetUPRSM {
		return errAsset
	}
	if t.M == 0 || t.N == 0 || t.K == 0 {
		return errDims
	}
	if t.ChallengeWindow == 0 {
		return errWindow
	}
	if len(t.RequesterPubKey) != 32 {
		return errBadPubKey
	}
	if len(t.RequesterNonce) == 0 || len(t.RequesterNonce) > 64 {
		return errNonce
	}
	if err := checkHashField("matrix_a_root", t.MatrixARoot); err != nil {
		return err
	}
	if err := checkHashField("matrix_b_root", t.MatrixBRoot); err != nil {
		return err
	}
	if t.K > MaxSafeK {
		return errKUnsafe
	}
	if mac := t.CanonicalMACCount(); mac > MaxMACCount {
		return errMACTooLarge
	}
	// Guard M*N*4 against uint64 overflow before the output cap.
	if t.M > MaxOutputBytes/4 || t.N > (MaxOutputBytes/4)/t.M {
		return errOutputTooLarge
	}
	return nil
}

// CanonicalMACCount is M x N x K: the only work measure of v0.1.1. Padding
// tiles contribute nothing because the count is defined over real dimensions.
func (t *TaskDescriptor) CanonicalMACCount() uint64 {
	return t.M * t.N * t.K
}

// TaskID derives the protocol task identifier.
func (t *TaskDescriptor) TaskID() ([]byte, error) {
	h, err := canonicalObjectHash(DomainTask, t)
	if err != nil {
		return nil, err
	}
	return h[:], nil
}

// AssignmentID derives the assignment identifier binding worker to task.
func (a *Assignment) AssignmentID() ([]byte, error) {
	if err := checkHashField("task_id", a.TaskID); err != nil {
		return nil, err
	}
	if err := checkPubKey(a.WorkerPubKey); err != nil {
		return nil, err
	}
	if len(a.AssignmentNonce) == 0 || len(a.AssignmentNonce) > 64 {
		return nil, errNonce
	}
	h, err := canonicalObjectHash(DomainAssignment, a)
	if err != nil {
		return nil, err
	}
	return h[:], nil
}

// CWUnits converts canonical MACs to human-display CWU (1 CWU = 2^20 MACs).
// It is display-only; the ledger stores canonical_mac_count.
func CWUnits(mac uint64) float64 {
	const macPerCWU = 1 << 20
	return float64(mac) / macPerCWU
}

// ValidateResultCommit binds a ResultCommit to the task and assignment it
// claims, enforcing replay protection: a commit from task A cannot be used
// for task B and a commit from worker A cannot be used under worker B's
// assignment.
func ValidateResultCommit(t *TaskDescriptor, a *Assignment, rc *ResultCommit) error {
	if rc.ProtocolVersion != ProtocolVersion {
		return errVersion
	}
	taskID, err := t.TaskID()
	if err != nil {
		return err
	}
	assignmentID, err := a.AssignmentID()
	if err != nil {
		return err
	}
	if !equalBytes(rc.TaskID, taskID) {
		return errors.New("gemmv1: ResultCommit task_id does not match the task")
	}
	if !equalBytes(rc.AssignmentID, assignmentID) {
		return errors.New("gemmv1: ResultCommit assignment_id does not match the assignment")
	}
	if !equalBytes(rc.WorkerPubKey, a.WorkerPubKey) {
		return errors.New("gemmv1: ResultCommit worker does not match the assignment")
	}
	if err := checkPubKey(rc.WorkerPubKey); err != nil {
		return err
	}
	if err := checkHashField("output_root", rc.OutputRoot); err != nil {
		return err
	}
	if rc.CanonicalMACCount != t.CanonicalMACCount() {
		return fmt.Errorf("gemmv1: canonical_mac_count %d != M*N*K %d", rc.CanonicalMACCount, t.CanonicalMACCount())
	}
	return nil
}

// ValidateChallengeOpen checks coordinates and tile encoding of a challenge
// before any Merkle verification; membership of the worker tile in the
// committed output_root is verified separately by the coordinator because it
// needs the output root from the ResultCommit.
func ValidateChallengeOpen(t *TaskDescriptor, c *ChallengeOpen) error {
	if c.ProtocolVersion != ProtocolVersion {
		return errVersion
	}
	if err := checkHashField("task_id", c.TaskID); err != nil {
		return err
	}
	if err := checkHashField("assignment_id", c.AssignmentID); err != nil {
		return err
	}
	if err := checkPubKey(c.ChallengerPubKey); err != nil {
		return err
	}
	if err := checkPubKey(c.WorkerPubKey); err != nil {
		return err
	}
	if c.DisputedTileI >= tileRows(t.M) || c.DisputedTileJ >= tileCols(t.N) {
		return errBadChallengeTile
	}
	if len(c.WorkerOutputTile) != intTileBytes || len(c.ChallengerOutputTile) != intTileBytes {
		return errBadLength("output tiles", intTileBytes, len(c.WorkerOutputTile))
	}
	if equalBytes(c.WorkerOutputTile, c.ChallengerOutputTile) {
		return errors.New("gemmv1: challenge must dispute a differing tile")
	}
	if c.ChallengeBond == 0 {
		return errors.New("gemmv1: challenge_bond must be positive")
	}
	return nil
}

func equalBytes(a, b []byte) bool {
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
