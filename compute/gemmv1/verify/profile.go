// Package verify implements the GEMM v0.1.2 cheap verification layer:
// probabilistic Freivalds detection over exact signed int64 arithmetic,
// bad-row/bad-tile localization, and output data availability checks.
//
// Separation of concerns (protocol invariant): Freivalds is a DETECTION
// strategy only. It is never a slashing proof. A mismatch may only route a
// challenger into the deterministic v0.1.1 tile dispute (ChallengeOpen ->
// on-demand trace -> bisection -> 512-MAC arbitration), which remains the
// sole final evidence. This package deliberately does not call
// gemmv1.ReferenceGEMM; its only exact recomputation is one row
// (ReferenceRowGEMM, O(KN)).
package verify

// VerificationProfile names the detection algorithm and round count.
type VerificationProfile struct {
	Algorithm string `gemm:"algorithm"`
	Rounds    uint16 `gemm:"rounds"`
}

// AlgorithmFreivaldsBinaryV1 is the only v0.1.2 detection algorithm.
const AlgorithmFreivaldsBinaryV1 = "FREIVALDS_BINARY_V1"

// ProfileVersion is the version string carried by v0.1.2 protocol objects.
const ProfileVersion = "0.1.2"

// VerificationResult reports the outcome of one fast verification run.
// Duration is wall-clock information only; it never enters a protocol hash.
type VerificationResult struct {
	Passed           bool
	RoundsExecuted   uint16
	ResidualRows     []uint64
	MismatchRound    int
	VerificationMACs uint64
	Duration         float64 // milliseconds, informational
}

// FalseAcceptUpperBound is the theoretical false-accept probability of the
// profile: for a fixed incorrect C, each independent round rejects with
// probability >= 1/2, so q rounds accept with probability <= 2^-q. This
// bounds the DETECTION layer only; it is not the overall attack success
// probability of the system, which is governed by the deterministic
// dispute that must also be won by a fraudulent worker.
func (p *VerificationProfile) FalseAcceptUpperBound() float64 {
	bound := 1.0
	for i := uint16(0); i < p.Rounds; i++ {
		bound /= 2
	}
	return bound
}
