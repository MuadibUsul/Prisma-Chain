package vm

import "errors"

// ReviewAction is the result of an independent, complete public-task replay.
// A trace with the correct final state but a different interior commitment is
// withheld rather than challenged: the v1 dispute game requires different
// final states to identify a disputed step.
type ReviewAction string

const (
	Attest    ReviewAction = "attest"
	Challenge ReviewAction = "challenge"
	Withhold  ReviewAction = "withhold"
)

type Review struct {
	Action ReviewAction
	// Canonical is the monitor's endpoint claim. Submit it only when Action is
	// Challenge; retain Trace to answer later midpoint requests.
	Canonical TraceClaim
	Trace     Execution
}

// ReviewClaim replays every instruction and verifies the worker's committed
// endpoints before deciding whether to attest, challenge, or withhold. Its
// caller must first bind the program, input and worker root to the on-chain
// model and task commitments. It performs no chain transaction and makes no
// data-availability claim.
func ReviewClaim(program Program, input []int64, worker TraceClaim) (Review, error) {
	trace, err := Execute(program, input)
	if err != nil {
		return Review{}, err
	}
	count := trace.Steps() + 1
	if !VerifyProof(worker.Root, State{}, 0, count, worker.InitialProof) ||
		!VerifyProof(worker.Root, worker.Final, trace.Steps(), count, worker.FinalProof) {
		return Review{}, errors.New("worker endpoint proofs are invalid")
	}
	_, initialProof, err := trace.Proof(0)
	if err != nil {
		return Review{}, err
	}
	final, finalProof, err := trace.Proof(trace.Steps())
	if err != nil {
		return Review{}, err
	}
	review := Review{
		Canonical: TraceClaim{Root: trace.Root(), InitialProof: initialProof, Final: final, FinalProof: finalProof},
		Trace:     trace,
	}
	switch {
	case worker.Root == trace.Root() && worker.Final == final:
		review.Action = Attest
	case worker.Final != final:
		review.Action = Challenge
	default:
		review.Action = Withhold
	}
	return review, nil
}
