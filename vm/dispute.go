package vm

import (
	"errors"
	"math"
)

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

// TraceClaim proves both endpoints of one committed execution. Authentication
// of the worker and challenger identities belongs to the chain module.
type TraceClaim struct {
	Root         Hash
	InitialProof Proof
	Final        State
	FinalProof   Proof
}

type Status struct {
	Low, High           uint32
	Mid                 uint32
	Deadline            uint64
	WorkerSubmitted     bool
	ChallengerSubmitted bool
	Outcome             Outcome
}

// Dispute narrows conflicting committed traces to the first differing step.
// Every submitted midpoint must carry an inclusion proof against its party's
// fixed root. The final decision executes just one VM instruction.
type Dispute struct {
	program          Program
	input            []int64
	roots            [2]Hash
	low, high        uint32
	lowState         State
	highStates       [2]State
	medians          [2]*State
	period, deadline uint64
	lastHeight       uint64
	outcome          Outcome
}

func NewDispute(p Program, input []int64, worker, challenger TraceClaim, height, roundPeriod uint64) (*Dispute, error) {
	if err := validateInput(p, input); err != nil {
		return nil, err
	}
	rounds := uint64(0)
	for width := uint32(p.Len()); width > 1; width = (width + 1) / 2 {
		rounds++
	}
	if rounds == 0 {
		rounds = 1
	}
	if roundPeriod == 0 || height >= math.MaxUint64 || roundPeriod > (math.MaxUint64-height-1)/rounds {
		return nil, errors.New("invalid round period or deadline overflow")
	}
	count := uint32(p.Len() + 1)
	initial := State{}
	for _, claim := range []TraceClaim{worker, challenger} {
		if !VerifyProof(claim.Root, initial, 0, count, claim.InitialProof) ||
			!VerifyProof(claim.Root, claim.Final, count-1, count, claim.FinalProof) {
			return nil, errors.New("invalid endpoint proof")
		}
	}
	if worker.Final == challenger.Final {
		return nil, errors.New("claims have the same final state")
	}
	d := &Dispute{
		program: p, input: append([]int64(nil), input...),
		roots: [2]Hash{worker.Root, challenger.Root},
		high:  count - 1, highStates: [2]State{worker.Final, challenger.Final},
		period: roundPeriod, deadline: height + roundPeriod, lastHeight: height,
	}
	if d.high == 1 {
		d.resolveStep()
	}
	return d, nil
}

func (d *Dispute) Status() Status {
	s := Status{Low: d.low, High: d.high, Deadline: d.deadline, Outcome: d.outcome}
	if d.outcome == Pending {
		s.Mid = d.low + (d.high-d.low)/2
	}
	s.WorkerSubmitted = d.medians[0] != nil
	s.ChallengerSubmitted = d.medians[1] != nil
	return s
}

// SubmitMid accepts a Merkle proof for the current midpoint. Either party may
// submit first; after both submissions the common prefix or disputed suffix is
// retained. A finalized dispute cannot be changed.
func (d *Dispute) SubmitMid(party Party, state State, proof Proof, height uint64) (Outcome, error) {
	if d.outcome != Pending {
		return d.outcome, errors.New("dispute already resolved")
	}
	if height < d.lastHeight || height > d.deadline {
		return Pending, errors.New("submission outside current round")
	}
	index, err := partyIndex(party)
	if err != nil {
		return Pending, err
	}
	if d.medians[index] != nil {
		return Pending, errors.New("party already submitted this round")
	}
	mid := d.low + (d.high-d.low)/2
	if !VerifyProof(d.roots[index], state, mid, uint32(d.program.Len()+1), proof) {
		return Pending, errors.New("invalid midpoint proof")
	}
	d.lastHeight = height
	copyState := state
	d.medians[index] = &copyState
	if d.medians[0] == nil || d.medians[1] == nil {
		return Pending, nil
	}
	worker, challenger := *d.medians[0], *d.medians[1]
	if worker == challenger {
		d.low, d.lowState = mid, worker
	} else {
		d.high, d.highStates = mid, [2]State{worker, challenger}
	}
	d.medians = [2]*State{}
	if d.high-d.low == 1 {
		d.resolveStep()
		return d.outcome, nil
	}
	d.deadline = height + d.period
	return Pending, nil
}

// Timeout is called after, never at, the deadline. A missing party loses the
// round; if both fail to respond the task needs external refund handling.
func (d *Dispute) Timeout(height uint64) (Outcome, error) {
	if d.outcome != Pending {
		return d.outcome, errors.New("dispute already resolved")
	}
	if height <= d.deadline || height < d.lastHeight {
		return Pending, errors.New("round has not expired")
	}
	switch {
	case d.medians[0] != nil && d.medians[1] == nil:
		d.outcome = WorkerWins
	case d.medians[0] == nil && d.medians[1] != nil:
		d.outcome = ChallengerWins
	default:
		d.outcome = BothInvalid
	}
	d.lastHeight = height
	return d.outcome, nil
}

func (d *Dispute) resolveStep() {
	workerValid := VerifyStep(d.program, d.input, d.lowState, d.highStates[0]) == nil
	challengerValid := VerifyStep(d.program, d.input, d.lowState, d.highStates[1]) == nil
	switch {
	case workerValid:
		d.outcome = WorkerWins
	case challengerValid:
		d.outcome = ChallengerWins
	default:
		d.outcome = BothInvalid
	}
}

func partyIndex(p Party) (int, error) {
	switch p {
	case Worker:
		return 0, nil
	case Challenger:
		return 1, nil
	default:
		return 0, errors.New("invalid party")
	}
}
