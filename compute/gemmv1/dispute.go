package gemmv1

// GEMM dispute: on-demand trace locking, interactive bisection over K-step
// states of one disputed tile, and 8x8x8 micro-step arbitration. The state
// machine mirrors the proven structure of vm/dispute.go but works on 8x8
// int32 tile states instead of VM scalar states.

import (
	"errors"
	"math"
)

var (
	errDisputeResolved  = errors.New("gemmv1: dispute already resolved")
	errDisputeNotReady  = errors.New("gemmv1: dispute is not ready for arbitration")
	errOutsideRound     = errors.New("gemmv1: submission outside the current round")
	errAlreadySubmitted = errors.New("gemmv1: party already submitted this round")
	errInvalidMidProof  = errors.New("gemmv1: invalid midpoint trace proof")
	errRoundNotExpired  = errors.New("gemmv1: round has not expired")
	errInvalidEndpoint  = errors.New("gemmv1: invalid trace endpoint claim")
	errSameFinalState   = errors.New("gemmv1: worker and challenger final states are equal")
	errBadArbTile       = errors.New("gemmv1: arbitration input tile rejected by matrix root")
	errPeriod           = errors.New("gemmv1: invalid round period or deadline overflow")
)

// DisputeConfig fixes the context of one tile dispute. WorkerTile and
// ChallengerTile are the 256-byte canonical tiles from the ChallengeOpen.
type DisputeConfig struct {
	Task           *TaskDescriptor
	Assignment     *Assignment
	TaskID         []byte
	TileI, TileJ   uint32
	WorkerTile     []byte
	ChallengerTile []byte
	MatrixARoot    Hash
	MatrixBRoot    Hash
	RoundPeriod    uint64 // epochs per bisection round
}

// GEMMDispute is the full dispute state for one output tile.
type GEMMDispute struct {
	cfg          DisputeConfig
	assignmentID []byte
	r            uint32
	roots        [2]Hash
	low, high    uint32
	lowState     State
	highStates   [2]State
	medians      [2]*State
	deadline     uint64
	lastEpoch    uint64
	outcome      Outcome
	arbReady     bool
	// LastRoundKeptLow records the direction of the last completed
	// bisection round for the consensus transcript; it is persisted in the
	// snapshot.
	LastRoundKeptLow bool
	transcript       []byte
}

// NewGEMMDispute opens a dispute from a validated ChallengeOpen context and
// both parties' locked TraceClaims. The caller (coordinator/chain) must
// already have verified that the worker tile belongs to the committed
// output_root and that the two tiles differ.
func NewGEMMDispute(cfg DisputeConfig, worker, challenger TraceClaim, openedEpoch uint64) (*GEMMDispute, error) {
	if cfg.RoundPeriod == 0 {
		return nil, errPeriod
	}
	assignmentID, err := cfg.Assignment.AssignmentID()
	if err != nil {
		return nil, err
	}
	if err := checkHashField("task_id", cfg.TaskID); err != nil {
		return nil, err
	}
	r := uint32(RSteps(cfg.Task.K))
	rounds := uint64(0)
	for width := r + 1; width > 1; width = (width + 1) / 2 {
		rounds++
	}
	if rounds == 0 {
		rounds = 1
	}
	if openedEpoch >= math.MaxUint64 || cfg.RoundPeriod > (math.MaxUint64-openedEpoch-1)/rounds {
		return nil, errPeriod
	}
	if err := validateClaim(&cfg, assignmentID, Worker, worker, r); err != nil {
		return nil, err
	}
	if err := validateClaim(&cfg, assignmentID, Challenger, challenger, r); err != nil {
		return nil, err
	}
	workerFinal, _ := StateFromCanonical(worker.FinalState)
	challengerFinal, _ := StateFromCanonical(challenger.FinalState)
	if workerFinal == nil || challengerFinal == nil || *workerFinal == *challengerFinal {
		return nil, errSameFinalState
	}
	d := &GEMMDispute{
		cfg:          cfg,
		assignmentID: assignmentID,
		r:            r,
		roots:        [2]Hash{worker.TraceRoot, challenger.TraceRoot},
		high:         r,
		lowState:     State{},
		highStates:   [2]State{*workerFinal, *challengerFinal},
		deadline:     openedEpoch + cfg.RoundPeriod,
		lastEpoch:    openedEpoch,
	}
	d.appendTranscript("dispute_open", disputeOpenTranscript{
		TaskID:              append([]byte(nil), cfg.TaskID...),
		AssignmentID:        append([]byte(nil), assignmentID...),
		TileI:               cfg.TileI,
		TileJ:               cfg.TileJ,
		WorkerTraceRoot:     worker.TraceRoot[:],
		ChallengerTraceRoot: challenger.TraceRoot[:],
	})
	return d, nil
}

func validateClaim(cfg *DisputeConfig, assignmentID []byte, party Party, claim TraceClaim, r uint32) error {
	tc := &TraceCommit{
		ProtocolVersion: ProtocolVersion,
		TaskID:          append([]byte(nil), cfg.TaskID...),
		AssignmentID:    append([]byte(nil), assignmentID...),
		DisputedTileI:   uint64(cfg.TileI),
		DisputedTileJ:   uint64(cfg.TileJ),
		Party:           uint64(party),
		TraceRoot:       append([]byte(nil), claim.TraceRoot[:]...),
		InitialState:    claim.InitialState,
		InitialProof:    claim.InitialProof,
		FinalState:      claim.FinalState,
		FinalProof:      claim.FinalProof,
	}
	if err := VerifyTraceCommit(cfg.Task, assignmentID, tc); err != nil {
		return errInvalidEndpoint
	}
	expectedFinal := cfg.WorkerTile
	if party == Challenger {
		expectedFinal = cfg.ChallengerTile
	}
	if !equalBytes(claim.FinalState, expectedFinal) {
		return errors.New("gemmv1: locked trace S_R does not match the party's claimed output tile")
	}
	return nil
}

// SubmitMid accepts one bisection round submission. Both parties submit the
// state at the current midpoint with an inclusion proof against their own
// locked trace root. Once both submissions arrive, the interval is halved.
func (d *GEMMDispute) SubmitMid(party Party, state []byte, proof MerkleProof, epoch uint64) (Outcome, error) {
	if d.outcome != Pending {
		return d.outcome, errDisputeResolved
	}
	if d.arbReady {
		return Pending, errDisputeNotReady
	}
	if epoch < d.lastEpoch || epoch > d.deadline {
		return Pending, errOutsideRound
	}
	idx, err := partyIndex(party)
	if err != nil {
		return Pending, err
	}
	if d.medians[idx] != nil {
		return Pending, errAlreadySubmitted
	}
	mid := d.low + (d.high-d.low)/2
	leaf := LeafTraceState(d.cfg.TaskID, d.assignmentID, d.cfg.TileI, d.cfg.TileJ, mid, state)
	if !VerifyLeafInclusion(d.roots[idx], leaf, proof) {
		return Pending, errInvalidMidProof
	}
	s, err := StateFromCanonical(state)
	if err != nil {
		return Pending, err
	}
	d.lastEpoch = epoch
	stored := *s
	d.medians[idx] = &stored
	if d.medians[0] == nil || d.medians[1] == nil {
		return Pending, nil
	}
	workerMid, challengerMid := *d.medians[0], *d.medians[1]
	if workerMid == challengerMid {
		d.low = mid
		d.lowState = workerMid
		d.LastRoundKeptLow = true
	} else {
		d.high = mid
		d.highStates = [2]State{workerMid, challengerMid}
		d.LastRoundKeptLow = false
	}
	d.medians = [2]*State{}
	if d.high-d.low == 1 {
		d.arbReady = true
	}
	d.deadline = epoch + d.cfg.RoundPeriod
	d.appendTranscript("bisection_round", midTranscript{
		Step:            mid,
		WorkerState:     workerMid.CanonicalBytes(),
		ChallengerState: challengerMid.CanonicalBytes(),
	})
	return Pending, nil
}

// Arbitrate resolves the final disputed transition S_low -> S_low+1. The
// arbiter first verifies both 8x8 input tiles against the committed matrix
// roots, then recomputes exactly one micro-step: 512 canonical MACs. It
// never re-executes the task.
func (d *GEMMDispute) Arbitrate(aTile, bTile [int8TileSize]int8, aProof, bProof InputTileProof, epoch uint64) (Outcome, error) {
	if d.outcome != Pending {
		return d.outcome, errDisputeResolved
	}
	if !d.arbReady || d.high-d.low != 1 {
		return Pending, errDisputeNotReady
	}
	if epoch < d.lastEpoch || epoch > d.deadline {
		return Pending, errOutsideRound
	}
	step := d.low
	if !VerifyInputTileProof(d.cfg.MatrixARoot, MatrixIDA, d.cfg.TileI, step, Int8TileBytes(aTile), aProof) {
		return Pending, errBadArbTile
	}
	if !VerifyInputTileProof(d.cfg.MatrixBRoot, MatrixIDB, step, d.cfg.TileJ, Int8TileBytes(bTile), bProof) {
		return Pending, errBadArbTile
	}
	expected := MicroStep(&d.lowState, aTile, bTile)
	workerValid := expected == d.highStates[0]
	challengerValid := expected == d.highStates[1]
	switch {
	case workerValid:
		d.outcome = WorkerWins
	case challengerValid:
		d.outcome = ChallengerWins
	default:
		d.outcome = BothInvalid
	}
	d.lastEpoch = epoch
	d.appendTranscript("arbitration", arbitrationTranscript{
		Step:     step,
		Outcome:  uint64(d.outcome),
		Expected: expected.CanonicalBytes(),
	})
	return d.outcome, nil
}

// Timeout closes an expired round. A party that did not respond loses; if
// both fail the outcome is BothInvalid, which maps to the existing refund
// path of the chain module. A dispute already ready for arbitration cannot
// time out; it waits for the arbiter.
func (d *GEMMDispute) Timeout(epoch uint64) (Outcome, error) {
	if d.outcome != Pending {
		return d.outcome, errDisputeResolved
	}
	if d.arbReady {
		return Pending, errDisputeNotReady
	}
	if epoch <= d.deadline || epoch < d.lastEpoch {
		return Pending, errRoundNotExpired
	}
	switch {
	case d.medians[0] != nil && d.medians[1] == nil:
		d.outcome = WorkerWins
	case d.medians[0] == nil && d.medians[1] != nil:
		d.outcome = ChallengerWins
	default:
		d.outcome = BothInvalid
	}
	d.lastEpoch = epoch
	d.appendTranscript("timeout", timeoutTranscript{Outcome: uint64(d.outcome)})
	return d.outcome, nil
}

// DisputeStatus snapshots the current dispute state.
type DisputeStatus struct {
	Low, High, Mid      uint32
	Deadline            uint64
	WorkerSubmitted     bool
	ChallengerSubmitted bool
	ArbitrationReady    bool
	Outcome             Outcome
}

// Status returns the snapshot.
func (d *GEMMDispute) Status() DisputeStatus {
	s := DisputeStatus{
		Low: d.low, High: d.high, Deadline: d.deadline,
		WorkerSubmitted:     d.medians[0] != nil,
		ChallengerSubmitted: d.medians[1] != nil,
		ArbitrationReady:    d.arbReady,
		Outcome:             d.outcome,
	}
	if !d.arbReady && d.outcome == Pending {
		s.Mid = d.low + (d.high-d.low)/2
	}
	return s
}

// TranscriptDigest is the SHA-256 over every ordered dispute event; it is
// optional evidence in a VerifiedWorkReceipt.
func (d *GEMMDispute) TranscriptDigest() []byte {
	h := hashBytes([]byte(DomainSig), d.transcript)
	return h[:]
}

// appendTranscript records one event with a tag prefix; it is a local
// transparency log, not a consensus structure.
func (d *GEMMDispute) appendTranscript(tag string, v any) {
	d.transcript = append(d.transcript, []byte(tag)...)
	d.transcript = append(d.transcript, 0x00)
	if enc, err := EncodeCanonical(v); err == nil {
		d.transcript = append(d.transcript, enc...)
	}
}

func partyIndex(p Party) (int, error) {
	switch p {
	case Worker:
		return 0, nil
	case Challenger:
		return 1, nil
	default:
		return 0, errors.New("gemmv1: invalid party")
	}
}

type disputeOpenTranscript struct {
	TaskID              []byte `gemm:"task_id"`
	AssignmentID        []byte `gemm:"assignment_id"`
	TileI               uint32 `gemm:"tile_i"`
	TileJ               uint32 `gemm:"tile_j"`
	WorkerTraceRoot     []byte `gemm:"worker_trace_root"`
	ChallengerTraceRoot []byte `gemm:"challenger_trace_root"`
}

type midTranscript struct {
	Step            uint32 `gemm:"step"`
	WorkerState     []byte `gemm:"worker_state"`
	ChallengerState []byte `gemm:"challenger_state"`
}

type arbitrationTranscript struct {
	Step     uint32 `gemm:"step"`
	Outcome  uint64 `gemm:"outcome"`
	Expected []byte `gemm:"expected"`
}

type timeoutTranscript struct {
	Outcome uint64 `gemm:"outcome"`
}

// Exported accessors for consensus persistence and transcript events. The
// chain state machine must read dispute internals without touching them
// directly.

// LowState returns a copy of the agreed state at the low step.
func (d *GEMMDispute) LowState() State { return d.lowState }

// HighState returns a copy of the party's state at the high step.
func (d *GEMMDispute) HighState(party Party) State {
	idx := 0
	if party == Challenger {
		idx = 1
	}
	return d.highStates[idx]
}

// TileI and TileJ return the disputed tile coordinates.
func (d *GEMMDispute) TileI() uint32 { return d.cfg.TileI }
func (d *GEMMDispute) TileJ() uint32 { return d.cfg.TileJ }

// ExpectedArbitrationState recomputes the arbiter's expected state for the
// final disputed transition: exactly one 512-MAC micro-step.
func (d *GEMMDispute) ExpectedArbitrationState(aTile, bTile [int8TileSize]int8) State {
	return MicroStep(&d.lowState, aTile, bTile)
}
