package canonical

// WIDE_GEMM_DISPUTE_V1 state machine (roadmap A2).  A K-step partial-state
// chain per party (S0 = zero, S_{r+1} = S_r + A_tile * W_tile over one
// 8-wide K step, 512 logical MAC each) bisects to the first divergent
// step exactly like the graph dispute; every field is exported so the
// chain stores the whole state machine as plain JSON (restart-safe).

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
)

// WideTraceClaimV1 locks one party's wide trace.
type WideTraceClaimV1 struct {
	Root         Hash
	InitialState WideTileState
	InitialProof []Hash
	FinalState   WideTileState
	FinalProof   []Hash
}

// WideGEMMDisputeV1 is the wide dispute state (exported for persistence).
type WideGEMMDisputeV1 struct {
	Steps       uint32 `json:"steps"` // trace length = ceil(K/8)+1
	RoundPeriod uint64 `json:"round_period"`
	TileI       uint32 `json:"tile_i"`
	TileJ       uint32 `json:"tile_j"`

	Roots     [2]string `json:"roots"` // locked trace merkle roots (hex)
	Low       uint32    `json:"low"`
	High      uint32    `json:"high"`
	LowState  string    `json:"low_state"`       // BE64 hex at the agreed step
	Medians   [2]string `json:"medians"`         // in-flight round states (hex, "" = none)
	HighS0    string    `json:"high_s0"`         // party 0 state at High
	HighS1    string    `json:"high_s1"`         // party 1 state at High
	Deadline  uint64    `json:"deadline"`
	LastEpoch uint64    `json:"last_epoch"`
	ArbReady  bool      `json:"arb_ready"`
	Outcome   Outcome   `json:"outcome"`
}

func stateHex(s WideTileState) string {
	buf := make([]byte, 0, WideTileN*WideTileN*8)
	for _, v := range s {
		for shift := 56; shift >= 0; shift -= 8 {
			buf = append(buf, byte(v>>uint(shift)))
		}
	}
	return hex.EncodeToString(buf)
}

func stateFromHex(s string) (WideTileState, error) {
	var out WideTileState
	b, err := hex.DecodeString(s)
	if err != nil || len(b) != WideTileN*WideTileN*8 {
		return out, errors.New("canonical/v2: malformed wide state encoding")
	}
	for i := 0; i < WideTileN*WideTileN; i++ {
		out[i] = decodeBEInt(b[i*8 : (i+1)*8])
	}
	return out, nil
}

func hashHex(h Hash) string { return hex.EncodeToString(h[:]) }

func hashFromHex(s string) (Hash, error) {
	var h Hash
	b, err := hex.DecodeString(s)
	if err != nil || len(b) != 32 {
		return h, errors.New("canonical/v2: malformed hash encoding")
	}
	copy(h[:], b)
	return h, nil
}

// NewWideDisputeV1 opens the dispute from both locked trace claims.
func NewWideDisputeV1(steps uint32, roundPeriod uint64, tileI, tileJ int,
	worker, challenger WideTraceClaimV1, openedEpoch uint64) (*WideGEMMDisputeV1, error) {
	if steps < 2 || roundPeriod == 0 {
		return nil, errors.New("canonical/v2: wide dispute config incomplete")
	}
	count := steps
	for _, claim := range []WideTraceClaimV1{worker, challenger} {
		if len(claim.InitialProof) != depthFor(count) || len(claim.FinalProof) != depthFor(count) {
			return nil, errBadTrailProof
		}
		if claim.InitialState != (WideTileState{}) {
			return nil, errors.New("canonical/v2: wide trace must start at the zero state")
		}
		if !VerifyLeafInclusion(claim.Root, WideStateLeafV1(tileI, tileJ, 0, claim.InitialState),
			MerkleProof{Index: 0, Count: count, Siblings: claim.InitialProof}) {
			return nil, errBadTrailProof
		}
		if !VerifyLeafInclusion(claim.Root, WideStateLeafV1(tileI, tileJ, int(steps-1), claim.FinalState),
			MerkleProof{Index: steps - 1, Count: count, Siblings: claim.FinalProof}) {
			return nil, errBadTrailProof
		}
	}
	if worker.FinalState == challenger.FinalState {
		return nil, errTrailSameEnd
	}
	rounds := uint64(0)
	for width := count; width > 1; width = (width + 1) / 2 {
		rounds++
	}
	if rounds == 0 {
		rounds = 1
	}
	if openedEpoch >= ^uint64(0) || roundPeriod > (^uint64(0)-openedEpoch-1)/rounds {
		return nil, errors.New("canonical/v2: epoch overflow")
	}
	d := &WideGEMMDisputeV1{
		Steps: steps, RoundPeriod: roundPeriod,
		TileI: uint32(tileI), TileJ: uint32(tileJ),
		Roots:     [2]string{hashHex(worker.Root), hashHex(challenger.Root)},
		High:      steps - 1,
		LowState:  stateHex(WideTileState{}),
		HighS0:    stateHex(worker.FinalState),
		HighS1:    stateHex(challenger.FinalState),
		Deadline:  openedEpoch + roundPeriod,
		LastEpoch: openedEpoch,
	}
	if d.High == 1 {
		d.ArbReady = true
	}
	return d, nil
}

// SubmitMid accepts one party's midpoint state with a trace inclusion proof.
func (d *WideGEMMDisputeV1) SubmitMid(party Party, state WideTileState, proof []Hash,
	epoch uint64) (Outcome, error) {
	if d.Outcome != Pending {
		return d.Outcome, errDisputeDone
	}
	if d.ArbReady {
		return Pending, errNotArbReady
	}
	if epoch < d.LastEpoch || epoch > d.Deadline {
		return Pending, errRoundWindow
	}
	idx, err := partyIndex(party)
	if err != nil {
		return Pending, err
	}
	if d.Medians[idx] != "" {
		return Pending, errors.New("canonical/v2: party already submitted this round")
	}
	mid := d.Low + (d.High-d.Low)/2
	root, err := hashFromHex(d.Roots[idx])
	if err != nil {
		return Pending, err
	}
	if !VerifyLeafInclusion(root, WideStateLeafV1(int(d.TileI), int(d.TileJ), int(mid), state),
		MerkleProof{Index: mid, Count: d.Steps, Siblings: proof}) {
		return Pending, errBadTrailProof
	}
	d.LastEpoch = epoch
	d.Medians[idx] = stateHex(state)
	if d.Medians[0] == "" || d.Medians[1] == "" {
		return Pending, nil
	}
	workerMid, err := stateFromHex(d.Medians[0])
	if err != nil {
		return Pending, err
	}
	challengerMid, err := stateFromHex(d.Medians[1])
	if err != nil {
		return Pending, err
	}
	if workerMid == challengerMid {
		d.Low = mid
		d.LowState = stateHex(workerMid)
	} else {
		d.High = mid
		d.HighS0 = stateHex(workerMid)
		d.HighS1 = stateHex(challengerMid)
	}
	d.Medians = [2]string{}
	if d.High-d.Low == 1 {
		d.ArbReady = true
	}
	d.Deadline = epoch + d.RoundPeriod
	return Pending, nil
}

// FirstDivergentStep is valid once the bisection completed.
func (d *WideGEMMDisputeV1) FirstDivergentStep() (uint32, error) {
	if !d.ArbReady {
		return 0, errNotArbReady
	}
	return d.Low, nil
}

func (d *WideGEMMDisputeV1) Timeout(epoch uint64) (Outcome, error) {
	if d.Outcome != Pending {
		return d.Outcome, errDisputeDone
	}
	if d.ArbReady {
		return Pending, errNotArbReady
	}
	if epoch <= d.Deadline || epoch < d.LastEpoch {
		return Pending, errRoundExpired
	}
	switch {
	case d.Medians[0] != "" && d.Medians[1] == "":
		d.Outcome = WorkerWins
	case d.Medians[0] == "" && d.Medians[1] != "":
		d.Outcome = ChallengerWins
	default:
		d.Outcome = BothInvalid
	}
	d.LastEpoch = epoch
	return d.Outcome, nil
}

func (d *WideGEMMDisputeV1) SetOutcome(o Outcome) { d.Outcome = o; d.ArbReady = false }

// MarshalJSON gives a stable persisted form (the chain stores this blob).
func (d *WideGEMMDisputeV1) MarshalState() ([]byte, error) {
	return json.Marshal(d)
}

func UnmarshalWideDisputeV1(raw []byte) (*WideGEMMDisputeV1, error) {
	var d WideGEMMDisputeV1
	if err := json.Unmarshal(raw, &d); err != nil {
		return nil, err
	}
	if d.Steps < 2 || d.High >= d.Steps || d.Low >= d.High {
		return nil, fmt.Errorf("canonical/v2: malformed wide dispute state")
	}
	return &d, nil
}

// LowStateValue decodes the agreed low state.
func (d *WideGEMMDisputeV1) LowStateValue() (WideTileState, error) { return stateFromHex(d.LowState) }

// HighStates decodes both parties' states at High.
func (d *WideGEMMDisputeV1) HighStates() (worker, challenger WideTileState, err error) {
	worker, err = stateFromHex(d.HighS0)
	if err != nil {
		return worker, challenger, err
	}
	challenger, err = stateFromHex(d.HighS1)
	return worker, challenger, err
}

// WideTraceProofV1 proves one step against a trace root.
func WideTraceProofV1(ti, tj int, states []WideTileState, index uint32) ([]Hash, error) {
	if len(states) == 0 {
		return nil, errors.New("canonical/v2: empty wide trace")
	}
	leaves := make([]Hash, len(states))
	for i, s := range states {
		leaves[i] = WideStateLeafV1(ti, tj, i, s)
	}
	levels, err := buildLevels(leaves)
	if err != nil {
		return nil, err
	}
	if int(index) >= len(states) {
		return nil, errors.New("canonical/v2: wide trace index out of range")
	}
	return proveLeaf(levels, int(index))
}

// WideStateFromBytes decodes 64 x BE64 two's complement into a tile state.
func WideStateFromBytes(b []byte) (WideTileState, error) {
	var out WideTileState
	if len(b) != WideTileN*WideTileN*8 {
		return out, errors.New("canonical/v2: wide state must be 512 bytes of BE64")
	}
	for i := 0; i < WideTileN*WideTileN; i++ {
		out[i] = decodeBEInt(b[i*8 : (i+1)*8])
	}
	return out, nil
}

// Bytes encodes the tile state as 64 x BE64 two's complement.
func (s WideTileState) Bytes() []byte {
	out := make([]byte, 0, WideTileN*WideTileN*8)
	for _, v := range s {
		out = append(out, encodeBEInt(v, 8)...)
	}
	return out
}
