package gemmv1

// GEMMDispute snapshot for consensus KV storage. The chain state machine
// must survive node restarts, so every field needed to continue a dispute
// is persisted in one fixed-size canonical binary blob ("GMD1"), mirroring
// the vm package's approach. The full trace is never stored — only the
// locked roots, the current bisection interval and midpoint submissions.

import (
	"encoding/binary"
	"errors"
)

var (
	errSnapshotSize = errors.New("gemmv1: invalid dispute snapshot size")
	errSnapshotBad  = errors.New("gemmv1: invalid dispute snapshot state")
)

const gemmSnapshotSize = 4 + 1 + 32 + 32 + 4 + 4 + 32 + 32 + 4 + 4 + 3*256 + 2*(1+256) + 24 + 1 + 1

// SnapshotV1 encodes the dispute for consensus storage.
func (d *GEMMDispute) SnapshotV1() ([]byte, error) {
	if d == nil || d.cfg.Task == nil || len(d.assignmentID) != 32 {
		return nil, errors.New("gemmv1: uninitialized dispute")
	}
	b := make([]byte, gemmSnapshotSize)
	o := 0
	copy(b[o:o+4], "GMD1")
	o += 4
	b[o] = 1 // snapshot version
	o++
	copy(b[o:o+32], d.cfg.TaskID)
	o += 32
	copy(b[o:o+32], d.assignmentID)
	o += 32
	binary.BigEndian.PutUint32(b[o:o+4], d.cfg.TileI)
	o += 4
	binary.BigEndian.PutUint32(b[o:o+4], d.cfg.TileJ)
	o += 4
	copy(b[o:o+32], d.roots[0][:])
	o += 32
	copy(b[o:o+32], d.roots[1][:])
	o += 32
	binary.BigEndian.PutUint32(b[o:o+4], d.low)
	o += 4
	binary.BigEndian.PutUint32(b[o:o+4], d.high)
	o += 4
	copy(b[o:o+256], d.lowState.CanonicalBytes())
	o += 256
	copy(b[o:o+256], d.highStates[0].CanonicalBytes())
	o += 256
	copy(b[o:o+256], d.highStates[1].CanonicalBytes())
	o += 256
	for _, m := range d.medians {
		if m != nil {
			b[o] = 1
			copy(b[o+1:o+257], m.CanonicalBytes())
		}
		o += 1 + 256
	}
	for _, v := range []uint64{d.cfg.RoundPeriod, d.deadline, d.lastEpoch} {
		binary.BigEndian.PutUint64(b[o:o+8], v)
		o += 8
	}
	if d.arbReady {
		b[o] = 1
	}
	o++
	b[o] = byte(d.outcome)
	return b, nil
}

// RestoreGEMMDispute rebuilds a dispute from a canonical snapshot. The
// caller must re-supply the task and assignment context; identity fields
// stored in the snapshot are validated against it.
func RestoreGEMMDispute(snapshot []byte, cfg DisputeConfig) (*GEMMDispute, error) {
	if cfg.Task == nil {
		return nil, errors.New("gemmv1: restore requires the task context")
	}
	if len(snapshot) != gemmSnapshotSize || string(snapshot[:4]) != "GMD1" || snapshot[4] != 1 {
		return nil, errSnapshotSize
	}
	o := 5
	var storedTask, storedAssignment [32]byte
	copy(storedTask[:], snapshot[o:o+32])
	o += 32
	copy(storedAssignment[:], snapshot[o:o+32])
	o += 32
	taskID, err := cfg.Task.TaskID()
	if err != nil {
		return nil, err
	}
	assignmentID, err := cfg.Assignment.AssignmentID()
	if err != nil {
		return nil, err
	}
	if string(storedTask[:]) != string(taskID) || string(storedAssignment[:]) != string(assignmentID) {
		return nil, errors.New("gemmv1: snapshot task or assignment mismatch")
	}
	tileI := binary.BigEndian.Uint32(snapshot[o : o+4])
	o += 4
	tileJ := binary.BigEndian.Uint32(snapshot[o : o+4])
	o += 4
	if tileI != cfg.TileI || tileJ != cfg.TileJ {
		return nil, errors.New("gemmv1: snapshot tile mismatch")
	}
	var roots [2]Hash
	copy(roots[0][:], snapshot[o:o+32])
	o += 32
	copy(roots[1][:], snapshot[o:o+32])
	o += 32
	low := binary.BigEndian.Uint32(snapshot[o : o+4])
	o += 4
	high := binary.BigEndian.Uint32(snapshot[o : o+4])
	o += 4
	lowState, err := StateFromCanonical(snapshot[o : o+256])
	if err != nil {
		return nil, err
	}
	o += 256
	workerHigh, err := StateFromCanonical(snapshot[o : o+256])
	if err != nil {
		return nil, err
	}
	o += 256
	challengerHigh, err := StateFromCanonical(snapshot[o : o+256])
	if err != nil {
		return nil, err
	}
	o += 256
	var medians [2]*State
	for i := range medians {
		present := snapshot[o]
		state, err := StateFromCanonical(snapshot[o+1 : o+257])
		if err != nil {
			return nil, err
		}
		if present > 1 {
			return nil, errSnapshotBad
		}
		if present == 1 {
			medians[i] = state
		} else if *state != (State{}) {
			return nil, errSnapshotBad
		}
		o += 1 + 256
	}
	period := binary.BigEndian.Uint64(snapshot[o : o+8])
	o += 8
	deadline := binary.BigEndian.Uint64(snapshot[o : o+8])
	o += 8
	lastEpoch := binary.BigEndian.Uint64(snapshot[o : o+8])
	o += 8
	arbReady := snapshot[o] == 1
	o++
	outcome := Outcome(snapshot[o])
	if outcome > BothInvalid || period == 0 || deadline < lastEpoch ||
		low >= high && !(arbReady || outcome != Pending) ||
		outcome == Pending && !arbReady && high-low <= 1 {
		return nil, errSnapshotBad
	}
	r := uint32(RSteps(cfg.Task.K))
	if high > r {
		return nil, errSnapshotBad
	}
	return &GEMMDispute{
		cfg:          cfg,
		assignmentID: assignmentID,
		r:            r,
		roots:        roots,
		low:          low,
		high:         high,
		lowState:     *lowState,
		highStates:   [2]State{*workerHigh, *challengerHigh},
		medians:      medians,
		deadline:     deadline,
		lastEpoch:    lastEpoch,
		outcome:      outcome,
		arbReady:     arbReady,
	}, nil
}
