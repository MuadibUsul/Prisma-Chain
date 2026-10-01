package vm

import (
	"encoding/binary"
	"errors"
)

const snapshotSize = 827

// Snapshot is a fixed-size, canonical v1 encoding for consensus KV storage.
// It binds the dispute to the exact program and public input supplied at
// restoration. The chain must authenticate parties before calling SubmitMid.
func (d *Dispute) Snapshot() ([]byte, error) {
	if d == nil || d.program.Len() == 0 {
		return nil, errors.New("uninitialized dispute")
	}
	b := make([]byte, snapshotSize)
	o := 0
	copy(b[o:o+4], "PVD1")
	o += 4
	programHash, inputHash := d.program.Digest(), InputDigest(d.input)
	copy(b[o:o+32], programHash[:])
	o += 32
	copy(b[o:o+32], inputHash[:])
	o += 32
	for _, root := range d.roots {
		copy(b[o:o+32], root[:])
		o += 32
	}
	binary.BigEndian.PutUint32(b[o:o+4], d.low)
	o += 4
	binary.BigEndian.PutUint32(b[o:o+4], d.high)
	o += 4
	putState(b[o:o+132], d.lowState)
	o += 132
	for _, s := range d.highStates {
		putState(b[o:o+132], s)
		o += 132
	}
	for _, s := range d.medians {
		if s != nil {
			b[o] = 1
			putState(b[o+1:o+133], *s)
		}
		o += 133
	}
	for _, n := range []uint64{d.period, d.deadline, d.lastHeight} {
		binary.BigEndian.PutUint64(b[o:o+8], n)
		o += 8
	}
	b[o] = byte(d.outcome)
	return b, nil
}

// RestoreDispute rejects incompatible, truncated, or noncanonical snapshots.
// The program and input must match the values previously committed on chain.
func RestoreDispute(b []byte, p Program, input []int64) (*Dispute, error) {
	if err := validateInput(p, input); err != nil {
		return nil, err
	}
	if len(b) != snapshotSize || string(b[:4]) != "PVD1" {
		return nil, errors.New("invalid dispute snapshot format")
	}
	o := 4
	programHash, inputHash := p.Digest(), InputDigest(input)
	if string(b[o:o+32]) != string(programHash[:]) || string(b[o+32:o+64]) != string(inputHash[:]) {
		return nil, errors.New("snapshot program or input mismatch")
	}
	o += 64
	d := &Dispute{program: p, input: append([]int64(nil), input...)}
	for i := range d.roots {
		copy(d.roots[i][:], b[o:o+32])
		o += 32
	}
	d.low = binary.BigEndian.Uint32(b[o : o+4])
	o += 4
	d.high = binary.BigEndian.Uint32(b[o : o+4])
	o += 4
	d.lowState = readState(b[o : o+132])
	o += 132
	for i := range d.highStates {
		d.highStates[i] = readState(b[o : o+132])
		o += 132
	}
	for i := range d.medians {
		if b[o] > 1 {
			return nil, errors.New("invalid midpoint presence flag")
		}
		state := readState(b[o+1 : o+133])
		if b[o] == 1 {
			d.medians[i] = &state
		} else if state != (State{}) {
			return nil, errors.New("noncanonical absent midpoint")
		}
		o += 133
	}
	d.period = binary.BigEndian.Uint64(b[o : o+8])
	o += 8
	d.deadline = binary.BigEndian.Uint64(b[o : o+8])
	o += 8
	d.lastHeight = binary.BigEndian.Uint64(b[o : o+8])
	o += 8
	d.outcome = Outcome(b[o])
	if d.period == 0 || d.low >= d.high || d.high > uint32(p.Len()) || d.outcome > BothInvalid ||
		d.lowState.PC != d.low || d.highStates[0].PC != d.high || d.highStates[1].PC != d.high ||
		(d.outcome == Pending && (d.high-d.low <= 1 || d.lastHeight > d.deadline ||
			d.medians[0] != nil && d.medians[1] != nil)) {
		return nil, errors.New("invalid dispute snapshot state")
	}
	for _, state := range d.medians {
		if state != nil && state.PC != d.low+(d.high-d.low)/2 {
			return nil, errors.New("invalid midpoint snapshot state")
		}
	}
	return d, nil
}

func putState(dst []byte, state State) {
	binary.BigEndian.PutUint32(dst[:4], state.PC)
	for i, value := range state.Reg {
		binary.BigEndian.PutUint64(dst[4+i*8:], uint64(value))
	}
}

func readState(src []byte) State {
	var state State
	state.PC = binary.BigEndian.Uint32(src[:4])
	for i := range state.Reg {
		state.Reg[i] = int64(binary.BigEndian.Uint64(src[4+i*8:]))
	}
	return state
}
