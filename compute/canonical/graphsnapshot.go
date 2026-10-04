package canonical

// GraphDispute snapshot for consensus KV storage ("GDS1"). The chain state
// machine must survive node restarts, so every field needed to continue a
// graph dispute is persisted as one fixed-size canonical binary blob,
// mirroring the frozen gemmv1 GMD1 format. The on-demand trace is never
// stored: only the locked trail roots, the bisection interval, the
// midpoint submissions and the round clock.

import (
	"encoding/binary"
	"errors"
)

var (
	errGraphSnapshotSize = errors.New("canonical: invalid graph dispute snapshot size")
	errGraphSnapshotBad  = errors.New("canonical: invalid graph dispute snapshot state")
)

const graphSnapshotSize = 4 + 1 + 32 + 8 + 2*32 + 4 + 4 + 32 + 2*32 + 2*(1+32) + 8 + 8 + 1 + 1

// SnapshotV1 encodes the dispute for consensus storage. The graph
// descriptor itself is NOT part of the snapshot: the caller re-supplies it
// at restore time and the stored graph id must match it.
func (d *GraphDispute) SnapshotV1() ([]byte, error) {
	if d == nil || d.cfg.Graph == nil {
		return nil, errors.New("canonical: uninitialized graph dispute")
	}
	b := make([]byte, graphSnapshotSize)
	o := 0
	copy(b[o:o+4], "GDS1")
	o += 4
	b[o] = 1
	o++
	copy(b[o:o+32], d.cfg.GraphID[:])
	o += 32
	binary.BigEndian.PutUint64(b[o:o+8], d.cfg.RoundPeriod)
	o += 8
	copy(b[o:o+32], d.roots[0][:])
	o += 32
	copy(b[o:o+32], d.roots[1][:])
	o += 32
	binary.BigEndian.PutUint32(b[o:o+4], d.low)
	o += 4
	binary.BigEndian.PutUint32(b[o:o+4], d.high)
	o += 4
	copy(b[o:o+32], d.lowRoot[:])
	o += 32
	copy(b[o:o+32], d.highRoots[0][:])
	o += 32
	copy(b[o:o+32], d.highRoots[1][:])
	o += 32
	for _, m := range d.medians {
		if m != nil {
			b[o] = 1
			copy(b[o+1:o+33], m[:])
		}
		o += 1 + 32
	}
	binary.BigEndian.PutUint64(b[o:o+8], d.deadline)
	o += 8
	binary.BigEndian.PutUint64(b[o:o+8], d.lastEpoch)
	o += 8
	if d.arbReady {
		b[o] = 1
	}
	o++
	b[o] = byte(d.outcome)
	return b, nil
}

// RestoreGraphDispute rebuilds a dispute from a canonical snapshot. The
// caller must re-supply the graph descriptor and the round period; the
// stored graph id, the interval and the clock are validated against it.
func RestoreGraphDispute(snapshot []byte, graph *GraphDescriptor, roundPeriod uint64) (*GraphDispute, error) {
	if graph == nil || roundPeriod == 0 {
		return nil, errors.New("canonical: restore requires the graph context")
	}
	if len(snapshot) != graphSnapshotSize || string(snapshot[:4]) != "GDS1" || snapshot[4] != 1 {
		return nil, errGraphSnapshotSize
	}
	if err := graph.Validate(); err != nil {
		return nil, err
	}
	graphID, err := graph.GraphID()
	if err != nil {
		return nil, err
	}
	o := 5
	var storedGraph Hash
	copy(storedGraph[:], snapshot[o:o+32])
	o += 32
	if storedGraph != graphID {
		return nil, errors.New("canonical: snapshot graph mismatch")
	}
	period := binary.BigEndian.Uint64(snapshot[o : o+8])
	o += 8
	if period != roundPeriod {
		return nil, errors.New("canonical: snapshot round period mismatch")
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
	var lowRoot Hash
	copy(lowRoot[:], snapshot[o:o+32])
	o += 32
	var highRoots [2]Hash
	copy(highRoots[0][:], snapshot[o:o+32])
	o += 32
	copy(highRoots[1][:], snapshot[o:o+32])
	o += 32
	var medians [2]*Hash
	for i := range medians {
		present := snapshot[o]
		var m Hash
		copy(m[:], snapshot[o+1:o+33])
		if present > 1 {
			return nil, errGraphSnapshotBad
		}
		if present == 1 {
			medians[i] = &m
		} else if m != (Hash{}) {
			return nil, errGraphSnapshotBad
		}
		o += 1 + 32
	}
	deadline := binary.BigEndian.Uint64(snapshot[o : o+8])
	o += 8
	lastEpoch := binary.BigEndian.Uint64(snapshot[o : o+8])
	o += 8
	arbReady := snapshot[o] == 1
	o++
	outcome := Outcome(snapshot[o])
	steps := uint32(len(graph.Nodes))
	if outcome > BothInvalid || deadline < lastEpoch || high > steps ||
		(low >= high && !(arbReady || outcome != Pending)) ||
		(outcome == Pending && !arbReady && high-low <= 1) {
		return nil, errGraphSnapshotBad
	}
	return &GraphDispute{
		cfg:       GraphDisputeConfig{Graph: graph, GraphID: graphID, RoundPeriod: period},
		roots:     roots,
		low:       low,
		high:      high,
		lowRoot:   lowRoot,
		highRoots: highRoots,
		medians:   medians,
		deadline:  deadline,
		lastEpoch: lastEpoch,
		arbReady:  arbReady,
		outcome:   outcome,
	}, nil
}

// DisputeStatus is the persisted status vocabulary of a graph dispute.
func (d *GraphDispute) DisputeStatus() string {
	switch {
	case d.arbReady:
		return GraphDisputeArbReady
	case d.outcome != Pending:
		return GraphDisputeResolved
	case d.high-d.low <= 1:
		return GraphDisputeArbReady
	default:
		return GraphDisputeBisection
	}
}

// Dispute sub-phase statuses (chain-visible strings).
const (
	GraphDisputeOpen      = "open"
	GraphDisputeBisection = "bisection"
	GraphDisputeArbReady  = "arb_ready"
	GraphDisputeResolved  = "resolved"
)
