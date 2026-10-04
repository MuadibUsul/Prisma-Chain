package canonical

// GraphDisputeV2 (roadmap A1-04..A1-08): the V2 fork of the graph bisection
// state machine.  Semantics are identical to GraphDispute; the version
// boundary lives in the trail leaf domain (DomainGraphTraceV2), the V2
// descriptor and the independent snapshot format.  The V1 type stays
// frozen — that is what the legacy golden tests pin.

import (
	"encoding/hex"
	"encoding/json"
	"errors"
)

func hexStr(b []byte) string { return hex.EncodeToString(b) }

func hexBytes(s string) ([]byte, error) { return hex.DecodeString(s) }

// GraphDisputeConfigV2 mirrors GraphDisputeConfig over the V2 descriptor.
type GraphDisputeConfigV2 struct {
	Graph       *GraphDescriptorV2
	GraphID     Hash
	RoundPeriod uint64
}

// GraphDisputeV2 is the persisted state of a graph-level dispute on a V2
// task.
type GraphDisputeV2 struct {
	cfg       GraphDisputeConfigV2
	roots     [2]Hash
	low, high uint32
	lowRoot   Hash
	highRoots [2]Hash
	medians   [2]*Hash
	deadline  uint64
	lastEpoch uint64
	arbReady  bool
	outcome   Outcome
}

// TrailClaimV2 locks one party's V2 trail (same shape as TrailClaim, V2
// leaf material).
type TrailClaimV2 = TrailClaim

func NewGraphDisputeV2(cfg GraphDisputeConfigV2, worker, challenger TrailClaimV2,
	openedEpoch uint64) (*GraphDisputeV2, error) {
	if cfg.Graph == nil || cfg.RoundPeriod == 0 {
		return nil, errors.New("canonical/v2: dispute config incomplete")
	}
	if err := cfg.Graph.ValidateV2(); err != nil {
		return nil, err
	}
	steps := uint32(len(cfg.Graph.Nodes))
	count := uint32(len(cfg.Graph.Nodes) + 1)
	for _, claim := range []TrailClaimV2{worker, challenger} {
		if len(claim.InitialProof) != depthFor(count) || len(claim.FinalProof) != depthFor(count) {
			return nil, errBadTrailProof
		}
		if !VerifyLeafInclusion(claim.Root, TrailLeafV2(cfg.GraphID, 0, claim.InitialRoot),
			MerkleProof{Index: 0, Count: count, Siblings: claim.InitialProof}) {
			return nil, errBadTrailProof
		}
		if !VerifyLeafInclusion(claim.Root, TrailLeafV2(cfg.GraphID, steps, claim.FinalRoot),
			MerkleProof{Index: steps, Count: count, Siblings: claim.FinalProof}) {
			return nil, errBadTrailProof
		}
	}
	if worker.InitialRoot != challenger.InitialRoot {
		return nil, errors.New("canonical/v2: trails must share the committed input state")
	}
	if worker.FinalRoot == challenger.FinalRoot {
		return nil, errTrailSameEnd
	}
	rounds := uint64(0)
	for width := count; width > 1; width = (width + 1) / 2 {
		rounds++
	}
	if rounds == 0 {
		rounds = 1
	}
	if openedEpoch >= ^uint64(0) || cfg.RoundPeriod > (^uint64(0)-openedEpoch-1)/rounds {
		return nil, errors.New("canonical/v2: epoch overflow")
	}
	d := &GraphDisputeV2{
		cfg:       cfg,
		roots:     [2]Hash{worker.Root, challenger.Root},
		high:      steps,
		lowRoot:   worker.InitialRoot,
		highRoots: [2]Hash{worker.FinalRoot, challenger.FinalRoot},
		deadline:  openedEpoch + cfg.RoundPeriod,
		lastEpoch: openedEpoch,
	}
	if d.high == 1 {
		d.arbReady = true
	}
	return d, nil
}

// SubmitMid accepts one party's midpoint state root with a V2 inclusion
// proof against its locked trail root.
func (d *GraphDisputeV2) SubmitMid(party Party, stateRoot Hash, proof []Hash,
	epoch uint64) (Outcome, error) {
	if d.outcome != Pending {
		return d.outcome, errDisputeDone
	}
	if d.arbReady {
		return Pending, errNotArbReady
	}
	if epoch < d.lastEpoch || epoch > d.deadline {
		return Pending, errRoundWindow
	}
	idx, err := partyIndex(party)
	if err != nil {
		return Pending, err
	}
	if d.medians[idx] != nil {
		return Pending, errors.New("canonical/v2: party already submitted this round")
	}
	mid := d.low + (d.high-d.low)/2
	count := uint32(len(d.cfg.Graph.Nodes) + 1)
	if !VerifyLeafInclusion(d.roots[idx], TrailLeafV2(d.cfg.GraphID, mid, stateRoot),
		MerkleProof{Index: mid, Count: count, Siblings: proof}) {
		return Pending, errBadTrailProof
	}
	d.lastEpoch = epoch
	stored := stateRoot
	d.medians[idx] = &stored
	if d.medians[0] == nil || d.medians[1] == nil {
		return Pending, nil
	}
	workerMid, challengerMid := *d.medians[0], *d.medians[1]
	if workerMid == challengerMid {
		d.low = mid
		d.lowRoot = workerMid
	} else {
		d.high = mid
		d.highRoots = [2]Hash{workerMid, challengerMid}
	}
	d.medians = [2]*Hash{}
	if d.high-d.low == 1 {
		d.arbReady = true
	}
	d.deadline = epoch + d.cfg.RoundPeriod
	return Pending, nil
}

func (d *GraphDisputeV2) FirstDivergentNode() (uint32, error) {
	if !d.arbReady {
		return 0, errNotArbReady
	}
	return d.low, nil
}

func (d *GraphDisputeV2) Timeout(epoch uint64) (Outcome, error) {
	if d.outcome != Pending {
		return d.outcome, errDisputeDone
	}
	if d.arbReady {
		return Pending, errNotArbReady
	}
	if epoch <= d.deadline || epoch < d.lastEpoch {
		return Pending, errRoundExpired
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
	return d.outcome, nil
}

func (d *GraphDisputeV2) SetOutcome(o Outcome) { d.outcome = o; d.arbReady = false }
func (d *GraphDisputeV2) Outcome() Outcome     { return d.outcome }
func (d *GraphDisputeV2) ArbReady() bool       { return d.arbReady }
func (d *GraphDisputeV2) LowStateRoot() Hash   { return d.lowRoot }
func (d *GraphDisputeV2) DisputeDeadline() uint64 { return d.deadline }
func (d *GraphDisputeV2) LastAcceptedEpoch() uint64 { return d.lastEpoch }
func (d *GraphDisputeV2) Interval() (low, high uint32) { return d.low, d.high }
func (d *GraphDisputeV2) HighRoots() [2]Hash   { return d.highRoots }

// DisputeStatusV2 mirrors DisputeStatus.
func (d *GraphDisputeV2) DisputeStatusV2() string {
	switch {
	case d.outcome != Pending:
		return "resolved"
	case d.arbReady:
		return "arb_ready"
	default:
		return "bisection"
	}
}

// --- snapshot V2 (independent format/domain; V1 GDS1 untouched) --------------

type graphDisputeSnapshotV2 struct {
	ProtocolVersion string   `json:"protocol_version"`
	GraphID         string   `json:"graph_id"`
	Nodes           int      `json:"nodes"`
	Roots           []string `json:"roots"`
	Low             uint32   `json:"low"`
	High            uint32   `json:"high"`
	LowRoot         string   `json:"low_root"`
	HighRoots       []string `json:"high_roots"`
	Deadline        uint64   `json:"deadline"`
	LastEpoch       uint64   `json:"last_epoch"`
	ArbReady        bool     `json:"arb_ready"`
	Outcome         uint8    `json:"outcome"`
	// an in-flight round may already hold one party's submission; the
	// snapshot must carry it or the resumed instance would replay the round
	Median0 string `json:"median_0,omitempty"`
	Median1 string `json:"median_1,omitempty"`
}

// SnapshotV2 persists the V2 dispute mid-bisection.  Restore re-validates
// the graph identity and node count against the descriptor.
func (d *GraphDisputeV2) SnapshotV2() ([]byte, error) {
	doc := graphDisputeSnapshotV2{
		ProtocolVersion: ProtocolVersionGraphV2,
		GraphID:         hexStr(d.cfg.GraphID[:]),
		Nodes:           len(d.cfg.Graph.Nodes),
		Low:             d.low, High: d.high,
		LowRoot:   hexStr(d.lowRoot[:]),
		Deadline:  d.deadline, LastEpoch: d.lastEpoch,
		ArbReady: d.arbReady, Outcome: uint8(d.outcome),
	}
	for _, r := range d.roots {
		doc.Roots = append(doc.Roots, hexStr(r[:]))
	}
	for _, r := range d.highRoots {
		doc.HighRoots = append(doc.HighRoots, hexStr(r[:]))
	}
	if d.medians[0] != nil {
		doc.Median0 = hexStr(d.medians[0][:])
	}
	if d.medians[1] != nil {
		doc.Median1 = hexStr(d.medians[1][:])
	}
	return json.Marshal(doc)
}

func RestoreGraphDisputeV2(snapshot []byte, graph *GraphDescriptorV2,
	roundPeriod uint64) (*GraphDisputeV2, error) {
	var doc graphDisputeSnapshotV2
	if err := json.Unmarshal(snapshot, &doc); err != nil {
		return nil, err
	}
	if doc.ProtocolVersion != ProtocolVersionGraphV2 || graph == nil {
		return nil, errors.New("canonical/v2: snapshot protocol mismatch")
	}
	if err := graph.ValidateV2(); err != nil {
		return nil, err
	}
	if doc.Nodes != len(graph.Nodes) {
		return nil, errors.New("canonical/v2: snapshot node count mismatch")
	}
	graphID, err := graph.GraphIDV2()
	if err != nil {
		return nil, err
	}
	if doc.GraphID != hexStr(graphID[:]) {
		return nil, errors.New("canonical/v2: snapshot graph id mismatch")
	}
	d := &GraphDisputeV2{
		cfg:       GraphDisputeConfigV2{Graph: graph, GraphID: graphID, RoundPeriod: roundPeriod},
		low:       doc.Low,
		high:      doc.High,
		deadline:  doc.Deadline,
		lastEpoch: doc.LastEpoch,
		arbReady:  doc.ArbReady,
		outcome:   Outcome(doc.Outcome),
	}
	if len(doc.Roots) != 2 || len(doc.HighRoots) != 2 {
		return nil, errors.New("canonical/v2: snapshot roots malformed")
	}
	for i := 0; i < 2; i++ {
		b, err := hexBytes(doc.Roots[i])
		if err != nil || len(b) != 32 {
			return nil, errors.New("canonical/v2: snapshot root malformed")
		}
		copy(d.roots[i][:], b)
		hb, err := hexBytes(doc.HighRoots[i])
		if err != nil || len(hb) != 32 {
			return nil, errors.New("canonical/v2: snapshot high root malformed")
		}
		copy(d.highRoots[i][:], hb)
	}
	lb, err := hexBytes(doc.LowRoot)
	if err != nil || len(lb) != 32 {
		return nil, errors.New("canonical/v2: snapshot low root malformed")
	}
	copy(d.lowRoot[:], lb)
	if d.high > uint32(len(graph.Nodes)) || d.low >= d.high {
		return nil, errors.New("canonical/v2: snapshot interval out of range")
	}
	for idx, enc := range [2]string{doc.Median0, doc.Median1} {
		if enc == "" {
			continue
		}
		b, err := hexBytes(enc)
		if err != nil || len(b) != 32 {
			return nil, errors.New("canonical/v2: snapshot median malformed")
		}
		var h Hash
		copy(h[:], b)
		d.medians[idx] = &h
	}
	return d, nil
}
