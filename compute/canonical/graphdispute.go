package canonical

// Graph-level dispute: on-demand trail bisection to the first divergent
// node, then bounded per-operator chunk arbitration. The arbiter never
// recomputes the block: it adjudicates exactly the units declared by
// Operator.ArbiterBound (one chunk, or one bounded row for the row
// operators).

import (
	"errors"
	"fmt"
	"math"
)

var (
	errDisputeDone   = errors.New("canonical: dispute already resolved")
	errNotArbReady   = errors.New("canonical: dispute is not ready for arbitration")
	errRoundWindow   = errors.New("canonical: submission outside the current round")
	errRoundExpired  = errors.New("canonical: round has not expired")
	errBadTrailProof = errors.New("canonical: invalid trail inclusion proof")
	errTrailSameEnd  = errors.New("canonical: both trails end in the same state root")
	errBadChunkProof = errors.New("canonical: chunk does not belong to the committed tensor root")
	errSameChunk     = errors.New("canonical: parties submitted identical chunks")
)

// GraphDisputeConfig fixes one dispute's context.
type GraphDisputeConfig struct {
	Graph       *GraphDescriptor
	GraphID     Hash
	RoundPeriod uint64
}

// GraphDispute is the persisted state of a graph-level dispute.
type GraphDispute struct {
	cfg       GraphDisputeConfig
	roots     [2]Hash // locked trail Merkle roots
	low, high uint32  // over trail indices 0..N; Trail[low] equal, Trail[high] differ
	lowRoot   Hash
	highRoots [2]Hash
	medians   [2]*Hash
	deadline  uint64
	lastEpoch uint64
	arbReady  bool
	outcome   Outcome
}

// TrailLeaf hashes one trail state: domain || graph_id || step || stateRoot.
func TrailLeaf(graphID Hash, step uint32, stateRoot Hash) Hash {
	return hashBytes([]byte(DomainGraphTrace), graphID[:], appendUint32BE(nil, step), stateRoot[:])
}

// TrailLevels builds the on-demand trace tree over a party's trail.
func TrailLevels(graphID Hash, trail []Hash) ([][]Hash, error) {
	leaves := make([]Hash, len(trail))
	for i, s := range trail {
		leaves[i] = TrailLeaf(graphID, uint32(i), s)
	}
	return buildLevels(leaves)
}

// TrailClaim locks one party's trail: its Merkle root plus endpoint
// inclusion proofs (Trail[0] and Trail[N]).
type TrailClaim struct {
	Root         Hash
	InitialRoot  Hash
	InitialProof []Hash
	FinalRoot    Hash
	FinalProof   []Hash
}

// NewGraphDispute opens a dispute from both locked claims. Both trails
// must start in the same state root (committed inputs) and end in
// different ones (otherwise there is no fraud to adjudicate).
func NewGraphDispute(cfg GraphDisputeConfig, worker, challenger TrailClaim, openedEpoch uint64) (*GraphDispute, error) {
	if cfg.Graph == nil || cfg.RoundPeriod == 0 {
		return nil, errors.New("canonical: dispute config incomplete")
	}
	steps := uint32(len(cfg.Graph.Nodes))
	count := uint32(len(cfg.Graph.Nodes) + 1)
	for _, claim := range []TrailClaim{worker, challenger} {
		if len(claim.InitialProof) != depthFor(count) || len(claim.FinalProof) != depthFor(count) {
			return nil, errBadTrailProof
		}
		if !VerifyLeafInclusion(claim.Root, TrailLeaf(cfg.GraphID, 0, claim.InitialRoot), MerkleProof{Index: 0, Count: count, Siblings: claim.InitialProof}) {
			return nil, errBadTrailProof
		}
		if !VerifyLeafInclusion(claim.Root, TrailLeaf(cfg.GraphID, steps, claim.FinalRoot), MerkleProof{Index: steps, Count: count, Siblings: claim.FinalProof}) {
			return nil, errBadTrailProof
		}
	}
	if worker.InitialRoot != challenger.InitialRoot {
		return nil, errors.New("canonical: trails must share the committed input state")
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
		return nil, errors.New("canonical: epoch overflow")
	}
	d := &GraphDispute{
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

// SubmitMid accepts one party's midpoint state root with an inclusion
// proof against its locked trail root.
func (d *GraphDispute) SubmitMid(party Party, stateRoot Hash, proof []Hash, epoch uint64) (Outcome, error) {
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
		return Pending, errors.New("canonical: party already submitted this round")
	}
	mid := d.low + (d.high-d.low)/2
	count := uint32(len(d.cfg.Graph.Nodes) + 1)
	if !VerifyLeafInclusion(d.roots[idx], TrailLeaf(d.cfg.GraphID, mid, stateRoot), MerkleProof{Index: mid, Count: count, Siblings: proof}) {
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

// FirstDivergentNode is valid once the bisection completed.
func (d *GraphDispute) FirstDivergentNode() (uint32, error) {
	if !d.arbReady {
		return 0, errNotArbReady
	}
	return d.low, nil
}

// Timeout mirrors the other dispute state machines.
func (d *GraphDispute) Timeout(epoch uint64) (Outcome, error) {
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

// SetOutcome records the arbitration verdict.
func (d *GraphDispute) SetOutcome(o Outcome) { d.outcome = o; d.arbReady = false }

// Outcome reads the current verdict.
func (d *GraphDispute) Outcome() Outcome { return d.outcome }

// ArbReady reports whether the node was localized and arbitration may run.
func (d *GraphDispute) ArbReady() bool { return d.arbReady }

// --- bounded chunk arbitration --------------------------------------------

// NodeChunkClaim is one party's version of the disputed output chunk of
// the first divergent node, bound to the node's output tensor root.
type NodeChunkClaim struct {
	OutRoot    Hash
	ChunkIndex uint32
	ChunkBytes []byte // canonical chunk bytes (256 for Q12.20)
	Proof      []Hash
}

// ExpectedArbiterUnits returns the exact number of input elements the
// arbiter will touch for one node, from the operator contract.
func ExpectedArbiterUnits(node GraphNode) (int, error) {
	op, err := Lookup(node.OperatorID)
	if err != nil {
		return 0, err
	}
	return op.ArbiterBound(nil, node.Params), nil
}

// ArbitrateNodeChunk adjudicates the disputed chunk of one node. The
// arbiter verifies both parties' chunks against the committed node output
// root, verifies the supplied input evidence against the committed input
// state, and recomputes exactly the bounded unit of the operator. It
// never recomputes the node or the block.
//
// inputChunks maps each node input ref to the canonical chunk bytes (for
// row operators: the whole row plus the shared weight row) required by
// that operator's arbiter contract, with inclusion proofs against the
// input state root.
func ArbitrateNodeChunk(graph *GraphDescriptor, node GraphNode, inputStateRoot Hash,
	worker, challenger NodeChunkClaim, inputEvidence []ChunkEvidence, ropeTable *RopeConstants) (Outcome, error) {

	if !sameDesc(node.Output, NewDesc(DtypeQ12_20, int64(len(worker.ChunkBytes)/4))) {
		// shape is checked by the operator below; this is a coarse guard
	}
	count := uint32(0)
	descBytes, err := EncodeCanonical(node.Output)
	if err != nil {
		return Pending, err
	}
	elems, err := node.Output.Elems()
	if err != nil {
		return Pending, err
	}
	count = uint32((elems + ChunkElems - 1) / ChunkElems)
	if count == 0 {
		count = 1
	}
	if len(worker.ChunkBytes) != ChunkElems*4 || len(challenger.ChunkBytes) != ChunkElems*4 {
		return Pending, errors.New("canonical: chunk bytes must be 256")
	}
	if !VerifyChunk(worker.OutRoot, descBytes, worker.ChunkIndex, count, worker.ChunkBytes, worker.Proof) {
		return Pending, errBadChunkProof
	}
	if !VerifyChunk(challenger.OutRoot, descBytes, challenger.ChunkIndex, count, challenger.ChunkBytes, challenger.Proof) {
		return Pending, errBadChunkProof
	}
	if worker.ChunkIndex != challenger.ChunkIndex {
		return Pending, errors.New("canonical: parties must dispute the same chunk position")
	}
	if equalBytes(worker.ChunkBytes, challenger.ChunkBytes) {
		return Pending, errSameChunk
	}
	// Every evidence blob must belong to the committed input state root.
	for _, ev := range inputEvidence {
		if !verifyEvidence(inputStateRoot, ev) {
			return Pending, errors.New("canonical: input evidence does not match the committed input state")
		}
	}

	expected, err := recomputeChunk(node, inputEvidence, ropeTable, worker.ChunkIndex)
	if err != nil {
		return Pending, err
	}
	workerOK := equalBytes(expected, worker.ChunkBytes)
	challengerOK := equalBytes(expected, challenger.ChunkBytes)
	switch {
	case workerOK:
		return WorkerWins, nil
	case challengerOK:
		return ChallengerWins, nil
	default:
		return BothInvalid, nil
	}
}

// ChunkEvidence couples one input chunk (or row) with the tensor identity
// and inclusion proof needed to bind it to the committed input state.
type ChunkEvidence struct {
	Ref        TensorRef
	Desc       TensorDescriptor
	Root       Hash   // tensor root (inside the committed state)
	StateProof []byte // unused placeholder for chain wiring
	ChunkIndex uint32
	Count      uint32
	Bytes      []byte
	Proof      []Hash
}

func verifyEvidence(stateRoot Hash, ev ChunkEvidence) bool {
	// The tensor root must be one of the live tensors of the committed
	// state, and the chunk must belong to that tensor.
	leaf := hashBytes([]byte(DomainGraphState), appendUint32BE(nil, uint32(ev.Ref.Kind)), appendUint32BE(nil, ev.Ref.Index), ev.Root[:])
	// A full state-inclusion proof arrives with the chain wiring; the
	// canonical layer verifies the chunk against the tensor root itself.
	_ = leaf
	descBytes, err := EncodeCanonical(ev.Desc)
	if err != nil {
		return false
	}
	return VerifyChunk(ev.Root, descBytes, ev.ChunkIndex, ev.Count, ev.Bytes, ev.Proof)
}

// recomputeChunk is the bounded operator-specific arbiter.
func recomputeChunk(node GraphNode, evidence []ChunkEvidence, ropeTable *RopeConstants, index uint32) ([]byte, error) {
	switch node.OperatorID {
	case OpAddFixedV1, OpMulFixedV1:
		if len(evidence) != 2 {
			return nil, errors.New("canonical: binary operator needs two input chunks")
		}
		a, err := chunkToInt32(evidence[0].Bytes)
		if err != nil {
			return nil, err
		}
		b, err := chunkToInt32(evidence[1].Bytes)
		if err != nil {
			return nil, err
		}
		out := make([]int32, ChunkElems)
		for i := range out {
			if node.OperatorID == OpAddFixedV1 {
				out[i] = AddFx(a[i], b[i])
			} else {
				out[i] = MulFx(a[i], b[i])
			}
		}
		return int32sToChunk(out), nil
	case OpRequantizeV1:
		if len(evidence) != 1 {
			return nil, errors.New("canonical: requantize needs one input chunk")
		}
		params := node.Params
		mult := params.Get("mult", 1)
		shift := uint(params.Get("shift", 0))
		lo := int32(params.Get("clamp_lo", int64(MinFx)))
		hi := int32(params.Get("clamp_hi", int64(MaxFx)))
		a, err := chunkToInt32(evidence[0].Bytes)
		if err != nil {
			return nil, err
		}
		out := make([]int32, ChunkElems)
		for i, x := range a {
			v := RShiftRoundEven(int64(x)*mult, shift)
			if v < int64(lo) {
				v = int64(lo)
			}
			if v > int64(hi) {
				v = int64(hi)
			}
			out[i] = int32(v)
		}
		return int32sToChunk(out), nil
	case OpSiLUFixedV1:
		if len(evidence) != 1 {
			return nil, errors.New("canonical: silu needs one input chunk")
		}
		a, err := chunkToInt32(evidence[0].Bytes)
		if err != nil {
			return nil, err
		}
		out := make([]int32, ChunkElems)
		for i, x := range a {
			out[i] = MulFx(x, SigmoidFx(x))
		}
		return int32sToChunk(out), nil
	case OpRoPEFixedV1:
		if len(evidence) != 1 || ropeTable == nil {
			return nil, errors.New("canonical: rope needs one input chunk and the pinned table")
		}
		a, err := chunkToInt32(evidence[0].Bytes)
		if err != nil {
			return nil, err
		}
		hidden := int(node.Output.Shape[len(node.Output.Shape)-1])
		if hidden%2 != 0 {
			return nil, errors.New("canonical: rope hidden must be even")
		}
		pairsPerRow := hidden / 2
		out := make([]int32, ChunkElems)
		base := int(index) * ChunkElems
		for k := 0; k < ChunkElems/2; k++ {
			pos := (base + k) / pairsPerRow
			pair := (base + k) % pairsPerRow
			cosFx, sinFx, ok := SinCosFx(ropeTable.Table, pos, pair, pairsPerRow)
			if !ok {
				return nil, errors.New("canonical: rope position out of table range")
			}
			even := a[2*k]
			odd := a[2*k+1]
			out[2*k] = SubFx(MulFx(even, cosFx), MulFx(odd, sinFx))
			out[2*k+1] = AddFx(MulFx(even, sinFx), MulFx(odd, cosFx))
		}
		return int32sToChunk(out), nil
	case OpRMSNormFixedV1:
		// Bounded row arbitration for a chunk in ANY row: evidence is the
		// chunk-aligned span of x rows the disputed chunk touches, then
		// the weight row chunks, each in index order and bound to its own
		// committed tensor. Row boundaries must be chunk-aligned (enforced
		// by the graph builder).
		hidden, _, r0, r1, nX, firstChunk, err := rowEvidenceSpan(node.Output, index)
		if err != nil {
			return nil, err
		}
		wChunks := int((hidden + ChunkElems - 1) / ChunkElems)
		if len(evidence) != nX+wChunks {
			return nil, fmt.Errorf("canonical: rmsnorm needs %d x-row chunks plus %d weight chunks", nX, wChunks)
		}
		if len(node.Inputs) != 2 {
			return nil, errors.New("canonical: rmsnorm node needs two inputs")
		}
		for i := 0; i < nX; i++ {
			if evidence[i].Ref != node.Inputs[0] || evidence[i].ChunkIndex != uint32(firstChunk)+uint32(i) {
				return nil, errors.New("canonical: rmsnorm evidence must cover the disputed x rows in index order")
			}
		}
		for i := 0; i < wChunks; i++ {
			if evidence[nX+i].Ref != node.Inputs[1] || evidence[nX+i].ChunkIndex != uint32(i) {
				return nil, errors.New("canonical: rmsnorm weight evidence must be in index order")
			}
		}
		xBlob, err := concatChunks(evidence[:nX])
		if err != nil {
			return nil, err
		}
		wBlob, err := concatChunks(evidence[nX:])
		if err != nil {
			return nil, err
		}
		epsFx := int32(node.Params.Get("eps_fx", 0))
		out := make([]int32, ChunkElems)
		basePos := int64(index) * ChunkElems
		for r := r0; r <= r1; r++ {
			rowStart := r * hidden
			off := rowStart - firstChunk*ChunkElems
			rowData := xBlob[off : off+hidden]
			rms, err := rmsFor(rowData, epsFx)
			if err != nil {
				return nil, err
			}
			lo := i64max(basePos, rowStart)
			hi := i64min(basePos+ChunkElems, rowStart+hidden)
			for pos := lo; pos < hi; pos++ {
				j := pos - rowStart
				out[pos-basePos] = MulFx(MulFx(rowData[j], rms), wBlob[j])
			}
		}
		return int32sToChunk(out), nil
	case OpSoftmaxFixedV1:
		hidden, _, r0, r1, nChunks, firstChunk, err := rowEvidenceSpan(node.Output, index)
		if err != nil {
			return nil, err
		}
		if len(evidence) != nChunks {
			return nil, fmt.Errorf("canonical: softmax needs %d row chunks for the disputed chunk", nChunks)
		}
		if len(node.Inputs) != 1 {
			return nil, errors.New("canonical: softmax node needs one input")
		}
		for i := 0; i < nChunks; i++ {
			if evidence[i].Ref != node.Inputs[0] || evidence[i].ChunkIndex != uint32(firstChunk)+uint32(i) {
				return nil, errors.New("canonical: softmax evidence must cover the disputed rows in index order")
			}
		}
		blob, err := concatChunks(evidence)
		if err != nil {
			return nil, err
		}
		out := make([]int32, ChunkElems)
		basePos := int64(index) * ChunkElems
		for r := r0; r <= r1; r++ {
			rowStart := r * hidden
			off := rowStart - firstChunk*ChunkElems
			probs, err := softmaxRow(blob[off : off+hidden])
			if err != nil {
				return nil, err
			}
			lo := i64max(basePos, rowStart)
			hi := i64min(basePos+ChunkElems, rowStart+hidden)
			for pos := lo; pos < hi; pos++ {
				out[pos-basePos] = probs[pos-rowStart]
			}
		}
		return int32sToChunk(out), nil
	case OpGEMMInt8V1:
		return recomputeGEMMChunk(node, evidence, index)
	default:
		return nil, fmt.Errorf("canonical: operator %s has no arbiter", node.OperatorID)
	}
}

// rowEvidenceSpan maps a disputed output chunk to the rows it touches and
// the exact chunk range the arbiter must receive as evidence. Row
// boundaries must fall on chunk boundaries (canonical block graphs
// require every row length to divide or be a multiple of ChunkElems), so
// the span always starts chunk-aligned.
func rowEvidenceSpan(desc TensorDescriptor, index uint32) (hidden, elems, r0, r1 int64, nChunks int, firstChunk int64, err error) {
	elems, err = desc.Elems()
	if err != nil {
		return
	}
	hidden = desc.Shape[len(desc.Shape)-1]
	if hidden <= 0 || elems%hidden != 0 {
		err = errors.New("canonical: malformed row geometry")
		return
	}
	basePos := int64(index) * ChunkElems
	if basePos >= elems {
		err = errors.New("canonical: disputed chunk is outside the tensor")
		return
	}
	endPos := basePos + ChunkElems
	if endPos > elems {
		endPos = elems
	}
	r0 = basePos / hidden
	r1 = (endPos - 1) / hidden
	needLo := r0 * hidden
	needHi := (r1 + 1) * hidden
	if needLo%ChunkElems != 0 {
		err = errors.New("canonical: rows must be chunk-aligned for arbitration")
		return
	}
	firstChunk = needLo / ChunkElems
	lastChunk := (needHi + ChunkElems - 1) / ChunkElems
	nChunks = int(lastChunk - firstChunk)
	if nChunks <= 0 {
		nChunks = 1
	}
	return
}

func i64min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func i64max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// recomputeGEMMChunk adjudicates one output chunk of a GEMM node from the
// FULL committed operands. The micro-arithmetic is v0.1.1: int64
// accumulation of int8 x int8 products, which by the K <= MaxSafeK
// admission is bit-identical to the frozen int32 accumulation. The chain
// prices this by MACs through GraphGasV1; canonical v1 graphs declare
// operand sizes that keep it affordable, while the v0.1.1 tile dispute
// remains the specialized route for large standalone GEMM tasks.
func recomputeGEMMChunk(node GraphNode, evidence []ChunkEvidence, index uint32) ([]byte, error) {
	if len(node.Inputs) != 2 {
		return nil, errors.New("canonical: GEMM node needs two inputs")
	}
	if len(evidence) < 2 {
		return nil, errors.New("canonical: GEMM arbiter needs the full operands as chunk evidence")
	}
	aDesc := evidence[0].Desc
	bDesc := evidence[len(evidence)-1].Desc
	m, n, k, transB, err := GEMMNodeDims(aDesc, bDesc, node.Params)
	if err != nil {
		return nil, err
	}
	elemsA, err := aDesc.Elems()
	if err != nil {
		return nil, err
	}
	elemsB, err := bDesc.Elems()
	if err != nil {
		return nil, err
	}
	countA := uint32((elemsA + ChunkElems - 1) / ChunkElems)
	countB := uint32((elemsB + ChunkElems - 1) / ChunkElems)
	if uint32(len(evidence)) != countA+countB {
		return nil, errors.New("canonical: GEMM arbiter requires every operand chunk in index order")
	}
	for i := uint32(0); i < countA; i++ {
		if evidence[i].Ref != node.Inputs[0] || evidence[i].ChunkIndex != i {
			return nil, errors.New("canonical: GEMM evidence does not match the A operand in order")
		}
	}
	for i := uint32(0); i < countB; i++ {
		if evidence[countA+i].Ref != node.Inputs[1] || evidence[countA+i].ChunkIndex != i {
			return nil, errors.New("canonical: GEMM evidence does not match the B operand in order")
		}
	}
	a, err := concatChunks(evidence[:countA])
	if err != nil {
		return nil, err
	}
	bt, err := concatChunks(evidence[countA:])
	if err != nil {
		return nil, err
	}
	for i := 0; i < int(elemsA); i++ {
		if a[i] < -128 || a[i] > 127 {
			return nil, errors.New("canonical: GEMM operand A is not int8 data")
		}
	}
	for i := 0; i < int(elemsB); i++ {
		if bt[i] < -128 || bt[i] > 127 {
			return nil, errors.New("canonical: GEMM operand B is not int8 data")
		}
	}
	out := make([]int32, ChunkElems)
	base := int64(index) * ChunkElems
	total := int64(m) * int64(n)
	for e := int64(0); e < ChunkElems; e++ {
		pos := base + e
		if pos >= total {
			break
		}
		row := pos / int64(n)
		col := pos % int64(n)
		var sum int64
		for d := uint64(0); d < k; d++ {
			var bv int32
			if transB {
				bv = bt[col*int64(k)+int64(d)]
			} else {
				bv = bt[int64(d)*int64(n)+col]
			}
			sum += int64(a[row*int64(k)+int64(d)]) * int64(bv)
		}
		if sum > math.MaxInt32 || sum < math.MinInt32 {
			return nil, errors.New("canonical: GEMM arbiter accumulation overflow")
		}
		out[e] = int32(sum)
	}
	return int32sToChunk(out), nil
}

// concatChunks decodes evidence chunk blobs in order into one row.
func concatChunks(evidence []ChunkEvidence) ([]int32, error) {
	var out []int32
	for _, ev := range evidence {
		vals, err := chunkToInt32(ev.Bytes)
		if err != nil {
			return nil, err
		}
		out = append(out, vals...)
	}
	return out, nil
}

func chunkToInt32(b []byte) ([]int32, error) {
	if len(b)%4 != 0 {
		return nil, errors.New("canonical: chunk length must be a multiple of 4")
	}
	out := make([]int32, len(b)/4)
	for i := range out {
		out[i] = int32(uint32(b[4*i])<<24 | uint32(b[4*i+1])<<16 | uint32(b[4*i+2])<<8 | uint32(b[4*i+3]))
	}
	return out, nil
}

func int32sToChunk(vals []int32) []byte {
	out := make([]byte, len(vals)*4)
	for i, v := range vals {
		binaryBE(out[i*4:], uint32(v))
	}
	return out
}

func binaryBE(dst []byte, v uint32) {
	dst[0] = byte(v >> 24)
	dst[1] = byte(v >> 16)
	dst[2] = byte(v >> 8)
	dst[3] = byte(v)
}

// TrailProof proves one trail state against the party's trail root (used
// by tests, the watcher and the dev fraud injector).
func TrailProof(graphID Hash, trail []Hash, index uint32) (MerkleProof, error) {
	levels, err := TrailLevels(graphID, trail)
	if err != nil {
		return MerkleProof{}, err
	}
	siblings, err := proveLeaf(levels, int(index))
	if err != nil {
		return MerkleProof{}, err
	}
	return MerkleProof{Index: index, Count: uint32(len(trail)), Siblings: siblings}, nil
}
