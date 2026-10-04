package canonical

// CANONICAL_GRAPH_V2: the versioned wide-integer graph family (Phase F.5C).
//
// V2 exists because the A13W10 int64 GEMM accumulator cannot be legally
// committed under the V1 int32 TensorRoot encoding.  Everything here is
// additive: V1 descriptors, GraphIDs, commits and receipts keep their
// domains and bytes; a V1 object can never verify under a V2 domain.
//
// Operator set: the frozen cheap operators (ADD/MUL/RMSNORM/ROPE/SILU/
// SOFTMAX, same IDs and versions as V1, quantized values in Q12.20) plus
// exactly two new versioned operators required by A13W10:
//   GEMM_A13W10_I64_V1   (A13 x W10 -> INT64_ACCUM)
//   REQUANTIZE_WIDE_V1   (INT64_ACCUM/Q12.20 -> Q12.20/A13)
// GPU decomposition (Karatsuba, tensor cores) is a worker backend detail
// and never enters any consensus object.

import (
	"crypto/ed25519"
	"errors"
	"fmt"
)

// DomainSigV3 separates V3 commitment signatures.
const DomainSigV3 = "PRISMA_CANONICAL_SIG_V3\x00"

// maxGraphNodesV2 mirrors the chain admission bound for standalone
// canonical-layer validation.
const maxGraphNodesV2 = 512

// V2 operator IDs.
const (
	OpGEMMWideA13W10  = "GEMM_A13W10_I64_V1"
	OpRequantizeWideV1 = "REQUANTIZE_WIDE_V1"
)

// --- V2 operator interface and registry --------------------------------------

// OperatorV2 is the V2 mirror of Operator with V2 tensors.
type OperatorV2 interface {
	ID() string
	Version() string
	ValidateParamsV2(p ParamList) error
	OutputSpecV2(in []TensorDescriptorV2, p ParamList) (TensorDescriptorV2, error)
	WorkV2(in []*TensorV2, p ParamList) WorkVector
	ExecuteV2(in []*TensorV2, p ParamList) (*TensorV2, error)
	ArbiterBoundV2(in []TensorDescriptorV2, p ParamList) int
}

var registryV2 = map[string]OperatorV2{}

func RegisterV2(op OperatorV2) {
	if _, exists := registryV2[op.ID()]; exists {
		panic("canonical: duplicate V2 operator registration " + op.ID())
	}
	registryV2[op.ID()] = op
}

func LookupV2(id string) (OperatorV2, error) {
	op, ok := registryV2[id]
	if !ok {
		return nil, fmt.Errorf("canonical: V2 operator %q is not supported", id)
	}
	return op, nil
}

func OperatorIDsV2() []string {
	ids := make([]string, 0, len(registryV2))
	for id := range registryV2 {
		ids = append(ids, id)
	}
	return ids
}

func init() {
	RegisterV2(gemmA13W10V1{})
	RegisterV2(requantizeWideV1{})
	// The frozen cheap operators are reused by ID under their V1 versions.
	RegisterV2(cheapV2{id: OpAddFixedV1, v1: addFixed{}})
	RegisterV2(cheapV2{id: OpMulFixedV1, v1: mulFixed{}})
	RegisterV2(cheapV2{id: OpRMSNormFixedV1, v1: rmsNormFixed{}})
	RegisterV2(cheapV2{id: OpRoPEFixedV1, v1: ropeFixed{}})
	RegisterV2(cheapV2{id: OpSiLUFixedV1, v1: siluFixed{}})
	RegisterV2(cheapV2{id: OpSoftmaxFixedV1, v1: softmaxFixed{}})
}

// --- tensor bridge (cheap operators reuse the V1 kernels on Q12.20) ----------

func v2ToV1Tensor(t *TensorV2) (*Tensor, error) {
	if t.Desc.Dtype != DtypeV2Q12_20 {
		return nil, fmt.Errorf("canonical/v2: dtype %d cannot bridge to the Q12.20 kernels", t.Desc.Dtype)
	}
	data := make([]int32, len(t.Data))
	for i, v := range t.Data {
		data[i] = int32(v)
	}
	desc := NewDesc(DtypeQ12_20, t.Desc.Shape...)
	return &Tensor{Desc: desc, Data: data}, nil
}

func v1ToV2Tensor(t *Tensor) (*TensorV2, error) {
	if Dtype(t.Desc.Dtype) != DtypeQ12_20 {
		return nil, errors.New("canonical/v2: bridge expects a Q12.20 result")
	}
	data := make([]int64, len(t.Data))
	for i, v := range t.Data {
		data[i] = int64(v)
	}
	return NewTensorV2(NewDescV2(DtypeV2Q12_20, t.Desc.Shape...), data)
}

func v2DescToV1(d TensorDescriptorV2) (TensorDescriptor, error) {
	if d.Dtype != DtypeV2Q12_20 {
		return TensorDescriptor{}, fmt.Errorf("canonical/v2: dtype %d has no V1 descriptor", d.Dtype)
	}
	return NewDesc(DtypeQ12_20, d.Shape...), nil
}

func v1DescToV2(d TensorDescriptor) TensorDescriptorV2 {
	return NewDescV2(DtypeV2Q12_20, d.Shape...)
}

// cheapV2 wraps a frozen V1 Q12.20 operator into the V2 registry. The
// arithmetic is literally the same code path; only the typed container
// changes (BE32 Q12.20 bytes are identical in both families).
type cheapV2 struct {
	id string
	v1 Operator
}

func (c cheapV2) ID() string      { return c.id }
func (c cheapV2) Version() string { return c.v1.Version() }

func (c cheapV2) ValidateParamsV2(p ParamList) error { return c.v1.ValidateParams(p) }

func (c cheapV2) OutputSpecV2(in []TensorDescriptorV2, p ParamList) (TensorDescriptorV2, error) {
	v1in := make([]TensorDescriptor, len(in))
	for i, d := range in {
		v1d, err := v2DescToV1(d)
		if err != nil {
			return TensorDescriptorV2{}, err
		}
		v1in[i] = v1d
	}
	out, err := c.v1.OutputSpec(v1in, p)
	if err != nil {
		return TensorDescriptorV2{}, err
	}
	return v1DescToV2(out), nil
}

func (c cheapV2) WorkV2(in []*TensorV2, p ParamList) WorkVector {
	v1in := make([]Tensor, len(in))
	for i, t := range in {
		v1t, err := v2ToV1Tensor(t)
		if err != nil {
			// Work accounting only needs shapes; fall back to a shape-only
			// tensor when the bridge refuses (never happens for validated
			// graphs, but keeps Work total).
			v1in[i] = Tensor{Desc: NewDesc(DtypeQ12_20, t.Desc.Shape...)}
			continue
		}
		v1in[i] = *v1t
	}
	return c.v1.Work(v1in, p)
}

func (c cheapV2) ExecuteV2(in []*TensorV2, p ParamList) (*TensorV2, error) {
	v1in := make([]Tensor, len(in))
	for i, t := range in {
		v1t, err := v2ToV1Tensor(t)
		if err != nil {
			return nil, err
		}
		v1in[i] = *v1t
	}
	if c.id == OpRoPEFixedV1 {
		// The RoPE constant table is the last input; the frozen V1 kernel
		// consumes it as (cos, sin) Q12.20 pairs.
		table := v1in[len(v1in)-1]
		constants, err := NewRopeConstants(table.Data, int(p.Get("half_dim", 0)))
		if err != nil {
			return nil, err
		}
		out, err := ExecuteRope(&v1in[0], constants, p)
		if err != nil {
			return nil, err
		}
		return v1ToV2Tensor(out)
	}
	out, err := c.v1.Execute(v1in, p)
	if err != nil {
		return nil, err
	}
	return v1ToV2Tensor(out)
}

func (c cheapV2) ArbiterBoundV2(in []TensorDescriptorV2, p ParamList) int {
	v1in := make([]TensorDescriptor, len(in))
	for i, d := range in {
		v1d, err := v2DescToV1(d)
		if err != nil {
			return 0
		}
		v1in[i] = v1d
	}
	return c.v1.ArbiterBound(v1in, p)
}

// --- GEMM_A13W10_I64_V1 -------------------------------------------------------

type gemmA13W10V1 struct{}

func (gemmA13W10V1) ID() string      { return OpGEMMWideA13W10 }
func (gemmA13W10V1) Version() string { return OpGEMMVersionWide }

func (gemmA13W10V1) ValidateParamsV2(p ParamList) error {
	tb := p.Get("transpose_b", 0)
	if tb != 0 && tb != 1 {
		return errors.New("canonical/v2: GEMM_A13W10 transpose_b must be 0 or 1")
	}
	return nil
}

func (gemmA13W10V1) OutputSpecV2(in []TensorDescriptorV2, p ParamList) (TensorDescriptorV2, error) {
	if len(in) != 2 {
		return TensorDescriptorV2{}, errors.New("canonical/v2: GEMM_A13W10 needs A and W")
	}
	a, w := in[0], in[1]
	if a.Dtype != DtypeV2A13 {
		return TensorDescriptorV2{}, errors.New("canonical/v2: GEMM_A13W10 left operand must be A13")
	}
	// The right operand is the logical signed weight operand: model weights
	// are W10, attention inner products (Q x K^T, P x V) use A13 containers.
	// Both are logical signed wide operands; the same exact int64 math
	// applies and admission bounds use the actual bit widths.
	if w.Dtype != DtypeV2W10 && w.Dtype != DtypeV2A13 {
		return TensorDescriptorV2{}, errors.New("canonical/v2: GEMM_A13W10 right operand must be W10 or A13")
	}
	wBits := 10
	if w.Dtype == DtypeV2A13 {
		wBits = 13
	}
	if len(a.Shape) != 2 || len(w.Shape) != 2 {
		return TensorDescriptorV2{}, errors.New("canonical/v2: GEMM_A13W10 operands must be matrices")
	}
	m, k := a.Shape[0], a.Shape[1]
	tb := p.Get("transpose_b", 0) == 1
	var n int64
	if tb {
		if w.Shape[0] <= 0 || w.Shape[1] != k {
			return TensorDescriptorV2{}, errors.New("canonical/v2: GEMM_A13W10 (N,K) weight shape mismatch")
		}
		n = w.Shape[0]
	} else {
		if w.Shape[0] != k || w.Shape[1] <= 0 {
			return TensorDescriptorV2{}, errors.New("canonical/v2: GEMM_A13W10 (K,N) weight shape mismatch")
		}
		n = w.Shape[1]
	}
	if m <= 0 || n <= 0 || k <= 0 {
		return TensorDescriptorV2{}, errors.New("canonical/v2: GEMM_A13W10 empty dimension")
	}
	// Admission: the int64 accumulator must be provably safe for every
	// admitted K, not only the observed model range.
	if uint64(k) > MaxSafeK64(13, wBits) {
		return TensorDescriptorV2{}, fmt.Errorf("canonical/v2: GEMM_A13W10 K=%d exceeds MaxSafeK64", k)
	}
	return NewDescV2(DtypeV2Int64Accum, m, n), nil
}

func (gemmA13W10V1) WorkV2(in []*TensorV2, p ParamList) WorkVector {
	if len(in) != 2 {
		return nil
	}
	m, k := in[0].Desc.Shape[0], in[0].Desc.Shape[1]
	n := in[1].Desc.Shape[1]
	if p.Get("transpose_b", 0) == 1 {
		n = in[1].Desc.Shape[0]
	}
	return WorkVector{}.Add("GEMM_A13W10_MAC", m*n*k)
}

func (gemmA13W10V1) ExecuteV2(in []*TensorV2, p ParamList) (*TensorV2, error) {
	if len(in) != 2 {
		return nil, errors.New("canonical/v2: GEMM_A13W10 needs A and W")
	}
	out, err := gemmA13W10V1{}.OutputSpecV2([]TensorDescriptorV2{in[0].Desc, in[1].Desc}, p)
	if err != nil {
		return nil, err
	}
	m, k := in[0].Desc.Shape[0], in[0].Desc.Shape[1]
	tb := p.Get("transpose_b", 0) == 1
	n := in[1].Desc.Shape[1]
	if tb {
		n = in[1].Desc.Shape[0]
	}
	data, err := ReferenceWideGEMM(in[0].Data, in[1].Data, int(m), int(n), int(k), tb,
		in[1].Desc.Dtype == DtypeV2A13)
	if err != nil {
		return nil, err
	}
	return NewTensorV2(out, data)
}

func (gemmA13W10V1) ArbiterBoundV2(in []TensorDescriptorV2, p ParamList) int {
	if len(in) != 2 {
		return 0
	}
	// One disputed output element depends on the whole K row/column; the
	// bounded on-chain arbiter never sees that, it only adjudicates a
	// single 8x8x8 step through the wide dispute path.
	return 0
}

// --- REQUANTIZE_WIDE_V1 -------------------------------------------------------

type requantizeWideV1 struct{}

func (requantizeWideV1) ID() string      { return OpRequantizeWideV1 }
func (requantizeWideV1) Version() string { return OpRequantVersionWide }

func (requantizeWideV1) ValidateParamsV2(p ParamList) error {
	mult := p.Get("mult", 0)
	if mult <= 0 || mult > 1<<62 {
		return errors.New("canonical/v2: REQUANTIZE_WIDE mult outside admission bounds")
	}
	shift := p.Get("shift", 0)
	if shift < 0 || shift > 62 {
		return errors.New("canonical/v2: REQUANTIZE_WIDE shift outside admission bounds")
	}
	lo, hi := p.Get("clamp_lo", MinFx), p.Get("clamp_hi", MaxFx)
	if lo > hi {
		return errors.New("canonical/v2: REQUANTIZE_WIDE clamp_lo > clamp_hi")
	}
	return nil
}

func (r requantizeWideV1) OutputSpecV2(in []TensorDescriptorV2, p ParamList) (TensorDescriptorV2, error) {
	if len(in) != 1 {
		return TensorDescriptorV2{}, errors.New("canonical/v2: REQUANTIZE_WIDE takes one input")
	}
	switch in[0].Dtype {
	case DtypeV2Int64Accum, DtypeV2Q12_20, DtypeV2A13:
	default:
		return TensorDescriptorV2{}, fmt.Errorf("canonical/v2: REQUANTIZE_WIDE cannot take dtype %d", in[0].Dtype)
	}
	outDtype := DtypeV2(p.Get("out_dtype", int64(DtypeV2Q12_20)))
	if outDtype != DtypeV2Q12_20 && outDtype != DtypeV2A13 {
		return TensorDescriptorV2{}, errors.New("canonical/v2: REQUANTIZE_WIDE output must be Q12.20 or A13")
	}
	if err := r.ValidateParamsV2(p); err != nil {
		return TensorDescriptorV2{}, err
	}
	out := NewDescV2(outDtype, in[0].Shape...)
	// The clamp is part of the declared output range; the target dtype
	// range is a hard bound on top.
	lo, hi := p.Get("clamp_lo", MinFx), p.Get("clamp_hi", MaxFx)
	if outDtype == DtypeV2A13 && (lo < A13Min || hi > A13Max) {
		return TensorDescriptorV2{}, errors.New("canonical/v2: A13 requant clamp outside A13 range")
	}
	return out, nil
}

func (requantizeWideV1) WorkV2(in []*TensorV2, _ ParamList) WorkVector {
	if len(in) != 1 {
		return nil
	}
	n, _ := in[0].Desc.Elems()
	return WorkVector{}.Add("REQUANTIZE_WIDE_ELEMENT", n)
}

func (requantizeWideV1) ExecuteV2(in []*TensorV2, p ParamList) (*TensorV2, error) {
	if len(in) != 1 {
		return nil, errors.New("canonical/v2: REQUANTIZE_WIDE takes one input")
	}
	out, err := requantizeWideV1{}.OutputSpecV2([]TensorDescriptorV2{in[0].Desc}, p)
	if err != nil {
		return nil, err
	}
	data, err := RequantWide(in[0].Data, uint64(p.Get("mult", 0)), uint(p.Get("shift", 0)),
		p.Get("clamp_lo", MinFx), p.Get("clamp_hi", MaxFx), out.Dtype)
	if err != nil {
		return nil, err
	}
	return NewTensorV2(out, data)
}

func (requantizeWideV1) ArbiterBoundV2(in []TensorDescriptorV2, _ ParamList) int { return 0 }

// --- graph descriptor V2 ------------------------------------------------------

type GraphInputV2 struct {
	Name string             `gemm:"name" json:"name"`
	Desc TensorDescriptorV2 `gemm:"desc" json:"desc"`
	Root []byte             `gemm:"root" json:"root"`
}

type GraphNodeV2 struct {
	NodeID     uint32             `gemm:"node_id" json:"node_id"`
	OperatorID string             `gemm:"operator_id" json:"operator_id"`
	Version    string             `gemm:"operator_version" json:"operator_version"`
	Inputs     []TensorRef        `gemm:"inputs" json:"inputs"`
	Output     TensorDescriptorV2 `gemm:"output" json:"output"`
	Params     ParamList          `gemm:"params" json:"params"`
}

// GraphDescriptorV2 binds the arithmetic profile and policy id (§18): the
// A13W10 policy is never an off-chain implicit config.
type GraphDescriptorV2 struct {
	ProtocolVersion string             `gemm:"protocol_version" json:"protocol_version"`
	Spec            string             `gemm:"spec" json:"spec"`
	Arithmetic      ArithmeticProfileV1 `gemm:"arithmetic" json:"arithmetic"`
	Inputs          []GraphInputV2     `gemm:"inputs" json:"inputs"`
	Nodes           []GraphNodeV2      `gemm:"nodes" json:"nodes"`
	Outputs         []TensorRef        `gemm:"outputs" json:"outputs"`
}

// GraphIDV2 = SHA256(PRISMA_CANONICAL_GRAPH_V2 || CBOR(descriptor)).
func (g *GraphDescriptorV2) GraphIDV2() (Hash, error) {
	return canonicalObjectHash(DomainGraphV2, g)
}

func (g *GraphDescriptorV2) ValidateV2() error {
	if g.ProtocolVersion != ProtocolVersionGraphV2 {
		return errors.New("canonical/v2: graph protocol_version mismatch")
	}
	if g.Arithmetic != A13W10I64Profile() {
		return errors.New("canonical/v2: graph arithmetic profile mismatch")
	}
	if len(g.Nodes) == 0 || len(g.Nodes) > maxGraphNodesV2 {
		return errors.New("canonical/v2: graph node count outside bounds")
	}
	seen := map[uint32]bool{}
	for i, node := range g.Nodes {
		if node.NodeID != uint32(i) {
			return fmt.Errorf("canonical/v2: node ids must be dense and ordered, got %d at %d", node.NodeID, i)
		}
		if seen[node.NodeID] {
			return fmt.Errorf("canonical/v2: duplicate node id %d", node.NodeID)
		}
		seen[node.NodeID] = true
		op, err := LookupV2(node.OperatorID)
		if err != nil {
			return err
		}
		if node.Version != op.Version() {
			return fmt.Errorf("canonical/v2: node %d version mismatch for %s", node.NodeID, node.OperatorID)
		}
		if err := node.Output.Validate(); err != nil {
			return fmt.Errorf("canonical/v2: node %d output: %w", node.NodeID, err)
		}
		descs := make([]TensorDescriptorV2, 0, len(node.Inputs))
		for _, ref := range node.Inputs {
			desc, err := g.descForV2(ref, node.NodeID)
			if err != nil {
				return err
			}
			descs = append(descs, desc)
		}
		want, err := op.OutputSpecV2(descs, node.Params)
		if err != nil {
			return fmt.Errorf("canonical/v2: node %d: %w", node.NodeID, err)
		}
		if !sameDescV2(want, node.Output) {
			return fmt.Errorf("canonical/v2: node %d declared output does not match the operator spec", node.NodeID)
		}
	}
	for _, out := range g.Outputs {
		if _, err := g.descForV2(out, 0); err != nil {
			return err
		}
	}
	return nil
}

func (g *GraphDescriptorV2) descForV2(ref TensorRef, before uint32) (TensorDescriptorV2, error) {
	switch ref.Kind {
	case 0:
		if int(ref.Index) >= len(g.Inputs) {
			return TensorDescriptorV2{}, fmt.Errorf("canonical/v2: input ref %d out of range", ref.Index)
		}
		return g.Inputs[ref.Index].Desc, nil
	case 1:
		if ref.Index >= before && before != 0 {
			return TensorDescriptorV2{}, fmt.Errorf("canonical/v2: forward reference to node %d", ref.Index)
		}
		if int(ref.Index) >= len(g.Nodes) {
			return TensorDescriptorV2{}, fmt.Errorf("canonical/v2: node ref %d out of range", ref.Index)
		}
		return g.Nodes[ref.Index].Output, nil
	default:
		return TensorDescriptorV2{}, fmt.Errorf("canonical/v2: unknown ref kind %d", ref.Kind)
	}
}

func sameDescV2(a, b TensorDescriptorV2) bool {
	if a.Dtype != b.Dtype || a.Layout != b.Layout || len(a.Shape) != len(b.Shape) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	return true
}

// --- execution ---------------------------------------------------------------

type GraphExecutionV2 struct {
	Tensors    map[TensorRef]*TensorV2
	Trail      []Hash // len = nodes+1
	Outputs    []Hash
	WorkVector WorkVector
}

// StateLeafV2 hashes one live tensor of a V2 graph state tree.
func StateLeafV2(kind uint8, index uint32, root Hash) Hash {
	return hashBytes([]byte(DomainGraphStateV2), appendUint32BE(nil, uint32(kind)),
		appendUint32BE(nil, index), root[:])
}

func stateRootForV2(tensors map[TensorRef]*TensorV2) (Hash, error) {
	leaves := make([]Hash, 0, len(tensors))
	ids := make([]uint64, 0, len(tensors))
	byID := map[uint64]TensorRef{}
	for ref := range tensors {
		id := uint64(ref.Kind)<<32 | uint64(ref.Index)
		ids = append(ids, id)
		byID[id] = ref
	}
	sortUint64s(ids)
	for _, id := range ids {
		root, err := tensors[byID[id]].TensorRootV2()
		if err != nil {
			return Hash{}, err
		}
		leaves = append(leaves, StateLeafV2(uint8(id>>32), uint32(id), root))
	}
	return MerkleRootOf(leaves)
}

func sortUint64s(v []uint64) {
	for i := 1; i < len(v); i++ {
		for j := i; j > 0 && v[j] < v[j-1]; j-- {
			v[j], v[j-1] = v[j-1], v[j]
		}
	}
}

// ExecuteGraphV2 runs the V2 graph with the reference operators.
func ExecuteGraphV2(g *GraphDescriptorV2, inputs map[uint32]*TensorV2) (*GraphExecutionV2, error) {
	if err := g.ValidateV2(); err != nil {
		return nil, err
	}
	tensors := map[TensorRef]*TensorV2{}
	var work WorkVector
	for idx, in := range g.Inputs {
		t, ok := inputs[uint32(idx)]
		if !ok {
			return nil, fmt.Errorf("canonical/v2: missing graph input %d (%s)", idx, in.Name)
		}
		if !sameDescV2(t.Desc, in.Desc) {
			return nil, fmt.Errorf("canonical/v2: input %d descriptor mismatch", idx)
		}
		root, err := t.TensorRootV2()
		if err != nil {
			return nil, err
		}
		if !equalBytes(root[:], in.Root) {
			return nil, fmt.Errorf("canonical/v2: input %d does not match its committed root", idx)
		}
		tensors[TensorRef{Kind: 0, Index: uint32(idx)}] = t
	}
	exec := &GraphExecutionV2{Tensors: tensors}
	initial, err := stateRootForV2(tensors)
	if err != nil {
		return nil, err
	}
	exec.Trail = append(exec.Trail, initial)
	for _, node := range g.Nodes {
		op, err := LookupV2(node.OperatorID)
		if err != nil {
			return nil, err
		}
		ins := make([]*TensorV2, 0, len(node.Inputs))
		for _, ref := range node.Inputs {
			t, ok := tensors[ref]
			if !ok {
				return nil, fmt.Errorf("canonical/v2: node %d missing input tensor", node.NodeID)
			}
			ins = append(ins, t)
		}
		work = workAppend(work, op.WorkV2(ins, node.Params))
		out, err := op.ExecuteV2(ins, node.Params)
		if err != nil {
			return nil, fmt.Errorf("canonical/v2: node %d: %w", node.NodeID, err)
		}
		tensors[TensorRef{Kind: 1, Index: node.NodeID}] = out
		state, err := stateRootForV2(tensors)
		if err != nil {
			return nil, err
		}
		exec.Trail = append(exec.Trail, state)
	}
	exec.WorkVector = work.Canonical()
	for _, ref := range g.Outputs {
		t, ok := tensors[ref]
		if !ok {
			return nil, errors.New("canonical/v2: graph output is not defined")
		}
		root, err := t.TensorRootV2()
		if err != nil {
			return nil, err
		}
		exec.Outputs = append(exec.Outputs, root)
	}
	return exec, nil
}

func hashToBytes(h Hash) []byte { return append([]byte(nil), h[:]...) }

func workAppend(w WorkVector, more WorkVector) WorkVector {
	for _, p := range more {
		w = w.Add(p.Key, p.Value)
	}
	return w
}

// --- node output manifest V2 ---------------------------------------------------

// NodeOutputLeafV2 binds (graph id V2, node, op, version, descriptor V2 hash).
func NodeOutputLeafV2(graphID Hash, node GraphNodeV2) (Hash, error) {
	descBytes, err := EncodeCanonical(node.Output)
	if err != nil {
		return Hash{}, err
	}
	return hashBytes([]byte(DomainManifestV2), graphID[:], appendUint32BE(nil, node.NodeID),
		[]byte(node.OperatorID), []byte(node.Version), descBytes), nil
}

// BuildNodeOutputManifestV2 commits every node output root; the leaves are
// combined with the node's output TensorRootV2 in node order.
func BuildNodeOutputManifestV2(g *GraphDescriptorV2, nodeRoots []Hash) (Hash, []Hash, error) {
	if err := g.ValidateV2(); err != nil {
		return Hash{}, nil, err
	}
	if len(nodeRoots) != len(g.Nodes) {
		return Hash{}, nil, errors.New("canonical/v2: manifest needs one root per node")
	}
	graphID, err := g.GraphIDV2()
	if err != nil {
		return Hash{}, nil, err
	}
	leaves := make([]Hash, len(g.Nodes))
	for i, node := range g.Nodes {
		leaf, err := NodeOutputLeafV2(graphID, node)
		if err != nil {
			return Hash{}, nil, err
		}
		leaves[i] = hashBytes(leaf[:], nodeRoots[i][:])
	}
	root, err := MerkleRootOf(leaves)
	if err != nil {
		return Hash{}, nil, err
	}
	return root, leaves, nil
}

// VerifyNodeOutputManifestLeafV2 proves one node's committed output root.
func VerifyNodeOutputManifestLeafV2(root Hash, graphID Hash, node GraphNodeV2,
	tensorRoot Hash, proof MerkleProof) bool {
	leaf, err := NodeOutputLeafV2(graphID, node)
	if err != nil {
		return false
	}
	combined := hashBytes(leaf[:], tensorRoot[:])
	return VerifyLeafInclusion(root, combined, proof)
}

// --- GraphResultCommitV3 -------------------------------------------------------

const GraphResultCommitV3Version = "3.0.0"

// GraphResultCommitV3 is the signed wide-graph result commitment.
type GraphResultCommitV3 struct {
	ProtocolVersion string `gemm:"protocol_version"`

	GraphID       []byte `gemm:"graph_id"`
	ArithmeticID  string `gemm:"arithmetic_id"`
	PolicyID      string `gemm:"policy_id"`
	TaskRef       []byte `gemm:"task_ref"`
	AssignmentRef []byte `gemm:"assignment_ref"`
	WorkerPubKey  []byte `gemm:"worker_pubkey"`

	NodeOutputManifestRootV2 []byte `gemm:"node_output_manifest_root_v2"`

	FinalOutputRoot []byte   `gemm:"final_output_root"`
	OutputRoots     [][]byte `gemm:"output_roots"`

	CompletedEpoch uint64 `gemm:"completed_epoch"`

	Signature []byte `gemm:"signature"`
}

func NewGraphResultCommitV3(g *GraphDescriptorV2, taskRef, assignmentRef, workerPubKey []byte,
	exec *GraphExecutionV2, completedEpoch uint64) (*GraphResultCommitV3, error) {
	if exec == nil {
		return nil, errors.New("canonical/v2: V3 commitment needs the execution")
	}
	nodeRoots := make([]Hash, len(g.Nodes))
	for i := range g.Nodes {
		t, ok := exec.Tensors[TensorRef{Kind: 1, Index: uint32(i)}]
		if !ok {
			return nil, errors.New("canonical/v2: execution is missing a node output")
		}
		root, err := t.TensorRootV2()
		if err != nil {
			return nil, err
		}
		nodeRoots[i] = root
	}
	manifestRoot, _, err := BuildNodeOutputManifestV2(g, nodeRoots)
	if err != nil {
		return nil, err
	}
	graphID, err := g.GraphIDV2()
	if err != nil {
		return nil, err
	}
	if len(exec.Outputs) == 0 {
		return nil, errors.New("canonical/v2: execution has no outputs")
	}
	roots := make([][]byte, len(exec.Outputs))
	for i, r := range exec.Outputs {
		roots[i] = append([]byte(nil), r[:]...)
	}
	return &GraphResultCommitV3{
		ProtocolVersion:          GraphResultCommitV3Version,
		GraphID:                  append([]byte(nil), graphID[:]...),
		ArithmeticID:             g.Arithmetic.ID,
		PolicyID:                 g.Arithmetic.PolicyID,
		TaskRef:                  append([]byte(nil), taskRef...),
		AssignmentRef:            append([]byte(nil), assignmentRef...),
		WorkerPubKey:             append([]byte(nil), workerPubKey...),
		NodeOutputManifestRootV2: append([]byte(nil), manifestRoot[:]...),
		FinalOutputRoot:          hashToBytes(FinalOutputRoot(exec.Outputs)),
		OutputRoots:              roots,
		CompletedEpoch:           completedEpoch,
	}, nil
}

func (rc *GraphResultCommitV3) signBytes() ([]byte, error) {
	unsigned := *rc
	unsigned.Signature = nil
	enc, err := EncodeCanonical(&unsigned)
	if err != nil {
		return nil, err
	}
	return append([]byte(DomainSigV3), enc...), nil
}

// SignGraphResultCommitV3 fills the worker signature.
func SignGraphResultCommitV3(rc *GraphResultCommitV3, key ed25519.PrivateKey) error {
	if len(rc.WorkerPubKey) != 32 {
		return errors.New("canonical/v2: result commit needs a 32-byte worker key")
	}
	preimage, err := rc.signBytes()
	if err != nil {
		return err
	}
	rc.Signature = ed25519.Sign(key, preimage)
	return nil
}

// VerifyGraphResultCommitV3Signature checks the V3 signature (separate
// domain: a V1/V2 signature can never validate V3 bytes).
func VerifyGraphResultCommitV3Signature(rc *GraphResultCommitV3) bool {
	if len(rc.WorkerPubKey) != 32 || len(rc.Signature) != ed25519.SignatureSize {
		return false
	}
	preimage, err := rc.signBytes()
	if err != nil {
		return false
	}
	return ed25519.Verify(rc.WorkerPubKey, preimage, rc.Signature)
}

// ValidateGraphResultCommitV3 re-derives every statically checkable bound.
func ValidateGraphResultCommitV3(g *GraphDescriptorV2, rc *GraphResultCommitV3,
	expectedManifestRoot []byte) error {
	if rc.ProtocolVersion != GraphResultCommitV3Version {
		return errors.New("canonical/v2: V3 result commit version mismatch")
	}
	graphID, err := g.GraphIDV2()
	if err != nil {
		return err
	}
	if !equalBytes(rc.GraphID, graphID[:]) {
		return errors.New("canonical/v2: V3 result commit does not bind this graph")
	}
	if rc.ArithmeticID != g.Arithmetic.ID || rc.PolicyID != g.Arithmetic.PolicyID {
		return errors.New("canonical/v2: V3 result commit arithmetic/policy mismatch")
	}
	if len(rc.NodeOutputManifestRootV2) != 32 {
		return errors.New("canonical/v2: V3 result commit manifest root malformed")
	}
	if expectedManifestRoot != nil && !equalBytes(rc.NodeOutputManifestRootV2, expectedManifestRoot) {
		return errors.New("canonical/v2: V3 result commit manifest mismatch")
	}
	if len(rc.OutputRoots) != len(g.Outputs) || len(rc.OutputRoots) == 0 {
		return errors.New("canonical/v2: V3 result commit output count mismatch")
	}
	outputs := make([]Hash, len(rc.OutputRoots))
	for i, r := range rc.OutputRoots {
		if len(r) != 32 {
			return errors.New("canonical/v2: V3 result commit output root malformed")
		}
		copy(outputs[i][:], r)
	}
	if !equalBytes(rc.FinalOutputRoot, hashToBytes(FinalOutputRoot(outputs))) {
		return errors.New("canonical/v2: V3 result commit final root mismatch")
	}
	if !VerifyGraphResultCommitV3Signature(rc) {
		return errors.New("canonical/v2: V3 result commit signature invalid")
	}
	return nil
}

// --- VerifiedGraphWorkReceiptV3 -------------------------------------------------

type VerifiedGraphWorkReceiptV3 struct {
	ProtocolVersion string `gemm:"protocol_version"`

	GraphID      []byte `gemm:"graph_id"`
	Spec         string `gemm:"spec"`
	ArithmeticID string `gemm:"arithmetic_id"`
	PolicyID     string `gemm:"policy_id"`

	TaskRef       []byte `gemm:"task_ref"`
	AssignmentRef []byte `gemm:"assignment_ref"`
	WorkerPubKey  []byte `gemm:"worker_pubkey"`

	NodeOutputManifestRootV2 []byte `gemm:"node_output_manifest_root_v2"`

	FinalOutputRoot []byte   `gemm:"final_output_root"`
	OutputRoots     [][]byte `gemm:"output_roots"`

	WorkVector WorkVector `gemm:"work_vector"`

	VerificationMode string `gemm:"verification_mode"`

	FinalizedEpoch uint64 `gemm:"finalized_epoch"`

	SettlementReference     []byte `gemm:"settlement_reference"`
	DAReference             []byte `gemm:"da_reference"`
	DisputeTranscriptDigest []byte `gemm:"dispute_transcript_digest"`
}

func BuildVerifiedGraphWorkReceiptV3(g *GraphDescriptorV2, taskRef, assignmentRef, workerPubKey []byte,
	outputs []Hash, work WorkVector, manifestRoot []byte, mode string, finalizedEpoch uint64,
	settlementRef, daRef, transcriptDigest []byte) (*VerifiedGraphWorkReceiptV3, error) {
	if len(manifestRoot) != 32 {
		return nil, errors.New("canonical/v2: V3 receipt needs a 32-byte manifest root")
	}
	if len(outputs) == 0 {
		return nil, errors.New("canonical/v2: V3 receipt needs outputs")
	}
	graphID, err := g.GraphIDV2()
	if err != nil {
		return nil, err
	}
	roots := make([][]byte, len(outputs))
	for i, r := range outputs {
		roots[i] = append([]byte(nil), r[:]...)
	}
	return &VerifiedGraphWorkReceiptV3{
		ProtocolVersion:          "3.0.0",
		GraphID:                  append([]byte(nil), graphID[:]...),
		Spec:                     g.Spec,
		ArithmeticID:             g.Arithmetic.ID,
		PolicyID:                 g.Arithmetic.PolicyID,
		TaskRef:                  append([]byte(nil), taskRef...),
		AssignmentRef:            append([]byte(nil), assignmentRef...),
		WorkerPubKey:             append([]byte(nil), workerPubKey...),
		NodeOutputManifestRootV2: append([]byte(nil), manifestRoot...),
		FinalOutputRoot:          hashToBytes(FinalOutputRoot(outputs)),
		OutputRoots:              roots,
		WorkVector:               work.Canonical(),
		VerificationMode:         mode,
		FinalizedEpoch:           finalizedEpoch,
		SettlementReference:      append([]byte(nil), settlementRef...),
		DAReference:              append([]byte(nil), daRef...),
		DisputeTranscriptDigest:  append([]byte(nil), transcriptDigest...),
	}, nil
}

// ReceiptIDV3 derives the canonical V3 receipt identifier (own domain).
func (r *VerifiedGraphWorkReceiptV3) ReceiptIDV3() (Hash, error) {
	return canonicalObjectHash(DomainReceiptV3, r)
}

// GraphWorkVectorV2 derives the canonical work vector of a V2 graph
// STATICALLY from the descriptor (the chain never trusts a worker-declared
// vector): same formulas the operators report at run time.
func GraphWorkVectorV2(g *GraphDescriptorV2) (WorkVector, error) {
	if err := g.ValidateV2(); err != nil {
		return nil, err
	}
	var work WorkVector
	for _, node := range g.Nodes {
		elems := int64(1)
		for _, d := range node.Output.Shape {
			elems *= d
		}
		switch node.OperatorID {
		case OpGEMMWideA13W10:
			a := g.inputDescOfV2(node.Inputs[0])
			b := g.inputDescOfV2(node.Inputs[1])
			m, k := a.Shape[0], a.Shape[1]
			n := b.Shape[1]
			if node.Params.Get("transpose_b", 0) == 1 {
				n = b.Shape[0]
			}
			work = work.Add("GEMM_A13W10_MAC", m*n*k)
		case OpRequantizeWideV1:
			work = work.Add("REQUANTIZE_WIDE_ELEMENT", elems)
		default:
			switch node.OperatorID {
			case OpAddFixedV1:
				work = work.Add("ADD_ELEMENT", elems)
			case OpMulFixedV1:
				work = work.Add("MUL_ELEMENT", elems)
			case OpRMSNormFixedV1:
				work = work.Add("RMSNORM_ELEMENT", elems)
				work = work.Add("RMSNORM_REDUCTION", elems)
			case OpRoPEFixedV1:
				work = work.Add("ROPE_PAIR", elems/2)
			case OpSiLUFixedV1:
				work = work.Add("SILU_ELEMENT", elems)
			case OpSoftmaxFixedV1:
				work = work.Add("SOFTMAX_ELEMENT", elems)
				work = work.Add("SOFTMAX_EXP", elems)
			default:
				return nil, fmt.Errorf("canonical/v2: unknown operator %s in work derivation", node.OperatorID)
			}
		}
	}
	return work.Canonical(), nil
}

func (g *GraphDescriptorV2) inputDescOfV2(ref TensorRef) TensorDescriptorV2 {
	if ref.Kind == 0 {
		return g.Inputs[ref.Index].Desc
	}
	return g.Nodes[ref.Index].Output
}
