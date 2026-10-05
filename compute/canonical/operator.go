package canonical

// Static, compile-time operator registry for CANONICAL_GRAPH_V1.
//
// There is no reflection, no dynamic plugin loading and no interface{} in
// the consensus path: the set of legal operators is fixed at build time
// and every one of them exposes exactly the same pure-integer contract.

import (
	"fmt"
	"sort"
)

// Operator IDs (frozen strings; each has its own version space).
const (
	OpGEMMInt8V1     = "GEMM_INT8_V1"
	OpAddFixedV1     = "ADD_FIXED_V1"
	OpMulFixedV1     = "MUL_FIXED_V1"
	OpRequantizeV1   = "REQUANTIZE_V1"
	OpRMSNormFixedV1 = "RMSNORM_FIXED_V1"
	OpRoPEFixedV1    = "ROPE_FIXED_V1"
	OpSiLUFixedV1    = "SILU_FIXED_V1"
	OpSoftmaxFixedV1 = "SOFTMAX_FIXED_V1"
)

// Operator versions. GEMM keeps its frozen protocol version untouched.
const (
	VersionGEMM     = "0.1.1"
	VersionFxFusion = "1.0.0"
)

// ParamList is the canonical parameter form of a node: sorted integer
// pairs only (any real value is scaled into the agreed fixed-point unit by
// the graph compiler and committed there).
type ParamList []Pair

// Get returns the parameter or the default.
func (p ParamList) Get(key string, def int64) int64 {
	for _, pair := range p {
		if pair.Key == key {
			return pair.Value
		}
	}
	return def
}

// Canonical returns a sorted copy.
func (p ParamList) Canonical() ParamList {
	out := append(ParamList(nil), p...)
	SortPairs(out)
	return out
}

// WorkVectorV1: per-unit work counters. There is deliberately no universal
// CWU: fusion pricing stays an open economic question, so a graph task
// reports the raw vector and settlement uses the agreed fee.
type WorkVector []Pair

// Add accumulates counters by key.
func (w WorkVector) Add(key string, n int64) WorkVector {
	for i := range w {
		if w[i].Key == key {
			w[i].Value += n
			return w
		}
	}
	return append(w, Pair{Key: key, Value: n})
}

// Get returns a counter or the default.
func (w WorkVector) Get(key string, def int64) int64 {
	for _, pair := range w {
		if pair.Key == key {
			return pair.Value
		}
	}
	return def
}

// Canonical sorts the counters.
func (w WorkVector) Canonical() WorkVector {
	out := append(WorkVector(nil), w...)
	SortPairs(out)
	return out
}

// Operator is the pure contract every canonical operator implements. All
// methods are deterministic, integer-only and stateless.
type Operator interface {
	ID() string
	Version() string
	// ValidateParams rejects malformed node parameters.
	ValidateParams(p ParamList) error
	// OutputSpec derives the output descriptor from input descriptors.
	OutputSpec(in []TensorDescriptor, p ParamList) (TensorDescriptor, error)
	// Work reports the canonical work counters of one execution.
	Work(in []Tensor, p ParamList) WorkVector
	// Execute runs the canonical reference computation.
	Execute(in []Tensor, p ParamList) (*Tensor, error)
	// ArbiterBound is the maximum number of input elements the chain
	// arbiter may touch to adjudicate one disputed output chunk of this
	// operator (0 means element-local with no wider dependency).
	ArbiterBound(in []TensorDescriptor, p ParamList) int
}

var registry = map[string]Operator{}

// Register installs an operator; duplicate IDs are a programming error.
func Register(op Operator) {
	if _, exists := registry[op.ID()]; exists {
		panic("canonical: duplicate operator registration " + op.ID())
	}
	registry[op.ID()] = op
}

// Lookup returns the registered operator for an ID.
func Lookup(id string) (Operator, error) {
	op, ok := registry[id]
	if !ok {
		return nil, fmt.Errorf("canonical: operator %q is not supported", id)
	}
	return op, nil
}

// OperatorIDs lists the compile-time supported operators.
func OperatorIDs() []string {
	ids := make([]string, 0, len(registry))
	for id := range registry {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func init() {
	Register(gemmInt8V1{})
	Register(addFixed{})
	Register(mulFixed{})
	Register(requantize{})
	Register(rmsNormFixed{})
	Register(ropeFixed{})
	Register(siluFixed{})
	Register(softmaxFixed{})
}

// helpers shared by operators

func sameDesc(a, b TensorDescriptor) bool {
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

func requireQ12(desc TensorDescriptor) error {
	if Dtype(desc.Dtype) != DtypeQ12_20 {
		return fmt.Errorf("canonical: operator requires q12.20 tensors, got dtype %d", desc.Dtype)
	}
	return desc.Validate()
}
