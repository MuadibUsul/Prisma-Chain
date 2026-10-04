package canonical

// CANONICAL_GRAPH_V1: a static, versioned DAG of canonical operators.
//
// There are no loops, branches, host callbacks, randomness or I/O: the
// graph is fully fixed before execution. Normal-path workers commit only
// the final output tensor roots (optimistic output-first); the
// per-node state trail exists only inside a dispute, generated on demand
// by both parties.

import (
	"errors"
	"fmt"
	"sort"
)

// GraphProtocolVersion is the frozen version string of graph objects.
const GraphProtocolVersion = "1.0.0"

// TensorRef addresses one tensor inside a graph execution: either one of
// the declared inputs/constants or the output of a node.
type TensorRef struct {
	Kind  uint8  `gemm:"kind"`  // 0 = graph input/constant, 1 = node output
	Index uint32 `gemm:"index"` // input index or node id
}

// GraphInput is a committed external tensor (task input or model weight
// constant). Only its descriptor and root are part of the graph; the bytes
// live in the DA layer.
type GraphInput struct {
	Name string           `gemm:"name"`
	Desc TensorDescriptor `gemm:"desc"`
	Root []byte           `gemm:"root"`
}

// GraphNode is one canonical operator application.
type GraphNode struct {
	NodeID     uint32           `gemm:"node_id"`
	OperatorID string           `gemm:"operator_id"`
	Version    string           `gemm:"operator_version"`
	Inputs     []TensorRef      `gemm:"inputs"`
	Output     TensorDescriptor `gemm:"output"`
	Params     ParamList        `gemm:"params"`
}

// GraphDescriptor is the canonical, hashable graph.
type GraphDescriptor struct {
	ProtocolVersion string       `gemm:"protocol_version"`
	Spec            string       `gemm:"spec"` // e.g. TRANSFORMER_BLOCK_V1
	Inputs          []GraphInput `gemm:"inputs"`
	Nodes           []GraphNode  `gemm:"nodes"`
	Outputs         []TensorRef  `gemm:"outputs"`
}

// GraphID = SHA256(domain || CanonicalCBOR(GraphDescriptor)); any change
// to an operator, weight root, scale, shape or constant changes it.
func (g *GraphDescriptor) GraphID() (Hash, error) {
	return canonicalObjectHash(DomainGraph, g)
}

// Validate checks the static well-formedness: forward references only,
// unknown operators rejected, descriptors computed by the operators
// themselves (never trusted from the raw graph), parameters validated.
func (g *GraphDescriptor) Validate() error {
	if g.ProtocolVersion != GraphProtocolVersion {
		return errors.New("canonical: graph protocol_version mismatch")
	}
	seen := map[uint32]bool{}
	for i, node := range g.Nodes {
		if node.NodeID != uint32(i) {
			return fmt.Errorf("canonical: node ids must be dense and ordered, got %d at position %d", node.NodeID, i)
		}
		if seen[node.NodeID] {
			return fmt.Errorf("canonical: duplicate node id %d", node.NodeID)
		}
		seen[node.NodeID] = true
		op, err := Lookup(node.OperatorID)
		if err != nil {
			return err
		}
		if node.Version != op.Version() {
			return fmt.Errorf("canonical: node %d version mismatch for %s", node.NodeID, node.OperatorID)
		}
		descs := make([]TensorDescriptor, 0, len(node.Inputs))
		for _, ref := range node.Inputs {
			desc, err := g.descFor(ref, node.NodeID)
			if err != nil {
				return err
			}
			descs = append(descs, desc)
		}
		want, err := op.OutputSpec(descs, node.Params)
		if err != nil {
			return fmt.Errorf("canonical: node %d: %w", node.NodeID, err)
		}
		if !sameDesc(want, node.Output) {
			return fmt.Errorf("canonical: node %d declared output does not match the operator spec", node.NodeID)
		}
	}
	for _, out := range g.Outputs {
		if _, err := g.descFor(out, uint32(len(g.Nodes))); err != nil {
			return err
		}
	}
	return nil
}

func (g *GraphDescriptor) descFor(ref TensorRef, beforeNode uint32) (TensorDescriptor, error) {
	switch ref.Kind {
	case 0:
		if int(ref.Index) >= len(g.Inputs) {
			return TensorDescriptor{}, errors.New("canonical: input ref out of range")
		}
		return g.Inputs[ref.Index].Desc, nil
	case 1:
		if ref.Index >= beforeNode {
			return TensorDescriptor{}, errors.New("canonical: node ref must be a forward reference")
		}
		return g.Nodes[ref.Index].Output, nil
	default:
		return TensorDescriptor{}, errors.New("canonical: unknown tensor ref kind")
	}
}

// --- execution -------------------------------------------------------------

// GraphExecution is the product of one reference execution: every tensor
// by ref, the per-node state trail and the final output roots.
type GraphExecution struct {
	Tensors    map[TensorRef]Tensor
	Trail      []Hash // len = nodes+1; Trail[k] is the state after node k-1
	Outputs    []Hash // final output tensor roots, in graph order
	WorkVector WorkVector
}

// stateRootFor computes GraphStateRoot = MerkleRoot over sorted live
// (tensorID, TensorRoot) pairs. The tensor id encodes kind||index so the
// state covers every live tensor, not just the last one (transformer
// blocks branch).
func stateRootFor(tensors map[TensorRef]Tensor) (Hash, error) {
	leaves := make([]Hash, 0, len(tensors))
	ids := make([]uint64, 0, len(tensors))
	byID := map[uint64]TensorRef{}
	for ref := range tensors {
		id := uint64(ref.Kind)<<32 | uint64(ref.Index)
		ids = append(ids, id)
		byID[id] = ref
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	for _, id := range ids {
		t := tensors[byID[id]]
		root, err := t.TensorRoot()
		if err != nil {
			return Hash{}, err
		}
		leaves = append(leaves, hashBytes([]byte(DomainGraphState), appendUint32BE(nil, uint32(id>>32)), appendUint32BE(nil, uint32(id)), root[:]))
	}
	return MerkleRootOf(leaves)
}

// ExecuteGraph runs the graph with the reference operators. constants
// resolves operator constant tables (RoPE) by graph input index.
func ExecuteGraph(g *GraphDescriptor, inputs map[uint32]Tensor, ropeTables map[uint32]*RopeConstants) (*GraphExecution, error) {
	if err := g.Validate(); err != nil {
		return nil, err
	}
	tensors := map[TensorRef]Tensor{}
	var work WorkVector
	for idx, in := range g.Inputs {
		t, ok := inputs[uint32(idx)]
		if !ok {
			return nil, fmt.Errorf("canonical: missing graph input %d (%s)", idx, in.Name)
		}
		if !sameDesc(t.Desc, in.Desc) {
			return nil, fmt.Errorf("canonical: input %d descriptor mismatch", idx)
		}
		root, err := t.TensorRoot()
		if err != nil {
			return nil, err
		}
		if !equalBytes(root[:], in.Root) {
			return nil, fmt.Errorf("canonical: input %d does not match its committed root", idx)
		}
		tensors[TensorRef{Kind: 0, Index: uint32(idx)}] = t
	}
	exec := &GraphExecution{Tensors: tensors}
	initial, err := stateRootFor(tensors)
	if err != nil {
		return nil, err
	}
	exec.Trail = append(exec.Trail, initial)
	for _, node := range g.Nodes {
		op, _ := Lookup(node.OperatorID)
		ins := make([]Tensor, 0, len(node.Inputs))
		for _, ref := range node.Inputs {
			ins = append(ins, tensors[ref])
		}
		var out *Tensor
		if node.OperatorID == OpRoPEFixedV1 {
			table, ok := ropeTables[node.Inputs[len(node.Inputs)-1].Index]
			if !ok {
				return nil, errors.New("canonical: missing RoPE constant table")
			}
			out, err = ExecuteRope(&ins[0], table, node.Params)
		} else {
			out, err = op.Execute(ins, node.Params)
		}
		if err != nil {
			return nil, fmt.Errorf("canonical: node %d: %w", node.NodeID, err)
		}
		tensors[TensorRef{Kind: 1, Index: node.NodeID}] = *out
		for _, counter := range op.Work(ins, node.Params) {
			work = work.Add(counter.Key, counter.Value)
		}
		state, err := stateRootFor(tensors)
		if err != nil {
			return nil, err
		}
		exec.Trail = append(exec.Trail, state)
	}
	exec.WorkVector = work.Canonical()
	for _, ref := range g.Outputs {
		t := tensors[ref]
		root, err := t.TensorRoot()
		if err != nil {
			return nil, err
		}
		exec.Outputs = append(exec.Outputs, root)
	}
	return exec, nil
}

// FinalOutputRoot commits the ordered output tensor roots:
// SHA256(domain || roots...).
func FinalOutputRoot(outputs []Hash) Hash {
	parts := make([][]byte, 0, len(outputs)+1)
	parts = append(parts, []byte(DomainVWR))
	for _, h := range outputs {
		parts = append(parts, h[:])
	}
	return hashBytes(parts...)
}

// MerkleRootOf exposes the shared tree root for callers outside tensor.go.
func MerkleRootOf(leaves []Hash) (Hash, error) {
	levels, err := buildLevels(leaves)
	if err != nil {
		return Hash{}, err
	}
	return levels[len(levels)-1][0], nil
}

func equalBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
