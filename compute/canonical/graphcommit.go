package canonical

// GraphResultCommit: the worker's signed commitment for one graph task.
// Output-first, exactly like the frozen GEMM ResultCommit: it commits only
// the final output roots (plus the per-output roots and the descriptor-
// derived work counters); the state trail never leaves the dispute.

import (
	"crypto/ed25519"
	"errors"
)

// GraphResultCommitVersion is the wire version of the commitment.
const GraphResultCommitVersion = "1.0.0"

// GraphResultCommit is the canonical signed object.
type GraphResultCommit struct {
	ProtocolVersion string `gemm:"protocol_version"`

	GraphID       []byte `gemm:"graph_id"`
	TaskRef       []byte `gemm:"task_ref"`
	AssignmentRef []byte `gemm:"assignment_ref"`
	WorkerPubKey  []byte `gemm:"worker_pubkey"`

	FinalOutputRoot []byte   `gemm:"final_output_root"`
	OutputRoots     [][]byte `gemm:"output_roots"`

	CompletedEpoch uint64 `gemm:"completed_epoch"`

	Signature []byte `gemm:"signature"`
}

// NewGraphResultCommit builds an unsigned commitment over committed
// output roots.
func NewGraphResultCommit(g *GraphDescriptor, taskRef, assignmentRef, workerPubKey []byte, outputs []Hash, completedEpoch uint64) (*GraphResultCommit, error) {
	if err := g.Validate(); err != nil {
		return nil, err
	}
	if len(workerPubKey) != 32 {
		return nil, errors.New("canonical: result commit needs a 32-byte worker key")
	}
	if len(taskRef) == 0 || len(taskRef) > 64 || len(assignmentRef) == 0 || len(assignmentRef) > 64 {
		return nil, errors.New("canonical: result commit references are malformed")
	}
	if len(outputs) == 0 {
		return nil, errors.New("canonical: result commit needs at least one output root")
	}
	graphID, err := g.GraphID()
	if err != nil {
		return nil, err
	}
	roots := make([][]byte, len(outputs))
	for i, h := range outputs {
		roots[i] = append([]byte(nil), h[:]...)
	}
	finalRoot := FinalOutputRoot(outputs)
	return &GraphResultCommit{
		ProtocolVersion: GraphResultCommitVersion,
		GraphID:         append([]byte(nil), graphID[:]...),
		TaskRef:         append([]byte(nil), taskRef...),
		AssignmentRef:   append([]byte(nil), assignmentRef...),
		WorkerPubKey:    append([]byte(nil), workerPubKey...),
		FinalOutputRoot: append([]byte(nil), finalRoot[:]...),
		OutputRoots:     roots,
		CompletedEpoch:  completedEpoch,
	}, nil
}

// signBytes is the signature preimage: domain || CBOR(commit with an empty
// signature). Mirrors the frozen gemmv1 convention.
func (rc *GraphResultCommit) signBytes() ([]byte, error) {
	unsigned := *rc
	unsigned.Signature = nil
	enc, err := EncodeCanonical(&unsigned)
	if err != nil {
		return nil, err
	}
	return append([]byte(DomainSig), enc...), nil
}

// SignGraphResultCommit fills the worker signature.
func SignGraphResultCommit(rc *GraphResultCommit, key ed25519.PrivateKey) error {
	if len(rc.WorkerPubKey) != 32 {
		return errors.New("canonical: result commit needs a 32-byte worker key")
	}
	preimage, err := rc.signBytes()
	if err != nil {
		return err
	}
	rc.Signature = ed25519.Sign(key, preimage)
	return nil
}

// VerifyGraphResultCommitSignature checks the commitment against its
// worker key.
func VerifyGraphResultCommitSignature(rc *GraphResultCommit) bool {
	if len(rc.WorkerPubKey) != 32 || len(rc.Signature) != ed25519.SignatureSize {
		return false
	}
	preimage, err := rc.signBytes()
	if err != nil {
		return false
	}
	return ed25519.Verify(rc.WorkerPubKey, preimage, rc.Signature)
}

// ValidateGraphResultCommit re-derives every commitment: version, graph
// identity, output-root set and the final output root. Nothing in the
// commitment is trusted.
func ValidateGraphResultCommit(g *GraphDescriptor, rc *GraphResultCommit) error {
	if rc.ProtocolVersion != GraphResultCommitVersion {
		return errors.New("canonical: result commit version mismatch")
	}
	graphID, err := g.GraphID()
	if err != nil {
		return err
	}
	if !equalBytes(rc.GraphID, graphID[:]) {
		return errors.New("canonical: result commit does not bind this graph")
	}
	if len(rc.OutputRoots) != len(g.Outputs) || len(rc.OutputRoots) == 0 {
		return errors.New("canonical: result commit output count mismatch")
	}
	outputs := make([]Hash, len(rc.OutputRoots))
	for i, raw := range rc.OutputRoots {
		if len(raw) != 32 {
			return errors.New("canonical: result commit output root malformed")
		}
		copy(outputs[i][:], raw)
	}
	finalRoot := FinalOutputRoot(outputs)
	if !equalBytes(rc.FinalOutputRoot, finalRoot[:]) {
		return errors.New("canonical: result commit final output root mismatch")
	}
	if !VerifyGraphResultCommitSignature(rc) {
		return errors.New("canonical: result commit signature does not verify")
	}
	return nil
}

// GraphWorkVector derives the canonical per-unit work counters of a graph
// purely from its static descriptor. Settlement prices this derived vector,
// so a worker can never bias its own counters.
func GraphWorkVector(g *GraphDescriptor) (WorkVector, error) {
	if err := g.Validate(); err != nil {
		return nil, err
	}
	var work WorkVector
	descOf := func(ref TensorRef) (TensorDescriptor, error) {
		switch ref.Kind {
		case 0:
			if int(ref.Index) >= len(g.Inputs) {
				return TensorDescriptor{}, errors.New("canonical: input ref out of range")
			}
			return g.Inputs[ref.Index].Desc, nil
		case 1:
			if int(ref.Index) >= len(g.Nodes) {
				return TensorDescriptor{}, errors.New("canonical: node ref out of range")
			}
			return g.Nodes[ref.Index].Output, nil
		default:
			return TensorDescriptor{}, errors.New("canonical: unknown tensor ref kind")
		}
	}
	for _, node := range g.Nodes {
		elems, err := node.Output.Elems()
		if err != nil {
			return nil, err
		}
		switch node.OperatorID {
		case OpGEMMInt8V1:
			aDesc, err := descOf(node.Inputs[0])
			if err != nil {
				return nil, err
			}
			bDesc, err := descOf(node.Inputs[1])
			if err != nil {
				return nil, err
			}
			m, n, k, _, err := GEMMNodeDims(aDesc, bDesc, node.Params)
			if err != nil {
				return nil, err
			}
			work = work.Add("GEMM_MAC", int64(m*n*k))
		case OpAddFixedV1:
			work = work.Add("ADD_ELEMENT", elems)
		case OpMulFixedV1:
			work = work.Add("MUL_ELEMENT", elems)
		case OpRequantizeV1:
			work = work.Add("REQUANTIZE_ELEMENT", elems)
		case OpRMSNormFixedV1:
			work = work.Add("RMSNORM_ELEMENT", elems).Add("RMSNORM_REDUCTION", elems)
		case OpRoPEFixedV1:
			work = work.Add("ROPE_PAIR", elems/2)
		case OpSiLUFixedV1:
			work = work.Add("SILU_ELEMENT", elems)
		case OpSoftmaxFixedV1:
			work = work.Add("SOFTMAX_ELEMENT", elems).Add("SOFTMAX_EXP", elems)
		default:
			return nil, errors.New("canonical: no work model for " + node.OperatorID)
		}
	}
	return work.Canonical(), nil
}
