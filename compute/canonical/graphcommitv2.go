package canonical

// GraphResultCommitV2 (Phase F.1): V1 plus a compact node-output manifest
// root, so a cheap watcher's Freivalds randomness can be bound to claims
// that were committed BEFORE the randomness existed. Additive and
// versioned: the V1 object, its signature format and every V1 test stay
// exactly as they are.

import (
	"crypto/ed25519"
	"errors"
)

// GraphResultCommitV2Version is the wire version of the V2 commitment.
const GraphResultCommitV2Version = "2.0.0"

// GraphResultCommitV2 is the canonical signed object.
type GraphResultCommitV2 struct {
	ProtocolVersion string `gemm:"protocol_version"`

	GraphID       []byte `gemm:"graph_id"`
	TaskRef       []byte `gemm:"task_ref"`
	AssignmentRef []byte `gemm:"assignment_ref"`
	WorkerPubKey  []byte `gemm:"worker_pubkey"`

	// NodeOutputManifestRoot commits every node's output TensorRoot
	// (docs/canonical-graph-v1.md, Phase F.1 manifest).
	NodeOutputManifestRoot []byte `gemm:"node_output_manifest_root"`

	FinalOutputRoot []byte   `gemm:"final_output_root"`
	OutputRoots     [][]byte `gemm:"output_roots"`

	CompletedEpoch uint64 `gemm:"completed_epoch"`

	Signature []byte `gemm:"signature"`
}

// NewGraphResultCommitV2 builds an unsigned V2 commitment. The manifest
// root is derived from the executed node roots; the signer must have run
// the graph (or received the exact node roots) before signing.
func NewGraphResultCommitV2(g *GraphDescriptor, taskRef, assignmentRef, workerPubKey []byte,
	exec *GraphExecution, completedEpoch uint64) (*GraphResultCommitV2, error) {
	if exec == nil {
		return nil, errors.New("canonical: V2 commitment needs the execution")
	}
	nodeRoots := make([]Hash, len(g.Nodes))
	for i := range g.Nodes {
		tens, ok := exec.Tensors[TensorRef{Kind: 1, Index: uint32(i)}]
		if !ok {
			return nil, errors.New("canonical: execution is missing a node output")
		}
		root, err := tens.TensorRoot()
		if err != nil {
			return nil, err
		}
		nodeRoots[i] = root
	}
	manifestRoot, _, err := BuildNodeOutputManifest(g, nodeRoots)
	if err != nil {
		return nil, err
	}
	base, err := NewGraphResultCommit(g, taskRef, assignmentRef, workerPubKey, exec.Outputs, completedEpoch)
	if err != nil {
		return nil, err
	}
	return &GraphResultCommitV2{
		ProtocolVersion:        GraphResultCommitV2Version,
		GraphID:                base.GraphID,
		TaskRef:                base.TaskRef,
		AssignmentRef:          base.AssignmentRef,
		WorkerPubKey:           base.WorkerPubKey,
		NodeOutputManifestRoot: append([]byte(nil), manifestRoot[:]...),
		FinalOutputRoot:        base.FinalOutputRoot,
		OutputRoots:            base.OutputRoots,
		CompletedEpoch:         base.CompletedEpoch,
	}, nil
}

func (rc *GraphResultCommitV2) signBytes() ([]byte, error) {
	unsigned := *rc
	unsigned.Signature = nil
	enc, err := EncodeCanonical(&unsigned)
	if err != nil {
		return nil, err
	}
	return append([]byte(DomainSig), enc...), nil
}

// SignGraphResultCommitV2 fills the worker signature.
func SignGraphResultCommitV2(rc *GraphResultCommitV2, key ed25519.PrivateKey) error {
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

// VerifyGraphResultCommitV2Signature checks the commitment signature.
func VerifyGraphResultCommitV2Signature(rc *GraphResultCommitV2) bool {
	if len(rc.WorkerPubKey) != 32 || len(rc.Signature) != ed25519.SignatureSize {
		return false
	}
	preimage, err := rc.signBytes()
	if err != nil {
		return false
	}
	return ed25519.Verify(rc.WorkerPubKey, preimage, rc.Signature)
}

// ValidateGraphResultCommitV2 re-derives every commitment the chain can
// check statically: version, graph identity, output-root set, final root
// and the signature. The manifest root is a signed 32-byte commitment;
// its correspondence to actually executed node outputs is checked by the
// watcher/DA from the bundle and refuted by the dispute path — the chain
// never recomputes the graph.
func ValidateGraphResultCommitV2(g *GraphDescriptor, rc *GraphResultCommitV2, expectedManifestRoot []byte) error {
	if rc.ProtocolVersion != GraphResultCommitV2Version {
		return errors.New("canonical: V2 result commit version mismatch")
	}
	graphID, err := g.GraphID()
	if err != nil {
		return err
	}
	if !equalBytes(rc.GraphID, graphID[:]) {
		return errors.New("canonical: V2 result commit does not bind this graph")
	}
	if len(rc.NodeOutputManifestRoot) != 32 {
		return errors.New("canonical: V2 result commit manifest root malformed")
	}
	if expectedManifestRoot != nil && !equalBytes(rc.NodeOutputManifestRoot, expectedManifestRoot) {
		return errors.New("canonical: V2 result commit manifest mismatch")
	}
	if len(rc.OutputRoots) != len(g.Outputs) || len(rc.OutputRoots) == 0 {
		return errors.New("canonical: V2 result commit output count mismatch")
	}
	outputs := make([]Hash, len(rc.OutputRoots))
	for i, raw := range rc.OutputRoots {
		if len(raw) != 32 {
			return errors.New("canonical: V2 result commit output root malformed")
		}
		copy(outputs[i][:], raw)
	}
	finalRoot := FinalOutputRoot(outputs)
	if !equalBytes(rc.FinalOutputRoot, finalRoot[:]) {
		return errors.New("canonical: V2 result commit final output root mismatch")
	}
	// The V2 signature covers the V2 preimage (manifest root included);
	// the V1 signature format is never reused over V2 bytes.
	if !VerifyGraphResultCommitV2Signature(rc) {
		return errors.New("canonical: V2 result commit signature does not verify")
	}
	return nil
}

// NodeOutputRootsFromExecution extracts the node output tensor roots of an
// execution in node order (tooling helper for bundles and tests).
func NodeOutputRootsFromExecution(g *GraphDescriptor, exec *GraphExecution) ([]Hash, error) {
	out := make([]Hash, len(g.Nodes))
	for i := range g.Nodes {
		tens, ok := exec.Tensors[TensorRef{Kind: 1, Index: uint32(i)}]
		if !ok {
			return nil, errors.New("canonical: execution is missing a node output")
		}
		root, err := tens.TensorRoot()
		if err != nil {
			return nil, err
		}
		out[i] = root
	}
	return out, nil
}
