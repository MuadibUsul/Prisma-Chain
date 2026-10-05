package canonical

// Node-output manifest (Phase F.1): a compact commitment to EVERY node's
// output tensor root, locked BEFORE any verification randomness exists.
//
// This is deliberately NOT an execution trace: leaves bind only
// (graph_id, node_id, operator_id, descriptor hash, node output tensor
// root). No tensor bytes, no micro states, no GEMM partial sums. The
// on-demand dispute trail is still generated only after a challenge.
//
// GraphResultCommitV1 is untouched; V2 carries this root additively.

import (
	"errors"
)

// DomainNodeOutput separates manifest leaves.
const DomainNodeOutput = "PRISMA_NODE_OUTPUT_V1\x00"

// NodeOutputManifestVersion names the manifest format.
const NodeOutputManifestVersion = "1.0.0"

// NodeOutputLeaf hashes one node's output commitment.
func NodeOutputLeaf(graphID Hash, node GraphNode) (Hash, error) {
	descBytes, err := EncodeCanonical(node.Output)
	if err != nil {
		return Hash{}, err
	}
	return hashBytes([]byte(DomainNodeOutput), graphID[:], appendUint32BE(nil, node.NodeID),
		[]byte(node.OperatorID), descBytes), nil
}

// BuildNodeOutputManifest commits every node output root:
// leaf_i = H(domain || graph_id || node_id || operator_id || CBOR(desc))
// combined with the node's output TensorRoot, and the manifest root is the
// canonical Merkle root over the leaves in node order. nodeRoots must be
// the executed output tensor roots of every node, in node order.
func BuildNodeOutputManifest(g *GraphDescriptor, nodeRoots []Hash) (Hash, []Hash, error) {
	if err := g.Validate(); err != nil {
		return Hash{}, nil, err
	}
	if len(nodeRoots) != len(g.Nodes) {
		return Hash{}, nil, errors.New("canonical: manifest needs one root per node")
	}
	graphID, err := g.GraphID()
	if err != nil {
		return Hash{}, nil, err
	}
	leaves := make([]Hash, len(g.Nodes))
	for i, node := range g.Nodes {
		leaf, err := NodeOutputLeaf(graphID, node)
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

// VerifyNodeOutputManifestLeaf proves one node's committed output root
// against the manifest root.
func VerifyNodeOutputManifestLeaf(root Hash, graphID Hash, node GraphNode, tensorRoot Hash, proof MerkleProof) bool {
	leaf, err := NodeOutputLeaf(graphID, node)
	if err != nil {
		return false
	}
	combined := hashBytes(leaf[:], tensorRoot[:])
	return VerifyLeafInclusion(root, combined, proof)
}
