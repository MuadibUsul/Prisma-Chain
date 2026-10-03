package gemmv1

// GEMM v1 Merkle utilities.
//
// Leaves are domain-separated SHA-256 digests that bind matrix identity,
// tile coordinates and canonical tile bytes. Internal nodes are
// SHA256(left || right) over two fixed 32-byte digests.
//
// Canonical odd behavior: at every level with an odd node count greater
// than one, the last node is paired with a copy of itself before hashing.
// A single-leaf tree has root == leaf. The rule is deterministic and
// platform independent; trees built from the same leaves always produce
// the same root (tested).

import (
	"errors"
)

var errNoLeaves = errors.New("gemmv1: Merkle tree needs at least one leaf")

// LeafInputTile hashes one input tile of A or B. matrixID distinguishes the
// two matrices; coordinates are fixed-width big-endian uint32.
func LeafInputTile(matrixID byte, tileRow, tileCol uint32, tileBytes []byte) Hash {
	buf := make([]byte, 0, len(DomainInputTile)+1+8+len(tileBytes))
	buf = append(buf, DomainInputTile...)
	buf = append(buf, matrixID)
	buf = appendUint32BE(buf, tileRow)
	buf = appendUint32BE(buf, tileCol)
	buf = append(buf, tileBytes...)
	return hashBytes(buf)
}

// LeafOutputTile hashes one int32 output tile bound to task and assignment.
func LeafOutputTile(taskID, assignmentID []byte, tileI, tileJ uint32, tileBytes []byte) Hash {
	buf := make([]byte, 0, len(DomainOutputTile)+64+8+len(tileBytes))
	buf = append(buf, DomainOutputTile...)
	buf = append(buf, taskID...)
	buf = append(buf, assignmentID...)
	buf = appendUint32BE(buf, tileI)
	buf = appendUint32BE(buf, tileJ)
	buf = append(buf, tileBytes...)
	return hashBytes(buf)
}

// LeafTraceState hashes one tile trace state, binding task, assignment,
// disputed tile coordinates, K-step index and the canonical state bytes.
func LeafTraceState(taskID, assignmentID []byte, tileI, tileJ uint32, step uint32, stateBytes []byte) Hash {
	buf := make([]byte, 0, len(DomainTraceState)+64+12+len(stateBytes))
	buf = append(buf, DomainTraceState...)
	buf = append(buf, taskID...)
	buf = append(buf, assignmentID...)
	buf = appendUint32BE(buf, tileI)
	buf = appendUint32BE(buf, tileJ)
	buf = appendUint32BE(buf, step)
	buf = append(buf, stateBytes...)
	return hashBytes(buf)
}

// buildLevels computes all tree levels bottom-up, levels[0] = leaves.
func buildLevels(leaves []Hash) ([][]Hash, error) {
	if len(leaves) == 0 {
		return nil, errNoLeaves
	}
	level := make([]Hash, len(leaves))
	copy(level, leaves)
	levels := [][]Hash{level}
	for len(level) > 1 {
		next := make([]Hash, 0, (len(level)+1)/2)
		for i := 0; i < len(level); i += 2 {
			left := level[i]
			right := left
			if i+1 < len(level) {
				right = level[i+1]
			}
			next = append(next, hashNode(left, right))
		}
		levels = append(levels, next)
		level = next
	}
	return levels, nil
}

func hashNode(left, right Hash) Hash {
	return hashBytes(left[:], right[:])
}

// MerkleRoot returns the root of the tree over leaves.
func MerkleRoot(leaves []Hash) (Hash, error) {
	levels, err := buildLevels(leaves)
	if err != nil {
		return Hash{}, err
	}
	return levels[len(levels)-1][0], nil
}

// MerkleProof is an inclusion proof for one leaf position. Index and Count
// bind the position; direction is derived from Index at each level.
type MerkleProof = OutputTileProof

// ProveLeaf builds an inclusion proof for the leaf at index against the
// tree levels.
func ProveLeaf(levels [][]Hash, index uint32) (MerkleProof, error) {
	count := uint32(len(levels[0]))
	if index >= count {
		return MerkleProof{}, errors.New("gemmv1: leaf index out of range")
	}
	proof := MerkleProof{Index: index, Count: count}
	for lvl := 0; lvl < len(levels)-1; lvl++ {
		nodes := levels[lvl]
		idx := index
		sib := idx
		if idx%2 == 0 {
			sib = idx + 1
			if sib >= uint32(len(nodes)) {
				sib = idx // odd last node pairs with itself
			}
		} else {
			sib = idx - 1
		}
		proof.Siblings = append(proof.Siblings, append([]byte(nil), nodes[sib][:]...))
		index /= 2
	}
	return proof, nil
}

// VerifyLeafInclusion verifies an inclusion proof of leaf at the proof's
// bound position against root.
func VerifyLeafInclusion(root Hash, leaf Hash, p MerkleProof) bool {
	if p.Count == 0 || p.Index >= p.Count || len(p.Siblings) != depthFor(p.Count) {
		return false
	}
	h := leaf
	idx, cnt := p.Index, p.Count
	for _, sib := range p.Siblings {
		var s Hash
		copy(s[:], sib)
		if idx%2 == 0 {
			// Right sibling may be the duplicated node itself; the honest
			// path supplies h, any forgery fails the root check.
			right := s
			if idx+1 >= cnt {
				right = h
			}
			h = hashNode(h, right)
		} else {
			h = hashNode(s, h)
		}
		idx /= 2
		cnt = (cnt + 1) / 2
	}
	return h == root
}

func depthFor(count uint32) int {
	depth := 0
	for count > 1 {
		count = (count + 1) / 2
		depth++
	}
	return depth
}
