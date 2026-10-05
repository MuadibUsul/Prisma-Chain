package gemmv1

// Merkle tree rules: determinism across sizes, odd-node behavior, position
// binding and proof rejection.

import (
	"testing"
)

func leafSet(count int) []Hash {
	leaves := make([]Hash, count)
	for i := range leaves {
		leaves[i] = hashBytes([]byte{byte(i), byte(i >> 8)})
	}
	return leaves
}

func TestMerkleOddSizesDeterministic(t *testing.T) {
	for _, count := range []int{1, 2, 3, 5, 8, 9, 17} {
		leaves := leafSet(count)
		root1, err := MerkleRoot(leaves)
		if err != nil {
			t.Fatal(err)
		}
		root2, err := MerkleRoot(leaves)
		if err != nil {
			t.Fatal(err)
		}
		if root1 != root2 {
			t.Fatalf("root not deterministic for %d leaves", count)
		}
	}
	single := leafSet(1)
	root, err := MerkleRoot(single)
	if err != nil {
		t.Fatal(err)
	}
	if root != single[0] {
		t.Fatal("single-leaf root must be the leaf")
	}
	if _, err := MerkleRoot(nil); err == nil {
		t.Fatal("empty tree must be rejected")
	}
}

func TestMerkleProofsAllPositions(t *testing.T) {
	for _, count := range []int{1, 2, 3, 5, 8, 9, 17} {
		leaves := leafSet(count)
		root, err := MerkleRoot(leaves)
		if err != nil {
			t.Fatal(err)
		}
		levels, err := buildLevels(leaves)
		if err != nil {
			t.Fatal(err)
		}
		for idx := 0; idx < count; idx++ {
			proof, err := ProveLeaf(levels, uint32(idx))
			if err != nil {
				t.Fatal(err)
			}
			if !VerifyLeafInclusion(root, leaves[idx], proof) {
				t.Fatalf("proof rejected: %d leaves, index %d", count, idx)
			}
			// Position binding: proof must fail for a different leaf.
			other := leaves[(idx+1)%count]
			if other != leaves[idx] && VerifyLeafInclusion(root, other, proof) {
				t.Fatalf("foreign leaf accepted: %d leaves, index %d", count, idx)
			}
		}
	}
}

func TestMerkleProofTamperRejected(t *testing.T) {
	leaves := leafSet(5)
	root, _ := MerkleRoot(leaves)
	levels, _ := buildLevels(leaves)
	proof, err := ProveLeaf(levels, 3)
	if err != nil {
		t.Fatal(err)
	}
	proof.Siblings[0][0] ^= 0xff
	if VerifyLeafInclusion(root, leaves[3], proof) {
		t.Fatal("tampered sibling accepted")
	}
	proof, _ = ProveLeaf(levels, 3)
	proof.Index = 4 // wrong position
	if VerifyLeafInclusion(root, leaves[3], proof) {
		t.Fatal("proof accepted under wrong index")
	}
	proof, _ = ProveLeaf(levels, 3)
	proof.Count = 4 // wrong count
	if VerifyLeafInclusion(root, leaves[3], proof) {
		t.Fatal("proof accepted under wrong count")
	}
	proof, _ = ProveLeaf(levels, 3)
	proof.Siblings = proof.Siblings[:len(proof.Siblings)-1]
	if VerifyLeafInclusion(root, leaves[3], proof) {
		t.Fatal("truncated proof accepted")
	}
}
