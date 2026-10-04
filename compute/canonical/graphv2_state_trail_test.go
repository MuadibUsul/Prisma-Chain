package canonical

// A1-02/A1-03 vector replay: Go must reproduce the Python-generated V2
// state and trail vectors bit-for-bit, and reject wrong ids/indexes and
// V1-version material.

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestGraphStateRootV2Vectors(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "canonical_graph_v2_state_vectors.json"))
	if err != nil {
		t.Skipf("vectors not present: %v", err)
	}
	var doc struct {
		Cases []struct {
			Name   string `json:"name"`
			Live   []struct {
				Kind  uint8  `json:"kind"`
				Index uint32 `json:"index"`
				Root  string `json:"root"`
			} `json:"live"`
			StateRootV2       string `json:"state_root_v2"`
			StateRootV1SameIn string `json:"state_root_v1_same_inputs"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if len(doc.Cases) == 0 {
		t.Fatal("no state vector cases")
	}
	for _, c := range doc.Cases {
		live := map[TensorRef]Hash{}
		for _, l := range c.Live {
			var h Hash
			b, err := hex.DecodeString(l.Root)
			if err != nil || len(b) != 32 {
				t.Fatalf("%s: bad root", c.Name)
			}
			copy(h[:], b)
			live[TensorRef{Kind: l.Kind, Index: l.Index}] = h
		}
		got, err := GraphStateRootV2FromRoots(live)
		if err != nil {
			t.Fatal(err)
		}
		if hex.EncodeToString(got[:]) != c.StateRootV2 {
			t.Fatalf("%s: V2 state root mismatch", c.Name)
		}
		if c.StateRootV1SameIn != "" && hex.EncodeToString(got[:]) == c.StateRootV1SameIn {
			t.Fatalf("%s: V2 root equals the V1 root for the same inputs", c.Name)
		}
	}
}

func TestTrailV2Vectors(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "canonical_graph_v2_trail_vectors.json"))
	if err != nil {
		t.Skipf("vectors not present: %v", err)
	}
	var doc struct {
		GraphID     string   `json:"graph_id"`
		Trail       []string `json:"trail"`
		TrailRootV2 string   `json:"trail_root_v2"`
		Proofs      []struct {
			Step      uint32   `json:"step"`
			StateRoot string   `json:"state_root"`
			Siblings  []string `json:"siblings"`
		} `json:"proofs"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	gidBytes, _ := hex.DecodeString(doc.GraphID)
	var gid Hash
	copy(gid[:], gidBytes)
	trail := make([]Hash, len(doc.Trail))
	for i, s := range doc.Trail {
		b, _ := hex.DecodeString(s)
		if len(b) != 32 {
			t.Fatal("bad trail root")
		}
		copy(trail[i][:], b)
	}
	root, err := TrailRootV2(gid, trail)
	if err != nil {
		t.Fatal(err)
	}
	if hex.EncodeToString(root[:]) != doc.TrailRootV2 {
		t.Fatal("trail root mismatch")
	}
	for _, p := range doc.Proofs {
		var sr Hash
		b, _ := hex.DecodeString(p.StateRoot)
		copy(sr[:], b)
		siblings := make([]Hash, len(p.Siblings))
		for i, s := range p.Siblings {
			sb, _ := hex.DecodeString(s)
			copy(siblings[i][:], sb)
		}
		if !VerifyTrailProofV2(root, gid, p.Step, uint32(len(trail)), sr, siblings) {
			t.Fatalf("step %d proof rejected", p.Step)
		}
		// wrong graph id is rejected
		if VerifyTrailProofV2(root, Hash{}, p.Step, uint32(len(trail)), sr, siblings) {
			t.Fatal("proof verified under a wrong graph id")
		}
		// wrong step index is rejected
		badStep := p.Step + 1
		if badStep < uint32(len(trail)) &&
			VerifyTrailProofV2(root, gid, badStep, uint32(len(trail)), sr, siblings) {
			t.Fatal("proof verified at a wrong step")
		}
		// the correct step but a V1-domain sibling is rejected
		v1Leaf := hashBytes([]byte(DomainGraphTrace), gid[:], appendUint32BE(nil, 0), trail[0][:])
		if VerifyTrailProofV2(root, gid, p.Step, uint32(len(trail)), sr,
			[]Hash{v1Leaf}) {
			t.Fatal("V1-domain material verified under the V2 trail verifier")
		}
	}
}
