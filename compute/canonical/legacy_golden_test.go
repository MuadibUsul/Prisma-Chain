package canonical

// Legacy golden regression (roadmap A0-02): the frozen V1 corpus must
// reproduce byte-for-byte on every future tree, and V1/V2 domains must
// reject each other's proofs.

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type legacyGolden struct {
	V1 struct {
		TensorRoots []struct {
			Name string `json:"name"`
			Root string `json:"root"`
		} `json:"tensor_roots"`
		GraphID         string `json:"graph_id"`
		CommitPreimage  string `json:"commit_preimage"`
		ReceiptID       string `json:"receipt_id"`
		CommitVersion   string `json:"commit_version"`
		SignatureDomain string `json:"signature_domain"`
	} `json:"v1"`
	V2 struct {
		A13ProbeRoot   string `json:"a13_probe_root"`
		CommitPreimage string `json:"commit_preimage"`
		CommitVersion  string `json:"commit_version"`
		GraphIDV2      string `json:"graph_id_v2"`
		PolicyID       string `json:"policy_id"`
	} `json:"v2"`
}

func loadLegacyGolden(t *testing.T) (*legacyGolden, map[string]any) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "f5c_legacy_golden.json"))
	if err != nil {
		t.Fatalf("golden fixture missing: %v", err)
	}
	var doc legacyGolden
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	vecRaw, err := os.ReadFile(filepath.Join("testdata", "canonical_vectors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var vectors map[string]any
	if err := json.Unmarshal(vecRaw, &vectors); err != nil {
		t.Fatal(err)
	}
	return &doc, vectors
}

func TestLegacyGoldenUnchanged(t *testing.T) {
	golden, vectors := loadLegacyGolden(t)

	// 1. V1 tensor roots recompute
	trs := vectors["tensor_roots"].([]any)
	for _, trAny := range trs {
		tr := trAny.(map[string]any)
		name := tr["name"].(string)
		found := false
		for _, want := range golden.V1.TensorRoots {
			if want.Name != name {
				continue
			}
			found = true
			desc := tr["desc"].(map[string]any)
			shapeParts := desc["shape"].([]any)
			shape := make([]int64, len(shapeParts))
			for i, sp := range shapeParts {
				shape[i] = int64(sp.(float64))
			}
			var data []int32
			for _, d := range tr["data"].([]any) {
				data = append(data, int32(d.(float64)))
			}
			root, err := TensorRootOf(NewDesc(Dtype(desc["dtype"].(float64)), shape...), data)
			if err != nil {
				t.Fatal(err)
			}
			if hex.EncodeToString(root[:]) != want.Root {
				t.Fatalf("V1 tensor root %s changed", name)
			}
		}
		if !found {
			t.Fatalf("golden is missing tensor root %s", name)
		}
	}

	// 2. V1 commit preimage hash
	rcV1 := &GraphResultCommit{
		ProtocolVersion: golden.V1.CommitVersion,
		GraphID:         repeatHex("11", 32),
		TaskRef:         []byte("legacy-golden-task"),
		AssignmentRef:   repeatHex("22", 32),
		WorkerPubKey:    repeatHex("33", 32),
		FinalOutputRoot: repeatHex("44", 32),
		OutputRoots:     [][]byte{repeatHex("55", 32), repeatHex("66", 32)},
		CompletedEpoch:  7,
	}
	enc, err := EncodeCanonical(rcV1)
	if err != nil {
		t.Fatal(err)
	}
	pre := sha256.Sum256(append([]byte(DomainSig), enc...))
	if hex.EncodeToString(pre[:]) != golden.V1.CommitPreimage {
		t.Fatal("V1 commit preimage changed")
	}

	// 3. V2 A13 probe root
	root, err := TensorRootV2Of(NewDescV2(DtypeV2A13, 3), []int64{-4096, -1, 4095})
	if err != nil {
		t.Fatal(err)
	}
	if hex.EncodeToString(root[:]) != golden.V2.A13ProbeRoot {
		t.Fatal("V2 A13 probe root changed")
	}

	// 4. V3 commit preimage hash
	rcV3 := &GraphResultCommitV3{
		ProtocolVersion:          golden.V2.CommitVersion,
		GraphID:                  repeatHex("77", 32),
		ArithmeticID:             ArithmeticProfileA13W10,
		PolicyID:                 PolicyIDA13W10,
		TaskRef:                  []byte("legacy-golden-task"),
		AssignmentRef:            repeatHex("22", 32),
		WorkerPubKey:             repeatHex("33", 32),
		NodeOutputManifestRootV2: repeatHex("88", 32),
		FinalOutputRoot:          repeatHex("44", 32),
		OutputRoots:              [][]byte{repeatHex("55", 32)},
		CompletedEpoch:           7,
	}
	enc3, err := EncodeCanonical(rcV3)
	if err != nil {
		t.Fatal(err)
	}
	pre3 := sha256.Sum256(append([]byte(DomainSigV3), enc3...))
	if hex.EncodeToString(pre3[:]) != golden.V2.CommitPreimage {
		t.Fatal("V3 commit preimage changed")
	}
	// The V1 and V3 preimages differ (separate domains + structures)
	if bytes.Equal(pre[:], pre3[:]) {
		t.Fatal("V1 and V3 preimages collide")
	}
}

// TestLegacyCrossVersionRejection: proofs never cross the version boundary.
//
// Note on chunk bytes: the Q12.20 chunk *payload* encoding is identical in
// V1 and V2 (64 x BE32) by design — the version boundary lives in the
// descriptor encoding, the leaf domain and the root domain.  Single-chunk
// tensors therefore have equal payload bytes; the rejection tests below
// use multi-chunk tensors so the version's own Merkle material (sibling
// hashes + roots) must actually cross the boundary.
func TestLegacyCrossVersionRejection(t *testing.T) {
	data130 := make([]int32, 130)
	for i := range data130 {
		data130[i] = int32(i*37) - 2000
	}
	v1t := &Tensor{Desc: NewDesc(DtypeQ12_20, 130), Data: data130}
	v1Root, err := v1t.TensorRoot()
	if err != nil {
		t.Fatal(err)
	}
	v1Sib, err := v1t.ChunkProof(0)
	if err != nil {
		t.Fatal(err)
	}
	v1Chunk, err := v1t.ChunkBytes(0)
	if err != nil {
		t.Fatal(err)
	}
	data64 := make([]int64, 130)
	for i := range data64 {
		data64[i] = int64(data130[i])
	}
	v2desc := NewDescV2(DtypeV2Q12_20, 130)
	v2t, err := NewTensorV2(v2desc, data64)
	if err != nil {
		t.Fatal(err)
	}
	// (a) V1 root + V1 Merkle siblings through the V2 verifier: rejected
	if VerifyChunkV2(v1Root, v2desc, 0, uint32(v1t.ChunkCount()), v1Chunk, v1Sib) {
		t.Fatal("V1 chunk proof verified as V2")
	}
	// (b) V1 TensorRoot used as a V2 root: rejected by direct comparison
	v2Root, err := v2t.TensorRootV2()
	if err != nil {
		t.Fatal(err)
	}
	if v1Root == v2Root {
		t.Fatal("V1 root equals V2 root for the same values")
	}
	// (c) V2 evidence through the V1 verifier: the V2 merkle material can
	// never fold to the V1 chunk-merkle root.
	v2Chunk, v2Sib, err := v2t.ChunkProofV2(0)
	if err != nil {
		t.Fatal(err)
	}
	v1Merkle, err := v1t.MerkleRoot()
	if err != nil {
		t.Fatal(err)
	}
	descBytes, err := EncodeCanonical(v1t.Desc)
	if err != nil {
		t.Fatal(err)
	}
	leafV2AsV1 := TensorLeaf(descBytes, 0, v2Chunk)
	proof := MerkleProof{Index: 0, Count: uint32(v2t.ChunkCount()), Siblings: v2Sib}
	if VerifyLeafInclusion(v1Merkle, leafV2AsV1, proof) {
		t.Fatal("V2 evidence verified under the V1 verifier")
	}
	// Receipt ids live in separate domains: same-shaped data must differ.
	v1Receipt := &VerifiedGraphWorkReceiptV1{ProtocolVersion: "1.1.0", GraphID: repeatHex("aa", 32),
		Spec: "X", TaskRef: []byte("t"), AssignmentRef: []byte("a"), WorkerPubKey: repeatHex("bb", 32),
		FinalOutputRoot: repeatHex("cc", 32), OutputRoots: [][]byte{repeatHex("dd", 32)},
		WorkVector: WorkVector{}.Add("K", 1), VerificationMode: ModeOptimisticUnchallenged,
		FinalizedEpoch: 1, SettlementReference: []byte("s")}
	id1, err := v1Receipt.ReceiptID()
	if err != nil {
		t.Fatal(err)
	}
	v3Receipt := &VerifiedGraphWorkReceiptV3{ProtocolVersion: "3.0.0", GraphID: repeatHex("aa", 32),
		Spec: "X", ArithmeticID: ArithmeticProfileA13W10, PolicyID: PolicyIDA13W10,
		TaskRef: []byte("t"), AssignmentRef: []byte("a"), WorkerPubKey: repeatHex("bb", 32),
		NodeOutputManifestRootV2: repeatHex("ee", 32),
		FinalOutputRoot:          repeatHex("cc", 32), OutputRoots: [][]byte{repeatHex("dd", 32)},
		WorkVector: WorkVector{}.Add("K", 1), VerificationMode: ModeOptimisticUnchallenged,
		FinalizedEpoch: 1, SettlementReference: []byte("s")}
	id3, err := v3Receipt.ReceiptIDV3()
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(id1[:], id3[:]) {
		t.Fatal("V1 and V3 receipt ids collide")
	}
}

func repeatHex(hexByte string, n int) []byte {
	b, _ := hex.DecodeString(hexByte)
	out := make([]byte, n)
	for i := range out {
		out[i] = b[0]
	}
	return out
}

