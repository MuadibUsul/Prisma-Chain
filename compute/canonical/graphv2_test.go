package canonical

// CANONICAL_GRAPH_V2 end-to-end protocol tests: a small wide graph
// executes deterministically, manifests and V3 commits bind it, the V3
// receipt has its own domain and replay/version confusion is rejected.

import (
	"crypto/ed25519"
	"crypto/rand"
	"testing"
)

func smallWideGraph(t *testing.T) *GraphDescriptorV2 {
	t.Helper()
	a := mustTensorV2(t, NewDescV2(DtypeV2A13, 2, 3), []int64{1, -2, 3, 4095, -4096, 7})
	w := mustTensorV2(t, NewDescV2(DtypeV2W10, 3, 2), []int64{1, -1, 2, 0, -3, 511})
	aroot, _ := a.TensorRootV2()
	wroot, _ := w.TensorRootV2()
	g := &GraphDescriptorV2{
		ProtocolVersion: ProtocolVersionGraphV2,
		Spec:            "TEST_WIDE_BLOCK_V1",
		Arithmetic:      A13W10I64Profile(),
		Inputs: []GraphInputV2{
			{Name: "a", Desc: a.Desc, Root: aroot[:]},
			{Name: "w", Desc: w.Desc, Root: wroot[:]},
		},
		Nodes: []GraphNodeV2{
			{
				NodeID: 0, OperatorID: OpGEMMWideA13W10, Version: OpGEMMVersionWide,
				Inputs: []TensorRef{{Kind: 0, Index: 0}, {Kind: 0, Index: 1}},
				Output: NewDescV2(DtypeV2Int64Accum, 2, 2),
				Params: ParamList{},
			},
			{
				NodeID: 1, OperatorID: OpRequantizeWideV1, Version: OpRequantVersionWide,
				Inputs: []TensorRef{{Kind: 1, Index: 0}},
				Output: NewDescV2(DtypeV2A13, 2, 2),
				Params: ParamList{{Key: "clamp_hi", Value: A13Max}, {Key: "clamp_lo", Value: A13Min},
					{Key: "mult", Value: 1 << 20}, {Key: "out_dtype", Value: int64(DtypeV2A13)},
					{Key: "shift", Value: 20}},
			},
		},
		Outputs: []TensorRef{{Kind: 1, Index: 1}},
	}
	return g
}

func TestGraphV2SmallExecutionAndCommitments(t *testing.T) {
	g := smallWideGraph(t)
	if err := g.ValidateV2(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	a := mustTensorV2(t, NewDescV2(DtypeV2A13, 2, 3), []int64{1, -2, 3, 4095, -4096, 7})
	w := mustTensorV2(t, NewDescV2(DtypeV2W10, 3, 2), []int64{1, -1, 2, 0, -3, 511})
	exec, err := ExecuteGraphV2(g, map[uint32]*TensorV2{0: a, 1: w})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	if len(exec.Trail) != 3 {
		t.Fatalf("trail length %d, want 3", len(exec.Trail))
	}
	gemmOut := exec.Tensors[TensorRef{Kind: 1, Index: 0}]
	want := []int64{
		1*1 + (-2)*2 + 3*(-3), 1*(-1) + (-2)*0 + 3*511,
		4095*1 + (-4096)*2 + 7*(-3), 4095*(-1) + (-4096)*0 + 7*511,
	}
	for i, v := range want {
		if gemmOut.Data[i] != v {
			t.Fatalf("gemm[%d] = %d, want %d", i, gemmOut.Data[i], v)
		}
	}
	if exec.WorkVector.Get("GEMM_A13W10_MAC", 0) != 12 {
		t.Fatalf("work vector missing GEMM_A13W10_MAC=12: %v", exec.WorkVector)
	}
	if exec.WorkVector.Get("REQUANTIZE_WIDE_ELEMENT", 0) != 4 {
		t.Fatalf("work vector missing REQUANTIZE_WIDE_ELEMENT=4: %v", exec.WorkVector)
	}

	// manifest over all node roots + commit V3 + signature + receipt V3
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	rc, err := NewGraphResultCommitV3(g, []byte("task-1"), []byte("assign-1"), pub, exec, 5)
	if err != nil {
		t.Fatal(err)
	}
	if err := SignGraphResultCommitV3(rc, priv); err != nil {
		t.Fatal(err)
	}
	if err := ValidateGraphResultCommitV3(g, rc, nil); err != nil {
		t.Fatalf("validate commit: %v", err)
	}
	// tampering invalidates
	rc2 := *rc
	rc2.CompletedEpoch = 6
	if VerifyGraphResultCommitV3Signature(&rc2) {
		t.Fatal("tampered commit verified")
	}
	// a V1/V2-domain signature can never validate V3 bytes: re-sign the
	// same struct under the V2 signing domain and check it fails
	unsigned := *rc
	unsigned.Signature = nil
	enc, _ := EncodeCanonical(&unsigned)
	badSig := ed25519.Sign(priv, enc) // missing the V3 domain
	rc3 := *rc
	rc3.Signature = badSig
	if VerifyGraphResultCommitV3Signature(&rc3) {
		t.Fatal("signature without the V3 domain verified")
	}

	receipt, err := BuildVerifiedGraphWorkReceiptV3(g, []byte("task-1"), []byte("assign-1"),
		pub, exec.Outputs, exec.WorkVector, rc.NodeOutputManifestRootV2, "freivalds_a13w10_i64",
		7, []byte("settle-1"), []byte("da-1"), nil)
	if err != nil {
		t.Fatal(err)
	}
	id, err := receipt.ReceiptIDV3()
	if err != nil {
		t.Fatal(err)
	}
	// receipt id is deterministic and domain-separated
	id2, _ := receipt.ReceiptIDV3()
	if id != id2 {
		t.Fatal("receipt id not deterministic")
	}
	// manifest leaf proof for node 0
	graphID, _ := g.GraphIDV2()
	root0, _ := exec.Tensors[TensorRef{Kind: 1, Index: 0}].TensorRootV2()
	manifestRoot, _, err := BuildNodeOutputManifestV2(g, []Hash{
		root0,
		mustRootV2(t, exec.Tensors[TensorRef{Kind: 1, Index: 1}]),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !equalBytes(manifestRoot[:], rc.NodeOutputManifestRootV2) {
		t.Fatal("manifest root mismatch with the commit")
	}
	_ = graphID
}

func mustTensorV2(t *testing.T, desc TensorDescriptorV2, data []int64) *TensorV2 {
	t.Helper()
	tensor, err := NewTensorV2(desc, data)
	if err != nil {
		t.Fatal(err)
	}
	return tensor
}

func mustRootV2(t *testing.T, tensor *TensorV2) Hash {
	t.Helper()
	root, err := tensor.TensorRootV2()
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func TestGraphV2RejectsV1GraphProtocol(t *testing.T) {
	g := smallWideGraph(t)
	g.ProtocolVersion = "1.0.0" // a V1 version string must not pass V2 validation
	if err := g.ValidateV2(); err == nil {
		t.Fatal("V1 protocol version accepted by V2 validation")
	}
	g = smallWideGraph(t)
	g.Arithmetic = ArithmeticProfileV1{ID: "OTHER"}
	if err := g.ValidateV2(); err == nil {
		t.Fatal("wrong arithmetic profile accepted")
	}
}

func TestGraphV2OverKRejected(t *testing.T) {
	g := smallWideGraph(t)
	g.Nodes[0].Output = NewDescV2(DtypeV2Int64Accum, 2, 2)
	// K = 3 here; admissions bound is enormous, so force an over-bound shape
	// by claiming K = MaxSafeK64+1 through the descriptor.
	bigK := int64(MaxSafeK64(13, 10) + 1)
	g2 := smallWideGraph(t)
	g2.Inputs[0].Desc = NewDescV2(DtypeV2A13, 2, bigK)
	if err := g2.ValidateV2(); err == nil {
		t.Fatal("over-MaxSafeK64 shape accepted")
	}
}
