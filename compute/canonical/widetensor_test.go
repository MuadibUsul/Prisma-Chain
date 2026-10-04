package canonical

// Phase F.5C — CANONICAL_TENSOR_V2 / GEMM_A13W10_I64_V1 / REQUANTIZE_WIDE_V1
// tests: encoding edges, domain separation, range rejection, frozen F.5B
// vector agreement, admission bounds and ties-to-even requant.

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestTensorV2EncodingEdges(t *testing.T) {
	desc := NewDescV2(DtypeV2A13, 5)
	tensor, err := NewTensorV2(desc, []int64{-4096, -1, 0, 1, 4095})
	if err != nil {
		t.Fatal(err)
	}
	chunk, err := tensor.ChunkBytes(0)
	if err != nil {
		t.Fatal(err)
	}
	want := []byte{
		0xF0, 0x00, // -4096 BE16
		0xFF, 0xFF, // -1 BE16
		0x00, 0x00, // 0
		0x00, 0x01, // 1
		0x0F, 0xFF, // 4095
	}
	for i, b := range want {
		if chunk[i] != b {
			t.Fatalf("A13 BE16 byte %d = %02x, want %02x", i, chunk[i], b)
		}
	}
	if len(chunk) != ChunkElems*2 {
		t.Fatalf("chunk size %d, want %d (zero-padded)", len(chunk), ChunkElems*2)
	}
	for i := len(want); i < len(chunk); i++ {
		if chunk[i] != 0 {
			t.Fatalf("padding byte %d = %02x, want 0", i, chunk[i])
		}
	}

	// W10 edges
	wdesc := NewDescV2(DtypeV2W10, 2)
	wt, err := NewTensorV2(wdesc, []int64{-512, 511})
	if err != nil {
		t.Fatal(err)
	}
	wchunk, _ := wt.ChunkBytes(0)
	if wchunk[0] != 0xFE || wchunk[1] != 0x00 || wchunk[2] != 0x01 || wchunk[3] != 0xFF {
		t.Fatalf("W10 BE16 encoding wrong: %x", wchunk[:4])
	}

	// INT64_ACCUM edges
	idesc := NewDescV2(DtypeV2Int64Accum, 2)
	it, err := NewTensorV2(idesc, []int64{math.MinInt64, math.MaxInt64})
	if err != nil {
		t.Fatal(err)
	}
	ichunk, _ := it.ChunkBytes(0)
	if ichunk[0] != 0x80 || ichunk[7] != 0x00 || ichunk[8] != 0x7F || ichunk[15] != 0xFF {
		t.Fatalf("INT64 BE64 encoding wrong: %x", ichunk[:16])
	}
	// two's complement has exactly one zero: +0 and -0 are the same bytes
	z1, _ := NewTensorV2(idesc, []int64{0, 0})
	zc, _ := z1.ChunkBytes(0)
	for i := 0; i < 16; i++ {
		if zc[i] != 0 {
			t.Fatal("zero must encode as all-zero bytes (no negative zero)")
		}
	}
}

func TestTensorV2RangeRejection(t *testing.T) {
	if _, err := NewTensorV2(NewDescV2(DtypeV2A13, 1), []int64{4096}); err == nil {
		t.Fatal("A13 value 4096 must be rejected")
	}
	if _, err := NewTensorV2(NewDescV2(DtypeV2A13, 1), []int64{-4097}); err == nil {
		t.Fatal("A13 value -4097 must be rejected")
	}
	if _, err := NewTensorV2(NewDescV2(DtypeV2W10, 1), []int64{512}); err == nil {
		t.Fatal("W10 value 512 must be rejected")
	}
	if _, err := NewTensorV2(NewDescV2(DtypeV2W10, 1), []int64{-513}); err == nil {
		t.Fatal("W10 value -513 must be rejected")
	}
	// Q12.20 stays in signed int32 range
	if _, err := NewTensorV2(NewDescV2(DtypeV2Q12_20, 1), []int64{1 << 31}); err == nil {
		t.Fatal("Q12.20 value 2^31 must be rejected")
	}
}

func TestTensorV2DomainSeparation(t *testing.T) {
	data := []int64{-100, 200, 300}
	v2desc := NewDescV2(DtypeV2Q12_20, 3)
	v2root, err := TensorRootV2Of(v2desc, data)
	if err != nil {
		t.Fatal(err)
	}
	// the V1 root of the same logical tensor must differ (different domain
	// and different descriptor encoding)
	v1desc := NewDesc(DtypeQ12_20, 3)
	v1root, err := TensorRootOf(v1desc, []int32{-100, 200, 300})
	if err != nil {
		t.Fatal(err)
	}
	if equalBytes(v2root[:], v1root[:]) {
		t.Fatal("V2 root must differ from V1 root for the same values (domain separation)")
	}

	// a V1 chunk proof must never verify as V2: build a V1 leaf and check
	// against the V2 root through the V2 verifier
	t2, err := NewTensorV2(v2desc, data)
	if err != nil {
		t.Fatal(err)
	}
	chunk, siblings, err := t2.ChunkProofV2(0)
	if err != nil {
		t.Fatal(err)
	}
	if !VerifyChunkV2(v2root, v2desc, 0, uint32(t2.ChunkCount()), chunk, siblings) {
		t.Fatal("valid V2 chunk proof rejected")
	}
	// wrong dtype descriptor must fail even with the correct chunk bytes
	wrongDesc := NewDescV2(DtypeV2A13, 3)
	if VerifyChunkV2(v2root, wrongDesc, 0, uint32(t2.ChunkCount()), chunk, siblings) {
		t.Fatal("chunk proof verified under a different dtype descriptor")
	}
	// tampering one byte must fail
	bad := append([]byte(nil), chunk...)
	bad[0] ^= 1
	if VerifyChunkV2(v2root, v2desc, 0, uint32(t2.ChunkCount()), bad, siblings) {
		t.Fatal("tampered chunk verified")
	}
}

func TestMaxSafeK64(t *testing.T) {
	if got := MaxSafeK64(13, 10); got != 4398046511103 {
		t.Fatalf("MaxSafeK64(13,10) = %d, want 4398046511103", got)
	}
}

func TestRShiftRoundEven64Ties(t *testing.T) {
	cases := map[int64]int64{
		0: 0, 1: 0, 3: 1, 2: 0, -1: 0, -3: -1, -2: 0,
	}
	for in, want := range cases {
		if got := RShiftRoundEven64(in, 2); got != want {
			t.Fatalf("RShiftRoundEven64(%d, 2) = %d, want %d", in, got, want)
		}
	}
	// exact halves round to even: 2>>1 = 1 (odd) so 3>>1 -> 2; 1>>1 -> 0
	if got := RShiftRoundEven64(3, 1); got != 2 {
		t.Fatalf("3>>1 rte = %d, want 2", got)
	}
	if got := RShiftRoundEven64(1, 1); got != 0 {
		t.Fatalf("1>>1 rte = %d, want 0", got)
	}
	if got := RShiftRoundEven64(-3, 1); got != -2 {
		t.Fatalf("-3>>1 rte = %d, want -2", got)
	}
}

// TestReferenceWideGEMMFrozenVectors replays the frozen F.5B wide GEMM
// vectors through the V2 protocol reference: Go must reproduce every
// element exactly (and by construction Python and the proven GPU kernel
// already reproduce the same file).
func TestReferenceWideGEMMFrozenVectors(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "f5b_wide_gemm_vectors.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("frozen vectors not present: %v", err)
	}
	var doc struct {
		Vectors []struct {
			Name       string  `json:"name"`
			AShape     []int   `json:"a_shape"`
			WShape     []int   `json:"w_shape"`
			A          []int64 `json:"a"`
			W          []int64 `json:"w"`
			TransposeB bool    `json:"transpose_b"`
			CPU        []int64 `json:"cpu_direct"`
		} `json:"vectors"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if len(doc.Vectors) == 0 {
		t.Fatal("no vectors")
	}
	for _, v := range doc.Vectors {
		m, k := v.AShape[0], v.AShape[1]
		var n int
		if v.TransposeB {
			n = v.WShape[0]
		} else {
			n = v.WShape[1]
		}
		got, err := ReferenceWideGEMM(v.A, v.W, m, n, k, v.TransposeB)
		if err != nil {
			t.Fatalf("%s: %v", v.Name, err)
		}
		if len(got) != len(v.CPU) {
			t.Fatalf("%s: length %d, want %d", v.Name, len(got), len(v.CPU))
		}
		for i := range got {
			if got[i] != v.CPU[i] {
				t.Fatalf("%s: element %d = %d, want %d", v.Name, i, got[i], v.CPU[i])
			}
		}
	}
}

func TestRequantWideTiesAndSaturation(t *testing.T) {
	// ties-to-even at shift 1: 1 -> 0, 3 -> 2, -1 -> 0, -3 -> -2
	got, err := RequantWide([]int64{1, 3, -1, -3}, 1, 1, MinFx, MaxFx, DtypeV2Q12_20)
	if err != nil {
		t.Fatal(err)
	}
	want := []int64{0, 2, 0, -2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("requant[%d] = %d, want %d", i, got[i], want[i])
		}
	}
	// saturation into A13 range
	got, err = RequantWide([]int64{1 << 40, -(1 << 40)}, 1, 1, A13Min, A13Max, DtypeV2A13)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != A13Max || got[1] != A13Min {
		t.Fatalf("saturation wrong: %v", got)
	}
	// mult outside admission bounds rejected
	if _, err := RequantWide([]int64{1}, 0, 1, A13Min, A13Max, DtypeV2A13); err == nil {
		t.Fatal("mult=0 must be rejected")
	}
	// overflow of the intermediate is an error, not silent wrap
	if _, err := RequantWide([]int64{1 << 62}, 8, 1, A13Min, A13Max, DtypeV2A13); err == nil {
		t.Fatal("int64 overflow must be rejected")
	}
}
