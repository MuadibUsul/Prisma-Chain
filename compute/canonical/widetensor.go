package canonical

// CANONICAL_TENSOR_V2 / GEMM_A13W10_I64_V1 / REQUANTIZE_WIDE_V1 — the
// versioned wide-integer protocol family (Phase F.5C).  Everything in
// this file is ADDITIVE: the V1 objects above keep their domains, bytes
// and semantics untouched.  V2 exists because the int64 GEMM accumulator
// cannot be legally committed under the V1 int32 TensorRoot encoding.

import (
	"errors"
	"fmt"
)

// V2 hash domains. Every V2 object is hashed under its own domain string;
// a V1 proof can never verify as V2 (version-confusion tests pin this).
const (
	DomainTensorV2        = "PRISMA_CANONICAL_TENSOR_V2\x00"
	DomainTensorRootV2    = "PRISMA_CANONICAL_TENSOR_ROOT_V2\x00"
	DomainGraphV2         = "PRISMA_CANONICAL_GRAPH_V2\x00"
	DomainGraphStateV2    = "PRISMA_CANONICAL_GRAPH_STATE_V2\x00"
	DomainGraphTraceV2    = "PRISMA_CANONICAL_GRAPH_TRACE_V2\x00"
	DomainManifestV2      = "PRISMA_GRAPH_NODE_MANIFEST_V2\x00"
	DomainCommitV3        = "PRISMA_GRAPH_RESULT_COMMIT_V3\x00"
	DomainReceiptV3       = "PRISMA_GRAPH_RECEIPT_V3\x00"
	DomainWideGEMMDispute = "PRISMA_WIDE_GEMM_DISPUTE_V1\x00"
	DomainWideTrace       = "PRISMA_WIDE_GEMM_TRACE_V1\x00"
)

// Protocol version strings of the V2 family.
const (
	ProtocolVersionTensorV2   = "CANONICAL_TENSOR_V2/1.0.0"
	ProtocolVersionGraphV2    = "CANONICAL_GRAPH_V2/1.0.0"
	OpGEMMVersionWide         = "GEMM_A13W10_I64_V1/1.0.0"
	OpRequantVersionWide      = "REQUANTIZE_WIDE_V1/1.0.0"
	ArithmeticProfileA13W10   = "A13W10_I64_PROFILE_V1"
	FreivaldsWideVersion      = "FREIVALDS_A13W10_I64_V1/1.0.0"
	// Frozen F.5A policy: the A13W10 quantization plan is identified by
	// this PolicyID and bound into every V2 graph descriptor.
	PolicyIDA13W10 = "eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0"
)

// DtypeV2 is the typed element space of CANONICAL_TENSOR_V2.  The values
// are frozen; Q12.20 keeps its V1 numeric identity and BE32 encoding, the
// wide dtypes add BE16/BE64 encodings.  Logical width != storage width:
// A13 is 13 logical bits in a 16-bit container, W10 is 10 logical bits in
// a 16-bit container (never advertised as "INT16 model").
type DtypeV2 uint8

const (
	DtypeV2Q12_20     DtypeV2 = 1
	DtypeV2A13        DtypeV2 = 129
	DtypeV2W10        DtypeV2 = 130
	DtypeV2Int64Accum DtypeV2 = 131
)

// Logical ranges. Values outside the logical range are REJECTED at
// admission, never reinterpreted.
const (
	A13Min int64 = -4096
	A13Max int64 = 4095
	W10Min int64 = -512
	W10Max int64 = 511
)

// ArithmeticProfileV1 binds the frozen A13W10 arithmetic into a graph
// descriptor so the policy can never be an off-chain implicit config.
type ArithmeticProfileV1 struct {
	ID        string `gemm:"id"`
	ABits     int    `gemm:"a_bits"`
	WBits     int    `gemm:"w_bits"`
	Accum     string `gemm:"accum"`
	Rounding  string `gemm:"rounding"`
	PolicyID  string `gemm:"policy_id"`
}

// A13W10I64Profile is the frozen profile instance.
func A13W10I64Profile() ArithmeticProfileV1 {
	return ArithmeticProfileV1{ID: ArithmeticProfileA13W10, ABits: 13, WBits: 10,
		Accum: "int64", Rounding: "ties_to_even", PolicyID: PolicyIDA13W10}
}

// TensorDescriptorV2 binds dtype + layout + shape; the canonical CBOR of
// the descriptor is hashed into every leaf and the root, so the same
// bytes can never be reinterpreted as another dtype.
type TensorDescriptorV2 struct {
	Dtype  DtypeV2 `gemm:"dtype"`
	Layout Layout  `gemm:"layout"`
	Shape  []int64 `gemm:"shape"`
}

func NewDescV2(dtype DtypeV2, shape ...int64) TensorDescriptorV2 {
	return TensorDescriptorV2{Dtype: dtype, Layout: LayoutRowMajor, Shape: shape}
}

func (d TensorDescriptorV2) Elems() (int64, error) {
	n := int64(1)
	for _, dim := range d.Shape {
		if dim < 0 {
			return 0, errors.New("canonical/v2: negative dimension")
		}
		n *= dim
	}
	return n, nil
}

func (d TensorDescriptorV2) BytesPerElem() (int, error) {
	switch d.Dtype {
	case DtypeV2Q12_20:
		return 4, nil
	case DtypeV2A13, DtypeV2W10:
		return 2, nil
	case DtypeV2Int64Accum:
		return 8, nil
	default:
		return 0, fmt.Errorf("canonical/v2: unknown dtype %d", d.Dtype)
	}
}

func (d TensorDescriptorV2) Validate() error {
	if d.Layout != LayoutRowMajor {
		return errors.New("canonical/v2: unsupported layout")
	}
	if _, err := d.BytesPerElem(); err != nil {
		return err
	}
	if _, err := d.Elems(); err != nil {
		return err
	}
	return nil
}

// ValidateValues rejects elements outside the logical range of the
// descriptor's dtype (Q12.20 must stay in signed int32 range).
func (d TensorDescriptorV2) ValidateValues(data []int64) error {
	var lo, hi int64
	switch d.Dtype {
	case DtypeV2Q12_20:
		lo, hi = MinFx, MaxFx
	case DtypeV2A13:
		lo, hi = A13Min, A13Max
	case DtypeV2W10:
		lo, hi = W10Min, W10Max
	case DtypeV2Int64Accum:
		return nil
	default:
		return fmt.Errorf("canonical/v2: unknown dtype %d", d.Dtype)
	}
	for i, v := range data {
		if v < lo || v > hi {
			return fmt.Errorf("canonical/v2: element %d value %d outside logical range [%d,%d]",
				i, v, lo, hi)
		}
	}
	return nil
}

// RShiftRoundEven64 is the shared int64 ties-to-even shift
// (Go == Python == GPU semantics).
func RShiftRoundEven64(v int64, s uint) int64 {
	if s == 0 {
		return v
	}
	q := v >> s
	r := v & ((1 << s) - 1)
	half := int64(1) << (s - 1)
	if r > half || (r == half && (q&1) == 1) {
		return q + 1
	}
	return q
}

// encodeElemBE writes one element big-endian two's complement at the
// dtype's canonical width.
func encodeElemBE(dst []byte, v int64, width int) {
	for i := 0; i < width; i++ {
		dst[i] = byte(v >> (8 * (width - 1 - i)))
	}
}

// TensorV2 is a typed committed tensor.
type TensorV2 struct {
	Desc TensorDescriptorV2
	Data []int64
}

func NewTensorV2(desc TensorDescriptorV2, data []int64) (*TensorV2, error) {
	if err := desc.Validate(); err != nil {
		return nil, err
	}
	n, _ := desc.Elems()
	if int64(len(data)) != n {
		return nil, fmt.Errorf("canonical/v2: %d elements, descriptor says %d", len(data), n)
	}
	if err := desc.ValidateValues(data); err != nil {
		return nil, err
	}
	return &TensorV2{Desc: desc, Data: data}, nil
}

// ChunkCount mirrors V1: 64 logical elements per chunk.
func (t *TensorV2) ChunkCount() int {
	n, _ := t.Desc.Elems()
	return int((n + ChunkElems - 1) / ChunkElems)
}

// chunkBytes encodes chunk `index`: 64 elements at the dtype width,
// zero-padded (all-zero bytes are the canonical zero of every V2 dtype).
func (t *TensorV2) chunkBytes(index int) ([]byte, error) {
	bpe, err := t.Desc.BytesPerElem()
	if err != nil {
		return nil, err
	}
	n, _ := t.Desc.Elems()
	base := int64(index) * ChunkElems
	if base >= n || index < 0 {
		return nil, errors.New("canonical/v2: chunk index out of range")
	}
	buf := make([]byte, ChunkElems*bpe)
	for i := int64(0); i < ChunkElems && base+i < n; i++ {
		encodeElemBE(buf[i*int64(bpe):(i+1)*int64(bpe)], t.Data[base+i], bpe)
	}
	return buf, nil
}

// ChunkBytes is the public typed-chunk accessor (DA evidence, disputes).
func (t *TensorV2) ChunkBytes(index int) ([]byte, error) {
	return t.chunkBytes(index)
}

// TensorLeafV2 = SHA256(DomainTensorV2 || desc || u32be(index) || chunk).
func TensorLeafV2(descBytes []byte, index uint32, chunk []byte) Hash {
	return hashBytes([]byte(DomainTensorV2), descBytes, appendUint32BE(nil, index), chunk)
}

func (t *TensorV2) merkleLevelsV2() ([][]Hash, Hash, error) {
	descBytes, err := EncodeCanonical(t.Desc)
	if err != nil {
		return nil, Hash{}, err
	}
	count := t.ChunkCount()
	leaves := make([]Hash, count)
	for i := 0; i < count; i++ {
		chunk, err := t.chunkBytes(i)
		if err != nil {
			return nil, Hash{}, err
		}
		leaves[i] = TensorLeafV2(descBytes, uint32(i), chunk)
	}
	levels, err := buildLevels(leaves)
	if err != nil {
		return nil, Hash{}, err
	}
	return levels, levels[len(levels)-1][0], nil
}

// TensorRootV2 = SHA256(DomainTensorRootV2 || desc || MerkleRoot(chunks)).
func (t *TensorV2) TensorRootV2() (Hash, error) {
	descBytes, err := EncodeCanonical(t.Desc)
	if err != nil {
		return Hash{}, err
	}
	_, merkle, err := t.merkleLevelsV2()
	if err != nil {
		return Hash{}, err
	}
	return hashBytes([]byte(DomainTensorRootV2), descBytes, merkle[:]), nil
}

func TensorRootV2Of(desc TensorDescriptorV2, data []int64) (Hash, error) {
	t, err := NewTensorV2(desc, data)
	if err != nil {
		return Hash{}, err
	}
	return t.TensorRootV2()
}

// ChunkProofV2 proves one typed chunk against the root.
func (t *TensorV2) ChunkProofV2(index int) (chunk []byte, siblings []Hash, err error) {
	levels, _, err := t.merkleLevelsV2()
	if err != nil {
		return nil, nil, err
	}
	if index < 0 || index >= len(levels[0]) {
		return nil, nil, errors.New("canonical/v2: leaf index out of range")
	}
	chunk, err = t.chunkBytes(index)
	if err != nil {
		return nil, nil, err
	}
	siblings, err = proveLeaf(levels, index)
	if err != nil {
		return nil, nil, err
	}
	return chunk, siblings, nil
}

// VerifyChunkV2 verifies a typed chunk against a TensorRootV2.  The
// descriptor is typed input (not re-parsed from wire bytes): its canonical
// encoding is recomputed and hashed into the leaf, the chunk width is
// derived from the dtype, and the inclusion proof is first folded back to
// the chunk merkle root which is then lifted to the TensorRootV2 domain.
func VerifyChunkV2(rootV2 Hash, desc TensorDescriptorV2, index uint32, count uint32,
	chunk []byte, siblings []Hash) bool {
	bpe, err := desc.BytesPerElem()
	if err != nil || desc.Layout != LayoutRowMajor {
		return false
	}
	if len(chunk) != ChunkElems*bpe {
		return false
	}
	descBytes, err := EncodeCanonical(desc)
	if err != nil {
		return false
	}
	leaf := TensorLeafV2(descBytes, index, chunk)
	proof := MerkleProof{Index: index, Count: count, Siblings: siblings}
	if proof.Count == 0 || proof.Index >= proof.Count || len(proof.Siblings) != depthFor(proof.Count) {
		return false
	}
	merkle := foldProof(leaf, proof)
	want := hashBytes([]byte(DomainTensorRootV2), descBytes, merkle[:])
	return equalBytes(rootV2[:], want[:])
}

// foldProof recomputes the merkle root from a leaf and its inclusion proof
// (same odd-node rule as VerifyLeafInclusion).
func foldProof(leaf Hash, proof MerkleProof) Hash {
	h := leaf
	idx, cnt := proof.Index, proof.Count
	for _, sib := range proof.Siblings {
		if idx%2 == 0 {
			right := sib
			if idx+1 >= cnt {
				right = h
			}
			h = hashBytes(h[:], right[:])
		} else {
			h = hashBytes(sib[:], h[:])
		}
		idx /= 2
		cnt = (cnt + 1) / 2
	}
	return h
}

// --- GEMM_A13W10_I64_V1 ------------------------------------------------------

// MaxSafeK64 is the admission bound of GEMM_A13W10_I64_V1:
// floor((2^63 - 1) / (2^(a-1) * 2^(w-1))).  For A13W10 this is 4398046511103.
func MaxSafeK64(aBits, wBits int) uint64 {
	if aBits <= 0 || wBits <= 0 || aBits > 64 || wBits > 64 {
		return 0
	}
	den := uint64(1) << (uint(aBits) - 1 + uint(wBits) - 1)
	return (uint64(1)<<63 - 1) / den
}

// ReferenceWideGEMM is the protocol reference of GEMM_A13W10_I64_V1:
// C[i,j] = sum_k int64(A[i,k]) * int64(W[k,j]) with exact int64
// accumulation.  It is the ONLY arbiter semantics; the GPU Karatsuba
// decomposition is a worker backend detail and never appears on chain.
// transpose_b selects W stored (N, K) instead of (K, N).
func ReferenceWideGEMM(a []int64, w []int64, m, n, k int, transposeB bool) ([]int64, error) {
	if m <= 0 || n <= 0 || k <= 0 {
		return nil, errors.New("canonical/v2: empty GEMM")
	}
	if uint64(k) > MaxSafeK64(13, 10) {
		return nil, fmt.Errorf("canonical/v2: K=%d exceeds MaxSafeK64", k)
	}
	if len(a) != m*k {
		return nil, errors.New("canonical/v2: A size mismatch")
	}
	if len(w) != (func() int {
		if transposeB {
			return n * k
		}
		return k * n
	})() {
		return nil, errors.New("canonical/v2: W size mismatch")
	}
	aDesc := NewDescV2(DtypeV2A13, int64(m), int64(k))
	if err := aDesc.ValidateValues(a); err != nil {
		return nil, err
	}
	wShape := int64(n)
	if !transposeB {
		wShape = int64(k)
	}
	wDesc := NewDescV2(DtypeV2W10, wShape, int64(k))
	if !transposeB {
		wDesc = NewDescV2(DtypeV2W10, int64(k), int64(n))
	}
	if err := wDesc.ValidateValues(w); err != nil {
		return nil, err
	}
	out := make([]int64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc int64
			for d := 0; d < k; d++ {
				var wv int64
				if transposeB {
					wv = w[j*k+d]
				} else {
					wv = w[d*n+j]
				}
				acc += a[i*k+d] * wv
			}
			out[i*n+j] = acc
		}
	}
	return out, nil
}

// --- REQUANTIZE_WIDE_V1 ------------------------------------------------------

// RequantWide is the protocol reference of REQUANTIZE_WIDE_V1:
// v = clamp(round_ties_even(input * mult >> shift), lo, hi), exact
// integer throughout.  Bridges INT64_ACCUM -> Q12.20 / A13 and
// Q12.20 -> A13 (and any V2 dtype pair via lo/hi + target range check).
func RequantWide(in []int64, mult uint64, shift uint, lo, hi int64, target DtypeV2) ([]int64, error) {
	if mult == 0 || mult > 1<<62 {
		return nil, errors.New("canonical/v2: requant multiplier outside admission bounds")
	}
	out := make([]int64, len(in))
	for i, v := range in {
		// int64 safety: the product must stay in int64 (no big.Int consensus
		// dependency; oversized intermediates are a descriptor admission
		// failure, not a silent widening).
		prod := v * int64(mult)
		if v != 0 && prod/v != int64(mult) {
			return nil, fmt.Errorf("canonical/v2: requant intermediate overflow at %d", i)
		}
		out[i] = clampInt64(RShiftRoundEven64(prod, shift), lo, hi)
	}
	tDesc := NewDescV2(target, int64(len(out)))
	if err := tDesc.ValidateValues(out); err != nil {
		return nil, err
	}
	return out, nil
}

func clampInt64(v, lo, hi int64) int64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
