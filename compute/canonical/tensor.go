package canonical

// Canonical tensors: CANONICAL_GRAPH_V1 binds every tensor to a
// descriptor (dtype, shape, layout) plus a chunked Merkle root, so the
// same bytes can never be reinterpreted under a different shape or dtype.

import (
	"errors"
)

// Dtypes. int8 and int32_accum exist to interface with the frozen GEMM
// wire format; q12.20 is the canonical activation format of
// CANONICAL_MATH_V1.
type Dtype uint8

const (
	DtypeInt8       Dtype = 1
	DtypeInt32Accum Dtype = 2
	DtypeQ12_20     Dtype = 3
)

// Layouts. Only row-major exists in v1.
type Layout uint8

const LayoutRowMajor Layout = 1

// ChunkElems is the frozen tensor-tree chunk size in elements: one chunk
// holds 64 Q12.20 values (256 canonical bytes), matching the 8x8 GEMM tile
// shape so a graph can hand chunks to GEMM nodes without re-encoding.
const ChunkElems = 64

// TensorDescriptor is the canonical shape binding of one tensor.
type TensorDescriptor struct {
	Dtype  uint8   `gemm:"dtype"`
	Layout uint8   `gemm:"layout"`
	Shape  []int64 `gemm:"shape"`
}

// NewDesc builds a descriptor with layout fixed to row-major.
func NewDesc(dtype Dtype, shape ...int64) TensorDescriptor {
	return TensorDescriptor{Dtype: uint8(dtype), Layout: uint8(LayoutRowMajor), Shape: append([]int64(nil), shape...)}
}

// Elems returns the element count; shapes with a non-positive dimension
// are rejected by Validate.
func (d TensorDescriptor) Elems() (int64, error) {
	if len(d.Shape) == 0 {
		return 0, errors.New("canonical: empty shape")
	}
	total := int64(1)
	for _, dim := range d.Shape {
		if dim <= 0 {
			return 0, errors.New("canonical: non-positive dimension")
		}
		if total > (1<<62)/dim {
			return 0, errors.New("canonical: shape overflow")
		}
		total *= dim
	}
	return total, nil
}

// BytesPerElem maps dtype to canonical storage width.
func (d TensorDescriptor) BytesPerElem() (int, error) {
	switch Dtype(d.Dtype) {
	case DtypeInt8:
		return 1, nil
	case DtypeInt32Accum, DtypeQ12_20:
		return 4, nil
	default:
		return 0, errors.New("canonical: unknown dtype")
	}
}

// Validate rejects malformed descriptors.
func (d TensorDescriptor) Validate() error {
	if _, err := d.Elems(); err != nil {
		return err
	}
	if d.Layout != uint8(LayoutRowMajor) {
		return errors.New("canonical: only row-major layout is supported")
	}
	if _, err := d.BytesPerElem(); err != nil {
		return err
	}
	return nil
}

// Tensor is a Q12.20 (or accumulator) tensor in canonical element order.
type Tensor struct {
	Desc TensorDescriptor
	Data []int32
}

// ChunkCount returns ceil(elems / ChunkElems).
func (t *Tensor) ChunkCount() int {
	n, err := t.Desc.Elems()
	if err != nil {
		return 0
	}
	chunks := int((n + ChunkElems - 1) / ChunkElems)
	if chunks == 0 {
		chunks = 1
	}
	return chunks
}

// chunkBytes zero-pads the final chunk so every leaf has the same width.
func (t *Tensor) chunkBytes(index int) ([]byte, error) {
	if index < 0 || index >= t.ChunkCount() {
		return nil, errors.New("canonical: chunk index out of range")
	}
	buf := make([]byte, ChunkElems*4)
	for i := 0; i < ChunkElems; i++ {
		pos := index*ChunkElems + i
		var v int32
		if pos < len(t.Data) {
			v = t.Data[pos]
		}
		buf[i*4] = byte(uint32(v) >> 24)
		buf[i*4+1] = byte(uint32(v) >> 16)
		buf[i*4+2] = byte(uint32(v) >> 8)
		buf[i*4+3] = byte(uint32(v))
	}
	return buf, nil
}

// TensorLeaf hashes chunk index of tensor desc: the leaf binds the
// descriptor, the position and the zero-padded canonical bytes.
func TensorLeaf(descBytes []byte, index uint32, chunk []byte) Hash {
	return hashBytes([]byte(DomainTensor), descBytes, appendUint32BE(nil, index), chunk)
}

// TensorTree builds the levels and root over the chunk leaves.
func (t *Tensor) TensorTree() ([][]Hash, Hash, error) {
	if err := t.Desc.Validate(); err != nil {
		return nil, Hash{}, err
	}
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
		leaves[i] = TensorLeaf(descBytes, uint32(i), chunk)
	}
	levels, err := buildLevels(leaves)
	if err != nil {
		return nil, Hash{}, err
	}
	return levels, levels[len(levels)-1][0], nil
}

// TensorRoot binds descriptor plus chunk-merkle root:
// SHA256(domain || CBOR(desc) || merkle_root).
func (t *Tensor) TensorRoot() (Hash, error) {
	descBytes, err := EncodeCanonical(t.Desc)
	if err != nil {
		return Hash{}, err
	}
	_, merkle, err := t.TensorTree()
	if err != nil {
		return Hash{}, err
	}
	return hashBytes([]byte(DomainTensorRoot), descBytes, merkle[:]), nil
}

// TensorRootOf rebuilds the root from a descriptor and raw canonical
// tensor bytes (used by verifiers that receive bytes, not int32 slices).
func TensorRootOf(desc TensorDescriptor, data []int32) (Hash, error) {
	t := &Tensor{Desc: desc, Data: data}
	return t.TensorRoot()
}

// ChunkProof proves one zero-padded chunk against a tensor root.
func (t *Tensor) ChunkProof(index int) ([]Hash, error) {
	levels, _, err := t.TensorTree()
	if err != nil {
		return nil, err
	}
	return proveLeaf(levels, index)
}

// MerkleRoot returns the bare chunk-tree root (bound into TensorRoot).
func (t *Tensor) MerkleRoot() (Hash, error) {
	_, merkle, err := t.TensorTree()
	return merkle, err
}

// VerifyChunk proves chunk bytes at index belong to the full TensorRoot
// (descriptor binding plus chunk Merkle root) for this descriptor: the
// sibling path rebuilds the chunk-tree root, which is then wrapped by the
// descriptor domain hash exactly as TensorRoot does.
func VerifyChunk(root Hash, descBytes []byte, index uint32, count uint32, chunk []byte, siblings []Hash) bool {
	if count == 0 || index >= count || len(siblings) != depthFor(count) {
		return false
	}
	leaf := TensorLeaf(descBytes, index, chunk)
	h := leaf
	idx, cnt := index, count
	for _, sib := range siblings {
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
	return hashBytes([]byte(DomainTensorRoot), descBytes, h[:]) == root
}

// --- minimal merkle helpers (same rules as the rest of the repo:
// odd last node pairs with itself; a single leaf is its own root) ---

func buildLevels(leaves []Hash) ([][]Hash, error) {
	if len(leaves) == 0 {
		return nil, errors.New("canonical: merkle tree needs at least one leaf")
	}
	level := append([]Hash(nil), leaves...)
	levels := [][]Hash{level}
	for len(level) > 1 {
		next := make([]Hash, 0, (len(level)+1)/2)
		for i := 0; i < len(level); i += 2 {
			left := level[i]
			right := left
			if i+1 < len(level) {
				right = level[i+1]
			}
			next = append(next, hashBytes(left[:], right[:]))
		}
		levels = append(levels, next)
		level = next
	}
	return levels, nil
}

func proveLeaf(levels [][]Hash, index int) ([]Hash, error) {
	count := len(levels[0])
	if index >= count {
		return nil, errors.New("canonical: leaf index out of range")
	}
	idx := index
	siblings := make([]Hash, 0, len(levels)-1)
	for lvl := 0; lvl < len(levels)-1; lvl++ {
		nodes := levels[lvl]
		sib := idx
		if idx%2 == 0 {
			sib = idx + 1
			if sib >= len(nodes) {
				sib = idx
			}
		} else {
			sib = idx - 1
		}
		siblings = append(siblings, nodes[sib])
		idx /= 2
	}
	return siblings, nil
}

func depthFor(count uint32) int {
	depth := 0
	for count > 1 {
		count = (count + 1) / 2
		depth++
	}
	return depth
}

// MerkleProof is the generic inclusion proof for one leaf (used by the
// graph trace and state machinery; tensor chunks use VerifyChunk).
type MerkleProof struct {
	Index    uint32
	Count    uint32
	Siblings []Hash
}

// VerifyLeafInclusion verifies leaf at proof position against root with
// the repo's canonical odd-node rule.
func VerifyLeafInclusion(root Hash, leaf Hash, proof MerkleProof) bool {
	if proof.Count == 0 || proof.Index >= proof.Count || len(proof.Siblings) != depthFor(proof.Count) {
		return false
	}
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
	return h == root
}
