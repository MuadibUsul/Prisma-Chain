package canonical

// GEMM_INT8_V1 as a canonical graph node. The arithmetic is the frozen
// v0.1.1 semantics (int8 x int8 -> int32 accumulator, admission K <=
// MaxSafeK) reused through gemmv1.ReferenceGEMM; nothing in gemmv1 is
// modified or re-implemented here.
//
// The one graph-level addition is the optional `transpose_b` parameter:
// attention needs scores = Q * K^T, and a static graph has no view or
// transpose operator. With transpose_b = 1 the node declares B as [N, K]
// and computes C[i, j] = sum_d A[i, d] * B[j, d]. The transpose is exact
// index arithmetic inside the adapter; the GEMM micro-arithmetic is
// byte-identical to v0.1.1 and the dispute dispatch for GEMM nodes stays
// with the frozen v0.1.1 tile machinery (chain layer).

import (
	"errors"
	"fmt"

	"prismachain/compute/gemmv1"
)

type gemmInt8V1 struct{}

func (gemmInt8V1) ID() string      { return OpGEMMInt8V1 }
func (gemmInt8V1) Version() string { return VersionGEMM }

// TransposeB reports whether the node uses the transposed-B form.
func TransposeB(p ParamList) bool { return p.Get("transpose_b", 0) == 1 }

func (gemmInt8V1) ValidateParams(p ParamList) error {
	if v := p.Get("transpose_b", 0); v != 0 && v != 1 {
		return errors.New("canonical: GEMM transpose_b must be 0 or 1")
	}
	return nil
}

// GEMMNodeDims derives M, N, K and the transposed-B flag from a GEMM
// node's committed input descriptors. The chain dispatch to the v0.1.1
// dispute uses exactly these derived dimensions.
func GEMMNodeDims(a, b TensorDescriptor, p ParamList) (m, n, k uint64, transB bool, err error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return 0, 0, 0, false, errors.New("canonical: GEMM operands must be 2-D [M,K] and [K,N]")
	}
	transB = TransposeB(p)
	m = uint64(a.Shape[0])
	if transB {
		n = uint64(b.Shape[0])
		k = uint64(a.Shape[1])
		if b.Shape[1] != a.Shape[1] {
			return 0, 0, 0, false, errors.New("canonical: GEMM transpose_b needs B [N,K] with matching K")
		}
	} else {
		n = uint64(b.Shape[1])
		k = uint64(a.Shape[1])
		if b.Shape[0] != a.Shape[1] {
			return 0, 0, 0, false, errors.New("canonical: GEMM needs A [M,K] and B [K,N]")
		}
	}
	if m == 0 || n == 0 || k == 0 {
		return 0, 0, 0, false, errors.New("canonical: GEMM dimensions must be positive")
	}
	if k > gemmv1.MaxSafeK {
		return 0, 0, 0, false, fmt.Errorf("canonical: GEMM K=%d exceeds MaxSafeK=%d", k, gemmv1.MaxSafeK)
	}
	return m, n, k, transB, nil
}

func (gemmInt8V1) OutputSpec(in []TensorDescriptor, p ParamList) (TensorDescriptor, error) {
	if len(in) != 2 {
		return TensorDescriptor{}, errors.New("canonical: GEMM needs two inputs")
	}
	if Dtype(in[0].Dtype) != DtypeInt8 || Dtype(in[1].Dtype) != DtypeInt8 {
		return TensorDescriptor{}, errors.New("canonical: GEMM operands must be int8")
	}
	if err := (gemmInt8V1{}).ValidateParams(p); err != nil {
		return TensorDescriptor{}, err
	}
	m, n, k, _, err := GEMMNodeDims(in[0], in[1], p)
	if err != nil {
		return TensorDescriptor{}, err
	}
	_ = k
	return NewDesc(DtypeInt32Accum, int64(m), int64(n)), nil
}

func (gemmInt8V1) Work(in []Tensor, p ParamList) WorkVector {
	if len(in) != 2 {
		return WorkVector{}
	}
	m, n, k, _, err := GEMMNodeDims(in[0].Desc, in[1].Desc, p)
	if err != nil {
		return WorkVector{}
	}
	return WorkVector{}.Add("GEMM_MAC", int64(m*n*k))
}

// int8Data extracts the int8 payload; canonical tensors store every dtype
// as int32 elements, so the adapter validates the range instead of
// reinterpreting bytes.
func int8Data(t *Tensor) ([]int8, error) {
	out := make([]int8, len(t.Data))
	for i, v := range t.Data {
		if v < -128 || v > 127 {
			return nil, fmt.Errorf("canonical: int8 tensor holds out-of-range value %d at element %d", v, i)
		}
		out[i] = int8(v)
	}
	return out, nil
}

func (gemmInt8V1) Execute(in []Tensor, p ParamList) (*Tensor, error) {
	desc, err := (gemmInt8V1{}).OutputSpec([]TensorDescriptor{in[0].Desc, in[1].Desc}, p)
	if err != nil {
		return nil, err
	}
	m, n, k, transB, err := GEMMNodeDims(in[0].Desc, in[1].Desc, p)
	if err != nil {
		return nil, err
	}
	a, err := int8Data(&in[0])
	if err != nil {
		return nil, err
	}
	bt, err := int8Data(&in[1])
	if err != nil {
		return nil, err
	}
	b := bt
	if transB {
		b = make([]int8, len(bt))
		for j := uint64(0); j < n; j++ {
			for d := uint64(0); d < k; d++ {
				b[d*n+j] = bt[j*k+d]
			}
		}
	}
	c := gemmv1.ReferenceGEMM(a, b, m, n, k)
	return &Tensor{Desc: desc, Data: c}, nil
}

// ArbiterBound: GEMM nodes are adjudicated by the frozen v0.1.1 tile
// dispute (bisection down to one 8x8 micro-step); the bound below is the
// two 8x8 input tiles that micro-step touches. The canonical arbiter's
// full-operand path is reserved for off-chain verification and small
// graphs, never for the chain.
func (gemmInt8V1) ArbiterBound([]TensorDescriptor, ParamList) int {
	return 2 * gemmv1.TileSize * gemmv1.TileSize
}
