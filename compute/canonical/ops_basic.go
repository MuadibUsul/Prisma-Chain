package canonical

// Elementwise and requantize operators: the cheap, exactly-recomputable
// primitives of the block.

import (
	"errors"
)

// --- ADD_FIXED_V1: residual + branch, saturating Q12.20 add ---------------

type addFixed struct{}

func (addFixed) ID() string      { return OpAddFixedV1 }
func (addFixed) Version() string { return VersionFxFusion }

func (addFixed) ValidateParams(ParamList) error { return nil }

func (addFixed) OutputSpec(in []TensorDescriptor, _ ParamList) (TensorDescriptor, error) {
	if len(in) != 2 {
		return TensorDescriptor{}, errors.New("canonical: ADD needs two inputs")
	}
	if !sameDesc(in[0], in[1]) {
		return TensorDescriptor{}, errors.New("canonical: ADD inputs must share dtype, layout and shape (use REQUANTIZE explicitly)")
	}
	return in[0], nil
}

func (addFixed) Work(in []Tensor, _ ParamList) WorkVector {
	n := int64(len(in[0].Data))
	return WorkVector{}.Add("ADD_ELEMENT", n)
}

func (addFixed) Execute(in []Tensor, p ParamList) (*Tensor, error) {
	if _, err := (addFixed{}).OutputSpec([]TensorDescriptor{in[0].Desc, in[1].Desc}, p); err != nil {
		return nil, err
	}
	out := make([]int32, len(in[0].Data))
	for i := range out {
		out[i] = AddFx(in[0].Data[i], in[1].Data[i])
	}
	return &Tensor{Desc: in[0].Desc, Data: out}, nil
}

func (addFixed) ArbiterBound([]TensorDescriptor, ParamList) int { return 2 * ChunkElems }

// --- MUL_FIXED_V1: gating multiply with fixed rounding --------------------

type mulFixed struct{}

func (mulFixed) ID() string      { return OpMulFixedV1 }
func (mulFixed) Version() string { return VersionFxFusion }

func (mulFixed) ValidateParams(ParamList) error { return nil }

func (mulFixed) OutputSpec(in []TensorDescriptor, _ ParamList) (TensorDescriptor, error) {
	if len(in) != 2 {
		return TensorDescriptor{}, errors.New("canonical: MUL needs two inputs")
	}
	if !sameDesc(in[0], in[1]) {
		return TensorDescriptor{}, errors.New("canonical: MUL inputs must share dtype, layout and shape")
	}
	return in[0], nil
}

func (mulFixed) Work(in []Tensor, _ ParamList) WorkVector {
	return WorkVector{}.Add("MUL_ELEMENT", int64(len(in[0].Data)))
}

func (mulFixed) Execute(in []Tensor, p ParamList) (*Tensor, error) {
	if _, err := (mulFixed{}).OutputSpec([]TensorDescriptor{in[0].Desc, in[1].Desc}, p); err != nil {
		return nil, err
	}
	out := make([]int32, len(in[0].Data))
	for i := range out {
		out[i] = MulFx(in[0].Data[i], in[1].Data[i])
	}
	return &Tensor{Desc: in[0].Desc, Data: out}, nil
}

func (mulFixed) ArbiterBound([]TensorDescriptor, ParamList) int { return 2 * ChunkElems }

// --- REQUANTIZE_V1: integer rescale + clamp -------------------------------
//
// y = clamp(round_ties_even(x * mult >> shift), lo, hi) with mult/shift/lo/hi
// all integer parameters committed in the graph; runtime floats are banned
// and saturation is explicit protocol semantics. Inputs are int32
// accumulators (GEMM output) or Q12.20 activations (norm/RoPE/softmax/MUL
// output); the rescale formula is identical for both, and out_dtype picks
// whether the result continues as Q12.20 or becomes int8 GEMM operands.

type requantize struct{}

func (requantize) ID() string      { return OpRequantizeV1 }
func (requantize) Version() string { return VersionFxFusion }

// out_dtype selects the output encoding: 0 = canonical Q12.20 (default),
// 1 = int8 for the next GEMM operands. The int8 mode is what turns every
// accumulator back into a quantized activation with explicit clamps.
func (requantize) ValidateParams(p ParamList) error {
	if p.Get("mult", 0) == 0 {
		return errors.New("canonical: REQUANTIZE needs a nonzero mult")
	}
	if s := p.Get("shift", -1); s < 0 || s > 62 {
		return errors.New("canonical: REQUANTIZE shift must be in [0,62]")
	}
	if p.Get("clamp_lo", 0) >= p.Get("clamp_hi", 1) {
		return errors.New("canonical: REQUANTIZE clamp range is empty")
	}
	switch p.Get("out_dtype", 0) {
	case 0:
	case 1:
		if lo, hi := p.Get("clamp_lo", 0), p.Get("clamp_hi", 1); lo < -128 || hi > 127 {
			return errors.New("canonical: REQUANTIZE int8 output needs clamps inside [-128,127]")
		}
	default:
		return errors.New("canonical: REQUANTIZE out_dtype must be 0 (q12.20) or 1 (int8)")
	}
	return nil
}

func (requantize) OutputSpec(in []TensorDescriptor, p ParamList) (TensorDescriptor, error) {
	if len(in) != 1 {
		return TensorDescriptor{}, errors.New("canonical: REQUANTIZE needs one input")
	}
	if dt := Dtype(in[0].Dtype); dt != DtypeInt32Accum && dt != DtypeQ12_20 {
		return TensorDescriptor{}, errors.New("canonical: REQUANTIZE input must be an int32 accumulator or a q12.20 tensor")
	}
	if err := (requantize{}).ValidateParams(p); err != nil {
		return TensorDescriptor{}, err
	}
	dtype := DtypeQ12_20
	if p.Get("out_dtype", 0) == 1 {
		dtype = DtypeInt8
	}
	return TensorDescriptor{Dtype: uint8(dtype), Layout: in[0].Layout, Shape: append([]int64(nil), in[0].Shape...)}, nil
}

func (requantize) Work(in []Tensor, _ ParamList) WorkVector {
	return WorkVector{}.Add("REQUANTIZE_ELEMENT", int64(len(in[0].Data)))
}

func (requantize) Execute(in []Tensor, p ParamList) (*Tensor, error) {
	desc, err := (requantize{}).OutputSpec([]TensorDescriptor{in[0].Desc}, p)
	if err != nil {
		return nil, err
	}
	mult := p.Get("mult", 1)
	shift := uint(p.Get("shift", 0))
	lo := int32(p.Get("clamp_lo", int64(MinFx)))
	hi := int32(p.Get("clamp_hi", int64(MaxFx)))
	out := make([]int32, len(in[0].Data))
	for i, x := range in[0].Data {
		v := RShiftRoundEven(int64(x)*mult, shift)
		if v < int64(lo) {
			v = int64(lo)
		}
		if v > int64(hi) {
			v = int64(hi)
		}
		out[i] = int32(v)
	}
	return &Tensor{Desc: desc, Data: out}, nil
}

func (requantize) ArbiterBound([]TensorDescriptor, ParamList) int { return ChunkElems }
