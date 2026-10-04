package canonical

// RMSNORM_FIXED_V1, SILU_FIXED_V1 and the row-local SOFTMAX_FIXED_V1: the
// nonlinear operators, all built on CanonicalMathV1 primitives.

import (
	"errors"
	"fmt"
)

// --- RMSNORM_FIXED_V1 ------------------------------------------------------
//
// rms = invsqrt(mean(x_i^2) + eps); y_i = x_i * rms * w_i.
// The sum-of-squares reduction is chunked (NORM_CHUNK_SIZE = 16) and can
// be traced for a bounded stage dispute; normalization is output-local
// given the agreed rms.

const normChunk = 16

type rmsNormFixed struct{}

func (rmsNormFixed) ID() string      { return OpRMSNormFixedV1 }
func (rmsNormFixed) Version() string { return VersionFxFusion }

func (rmsNormFixed) ValidateParams(p ParamList) error {
	if p.Get("eps_fx", 0) <= 0 {
		return errors.New("canonical: RMSNORM needs a positive eps_fx")
	}
	if c := p.Get("chunk", normChunk); c != normChunk {
		return fmt.Errorf("canonical: RMSNORM chunk must be %d", normChunk)
	}
	return nil
}

func shapeElems(shape []int64) int64 {
	total := int64(1)
	for _, dim := range shape {
		total *= dim
	}
	return total
}

func (rmsNormFixed) OutputSpec(in []TensorDescriptor, p ParamList) (TensorDescriptor, error) {
	if len(in) != 2 {
		return TensorDescriptor{}, errors.New("canonical: RMSNORM needs x and weight")
	}
	if err := requireQ12(in[0]); err != nil {
		return TensorDescriptor{}, err
	}
	if err := requireQ12(in[1]); err != nil {
		return TensorDescriptor{}, err
	}
	if shapeElems(in[1].Shape) != in[0].Shape[len(in[0].Shape)-1] {
		return TensorDescriptor{}, errors.New("canonical: RMSNORM weight must match the last dimension")
	}
	return in[0], (rmsNormFixed{}).ValidateParams(p)
}

// rmsFor computes the canonical rms of one row.
func rmsFor(row []int32, epsFx int32) (int32, error) {
	var sum int64
	for _, v := range row {
		prod := int64(v) * int64(v)
		sum += prod >> FracBits
	}
	mean := sum / int64(len(row))
	ms := mean + int64(epsFx)
	inv, err := InvSqrtFx(ms)
	if err != nil {
		return 0, err
	}
	return inv, nil
}

// rowStageStates is the traced sum-of-squares stage: state k is the Qf
// sum-of-squares contribution of the first k chunks.
func rowStageStates(row []int32) []int64 {
	chunks := (len(row) + normChunk - 1) / normChunk
	states := make([]int64, chunks+1)
	var sum int64
	for c := 0; c < chunks; c++ {
		for i := c * normChunk; i < (c+1)*normChunk && i < len(row); i++ {
			sum += int64(row[i]) * int64(row[i])
		}
		states[c+1] = sum
	}
	return states
}

func (rmsNormFixed) Work(in []Tensor, _ ParamList) WorkVector {
	n := int64(len(in[0].Data))
	return WorkVector{}.Add("RMSNORM_ELEMENT", n).Add("RMSNORM_REDUCTION", n)
}

func (rmsNormFixed) Execute(in []Tensor, p ParamList) (*Tensor, error) {
	if _, err := (rmsNormFixed{}).OutputSpec([]TensorDescriptor{in[0].Desc, in[1].Desc}, p); err != nil {
		return nil, err
	}
	epsFx := int32(p.Get("eps_fx", 0))
	hidden := int(in[0].Desc.Shape[len(in[0].Desc.Shape)-1])
	rows := len(in[0].Data) / hidden
	out := make([]int32, len(in[0].Data))
	for r := 0; r < rows; r++ {
		row := in[0].Data[r*hidden : (r+1)*hidden]
		inv, err := rmsFor(row, epsFx)
		if err != nil {
			return nil, err
		}
		for i := 0; i < hidden; i++ {
			out[r*hidden+i] = MulFx(MulFx(row[i], inv), in[1].Data[i%len(in[1].Data)])
		}
	}
	return &Tensor{Desc: in[0].Desc, Data: out}, nil
}

func (rmsNormFixed) ArbiterBound(in []TensorDescriptor, _ ParamList) int {
	// The stage costs one 16-element reduction chunk; normalization costs
	// one 64-element output chunk (x and weight chunks).
	return normChunk + 2*ChunkElems
}

// --- SILU_FIXED_V1: y = x * sigmoid(x) -------------------------------------

type siluFixed struct{}

func (siluFixed) ID() string      { return OpSiLUFixedV1 }
func (siluFixed) Version() string { return VersionFxFusion }

func (siluFixed) ValidateParams(ParamList) error { return nil }

func (siluFixed) OutputSpec(in []TensorDescriptor, _ ParamList) (TensorDescriptor, error) {
	if len(in) != 1 {
		return TensorDescriptor{}, errors.New("canonical: SILU needs one input")
	}
	if err := requireQ12(in[0]); err != nil {
		return TensorDescriptor{}, err
	}
	return in[0], nil
}

func (siluFixed) Work(in []Tensor, _ ParamList) WorkVector {
	return WorkVector{}.Add("SILU_ELEMENT", int64(len(in[0].Data)))
}

func (siluFixed) Execute(in []Tensor, p ParamList) (*Tensor, error) {
	if _, err := (siluFixed{}).OutputSpec([]TensorDescriptor{in[0].Desc}, p); err != nil {
		return nil, err
	}
	out := make([]int32, len(in[0].Data))
	for i, x := range in[0].Data {
		out[i] = MulFx(x, SigmoidFx(x))
	}
	return &Tensor{Desc: in[0].Desc, Data: out}, nil
}

func (siluFixed) ArbiterBound([]TensorDescriptor, ParamList) int { return ChunkElems }

// --- SOFTMAX_FIXED_V1 -------------------------------------------------------
//
// Canonical row pipeline: row max, subtract max, canonical exp, sum,
// normalize. max and sum reductions are chunked (16) and traceable as two
// bounded stages; normalization is output-local given agreed max and sum.

type softmaxFixed struct{}

func (softmaxFixed) ID() string      { return OpSoftmaxFixedV1 }
func (softmaxFixed) Version() string { return VersionFxFusion }

func (softmaxFixed) ValidateParams(p ParamList) error {
	if c := p.Get("chunk", normChunk); c != normChunk {
		return fmt.Errorf("canonical: SOFTMAX chunk must be %d", normChunk)
	}
	return nil
}

func (softmaxFixed) OutputSpec(in []TensorDescriptor, p ParamList) (TensorDescriptor, error) {
	if len(in) != 1 {
		return TensorDescriptor{}, errors.New("canonical: SOFTMAX needs one input")
	}
	if err := requireQ12(in[0]); err != nil {
		return TensorDescriptor{}, err
	}
	return in[0], (softmaxFixed{}).ValidateParams(p)
}

// softmaxRow returns the canonical probabilities of one row.
func softmaxRow(row []int32) ([]int32, error) {
	max := row[0]
	for _, v := range row[1:] {
		if v > max {
			max = v
		}
	}
	exps := make([]int32, len(row))
	var sum int64
	for i, v := range row {
		exps[i] = ExpFx(SubFx(v, max))
		sum += int64(exps[i])
	}
	if sum == 0 {
		sum = 1
	}
	out := make([]int32, len(row))
	for i, e := range exps {
		q := (int64(e) << FracBits) / sum
		r := (int64(e) << FracBits) % sum
		if r != 0 {
			two := 2 * abs64(r)
			if two > sum || (two == sum && q&1 == 1) {
				q++
			}
		}
		out[i] = Saturate(q)
	}
	return out, nil
}

func (softmaxFixed) Work(in []Tensor, _ ParamList) WorkVector {
	n := int64(len(in[0].Data))
	return WorkVector{}.Add("SOFTMAX_ELEMENT", n).Add("SOFTMAX_EXP", n)
}

func (softmaxFixed) Execute(in []Tensor, p ParamList) (*Tensor, error) {
	if _, err := (softmaxFixed{}).OutputSpec([]TensorDescriptor{in[0].Desc}, p); err != nil {
		return nil, err
	}
	hidden := int(in[0].Desc.Shape[len(in[0].Desc.Shape)-1])
	rows := len(in[0].Data) / hidden
	out := make([]int32, len(in[0].Data))
	for r := 0; r < rows; r++ {
		probs, err := softmaxRow(in[0].Data[r*hidden : (r+1)*hidden])
		if err != nil {
			return nil, err
		}
		copy(out[r*hidden:(r+1)*hidden], probs)
	}
	return &Tensor{Desc: in[0].Desc, Data: out}, nil
}

func (softmaxFixed) ArbiterBound(in []TensorDescriptor, _ ParamList) int {
	// One 16-element max chunk OR one 16-element exp-sum chunk OR one
	// 64-element normalization chunk.
	return ChunkElems + normChunk
}
