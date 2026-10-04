package canonical

// TRANSFORMER_BLOCK_V1: the canonical Transformer block as a graph MACRO.
//
// Attention and the SwiGLU MLP are NOT black-box operators: they expand
// into the static operator set above (GEMM_INT8_V1, REQUANTIZE_V1,
// ROPE_FIXED_V1, SOFTMAX_FIXED_V1, SILU_FIXED_V1, MUL_FIXED_V1,
// ADD_FIXED_V1, RMSNORM_FIXED_V1). Everything the block needs that the
// frozen GEMM layout cannot express directly is resolved at build time by
// the converter, never by runtime views or strided tensors:
//
//   - heads are separate weight slices (Wq_h, Wk_h, ... are committed
//     inputs), so no runtime slicing of activations exists;
//   - the head-concatenation output projection is the mathematically
//     equivalent sum over heads of Y_h = O_h · Wo_h (linearity);
//   - scores = Q · K^T uses the GEMM node's transpose_b form;
//   - the 1/sqrt(head_dim) attention scale is folded into the scores'
//     REQUANTIZE parameters (statically, by the converter);
//   - activations are statically quantized (fixed per-stage scales chosen
//     at conversion time, as in classic W8A8); dynamic per-input scales
//     are NOT part of v1 and are documented as such.
//
// The block is pre-norm: y = x + Attn(Norm(x)), out = y + MLP(Norm(y)).

import (
	"errors"
	"fmt"
	"math/big"
)

// SpecTransformerBlockV1 is the graph spec string of the block macro.
const SpecTransformerBlockV1 = "TRANSFORMER_BLOCK_V1"

// BlockConfig fixes the static shape of one block.
type BlockConfig struct {
	Seq       int `gemm:"seq"`
	DModel    int `gemm:"d_model"`
	Heads     int `gemm:"heads"`
	HeadDim   int `gemm:"head_dim"`
	MLPHidden int `gemm:"mlp_hidden"`
}

// Validate rejects unsupported shapes. HeadDim must be a multiple of 4
// (even pair count for RoPE, chunk-friendly rows).
func (c BlockConfig) Validate() error {
	if c.Seq <= 0 || c.DModel <= 0 || c.Heads <= 0 || c.HeadDim <= 0 || c.MLPHidden <= 0 {
		return errors.New("canonical: block dimensions must be positive")
	}
	if c.Heads*c.HeadDim != c.DModel {
		return errors.New("canonical: heads * head_dim must equal d_model")
	}
	if c.HeadDim%4 != 0 {
		return errors.New("canonical: head_dim must be a multiple of 4 (even RoPE pairs)")
	}
	if c.DModel%normChunk != 0 || c.HeadDim%normChunk != 0 || c.Seq%normChunk != 0 {
		return errors.New("canonical: block rows must be multiples of the reduction chunk")
	}
	// Row-operator arbitration needs row boundaries on chunk boundaries:
	// every softmax/norm row length must divide ChunkElems or be a
	// multiple of it.
	aligned := func(n int) bool { return n%ChunkElems == 0 || ChunkElems%n == 0 }
	if !aligned(c.DModel) || !aligned(c.Seq) {
		return errors.New("canonical: block rows must divide or be a multiple of the tensor chunk size")
	}
	return nil
}

// RequantSpec is one REQUANTIZE node's committed parameters.
type RequantSpec struct {
	Mult    int64
	Shift   int64
	Lo      int64
	Hi      int64
	OutInt8 bool
}

// Params renders the canonical parameter list.
func (r RequantSpec) Params() ParamList {
	out := ParamList{
		{Key: "clamp_hi", Value: r.Hi},
		{Key: "clamp_lo", Value: r.Lo},
		{Key: "mult", Value: r.Mult},
		{Key: "shift", Value: r.Shift},
	}
	if r.OutInt8 {
		out = append(out, Pair{Key: "out_dtype", Value: 1})
	}
	return out.Canonical()
}

// BlockQuant is the converter's static quantization plan: one entry per
// REQUANTIZE site of the block. Values are chosen at conversion time from
// calibration data and are committed inside GraphID.
type BlockQuant struct {
	NormEpsFx int64 // RMSNORM eps_fx (positive)

	Norm1ToInt8 RequantSpec // attn-norm output -> int8 operands
	Norm2ToInt8 RequantSpec // mlp-norm output -> int8 operands

	QKAccumToFx  RequantSpec // Q/K accumulator -> q12.20 (pre-RoPE)
	QKRopeToInt8 RequantSpec // Q/K after RoPE -> int8 operands
	VAccumToInt8 RequantSpec // V accumulator -> int8 operands

	ScoresAccumToFx RequantSpec // scores accumulator -> q12.20, attention scale folded
	SoftmaxToInt8   RequantSpec // probabilities -> int8 operands
	CtxAccumToInt8  RequantSpec // context accumulator -> int8 operands
	ProjAccumToFx   RequantSpec // output projection accumulator -> q12.20

	GateAccumToFx RequantSpec // gate projection accumulator -> q12.20
	UpAccumToFx   RequantSpec // up projection accumulator -> q12.20
	MlpMulToInt8  RequantSpec // SwiGLU product -> int8 operands
	DownAccumToFx RequantSpec // down projection accumulator -> q12.20
}

// BlockInputSet carries the committed roots of every block input. Names
// are assigned by the builder (role names are part of the hash); a
// non-empty caller name must match the role name.
type BlockInputSet struct {
	X         GraphInput
	Norm1     GraphInput // attention RMSNorm weight [d_model] q12.20
	Norm2     GraphInput // MLP RMSNorm weight [d_model] q12.20
	RopeTable GraphInput // pinned (cos, sin) table, Seq*HeadDim q12.20 values

	WQ []GraphInput // per head [DModel, HeadDim] int8
	WK []GraphInput
	WV []GraphInput
	WO []GraphInput // per head [HeadDim, DModel] int8

	WGate GraphInput // [DModel, MLPHidden] int8
	WUp   GraphInput // [DModel, MLPHidden] int8
	WDown GraphInput // [MLPHidden, DModel] int8
}

// AttentionScaleFx returns floor(2^20 / sqrt(headDim)) with integer-only
// arithmetic: the converter multiplies it into the scores' dequant
// MULT/SHIFT to fold 1/sqrt(headDim) into REQUANTIZE (no float, no new
// operator).
func AttentionScaleFx(headDim int) (int64, error) {
	if headDim <= 0 {
		return 0, errors.New("canonical: head_dim must be positive")
	}
	// floor(sqrt(headDim) * 2^20) via big.Int sqrt, then floor(2^40 / s).
	s := new(big.Int).Lsh(big.NewInt(int64(headDim)), 40)
	s.Sqrt(s)
	if s.Sign() == 0 {
		return 0, errors.New("canonical: degenerate attention scale")
	}
	num := new(big.Int).Lsh(big.NewInt(1), 40)
	return new(big.Int).Div(num, s).Int64(), nil
}

type blockBuilder struct {
	g     *GraphDescriptor
	cfg   BlockConfig
	quant BlockQuant

	x, n1, n2, rope   int
	wq, wk, wv, wo    []int
	wGate, wUp, wDown int
}

func (b *blockBuilder) addInput(role string, in GraphInput, dtype Dtype, shape ...int64) (int, error) {
	if in.Name != "" && in.Name != role {
		return 0, fmt.Errorf("canonical: input %q carries an unexpected name %q", role, in.Name)
	}
	if err := in.Desc.Validate(); err != nil {
		return 0, fmt.Errorf("canonical: input %q: %w", role, err)
	}
	if Dtype(in.Desc.Dtype) != dtype {
		return 0, fmt.Errorf("canonical: input %q has dtype %d, want %d", role, in.Desc.Dtype, dtype)
	}
	if len(in.Desc.Shape) != len(shape) {
		return 0, fmt.Errorf("canonical: input %q rank mismatch", role)
	}
	for i := range shape {
		if in.Desc.Shape[i] != shape[i] {
			return 0, fmt.Errorf("canonical: input %q shape mismatch at dim %d", role, i)
		}
	}
	if len(in.Root) != 32 {
		return 0, fmt.Errorf("canonical: input %q needs a 32-byte committed root", role)
	}
	idx := len(b.g.Inputs)
	b.g.Inputs = append(b.g.Inputs, GraphInput{Name: role, Desc: in.Desc, Root: append([]byte(nil), in.Root...)})
	return idx, nil
}

func (b *blockBuilder) descOf(ref TensorRef) (TensorDescriptor, error) {
	switch ref.Kind {
	case 0:
		if int(ref.Index) >= len(b.g.Inputs) {
			return TensorDescriptor{}, errors.New("canonical: input ref out of range")
		}
		return b.g.Inputs[ref.Index].Desc, nil
	case 1:
		if int(ref.Index) >= len(b.g.Nodes) {
			return TensorDescriptor{}, errors.New("canonical: node ref out of range")
		}
		return b.g.Nodes[ref.Index].Output, nil
	default:
		return TensorDescriptor{}, errors.New("canonical: unknown tensor ref kind")
	}
}

// node appends one operator application, deriving its output descriptor
// from the operator contract (never from the caller).
func (b *blockBuilder) node(opID string, inputs []TensorRef, params ParamList) (TensorRef, error) {
	op, err := Lookup(opID)
	if err != nil {
		return TensorRef{}, err
	}
	descs := make([]TensorDescriptor, 0, len(inputs))
	for _, ref := range inputs {
		desc, err := b.descOf(ref)
		if err != nil {
			return TensorRef{}, err
		}
		descs = append(descs, desc)
	}
	out, err := op.OutputSpec(descs, params)
	if err != nil {
		return TensorRef{}, fmt.Errorf("canonical: %s node: %w", opID, err)
	}
	id := uint32(len(b.g.Nodes))
	b.g.Nodes = append(b.g.Nodes, GraphNode{
		NodeID:     id,
		OperatorID: opID,
		Version:    op.Version(),
		Inputs:     inputs,
		Output:     out,
		Params:     params.Canonical(),
	})
	return TensorRef{Kind: 1, Index: id}, nil
}

func inRef(idx int) TensorRef     { return TensorRef{Kind: 0, Index: uint32(idx)} }
func nodeRef(id uint32) TensorRef { return TensorRef{Kind: 1, Index: id} }

// BuildTransformerBlockV1 expands the block macro into a validated
// CANONICAL_GRAPH_V1 descriptor. The returned graph is deterministic: the
// same config, roots and quant plan produce the same node order and the
// same GraphID.
func BuildTransformerBlockV1(cfg BlockConfig, in BlockInputSet, quant BlockQuant) (*GraphDescriptor, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if quant.NormEpsFx <= 0 {
		return nil, errors.New("canonical: block needs a positive RMSNORM eps_fx")
	}
	if len(in.WQ) != cfg.Heads || len(in.WK) != cfg.Heads || len(in.WV) != cfg.Heads || len(in.WO) != cfg.Heads {
		return nil, errors.New("canonical: per-head weight lists must match the head count")
	}
	b := &blockBuilder{
		g:   &GraphDescriptor{ProtocolVersion: GraphProtocolVersion, Spec: SpecTransformerBlockV1},
		cfg: cfg, quant: quant,
	}
	var err error
	soft := func(f func() error) {
		if err == nil {
			err = f()
		}
	}

	// --- graph inputs in the fixed builder order ---------------------------
	soft(func() error {
		b.x, err = b.addInput("x", in.X, DtypeQ12_20, int64(cfg.Seq), int64(cfg.DModel))
		return err
	})
	soft(func() error {
		b.n1, err = b.addInput("w_attn_norm", in.Norm1, DtypeQ12_20, int64(cfg.DModel))
		return err
	})
	soft(func() error {
		b.n2, err = b.addInput("w_mlp_norm", in.Norm2, DtypeQ12_20, int64(cfg.DModel))
		return err
	})
	soft(func() error {
		b.rope, err = b.addInput("rope_table", in.RopeTable, DtypeQ12_20, int64(cfg.Seq*cfg.HeadDim))
		return err
	})
	b.wq = make([]int, cfg.Heads)
	b.wk = make([]int, cfg.Heads)
	b.wv = make([]int, cfg.Heads)
	b.wo = make([]int, cfg.Heads)
	for h := 0; h < cfg.Heads; h++ {
		h := h
		soft(func() error {
			b.wq[h], err = b.addInput(fmt.Sprintf("wq_h%d", h), in.WQ[h], DtypeInt8, int64(cfg.DModel), int64(cfg.HeadDim))
			return err
		})
		soft(func() error {
			b.wk[h], err = b.addInput(fmt.Sprintf("wk_h%d", h), in.WK[h], DtypeInt8, int64(cfg.DModel), int64(cfg.HeadDim))
			return err
		})
		soft(func() error {
			b.wv[h], err = b.addInput(fmt.Sprintf("wv_h%d", h), in.WV[h], DtypeInt8, int64(cfg.DModel), int64(cfg.HeadDim))
			return err
		})
		soft(func() error {
			b.wo[h], err = b.addInput(fmt.Sprintf("wo_h%d", h), in.WO[h], DtypeInt8, int64(cfg.HeadDim), int64(cfg.DModel))
			return err
		})
	}
	soft(func() error {
		b.wGate, err = b.addInput("w_gate", in.WGate, DtypeInt8, int64(cfg.DModel), int64(cfg.MLPHidden))
		return err
	})
	soft(func() error {
		b.wUp, err = b.addInput("w_up", in.WUp, DtypeInt8, int64(cfg.DModel), int64(cfg.MLPHidden))
		return err
	})
	soft(func() error {
		b.wDown, err = b.addInput("w_down", in.WDown, DtypeInt8, int64(cfg.MLPHidden), int64(cfg.DModel))
		return err
	})
	if err != nil {
		return nil, err
	}

	req := func(ref TensorRef, spec RequantSpec) (TensorRef, error) {
		return b.node(OpRequantizeV1, []TensorRef{ref}, spec.Params())
	}
	ropeParams := ParamList{
		{Key: "half_dim", Value: int64(cfg.HeadDim / 2)},
		{Key: "max_pos", Value: int64(cfg.Seq)},
	}
	transB := ParamList{{Key: "transpose_b", Value: 1}}

	// --- attention: Norm -> quantized Q/K/V --------------------------------
	xn, err := b.node(OpRMSNormFixedV1, []TensorRef{inRef(b.x), inRef(b.n1)}, ParamList{{Key: "eps_fx", Value: quant.NormEpsFx}})
	if err != nil {
		return nil, err
	}
	xq, err := req(xn, quant.Norm1ToInt8)
	if err != nil {
		return nil, err
	}
	headOuts := make([]TensorRef, cfg.Heads)
	for h := 0; h < cfg.Heads; h++ {
		// Q
		qacc, err := b.node(OpGEMMInt8V1, []TensorRef{xq, inRef(b.wq[h])}, nil)
		if err != nil {
			return nil, err
		}
		qfx, err := req(qacc, quant.QKAccumToFx)
		if err != nil {
			return nil, err
		}
		qr, err := b.node(OpRoPEFixedV1, []TensorRef{qfx, inRef(b.rope)}, ropeParams)
		if err != nil {
			return nil, err
		}
		q8, err := req(qr, quant.QKRopeToInt8)
		if err != nil {
			return nil, err
		}
		// K
		kacc, err := b.node(OpGEMMInt8V1, []TensorRef{xq, inRef(b.wk[h])}, nil)
		if err != nil {
			return nil, err
		}
		kfx, err := req(kacc, quant.QKAccumToFx)
		if err != nil {
			return nil, err
		}
		kr, err := b.node(OpRoPEFixedV1, []TensorRef{kfx, inRef(b.rope)}, ropeParams)
		if err != nil {
			return nil, err
		}
		k8, err := req(kr, quant.QKRopeToInt8)
		if err != nil {
			return nil, err
		}
		// V
		vacc, err := b.node(OpGEMMInt8V1, []TensorRef{xq, inRef(b.wv[h])}, nil)
		if err != nil {
			return nil, err
		}
		v8, err := req(vacc, quant.VAccumToInt8)
		if err != nil {
			return nil, err
		}
		// scores = Q . K^T (transpose_b), attention scale folded in REQ
		sacc, err := b.node(OpGEMMInt8V1, []TensorRef{q8, k8}, transB)
		if err != nil {
			return nil, err
		}
		sfx, err := req(sacc, quant.ScoresAccumToFx)
		if err != nil {
			return nil, err
		}
		probs, err := b.node(OpSoftmaxFixedV1, []TensorRef{sfx}, nil)
		if err != nil {
			return nil, err
		}
		p8, err := req(probs, quant.SoftmaxToInt8)
		if err != nil {
			return nil, err
		}
		// context = P . V
		cacc, err := b.node(OpGEMMInt8V1, []TensorRef{p8, v8}, nil)
		if err != nil {
			return nil, err
		}
		c8, err := req(cacc, quant.CtxAccumToInt8)
		if err != nil {
			return nil, err
		}
		// per-head output projection: Y_h = O_h . Wo_h
		oacc, err := b.node(OpGEMMInt8V1, []TensorRef{c8, inRef(b.wo[h])}, nil)
		if err != nil {
			return nil, err
		}
		ofx, err := req(oacc, quant.ProjAccumToFx)
		if err != nil {
			return nil, err
		}
		headOuts[h] = ofx
	}
	// head concatenation by linearity: sum of per-head projections
	headSum := headOuts[0]
	for h := 1; h < cfg.Heads; h++ {
		headSum, err = b.node(OpAddFixedV1, []TensorRef{headSum, headOuts[h]}, nil)
		if err != nil {
			return nil, err
		}
	}
	x1, err := b.node(OpAddFixedV1, []TensorRef{inRef(b.x), headSum}, nil)
	if err != nil {
		return nil, err
	}

	// --- SwiGLU MLP ---------------------------------------------------------
	xm, err := b.node(OpRMSNormFixedV1, []TensorRef{x1, inRef(b.n2)}, ParamList{{Key: "eps_fx", Value: quant.NormEpsFx}})
	if err != nil {
		return nil, err
	}
	xmq, err := req(xm, quant.Norm2ToInt8)
	if err != nil {
		return nil, err
	}
	gacc, err := b.node(OpGEMMInt8V1, []TensorRef{xmq, inRef(b.wGate)}, nil)
	if err != nil {
		return nil, err
	}
	gfx, err := req(gacc, quant.GateAccumToFx)
	if err != nil {
		return nil, err
	}
	gs, err := b.node(OpSiLUFixedV1, []TensorRef{gfx}, nil)
	if err != nil {
		return nil, err
	}
	uacc, err := b.node(OpGEMMInt8V1, []TensorRef{xmq, inRef(b.wUp)}, nil)
	if err != nil {
		return nil, err
	}
	ufx, err := req(uacc, quant.UpAccumToFx)
	if err != nil {
		return nil, err
	}
	hprod, err := b.node(OpMulFixedV1, []TensorRef{gs, ufx}, nil)
	if err != nil {
		return nil, err
	}
	hq, err := req(hprod, quant.MlpMulToInt8)
	if err != nil {
		return nil, err
	}
	dacc, err := b.node(OpGEMMInt8V1, []TensorRef{hq, inRef(b.wDown)}, nil)
	if err != nil {
		return nil, err
	}
	dfx, err := req(dacc, quant.DownAccumToFx)
	if err != nil {
		return nil, err
	}
	y, err := b.node(OpAddFixedV1, []TensorRef{x1, dfx}, nil)
	if err != nil {
		return nil, err
	}
	b.g.Outputs = []TensorRef{y}
	if err := b.g.Validate(); err != nil {
		return nil, err
	}
	return b.g, nil
}
