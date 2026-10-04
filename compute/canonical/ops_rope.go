package canonical

// ROPE_FIXED_V1: rotary position embedding over element pairs with pinned
// sin/cos constants. Pair-local, so arbitration recomputes a bounded group
// of pairs from the committed constants.

import (
	"errors"
)

type ropeFixed struct{}

func (ropeFixed) ID() string      { return OpRoPEFixedV1 }
func (ropeFixed) Version() string { return VersionFxFusion }

func (ropeFixed) ValidateParams(p ParamList) error {
	if p.Get("max_pos", 0) <= 0 {
		return errors.New("canonical: ROPE needs max_pos")
	}
	if p.Get("half_dim", 0) <= 0 || p.Get("half_dim", 0)%2 != 0 {
		return errors.New("canonical: ROPE half_dim must be a positive even count of pairs")
	}
	return nil
}

// OutputSpec: the node takes the activations plus the pinned sin/cos
// constant tensor as a second (structural) input.
func (ropeFixed) OutputSpec(in []TensorDescriptor, p ParamList) (TensorDescriptor, error) {
	if len(in) != 2 {
		return TensorDescriptor{}, errors.New("canonical: ROPE needs activations and the constant table")
	}
	if err := requireQ12(in[0]); err != nil {
		return TensorDescriptor{}, err
	}
	if err := requireQ12(in[1]); err != nil {
		return TensorDescriptor{}, err
	}
	if len(in[0].Shape) < 2 {
		return TensorDescriptor{}, errors.New("canonical: ROPE input needs [positions, hidden]")
	}
	if in[0].Shape[len(in[0].Shape)-1] != p.Get("half_dim", 0)*2 {
		return TensorDescriptor{}, errors.New("canonical: ROPE hidden must equal 2*half_dim")
	}
	if in[0].Shape[len(in[0].Shape)-2] > p.Get("max_pos", 0) {
		return TensorDescriptor{}, errors.New("canonical: ROPE input exceeds max_pos")
	}
	return in[0], (ropeFixed{}).ValidateParams(p)
}

// RopeConstants holds the pinned sin/cos table: for (position, pair) the
// table stores (cos, sin) as Q12.20 int32 values. The generator
// (tools/gen_rope_table.go) writes it with high-precision math; the bytes
// are committed in the graph as a constant tensor and hashed with it.
type RopeConstants struct {
	Table []int32
	Pairs int
}

// NewRopeConstants builds from a flat (cos, sin) table holding
// positions x pairs entries: len(table) must be a positive multiple of
// pairs*2, where pairs is the number of element pairs per row.
func NewRopeConstants(table []int32, pairs int) (*RopeConstants, error) {
	if pairs <= 0 || len(table) == 0 || len(table)%(pairs*2) != 0 {
		return nil, errors.New("canonical: malformed RoPE table")
	}
	return &RopeConstants{Table: table, Pairs: pairs}, nil
}

func (ropeFixed) Work(in []Tensor, _ ParamList) WorkVector {
	pairs := int64(len(in[0].Data) / 2)
	return WorkVector{}.Add("ROPE_PAIR", pairs)
}

// ExecuteRope applies the rotation with the given constants; Execute on
// the operator uses the constants attached to the graph, which the graph
// layer resolves and passes here.
func ExecuteRope(in *Tensor, constants *RopeConstants, p ParamList) (*Tensor, error) {
	tableDesc := NewDesc(DtypeQ12_20, int64(len(constants.Table)))
	if _, err := (ropeFixed{}).OutputSpec([]TensorDescriptor{in.Desc, tableDesc}, p); err != nil {
		return nil, err
	}
	hidden := int(in.Desc.Shape[len(in.Desc.Shape)-1])
	positions := len(in.Data) / hidden
	pairsPerRow := hidden / 2
	out := make([]int32, len(in.Data))
	for pos := 0; pos < positions; pos++ {
		for pair := 0; pair < pairsPerRow; pair++ {
			cosFx, sinFx, ok := SinCosFx(constants.Table, pos, pair, pairsPerRow)
			if !ok {
				return nil, errors.New("canonical: RoPE position out of table range")
			}
			even := in.Data[pos*hidden+2*pair]
			odd := in.Data[pos*hidden+2*pair+1]
			out[pos*hidden+2*pair] = SubFx(MulFx(even, cosFx), MulFx(odd, sinFx))
			out[pos*hidden+2*pair+1] = AddFx(MulFx(even, sinFx), MulFx(odd, cosFx))
		}
	}
	return &Tensor{Desc: in.Desc, Data: out}, nil
}

func (ropeFixed) Execute(in []Tensor, p ParamList) (*Tensor, error) {
	return nil, errors.New("canonical: ROPE requires graph constants; use ExecuteRope")
}

func (ropeFixed) ArbiterBound([]TensorDescriptor, ParamList) int {
	// One chunk of pairs: every output element depends on exactly one input
	// pair plus committed constants.
	return ChunkElems
}
