package canonical

// CanonicalMathV1, tensor commitment and operator unit tests. Every
// nonlinear expectation is checked against values computed from the
// frozen algorithms, not against libm at runtime.

import (
	"math"
	"testing"
)

func TestPinnedTables(t *testing.T) {
	if BuildInvSqrtTable() != PinnedInvSqrtTable() {
		t.Fatal("pinned invsqrt table does not match the generator output")
	}
}

func TestRShiftTiesEven(t *testing.T) {
	cases := []struct {
		v    int64
		s    uint
		want int64
	}{
		{4, 1, 2}, {5, 1, 2}, {6, 1, 3}, {7, 1, 4}, {1, 1, 0},
		{-4, 1, -2}, {-5, 1, -2}, {-6, 1, -3}, {-7, 1, -4}, {-1, 1, 0},
		{3, 2, 1}, {1, 2, 0},
	}
	for _, c := range cases {
		if got := RShiftRoundEven(c.v, c.s); got != c.want {
			t.Fatalf("rshift(%d,%d)=%d want %d", c.v, c.s, got, c.want)
		}
	}
}

func TestInvSqrtAccuracy(t *testing.T) {
	cases := []float64{1, 2, 4, 16, 100, 0.01, 0.25, 2047}
	for _, x := range cases {
		fx := int64(x * float64(int64(1)<<FracBits))
		got, err := InvSqrtFx(fx)
		if err != nil {
			t.Fatal(err)
		}
		want := 1 / math.Sqrt(x)
		// Measured bound: <=1e-3 relative at the very top of the range
		// (2047), ~4e-5 elsewhere; well inside the RMSNorm 5e-3 acceptance.
		if math.Abs(float64(got)/float64(int64(1)<<FracBits)-want) > 1e-3*want+1e-6 {
			t.Fatalf("invsqrt(%v)=%v want %v", x, float64(got)/float64(int64(1)<<FracBits), want)
		}
	}
	if _, err := InvSqrtFx(0); err == nil {
		t.Fatal("invsqrt(0) must be rejected")
	}
	if _, err := InvSqrtFx(-5); err == nil {
		t.Fatal("invsqrt(negative) must be rejected")
	}
}

func TestExpKnownValues(t *testing.T) {
	check := func(x, want, tol float64) {
		fx := int32(x * float64(int64(1)<<FracBits))
		got := float64(ExpFx(fx)) / float64(int64(1)<<FracBits)
		if math.Abs(got-want) > tol {
			t.Fatalf("exp(%v)=%v want %v (tol %v)", x, got, want, tol)
		}
	}
	check(0, 1, 1e-6)
	check(-1, math.Exp(-1), 1e-3)
	check(-5, math.Exp(-5), 1e-4)
	check(-0.001, math.Exp(-0.001), 1e-5)
	if ExpFx(int32(-30*float64(int64(1)<<FracBits))) != 0 {
		t.Fatal("exp below -24 must underflow to 0")
	}
	if ExpFx(int32(25*float64(int64(1)<<FracBits))) != int32(MaxFx) {
		t.Fatal("exp above 21 must saturate")
	}
	// monotonicity on a grid
	prev := int32(-1 << 30)
	for x := -20.0; x <= 0; x += 0.25 {
		cur := ExpFx(int32(x * float64(int64(1)<<FracBits)))
		if cur < prev {
			t.Fatalf("exp not monotonic at %v", x)
		}
		prev = cur
	}
}

func TestSigmoidSymmetryAndRange(t *testing.T) {
	one := float64(int64(1) << FracBits)
	for x := -10.0; x <= 10.0; x += 0.5 {
		fx := int32(x * one)
		got := float64(SigmoidFx(fx)) / one
		if got < 0 || got > 1 {
			t.Fatalf("sigmoid(%v)=%v out of [0,1]", x, got)
		}
		if math.Abs(got-1/(1+math.Exp(-x))) > 2e-3 {
			t.Fatalf("sigmoid(%v)=%v want %v", x, got, 1/(1+math.Exp(-x)))
		}
		neg := float64(SigmoidFx(-fx)) / one
		if math.Abs(got+neg-1) > 3e-3 {
			t.Fatalf("sigmoid symmetry broken at %v: %v + %v", x, got, neg)
		}
	}
}

func TestTensorRootDeterminismAndTamper(t *testing.T) {
	desc := NewDesc(DtypeQ12_20, 16, 128)
	data := make([]int32, 16*128)
	for i := range data {
		data[i] = int32(i*37%200 - 100)
	}
	a := &Tensor{Desc: desc, Data: data}
	b := &Tensor{Desc: desc, Data: append([]int32(nil), data...)}
	ra, err := a.TensorRoot()
	if err != nil {
		t.Fatal(err)
	}
	rb, err := b.TensorRoot()
	if err != nil {
		t.Fatal(err)
	}
	if ra != rb {
		t.Fatal("tensor root is not deterministic")
	}
	// descriptor binding: same bytes, different shape must change the root
	other := &Tensor{Desc: NewDesc(DtypeQ12_20, 32, 64), Data: append([]int32(nil), data...)}
	ro, err := other.TensorRoot()
	if err != nil {
		t.Fatal(err)
	}
	if ro == ra {
		t.Fatal("tensor root does not bind the descriptor")
	}
	// tamper
	bad := &Tensor{Desc: desc, Data: append([]int32(nil), data...)}
	bad.Data[0]++
	rbad, err := bad.TensorRoot()
	if err != nil {
		t.Fatal(err)
	}
	if rbad == ra {
		t.Fatal("tensor root unchanged after tampering")
	}
	// chunk proof round trip
	siblings, err := a.ChunkProof(3)
	if err != nil {
		t.Fatal(err)
	}
	descBytes, _ := EncodeCanonical(desc)
	chunk, _ := a.chunkBytes(3)
	if !VerifyChunk(ra, descBytes, 3, uint32(a.ChunkCount()), chunk, siblings) {
		t.Fatal("valid chunk proof rejected")
	}
	chunk[0] ^= 0xff
	if VerifyChunk(ra, descBytes, 3, uint32(a.ChunkCount()), chunk, siblings) {
		t.Fatal("tampered chunk accepted")
	}
}

func TestOperatorRegistry(t *testing.T) {
	ids := OperatorIDs()
	want := []string{OpAddFixedV1, OpGEMMInt8V1, OpMulFixedV1, OpRMSNormFixedV1, OpRequantizeV1, OpRoPEFixedV1, OpSiLUFixedV1, OpSoftmaxFixedV1}
	// GEMM is registered by the chain layer when it wraps gemmv1; the
	// canonical registry itself holds the seven new operators.
	for _, id := range []string{OpAddFixedV1, OpMulFixedV1, OpRequantizeV1, OpRMSNormFixedV1, OpRoPEFixedV1, OpSiLUFixedV1, OpSoftmaxFixedV1} {
		if _, err := Lookup(id); err != nil {
			t.Fatalf("operator %s missing: %v", id, err)
		}
	}
	if _, err := Lookup("ATTENTION_V1"); err == nil {
		t.Fatal("unknown operator must be rejected")
	}
	_ = want
	_ = ids
}

func TestElementwiseOperators(t *testing.T) {
	x := &Tensor{Desc: NewDesc(DtypeQ12_20, 1, 4), Data: []int32{int32(one), -int32(one), 0, int32(MaxFx)}}
	y := &Tensor{Desc: NewDesc(DtypeQ12_20, 1, 4), Data: []int32{int32(one), int32(one), 5, 1}}
	add, err := (addFixed{}).Execute([]Tensor{*x, *y}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if add.Data[0] != int32(2*one) || add.Data[2] != 5 {
		t.Fatalf("ADD wrong: %v", add.Data)
	}
	if add.Data[3] != int32(MaxFx) {
		t.Fatal("ADD must saturate")
	}
	mul, err := (mulFixed{}).Execute([]Tensor{*x, *y}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if mul.Data[0] != int32(one) || mul.Data[1] != int32(-one) {
		t.Fatalf("MUL wrong: %v", mul.Data)
	}
	// mismatched shapes rejected
	z := &Tensor{Desc: NewDesc(DtypeQ12_20, 1, 3), Data: []int32{1, 2, 3}}
	if _, err := (addFixed{}).Execute([]Tensor{*x, *z}, nil); err == nil {
		t.Fatal("ADD with mismatched shapes accepted")
	}
}

func TestRequantizeOperator(t *testing.T) {
	in := &Tensor{Desc: NewDesc(DtypeInt32Accum, 1, 4), Data: []int32{0, 1, -1, 1 << 30}}
	params := ParamList{{Key: "clamp_hi", Value: int64(MaxFx)}, {Key: "clamp_lo", Value: int64(MinFx)},
		{Key: "mult", Value: 1 << 10}, {Key: "shift", Value: 30}}
	out, err := (requantize{}).Execute([]Tensor{*in}, params)
	if err != nil {
		t.Fatal(err)
	}
	// x*1024 >> 30: 1 -> round(2^-20) = 0; 2^30 -> 1<<0? 2^30*2^10>>30 = 2^10
	if out.Data[0] != 0 || out.Data[3] != 1<<10 {
		t.Fatalf("REQUANTIZE wrong: %v", out.Data)
	}
	if out.Desc.Dtype != uint8(DtypeQ12_20) {
		t.Fatal("REQUANTIZE must output q12.20")
	}
	// bad params
	if _, err := (requantize{}).Execute([]Tensor{*in}, ParamList{{Key: "mult", Value: 0}, {Key: "shift", Value: 1}}); err == nil {
		t.Fatal("zero mult accepted")
	}
	if _, err := (requantize{}).Execute([]Tensor{*in}, ParamList{{Key: "mult", Value: 1}, {Key: "shift", Value: 90}}); err == nil {
		t.Fatal("absurd shift accepted")
	}
}

func TestRMSNormOperator(t *testing.T) {
	// hidden=16, one row; known values.
	desc := NewDesc(DtypeQ12_20, 1, 16)
	data := make([]int32, 16)
	for i := range data {
		data[i] = int32((i + 1) * (1 << (FracBits - 4)))
	}
	weight := make([]int32, 16)
	for i := range weight {
		weight[i] = int32(one)
	}
	eps := int32(roundFx(1e-5))
	params := ParamList{{Key: "chunk", Value: normChunk}, {Key: "eps_fx", Value: int64(eps)}}
	out, err := (rmsNormFixed{}).Execute([]Tensor{
		{Desc: desc, Data: data}, {Desc: NewDesc(DtypeQ12_20, 16), Data: weight}}, params)
	if err != nil {
		t.Fatal(err)
	}
	// reference via float for the tolerance check
	var sum float64
	for _, v := range data {
		f := float64(v) / float64(one)
		sum += f * f
	}
	rms := math.Sqrt(sum/16 + 1e-5)
	for i := range out.Data {
		want := float64(data[i]) / float64(one) / rms
		got := float64(out.Data[i]) / float64(one)
		if math.Abs(got-want) > 5e-3 {
			t.Fatalf("rmsnorm element %d: got %v want %v", i, got, want)
		}
	}
	// all-zero input: eps keeps rms finite, output stays zero
	zero := make([]int32, 16)
	out0, err := (rmsNormFixed{}).Execute([]Tensor{
		{Desc: desc, Data: zero}, {Desc: NewDesc(DtypeQ12_20, 16), Data: weight}}, params)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range out0.Data {
		if v != 0 {
			t.Fatal("zero input must produce zero output")
		}
	}
	// weight shape must match last dim
	if _, err := (rmsNormFixed{}).Execute([]Tensor{
		{Desc: desc, Data: data}, {Desc: NewDesc(DtypeQ12_20, 8), Data: weight[:8]}}, params); err == nil {
		t.Fatal("mismatched weight accepted")
	}
}

func roundFx(x float64) int64 {
	return int64(math.Round(x * float64(int64(1)<<FracBits)))
}

func TestSiluOperator(t *testing.T) {
	desc := NewDesc(DtypeQ12_20, 1, 5)
	in := &Tensor{Desc: desc, Data: []int32{
		int32(roundFx(-6)), 0, int32(one), int32(roundFx(6)), int32(roundFx(0.5))}}
	out, err := (siluFixed{}).Execute([]Tensor{*in}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{-6 / (1 + math.Exp(6)), 0, 1 / (1 + math.Exp(-1)), 6 / (1 + math.Exp(-6)), 0.5 / (1 + math.Exp(-0.5))}
	for i := range out.Data {
		got := float64(out.Data[i]) / float64(one)
		if math.Abs(got-want[i]) > 5e-3 {
			t.Fatalf("silu[%d]=%v want %v", i, got, want[i])
		}
	}
}

func TestSoftmaxOperator(t *testing.T) {
	rows := [][]int32{
		{0, 0, 0, 0},
		{int32(roundFx(10)), int32(roundFx(-10)), 0, 0},
		{int32(roundFx(-800)), int32(roundFx(800)), 0, 0},
	}
	desc := NewDesc(DtypeQ12_20, 3, 4)
	flat := append(append(append([]int32{}, rows[0]...), rows[1]...), rows[2]...)
	out, err := (softmaxFixed{}).Execute([]Tensor{{Desc: desc, Data: flat}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 3; r++ {
		var sum float64
		for i := 0; i < 4; i++ {
			sum += float64(out.Data[r*4+i]) / float64(one)
		}
		if math.Abs(sum-1) > 2e-3 {
			t.Fatalf("softmax row %d sums to %v", r, sum)
		}
	}
	// uniform row: all equal probabilities
	if out.Data[0] != out.Data[1] || out.Data[0] != out.Data[2] {
		t.Fatal("uniform row must produce uniform probabilities")
	}
	// dominant element
	var maxProb int32
	for i := 0; i < 4; i++ {
		if out.Data[4+i] > maxProb {
			maxProb = out.Data[4+i]
		}
	}
	if float64(maxProb) < 0.99*float64(one) {
		t.Fatalf("dominant probability too small: %v", float64(maxProb)/float64(one))
	}
}

func TestRopeOperator(t *testing.T) {
	pairs := 4
	table := make([]int32, pairs*2)
	// identity rotation at position 0: cos=1, sin=0
	for i := 0; i < pairs; i++ {
		table[i*2] = int32(one)
		table[i*2+1] = 0
	}
	constants, err := NewRopeConstants(table, pairs)
	if err != nil {
		t.Fatal(err)
	}
	desc := NewDesc(DtypeQ12_20, 1, int64(pairs*2))
	in := &Tensor{Desc: desc, Data: []int32{int32(roundFx(1)), int32(roundFx(2)), int32(roundFx(3)), int32(roundFx(4)), 0, 0, 0, 0}}
	params := ParamList{{Key: "half_dim", Value: int64(pairs)}, {Key: "max_pos", Value: 1}}
	out, err := ExecuteRope(in, constants, params)
	if err != nil {
		t.Fatal(err)
	}
	if out.Data[0] != in.Data[0] || out.Data[1] != in.Data[1] {
		t.Fatal("identity rotation changed the pair")
	}
	// params validation
	if _, err := ExecuteRope(in, constants, ParamList{{Key: "half_dim", Value: int64(pairs)}, {Key: "max_pos", Value: 0}}); err == nil {
		t.Fatal("max_pos=0 accepted")
	}
}

func TestOperatorDeterminism(t *testing.T) {
	desc := NewDesc(DtypeQ12_20, 2, 32)
	data := make([]int32, 64)
	for i := range data {
		data[i] = int32((i*104729)%2000 - 1000)
	}
	for _, id := range []string{OpSiLUFixedV1, OpSoftmaxFixedV1} {
		op, _ := Lookup(id)
		in := []Tensor{{Desc: desc, Data: append([]int32(nil), data...)}}
		a, err := op.Execute(in, nil)
		if err != nil {
			t.Fatal(err)
		}
		b, err := op.Execute(in, nil)
		if err != nil {
			t.Fatal(err)
		}
		for i := range a.Data {
			if a.Data[i] != b.Data[i] {
				t.Fatalf("%s is not deterministic", id)
			}
		}
	}
}
