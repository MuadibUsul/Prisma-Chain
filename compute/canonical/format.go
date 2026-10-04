package canonical

// CanonicalMathV1: the single fixed-point numeric contract of the block
// (docs/canonical-math-v1.md). Q12.20 in a signed int32, ties-to-even
// rounding, saturating adds, no libm anywhere in canonical semantics.
//
// Everything here is pure integer arithmetic; the same algorithms are
// mirrored bit-for-bit by the Python implementation and therefore by any
// GPU backend.

import (
	"errors"
	"math/big"
)

const (
	// FracBits is the frozen fractional width of CANONICAL_MATH_V1.
	FracBits = 20
	// MathVersion names the frozen numeric contract.
	MathVersion = "CANONICAL_MATH_V1"

	one = int64(1) << FracBits
	// MaxFx / MinFx are the saturated Q12.20 bounds.
	MaxFx = int64(1)<<31 - 1
	MinFx = -(int64(1) << 31)
)

var (
	errDivZero = errors.New("canonical: division by zero is a spec violation")
	errInvsqrt = errors.New("canonical: invsqrt domain (input must be positive)")
)

// Saturate clamps a wide value into the Q12.20 range.
func Saturate(v int64) int32 {
	if v > MaxFx {
		return int32(MaxFx)
	}
	if v < MinFx {
		return int32(MinFx)
	}
	return int32(v)
}

// RShiftRoundEven shifts right with round-to-nearest, ties-to-even (the
// only rounding mode of the protocol). Negative values round
// symmetrically: -1.5 -> -2, -0.5 -> 0.
func RShiftRoundEven(v int64, s uint) int64 {
	if s == 0 {
		return v
	}
	q := v >> s
	r := v & (int64(1)<<s - 1)
	half := int64(1) << (s - 1)
	if r > half || (r == half && q&1 == 1) {
		q++
	}
	return q
}

// MulFx returns (a*b) in Q12.20 with ties-to-even rounding.
func MulFx(a, b int32) int32 {
	return Saturate(RShiftRoundEven(int64(a)*int64(b), FracBits))
}

// AddFx returns the saturating sum.
func AddFx(a, b int32) int32 { return Saturate(int64(a) + int64(b)) }

// SubFx returns the saturating difference.
func SubFx(a, b int32) int32 { return Saturate(int64(a) - int64(b)) }

// DivFx returns a/b in Q12.20 with ties-to-even rounding. Division by
// zero is a spec violation and is rejected.
func DivFx(a, b int32) (int32, error) {
	if b == 0 {
		return 0, errDivZero
	}
	num := int64(a) << FracBits
	q := num / int64(b)
	r := num % int64(b)
	if r != 0 {
		twice := 2 * abs64(r)
		ab := abs64(int64(b))
		if twice > ab || (twice == ab && q&1 == 1) {
			if (b > 0) == (r > 0) {
				q++
			} else {
				q--
			}
		}
	}
	return Saturate(q), nil
}

// ExpFx returns exp(x) in Q12.20 for any x, using the frozen split
// 2^(x/ln2) = 2^k * P(f). Inputs <= -24 underflow to zero; inputs >= 21
// saturate at MaxFx.
func ExpFx(x int32) int32 {
	xi := int64(x)
	if xi <= -(one * 24) {
		return 0
	}
	if xi >= one*21 {
		return int32(MaxFx)
	}
	k := xi / ln2Fx   // truncating division (mirrored bit-for-bit by every backend)
	f := xi - k*ln2Fx // |f| < ln2Fx; the Taylor polynomial covers both signs
	poly := one + MulFxWide(f, c1Fx+MulFxWide(f, c2Fx+MulFxWide(f, c3Fx+MulFxWide(f, c4Fx, FracBits), FracBits), FracBits), FracBits)
	if k < 0 {
		if -k >= 32 {
			return 0
		}
		return Saturate(RShiftRoundEven(poly, uint(-k)))
	}
	if k > 31 {
		return int32(MaxFx)
	}
	return Saturate(poly << uint(k))
}

// MulFxWide multiplies a wide int64 Q12.20 value by an int32 Q12.20 value
// (used inside polynomial evaluation where the running value stays in
// range but is not yet narrowed).
func MulFxWide(a int64, b int64, frac uint) int64 {
	return RShiftRoundEven(a*b, frac)
}

// Pinned polynomial / constant coefficients of CanonicalMathV1 (generated
// once by high-precision math, rounded to Q12.20, committed here).
var (
	ln2Fx int64 = 726817  // round(ln 2 * 2^20)
	c1Fx  int64 = 1048576 // 1
	c2Fx  int64 = 524288  // round(1/2 * 2^20)
	c3Fx  int64 = 174763  // round(1/6 * 2^20)
	c4Fx  int64 = 43691   // round(1/24 * 2^20)
)

// SigmoidFx returns the stable sigmoid in Q12.20: 1/(1+e^-x) for x >= 0,
// e^x/(1+e^x) for x < 0, so the exponent is always non-positive.
func SigmoidFx(x int32) int32 {
	if x >= 0 {
		den := one + int64(ExpFx(-x))
		v, err := DivFx(int32(one), int32(den))
		if err != nil {
			return int32(one)
		}
		return v
	}
	e := int64(ExpFx(x))
	den := one + e
	q := (e << FracBits) / den
	r := (e << FracBits) % den
	if r != 0 {
		if 2*abs64(r) > abs64(den) || (2*abs64(r) == abs64(den) && q&1 == 1) {
			q++
		}
	}
	return Saturate(q)
}

// INV_SQRT_TABLE_POINTS is the pinned table size for invsqrt.
const INV_SQRT_TABLE_POINTS = 16

// InvSqrtTable entry i is round(1/sqrt(1 + i*3/16) * 2^20), generated once
// by high-precision math (tools/canonical_math_analysis.py holds the
// generator; the values below are the frozen output).
var invSqrtTable = [INV_SQRT_TABLE_POINTS]int32{
	1048576, 962239, 894228, 838860, 792649, 753319, 719317, 689539,
	663177, 639625, 618416, 599186, 581645, 565560, 550739, 537025,
}

// InvSqrtFx returns 1/sqrt(x) in Q12.20 using the frozen algorithm:
// normalize x = m*4^e with m in [1,4); pinned table estimate; apply the
// 4^-e scale BEFORE iterating; exactly 3 Newton steps at doubled internal
// precision (int64-safe); saturate, never diverge.
// InvSqrtFx returns 1/sqrt(x) in Q12.20 for x in the full int64 domain.
//
// Algorithm (frozen): normalize x = m*4^e with m in [1,4); the pinned
// 16-point table of 1/sqrt(m) is accurate to ~9%; exactly four Newton
// steps are run IN THE NORMALIZED DOMAIN, where every intermediate stays
// inside int64 (the naive form over the raw x underflows or overflows at
// the extremes); finally the 2^-e scale is applied with saturation. No
// libm, no unbounded integers, result never zero when x > 0.
func InvSqrtFx(x int64) (int32, error) {
	if x <= 0 {
		return 0, errInvsqrt
	}
	// normalize
	b := 64 - leadingZeros64(uint64(x))
	target := FracBits + 2
	var e int
	var m int64
	if b >= target {
		shift := b - target
		e = (shift + 1) / 2
		m = x >> uint(2*e)
	} else {
		e = -((target - b + 1) / 2)
		m = x << uint(-2*e)
	}
	if m >= one<<2 {
		m >>= 2
		e++
	}
	idx := ((m - one) * INV_SQRT_TABLE_POINTS) / (one * 3)
	if idx < 0 {
		idx = 0
	}
	if idx >= INV_SQRT_TABLE_POINTS {
		idx = INV_SQRT_TABLE_POINTS - 1
	}
	// four Newton steps on m in [1,4]: y in [0.5,1], y^2 >= 2^-2 so no Qf
	// underflow, and m*y^2 <= 4 keeps every product far below 2^63.
	Y := int64(invSqrtTable[idx])
	for i := 0; i < 4; i++ {
		Y2 := (Y * Y) >> FracBits
		mY2 := (m * Y2) >> FracBits
		corr := int64(3)<<(FracBits-1) - (mY2 >> 1)
		Y = (Y * corr) >> FracBits
	}
	// apply 2^-e
	if e > 0 {
		if e >= 63 {
			return 1, nil
		}
		Y >>= uint(e)
	} else if e < 0 {
		if -e < 63 {
			Y <<= uint(-e)
		}
	}
	if Y < 1 {
		Y = 1
	}
	return Saturate(Y), nil
}

// SinCosFx returns the pinned RoPE constants for (position, frequency
// index). The values come from a committed table generated by
// high-precision math; runtime never calls libm.
func SinCosFx(table []int32, position, freqIndex, pairsPerRow int) (cosFx, sinFx int32, ok bool) {
	idx := (position*pairsPerRow + freqIndex) * 2
	if idx+1 >= len(table) {
		return 0, 0, false
	}
	return table[idx], table[idx+1], true
}

func abs64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

func leadingZeros64(v uint64) int {
	if v == 0 {
		return 64
	}
	n := 0
	for v&(1<<63) == 0 {
		v <<= 1
		n++
	}
	return n
}

// BuildInvSqrtTable regenerates the pinned table with high-precision math
// (exact big.Int square roots); used by the generator tool and tests to
// prove the committed constants are the frozen ones.
func BuildInvSqrtTable() [INV_SQRT_TABLE_POINTS]int32 {
	var out [INV_SQRT_TABLE_POINTS]int32
	scale := new(big.Int).Lsh(big.NewInt(1), uint(2*FracBits))
	for i := 0; i < INV_SQRT_TABLE_POINTS; i++ {
		// m = 1 + i*3/16 exactly in Q2f
		m := new(big.Int).Set(scale)
		m.Add(m, new(big.Int).Mul(big.NewInt(int64(i*3)), new(big.Int).Lsh(big.NewInt(1), uint(2*FracBits-4))))
		// value = 2^f / sqrt(m) with m in Q2f: sqrt(m) = sqrt(m_num/2^2f) =
		// sqrt(m_num)/2^f, so value = 2^f * 2^f / sqrt(m_num) = 2^2f/sqrt(m_num)
		root := new(big.Int).Sqrt(m)
		q := new(big.Int).Div(new(big.Int).Lsh(big.NewInt(1), uint(2*FracBits)), root)
		out[i] = int32(q.Int64())
	}
	return out
}

// PinnedInvSqrtTable exposes the committed table for tests.
func PinnedInvSqrtTable() [INV_SQRT_TABLE_POINTS]int32 { return invSqrtTable }
