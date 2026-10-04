package canonical

// FREIVALDS_A13W10_I64_V1 — randomized exact check of one wide GEMM.
//
//   r in {0,1}^N  (one random bit per output column)
//   x = W r       (exact int64)
//   y = A x       (exact int64)
//   z = C r       (exact int64)
//   check y == z
//
// Mismatch is DETECTION ONLY: it never slashes by itself; it triggers the
// deterministic wide dispute path.  Every intermediate is asserted to stay
// inside signed int64 (F.5A bound: <= 46 signed bits for the real block).
//
// A v1 similarity: the same shape as the frozen gemmv1 Freivalds, but the
// wire/hash/arithmetic all live in the V2 version space and the operands
// are logical wide integers.

import (
	"errors"
	"fmt"
)

// WideFreivaldsBits is the number of signed bits the largest intermediate
// may touch for admitted A13W10/X shapes.  The F.5A evidence measured
// <= 46; the admission check below recomputes the theoretical bound for
// each shape instead of trusting the measurement.
func wideFreivaldsBoundBits(aBits, wBits, m, n, k int) int {
	// r has n entries of 0/1.  |W r| <= 2^(wBits-1) * n.  |A x| <= 2^(aBits-1)
	// * k * max|x|.  Bits of the product sum, one bit per factor.
	bx := (wBits - 1) + ceilLog2(n)
	by := (aBits - 1) + ceilLog2(k) + bx
	return by + 1 // sign
}

func ceilLog2(n int) int {
	if n <= 1 {
		return 0
	}
	bits := 0
	for v := n - 1; v > 0; v >>= 1 {
		bits++
	}
	return bits
}

// FreivaldsWideCheck runs one round with the caller-supplied random bits
// (deterministic in tests, CSPRNG in production; the watcher derives them
// after the result commitment exists — see the watcher randomness rule).
func FreivaldsWideCheck(a, w, c []int64, m, n, k int, transposeB, wA13 bool, r []uint8) (bool, error) {
	if len(r) != n {
		return false, errors.New("canonical/v2: freivalds needs exactly N random bits")
	}
	if len(a) != m*k || len(c) != m*n {
		return false, errors.New("canonical/v2: freivalds operand size mismatch")
	}
	wbits := 10
	if wA13 {
		wbits = 13
	}
	bneeded := wideFreivaldsBoundBits(13, wbits, m, n, k)
	if bneeded > 62 {
		return false, fmt.Errorf("canonical/v2: freivalds intermediates need %d signed bits (>62)", bneeded)
	}
	// x = W r  (k vector)
	x := make([]int64, k)
	for d := 0; d < k; d++ {
		var acc int64
		for j := 0; j < n; j++ {
			if r[j] == 0 {
				continue
			}
			var wv int64
			if transposeB {
				wv = w[j*k+d]
			} else {
				wv = w[d*n+j]
			}
			acc += wv
		}
		x[d] = acc
	}
	// y = A x (m vector)
	y := make([]int64, m)
	for i := 0; i < m; i++ {
		var acc int64
		for d := 0; d < k; d++ {
			acc += a[i*k+d] * x[d]
		}
		y[i] = acc
	}
	// z = C r (m vector)
	z := make([]int64, m)
	for i := 0; i < m; i++ {
		var acc int64
		for j := 0; j < n; j++ {
			if r[j] == 0 {
				continue
			}
			acc += c[i*n+j]
		}
		z[i] = acc
	}
	for i := 0; i < m; i++ {
		if y[i] != z[i] {
			return false, nil
		}
	}
	return true, nil
}

// FreivaldsWideRounds is the production round count: 40 rounds give
// per-GEMM false-accept <= 2^-40; with 83 GEMM nodes the union bound is
// <= 83 * 2^-40 < 2^-33.6.  (The F.1 bundle test's 2 rounds are NOT a
// production parameter.)
const FreivaldsWideRounds = 40
