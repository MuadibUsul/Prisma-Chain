package verify

// Freivalds detection over exact signed int64 arithmetic.
//
// For A: int8[M,K], B: int8[K,N], C: int32[M,N] and r in {0,1}^N:
//
//	x = B x r   (K vector)
//	y = A x x   (M vector)
//	z = C x r   (M vector)
//
// If C = A x B exactly then y = z for every r. For a fixed incorrect C,
// each independent round detects with probability >= 1/2, so q rounds have
// a false-accept upper bound of 2^-q.
//
// Overflow bounds (proved at admission, see Profile.AdmitFor). All values
// are exact int64 with NO wraparound:
//
//	|x_t| <= 128 * N                    (each |B| <= 128, r binary)
//	|y_i| <= 128 * 128 * N * K = 16384*N*K
//	|z_i| <= N * (2^31-1)               (C is canonical int32)
//	|D_i| = |y_i - z_i| <= 16384*N*K + N*(2^31-1)
//
// The admission check requires 16384*N*K + N*(2^31-1) < 2^62, which every
// currently admissible task satisfies by a wide margin (max N*K under task
// admission is about 1.4e14, giving |D| <= ~2.3e18 < 2^62 = 4.6e18).

import (
	"errors"
	"math"
	"time"
)

var (
	errShape        = errors.New("verify: matrix shape mismatch or empty dimension")
	errOverflowRisk = errors.New("verify: shape exceeds the proven int64 Freivalds bound")
)

// AdmitFor proves the int64 overflow bound for one shape before any
// arithmetic runs. Shapes outside the bound are rejected here, never
// handled by silent wraparound.
func (p *VerificationProfile) AdmitFor(m, n, k uint64) error {
	if m == 0 || n == 0 || k == 0 {
		return errShape
	}
	// y_i bound times the z residual headroom must stay below 2^62.
	yBound := uint64(16384) * n * k
	zBound := n * uint64(math.MaxInt32)
	if yBound/16384 != n*k || yBound+zBound < yBound {
		return errOverflowRisk
	}
	if yBound+zBound >= 1<<62 {
		return errOverflowRisk
	}
	return nil
}

// Validate checks the profile fields.
func (p *VerificationProfile) Validate() error {
	if p.Algorithm != AlgorithmFreivaldsBinaryV1 {
		return errors.New("verify: unknown verification algorithm")
	}
	if p.Rounds == 0 {
		return errors.New("verify: rounds must be at least 1")
	}
	return nil
}

// VerifyFreivalds is the scalar reference implementation: one independent
// binary vector per round, exact int64 accumulation. It is the protocol
// correctness oracle; the batched variant must agree with it exactly.
func VerifyFreivalds(a, b []int8, c []int32, m, n, k uint64, prof VerificationProfile, src RandomSource) (*VerificationResult, error) {
	if err := prof.Validate(); err != nil {
		return nil, err
	}
	if err := prof.AdmitFor(m, n, k); err != nil {
		return nil, err
	}
	if uint64(len(a)) != m*k || uint64(len(b)) != k*n || uint64(len(c)) != m*n {
		return nil, errShape
	}
	started := time.Now()
	res := &VerificationResult{}
	macsPerRound := n*k + m*k + m*n

	for round := uint16(0); round < prof.Rounds; round++ {
		r := RandomBinaryVector(src, n)
		// x = B x r
		x := make([]int64, k)
		for t := uint64(0); t < k; t++ {
			var acc int64
			bt := b[t*n : (t+1)*n]
			for j := uint64(0); j < n; j++ {
				acc += int64(bt[j]) * int64(r[j])
			}
			x[t] = acc
		}
		// y = A x x
		y := make([]int64, m)
		for i := uint64(0); i < m; i++ {
			var acc int64
			ai := a[i*k : (i+1)*k]
			for t := uint64(0); t < k; t++ {
				acc += int64(ai[t]) * x[t]
			}
			y[i] = acc
		}
		// z = C x r
		z := make([]int64, m)
		for i := uint64(0); i < m; i++ {
			var acc int64
			ci := c[i*n : (i+1)*n]
			for j := uint64(0); j < n; j++ {
				acc += int64(ci[j]) * int64(r[j])
			}
			z[i] = acc
		}
		res.RoundsExecuted = round + 1
		res.VerificationMACs += macsPerRound
		if !equalInt64(y, z) {
			res.Passed = false
			res.MismatchRound = int(round)
			res.ResidualRows = ResidualBadRows(y, z)
			res.Duration = msSince(started)
			return res, nil
		}
	}
	res.Passed = true
	res.Duration = msSince(started)
	return res, nil
}

// VerifyFreivaldsBatch runs q rounds as one batch: R in {0,1}^{N x q},
// X = B x R, Y = A x X, Z = C x R. The result must be identical to the
// scalar reference (tested). Batch only changes the execution shape so GPU
// backends can use one matrix multiplication per stage; it never changes
// the arithmetic.
func VerifyFreivaldsBatch(a, b []int8, c []int32, m, n, k uint64, prof VerificationProfile, src RandomSource) (*VerificationResult, error) {
	if err := prof.Validate(); err != nil {
		return nil, err
	}
	if err := prof.AdmitFor(m, n, k); err != nil {
		return nil, err
	}
	if uint64(len(a)) != m*k || uint64(len(b)) != k*n || uint64(len(c)) != m*n {
		return nil, errShape
	}
	q := int(prof.Rounds)
	started := time.Now()
	res := &VerificationResult{RoundsExecuted: prof.Rounds}
	macsPerRound := n*k + m*k + m*n
	res.VerificationMACs = macsPerRound * uint64(q)

	// R[N][q]
	R := make([][]int64, n)
	for j := uint64(0); j < n; j++ {
		R[j] = make([]int64, q)
		vec := RandomBinaryVector(src, n)
		for t := 0; t < q; t++ {
			R[j][t] = int64(vec[j])
		}
	}
	// X = B x R : X[t][s]
	X := make([][]int64, k)
	for t := uint64(0); t < k; t++ {
		X[t] = make([]int64, q)
		bt := b[t*n : (t+1)*n]
		for s := 0; s < q; s++ {
			var acc int64
			for j := uint64(0); j < n; j++ {
				acc += int64(bt[j]) * R[j][s]
			}
			X[t][s] = acc
		}
	}
	// Y = A x X
	Y := make([][]int64, m)
	for i := uint64(0); i < m; i++ {
		Y[i] = make([]int64, q)
		ai := a[i*k : (i+1)*k]
		for s := 0; s < q; s++ {
			var acc int64
			for t := uint64(0); t < k; t++ {
				acc += int64(ai[t]) * X[t][s]
			}
			Y[i][s] = acc
		}
	}
	// Z = C x R
	Z := make([][]int64, m)
	for i := uint64(0); i < m; i++ {
		Z[i] = make([]int64, q)
		ci := c[i*n : (i+1)*n]
		for s := 0; s < q; s++ {
			var acc int64
			for j := uint64(0); j < n; j++ {
				acc += int64(ci[j]) * R[j][s]
			}
			Z[i][s] = acc
		}
	}
	for s := 0; s < q; s++ {
		mismatch := false
		for i := uint64(0); i < m; i++ {
			if Y[i][s] != Z[i][s] {
				mismatch = true
				break
			}
		}
		if mismatch {
			res.Passed = false
			res.MismatchRound = s
			yCol := make([]int64, m)
			zCol := make([]int64, m)
			for i := uint64(0); i < m; i++ {
				yCol[i], zCol[i] = Y[i][s], Z[i][s]
			}
			res.ResidualRows = ResidualBadRows(yCol, zCol)
			res.Duration = msSince(started)
			return res, nil
		}
	}
	res.Passed = true
	res.Duration = msSince(started)
	return res, nil
}

// ResidualBadRows returns the rows where y and z differ.
func ResidualBadRows(y, z []int64) []uint64 {
	var rows []uint64
	for i := range y {
		if y[i] != z[i] {
			rows = append(rows, uint64(i))
		}
	}
	return rows
}

func equalInt64(x, y []int64) bool {
	if len(x) != len(y) {
		return false
	}
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

func msSince(t time.Time) float64 {
	return float64(time.Since(t).Microseconds()) / 1000.0
}
