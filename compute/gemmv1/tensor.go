package gemmv1

// Canonical tensor layout for GEMM_INT8_V1.
//
// A and B are canonical as consecutive raw signed int8 bytes, row-major.
// C and every partial state are int32 encoded as signed two's-complement
// big-endian. No platform-dependent representation (native endianness,
// alignment, float conversion) ever enters a hash input.

import "errors"

var (
	errLenA        = errors.New("gemmv1: A must hold M*K int8 values")
	errLenB        = errors.New("gemmv1: B must hold K*N int8 values")
	errStateLen    = errors.New("gemmv1: state must be exactly 64 int32 values")
	errTileLen     = errors.New("gemmv1: tile bytes must be exactly 256")
	errNotMultiple = errors.New("gemmv1: flat length is not a multiple of 64")
)

// CheckInputSizes validates raw int8 buffer lengths for an M x N x K task.
func CheckInputSizes(m, n, k uint64, a, b []int8) error {
	if uint64(len(a)) != m*k {
		return errLenA
	}
	if uint64(len(b)) != k*n {
		return errLenB
	}
	return nil
}

// Int32sToCanonical encodes int32 values as big-endian two's complement.
func Int32sToCanonical(vals []int32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		out[4*i] = byte(v >> 24)
		out[4*i+1] = byte(v >> 16)
		out[4*i+2] = byte(v >> 8)
		out[4*i+3] = byte(v)
	}
	return out
}

// CanonicalToInt32s decodes big-endian two's-complement int32 values.
func CanonicalToInt32s(b []byte) ([]int32, error) {
	if len(b)%4 != 0 {
		return nil, errNotMultiple
	}
	out := make([]int32, len(b)/4)
	for i := range out {
		out[i] = int32(uint32(b[4*i])<<24 | uint32(b[4*i+1])<<16 | uint32(b[4*i+2])<<8 | uint32(b[4*i+3]))
	}
	return out, nil
}

// State is one 8x8 int32 tile trace state, row-major, 64 values.
type State [intTileCount]int32

// CanonicalBytes encodes the state in canonical form (64 big-endian int32).
func (s *State) CanonicalBytes() []byte {
	return Int32sToCanonical(s[:])
}

// StateFromCanonical decodes a canonical state.
func StateFromCanonical(b []byte) (*State, error) {
	if len(b) != intTileBytes {
		return nil, errStateLen
	}
	vals, err := CanonicalToInt32s(b)
	if err != nil {
		return nil, err
	}
	var s State
	copy(s[:], vals)
	return &s, nil
}

// IsZero reports whether the state is the all-zero matrix (S0).
func (s *State) IsZero() bool {
	for _, v := range s {
		if v != 0 {
			return false
		}
	}
	return true
}
