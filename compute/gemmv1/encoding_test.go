package gemmv1

// Canonical CBOR encoder tests: RFC 8949 Appendix A vectors for the
// supported subset, map key ordering, and rejection of unsupported types.

import (
	"bytes"
	"testing"
)

func enc(t *testing.T, v any) []byte {
	t.Helper()
	b, err := EncodeCanonical(v)
	if err != nil {
		t.Fatalf("EncodeCanonical(%v): %v", v, err)
	}
	return b
}

func TestEncodeAppendAVectors(t *testing.T) {
	cases := []struct {
		name string
		v    any
		want []byte
	}{
		{"uint0", uint64(0), []byte{0x00}},
		{"uint23", uint64(23), []byte{0x17}},
		{"uint24", uint64(24), []byte{0x18, 0x18}},
		{"uint100", uint64(100), []byte{0x18, 0x64}},
		{"uint1000", uint64(1000), []byte{0x19, 0x03, 0xe8}},
		{"uint1e6", uint64(1000000), []byte{0x1a, 0x00, 0x0f, 0x42, 0x40}},
		{"uint1e12", uint64(1000000000000), []byte{0x1b, 0x00, 0x00, 0x00, 0xe8, 0xd4, 0xa5, 0x10, 0x00}},
		{"uintmax", uint64(18446744073709551615), []byte{0x1b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}},
		{"neg1", int64(-1), []byte{0x20}},
		{"neg10", int64(-10), []byte{0x29}},
		{"neg100", int64(-100), []byte{0x38, 0x63}},
		{"neg1000", int64(-1000), []byte{0x39, 0x03, 0xe7}},
		{"bstr_empty", []byte{}, []byte{0x40}},
		{"bstr", []byte{0x01, 0x02, 0x03, 0x04}, []byte{0x44, 0x01, 0x02, 0x03, 0x04}},
		{"tstr_empty", "", []byte{0x60}},
		{"tstr_a", "a", []byte{0x61, 0x61}},
		{"tstr_IETF", "IETF", []byte{0x64, 0x49, 0x45, 0x54, 0x46}},
		{"arr_empty", []uint64{}, []byte{0x80}},
		{"arr_123", []uint64{1, 2, 3}, []byte{0x83, 0x01, 0x02, 0x03}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := enc(t, c.v); !bytes.Equal(got, c.want) {
				t.Fatalf("got %x, want %x", got, c.want)
			}
		})
	}
}

type vecStruct struct {
	Zeta  uint64 `gemm:"zeta"`
	Alpha string `gemm:"alpha"`
	Data  []byte `gemm:"data"`
}

func TestEncodeStructKeySorting(t *testing.T) {
	// RFC 8949 4.2.1: keys sort bytewise by their canonical encodings.
	// Encoded: "data"=0x64'data' < "zeta"=0x64'zeta' < "alpha"=0x65'alpha'.
	got := enc(t, vecStruct{Zeta: 1, Alpha: "IETF", Data: []byte{0x01}})
	want := []byte{
		0xa3,
		0x64, 'd', 'a', 't', 'a', 0x41, 0x01,
		0x64, 'z', 'e', 't', 'a', 0x01,
		0x65, 'a', 'l', 'p', 'h', 'a', 0x64, 'I', 'E', 'T', 'F',
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("got %x, want %x", got, want)
	}
	// Field order in the struct must not matter.
	same := enc(t, vecStruct{Data: []byte{0x01}, Alpha: "IETF", Zeta: 1})
	if !bytes.Equal(got, same) {
		t.Fatalf("struct field order changed the encoding")
	}
}

func TestEncodeRejections(t *testing.T) {
	bad := []any{
		true,
		3.14,
		map[string]uint64{"a": 1},
		[]any{nil},
		(struct{ X uintptr }{}),
	}
	for _, v := range bad {
		if _, err := EncodeCanonical(v); err == nil {
			t.Fatalf("expected rejection for %T", v)
		}
	}
}

func TestEncodeInt32BE(t *testing.T) {
	cases := []struct {
		v    int32
		want []byte
	}{
		{0, []byte{0, 0, 0, 0}},
		{1, []byte{0, 0, 0, 1}},
		{-1, []byte{0xff, 0xff, 0xff, 0xff}},
		{127, []byte{0, 0, 0, 0x7f}},
		{-2147483648, []byte{0x80, 0, 0, 0}},
		{2147483647, []byte{0x7f, 0xff, 0xff, 0xff}},
	}
	for _, c := range cases {
		vals := []int32{c.v}
		got := Int32sToCanonical(vals)
		if !bytes.Equal(got, c.want) {
			t.Fatalf("int32 %d: got %x want %x", c.v, got, c.want)
		}
		back, err := CanonicalToInt32s(got)
		if err != nil || back[0] != c.v {
			t.Fatalf("roundtrip %d: %v %v", c.v, back, err)
		}
	}
}
