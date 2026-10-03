package gemmv1

// Canonical CBOR encoding for GEMM v1 protocol objects.
//
// This is a deliberately small implementation of the RFC 8949 core
// deterministic encoding rules restricted to the types the protocol uses:
// unsigned and negative integers, byte strings, text strings, arrays and
// maps with text-string keys. Floats, tags, booleans and indefinite lengths
// are rejected so that the same logical object can only ever produce one
// byte sequence. Map keys are sorted bytewise by their canonical encodings
// (RFC 8949 section 4.2.1).
//
// A matching encoder lives in compute/gemmv1/python/gemmv1/canonical_cbor.py;
// cross-language equality is enforced by testdata/cross_language_vectors.json.
// Struct fields carry `gemm:"snake_case_key"` tags. Every tagged field is
// always encoded; an absent optional value is a zero-length byte string.

import (
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"sort"
)

// EncodeCanonical returns the canonical CBOR encoding of v.
func EncodeCanonical(v any) ([]byte, error) {
	var buf []byte
	err := encodeValue(&buf, reflect.ValueOf(v))
	if err != nil {
		return nil, err
	}
	return buf, nil
}

func encodeHead(dst *[]byte, major byte, val uint64) {
	b := *dst
	switch {
	case val < 24:
		b = append(b, major<<5|byte(val))
	case val < 0x100:
		b = append(b, major<<5|24, byte(val))
	case val < 0x10000:
		b = append(b, major<<5|25, byte(val>>8), byte(val))
	case val < 0x100000000:
		b = append(b, major<<5|26)
		b = binary.BigEndian.AppendUint32(b, uint32(val))
	default:
		b = append(b, major<<5|27)
		b = binary.BigEndian.AppendUint64(b, val)
	}
	*dst = b
}

var errUnsupportedType = errors.New("gemmv1: type not supported by canonical CBOR encoding")

func encodeValue(dst *[]byte, v reflect.Value) error {
	if v.Kind() == reflect.Ptr || v.Kind() == reflect.Interface {
		if v.IsNil() {
			return errors.New("gemmv1: nil values cannot be encoded")
		}
		return encodeValue(dst, v.Elem())
	}
	switch v.Kind() {
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		encodeHead(dst, 0, v.Uint())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		n := v.Int()
		if n >= 0 {
			encodeHead(dst, 0, uint64(n))
		} else {
			encodeHead(dst, 1, ^uint64(n))
		}
	case reflect.Slice:
		if v.Type().Elem().Kind() == reflect.Uint8 {
			encodeHead(dst, 2, uint64(v.Len()))
			*dst = append(*dst, v.Bytes()...)
			return nil
		}
		encodeHead(dst, 4, uint64(v.Len()))
		for i := 0; i < v.Len(); i++ {
			if err := encodeValue(dst, v.Index(i)); err != nil {
				return err
			}
		}
	case reflect.Array:
		if v.Type().Elem().Kind() == reflect.Uint8 {
			encodeHead(dst, 2, uint64(v.Len()))
			for i := 0; i < v.Len(); i++ {
				*dst = append(*dst, byte(v.Index(i).Uint()))
			}
			return nil
		}
		encodeHead(dst, 4, uint64(v.Len()))
		for i := 0; i < v.Len(); i++ {
			if err := encodeValue(dst, v.Index(i)); err != nil {
				return err
			}
		}
	case reflect.String:
		encodeHead(dst, 3, uint64(len(v.String())))
		*dst = append(*dst, v.String()...)
	case reflect.Struct:
		return encodeStruct(dst, v)
	default:
		return fmt.Errorf("%w: %s", errUnsupportedType, v.Kind())
	}
	return nil
}

type cborField struct {
	key   []byte
	value []byte
}

func encodeStruct(dst *[]byte, v reflect.Value) error {
	t := v.Type()
	fields := make([]cborField, 0, t.NumField())
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if f.PkgPath != "" {
			continue
		}
		key := f.Tag.Get("gemm")
		switch key {
		case "-":
			continue
		case "":
			return fmt.Errorf("gemmv1: struct field %s.%s has no gemm tag", t.Name(), f.Name)
		}
		var kbuf, vbuf []byte
		encodeHead(&kbuf, 3, uint64(len(key)))
		kbuf = append(kbuf, key...)
		if err := encodeValue(&vbuf, v.Field(i)); err != nil {
			return fmt.Errorf("gemmv1: field %s: %w", f.Name, err)
		}
		fields = append(fields, cborField{key: kbuf, value: vbuf})
	}
	sort.Slice(fields, func(a, b int) bool {
		return lessBytes(fields[a].key, fields[b].key)
	})
	encodeHead(dst, 5, uint64(len(fields)))
	for _, f := range fields {
		*dst = append(*dst, f.key...)
		*dst = append(*dst, f.value...)
	}
	return nil
}

func lessBytes(a, b []byte) bool {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return len(a) < len(b)
}
