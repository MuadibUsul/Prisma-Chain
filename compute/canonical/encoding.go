package canonical

// Canonical encoding for CANONICAL_GRAPH_V1 objects: a minimal RFC 8949
// core-deterministic CBOR subset (unsigned/negative integers, byte and
// text strings, arrays, maps with text keys sorted bytewise by canonical
// encoding, and structs with `gemm:"snake_case"` tags). Independent of the
// frozen gemmv1 encoder on purpose: this module has its own version space
// and must be able to evolve without touching GEMM wire formats.

import (
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"sort"
)

// Pair is one entry of the canonical integer parameter list; slices of
// Pair are sorted before encoding so parameter maps are deterministic
// without relying on Go map iteration.
type Pair struct {
	Key   string `gemm:"key"`
	Value int64  `gemm:"value"`
}

// SortPairs orders a parameter list by key.
func SortPairs(pairs []Pair) {
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].Key < pairs[j].Key })
}

func EncodeCanonical(v any) ([]byte, error) {
	var buf []byte
	if err := encodeValue(&buf, reflect.ValueOf(v)); err != nil {
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

func encodeValue(dst *[]byte, v reflect.Value) error {
	if v.Kind() == reflect.Ptr || v.Kind() == reflect.Interface {
		if v.IsNil() {
			return errors.New("canonical: nil values cannot be encoded")
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
		if v.Type().Elem().Kind() == reflect.Int32 {
			// int32 slices are encoded as big-endian two's complement byte
			// strings: one canonical representation for tensor payloads.
			encodeHead(dst, 2, uint64(v.Len()*4))
			for i := 0; i < v.Len(); i++ {
				*dst = binary.BigEndian.AppendUint32(*dst, uint32(int32(v.Index(i).Int())))
			}
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
		return fmt.Errorf("canonical: type %s not supported", v.Kind())
	}
	return nil
}

func encodeStruct(dst *[]byte, v reflect.Value) error {
	t := v.Type()
	fields := make([]struct{ key, val []byte }, 0, t.NumField())
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
			return fmt.Errorf("canonical: field %s.%s has no gemm tag", t.Name(), f.Name)
		}
		var kb, vb []byte
		encodeHead(&kb, 3, uint64(len(key)))
		kb = append(kb, key...)
		if err := encodeValue(&vb, v.Field(i)); err != nil {
			return fmt.Errorf("canonical: field %s: %w", f.Name, err)
		}
		fields = append(fields, struct{ key, val []byte }{kb, vb})
	}
	sort.Slice(fields, func(i, j int) bool { return lessBytes(fields[i].key, fields[j].key) })
	encodeHead(dst, 5, uint64(len(fields)))
	for _, f := range fields {
		*dst = append(*dst, f.key...)
		*dst = append(*dst, f.val...)
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
