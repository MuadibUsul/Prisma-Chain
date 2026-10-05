package gemmv1

// Development helpers shared by tests, the CLI and cross-language vectors.
// They are protocol utilities, not consensus paths.

import (
	"crypto/ed25519"
	"crypto/rand"
	"errors"
	"fmt"
)

// GenTestMatrix deterministically generates length int8 values. Value i is
// int8(SHA256("PRISMA_GEMM_TESTGEN_V1\x00" || u32be(seed) || id || u64be(i))[0]).
// The identical generator exists in the Python package so that cross-language
// vectors use the same matrices.
func GenTestMatrix(id byte, seed uint32, length uint64) []int8 {
	out := make([]int8, length)
	prefix := make([]byte, 0, len(DomainTestGen)+4+1+8)
	prefix = append(prefix, DomainTestGen...)
	prefix = appendUint32BE(prefix, seed)
	prefix = append(prefix, id)
	for i := uint64(0); i < length; i++ {
		buf := appendUint64BE(append([]byte(nil), prefix...), i)
		h := hashBytes(buf)
		out[i] = int8(h[0])
	}
	return out
}

// DomainTestGen is the test-vector generator domain. It is dev-only and
// never used by protocol messages.
const DomainTestGen = "PRISMA_GEMM_TESTGEN_V1\x00"

func appendUint64BE(dst []byte, v uint64) []byte {
	var b [8]byte
	for i := 0; i < 8; i++ {
		b[7-i] = byte(v >> (8 * i))
	}
	return append(dst, b[:]...)
}

// Int8ToBytes reinterprets an int8 slice as a byte slice (two's complement).
func Int8ToBytes(v []int8) []byte {
	out := make([]byte, len(v))
	for i, x := range v {
		out[i] = byte(x)
	}
	return out
}

// GenerateKeyPair creates an Ed25519 key pair for tests and local CLI use.
func GenerateKeyPair() (ed25519.PublicKey, ed25519.PrivateKey, error) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, nil, fmt.Errorf("gemmv1: keygen: %w", err)
	}
	return pub, priv, nil
}

// FindDisputedTile compares worker and challenger tile sets and returns the
// first differing position. found is false when the results are identical,
// in which case no challenge may be opened.
func FindDisputedTile(worker, challenger []State, colsC uint32) (tileI, tileJ uint32, found bool, err error) {
	if len(worker) != len(challenger) {
		return 0, 0, false, errors.New("gemmv1: tile count mismatch")
	}
	for idx := range worker {
		if worker[idx] != challenger[idx] {
			return uint32(idx) / colsC, uint32(idx) % colsC, true, nil
		}
	}
	return 0, 0, false, nil
}

// SubmitMidFromTrace is the participant-side convenience that submits the
// current midpoint state from a party's local trace artifacts.
func SubmitMidFromTrace(d *GEMMDispute, party Party, art *TileTraceArtifacts, epoch uint64) (Outcome, error) {
	mid := d.Status().Mid
	proof, err := ProveLeaf(art.Levels, mid)
	if err != nil {
		return Pending, err
	}
	return d.SubmitMid(party, art.States[mid].CanonicalBytes(), proof, epoch)
}
