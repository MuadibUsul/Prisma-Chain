package gemmv1

import (
	"crypto/ed25519"
	"crypto/sha256"
	"errors"
	"fmt"
)

// Domain separation prefixes. Every hash input in GEMM v1 starts with one of
// these byte strings, which all end in a single 0x00 byte. The GEMM v1 domain
// space is independent of the prismavm v1 hashing format.
const (
	DomainTask       = "PRISMA_GEMM_TASK_V1\x00"
	DomainAssignment = "PRISMA_GEMM_ASSIGNMENT_V1\x00"
	DomainInputTile  = "PRISMA_GEMM_INPUT_TILE_V1\x00"
	DomainOutputTile = "PRISMA_GEMM_OUTPUT_TILE_V1\x00"
	DomainTraceState = "PRISMA_GEMM_TRACE_STATE_V1\x00"
	DomainVWR        = "PRISMA_GEMM_VWR_V1\x00"
	// DomainSig covers signatures over canonical protocol objects. It is an
	// extension beyond the six hash domains above: signatures hash
	// DomainSig || CanonicalCBOR(message with signature field zeroed).
	DomainSig = "PRISMA_GEMM_SIG_V1\x00"
)

// Hash is a SHA-256 digest with value semantics, mirroring vm.Hash.
type Hash [32]byte

// hashBytes returns SHA-256 over the concatenation of parts.
func hashBytes(parts ...[]byte) Hash {
	h := sha256.New()
	for _, p := range parts {
		h.Write(p)
	}
	var out Hash
	copy(out[:], h.Sum(nil))
	return out
}

// canonicalObjectHash hashes Domain || CanonicalCBOR(v).
func canonicalObjectHash(domain string, v any) (Hash, error) {
	enc, err := EncodeCanonical(v)
	if err != nil {
		return Hash{}, err
	}
	return hashBytes([]byte(domain), enc), nil
}

// appendUint32BE appends a fixed-width big-endian uint32. All tile
// coordinates and trace steps in leaf preimages use this fixed-width form so
// that concatenation is unambiguous.
func appendUint32BE(dst []byte, v uint32) []byte {
	return append(dst, byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
}

// SignObject signs canonicalBytes (already the exact preimage to sign) with
// an Ed25519 key.
func SignObject(key ed25519.PrivateKey, canonicalBytes []byte) []byte {
	return ed25519.Sign(key, canonicalBytes)
}

// VerifyObject verifies an Ed25519 signature over canonicalBytes.
func VerifyObject(pub ed25519.PublicKey, canonicalBytes, sig []byte) bool {
	return ed25519.Verify(pub, canonicalBytes, sig)
}

// SignedBytes returns the exact bytes a signature over v covers:
// DomainSig || CanonicalCBOR(v). v must be the message struct with its
// signature field left nil.
func SignedBytes(v any) ([]byte, error) {
	enc, err := EncodeCanonical(v)
	if err != nil {
		return nil, err
	}
	return append([]byte(DomainSig), enc...), nil
}

// errBadLength is returned when a byte-slice field has a fixed required size.
func errBadLength(field string, want, got int) error {
	return fmt.Errorf("gemmv1: %s must be %d bytes, got %d", field, want, got)
}

var (
	errNilBytes  = errors.New("gemmv1: required byte field is empty")
	errBadPubKey = errors.New("gemmv1: public key must be 32-byte Ed25519")
)

func checkPubKey(b []byte) error {
	if len(b) != ed25519.PublicKeySize {
		return errBadPubKey
	}
	return nil
}

func checkHashField(field string, b []byte) error {
	if len(b) != 32 {
		return errBadLength(field, 32, len(b))
	}
	return nil
}

// signProtocolObject signs a canonical protocol object with its signature
// field left empty, then stores the Ed25519 signature.
func signProtocolObject(v any, sigField *[]byte, key ed25519.PrivateKey) error {
	*sigField = nil
	sigBytes, err := SignedBytes(v)
	if err != nil {
		return err
	}
	*sigField = SignObject(key, sigBytes)
	return nil
}

// verifyProtocolObject re-canonicalizes the object with its signature field
// emptied and checks the Ed25519 signature under pub.
func verifyProtocolObject(v any, pub []byte, sig []byte) bool {
	if err := checkPubKey(pub); err != nil || len(sig) != ed25519SigSize {
		return false
	}
	sigBytes, err := SignedBytes(v)
	if err != nil {
		return false
	}
	return VerifyObject(ed25519.PublicKey(pub), sigBytes, sig)
}

const ed25519SigSize = ed25519.SignatureSize
