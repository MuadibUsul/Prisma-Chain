package gemmv1

// DA_REPLICA_V1: replicated output availability for GEMM_INT8_V1.
//
// The chain already commits every 8x8 int32 output tile through
// output_root; DA_REPLICA_V1 reuses that commitment instead of defining a
// second Merkle tree. A bonded DA provider stores the canonical full C blob
// plus the tile tree, attests availability with a signed canonical object,
// and answers on-chain sampling challenges with an OutputTileProof against
// the worker's committed output_root.
//
// What this proves: permissionless replicated availability with bonded
// provider attestations and objective on-chain tile challenge-response. It
// is NOT a proof of permanent or global availability; it relies on enough
// independent providers remaining truthful, and on challenges being
// includable by the chain.

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"math"
)

// DomainDAChallengeV1 separates the DA sampling-tile derivation.
const DomainDAChallengeV1 = "PRISMA_GEMM_DA_CHALLENGE_V1\x00"

// DomainDAAttestationV1 separates DA attestation signatures. The canonical
// object is signed with the same DomainSig scheme as other protocol
// objects, but its own name is recorded for documentation and for any
// future detached verification.
const DomainDAAttestationV1 = "PRISMA_GEMM_DA_ATTESTATION_V1\x00"

// DAProtocolVersion is the version string of DA availability objects.
const DAProtocolVersion = "DA_REPLICA_V1"

// DAAttestation is the canonical, provider-signed availability claim. It is
// a claim ("I hold the full blob and can serve it until this height"), not
// a mathematical proof of future availability.
type DAAttestation struct {
	ProtocolVersion string `gemm:"protocol_version"`

	TaskID       []byte `gemm:"task_id"`
	AssignmentID []byte `gemm:"assignment_id"`
	OutputRoot   []byte `gemm:"output_root"`

	ProviderAccount []byte `gemm:"provider_account"`
	ProviderPubKey  []byte `gemm:"provider_pub_key"`

	OutputBytes uint64 `gemm:"output_bytes"`

	AvailableUntilHeight uint64 `gemm:"available_until_height"`
	AttestedHeight       uint64 `gemm:"attested_height"`

	Signature []byte `gemm:"signature"`
}

// ValidateDAAttestation checks the shape of a DA attestation before any
// signature or state checks.
func ValidateDAAttestation(a *DAAttestation) error {
	if a.ProtocolVersion != DAProtocolVersion {
		return errors.New("gemmv1: da attestation protocol_version mismatch")
	}
	for field, b := range map[string][]byte{
		"task_id": a.TaskID, "assignment_id": a.AssignmentID, "output_root": a.OutputRoot,
	} {
		if err := checkHashField(field, b); err != nil {
			return err
		}
	}
	if len(a.ProviderAccount) == 0 || len(a.ProviderAccount) > 128 {
		return errors.New("gemmv1: da provider account must be 1..128 bytes")
	}
	if err := checkPubKey(a.ProviderPubKey); err != nil {
		return err
	}
	if a.OutputBytes == 0 {
		return errors.New("gemmv1: da output_bytes must be positive")
	}
	if a.AvailableUntilHeight < a.AttestedHeight {
		return errors.New("gemmv1: da available_until must not precede attested height")
	}
	return nil
}

// SignDAAttestation fills the provider signature.
func SignDAAttestation(a *DAAttestation, key ed25519.PrivateKey) error {
	return signProtocolObject(a, &a.Signature, key)
}

// VerifyDAAttestationSignature verifies the attestation under the provider
// protocol key it carries. The signature covers the canonical object with
// the signature field empty, so the preimage is re-encoded with it zeroed.
func VerifyDAAttestationSignature(a *DAAttestation) bool {
	if err := checkPubKey(a.ProviderPubKey); err != nil || len(a.Signature) != ed25519SigSize {
		return false
	}
	unsigned := *a
	unsigned.Signature = nil
	sigBytes, err := SignedBytes(&unsigned)
	if err != nil {
		return false
	}
	return VerifyObject(ed25519.PublicKey(a.ProviderPubKey), sigBytes, a.Signature)
}

// DAChallengeTile derives the sampled output tile from committed,
// challenger-chosen and chain facts. This is a deterministic sampling
// challenge, not cryptographic randomness: its purpose is to stop a
// challenger from always probing the same tile and to expose providers
// that only stored part of the blob.
func DAChallengeTile(taskID []byte, providerAccount []byte, challengerAccount []byte, nonce []byte, openedHeight uint64, rowsC, colsC uint32) (uint32, uint32, error) {
	if rowsC == 0 || colsC == 0 {
		return 0, 0, errors.New("gemmv1: empty tile grid")
	}
	h := sha256.New()
	h.Write([]byte(DomainDAChallengeV1))
	h.Write(taskID)
	h.Write(providerAccount)
	h.Write(challengerAccount)
	h.Write(nonce)
	var heightBytes [8]byte
	binary.BigEndian.PutUint64(heightBytes[:], openedHeight)
	h.Write(heightBytes[:])
	sum := h.Sum(nil)
	total := uint64(rowsC) * uint64(colsC)
	if total == 0 || total > math.MaxUint64 {
		return 0, 0, errors.New("gemmv1: tile grid overflow")
	}
	index := binary.BigEndian.Uint64(sum[:8]) % total
	return uint32(index / uint64(colsC)), uint32(index % uint64(colsC)), nil
}

// DARequiredReplicas is the v1 quorum: at least this many distinct valid
// provider attestations are required before a task can finalize.
const DARequiredReplicas = 2

// DAMinProviders is the configured minimum provider count per task.
const DAMinProviders = 3

// DAPenaltyUprsm is the fixed, versioned testnet penalty for a provider
// that fails an on-chain DA challenge by timing out. It is deliberately a
// fixed value, not a market mechanism.
const DAPenaltyUprsm = 100_000
