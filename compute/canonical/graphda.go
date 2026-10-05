package canonical

// GRAPH_DA_ATTESTATION_V2 (roadmap A4): typed DA availability attestation
// for a GraphVerificationBundleV2 task.  A provider signs a storage
// promise bound to the committed graph identity and manifest root; the
// provider never executes the Transformer and the chain only ever checks
// signatures, bindings and (via the challenge path) typed chunk proofs.

import (
	"crypto/ed25519"
	"errors"
)

const (
	ProtocolVersionGraphDA      = "GRAPH_DA_ATTESTATION_V2/1.0.0"
	DomainGraphDAAttestationV2  = "PRISMA_GRAPH_DA_ATTESTATION_V2\x00"
	DomainGraphDAChallengeV1    = "PRISMA_GRAPH_DA_CHALLENGE_V1\x00"
	GraphDARequiredReplicas     = 2
	GraphDAArtifactKind         = "GRAPH_VERIFICATION_BUNDLE_V2"
)

// GraphDAAttestationV2 is the canonical signed provider object.
type GraphDAAttestationV2 struct {
	ProtocolVersion string `gemm:"protocol_version" json:"protocol_version"`

	GraphTaskID uint64 `gemm:"graph_task_id" json:"graph_task_id"`
	GraphID     []byte `gemm:"graph_id" json:"graph_id"`
	// ManifestRootV2 binds the storage promise to the worker's locked
	// commitment (the chain knows it; the artifact hash stays an in-bundle
	// integrity detail).
	ManifestRootV2  []byte `gemm:"manifest_root_v2" json:"manifest_root_v2"`
	FinalOutputRoot []byte `gemm:"final_output_root" json:"final_output_root"`

	ProviderAccount []byte `gemm:"provider_account" json:"provider_account"`
	ProviderPubKey  []byte `gemm:"provider_pub_key" json:"provider_pub_key"`

	BundleBytes uint64 `gemm:"bundle_bytes" json:"bundle_bytes"`

	AvailableUntilHeight uint64 `gemm:"available_until_height" json:"available_until_height"`
	AttestedHeight       uint64 `gemm:"attested_height" json:"attested_height"`

	Signature []byte `gemm:"signature" json:"signature"`
}

func (a *GraphDAAttestationV2) signBytes() ([]byte, error) {
	unsigned := *a
	unsigned.Signature = nil
	enc, err := EncodeCanonical(&unsigned)
	if err != nil {
		return nil, err
	}
	return append([]byte(DomainGraphDAAttestationV2), enc...), nil
}

// SignGraphDAAttestationV2 fills the provider signature.
func SignGraphDAAttestationV2(a *GraphDAAttestationV2, key ed25519.PrivateKey) error {
	if len(a.ProviderPubKey) != 32 {
		return errors.New("canonical/v2: graph DA attestation needs a 32-byte provider key")
	}
	preimage, err := a.signBytes()
	if err != nil {
		return err
	}
	a.Signature = ed25519.Sign(key, preimage)
	return nil
}

// VerifyGraphDAAttestationV2Signature checks the signature (own domain:
// a gemmv1 DA attestation can never validate here and vice versa).
func VerifyGraphDAAttestationV2Signature(a *GraphDAAttestationV2) bool {
	if len(a.ProviderPubKey) != 32 || len(a.Signature) != ed25519.SignatureSize {
		return false
	}
	preimage, err := a.signBytes()
	if err != nil {
		return false
	}
	return ed25519.Verify(a.ProviderPubKey, preimage, a.Signature)
}

// ValidateGraphDAAttestationV2 checks the static shape.
func ValidateGraphDAAttestationV2(a *GraphDAAttestationV2) error {
	if a.ProtocolVersion != ProtocolVersionGraphDA {
		return errors.New("canonical/v2: graph DA attestation version mismatch")
	}
	if len(a.GraphID) != 32 || len(a.ManifestRootV2) != 32 || len(a.FinalOutputRoot) != 32 {
		return errors.New("canonical/v2: graph DA attestation roots malformed")
	}
	if len(a.ProviderAccount) == 0 || len(a.ProviderPubKey) != 32 {
		return errors.New("canonical/v2: graph DA attestation provider malformed")
	}
	if a.BundleBytes == 0 {
		return errors.New("canonical/v2: graph DA attestation bundle bytes missing")
	}
	if a.AvailableUntilHeight < a.AttestedHeight {
		return errors.New("canonical/v2: graph DA attestation availability window inverted")
	}
	return nil
}
