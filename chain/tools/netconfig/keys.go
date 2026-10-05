package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"

	cmted25519 "github.com/cometbft/cometbft/crypto/ed25519"
	"github.com/cosmos/cosmos-sdk/crypto/keys/secp256k1"
	sdk "github.com/cosmos/cosmos-sdk/types"
)

// Deterministic key derivation (B1-03c).
//
// The validator seed is the secret; every key is derived from it with a
// domain-separated secret through the library's own derivation, so the
// ceremony is reproducible and the bundle contains public keys only:
//
//	consensus ed25519 : cometbft ed25519.GenPrivKeyFromSecret(domain||seed)
//	node key ed25519  : cometbft ed25519.GenPrivKeyFromSecret(domain||seed)
//	account secp256k1 : sdk secp256k1.GenPrivKeyFromSecret(domain||seed)
//
// Changing a domain string changes every derived key: they are part of the
// network's reproducible ceremony and are versioned as v1.
const (
	consensusKeyDomain = "prisma/netconfig/v1/consensus-key:"
	nodeKeyDomain      = "prisma/netconfig/v1/node-key:"
	accountKeyDomain   = "prisma/netconfig/v1/account-key:"
)

func sealedSecret(domain string, seed []byte) []byte {
	h := sha256.Sum256(append([]byte(domain), seed...))
	return h[:]
}

// DerivedKeys are one validator's private keys (never written to the
// public bundle; derived in memory during generate and provision).
type DerivedKeys struct {
	Consensus cmted25519.PrivKey
	Node      cmted25519.PrivKey
	Account   *secp256k1.PrivKey
}

func DeriveKeys(seed []byte) DerivedKeys {
	return DerivedKeys{
		Consensus: cmted25519.GenPrivKeyFromSecret(sealedSecret(consensusKeyDomain, seed)),
		Node:      cmted25519.GenPrivKeyFromSecret(sealedSecret(nodeKeyDomain, seed)),
		Account:   secp256k1.GenPrivKeyFromSecret(sealedSecret(accountKeyDomain, seed)),
	}
}

var prefixOnce sync.Once

// configureSDKPrefixes mirrors chain/cmd/prismad/cmd/root.go so bech32
// addresses rendered here match the node exactly.
func configureSDKPrefixes() {
	prefixOnce.Do(func() {
		cfg := sdk.GetConfig()
		cfg.SetBech32PrefixForAccount("prsm", "prsmpub")
		cfg.SetBech32PrefixForValidator("prsmvaloper", "prsmvaloperpub")
		cfg.SetBech32PrefixForConsensusNode("prsmvalcons", "prsmvalconspub")
		cfg.Seal()
	})
}

// AccountAddress is the bech32 account address (prefix prsm) of the key.
func (k DerivedKeys) AccountAddress() string {
	return sdk.AccAddress(k.Account.PubKey().Address()).String()
}

// NodeID is the comet node id (hex address of the node key).
func (k DerivedKeys) NodeID() string {
	return hex.EncodeToString(k.Node.PubKey().Address())
}

// --- comet key file formats (public key material only in artifacts) -------

type cometKeyJSON struct {
	Type  string `json:"type"`
	Value string `json:"value"`
}

type cometPrivValidatorKey struct {
	Address string       `json:"address"`
	PubKey  cometKeyJSON `json:"pub_key"`
	PrivKey cometKeyJSON `json:"priv_key"`
}

type cometNodeKey struct {
	PrivKey cometKeyJSON `json:"priv_key"`
}

func b64(b []byte) string { return base64.StdEncoding.EncodeToString(b) }

// writePrivValidatorKey writes config/priv_validator_key.json in the exact
// format cometbft expects, from the derived consensus key.
func writePrivValidatorKey(home string, priv cmted25519.PrivKey) error {
	doc := cometPrivValidatorKey{
		Address: hex.EncodeToString(priv.PubKey().Address()),
		PubKey:  cometKeyJSON{"tendermint/PubKeyEd25519", b64(priv.PubKey().Bytes())},
		PrivKey: cometKeyJSON{"tendermint/PrivKeyEd25519", b64(priv.Bytes())},
	}
	return writeJSONFile(filepath.Join(home, "config", "priv_validator_key.json"), doc)
}

func writeNodeKey(home string, priv cmted25519.PrivKey) error {
	doc := cometNodeKey{PrivKey: cometKeyJSON{"tendermint/PrivKeyEd25519", b64(priv.Bytes())}}
	return writeJSONFile(filepath.Join(home, "config", "node_key.json"), doc)
}

// ensurePrivValidatorState writes the initial signing state if the node
// home does not have one yet (deterministic: height 0).
func ensurePrivValidatorState(home string) error {
	path := filepath.Join(home, "data", "priv_validator_state.json")
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	doc := map[string]any{"height": "0", "round": 0, "step": 0}
	raw, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}

func writeJSONFile(path string, doc any) error {
	raw, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}

// PublicIdentity is what the bundle records for one validator: public keys
// and addresses only, never seed or private key material.
type PublicIdentity struct {
	Moniker             string `json:"moniker"`
	NodeID              string `json:"node_id"`
	AccountAddress      string `json:"account_address"`
	ConsensusAddress    string `json:"consensus_address"`
	ConsensusPubKeyB64  string `json:"consensus_pubkey_b64"`
	P2PExternalAddress  string `json:"p2p_external_address"`
	AccountFundingUprsm uint64 `json:"account_funding_uprsm"`
	SelfDelegationUprsm uint64 `json:"self_delegation_uprsm"`
}

func PublicIdentityOf(moniker string, v Validator, k DerivedKeys) PublicIdentity {
	return PublicIdentity{
		Moniker:             moniker,
		NodeID:              k.NodeID(),
		AccountAddress:      k.AccountAddress(),
		ConsensusAddress:    hex.EncodeToString(k.Consensus.PubKey().Address()),
		ConsensusPubKeyB64:  b64(k.Consensus.PubKey().Bytes()),
		P2PExternalAddress:  v.P2PExternalAddress,
		AccountFundingUprsm: v.AccountFundingUprsm,
		SelfDelegationUprsm: v.SelfDelegationUprsm,
	}
}
