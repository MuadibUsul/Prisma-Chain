package main

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"prismachain/chain/x/compute"
)

// Spec is the generator input. It contains the validator seeds (secrets);
// the generated bundle contains public keys only.
type Spec struct {
	ChainID     string      `json:"chain_id"`
	GenesisTime string      `json:"genesis_time"`
	Denom       string      `json:"denom"`
	Validators  []Validator `json:"validators"`
	Faucet      Faucet      `json:"faucet"`
	RPC         RPC         `json:"rpc"`
	P2P         P2P         `json:"p2p"`
	PrismadPath string      `json:"prismad_binary"`
}

type Validator struct {
	Moniker             string `json:"moniker"`
	SeedB64             string `json:"seed_b64"`
	P2PExternalAddress  string `json:"p2p_external_address"`
	AccountFundingUprsm uint64 `json:"account_funding_uprsm"`
	SelfDelegationUprsm uint64 `json:"self_delegation_uprsm"`
}

type Faucet struct {
	Enabled            bool   `json:"enabled"`
	AmountUprsm        uint64 `json:"amount_uprsm"`
	CooldownSeconds    uint64 `json:"cooldown_seconds"`
	PerAddressCapUprsm uint64 `json:"per_address_cap_uprsm"`
}

type RPC struct {
	Laddr     string `json:"laddr"`
	GRPCLaddr string `json:"grpc_laddr"`
}

type P2P struct {
	Laddr string `json:"laddr"`
}

const testnetDenom = "uprsm"

// chain-id rule (B1-03a): prisma-<network>-<generation>, and never a devnet
// identifier. Devnet identifiers used historically: prisma-mv-N (multi
// validator), prisma-local-N (single node), prisma-b1smoke-N (CI smoke).
var chainIDPattern = regexp.MustCompile(`^prisma-[a-z0-9]+-[0-9]+$`)

var devnetChainIDs = regexp.MustCompile(`^prisma-(mv|local|b1smoke)-[0-9]+$`)

var monikerPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)

var hostPortPattern = regexp.MustCompile(`^[A-Za-z0-9.:\[\]-]+:[0-9]{1,5}$`)

func LoadSpec(path string) (*Spec, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read spec: %w", err)
	}
	var spec Spec
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&spec); err != nil {
		return nil, fmt.Errorf("parse spec: %w", err)
	}
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	return &spec, nil
}

// Validate enforces the B1-03 rules that keep devnet assumptions and
// unsafe defaults out of a public network config.
func (s *Spec) Validate() error {
	if !chainIDPattern.MatchString(s.ChainID) {
		return fmt.Errorf("chain_id %q violates the rule prisma-<network>-<generation>", s.ChainID)
	}
	if devnetChainIDs.MatchString(s.ChainID) {
		return fmt.Errorf("chain_id %q is a devnet identifier (prisma-mv-*/prisma-local-*)", s.ChainID)
	}
	if s.Denom != testnetDenom {
		return fmt.Errorf("denom %q is not the frozen testnet denom %q", s.Denom, testnetDenom)
	}
	if _, err := time.Parse(time.RFC3339, s.GenesisTime); err != nil {
		return fmt.Errorf("genesis_time %q is not RFC3339: %w", s.GenesisTime, err)
	}
	if len(s.Validators) == 0 {
		return fmt.Errorf("at least one validator is required")
	}
	seen := map[string]bool{}
	for i, v := range s.Validators {
		if !monikerPattern.MatchString(v.Moniker) {
			return fmt.Errorf("validator[%d]: moniker %q is not a safe name", i, v.Moniker)
		}
		if seen[v.Moniker] {
			return fmt.Errorf("validator[%d]: duplicate moniker %q", i, v.Moniker)
		}
		seen[v.Moniker] = true
		seed, err := base64.StdEncoding.DecodeString(v.SeedB64)
		if err != nil || len(seed) < 16 {
			return fmt.Errorf("validator[%d] (%s): seed_b64 must be base64 of >= 16 bytes", i, v.Moniker)
		}
		if !hostPortPattern.MatchString(v.P2PExternalAddress) {
			return fmt.Errorf("validator[%d] (%s): p2p_external_address %q is not host:port", i, v.Moniker, v.P2PExternalAddress)
		}
		host := v.P2PExternalAddress[:strings.LastIndex(v.P2PExternalAddress, ":")]
		switch host {
		case "", "127.0.0.1", "localhost", "0.0.0.0", "::1":
			return fmt.Errorf("validator[%d] (%s): p2p_external_address %q is a localhost/devnet address", i, v.Moniker, v.P2PExternalAddress)
		}
		if v.SelfDelegationUprsm < compute.MinBond {
			return fmt.Errorf("validator[%d] (%s): self_delegation %d is below the frozen MinBond %d",
				i, v.Moniker, v.SelfDelegationUprsm, compute.MinBond)
		}
		if v.AccountFundingUprsm <= v.SelfDelegationUprsm {
			return fmt.Errorf("validator[%d] (%s): funding must exceed self delegation", i, v.Moniker)
		}
	}
	if s.Faucet.Enabled {
		if s.Faucet.AmountUprsm == 0 || s.Faucet.CooldownSeconds == 0 {
			return fmt.Errorf("faucet: amount and cooldown must be positive when enabled")
		}
		if s.Faucet.PerAddressCapUprsm < s.Faucet.AmountUprsm {
			return fmt.Errorf("faucet: per_address_cap must be >= amount")
		}
	}
	if s.RPC.Laddr == "" || strings.Contains(s.RPC.Laddr, "0.0.0.0") {
		return fmt.Errorf("rpc.laddr %q must be set and must not be a 0.0.0.0 wildcard (B1-03d; the public profile is B1-02)", s.RPC.Laddr)
	}
	if s.P2P.Laddr == "" {
		return fmt.Errorf("p2p.laddr must be set explicitly (e.g. tcp://0.0.0.0:26656)")
	}
	return nil
}

// PrismadBinary returns the node binary to drive the ceremony with.
func (s *Spec) PrismadBinary() string {
	if s.PrismadPath != "" {
		return s.PrismadPath
	}
	return "prismad"
}

// NetworkParams is the derived, machine-checked parameter record written
// into the bundle. Windows/bounds/penalties come from the chain code.
type NetworkParams struct {
	Protocol             string            `json:"protocol"`
	Denom                string            `json:"denom"`
	MinBondUprsm         uint64            `json:"min_bond_uprsm"`
	ChallengeBlocks      uint64            `json:"challenge_blocks"`
	ChallengeRoundBlocks uint64            `json:"challenge_round_blocks"`
	DAWindowBlocks       uint64            `json:"da_window_blocks"`
	DAChallengeBlocks    uint64            `json:"da_challenge_blocks"`
	DAPenaltyUprsm       uint64            `json:"da_penalty_uprsm"`
	MaxQueuedChallenges  int               `json:"max_queued_challenges"`
	LightTokenPriceUprsm uint64            `json:"light_token_price_uprsm"`
	GraphBounds          map[string]int    `json:"graph_bounds"`
	DAQuorumNote         string            `json:"da_quorum_rule"`
	RPC                  RPC               `json:"rpc"`
	P2P                  P2P               `json:"p2p"`
	Faucet               Faucet            `json:"faucet"`
	Sources              map[string]string `json:"sources"`
}

// DeriveParams builds the parameter record from the frozen chain constants.
func DeriveParams(spec *Spec) NetworkParams {
	return NetworkParams{
		Protocol:             "PHASE_F FROZEN / CANONICAL_GRAPH_V2",
		Denom:                spec.Denom,
		MinBondUprsm:         compute.MinBond,
		ChallengeBlocks:      compute.ChallengeBlocks,
		ChallengeRoundBlocks: compute.ChallengeRoundBlocks,
		DAWindowBlocks:       compute.DAWindowBlocks,
		DAChallengeBlocks:    compute.DAChallengeBlocks,
		DAPenaltyUprsm:       compute.DAPenaltyUprsm,
		MaxQueuedChallenges:  compute.MaxQueuedChallenges,
		LightTokenPriceUprsm: compute.LightTokenPriceUprsm,
		GraphBounds: map[string]int{
			"max_json_bytes":          compute.MaxGraphJSONBytes,
			"max_nodes":               compute.MaxGraphNodes,
			"max_inputs":              compute.MaxGraphInputs,
			"max_evidence_chunks":     compute.MaxGraphEvidenceChunks,
			"max_outputs":             compute.MaxGraphOutputs,
			"challenge_bond_multiple": compute.MaxGraphChallengeBondMultiple,
		},
		DAQuorumNote: "quorum is the runtime registry rule len(attesters) == 2 " +
			"(chain/x/compute/keeper.go, gemm_receipt.go); it is a code literal, not a named constant",
		RPC:    spec.RPC,
		P2P:    spec.P2P,
		Faucet: spec.Faucet,
		Sources: map[string]string{
			"windows_and_bond": "chain/x/compute/keeper.go",
			"da":               "chain/x/compute/gemm_da.go, compute/gemmv1/da.go",
			"graph_bounds":     "chain/x/compute/graph.go",
			"protocol":         "docs/phase-f-freeze.json",
		},
	}
}

// NewSeeds prints fresh validator seeds; the operator keeps them secret.
func newSeed() (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(buf), nil
}
