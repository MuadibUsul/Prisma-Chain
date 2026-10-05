package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"prismachain/chain/x/compute"
)

// --- prismad fixture -------------------------------------------------------

var (
	prismadOnce sync.Once
	prismadPath string
	prismadErr  error
)

// prismadBin builds the node binary once per test run; the ceremony drives
// the real binary, never a mock.
func prismadBin(t *testing.T) string {
	t.Helper()
	prismadOnce.Do(func() {
		dir, err := os.MkdirTemp("", "netconfig-test-bin-*")
		if err != nil {
			prismadErr = err
			return
		}
		name := "prismad"
		if runtime.GOOS == "windows" {
			name += ".exe"
		}
		bin := filepath.Join(dir, name)
		cmd := exec.Command("go", "build", "-o", bin, "./cmd/prismad")
		cmd.Dir = "../.."
		if out, err := cmd.CombinedOutput(); err != nil {
			prismadErr = err
			t.Logf("go build output: %s", out)
			return
		}
		prismadPath = bin
	})
	if prismadErr != nil {
		t.Fatalf("build prismad: %v", prismadErr)
	}
	return prismadPath
}

func testSpec(bin string) *Spec {
	seed := func(name string) string {
		// Fixed seeds keep the ceremony reproducible across runs.
		h := [32]byte{}
		copy(h[:], name)
		return base64StdEncode(h[:])
	}
	return &Spec{
		ChainID:     "prisma-testnet-1",
		GenesisTime: "2026-11-01T00:00:00Z",
		Denom:       "uprsm",
		Validators: []Validator{
			{Moniker: "seed-a", SeedB64: seed("seed-a"), P2PExternalAddress: "203.0.113.10:26656",
				AccountFundingUprsm: 100000000000, SelfDelegationUprsm: 1000000},
			{Moniker: "seed-b", SeedB64: seed("seed-b"), P2PExternalAddress: "203.0.113.11:26656",
				AccountFundingUprsm: 100000000000, SelfDelegationUprsm: 1000000},
		},
		Faucet:      Faucet{Enabled: true, AmountUprsm: 10000000, CooldownSeconds: 3600, PerAddressCapUprsm: 100000000},
		RPC:         RPC{Laddr: "tcp://127.0.0.1:26657", GRPCLaddr: "tcp://127.0.0.1:9090"},
		P2P:         P2P{Laddr: "tcp://0.0.0.0:26656"},
		PrismadPath: bin,
	}
}

func writeSpecFile(t *testing.T, spec *Spec) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "spec.json")
	raw, err := json.MarshalIndent(spec, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func genBundle(t *testing.T, specPath, out string) {
	t.Helper()
	if err := cmdGenerate([]string{"--spec", specPath, "--out", out}); err != nil {
		t.Fatalf("generate: %v", err)
	}
}

// --- validation tests ------------------------------------------------------

func TestSpecValidationRejects(t *testing.T) {
	base := func() *Spec {
		s := testSpec("prismad")
		return s
	}
	cases := []struct {
		name   string
		mutate func(*Spec)
		want   string
	}{
		{"devnet chain id", func(s *Spec) { s.ChainID = "prisma-mv-1" }, "devnet identifier"},
		{"chain id rule", func(s *Spec) { s.ChainID = "testnet" }, "violates the rule"},
		{"stake denom", func(s *Spec) { s.Denom = "stake" }, "frozen testnet denom"},
		{"rpc wildcard", func(s *Spec) { s.RPC.Laddr = "tcp://0.0.0.0:26657" }, "wildcard"},
		{"localhost p2p", func(s *Spec) { s.Validators[0].P2PExternalAddress = "127.0.0.1:26656" }, "localhost/devnet address"},
		{"weak seed", func(s *Spec) { s.Validators[0].SeedB64 = base64StdEncode([]byte("short")) }, ">= 16 bytes"},
		{"low stake", func(s *Spec) { s.Validators[0].SelfDelegationUprsm = compute.MinBond - 1 }, "below the frozen MinBond"},
		{"bad genesis time", func(s *Spec) { s.GenesisTime = "yesterday" }, "RFC3339"},
		{"funding below stake", func(s *Spec) { s.Validators[0].AccountFundingUprsm = 1 }, "must exceed self delegation"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := base()
			tc.mutate(s)
			err := s.Validate()
			if err == nil {
				t.Fatalf("spec accepted, want rejection containing %q", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err, tc.want)
			}
		})
	}
}

func TestSpecRejectsUnknownFields(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "spec.json")
	if err := os.WriteFile(path, []byte(`{"chain_id":"prisma-testnet-1","surprise":true}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadSpec(path); err == nil || !strings.Contains(err.Error(), "surprise") {
		t.Fatalf("unknown field accepted: %v", err)
	}
}

// --- determinism -----------------------------------------------------------

func TestGenerateTwiceByteIdentical(t *testing.T) {
	if testing.Short() {
		t.Skip("ceremony test skipped in -short mode")
	}
	bin := prismadBin(t)
	spec := testSpec(bin)
	specPath := writeSpecFile(t, spec)

	out1 := filepath.Join(t.TempDir(), "bundle1")
	out2 := filepath.Join(t.TempDir(), "bundle2")
	genBundle(t, specPath, out1)
	genBundle(t, specPath, out2)

	for _, name := range []string{"genesis.json", "validators.json", "params.json", "README.md", "chain-id.txt", "manifest.json"} {
		a, err := os.ReadFile(filepath.Join(out1, name))
		if err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(filepath.Join(out2, name))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(a, b) {
			t.Fatalf("%s differs between two runs with the same spec", name)
		}
	}
}

// --- verification ----------------------------------------------------------

func TestVerifyRejectsTamperedGenesis(t *testing.T) {
	if testing.Short() {
		t.Skip("ceremony test skipped in -short mode")
	}
	bin := prismadBin(t)
	specPath := writeSpecFile(t, testSpec(bin))
	out := filepath.Join(t.TempDir(), "bundle")
	genBundle(t, specPath, out)

	genesisPath := filepath.Join(out, "genesis.json")
	raw, err := os.ReadFile(genesisPath)
	if err != nil {
		t.Fatal(err)
	}
	raw = bytes.Replace(raw, []byte(`"chain_id": "prisma-testnet-1"`), []byte(`"chain_id": "prisma-testnet-2"`), 1)
	if err := os.WriteFile(genesisPath, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = verifyBundle(out, bin, false)
	if err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("tampered genesis accepted: %v", err)
	}
}

func TestVerifyRejectsUnlistedFile(t *testing.T) {
	if testing.Short() {
		t.Skip("ceremony test skipped in -short mode")
	}
	bin := prismadBin(t)
	specPath := writeSpecFile(t, testSpec(bin))
	out := filepath.Join(t.TempDir(), "bundle")
	genBundle(t, specPath, out)

	if err := os.WriteFile(filepath.Join(out, "extra.txt"), []byte("not in the manifest"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := verifyBundle(out, bin, false)
	if err == nil || !strings.Contains(err.Error(), "unlisted file") {
		t.Fatalf("unlisted file accepted: %v", err)
	}
}

// --- params drift ----------------------------------------------------------

func TestDerivedParamsMatchChainConstants(t *testing.T) {
	spec := testSpec("prismad")
	p := DeriveParams(spec)
	if p.MinBondUprsm != compute.MinBond || p.ChallengeBlocks != compute.ChallengeBlocks ||
		p.DAWindowBlocks != compute.DAWindowBlocks || p.DAPenaltyUprsm != compute.DAPenaltyUprsm ||
		p.ChallengeRoundBlocks != compute.ChallengeRoundBlocks {
		t.Fatalf("derived params drift from the chain constants: %+v", p)
	}
	if err := checkParamsDrift(p); err != nil {
		t.Fatalf("checkParamsDrift rejects freshly derived params: %v", err)
	}
	bad := p
	bad.DAWindowBlocks = p.DAWindowBlocks + 1
	if err := checkParamsDrift(bad); err == nil {
		t.Fatalf("drift not detected")
	}
}

// --- provisioning ----------------------------------------------------------

func TestProvisionCreatesValidNodeHome(t *testing.T) {
	if testing.Short() {
		t.Skip("ceremony test skipped in -short mode")
	}
	bin := prismadBin(t)
	spec := testSpec(bin)
	specPath := writeSpecFile(t, spec)
	out := filepath.Join(t.TempDir(), "bundle")
	genBundle(t, specPath, out)

	home := filepath.Join(t.TempDir(), "home")
	if err := cmdProvision([]string{"--spec", specPath, "--bundle", out, "--moniker", "seed-b", "--home", home}); err != nil {
		t.Fatalf("provision: %v", err)
	}
	bundleGenesis, err := os.ReadFile(filepath.Join(out, "genesis.json"))
	if err != nil {
		t.Fatal(err)
	}
	homeGenesis, err := os.ReadFile(filepath.Join(home, "config", "genesis.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(bundleGenesis, homeGenesis) {
		t.Fatalf("provisioned genesis differs from the bundle genesis")
	}
	if _, err := os.Stat(filepath.Join(home, "config", "priv_validator_key.json")); err != nil {
		t.Fatalf("private validator key missing: %v", err)
	}
	cfg, err := os.ReadFile(filepath.Join(home, "config", "config.toml"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(cfg, []byte(`laddr = "tcp://127.0.0.1:26657"`)) {
		t.Fatalf("rpc laddr not applied to config.toml")
	}
	if !bytes.Contains(cfg, []byte("203.0.113.10:26656")) {
		t.Fatalf("persistent peer (the other validator) not applied")
	}
	// The provisioned home must re-init refusal-guard.
	if err := cmdProvision([]string{"--spec", specPath, "--bundle", out, "--moniker", "seed-b", "--home", home}); err == nil {
		t.Fatalf("provision overwrote a non-empty home")
	}
}

func base64StdEncode(raw []byte) string { return base64.StdEncoding.EncodeToString(raw) }
