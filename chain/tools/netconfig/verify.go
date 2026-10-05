package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"prismachain/chain/x/compute"
)

// setTomlInSection replaces `key = "..."` inside [section] (top-level when
// section is ""). Fails when the key is absent, so config drift is never
// silent. Comet's config.toml has several `laddr` keys in different
// sections, which is why this is section-scoped.
func setTomlInSection(content, section, key, value string) (string, error) {
	lines := strings.Split(content, "\n")
	current := ""
	found := false
	keyRe := regexp.MustCompile(`^\s*` + regexp.QuoteMeta(key) + `\s*=`)
	sectionRe := regexp.MustCompile(`^\s*\[([^\]]+)\]\s*$`)
	for i, line := range lines {
		if m := sectionRe.FindStringSubmatch(line); m != nil {
			current = m[1]
			continue
		}
		if current == section && keyRe.MatchString(line) {
			lines[i] = fmt.Sprintf("%s = %q", key, value)
			found = true
		}
	}
	if !found {
		return "", fmt.Errorf("config key %q not found in section [%s]", key, section)
	}
	return strings.Join(lines, "\n"), nil
}

// setTomlRawInSection is setTomlInSection for non-string TOML values
// (booleans, numbers): the value is written verbatim, without quotes.
func setTomlRawInSection(content, section, key, rawValue string) (string, error) {
	lines := strings.Split(content, "\n")
	current := ""
	found := false
	keyRe := regexp.MustCompile(`^\s*` + regexp.QuoteMeta(key) + `\s*=`)
	sectionRe := regexp.MustCompile(`^\s*\[([^\]]+)\]\s*$`)
	for i, line := range lines {
		if m := sectionRe.FindStringSubmatch(line); m != nil {
			current = m[1]
			continue
		}
		if current == section && keyRe.MatchString(line) {
			lines[i] = fmt.Sprintf("%s = %s", key, rawValue)
			found = true
		}
	}
	if !found {
		return "", fmt.Errorf("config key %q not found in section [%s]", key, section)
	}
	return strings.Join(lines, "\n"), nil
}

// applyNodeConfig applies the network config policy to a node home: the
// given listen addresses, the node's external address and its peers.
// No 0.0.0.0 RPC wildcard, explicit p2p laddr, no localhost peers.
func applyNodeConfig(home string, spec *Spec, v Validator, persistentPeers string) error {
	cfgPath := filepath.Join(home, "config", "config.toml")
	raw, err := os.ReadFile(cfgPath)
	if err != nil {
		return err
	}
	content := string(raw)
	for _, step := range []struct{ section, key, value string }{
		{"", "moniker", v.Moniker},
		{"p2p", "laddr", spec.P2P.Laddr},
		{"p2p", "external_address", v.P2PExternalAddress},
		{"p2p", "persistent_peers", persistentPeers},
		{"rpc", "laddr", spec.RPC.Laddr},
	} {
		content, err = setTomlInSection(content, step.section, step.key, step.value)
		if err != nil {
			return fmt.Errorf("%s: %w", cfgPath, err)
		}
	}
	if err := os.WriteFile(cfgPath, []byte(content), 0o644); err != nil {
		return err
	}
	appPath := filepath.Join(home, "config", "app.toml")
	appRaw, err := os.ReadFile(appPath)
	if err != nil {
		return err
	}
	appContent, err := setTomlInSection(string(appRaw), "", "minimum-gas-prices", "0uprsm")
	if err != nil {
		return fmt.Errorf("%s: %w", appPath, err)
	}
	return os.WriteFile(appPath, []byte(appContent), 0o644)
}

// peersFor returns the persistent_peers string for one validator: every
// other validator's node id at its external address, sorted, so the value
// is deterministic.
func peersFor(spec *Spec, moniker string) (string, error) {
	peers := []string{}
	for _, v := range spec.Validators {
		if v.Moniker == moniker {
			continue
		}
		seed, err := decodeSeed(v.SeedB64)
		if err != nil {
			return "", err
		}
		keys := DeriveKeys(seed)
		peers = append(peers, keys.NodeID()+"@"+v.P2PExternalAddress)
	}
	sort.Strings(peers)
	return strings.Join(peers, ","), nil
}

// --- bundle verification ---------------------------------------------------

type verifyResult struct {
	Files    int
	ChainID  string
	Checks   []string
	Warnings []string
}

func verifyBundle(dir, prismad string, sdkCheck bool) (*verifyResult, error) {
	res := &verifyResult{}
	manifestRaw, err := os.ReadFile(filepath.Join(dir, manifestName))
	if err != nil {
		return nil, fmt.Errorf("read manifest: %w", err)
	}
	var m bundleManifest
	if err := json.Unmarshal(manifestRaw, &m); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}
	res.ChainID = m.ChainID

	listed := map[string]bool{}
	for _, e := range m.Files {
		full := filepath.Join(dir, filepath.FromSlash(e.Path))
		sum, size, err := fileSha256(full)
		if err != nil {
			return nil, fmt.Errorf("manifest lists %s but it cannot be read: %w", e.Path, err)
		}
		if sum != e.Sha256 {
			return nil, fmt.Errorf("checksum mismatch for %s: manifest %s, actual %s", e.Path, e.Sha256, sum)
		}
		if size != e.Bytes {
			return nil, fmt.Errorf("size mismatch for %s: manifest %d, actual %d", e.Path, e.Bytes, size)
		}
		listed[e.Path] = true
		res.Files++
	}
	err = filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || d.Name() == manifestName {
			return err
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		if !listed[filepath.ToSlash(rel)] {
			return fmt.Errorf("unlisted file in bundle: %s", rel)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	res.Checks = append(res.Checks, "manifest checksums match for every file; no unlisted files")

	chainIDRaw, err := os.ReadFile(filepath.Join(dir, "chain-id.txt"))
	if err != nil {
		return nil, err
	}
	chainID := strings.TrimSpace(string(chainIDRaw))
	if chainID != m.ChainID {
		return nil, fmt.Errorf("chain-id.txt %q != manifest chain_id %q", chainID, m.ChainID)
	}
	if !chainIDPattern.MatchString(chainID) {
		return nil, fmt.Errorf("chain_id %q violates the rule prisma-<network>-<generation>", chainID)
	}
	if devnetChainIDs.MatchString(chainID) {
		return nil, fmt.Errorf("chain_id %q is a devnet identifier", chainID)
	}
	res.Checks = append(res.Checks, "chain-id rule satisfied and not a devnet id")

	genesisRaw, err := os.ReadFile(filepath.Join(dir, "genesis.json"))
	if err != nil {
		return nil, err
	}
	if bytes.Contains(genesisRaw, []byte(`"stake"`)) {
		return nil, fmt.Errorf("genesis contains the SDK default denom \"stake\" (devnet leak)")
	}
	var genesis map[string]any
	if err := json.Unmarshal(genesisRaw, &genesis); err != nil {
		return nil, fmt.Errorf("parse genesis: %w", err)
	}
	if got, _ := genesis["chain_id"].(string); got != chainID {
		return nil, fmt.Errorf("genesis chain_id %q != %q", got, chainID)
	}
	for _, p := range [][]string{
		{"app_state", "staking", "params", "bond_denom"},
		{"app_state", "mint", "params", "mint_denom"},
	} {
		v, err := getPath(genesis, p...)
		if err != nil {
			return nil, err
		}
		if v != testnetDenom {
			return nil, fmt.Errorf("%s = %v, want %q", strings.Join(p, "."), v, testnetDenom)
		}
	}
	if inflation, err := getPath(genesis, "app_state", "mint", "minter", "inflation"); err == nil && inflation != "0.000000000000000000" {
		res.Warnings = append(res.Warnings, fmt.Sprintf("mint inflation is %v (expected zero for testnet issuance policy)", inflation))
	}
	res.Checks = append(res.Checks, "genesis denom policy satisfied (uprsm everywhere, no \"stake\")")

	validatorsRaw, err := os.ReadFile(filepath.Join(dir, "validators.json"))
	if err != nil {
		return nil, err
	}
	var valDoc struct {
		ChainID    string           `json:"chain_id"`
		Validators []PublicIdentity `json:"validators"`
	}
	if err := json.Unmarshal(validatorsRaw, &valDoc); err != nil {
		return nil, err
	}
	if valDoc.ChainID != chainID {
		return nil, fmt.Errorf("validators.json chain_id %q != %q", valDoc.ChainID, chainID)
	}
	txsAny, err := getPath(genesis, "app_state", "genutil", "gen_txs")
	if err != nil {
		return nil, err
	}
	txs, _ := txsAny.([]any)
	if len(txs) != len(valDoc.Validators) {
		return nil, fmt.Errorf("genesis has %d gentxs but validators.json lists %d", len(txs), len(valDoc.Validators))
	}
	for i, tx := range txs {
		amount, err := gentxSelfDelegation(tx)
		if err != nil {
			return nil, fmt.Errorf("gentx[%d]: %w", i, err)
		}
		if amount < compute.MinBond {
			return nil, fmt.Errorf("gentx[%d]: self delegation %d below frozen MinBond %d", i, amount, compute.MinBond)
		}
	}
	for _, v := range valDoc.Validators {
		host := v.P2PExternalAddress[:strings.LastIndex(v.P2PExternalAddress, ":")]
		switch host {
		case "", "127.0.0.1", "localhost", "0.0.0.0", "::1":
			return nil, fmt.Errorf("validator %s: p2p address %q is a localhost/devnet address", v.Moniker, v.P2PExternalAddress)
		}
	}
	res.Checks = append(res.Checks, "gentx count, self-delegation floor and public p2p addresses verified")

	paramsRaw, err := os.ReadFile(filepath.Join(dir, "params.json"))
	if err != nil {
		return nil, err
	}
	var params NetworkParams
	if err := json.Unmarshal(paramsRaw, &params); err != nil {
		return nil, err
	}
	if err := checkParamsDrift(params); err != nil {
		return nil, err
	}
	if params.RPC.Laddr == "" || strings.Contains(params.RPC.Laddr, "0.0.0.0") {
		return nil, fmt.Errorf("params rpc.laddr %q is empty or a wildcard", params.RPC.Laddr)
	}
	res.Checks = append(res.Checks, "params.json matches the frozen chain constants; rpc laddr is not a wildcard")

	if sdkCheck {
		tmp, err := os.MkdirTemp("", "netconfig-validate-*")
		if err != nil {
			return nil, err
		}
		defer os.RemoveAll(tmp)
		if err := os.MkdirAll(filepath.Join(tmp, "config"), 0o755); err != nil {
			return nil, err
		}
		if err := os.WriteFile(filepath.Join(tmp, "config", "genesis.json"), genesisRaw, 0o644); err != nil {
			return nil, err
		}
		if _, err := runPrismad(prismad, "genesis", "validate", "--home", tmp); err != nil {
			return nil, fmt.Errorf("prismad genesis validate: %w", err)
		}
		res.Checks = append(res.Checks, "prismad genesis validate: PASS")
	}
	return res, nil
}

// checkParamsDrift re-derives the network parameters from the chain code
// and fails if the recorded bundle disagrees — the bundle cannot silently
// drift from the frozen constants.
func checkParamsDrift(p NetworkParams) error {
	mismatch := func(name string, got, want any) error {
		return fmt.Errorf("params.json %s = %v, chain code has %v", name, got, want)
	}
	if p.MinBondUprsm != compute.MinBond {
		return mismatch("min_bond_uprsm", p.MinBondUprsm, compute.MinBond)
	}
	if p.ChallengeBlocks != compute.ChallengeBlocks {
		return mismatch("challenge_blocks", p.ChallengeBlocks, compute.ChallengeBlocks)
	}
	if p.ChallengeRoundBlocks != compute.ChallengeRoundBlocks {
		return mismatch("challenge_round_blocks", p.ChallengeRoundBlocks, compute.ChallengeRoundBlocks)
	}
	if p.DAWindowBlocks != compute.DAWindowBlocks {
		return mismatch("da_window_blocks", p.DAWindowBlocks, compute.DAWindowBlocks)
	}
	if p.DAChallengeBlocks != compute.DAChallengeBlocks {
		return mismatch("da_challenge_blocks", p.DAChallengeBlocks, compute.DAChallengeBlocks)
	}
	if p.DAPenaltyUprsm != compute.DAPenaltyUprsm {
		return mismatch("da_penalty_uprsm", p.DAPenaltyUprsm, compute.DAPenaltyUprsm)
	}
	if p.MaxQueuedChallenges != compute.MaxQueuedChallenges {
		return mismatch("max_queued_challenges", p.MaxQueuedChallenges, compute.MaxQueuedChallenges)
	}
	if p.GraphBounds["max_nodes"] != compute.MaxGraphNodes {
		return mismatch("graph_bounds.max_nodes", p.GraphBounds["max_nodes"], compute.MaxGraphNodes)
	}
	return nil
}

func cmdVerify(args []string) error {
	fs := flag.NewFlagSet("verify", flag.ContinueOnError)
	dir := fs.String("dir", "", "bundle directory")
	prismad := fs.String("prismad", "prismad", "prismad binary for the SDK-level genesis validation")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dir == "" {
		return fmt.Errorf("verify requires --dir")
	}
	res, err := verifyBundle(*dir, *prismad, true)
	if err != nil {
		return err
	}
	fmt.Printf("netconfig: bundle %s verified (chain-id %s, %d files)\n", *dir, res.ChainID, res.Files)
	for _, c := range res.Checks {
		fmt.Printf("  PASS  %s\n", c)
	}
	for _, w := range res.Warnings {
		fmt.Printf("  WARN  %s\n", w)
	}
	return nil
}

func cmdProvision(args []string) error {
	fs := flag.NewFlagSet("provision", flag.ContinueOnError)
	specPath := fs.String("spec", "", "path to the spec JSON (contains the validator seed)")
	bundleDir := fs.String("bundle", "", "verified bundle directory")
	moniker := fs.String("moniker", "", "which validator to provision")
	home := fs.String("home", "", "node home directory to create (private keys are written here)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *specPath == "" || *bundleDir == "" || *moniker == "" || *home == "" {
		return fmt.Errorf("provision requires --spec, --bundle, --moniker and --home")
	}
	spec, err := LoadSpec(*specPath)
	if err != nil {
		return err
	}
	configureSDKPrefixes()
	var target *Validator
	for i := range spec.Validators {
		if spec.Validators[i].Moniker == *moniker {
			target = &spec.Validators[i]
		}
	}
	if target == nil {
		return fmt.Errorf("spec has no validator %q", *moniker)
	}
	if _, err := verifyBundle(*bundleDir, spec.PrismadBinary(), true); err != nil {
		return fmt.Errorf("bundle failed verification: %w", err)
	}
	if entries, err := os.ReadDir(*home); err == nil && len(entries) > 0 {
		return fmt.Errorf("home %s is not empty; refusing to overwrite", *home)
	}
	if _, err := runPrismad(spec.PrismadBinary(), "init", *moniker, "--chain-id", spec.ChainID,
		"--default-denom", spec.Denom, "--home", *home); err != nil {
		return err
	}
	seed, err := decodeSeed(target.SeedB64)
	if err != nil {
		return err
	}
	keys := DeriveKeys(seed)
	if err := writePrivValidatorKey(*home, keys.Consensus); err != nil {
		return err
	}
	if err := writeNodeKey(*home, keys.Node); err != nil {
		return err
	}
	if err := ensurePrivValidatorState(*home); err != nil {
		return err
	}
	if err := pinClientToml(*home); err != nil {
		return err
	}
	genesisRaw, err := os.ReadFile(filepath.Join(*bundleDir, "genesis.json"))
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*home, "config", "genesis.json"), genesisRaw, 0o644); err != nil {
		return err
	}
	peers, err := peersFor(spec, *moniker)
	if err != nil {
		return err
	}
	if err := applyNodeConfig(*home, spec, *target, peers); err != nil {
		return err
	}
	if _, err := runPrismad(spec.PrismadBinary(), "genesis", "validate", "--home", *home); err != nil {
		return fmt.Errorf("provisioned home failed genesis validation: %w", err)
	}
	if keys.NodeID() != "" && keys.AccountAddress() != "" {
		fmt.Printf("netconfig: provisioned %s\n", *moniker)
		fmt.Printf("  home            : %s\n", *home)
		fmt.Printf("  node id         : %s\n", keys.NodeID())
		fmt.Printf("  account address : %s (bech32)\n", keys.AccountAddress())
		fmt.Printf("  persistent peers: %s\n", peers)
		fmt.Printf("  start with: prismad start --home %s --minimum-gas-prices 0uprsm\n", *home)
	}
	return nil
}
