package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

func base64Std(s string) ([]byte, error) { return base64.StdEncoding.DecodeString(s) }

// applyGenesisPolicy enforces the network's genesis policy on top of the
// module defaults that `prismad init --default-denom` derived:
//
//   - every denom-bearing parameter must be the testnet denom (checked,
//     hard error otherwise — the generator never silently rewrites a
//     mismatch)
//   - testnet issuance is disabled: zero mint inflation until Prisma's
//     separate, audited base-task issuance schedule exists
func applyGenesisPolicy(path, denom string) error {
	doc, err := readJSONDoc(path)
	if err != nil {
		return err
	}
	denomPaths := [][]string{
		{"app_state", "staking", "params", "bond_denom"},
		{"app_state", "mint", "params", "mint_denom"},
	}
	for _, p := range denomPaths {
		v, err := getPath(doc, p...)
		if err != nil {
			return err
		}
		if v != denom {
			return fmt.Errorf("%s = %v, want %q (init --default-denom did not take effect)", strings.Join(p, "."), v, denom)
		}
	}
	for _, p := range [][]string{
		{"app_state", "gov", "params", "min_deposit"},
		{"app_state", "gov", "params", "expedited_min_deposit"},
	} {
		v, err := getPath(doc, p...)
		if err != nil {
			return err
		}
		coins, ok := v.([]any)
		if !ok {
			return fmt.Errorf("%s is not a coin array", strings.Join(p, "."))
		}
		for i, c := range coins {
			coin, ok := c.(map[string]any)
			if !ok {
				return fmt.Errorf("%s[%d] is not a coin object", strings.Join(p, "."), i)
			}
			if coin["denom"] != denom {
				return fmt.Errorf("%s[%d].denom = %v, want %q", strings.Join(p, "."), i, coin["denom"], denom)
			}
		}
	}
	zero := "0.000000000000000000"
	for _, p := range [][]string{
		{"app_state", "mint", "minter", "inflation"},
		{"app_state", "mint", "params", "inflation_rate_change"},
		{"app_state", "mint", "params", "inflation_max"},
		{"app_state", "mint", "params", "inflation_min"},
	} {
		if _, err := getPath(doc, p...); err != nil {
			return err
		}
		if err := setPath(doc, zero, p...); err != nil {
			return err
		}
	}
	return writeJSONDoc(path, doc)
}

// normalizeGenesis fixes the genesis time and orders gentxs so two runs of
// the generator with the same spec produce byte-identical genesis files.
func normalizeGenesis(path, genesisTime string) ([]byte, error) {
	if _, err := time.Parse(time.RFC3339, genesisTime); err != nil {
		return nil, fmt.Errorf("genesis_time %q: %w", genesisTime, err)
	}
	doc, err := readJSONDoc(path)
	if err != nil {
		return nil, err
	}
	doc["genesis_time"] = genesisTime
	txsAny, err := getPath(doc, "app_state", "genutil", "gen_txs")
	if err != nil {
		return nil, err
	}
	txs, ok := txsAny.([]any)
	if !ok || len(txs) == 0 {
		return nil, fmt.Errorf("genutil.gen_txs is not a non-empty array")
	}
	sort.SliceStable(txs, func(i, j int) bool { return gentxValidatorAddr(txs[i]) < gentxValidatorAddr(txs[j]) })
	raw, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(raw, '\n'), nil
}

func gentxValidatorAddr(tx any) string {
	m, _ := tx.(map[string]any)
	body, _ := m["body"].(map[string]any)
	msgs, _ := body["messages"].([]any)
	if len(msgs) == 0 {
		return ""
	}
	msg, _ := msgs[0].(map[string]any)
	s, _ := msg["validator_address"].(string)
	return s
}

// gentxSelfDelegation returns the self-delegation amount (uprsm) recorded
// in a MsgCreateValidator inside a gentx. Both the single-Coin form
// (`value: {denom, amount}`) and the legacy array form
// (`value: {amount: [{denom, amount}]}`) are accepted.
func gentxSelfDelegation(tx any) (uint64, error) {
	m, _ := tx.(map[string]any)
	body, _ := m["body"].(map[string]any)
	msgs, _ := body["messages"].([]any)
	if len(msgs) == 0 {
		return 0, fmt.Errorf("gentx has no message")
	}
	msg, _ := msgs[0].(map[string]any)
	value, _ := msg["value"].(map[string]any)
	parseAmount := func(s string) (uint64, error) {
		var n uint64
		if _, err := fmt.Sscanf(s, "%d", &n); err != nil {
			return 0, fmt.Errorf("amount %q: %w", s, err)
		}
		return n, nil
	}
	if denom, ok := value["denom"].(string); ok {
		if denom != testnetDenom {
			return 0, fmt.Errorf("gentx denom %q is not %s", denom, testnetDenom)
		}
		s, ok := value["amount"].(string)
		if !ok {
			return 0, fmt.Errorf("gentx value.amount is not a string")
		}
		return parseAmount(s)
	}
	if amounts, ok := value["amount"].([]any); ok {
		for _, a := range amounts {
			coin, _ := a.(map[string]any)
			if coin["denom"] != testnetDenom {
				continue
			}
			s, _ := coin["amount"].(string)
			return parseAmount(s)
		}
	}
	return 0, fmt.Errorf("gentx has no %s amount", testnetDenom)
}

// --- TOML line helpers -----------------------------------------------------

var tomlKeyPattern = regexp.MustCompile(`(?m)^([ \t]*)%s([ \t]*)=([ \t]*)(?:"[^"]*"|[^\s#]+)`)

// setTomlString replaces `key = "..."` (replace-only; missing key is a
// hard error so config drift is never silent).
func setTomlString(content, key, value string) (string, error) {
	re := regexp.MustCompile(fmt.Sprintf(tomlKeyPattern.String(), regexp.QuoteMeta(key)))
	if !re.MatchString(content) {
		return "", fmt.Errorf("config key %q not found", key)
	}
	return re.ReplaceAllString(content, fmt.Sprintf("${1}%s${2}=${3}%q", regexp.QuoteMeta(key), value)), nil
}

// setOrAppendTomlString replaces the key or appends it as a top-level entry
// (used for client.toml, which is flat and may omit optional keys).
func setOrAppendTomlString(content, key, value string) string {
	re := regexp.MustCompile(fmt.Sprintf(tomlKeyPattern.String(), regexp.QuoteMeta(key)))
	if re.MatchString(content) {
		return re.ReplaceAllString(content, fmt.Sprintf("${1}%s${2}=${3}%q", regexp.QuoteMeta(key), value))
	}
	if !strings.HasSuffix(content, "\n") {
		content += "\n"
	}
	return content + fmt.Sprintf("%s = %q\n", key, value)
}

// --- filesystem helpers ----------------------------------------------------

func copyTree(src, dst string) error {
	return filepath.WalkDir(src, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		in, err := os.Open(path)
		if err != nil {
			return err
		}
		defer in.Close()
		out, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
		if err != nil {
			return err
		}
		defer out.Close()
		_, err = io.Copy(out, in)
		return err
	})
}

// bundleReadme is deterministic: same spec, same bytes.
func bundleReadme(spec *Spec, idents []PublicIdentity) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Prisma testnet config bundle — %s\n\n", spec.ChainID)
	fmt.Fprintf(&b, "Generated by `netconfig/1` from the frozen protocol identity in `manifest.json`.\n")
	fmt.Fprintf(&b, "This bundle contains **public material only**: no seeds, no private keys.\n\n")
	b.WriteString("## Contents\n\n")
	b.WriteString("- `chain-id.txt` — the network identifier\n")
	b.WriteString("- `genesis.json` — the network genesis (all validators identical)\n")
	b.WriteString("- `params.json` — frozen protocol parameters, derived from the chain code\n")
	b.WriteString("- `validators.json` — validator public identities (node id, addresses, consensus key)\n")
	b.WriteString("- `manifest.json` — sha256 of every file above; `netconfig verify` enforces it\n\n")
	b.WriteString("## Validators\n\n")
	b.WriteString("| moniker | node id | p2p address | account |\n|---|---|---|---|\n")
	for _, id := range idents {
		fmt.Fprintf(&b, "| %s | `%s` | %s | `%s` |\n", id.Moniker, id.NodeID, id.P2PExternalAddress, id.AccountAddress)
	}
	b.WriteString("\n## Use\n\n")
	b.WriteString("```\n")
	b.WriteString("# verify the bundle (checksums, policy, SDK genesis validation)\n")
	b.WriteString("prismad-netconfig verify --dir <this-dir>\n\n")
	b.WriteString("# materialize a runnable node home for one validator (needs the spec with the seed)\n")
	b.WriteString("prismad-netconfig provision --spec spec.json --bundle <this-dir> --moniker <moniker> --home ~/.prisma\n")
	b.WriteString("```\n")
	return b.String()
}
