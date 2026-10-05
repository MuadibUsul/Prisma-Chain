package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"prismachain/chain/version"
)

// --- JSON helpers (number-literal preserving, deterministic output) -------

func readJSONDoc(path string) (map[string]any, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var doc map[string]any
	if err := dec.Decode(&doc); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return doc, nil
}

// writeJSONDoc marshals with sorted keys (encoding/json sorts map keys) so
// the same input always produces the same bytes.
func writeJSONDoc(path string, doc any) error {
	raw, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}

func getPath(doc map[string]any, path ...string) (any, error) {
	var cur any = doc
	for _, key := range path {
		m, ok := cur.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("path %s: not an object", strings.Join(path, "."))
		}
		cur, ok = m[key]
		if !ok {
			return nil, fmt.Errorf("path %s: key %q missing", strings.Join(path, "."), key)
		}
	}
	return cur, nil
}

func setPath(doc map[string]any, value any, path ...string) error {
	if len(path) == 0 {
		return fmt.Errorf("setPath: empty path")
	}
	cur := doc
	for _, key := range path[:len(path)-1] {
		next, ok := cur[key].(map[string]any)
		if !ok {
			return fmt.Errorf("setPath %s: key %q missing or not an object", strings.Join(path, "."), key)
		}
		cur = next
	}
	cur[path[len(path)-1]] = value
	return nil
}

// --- prismad driver --------------------------------------------------------

func runPrismad(bin string, args ...string) (string, error) {
	cmd := exec.Command(bin, args...)
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		return out.String(), fmt.Errorf("%s %s: %v: %s", filepath.Base(bin), strings.Join(args, " "), err, strings.TrimSpace(errb.String()))
	}
	return out.String(), nil
}

// keysFlags returns the keyring flags; the test backend plus an explicit
// keyring dir keeps the ceremony non-interactive and self-contained.
func keysFlags(home string) []string {
	return []string{"--keyring-backend", "test", "--keyring-dir", home, "--home", home}
}

// --- manifest --------------------------------------------------------------

type manifestEntry struct {
	Path   string `json:"path"`
	Sha256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

type bundleManifest struct {
	Tool         string          `json:"tool"`
	ChainID      string          `json:"chain_id"`
	FreezeTag    string          `json:"freeze_tag"`
	FreezeCommit string          `json:"freeze_commit"`
	GraphIDV2    string          `json:"graph_id_v2"`
	PolicyID     string          `json:"policy_id"`
	Files        []manifestEntry `json:"files"`
}

const manifestName = "manifest.json"

func fileSha256(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

func writeManifest(dir string, chainID string) (*bundleManifest, error) {
	entries := []manifestEntry{}
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || d.Name() == manifestName {
			return nil
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		sum, size, err := fileSha256(path)
		if err != nil {
			return err
		}
		entries = append(entries, manifestEntry{Path: filepath.ToSlash(rel), Sha256: sum, Bytes: size})
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Path < entries[j].Path })
	proto := version.Protocol()
	m := &bundleManifest{
		Tool:         "netconfig/1",
		ChainID:      chainID,
		FreezeTag:    proto.FreezeTag,
		FreezeCommit: proto.FreezeCommit,
		GraphIDV2:    proto.GraphIDV2,
		PolicyID:     proto.PolicyID,
		Files:        entries,
	}
	return m, writeJSONDoc(filepath.Join(dir, manifestName), m)
}

// --- newseeds --------------------------------------------------------------

func cmdNewSeeds(args []string) error {
	fs := flag.NewFlagSet("newseeds", flag.ContinueOnError)
	count := fs.Int("count", 4, "number of seeds to print")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *count < 1 || *count > 64 {
		return fmt.Errorf("count must be between 1 and 64")
	}
	for i := 0; i < *count; i++ {
		seed, err := newSeed()
		if err != nil {
			return err
		}
		fmt.Println(seed)
	}
	return nil
}

// --- generate --------------------------------------------------------------

func cmdGenerate(args []string) error {
	fs := flag.NewFlagSet("generate", flag.ContinueOnError)
	specPath := fs.String("spec", "", "path to the spec JSON (contains validator seeds)")
	outDir := fs.String("out", "", "output bundle directory")
	keepHomes := fs.String("keep-homes", "", "optional: also materialize runnable node homes here (contains private keys)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *specPath == "" || *outDir == "" {
		return fmt.Errorf("generate requires --spec and --out")
	}
	spec, err := LoadSpec(*specPath)
	if err != nil {
		return err
	}
	configureSDKPrefixes()
	prismad := spec.PrismadBinary()

	work, err := os.MkdirTemp("", "netconfig-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(work)

	type node struct {
		spec   Validator
		keys   DerivedKeys
		home   string
		public PublicIdentity
	}
	nodes := make([]node, 0, len(spec.Validators))
	for _, v := range spec.Validators {
		seed, err := decodeSeed(v.SeedB64)
		if err != nil {
			return err
		}
		keys := DeriveKeys(seed)
		home := filepath.Join(work, v.Moniker)
		if _, err := runPrismad(prismad, "init", v.Moniker, "--chain-id", spec.ChainID,
			"--default-denom", spec.Denom, "--home", home); err != nil {
			return err
		}
		// Replace the random comet keys with the derived ones before
		// anything else uses this home.
		if err := writePrivValidatorKey(home, keys.Consensus); err != nil {
			return err
		}
		if err := writeNodeKey(home, keys.Node); err != nil {
			return err
		}
		if err := ensurePrivValidatorState(home); err != nil {
			return err
		}
		if err := pinClientToml(home); err != nil {
			return err
		}
		if _, err := runPrismad(prismad, append([]string{"keys", "import-hex", "validator",
			hex.EncodeToString(keys.Account.Key)}, keysFlags(home)...)...); err != nil {
			return err
		}
		nodes = append(nodes, node{spec: v, keys: keys, home: home,
			public: PublicIdentityOf(v.Moniker, v, keys)})
	}
	owner := nodes[0].home

	// Genesis policy: the denom comes from --default-denom; testnet
	// issuance is disabled (zero inflation) until the audited base-task
	// issuance schedule exists. Everything else stays at module defaults.
	genesisPath := filepath.Join(owner, "config", "genesis.json")
	if err := applyGenesisPolicy(genesisPath, spec.Denom); err != nil {
		return fmt.Errorf("genesis policy: %w", err)
	}

	for _, n := range nodes {
		if _, err := runPrismad(prismad, append([]string{"genesis", "add-genesis-account",
			n.public.AccountAddress, fmt.Sprintf("%d%s", n.spec.AccountFundingUprsm, spec.Denom)}, flagsHome(owner)...)...); err != nil {
			return err
		}
	}
	gentxDir := filepath.Join(owner, "config", "gentx")
	if err := os.MkdirAll(gentxDir, 0o755); err != nil {
		return err
	}
	// Every gentx is signed against the same funded genesis document, so
	// distribute the owner's genesis to every validator home first.
	fundedGenesis, err := os.ReadFile(filepath.Join(owner, "config", "genesis.json"))
	if err != nil {
		return err
	}
	for _, n := range nodes {
		if err := os.WriteFile(filepath.Join(n.home, "config", "genesis.json"), fundedGenesis, 0o644); err != nil {
			return err
		}
	}
	type collectedGentx struct {
		name string
		data []byte
	}
	var collected []collectedGentx
	for i, n := range nodes {
		args := []string{"genesis", "gentx", "validator", fmt.Sprintf("%d%s", n.spec.SelfDelegationUprsm, spec.Denom),
			"--chain-id", spec.ChainID}
		args = append(args, keysFlags(n.home)...)
		if _, err := runPrismad(prismad, args...); err != nil {
			return err
		}
		produced, err := filepath.Glob(filepath.Join(n.home, "config", "gentx", "*.json"))
		if err != nil || len(produced) == 0 {
			return fmt.Errorf("validator %s: gentx produced no file", n.spec.Moniker)
		}
		sort.Strings(produced)
		for _, p := range produced {
			data, err := os.ReadFile(p)
			if err != nil {
				return err
			}
			collected = append(collected, collectedGentx{
				name: fmt.Sprintf("%02d-%s", i, filepath.Base(p)), data: data,
			})
		}
	}
	// collect-gentxs reads exactly the owner's gentx directory: clear it and
	// lay down the gathered gentxs so no gentx is present twice (the owner
	// produced one there itself, and its bytes are already in memory).
	existing, err := os.ReadDir(gentxDir)
	if err != nil {
		return err
	}
	for _, e := range existing {
		if !e.IsDir() {
			if err := os.Remove(filepath.Join(gentxDir, e.Name())); err != nil {
				return err
			}
		}
	}
	for _, g := range collected {
		if err := os.WriteFile(filepath.Join(gentxDir, g.name), g.data, 0o644); err != nil {
			return err
		}
	}
	if _, err := runPrismad(prismad, append([]string{"genesis", "collect-gentxs"}, flagsHome(owner)...)...); err != nil {
		return err
	}

	// Normalize: fixed genesis time, gentxs sorted by validator address,
	// deterministic serialization.
	finalGenesis, err := normalizeGenesis(genesisPath, spec.GenesisTime)
	if err != nil {
		return err
	}
	for _, n := range nodes {
		if err := os.WriteFile(filepath.Join(n.home, "config", "genesis.json"), finalGenesis, 0o644); err != nil {
			return err
		}
	}

	// Write the public bundle.
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*outDir, "chain-id.txt"), []byte(spec.ChainID+"\n"), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*outDir, "genesis.json"), finalGenesis, 0o644); err != nil {
		return err
	}
	if err := writeJSONDoc(filepath.Join(*outDir, "params.json"), DeriveParams(spec)); err != nil {
		return err
	}
	idents := make([]PublicIdentity, 0, len(nodes))
	for _, n := range nodes {
		idents = append(idents, n.public)
	}
	if err := writeJSONDoc(filepath.Join(*outDir, "validators.json"), map[string]any{
		"chain_id":   spec.ChainID,
		"validators": idents,
	}); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*outDir, "README.md"), []byte(bundleReadme(spec, idents)), 0o644); err != nil {
		return err
	}
	if _, err := writeManifest(*outDir, spec.ChainID); err != nil {
		return err
	}

	// Optional: materialize runnable node homes (private keys, only when
	// the operator explicitly asks for it).
	if *keepHomes != "" {
		for _, n := range nodes {
			dst := filepath.Join(*keepHomes, n.spec.Moniker)
			if err := copyTree(n.home, dst); err != nil {
				return err
			}
		}
	}

	// Self-check: the bundle must verify, including the SDK-level genesis
	// validation through prismad.
	if _, err := verifyBundle(*outDir, prismad, false); err != nil {
		return fmt.Errorf("generated bundle failed self-verification: %w", err)
	}
	fmt.Printf("netconfig: bundle written to %s (chain-id %s, %d validators)\n",
		*outDir, spec.ChainID, len(nodes))
	if *keepHomes != "" {
		fmt.Printf("netconfig: node homes (PRIVATE KEYS) written to %s\n", *keepHomes)
	}
	return nil
}

func decodeSeed(seedB64 string) ([]byte, error) {
	raw, err := base64Std(seedB64)
	if err != nil {
		return nil, fmt.Errorf("seed_b64: %w", err)
	}
	return raw, nil
}

// pinClientToml forces the test keyring + home so every SDK/autocli command
// finds the keys without interactive input (same trap the devnet hit).
func pinClientToml(home string) error {
	path := filepath.Join(home, "config", "client.toml")
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	content, err := setTomlString(string(raw), "keyring-backend", "test")
	if err != nil {
		return err
	}
	content = setOrAppendTomlString(content, "keyring-dir", filepath.ToSlash(home))
	return os.WriteFile(path, []byte(content), 0o644)
}

func flagsHome(home string) []string { return []string{"--home", home} }

func copyFile(src, dst string) error {
	raw, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, raw, 0o644)
}
