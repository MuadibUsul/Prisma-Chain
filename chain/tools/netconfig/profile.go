package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Node config profiles (B1-02a/B1-02b): each role gets a canonical
// config.toml/app.toml derived from the real SDK template plus the role's
// listen/peer policy, and roles that expose RPC also get a concrete
// rpc-proxy.json. No role embeds key material: a node home generated from
// a profile is provisioned separately (`netconfig provision`).
type profileRole struct {
	Name        string
	Description string
	P2PLaddr    string // default; overridable with --p2p-laddr
	Pex         bool
	PublicRPC   bool // front the local RPC with prismad rpc-proxy
	Notes       []string
}

var profileRoles = map[string]profileRole{
	"validator-private": {
		Name:        "validator-private",
		Description: "consensus validator with no public surface; peers only with its sentries",
		P2PLaddr:    "tcp://127.0.0.1:26656",
		Pex:         false,
		PublicRPC:   false,
		Notes: []string{
			"bind p2p to the private interface (or a private IP) with --p2p-laddr; the default is localhost",
			"persistent_peers must contain only sentries; a public peer on a validator is a policy violation",
			"rpc stays on localhost; operators reach it over SSH/VPN",
		},
	},
	"sentry": {
		Name:        "sentry",
		Description: "public p2p sentry: the only component that talks to the open network on a validator's behalf",
		P2PLaddr:    "tcp://0.0.0.0:26656",
		Pex:         true,
		PublicRPC:   true,
		Notes: []string{
			"--external-address must be the public host:port peers can dial",
			"persistent_peers = validator private address + other sentries",
			"the public RPC is the rpc-proxy, never the node's own RPC",
		},
	},
	"rpc-public": {
		Name:        "rpc-public",
		Description: "non-validator RPC full node: no validator key in the network set, peer with sentries only",
		P2PLaddr:    "tcp://0.0.0.0:26656",
		Pex:         false,
		PublicRPC:   true,
		Notes: []string{
			"this node never appears in the genesis validator set, so it holds no network signing key",
			"pex is off: peer discovery goes through the sentries you list",
			"the public RPC is the rpc-proxy, never the node's own RPC",
		},
	},
}

func cmdProfile(args []string) error {
	fs := flag.NewFlagSet("profile", flag.ContinueOnError)
	roleName := fs.String("role", "", "validator-private | sentry | rpc-public")
	out := fs.String("out", "", "output directory for the profile")
	chainID := fs.String("chain-id", "prisma-testnet-1", "chain id for the generated client config")
	moniker := fs.String("moniker", "", "node moniker (required)")
	p2pLaddr := fs.String("p2p-laddr", "", "override the p2p listen address")
	rpcLaddr := fs.String("rpc-laddr", "tcp://127.0.0.1:26657", "node RPC listen address (keep on localhost)")
	external := fs.String("external-address", "", "public/private host:port this node advertises")
	peers := fs.String("peers", "", "comma-separated persistent peers id@host:port")
	prismad := fs.String("prismad", "prismad", "prismad binary")
	proxyListen := fs.String("proxy-listen", "0.0.0.0:8545", "rpc-proxy listen address (public roles)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	role, ok := profileRoles[*roleName]
	if !ok {
		names := make([]string, 0, len(profileRoles))
		for n := range profileRoles {
			names = append(names, n)
		}
		sort.Strings(names)
		return fmt.Errorf("--role must be one of %s", strings.Join(names, ", "))
	}
	if *out == "" || *moniker == "" {
		return fmt.Errorf("profile requires --out and --moniker")
	}
	if role.Name == "sentry" && *external == "" {
		return fmt.Errorf("sentry profile requires --external-address (the public host:port peers dial)")
	}
	if strings.Contains(*rpcLaddr, "0.0.0.0") {
		return fmt.Errorf("rpc-laddr %q is a wildcard; the node RPC stays on localhost and the proxy faces the public", *rpcLaddr)
	}
	configureSDKPrefixes()

	work, err := os.MkdirTemp("", "netconfig-profile-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(work)
	home := filepath.Join(work, "home")
	if _, err := runPrismad(*prismad, "init", *moniker, "--chain-id", *chainID, "--default-denom", testnetDenom, "--home", home); err != nil {
		return err
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		return err
	}
	p2p := role.P2PLaddr
	if *p2pLaddr != "" {
		p2p = *p2pLaddr
	}
	cfgRaw, err := os.ReadFile(filepath.Join(home, "config", "config.toml"))
	if err != nil {
		return err
	}
	content := string(cfgRaw)
	for _, step := range []struct{ section, key, value string }{
		{"", "moniker", *moniker},
		{"p2p", "laddr", p2p},
		{"p2p", "external_address", *external},
		{"p2p", "persistent_peers", *peers},
		{"rpc", "laddr", *rpcLaddr},
	} {
		content, err = setTomlInSection(content, step.section, step.key, step.value)
		if err != nil {
			return err
		}
	}
	// pex is a TOML boolean: written raw, never quoted.
	if content, err = setTomlRawInSection(content, "p2p", "pex", fmt.Sprintf("%t", role.Pex)); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*out, "config.toml"), []byte(content), 0o644); err != nil {
		return err
	}
	appRaw, err := os.ReadFile(filepath.Join(home, "config", "app.toml"))
	if err != nil {
		return err
	}
	app, err := setTomlInSection(string(appRaw), "", "minimum-gas-prices", "0uprsm")
	if err != nil {
		return err
	}
	app, err = setTomlInSection(app, "api", "enable", "false")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*out, "app.toml"), []byte(app), 0o644); err != nil {
		return err
	}
	if role.PublicRPC {
		if err := writeJSONDoc(filepath.Join(*out, "rpc-proxy.json"), map[string]any{
			"listen":             *proxyListen,
			"upstream":           *rpcLaddr,
			"policy":             "read-only",
			"rate_per_second":    ReadOnlyProxyRate,
			"burst":              ReadOnlyProxyBurst,
			"max_body_bytes":     1 << 20,
			"upstream_timeout_s": 15,
		}); err != nil {
			return err
		}
	}
	var readme strings.Builder
	fmt.Fprintf(&readme, "# Node profile: %s\n\n%s\n\n", role.Name, role.Description)
	readme.WriteString("## Policy\n\n")
	for _, note := range role.Notes {
		fmt.Fprintf(&readme, "- %s\n", note)
	}
	readme.WriteString("\n## Values\n\n```text\n")
	fmt.Fprintf(&readme, "p2p.laddr          = %s\n", p2p)
	fmt.Fprintf(&readme, "p2p.external_addr  = %s\n", *external)
	fmt.Fprintf(&readme, "p2p.persistent     = %s\n", *peers)
	fmt.Fprintf(&readme, "p2p.pex            = %t\n", role.Pex)
	fmt.Fprintf(&readme, "rpc.laddr          = %s\n", *rpcLaddr)
	if role.PublicRPC {
		fmt.Fprintf(&readme, "rpc-proxy.listen   = %s\n", *proxyListen)
	}
	readme.WriteString("```\n\n## Install\n\n```text\n")
	readme.WriteString("1. provision the node home with its own key (netconfig provision)\n")
	readme.WriteString("2. copy config.toml / app.toml over <home>/config/\n")
	if role.PublicRPC {
		readme.WriteString("3. run: prismad rpc-proxy --listen " + *proxyListen + " --upstream http://127.0.0.1:26657\n")
		readme.WriteString("   (policy and limits are in the default read-only preset; adjust with flags and record why)\n")
	}
	readme.WriteString("```\n")
	if err := os.WriteFile(filepath.Join(*out, "README.md"), []byte(readme.String()), 0o644); err != nil {
		return err
	}
	fmt.Printf("netconfig: %s profile written to %s\n", role.Name, *out)
	return nil
}

// Defaults mirrored in rpc-proxy.json so the file is a complete record of
// what the operator should run (values match rpcproxy.ReadOnlyPolicy).
const (
	ReadOnlyProxyRate  = 20
	ReadOnlyProxyBurst = 40
)
