// Package version carries the release identity of the Prisma node binaries
// and the frozen protocol identity they implement.
//
// Release builds stamp softwareVersion and gitCommit via
//
//	-ldflags "-X prismachain/chain/version.softwareVersion=<v> \
//	          -X prismachain/chain/version.gitCommit=<sha>"
//
// No wall-clock build date is embedded anywhere: it would break the
// two-build reproducibility check (B1-01e). The git commit is the
// timestamp authority instead.
package version

import "runtime"

var (
	softwareVersion = "dev"
	gitCommit       = "unknown"
)

// SoftwareVersion returns the release version (a product version or git
// describe string); "dev" for unstamped local builds.
func SoftwareVersion() string { return softwareVersion }

// GitCommit returns the full commit SHA the binary was built from.
func GitCommit() string { return gitCommit }

// BuildTarget returns the "os/arch" the binary was compiled for.
func BuildTarget() string { return runtime.GOOS + "/" + runtime.GOARCH }

// ProtocolIdentity is the frozen protocol identity embedded at build time
// from docs/phase-f-freeze.json (see protocol_gen.go; never hand-typed).
type ProtocolIdentity struct {
	Phase        string            `json:"protocol_phase"`
	FreezeTag    string            `json:"freeze_tag"`
	FreezeCommit string            `json:"freeze_commit"`
	GraphIDV2    string            `json:"graph_id_v2"`
	PolicyID     string            `json:"policy_id"`
	Components   map[string]string `json:"protocol_versions"`
}

// Protocol returns the embedded frozen-protocol identity.
func Protocol() ProtocolIdentity {
	components := make(map[string]string, len(ProtocolVersions))
	for k, v := range ProtocolVersions {
		components[k] = v
	}
	return ProtocolIdentity{
		Phase:        "FROZEN",
		FreezeTag:    FreezeTag,
		FreezeCommit: FreezeCommit,
		GraphIDV2:    GraphIDV2,
		PolicyID:     PolicyID,
		Components:   components,
	}
}

// Info is the full machine-readable identity of a build.
type Info struct {
	SoftwareVersion string           `json:"software_version"`
	GitCommit       string           `json:"git_commit"`
	BuildTarget     string           `json:"build_target"`
	Protocol        ProtocolIdentity `json:"protocol"`
}

// Get returns the complete build identity.
func Get() Info {
	return Info{
		SoftwareVersion: SoftwareVersion(),
		GitCommit:       GitCommit(),
		BuildTarget:     BuildTarget(),
		Protocol:        Protocol(),
	}
}
