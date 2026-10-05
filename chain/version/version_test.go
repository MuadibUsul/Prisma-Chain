package version

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

const freezeArtifact = "../../docs/phase-f-freeze.json"

// The frozen protocol identity read from the authoritative artifact must
// match these constants; drift here means the protocol changed, which is
// forbidden without a versioned protocol change (roadmap section 6).
func TestFrozenIdentityConstants(t *testing.T) {
	if GraphIDV2 != "8fb86087697e277602bb4b2088ba48eb4172c21d9c4f83914582f2cbf00d4def" {
		t.Fatalf("GraphIDV2 drifted: %s", GraphIDV2)
	}
	if PolicyID != "eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0" {
		t.Fatalf("PolicyID drifted: %s", PolicyID)
	}
	if FreezeTag != "transformer-phase-f-wide-integer" {
		t.Fatalf("FreezeTag drifted: %s", FreezeTag)
	}
	if FreezeCommit != "89547fa97b62d72a4cd336a5dfd3b000b03eac60" {
		t.Fatalf("FreezeCommit drifted: %s", FreezeCommit)
	}
}

// The generated embed must equal the freeze artifact it claims to come
// from: proves the values are not hand-typed and cannot drift silently.
func TestGeneratedMatchesFreezeArtifact(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(freezeArtifact))
	if err != nil {
		t.Skipf("freeze artifact not available outside the repository tree: %v", err)
	}
	var doc struct {
		FreezeTag    string `json:"freeze_tag"`
		FreezeCommit string `json:"freeze_tag_commit"`
		Recorded     struct {
			GraphIDV2        string            `json:"graph_id_v2"`
			PolicyID         string            `json:"policy_id"`
			ProtocolVersions map[string]string `json:"protocol_versions"`
		} `json:"recorded"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("parse freeze artifact: %v", err)
	}
	if doc.FreezeTag != FreezeTag || doc.FreezeCommit != FreezeCommit {
		t.Fatalf("tag/commit mismatch: artifact %s/%s vs generated %s/%s",
			doc.FreezeTag, doc.FreezeCommit, FreezeTag, FreezeCommit)
	}
	if doc.Recorded.GraphIDV2 != GraphIDV2 || doc.Recorded.PolicyID != PolicyID {
		t.Fatalf("identity mismatch: artifact %s/%s vs generated %s/%s",
			doc.Recorded.GraphIDV2, doc.Recorded.PolicyID, GraphIDV2, PolicyID)
	}
	if len(doc.Recorded.ProtocolVersions) != len(ProtocolVersions) {
		t.Fatalf("protocol component count: artifact %d vs generated %d",
			len(doc.Recorded.ProtocolVersions), len(ProtocolVersions))
	}
	for k, v := range doc.Recorded.ProtocolVersions {
		if ProtocolVersions[k] != v {
			t.Fatalf("component %q: artifact %q vs generated %q", k, v, ProtocolVersions[k])
		}
	}
}

func TestBuildIdentity(t *testing.T) {
	if SoftwareVersion() == "" || GitCommit() == "" {
		t.Fatalf("build identity fields must never be empty")
	}
	if want := runtime.GOOS + "/" + runtime.GOARCH; BuildTarget() != want {
		t.Fatalf("BuildTarget = %q, want %q", BuildTarget(), want)
	}
	info := Get()
	if info.Protocol.Phase != "FROZEN" || info.Protocol.GraphIDV2 != GraphIDV2 {
		t.Fatalf("Get() lost the protocol identity: %+v", info.Protocol)
	}
	info.Protocol.Components["tensor_v2"] = "mutated"
	if ProtocolVersions["tensor_v2"] == "mutated" {
		t.Fatalf("Protocol() must return a copy of the component map")
	}
}
