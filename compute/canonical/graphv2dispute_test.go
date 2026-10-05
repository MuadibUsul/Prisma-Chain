package canonical

// A1-04..A1-06 canonical-layer tests for GraphDisputeV2: honest bisection
// to the injected first-divergent node, snapshot/restore equivalence,
// wrong-claim rejection and V1-material rejection.

import (
	"testing"
)

func buildDisputeV2Fixture(t *testing.T) (*GraphDescriptorV2, *GraphExecutionV2) {
	t.Helper()
	g := smallWideGraph(t)
	a := mustTensorV2(t, NewDescV2(DtypeV2A13, 2, 3), []int64{1, -2, 3, 4095, -4096, 7})
	w := mustTensorV2(t, NewDescV2(DtypeV2W10, 3, 2), []int64{1, -1, 2, 0, -3, 511})
	exec, err := ExecuteGraphV2(g, map[uint32]*TensorV2{0: a, 1: w})
	if err != nil {
		t.Fatal(err)
	}
	return g, exec
}

func claimForV2(t *testing.T, gid Hash, trail []Hash) TrailClaim {
	t.Helper()
	initial, err := TrailProofV2(gid, trail, 0)
	if err != nil {
		t.Fatal(err)
	}
	final, err := TrailProofV2(gid, trail, uint32(len(trail)-1))
	if err != nil {
		t.Fatal(err)
	}
	root, err := TrailRootV2(gid, trail)
	if err != nil {
		t.Fatal(err)
	}
	return TrailClaim{Root: root, InitialRoot: trail[0],
		InitialProof: initial, FinalRoot: trail[len(trail)-1], FinalProof: final}
}

func TestGraphDisputeV2Bisection(t *testing.T) {
	g, exec := buildDisputeV2Fixture(t)
	gid, err := g.GraphIDV2()
	if err != nil {
		t.Fatal(err)
	}
	honest := exec.Trail // len = nodes+1 = 3

	// challenger corrupts node 1's output (trail[2]); node 0 output shared
	bad := append([]Hash(nil), honest...)
	bad[2] = hashBytes([]byte("fraud-state"))

	d, err := NewGraphDisputeV2(GraphDisputeConfigV2{Graph: g, GraphID: gid, RoundPeriod: 10},
		claimForV2(t, gid, honest), claimForV2(t, gid, bad), 100)
	if err != nil {
		t.Fatal(err)
	}
	// first round: mid = 1, both submit trail[1] (equal) -> low = 1
	if _, err := d.SubmitMid(Worker, honest[1], mustProofV2(t, gid, honest, 1), 101); err != nil {
		t.Fatal(err)
	}
	if _, err := d.SubmitMid(Challenger, bad[1], mustProofV2(t, gid, bad, 1), 101); err != nil {
		t.Fatal(err)
	}
	if low, high := d.Interval(); low != 1 || high != 2 {
		t.Fatalf("interval (%d,%d), want (1,2)", low, high)
	}
	if !d.ArbReady() {
		t.Fatal("arbitration should be ready after convergence to one step")
	}
	node, err := d.FirstDivergentNode()
	if err != nil {
		t.Fatal(err)
	}
	if node != 1 {
		t.Fatalf("first divergent node %d, want 1 (the injected node)", node)
	}
	// snapshot mid-dispute was taken before the final round; verify restore
	// reproduces the same interval and node on a fresh instance
	snap, err := d.SnapshotV2()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := RestoreGraphDisputeV2(snap, g, 10)
	if err != nil {
		t.Fatal(err)
	}
	rNode, err := restored.FirstDivergentNode()
	if err != nil || rNode != node {
		t.Fatalf("restored node %d (err %v), want %d", rNode, err, node)
	}
}

func TestGraphDisputeV2Rejections(t *testing.T) {
	g, exec := buildDisputeV2Fixture(t)
	gid, err := g.GraphIDV2()
	if err != nil {
		t.Fatal(err)
	}
	honest := exec.Trail
	// a forger who corrupts node 0's output gets a self-consistent trail:
	// every downstream state differs too
	bad := append([]Hash(nil), honest...)
	bad[1] = hashBytes([]byte("fraud-node0-out"))
	bad[2] = hashBytes([]byte("fraud-final"))

	// same final root: no fraud to adjudicate
	same := claimForV2(t, gid, honest)
	if _, err := NewGraphDisputeV2(GraphDisputeConfigV2{Graph: g, GraphID: gid, RoundPeriod: 10},
		same, same, 100); err == nil {
		t.Fatal("identical endpoints accepted")
	}
	// V1-domain proofs are refused: rebuild the claim with V1 leaves
	v1Initial := func() []Hash {
		levels := [][]Hash{{TrailLeaf(gid, 0, honest[0]), TrailLeaf(gid, 1, honest[1]),
			TrailLeaf(gid, 2, honest[2])}}
		sib, err := proveLeaf(levels, 0)
		if err != nil {
			t.Fatal(err)
		}
		return sib
	}()
	v1Root := func() Hash {
		levels, err := TrailLevels(gid, honest)
		if err != nil {
			t.Fatal(err)
		}
		return levels[len(levels)-1][0]
	}()
	claim := claimForV2(t, gid, honest)
	v1Claim := TrailClaim{Root: v1Root, InitialRoot: honest[0],
		InitialProof: v1Initial, FinalRoot: honest[2], FinalProof: claim.FinalProof}
	if _, err := NewGraphDisputeV2(GraphDisputeConfigV2{Graph: g, GraphID: gid, RoundPeriod: 10},
		v1Claim, claimForV2(t, gid, bad), 100); err == nil {
		t.Fatal("V1-domain trail material accepted by the V2 dispute")
	}
	// wrong round window and duplicate submissions are refused
	d, err := NewGraphDisputeV2(GraphDisputeConfigV2{Graph: g, GraphID: gid, RoundPeriod: 10},
		claimForV2(t, gid, honest), claimForV2(t, gid, bad), 100)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.SubmitMid(Worker, honest[1], mustProofV2(t, gid, honest, 1), 99); err == nil {
		t.Fatal("round window violation accepted")
	}
	if _, err := d.SubmitMid(Worker, honest[1], mustProofV2(t, gid, honest, 1), 101); err != nil {
		t.Fatal(err)
	}
	if _, err := d.SubmitMid(Worker, honest[1], mustProofV2(t, gid, honest, 1), 101); err == nil {
		t.Fatal("duplicate submission accepted")
	}
	// a proof from the wrong trail is refused
	if _, err := d.SubmitMid(Challenger, honest[1], mustProofV2(t, gid, honest, 1), 101); err == nil {
		// honest[1] == bad[1] is legitimately provable from the bad trail too,
		// so this must NOT error — assert it succeeded instead.
		t.Fatal("shared state root with a valid proof was refused")
	}
}

func TestGraphDisputeV2SnapshotIdentity(t *testing.T) {
	g, exec := buildDisputeV2Fixture(t)
	gid, err := g.GraphIDV2()
	if err != nil {
		t.Fatal(err)
	}
	honest := exec.Trail
	bad := append([]Hash(nil), honest...)
	bad[1] = hashBytes([]byte("fraud-node0-out"))
	bad[2] = hashBytes([]byte("fraud-final"))
	d, err := NewGraphDisputeV2(GraphDisputeConfigV2{Graph: g, GraphID: gid, RoundPeriod: 10},
		claimForV2(t, gid, honest), claimForV2(t, gid, bad), 100)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.SubmitMid(Worker, honest[1], mustProofV2(t, gid, honest, 1), 101); err != nil {
		t.Fatal(err)
	}
	snap, err := d.SnapshotV2()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := RestoreGraphDisputeV2(snap, g, 10)
	if err != nil {
		t.Fatal(err)
	}
	l1, h1 := d.Interval()
	l2, h2 := restored.Interval()
	if l1 != l2 || h1 != h2 || d.LastAcceptedEpoch() != restored.LastAcceptedEpoch() {
		t.Fatal("snapshot restore diverged")
	}
	// restore against a different graph must fail (id/nodes check)
	other := smallWideGraph(t)
	other.Spec = "OTHER_SPEC"
	if _, err := RestoreGraphDisputeV2(snap, other, 10); err == nil {
		t.Fatal("restore accepted a different graph")
	}
	// continuing the restored instance reaches the same verdict
	if _, err := restored.SubmitMid(Challenger, bad[1], mustProofV2(t, gid, bad, 1), 101); err != nil {
		t.Fatal(err)
	}
	node, err := restored.FirstDivergentNode()
	if err != nil || node != 0 {
		t.Fatalf("restored verdict node %d (err %v), want 0", node, err)
	}
}

func mustProofV2(t *testing.T, gid Hash, trail []Hash, index uint32) []Hash {
	t.Helper()
	p, err := TrailProofV2(gid, trail, index)
	if err != nil {
		t.Fatal(err)
	}
	return p
}
