package canonical

// Snapshot persistence and receipt tests: the chain must be able to
// restore a graph dispute across restarts and continue it, and the
// settlement receipt must re-derive every commitment.

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"testing"
)

func midSubmission(t *testing.T, dispute *GraphDispute, graphID Hash, workerTrail, honestTrail []Hash, epoch uint64) {
	t.Helper()
	mid := dispute.low + (dispute.high-dispute.low)/2
	wLevels, err := TrailLevels(graphID, workerTrail)
	if err != nil {
		t.Fatal(err)
	}
	cLevels, err := TrailLevels(graphID, honestTrail)
	if err != nil {
		t.Fatal(err)
	}
	wp, err := proveLeaf(wLevels, int(mid))
	if err != nil {
		t.Fatal(err)
	}
	cp, err := proveLeaf(cLevels, int(mid))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := dispute.SubmitMid(Worker, workerTrail[mid], wp, epoch); err != nil {
		t.Fatal(err)
	}
	if _, err := dispute.SubmitMid(Challenger, honestTrail[mid], cp, epoch+1); err != nil {
		t.Fatal(err)
	}
}

func TestGraphSnapshotRoundTripAndContinuation(t *testing.T) {
	fix := buildBlockFixture(t, miniBlockConfig())
	fraudNode := firstNodeWith(t, fix.graph, OpSoftmaxFixedV1)
	fraudTrail := fraudExecution(t, fix.graph, fix.inputs, fix.tables, fraudNode)
	graphID, err := fix.graph.GraphID()
	if err != nil {
		t.Fatal(err)
	}
	cfg := GraphDisputeConfig{Graph: fix.graph, GraphID: graphID, RoundPeriod: 50}
	dispute, err := NewGraphDispute(cfg, trailClaim(t, graphID, fraudTrail), trailClaim(t, graphID, fix.exec.Trail), 1000)
	if err != nil {
		t.Fatal(err)
	}
	// Advance one round, then persist and restore mid-dispute.
	midSubmission(t, dispute, graphID, fraudTrail, fix.exec.Trail, 1001)

	snapshot, err := dispute.SnapshotV1()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := RestoreGraphDispute(snapshot, fix.graph, 50)
	if err != nil {
		t.Fatal(err)
	}

	// Both sessions continue with identical submissions and must agree.
	// The round clock continues from the last accepted epoch.
	epoch := uint64(1003)
	for i := 0; i < 16 && !dispute.ArbReady(); i++ {
		midSubmission(t, dispute, graphID, fraudTrail, fix.exec.Trail, epoch)
		midSubmission(t, restored, graphID, fraudTrail, fix.exec.Trail, epoch)
		epoch += 2
	}
	if !dispute.ArbReady() || !restored.ArbReady() {
		t.Fatal("bisection did not converge after restore")
	}
	n1, err := dispute.FirstDivergentNode()
	if err != nil {
		t.Fatal(err)
	}
	n2, err := restored.FirstDivergentNode()
	if err != nil {
		t.Fatal(err)
	}
	if n1 != fraudNode || n2 != fraudNode {
		t.Fatalf("localization after restore: %d / %d, want %d", n1, n2, fraudNode)
	}
	// The restored dispute must snapshot identically.
	snap2, err := restored.SnapshotV1()
	if err != nil {
		t.Fatal(err)
	}
	orig, err := dispute.SnapshotV1()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(orig, snap2) {
		t.Fatal("restored dispute diverged from the original")
	}
}

func TestGraphSnapshotTamper(t *testing.T) {
	fix := buildBlockFixture(t, miniBlockConfig())
	fraudNode := firstNodeWith(t, fix.graph, OpSoftmaxFixedV1)
	fraudTrail := fraudExecution(t, fix.graph, fix.inputs, fix.tables, fraudNode)
	graphID, _ := fix.graph.GraphID()
	cfg := GraphDisputeConfig{Graph: fix.graph, GraphID: graphID, RoundPeriod: 50}
	dispute, err := NewGraphDispute(cfg, trailClaim(t, graphID, fraudTrail), trailClaim(t, graphID, fix.exec.Trail), 1000)
	if err != nil {
		t.Fatal(err)
	}
	midSubmission(t, dispute, graphID, fraudTrail, fix.exec.Trail, 1001)
	snapshot, err := dispute.SnapshotV1()
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot) != graphSnapshotSize {
		t.Fatalf("snapshot size %d", len(snapshot))
	}

	flip := func(offset int) []byte {
		cp := append([]byte(nil), snapshot...)
		cp[offset] ^= 0x01
		return cp
	}
	if _, err := RestoreGraphDispute(flip(0), fix.graph, 50); err == nil {
		t.Fatal("bad magic accepted")
	}
	if _, err := RestoreGraphDispute(flip(4), fix.graph, 50); err == nil {
		t.Fatal("bad version accepted")
	}
	if _, err := RestoreGraphDispute(flip(5), fix.graph, 50); err == nil {
		t.Fatal("tampered graph id accepted")
	}
	if _, err := RestoreGraphDispute(snapshot[:len(snapshot)-1], fix.graph, 50); err == nil {
		t.Fatal("truncated snapshot accepted")
	}
	// high beyond the trail length
	bad := append([]byte(nil), snapshot...)
	steps := uint32(len(fix.graph.Nodes))
	o := 5 + 32 + 8 + 64 // magic+version+graphid+period+roots
	bad[o+4] = byte(steps >> 24)
	bad[o+5] = byte(steps >> 16)
	bad[o+6] = byte(steps >> 8)
	bad[o+7] = byte(steps + 3)
	if _, err := RestoreGraphDispute(bad, fix.graph, 50); err == nil {
		t.Fatal("out-of-range interval accepted")
	}
	// wrong round period
	if _, err := RestoreGraphDispute(snapshot, fix.graph, 99); err == nil {
		t.Fatal("wrong round period accepted")
	}
}

func TestGraphReceiptBindings(t *testing.T) {
	fix := buildBlockFixture(t, miniBlockConfig())
	workerKey := bytes.Repeat([]byte{7}, 32)
	receipt, err := BuildVerifiedGraphWorkReceiptV1(fix.graph,
		[]byte{1, 2, 3}, []byte{4, 5}, workerKey,
		fix.exec.Outputs, fix.exec.WorkVector, ModeOptimisticUnchallenged, 42,
		[]byte("settle"), nil)
	if err != nil {
		t.Fatal(err)
	}
	id1, err := receipt.ReceiptID()
	if err != nil {
		t.Fatal(err)
	}
	again, err := BuildVerifiedGraphWorkReceiptV1(fix.graph,
		[]byte{1, 2, 3}, []byte{4, 5}, workerKey,
		fix.exec.Outputs, fix.exec.WorkVector, ModeOptimisticUnchallenged, 42,
		[]byte("settle"), nil)
	if err != nil {
		t.Fatal(err)
	}
	id2, err := again.ReceiptID()
	if err != nil {
		t.Fatal(err)
	}
	if id1 != id2 {
		t.Fatal("receipt id is not deterministic")
	}
	if err := receipt.VerifyReceiptBinding(fix.graph, fix.exec.Outputs); err != nil {
		t.Fatal(err)
	}
	if receipt.WorkVector.Get("GEMM_MAC", -1) != fix.exec.WorkVector.Get("GEMM_MAC", -2) {
		t.Fatal("receipt work vector lost the GEMM counters")
	}

	// Tampered outputs must not verify.
	bad := []Hash{fix.exec.Outputs[0]}
	bad[0][0] ^= 0x01
	if err := receipt.VerifyReceiptBinding(fix.graph, bad); err == nil {
		t.Fatal("tampered outputs accepted")
	}
	// A different graph must not verify either.
	tampered := *receipt
	tampered.GraphID = append([]byte(nil), receipt.GraphID...)
	tampered.GraphID[0] ^= 0x01
	if err := tampered.VerifyReceiptBinding(fix.graph, fix.exec.Outputs); err == nil {
		t.Fatal("tampered graph id accepted")
	}

	if _, err := BuildVerifiedGraphWorkReceiptV1(fix.graph, nil, nil, workerKey,
		fix.exec.Outputs, fix.exec.WorkVector, "bogus_mode", 1, nil, nil); err == nil {
		t.Fatal("bad mode accepted")
	}
	if _, err := BuildVerifiedGraphWorkReceiptV1(fix.graph, nil, nil, workerKey[:16],
		fix.exec.Outputs, fix.exec.WorkVector, ModeChallengedWorkerWon, 1, nil, nil); err == nil {
		t.Fatal("short worker key accepted")
	}
	if _, err := BuildVerifiedGraphWorkReceiptV1(fix.graph, nil, nil, workerKey,
		fix.exec.Outputs, fix.exec.WorkVector, ModeChallengedWorkerWon, 1, nil, bytes.Repeat([]byte{1}, 33)); err == nil {
		t.Fatal("bad digest length accepted")
	}
	// The receipt must encode canonically (id above proves it) and the
	// challenged mode is allowed for a worker who survived a dispute.
	if _, err := BuildVerifiedGraphWorkReceiptV1(fix.graph, nil, nil, workerKey,
		fix.exec.Outputs, fix.exec.WorkVector, ModeChallengedWorkerWon, 9, nil, bytes.Repeat([]byte{2}, 32)); err != nil {
		t.Fatal(err)
	}
}

func TestGraphWorkVectorMatchesExecution(t *testing.T) {
	for _, cfg := range []BlockConfig{miniBlockConfig(), mediumBlockConfig()} {
		if cfg.MLPHidden == mediumBlockConfig().MLPHidden && testing.Short() {
			t.Skip("medium block runs only outside -short")
		}
		fix := buildBlockFixture(t, cfg)
		derived, err := GraphWorkVector(fix.graph)
		if err != nil {
			t.Fatal(err)
		}
		if len(derived) != len(fix.exec.WorkVector) {
			t.Fatalf("work vector key count %d != execution %d", len(derived), len(fix.exec.WorkVector))
		}
		for _, pair := range fix.exec.WorkVector {
			if got := derived.Get(pair.Key, -1); got != pair.Value {
				t.Fatalf("derived work %s = %d, executed %d", pair.Key, got, pair.Value)
			}
		}
	}
}

func TestGraphResultCommit(t *testing.T) {
	fix := buildBlockFixture(t, miniBlockConfig())
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	rc, err := NewGraphResultCommit(fix.graph, []byte("task-1"), []byte("assign-1"), pub, fix.exec.Outputs, 7)
	if err != nil {
		t.Fatal(err)
	}
	if err := SignGraphResultCommit(rc, priv); err != nil {
		t.Fatal(err)
	}
	if err := ValidateGraphResultCommit(fix.graph, rc); err != nil {
		t.Fatal(err)
	}

	// Tamper: output root bytes.
	bad := *rc
	bad.OutputRoots = make([][]byte, len(rc.OutputRoots))
	for i, raw := range rc.OutputRoots {
		bad.OutputRoots[i] = append([]byte(nil), raw...)
	}
	bad.OutputRoots[0][0] ^= 0x01
	if err := ValidateGraphResultCommit(fix.graph, &bad); err == nil {
		t.Fatal("tampered output root accepted")
	}
	// Tamper: signature.
	bad2 := *rc
	bad2.Signature = append([]byte(nil), rc.Signature...)
	bad2.Signature[0] ^= 0x01
	if err := ValidateGraphResultCommit(fix.graph, &bad2); err == nil {
		t.Fatal("tampered signature accepted")
	}
	// Tamper: graph id.
	bad3 := *rc
	bad3.GraphID = append([]byte(nil), rc.GraphID...)
	bad3.GraphID[0] ^= 0x01
	if err := ValidateGraphResultCommit(fix.graph, &bad3); err == nil {
		t.Fatal("tampered graph id accepted")
	}
	// An unsigned commit must not validate.
	if err := ValidateGraphResultCommit(fix.graph, &GraphResultCommit{
		ProtocolVersion: GraphResultCommitVersion,
		GraphID:         rc.GraphID, TaskRef: rc.TaskRef, AssignmentRef: rc.AssignmentRef,
		WorkerPubKey: rc.WorkerPubKey, FinalOutputRoot: rc.FinalOutputRoot, OutputRoots: rc.OutputRoots,
	}); err == nil {
		t.Fatal("unsigned commit accepted")
	}
}
