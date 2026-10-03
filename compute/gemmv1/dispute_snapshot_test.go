package gemmv1

// Phase D consensus-support tests: dispute snapshot round trip and
// transcript chain determinism.

import (
	"bytes"
	"testing"
)

func disputeForSnapshot(t *testing.T, k uint64) (*GEMMDispute, *TaskDescriptor, *Assignment, []byte, *TileTraceArtifacts, *TileTraceArtifacts) {
	t.Helper()
	const m, n = 8, 8
	a := GenTestMatrix('A', 5, m*k)
	b := GenTestMatrix('B', 6, k*n)
	rootA, rootB, err := BuildMatrixRoots(a, b, m, n, k)
	if err != nil {
		t.Fatal(err)
	}
	task, err := NewTaskDescriptor(make([]byte, 32), []byte("nonce"), 1000, m, n, k, rootA[:], rootB[:], 100, 1000)
	if err != nil {
		t.Fatal(err)
	}
	taskID, _ := task.TaskID()
	assignment := &Assignment{TaskID: taskID, WorkerPubKey: make([]byte, 32), AssignmentNonce: []byte("an"), AcceptedEpoch: 1001}
	assignmentID, _ := assignment.AssignmentID()
	honest, err := BuildTileTrace(a, b, m, n, k, taskID, assignmentID, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	// A fabricated worker trace ending at a corrupted tile.
	fraudFinal := honest.States[rStepsOf(k)]
	fraudFinal[0]++
	fraud := &TileTraceArtifacts{States: append([]State(nil), honest.States...), Levels: honest.Levels, Root: honest.Root}
	fraud.States[rStepsOf(k)] = fraudFinal
	fLeaves := make([]Hash, len(fraud.States))
	for step, s := range fraud.States {
		fLeaves[step] = LeafTraceState(taskID, assignmentID, 0, 0, uint32(step), s.CanonicalBytes())
	}
	fLevels, err := BuildLevels(fLeaves)
	if err != nil {
		t.Fatal(err)
	}
	fraud.Levels = fLevels
	fraud.Root = fLevels[len(fLevels)-1][0]
	cfg := DisputeConfig{
		Task: task, Assignment: assignment, TaskID: taskID, TileI: 0, TileJ: 0,
		WorkerTile:     fraud.States[rStepsOf(k)].CanonicalBytes(),
		ChallengerTile: honest.States[rStepsOf(k)].CanonicalBytes(),
		MatrixARoot:    rootA, MatrixBRoot: rootB, RoundPeriod: 50,
	}
	wc, err := BuildTraceCommit(task, assignmentID, Worker, 0, 0, fraud, 1011)
	if err != nil {
		t.Fatal(err)
	}
	cc, err := BuildTraceCommit(task, assignmentID, Challenger, 0, 0, honest, 1011)
	if err != nil {
		t.Fatal(err)
	}
	d, err := NewGEMMDispute(cfg,
		TraceClaim{Party: Worker, TraceRoot: fraud.Root, InitialState: wc.InitialState, InitialProof: wc.InitialProof, FinalState: wc.FinalState, FinalProof: wc.FinalProof},
		TraceClaim{Party: Challenger, TraceRoot: honest.Root, InitialState: cc.InitialState, InitialProof: cc.InitialProof, FinalState: cc.FinalState, FinalProof: cc.FinalProof},
		1010)
	if err != nil {
		t.Fatal(err)
	}
	return d, task, assignment, taskID, fraud, honest
}

func rStepsOf(k uint64) uint32 { return uint32((k + 7) / 8) }

func TestGEMMDisputeSnapshotRoundTrip(t *testing.T) {
	d, task, assignment, taskID, workerArt, challArt := disputeForSnapshot(t, 16)
	// One bisection round so the snapshot carries midpoint submissions.
	if _, err := SubmitMidFromTrace(d, Worker, workerArt, 1012); err != nil {
		t.Fatal(err)
	}
	blob, err := d.SnapshotV1()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := RestoreGEMMDispute(blob, d.cfg)
	if err != nil {
		t.Fatal(err)
	}
	if restored.Status().Low != d.Status().Low || restored.Status().High != d.Status().High ||
		restored.Status().WorkerSubmitted != d.Status().WorkerSubmitted ||
		restored.deadline != d.deadline {
		t.Fatalf("snapshot round trip changed dispute state: %+v vs %+v", restored.Status(), d.Status())
	}
	// The restored dispute must accept the remaining round and produce a
	// byte-identical snapshot to the in-memory dispute driven through the
	// same steps.
	if _, err := SubmitMidFromTrace(restored, Challenger, challArt, 1013); err != nil {
		t.Fatal(err)
	}
	if _, err := SubmitMidFromTrace(d, Challenger, challArt, 1013); err != nil {
		t.Fatal(err)
	}
	restoredBlob, err := restored.SnapshotV1()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(restoredBlob, mustSnap(t, d)) {
		t.Fatal("restored dispute diverged from the in-memory dispute")
	}
	// Tampered or foreign snapshots must be rejected.
	if _, err := RestoreGEMMDispute(append(append([]byte(nil), blob...), 0), d.cfg); err == nil {
		t.Fatal("oversized snapshot accepted")
	}
	if _, err := RestoreGEMMDispute(append([]byte(nil), blob[:len(blob)-1]...), d.cfg); err == nil {
		t.Fatal("truncated snapshot accepted")
	}
	other := *task
	other.K = 8
	if _, err := RestoreGEMMDispute(blob, DisputeConfig{Task: &other, Assignment: assignment, TaskID: taskID, TileI: 0, TileJ: 0, RoundPeriod: 50}); err == nil {
		t.Fatal("snapshot accepted under a different task")
	}
}

func mustSnap(t *testing.T, d *GEMMDispute) []byte {
	t.Helper()
	blob, err := d.SnapshotV1()
	if err != nil {
		t.Fatal(err)
	}
	return blob
}

func TestTranscriptChainDeterministic(t *testing.T) {
	ev := TranscriptTraceLocked{Party: 1, TraceRoot: make([]byte, 32)}
	h1, err := TranscriptStep([32]byte{}, TranscriptTagTraceLocked, ev)
	if err != nil {
		t.Fatal(err)
	}
	h2, err := TranscriptStep(h1, TranscriptTagArbitration, TranscriptArbitration{Step: 3, Outcome: 2, Expected: make([]byte, 256)})
	if err != nil {
		t.Fatal(err)
	}
	again1, _ := TranscriptStep([32]byte{}, TranscriptTagTraceLocked, ev)
	again2, _ := TranscriptStep(again1, TranscriptTagArbitration, TranscriptArbitration{Step: 3, Outcome: 2, Expected: make([]byte, 256)})
	if h1 != again1 || h2 != again2 {
		t.Fatal("transcript chain is not deterministic")
	}
	if h1 == h2 {
		t.Fatal("distinct events produced the same digest")
	}
}
