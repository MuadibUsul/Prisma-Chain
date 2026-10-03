package gemmv1

// Verified Work Receipt construction and identity.

import (
	"bytes"
	"testing"
)

func TestReceiptOptimisticPath(t *testing.T) {
	f := newFixture(t, 8, 8, 16, 42, true)
	rc := &f.result.ResultCommit
	receipt, err := BuildVerifiedWorkReceipt(f.task, f.assignment, rc, ModeOptimisticUnchallenged, 2000, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.CanonicalMACCount != f.task.CanonicalMACCount() {
		t.Fatal("receipt mac count mismatch")
	}
	if !bytes.Equal(receipt.OutputRoot, rc.OutputRoot) {
		t.Fatal("receipt output root mismatch")
	}
	if len(receipt.DisputeTranscriptDigest) != 0 {
		t.Fatal("unchallenged receipt must have an empty transcript digest")
	}
	id1, err := receipt.ReceiptID()
	if err != nil {
		t.Fatal(err)
	}
	again, _ := BuildVerifiedWorkReceipt(f.task, f.assignment, rc, ModeOptimisticUnchallenged, 2000, nil, nil)
	id2, _ := again.ReceiptID()
	if !bytes.Equal(id1, id2) {
		t.Fatal("receipt id not deterministic")
	}
}

func TestReceiptChallengedWorkerWon(t *testing.T) {
	f := newFixture(t, 8, 8, 16, 42, true)
	honest := f.challengerHonestTiles()
	wrongTiles := append([]State(nil), honest...)
	wrongTiles[0][0] -= 5
	d, workerArt, challArt := openChallenge(t, f, honest, wrongTiles, 1010)
	runBisection(t, d, workerArt, challArt)
	step := d.Status().Low
	aTile := ExtractATile(f.matrixA, f.task.M, f.task.K, 0, uint64(step))
	bTile := ExtractBTile(f.matrixB, f.task.K, f.task.N, uint64(step), 0)
	aProof, _ := ProveInputTile(MatrixIDA, f.matrixA, f.matrixB, f.task.M, f.task.N, f.task.K, 0, uint64(step))
	bProof, _ := ProveInputTile(MatrixIDB, f.matrixA, f.matrixB, f.task.M, f.task.N, f.task.K, uint64(step), 0)
	outcome, err := d.Arbitrate(aTile, bTile, aProof, bProof, 1030)
	if err != nil || outcome != WorkerWins {
		t.Fatalf("worker should win: outcome=%d err=%v", outcome, err)
	}
	receipt, err := BuildVerifiedWorkReceipt(f.task, f.assignment, &f.result.ResultCommit, ModeChallengedWorkerWon, 2000, nil, d.TranscriptDigest())
	if err != nil {
		t.Fatalf("winning worker must be able to finalize: %v", err)
	}
	if len(receipt.DisputeTranscriptDigest) != 32 {
		t.Fatal("challenged receipt must carry the transcript digest")
	}
	if _, err := receipt.ReceiptID(); err != nil {
		t.Fatal(err)
	}
}

func TestReceiptRejections(t *testing.T) {
	f := newFixture(t, 8, 8, 16, 42, true)
	if _, err := BuildVerifiedWorkReceipt(f.task, f.assignment, &f.result.ResultCommit, "made_up_mode", 2000, nil, nil); err == nil {
		t.Fatal("unknown verification mode accepted")
	}
	badDigest := []byte{0x01}
	if _, err := BuildVerifiedWorkReceipt(f.task, f.assignment, &f.result.ResultCommit, ModeOptimisticUnchallenged, 2000, nil, badDigest); err == nil {
		t.Fatal("non-32-byte transcript digest accepted")
	}
	badCommit := f.result.ResultCommit
	badCommit.OutputRoot[0] ^= 0xff
	if _, err := BuildVerifiedWorkReceipt(f.task, f.assignment, &badCommit, ModeOptimisticUnchallenged, 2000, nil, nil); err == nil {
		t.Fatal("receipt accepted over a foreign output root")
	}
}
