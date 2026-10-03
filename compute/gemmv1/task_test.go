package gemmv1

// Task validation, admission (including the int32 overflow bound), TaskID
// determinism and ResultCommit replay binding.

import (
	"bytes"
	"errors"
	"testing"
)

func TestTaskValidateAndAdmission(t *testing.T) {
	f := newFixture(t, 8, 8, 16, 42, true)
	if err := f.task.Validate(); err != nil {
		t.Fatalf("valid task rejected: %v", err)
	}
	if f.task.CanonicalMACCount() != 8*8*16 {
		t.Fatalf("canonical_mac_count = %d, want %d", f.task.CanonicalMACCount(), 8*8*16)
	}

	// K above the int32-safe accumulator bound must be rejected at admission.
	over := *f.task
	over.K = MaxSafeK + 1
	if !errors.Is(over.Validate(), errKUnsafe) {
		t.Fatalf("K=%d should be rejected with errKUnsafe, got %v", over.K, over.Validate())
	}
	safe := *f.task
	safe.K = MaxSafeK
	if err := safe.Validate(); err != nil {
		t.Fatalf("K=MaxSafeK should be admitted, got %v", err)
	}
	// Sanity: the bound really prevents accumulator overflow.
	if uint64(127*127)*uint64(MaxSafeK) > 1<<31-1 {
		t.Fatalf("MaxSafeK %d does not prevent int32 overflow", MaxSafeK)
	}

	zero := *f.task
	zero.M = 0
	if zero.Validate() == nil {
		t.Fatal("M=0 must be rejected")
	}
	badTile := *f.task
	badTile.TileSize = 4
	if badTile.Validate() == nil {
		t.Fatal("tile_size 4 must be rejected")
	}
}

func TestTaskIDDeterministic(t *testing.T) {
	f1 := newFixture(t, 8, 8, 16, 42, false)
	f2 := newFixture(t, 8, 8, 16, 42, false)
	id1, _ := f1.task.TaskID()
	id2, _ := f2.task.TaskID()
	if !bytes.Equal(id1, id2) {
		t.Fatalf("task id differs across builds: %x vs %x", id1, id2)
	}
	// Any single field change must change the task id.
	changed := *f1.task
	changed.IssuedEpoch++
	id3, _ := changed.TaskID()
	if bytes.Equal(id1, id3) {
		t.Fatal("task id did not change with the descriptor")
	}
}

func TestCWUnitsDisplay(t *testing.T) {
	if got := CWUnits(1 << 20); got != 1 {
		t.Fatalf("1 CWU should equal 2^20 MACs, got %v", got)
	}
	if got := CWUnits(4 << 20); got != 4 {
		t.Fatalf("4 CWU expected, got %v", got)
	}
}

// F: ResultCommit replay protection.
func TestResultCommitReplayBinding(t *testing.T) {
	fA := newFixture(t, 8, 8, 16, 42, true)
	rcA := fA.result.ResultCommit

	// Same task, same assignment: accepted.
	if err := ValidateResultCommit(fA.task, fA.assignment, &rcA); err != nil {
		t.Fatalf("valid commit rejected: %v", err)
	}

	// Task B with the same shape but different seed: commit A must fail.
	fB := newFixture(t, 8, 8, 16, 4242, true)
	if err := ValidateResultCommit(fB.task, fB.assignment, &rcA); err == nil {
		t.Fatal("task A commit accepted under task B")
	}

	// Worker B's assignment must not accept worker A's commit.
	assignmentAWorkerB := *fB.assignment
	assignmentAWorkerB.TaskID = fA.taskID
	if err := ValidateResultCommit(fA.task, &assignmentAWorkerB, &rcA); err == nil {
		t.Fatal("worker A commit accepted under worker B assignment")
	}

	// A wrong canonical_mac_count must be rejected.
	badCount := rcA
	badCount.CanonicalMACCount++
	if err := ValidateResultCommit(fA.task, fA.assignment, &badCount); err == nil {
		t.Fatal("wrong canonical_mac_count accepted")
	}
}
