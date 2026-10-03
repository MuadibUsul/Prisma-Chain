package gemmv1

// CPU reference GEMM and the worker normal path.

import (
	"bytes"
	"testing"
)

// naiveGEMM is an independent implementation used as a cross-check oracle.
func naiveGEMM(a, b []int8, m, n, k uint64) []int32 {
	c := make([]int32, m*n)
	for i := uint64(0); i < m; i++ {
		for j := uint64(0); j < n; j++ {
			var acc int32
			for t := uint64(0); t < k; t++ {
				acc += int32(a[i*k+t]) * int32(b[t*n+j])
			}
			c[i*n+j] = acc
		}
	}
	return c
}

// A: repeated execution is deterministic and matches an independent oracle.
func TestReferenceGEMMDeterministicAndExact(t *testing.T) {
	sizes := []struct{ m, n, k uint64 }{
		{5, 4, 12},
		{8, 8, 8},
		{17, 9, 23},
		{32, 48, 64},
		{1, 1, MaxSafeK},
	}
	for _, s := range sizes {
		a := GenTestMatrix('A', uint32(s.m+s.k), s.m*s.k)
		b := GenTestMatrix('B', uint32(s.n+s.k), s.k*s.n)
		c1 := ReferenceGEMM(a, b, s.m, s.n, s.k)
		c2 := ReferenceGEMM(a, b, s.m, s.n, s.k)
		if !bytes.Equal(Int32sToCanonical(c1), Int32sToCanonical(c2)) {
			t.Fatalf("reference GEMM not deterministic at %v", s)
		}
		want := naiveGEMM(a, b, s.m, s.n, s.k)
		for i := range c1 {
			if c1[i] != want[i] {
				t.Fatalf("reference != naive at %v, index %d", s, i)
			}
		}
	}
}

func TestReferenceGEMMKnownSmallCase(t *testing.T) {
	a := []int8{1, 2, 3, 4}
	b := []int8{5, 6, 7, 8}
	want := []int32{19, 22, 43, 50}
	got := ReferenceGEMM(a, b, 2, 2, 2)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v want %v", got, want)
		}
	}
}

func TestExecuteTaskHappyPath(t *testing.T) {
	f := newFixture(t, 8, 8, 16, 42, true)
	if !VerifyResultCommitSignature(&f.result.ResultCommit) {
		t.Fatal("worker signature does not verify")
	}
	// A: repeated execution gives the identical output root and commit bytes.
	again, err := ExecuteTask(f.task, f.assignment, f.matrixA, f.matrixB, 1002)
	if err != nil {
		t.Fatal(err)
	}
	if again.OutputRoot != f.result.OutputRoot {
		t.Fatal("output root not deterministic")
	}
	if !bytes.Equal(Int32sToCanonical(again.Tiles[0][:]), Int32sToCanonical(f.result.Tiles[0][:])) {
		t.Fatal("output tiles not deterministic")
	}

	// Output tree must verify for every tile.
	taskID, _ := f.task.TaskID()
	levels, err := buildLevels(OutputLeaves(taskID, f.assignmentID, f.result.Tiles, f.colsC))
	if err != nil {
		t.Fatal(err)
	}
	for idx := range f.result.Tiles {
		proof, err := ProveLeaf(levels, uint32(idx))
		if err != nil {
			t.Fatal(err)
		}
		leaf := LeafOutputTile(taskID, f.assignmentID, uint32(idx)/f.colsC, uint32(idx)%f.colsC, f.result.Tiles[idx].CanonicalBytes())
		if !VerifyLeafInclusion(f.result.OutputRoot, leaf, proof) {
			t.Fatalf("output proof rejected for tile %d", idx)
		}
	}
}

func TestExecuteTaskRejectsTamperedInputs(t *testing.T) {
	f := newFixture(t, 8, 8, 16, 42, true)
	bad := append([]int8(nil), f.matrixA...)
	bad[0] ^= 0x01
	if _, err := ExecuteTask(f.task, f.assignment, bad, f.matrixB, 1002); err == nil {
		t.Fatal("tampered matrix A accepted")
	}
	badB := append([]int8(nil), f.matrixB...)
	badB[len(badB)-1] ^= 0x01
	if _, err := ExecuteTask(f.task, f.assignment, f.matrixA, badB, 1002); err == nil {
		t.Fatal("tampered matrix B accepted")
	}
}

func TestOutputTreeBindsTaskAndAssignment(t *testing.T) {
	fA := newFixture(t, 8, 8, 16, 42, true)
	fB := newFixture(t, 8, 8, 16, 42, true) // same shape/seed, different keys
	if bytes.Equal(fA.taskID, fB.taskID) {
		t.Fatal("different requesters must produce different task ids")
	}
	// The same tile bytes under a different task/assignment must hash to a
	// different leaf, so proofs cannot be replayed across tasks.
	leafA := LeafOutputTile(fA.taskID, fA.assignmentID, 0, 0, fA.result.Tiles[0].CanonicalBytes())
	leafB := LeafOutputTile(fB.taskID, fB.assignmentID, 0, 0, fA.result.Tiles[0].CanonicalBytes())
	if leafA == leafB {
		t.Fatal("output leaf does not bind task/assignment")
	}
}
