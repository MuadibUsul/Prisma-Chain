package gemmv1

// On-demand tile trace generation and lock verification.

import (
	"testing"
)

func TestBuildTileTraceEndpoints(t *testing.T) {
	const m, n, k = 8, 8, 24 // R = 3
	f := newFixture(t, m, n, k, 11, true)
	tileI, tileJ := uint32(0), uint32(0)
	art, err := BuildTileTrace(f.matrixA, f.matrixB, m, n, k, f.taskID, f.assignmentID, tileI, tileJ)
	if err != nil {
		t.Fatal(err)
	}
	if len(art.States) != int(RSteps(k))+1 {
		t.Fatalf("trace length %d, want R+1=%d", len(art.States), RSteps(k)+1)
	}
	if !art.States[0].IsZero() {
		t.Fatal("S0 must be the zero matrix")
	}
	// S_R must equal the committed output tile.
	if art.States[RSteps(k)] != f.result.Tiles[tileI*f.colsC+tileJ] {
		t.Fatal("S_R does not match the output tile")
	}

	commit, err := BuildTraceCommit(f.task, f.assignmentID, Worker, tileI, tileJ, art, 1011)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyTraceCommit(f.task, f.assignmentID, commit); err != nil {
		t.Fatalf("trace commit rejected: %v", err)
	}

	// Tampering with any committed field must break an endpoint proof.
	tampered := *commit
	tampered.FinalState = append([]byte(nil), commit.FinalState...)
	tampered.FinalState[0] ^= 0x01
	if err := VerifyTraceCommit(f.task, f.assignmentID, &tampered); err == nil {
		t.Fatal("tampered S_R accepted")
	}
	tampered = *commit
	tampered.InitialState[0] = 0x01
	if err := VerifyTraceCommit(f.task, f.assignmentID, &tampered); err == nil {
		t.Fatal("non-zero S0 accepted")
	}
}

func TestTraceRootsDifferByParty(t *testing.T) {
	const m, n, k = 8, 8, 16
	f := newFixture(t, m, n, k, 12, true)
	workerArt, err := BuildTileTrace(f.matrixA, f.matrixB, m, n, k, f.taskID, f.assignmentID, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Challenger computes independently with the same generator input.
	challArt, err := BuildTileTrace(f.matrixA, f.matrixB, m, n, k, f.taskID, f.assignmentID, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if workerArt.Root != challArt.Root {
		t.Fatal("independent honest traces must have identical roots")
	}
}
