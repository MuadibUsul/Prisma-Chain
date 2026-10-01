package vm

import "testing"

func TestRejectNoncanonicalPadding(t *testing.T) {
	states := []State{{PC: 0}, {PC: 1}, {PC: 2}}
	execution := newExecution(states)
	state, proof, err := execution.Proof(2)
	if err != nil || !VerifyProof(execution.Root(), state, 2, 3, proof) {
		t.Fatalf("valid final proof rejected: %v", err)
	}
	bad := newExecution(states)
	bad.levels[0][3] = Hash{1}
	bad.levels[1][1] = nodeHash(bad.levels[0][2], bad.levels[0][3])
	bad.levels[2][0] = nodeHash(bad.levels[1][0], bad.levels[1][1])
	state, proof, _ = bad.Proof(2)
	if VerifyProof(bad.Root(), state, 2, 3, proof) {
		t.Fatal("noncanonical padded leaf accepted")
	}
}
