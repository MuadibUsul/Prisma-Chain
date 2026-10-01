package vm

import "testing"

func TestMonitorReplaysBeforeAttestingOrChallenging(t *testing.T) {
	program, input, honest, forged := traces(t)
	good, err := ReviewClaim(program, input, claim(t, honest))
	if err != nil || good.Action != Attest || good.Canonical.Root != honest.Root() {
		t.Fatalf("honest trace was not attested: action=%q err=%v", good.Action, err)
	}
	bad, err := ReviewClaim(program, input, claim(t, forged))
	if err != nil || bad.Action != Challenge {
		t.Fatalf("bad result did not trigger a challenge: action=%q err=%v", bad.Action, err)
	}
	dispute, err := NewDispute(program, input, claim(t, forged), bad.Canonical, 100, 5)
	if err != nil {
		t.Fatal(err)
	}
	for dispute.Status().Outcome == Pending {
		mid := dispute.Status().Mid
		wState, wProof, _ := forged.Proof(mid)
		cState, cProof, _ := bad.Trace.Proof(mid)
		if _, err := dispute.SubmitMid(Worker, wState, wProof, dispute.Status().Deadline); err != nil {
			t.Fatal(err)
		}
		if _, err := dispute.SubmitMid(Challenger, cState, cProof, dispute.Status().Deadline); err != nil {
			t.Fatal(err)
		}
	}
	if dispute.Status().Outcome != ChallengerWins {
		t.Fatalf("full replay could not defeat forged trace: %v", dispute.Status().Outcome)
	}
	interior := append([]State(nil), honest.states...)
	interior[3].Reg[7]++
	withheld, err := ReviewClaim(program, input, claim(t, newExecution(interior)))
	if err != nil || withheld.Action != Withhold {
		t.Fatalf("malformed interior trace must not be attested or challenged: action=%q err=%v", withheld.Action, err)
	}
}
