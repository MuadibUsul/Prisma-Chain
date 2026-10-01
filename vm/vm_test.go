package vm

import (
	"math"
	"testing"
)

func classifier(t *testing.T) (Program, []int64) {
	t.Helper()
	p, err := NewProgram([]Instruction{
		{Op: OpInput, Dst: 1, Imm: 0},
		{Op: OpSet, Dst: 2, Imm: 2},
		{Op: OpMul, Dst: 3, A: 1, B: 2},
		{Op: OpInput, Dst: 4, Imm: 1},
		{Op: OpAdd, Dst: 0, A: 3, B: 4},
		{Op: OpSet, Dst: 5},
		{Op: OpGT, Dst: 6, A: 0, B: 5},
		{Op: OpSelect, Dst: 0, A: 6, B: 0, C: 5},
	})
	if err != nil {
		t.Fatal(err)
	}
	return p, []int64{3, -2}
}

func claim(t *testing.T, e Execution) TraceClaim {
	t.Helper()
	_, initial, err := e.Proof(0)
	if err != nil {
		t.Fatal(err)
	}
	final, terminal, err := e.Proof(e.Steps())
	if err != nil {
		t.Fatal(err)
	}
	return TraceClaim{Root: e.Root(), InitialProof: initial, Final: final, FinalProof: terminal}
}

func traces(t *testing.T) (Program, []int64, Execution, Execution) {
	t.Helper()
	p, input := classifier(t)
	honest, err := Execute(p, input)
	if err != nil {
		t.Fatal(err)
	}
	states := append([]State(nil), honest.states...)
	for i := 5; i < len(states); i++ {
		states[i].Reg[0] = 999
	}
	return p, input, honest, newExecution(states)
}

func TestExecutionAndProofs(t *testing.T) {
	p, input, honest, _ := traces(t)
	if honest.Output() != 4 || honest.Steps() != uint32(p.Len()) {
		t.Fatalf("unexpected execution: output %d, steps %d", honest.Output(), honest.Steps())
	}
	for index := uint32(0); index <= honest.Steps(); index++ {
		state, proof, err := honest.Proof(index)
		if err != nil || !VerifyProof(honest.Root(), state, index, honest.Steps()+1, proof) {
			t.Fatalf("valid proof at %d rejected: %v", index, err)
		}
		state.Reg[0]++
		if VerifyProof(honest.Root(), state, index, honest.Steps()+1, proof) {
			t.Fatalf("modified state at %d accepted", index)
		}
	}
	if _, _, err := honest.Proof(honest.Steps() + 1); err == nil {
		t.Fatal("out-of-range proof accepted")
	}
	if InputDigest(input) == InputDigest([]int64{-2, 3}) {
		t.Fatal("input commitment ignored order")
	}
}

func TestProgramCopyAndArithmetic(t *testing.T) {
	code := []Instruction{{Op: OpSet, Dst: 1, Imm: -7}, {Op: OpSet, Dst: 2, Imm: 3}, {Op: OpDiv, Dst: 0, A: 1, B: 2}, {Op: OpMulDiv, Dst: 3, A: 1, B: 2, Imm: 2}}
	p, err := NewProgram(code)
	if err != nil {
		t.Fatal(err)
	}
	digest := p.Digest()
	code[0].Imm = 99
	e, err := Execute(p, nil)
	if err != nil || e.Output() != -2 || e.Final().Reg[3] != -10 || p.Digest() != digest {
		t.Fatalf("copy or truncation failure: output %d, scaled %d, err %v", e.Output(), e.Final().Reg[3], err)
	}
	invalid := []Instruction{{Op: OpSet, Dst: 1, Imm: math.MaxInt64}, {Op: OpSet, Dst: 2, Imm: 1}, {Op: OpAdd, Dst: 0, A: 1, B: 2}}
	p, err = NewProgram(invalid)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Execute(p, nil); err == nil {
		t.Fatal("overflow accepted")
	}
	if _, err := NewProgram([]Instruction{{Op: OpInput, Dst: 0, Imm: MaxInputValues}}); err == nil {
		t.Fatal("out-of-range input accepted")
	}
}

func runDispute(t *testing.T, p Program, input []int64, worker, challenger Execution) Outcome {
	t.Helper()
	d, err := NewDispute(p, input, claim(t, worker), claim(t, challenger), 100, 5)
	if err != nil {
		t.Fatal(err)
	}
	for d.Status().Outcome == Pending {
		mid := d.Status().Mid
		wState, wProof, _ := worker.Proof(mid)
		cState, cProof, _ := challenger.Proof(mid)
		if _, err := d.SubmitMid(Worker, wState, wProof, d.Status().Deadline); err != nil {
			t.Fatal(err)
		}
		if _, err := d.SubmitMid(Challenger, cState, cProof, d.Status().Deadline); err != nil {
			t.Fatal(err)
		}
	}
	return d.Status().Outcome
}

func TestBisectionFindsDishonestTrace(t *testing.T) {
	p, input, honest, forged := traces(t)
	if got := runDispute(t, p, input, forged, honest); got != ChallengerWins {
		t.Fatalf("dishonest worker won: %v", got)
	}
	if got := runDispute(t, p, input, honest, forged); got != WorkerWins {
		t.Fatalf("false challenger won: %v", got)
	}
	badClaim := claim(t, forged)
	badClaim.Final.Reg[0]++
	if _, err := NewDispute(p, input, badClaim, claim(t, honest), 100, 5); err == nil {
		t.Fatal("invalid endpoint proof accepted")
	}
}

func TestDisputeTimeoutAndMidpointProof(t *testing.T) {
	p, input, honest, forged := traces(t)
	d, err := NewDispute(p, input, claim(t, honest), claim(t, forged), 100, 5)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.Timeout(105); err == nil {
		t.Fatal("premature timeout accepted")
	}
	state, proof, _ := honest.Proof(d.Status().Mid)
	state.Reg[0]++
	if _, err := d.SubmitMid(Worker, state, proof, 101); err == nil {
		t.Fatal("invalid midpoint accepted")
	}
	state.Reg[0]--
	if _, err := d.SubmitMid(Worker, state, proof, 101); err != nil {
		t.Fatal(err)
	}
	if got, err := d.Timeout(106); err != nil || got != WorkerWins {
		t.Fatalf("missing challenger did not lose: %v, %v", got, err)
	}
	d, err = NewDispute(p, input, claim(t, honest), claim(t, forged), 100, 5)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := d.Timeout(106); err != nil || got != BothInvalid {
		t.Fatalf("dual timeout: %v, %v", got, err)
	}
}

func TestDisputeSnapshotRoundTrip(t *testing.T) {
	p, input, honest, forged := traces(t)
	d, err := NewDispute(p, input, claim(t, forged), claim(t, honest), 10, 3)
	if err != nil {
		t.Fatal(err)
	}
	state, proof, _ := forged.Proof(d.Status().Mid)
	if _, err := d.SubmitMid(Worker, state, proof, 11); err != nil {
		t.Fatal(err)
	}
	blob, err := d.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := RestoreDispute(blob, p, input)
	if err != nil || restored.Status() != d.Status() {
		t.Fatalf("snapshot round trip failed: %v", err)
	}
	if _, err := RestoreDispute(blob[:len(blob)-1], p, input); err == nil {
		t.Fatal("truncated snapshot accepted")
	}
	if _, err := RestoreDispute(blob, p, []int64{4, -2}); err == nil {
		t.Fatal("mismatched input accepted")
	}
	for restored.Status().Outcome == Pending {
		mid := restored.Status().Mid
		if !restored.Status().WorkerSubmitted {
			s, pr, _ := forged.Proof(mid)
			if _, err := restored.SubmitMid(Worker, s, pr, restored.Status().Deadline); err != nil {
				t.Fatal(err)
			}
		}
		s, pr, _ := honest.Proof(mid)
		if _, err := restored.SubmitMid(Challenger, s, pr, restored.Status().Deadline); err != nil {
			t.Fatal(err)
		}
		blob, err = restored.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		restored, err = RestoreDispute(blob, p, input)
		if err != nil {
			t.Fatal(err)
		}
	}
	if restored.Status().Outcome != ChallengerWins {
		t.Fatalf("restored dispute produced %v", restored.Status().Outcome)
	}
}

func TestOneStepDisputeResolvesAtCreation(t *testing.T) {
	p, err := NewProgram([]Instruction{{Op: OpSet, Dst: 0, Imm: 7}})
	if err != nil {
		t.Fatal(err)
	}
	honest, err := Execute(p, nil)
	if err != nil {
		t.Fatal(err)
	}
	states := append([]State(nil), honest.states...)
	states[1].Reg[0] = 8
	forged := newExecution(states)
	d, err := NewDispute(p, nil, claim(t, forged), claim(t, honest), 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if d.Status().Outcome != ChallengerWins {
		t.Fatalf("one-step fraud not resolved: %v", d.Status().Outcome)
	}
}
