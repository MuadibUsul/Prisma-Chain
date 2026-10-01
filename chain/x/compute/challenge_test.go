package compute

import (
	"encoding/json"
	"testing"

	storetypes "cosmossdk.io/store/types"
	"github.com/cosmos/cosmos-sdk/testutil"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"prismachain/chain/x/compute/types"
	"prismachain/vm"
)

type challengeFixture struct {
	ctx             sdk.Context
	keeper          Keeper
	bank            *memoryBank
	msg             types.MsgServer
	requester       string
	worker          string
	challenger      string
	taskID          uint64
	workerTrace     vm.Execution
	challengerTrace vm.Execution
}

func traceClaimJSON(t *testing.T, execution vm.Execution) []byte {
	t.Helper()
	_, initialProof, err := execution.Proof(0)
	if err != nil {
		t.Fatal(err)
	}
	final, finalProof, err := execution.Proof(execution.Steps())
	if err != nil {
		t.Fatal(err)
	}
	claim, err := json.Marshal(vm.TraceClaim{Root: execution.Root(), InitialProof: initialProof, Final: final, FinalProof: finalProof})
	if err != nil {
		t.Fatal(err)
	}
	return claim
}

func newChallengeFixture(t *testing.T, steps int, workerWrong bool) challengeFixture {
	t.Helper()
	key := storetypes.NewKVStoreKey(ModuleName)
	ctx := testutil.DefaultContextWithKeys(map[string]*storetypes.KVStoreKey{ModuleName: key}, nil, nil).WithBlockHeight(1)
	requester, worker, challenger := account(21), account(22), account(23)
	bank := &memoryBank{accounts: map[string]uint64{requester: 1000, worker: MinBond, challenger: 1000}}
	keeper := NewKeeper(key, bank)
	msg := keeper.MsgServer()
	code, dishonestCode := make([]vm.Instruction, steps), make([]vm.Instruction, steps)
	for i := 0; i < steps; i++ {
		code[i] = vm.Instruction{Op: vm.OpSet, Dst: 0, Imm: int64(7 + i)}
		dishonestCode[i] = vm.Instruction{Op: vm.OpSet, Dst: 0, Imm: int64(20 + i)}
	}
	program, err := vm.NewProgram(code)
	if err != nil {
		t.Fatal(err)
	}
	dishonestProgram, err := vm.NewProgram(dishonestCode)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := vm.Execute(program, nil)
	if err != nil {
		t.Fatal(err)
	}
	dishonest, err := vm.Execute(dishonestProgram, nil)
	if err != nil {
		t.Fatal(err)
	}
	programJSON, err := json.Marshal(code)
	if err != nil {
		t.Fatal(err)
	}
	digest := make([]byte, 32)
	if _, err := msg.RegisterModel(ctx, &types.MsgRegisterModel{Owner: requester, ModelId: "challenge-vm", SpecVersion: "v1",
		Mode: "verifiable", ImageDigest: digest, TokenizerDigest: digest, WeightsDigest: digest, Program: programJSON}); err != nil {
		t.Fatal(err)
	}
	inputDigest := vm.InputDigest(nil)
	posted, err := msg.PostTask(ctx, &types.MsgPostTask{Requester: requester, Mode: "verifiable", ModelId: "challenge-vm",
		SpecVersion: "v1", InputCommitment: inputDigest[:], DataRef: "unavailable://outside-data", MaxFee: 1000,
		Deadline: 100, PrivacyTier: "public"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := msg.BondWorker(ctx, &types.MsgBondWorker{Worker: worker, Amount: MinBond}); err != nil {
		t.Fatal(err)
	}
	if _, err := msg.AcceptTask(ctx, &types.MsgAcceptTask{Worker: worker, TaskId: posted.TaskId}); err != nil {
		t.Fatal(err)
	}
	workerTrace, challengerTrace := canonical, dishonest
	if workerWrong {
		workerTrace, challengerTrace = dishonest, canonical
	}
	root := workerTrace.Root()
	output := OutputDigest(workerTrace.Output())
	ctx = ctx.WithBlockHeight(2)
	if _, err := msg.SubmitResult(ctx, &types.MsgSubmitResult{Worker: worker, TaskId: posted.TaskId,
		OutputDigest: output[:], TraceRoot: root[:], TraceClaimJson: traceClaimJSON(t, workerTrace)}); err != nil {
		t.Fatal(err)
	}
	ctx = ctx.WithBlockHeight(3)
	if _, err := msg.StartChallenge(ctx, &types.MsgStartChallenge{Challenger: challenger, TaskId: posted.TaskId,
		TraceClaimJson: traceClaimJSON(t, challengerTrace)}); err != nil {
		t.Fatal(err)
	}
	return challengeFixture{ctx, keeper, bank, msg, requester, worker, challenger, posted.TaskId, workerTrace, challengerTrace}
}

func (f challengeFixture) task(t *testing.T) Task {
	t.Helper()
	task, err := f.keeper.GetTask(f.ctx, f.taskID)
	if err != nil {
		t.Fatal(err)
	}
	return task
}

func (f challengeFixture) assertConserved(t *testing.T) {
	t.Helper()
	total := f.bank.module + f.bank.burned + f.bank.accounts[f.requester] + f.bank.accounts[f.worker] + f.bank.accounts[f.challenger]
	if total != MinBond+2000 {
		t.Fatalf("ledger lost or created PRSM: %d", total)
	}
}

func TestWrongResultChallengeRefundsAndSlashes(t *testing.T) {
	f := newChallengeFixture(t, 1, true)
	task := f.task(t)
	if task.Status != "refunded" || task.ReservedBond != 0 || f.keeper.getReserved(f.ctx, f.worker) != 0 ||
		f.keeper.GetBond(f.ctx, f.worker) != 900000 || f.bank.accounts[f.requester] != 1000 ||
		f.bank.accounts[f.challenger] != 101000 || f.bank.module != 900000 || f.bank.burned != 0 {
		t.Fatalf("wrong-result challenge accounting: task=%+v bank=%+v", task, f.bank)
	}
	if _, err := f.msg.FinalizeTask(f.ctx, &types.MsgFinalizeTask{Actor: f.requester, TaskId: f.taskID}); err == nil {
		t.Fatal("fraudulent result could still be paid")
	}
	f.assertConserved(t)
}

func TestFalseChallengeLosesBondAndLeavesTaskPending(t *testing.T) {
	f := newChallengeFixture(t, 1, false)
	task := f.task(t)
	if task.Status != "pending" || task.ChallengeEnd != 3+ChallengeBlocks || task.ReservedBond != MinBond ||
		f.keeper.GetBond(f.ctx, f.worker) != MinBond || f.bank.accounts[f.worker] != 0 ||
		f.bank.accounts[f.challenger] != 990 || f.bank.module != MinBond+1000 || f.bank.burned != 10 {
		t.Fatalf("false-challenge accounting: task=%+v bank=%+v", task, f.bank)
	}
	// A colluding challenger cannot reclaim the bond through the worker and
	// cannot shorten the next honest challenger's full review window.
	if _, err := f.msg.StartChallenge(f.ctx.WithBlockHeight(4), &types.MsgStartChallenge{
		Challenger: f.challenger, TaskId: f.taskID, TraceClaimJson: traceClaimJSON(t, f.challengerTrace),
	}); err != nil {
		t.Fatal(err)
	}
	if f.bank.burned != 20 || f.bank.accounts[f.worker] != 0 {
		t.Fatalf("repeated false challenge was profitable: bank=%+v", f.bank)
	}
	f.assertConserved(t)
}

func TestQueuedHonestChallengeSurvivesFalseFrontRun(t *testing.T) {
	key := storetypes.NewKVStoreKey(ModuleName)
	ctx := testutil.DefaultContextWithKeys(map[string]*storetypes.KVStoreKey{ModuleName: key}, nil, nil).WithBlockHeight(1)
	requester, worker, front, honest, spare := account(41), account(42), account(43), account(44), account(45)
	bank := &memoryBank{accounts: map[string]uint64{
		requester: 1000, worker: MinBond, front: 1000, honest: 1000, spare: 1000,
	}}
	keeper := NewKeeper(key, bank)
	msg := keeper.MsgServer()
	code := []vm.Instruction{{Op: vm.OpSet, Dst: 1, Imm: 42}, {Op: vm.OpAdd, Dst: 0, A: 1, B: 2}}
	wrong := []vm.Instruction{{Op: vm.OpSet, Dst: 1, Imm: 42}, {Op: vm.OpSet, Dst: 0, Imm: 43}}
	frontCode := []vm.Instruction{{Op: vm.OpSet, Dst: 1, Imm: 99}, {Op: vm.OpSet, Dst: 0, Imm: 43}}
	program, _ := vm.NewProgram(code)
	wrongProgram, _ := vm.NewProgram(wrong)
	frontProgram, _ := vm.NewProgram(frontCode)
	canonical, _ := vm.Execute(program, nil)
	workerTrace, _ := vm.Execute(wrongProgram, nil)
	frontTrace, _ := vm.Execute(frontProgram, nil)
	programJSON, _ := json.Marshal(code)
	digest := make([]byte, 32)
	if _, err := msg.RegisterModel(ctx, &types.MsgRegisterModel{Owner: requester, ModelId: "race-vm", SpecVersion: "v1",
		Mode: "verifiable", ImageDigest: digest, TokenizerDigest: digest, WeightsDigest: digest, Program: programJSON}); err != nil {
		t.Fatal(err)
	}
	inputDigest := vm.InputDigest(nil)
	posted, err := msg.PostTask(ctx, &types.MsgPostTask{Requester: requester, Mode: "verifiable", ModelId: "race-vm",
		SpecVersion: "v1", InputCommitment: inputDigest[:], MaxFee: 1000, Deadline: 100, PrivacyTier: "public"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := msg.BondWorker(ctx, &types.MsgBondWorker{Worker: worker, Amount: MinBond}); err != nil {
		t.Fatal(err)
	}
	if _, err := msg.AcceptTask(ctx, &types.MsgAcceptTask{Worker: worker, TaskId: posted.TaskId}); err != nil {
		t.Fatal(err)
	}
	root, output := workerTrace.Root(), OutputDigest(workerTrace.Output())
	ctx = ctx.WithBlockHeight(2)
	if _, err := msg.SubmitResult(ctx, &types.MsgSubmitResult{Worker: worker, TaskId: posted.TaskId,
		OutputDigest: output[:], TraceRoot: root[:], TraceClaimJson: traceClaimJSON(t, workerTrace)}); err != nil {
		t.Fatal(err)
	}
	ctx = ctx.WithBlockHeight(3)
	if _, err := msg.StartChallenge(ctx, &types.MsgStartChallenge{Challenger: front, TaskId: posted.TaskId,
		TraceClaimJson: traceClaimJSON(t, frontTrace)}); err != nil {
		t.Fatal(err)
	}
	ctx = ctx.WithBlockHeight(4)
	for _, who := range []string{honest, spare} {
		if _, err := msg.StartChallenge(ctx, &types.MsgStartChallenge{Challenger: who, TaskId: posted.TaskId,
			TraceClaimJson: traceClaimJSON(t, canonical)}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := msg.StartChallenge(ctx, &types.MsgStartChallenge{Challenger: honest, TaskId: posted.TaskId,
		TraceClaimJson: traceClaimJSON(t, canonical)}); err == nil {
		t.Fatal("duplicate queued challenger was accepted")
	}
	for _, turn := range []struct {
		height int64
		who    string
		trace  vm.Execution
	}{
		{5, worker, workerTrace}, {5, front, frontTrace},
		{6, worker, workerTrace}, {6, honest, canonical},
	} {
		state, proof, err := turn.trace.Proof(1)
		if err != nil {
			t.Fatal(err)
		}
		stateJSON, _ := json.Marshal(state)
		proofJSON, _ := json.Marshal(proof)
		ctx = ctx.WithBlockHeight(turn.height)
		if _, err := msg.ChallengeMidpoint(ctx, &types.MsgChallengeMidpoint{Actor: turn.who, TaskId: posted.TaskId,
			StateJson: stateJSON, ProofJson: proofJSON}); err != nil {
			t.Fatal(err)
		}
		if turn.who == front {
			task, _ := keeper.GetTask(ctx, posted.TaskId)
			if task.Status != "challenged" || task.Challenger != honest || len(task.QueuedChallenges) != 1 ||
				task.ChallengeEnd != uint64(turn.height)+ChallengeBlocks || bank.burned != 10 {
				t.Fatalf("honest challenge was not promoted after false front run: %+v bank=%+v", task, bank)
			}
		}
	}
	task, _ := keeper.GetTask(ctx, posted.TaskId)
	if task.Status != "refunded" || len(task.QueuedChallenges) != 0 || task.ReservedBond != 0 ||
		bank.accounts[requester] != 1000 || bank.accounts[front] != 990 ||
		bank.accounts[honest] != 101000 || bank.accounts[spare] != 1000 ||
		bank.module != 900000 || bank.burned != 10 || keeper.GetBond(ctx, worker) != 900000 {
		t.Fatalf("queued challenge accounting: task=%+v bank=%+v", task, bank)
	}
	if _, err := msg.FinalizeTask(ctx, &types.MsgFinalizeTask{Actor: requester, TaskId: posted.TaskId}); err == nil {
		t.Fatal("fraudulent result could be paid after queued challenge")
	}
	if total := bank.module + bank.burned + bank.accounts[requester] + bank.accounts[worker] +
		bank.accounts[front] + bank.accounts[honest] + bank.accounts[spare]; total != MinBond+4000 {
		t.Fatalf("queued challenges did not conserve PRSM: %d", total)
	}
}

func TestChallengeUsesPublicOnChainDataWhenDataRefIsUnavailable(t *testing.T) {
	key := storetypes.NewKVStoreKey(ModuleName)
	ctx := testutil.DefaultContextWithKeys(map[string]*storetypes.KVStoreKey{ModuleName: key}, nil, nil).WithBlockHeight(1)
	requester, worker, challenger := account(31), account(32), account(33)
	bank := &memoryBank{accounts: map[string]uint64{requester: 1000, worker: MinBond, challenger: 1000}}
	keeper := NewKeeper(key, bank)
	msg := keeper.MsgServer()
	program, err := vm.NewProgram([]vm.Instruction{{Op: vm.OpInput, Dst: 0, Imm: 0}})
	if err != nil {
		t.Fatal(err)
	}
	programJSON, _ := json.Marshal([]vm.Instruction{{Op: vm.OpInput, Dst: 0, Imm: 0}})
	digest := make([]byte, 32)
	if _, err := msg.RegisterModel(ctx, &types.MsgRegisterModel{Owner: requester, ModelId: "public-input-vm",
		SpecVersion: "v1", Mode: "verifiable", ImageDigest: digest, TokenizerDigest: digest,
		WeightsDigest: digest, Program: programJSON}); err != nil {
		t.Fatal(err)
	}
	publicInput := []int64{7}
	inputDigest := vm.InputDigest(publicInput)
	wrongDigest := vm.InputDigest([]int64{9})
	taskMsg := &types.MsgPostTask{Requester: requester, Mode: "verifiable", ModelId: "public-input-vm",
		SpecVersion: "v1", InputCommitment: wrongDigest[:], PublicInput: publicInput,
		DataRef: "unavailable://outside-data", MaxFee: 1000, Deadline: 100, PrivacyTier: "public"}
	if _, err := msg.PostTask(ctx, taskMsg); err == nil {
		t.Fatal("mismatched public input commitment was accepted")
	}
	taskMsg.InputCommitment = inputDigest[:]
	posted, err := msg.PostTask(ctx, taskMsg)
	if err != nil {
		t.Fatal(err)
	}
	storedModel, err := keeper.GetModel(ctx, "public-input-vm", "v1")
	if err != nil || string(storedModel.Program) != string(programJSON) {
		t.Fatal("challenge program not available from chain")
	}
	storedTask, err := keeper.GetTask(ctx, posted.TaskId)
	if err != nil || len(storedTask.PublicInput) != 1 || storedTask.PublicInput[0] != 7 {
		t.Fatal("public input not available from chain")
	}
	if _, err := msg.BondWorker(ctx, &types.MsgBondWorker{Worker: worker, Amount: MinBond}); err != nil {
		t.Fatal(err)
	}
	if _, err := msg.AcceptTask(ctx, &types.MsgAcceptTask{Worker: worker, TaskId: posted.TaskId}); err != nil {
		t.Fatal(err)
	}
	falseTrace, err := vm.Execute(program, []int64{9})
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := vm.Execute(program, publicInput)
	if err != nil {
		t.Fatal(err)
	}
	root := falseTrace.Root()
	output := OutputDigest(falseTrace.Output())
	ctx = ctx.WithBlockHeight(2)
	if _, err := msg.SubmitResult(ctx, &types.MsgSubmitResult{Worker: worker, TaskId: posted.TaskId,
		OutputDigest: output[:], TraceRoot: root[:], TraceClaimJson: traceClaimJSON(t, falseTrace)}); err != nil {
		t.Fatal(err)
	}
	ctx = ctx.WithBlockHeight(3)
	if _, err := msg.StartChallenge(ctx, &types.MsgStartChallenge{Challenger: challenger, TaskId: posted.TaskId,
		TraceClaimJson: traceClaimJSON(t, canonical)}); err != nil {
		t.Fatal(err)
	}
	resolved, err := keeper.GetTask(ctx, posted.TaskId)
	if err != nil || resolved.Status != "refunded" || bank.accounts[requester] != 1000 {
		t.Fatalf("challenge could not use on-chain data: task=%+v bank=%+v err=%v", resolved, bank, err)
	}
}

func TestChallengeResponseTimeoutLedger(t *testing.T) {
	for _, tc := range []struct {
		name, responder, status                             string
		requester, worker, challenger, module, burned, bond uint64
	}{
		{"neither responds", "", "refunded", 1000, 0, 990, 900000, 100010, 900000},
		{"only worker responds", "worker", "pending", 0, 0, 990, MinBond + 1000, 10, MinBond},
		{"only challenger responds", "challenger", "refunded", 1000, 0, 101000, 900000, 0, 900000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newChallengeFixture(t, 2, true)
			if f.task(t).Status != "challenged" {
				t.Fatal("two-step dispute resolved before midpoint")
			}
			if tc.responder != "" {
				actor, trace := f.worker, f.workerTrace
				if tc.responder == "challenger" {
					actor, trace = f.challenger, f.challengerTrace
				}
				state, proof, err := trace.Proof(1)
				if err != nil {
					t.Fatal(err)
				}
				stateJSON, _ := json.Marshal(state)
				proofJSON, _ := json.Marshal(proof)
				f.ctx = f.ctx.WithBlockHeight(4)
				if _, err := f.msg.ChallengeMidpoint(f.ctx, &types.MsgChallengeMidpoint{Actor: actor, TaskId: f.taskID,
					StateJson: stateJSON, ProofJson: proofJSON}); err != nil {
					t.Fatal(err)
				}
			}
			f.ctx = f.ctx.WithBlockHeight(8)
			if _, err := f.msg.TimeoutChallenge(f.ctx, &types.MsgTimeoutChallenge{Actor: f.requester, TaskId: f.taskID}); err == nil {
				t.Fatal("challenge timed out at, rather than after, the deadline")
			}
			f.ctx = f.ctx.WithBlockHeight(9)
			if _, err := f.msg.TimeoutChallenge(f.ctx, &types.MsgTimeoutChallenge{Actor: f.requester, TaskId: f.taskID}); err != nil {
				t.Fatal(err)
			}
			task := f.task(t)
			if task.Status != tc.status || f.bank.accounts[f.requester] != tc.requester ||
				f.bank.accounts[f.worker] != tc.worker || f.bank.accounts[f.challenger] != tc.challenger ||
				f.bank.module != tc.module || f.bank.burned != tc.burned || f.keeper.GetBond(f.ctx, f.worker) != tc.bond {
				t.Fatalf("timeout accounting: task=%+v bank=%+v", task, f.bank)
			}
			f.assertConserved(t)
		})
	}
}
