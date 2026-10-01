package compute

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	storetypes "cosmossdk.io/store/types"
	"github.com/cosmos/cosmos-sdk/testutil"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"prismachain/chain/x/compute/types"
	"prismachain/vm"
)

type memoryBank struct {
	accounts map[string]uint64
	module   uint64
	burned   uint64
}

func coinsUprsm(coins sdk.Coins) uint64 { return coins.AmountOf(Denom).BigInt().Uint64() }

func (b *memoryBank) SendCoinsFromAccountToModule(_ context.Context, account sdk.AccAddress, module string, coins sdk.Coins) error {
	if module != ModuleName {
		return errors.New("wrong module")
	}
	n := coinsUprsm(coins)
	if b.accounts[account.String()] < n {
		return errors.New("insufficient account funds")
	}
	b.accounts[account.String()] -= n
	b.module += n
	return nil
}

func (b *memoryBank) SendCoinsFromModuleToAccount(_ context.Context, module string, account sdk.AccAddress, coins sdk.Coins) error {
	if module != ModuleName {
		return errors.New("wrong module")
	}
	n := coinsUprsm(coins)
	if b.module < n {
		return errors.New("insufficient module funds")
	}
	b.module -= n
	b.accounts[account.String()] += n
	return nil
}

func (b *memoryBank) BurnCoins(_ context.Context, module string, coins sdk.Coins) error {
	if module != ModuleName {
		return errors.New("wrong module")
	}
	n := coinsUprsm(coins)
	if b.module < n {
		return errors.New("insufficient module funds")
	}
	b.module -= n
	b.burned += n
	return nil
}

func account(seed byte) string {
	return sdk.AccAddress([]byte{seed, seed, seed, seed, seed, seed, seed, seed, seed, seed,
		seed, seed, seed, seed, seed, seed, seed, seed, seed, seed}).String()
}

func TestVerifiableTaskEscrowChallengeWindowAndSingleSettlement(t *testing.T) {
	key := storetypes.NewKVStoreKey(ModuleName)
	ctx := testutil.DefaultContextWithKeys(map[string]*storetypes.KVStoreKey{ModuleName: key}, nil, nil).WithBlockHeight(1)
	requester, worker, monitorA, monitorB := account(1), account(2), account(3), account(4)
	bank := &memoryBank{accounts: map[string]uint64{requester: 1000, worker: MinBond, monitorA: MinBond, monitorB: MinBond}}
	keeper := NewKeeper(key, bank)
	msg := keeper.MsgServer()
	code := []vm.Instruction{{Op: vm.OpSet, Dst: 0, Imm: 7}}
	program, err := vm.NewProgram(code)
	if err != nil {
		t.Fatal(err)
	}
	programJSON, _ := json.Marshal(code)
	digest := make([]byte, 32)
	if _, err := msg.RegisterModel(ctx, &types.MsgRegisterModel{Owner: requester, ModelId: "demo", SpecVersion: "vm-v1",
		Mode: "verifiable", ImageDigest: digest, TokenizerDigest: digest, WeightsDigest: digest, Program: programJSON}); err != nil {
		t.Fatal(err)
	}
	inputDigest := vm.InputDigest(nil)
	posted, err := msg.PostTask(ctx, &types.MsgPostTask{Requester: requester, Mode: "verifiable", ModelId: "demo",
		SpecVersion: "vm-v1", InputCommitment: inputDigest[:], MaxFee: 1000, Deadline: 100, PrivacyTier: "public"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := msg.BondWorker(ctx, &types.MsgBondWorker{Worker: worker, Amount: MinBond}); err != nil {
		t.Fatal(err)
	}
	if _, err := msg.AcceptTask(ctx, &types.MsgAcceptTask{Worker: worker, TaskId: posted.TaskId}); err != nil {
		t.Fatal(err)
	}
	if keeper.getReserved(ctx, worker) != MinBond {
		t.Fatal("bond was not reserved")
	}
	if _, err := msg.AcceptTask(ctx, &types.MsgAcceptTask{Worker: worker, TaskId: posted.TaskId}); err == nil {
		t.Fatal("duplicate accept succeeded")
	}
	execution, err := vm.Execute(program, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, initialProof, _ := execution.Proof(0)
	final, finalProof, _ := execution.Proof(execution.Steps())
	claimJSON, _ := json.Marshal(vm.TraceClaim{Root: execution.Root(), InitialProof: initialProof, Final: final, FinalProof: finalProof})
	root := execution.Root()
	out := OutputDigest(execution.Output())
	ctx = ctx.WithBlockHeight(2)
	if _, err := msg.SubmitResult(ctx, &types.MsgSubmitResult{Worker: worker, TaskId: posted.TaskId,
		OutputDigest: out[:], TraceRoot: root[:], TraceClaimJson: claimJSON}); err != nil {
		t.Fatal(err)
	}
	for _, monitor := range []string{monitorA, monitorB} {
		if _, err := msg.BondWorker(ctx, &types.MsgBondWorker{Worker: monitor, Amount: MinBond}); err != nil {
			t.Fatal(err)
		}
		if _, err := msg.AttestResult(ctx, &types.MsgAttestResult{Monitor: monitor, TaskId: posted.TaskId}); err != nil {
			t.Fatal(err)
		}
	}
	ctx = ctx.WithBlockHeight(22)
	if _, err := msg.FinalizeTask(ctx, &types.MsgFinalizeTask{Actor: requester, TaskId: posted.TaskId}); err == nil {
		t.Fatal("finalized before challenge window closed")
	}
	ctx = ctx.WithBlockHeight(23)
	if _, err := msg.FinalizeTask(ctx, &types.MsgFinalizeTask{Actor: requester, TaskId: posted.TaskId}); err != nil {
		t.Fatal(err)
	}
	if _, err := msg.FinalizeTask(ctx, &types.MsgFinalizeTask{Actor: requester, TaskId: posted.TaskId}); err == nil {
		t.Fatal("paid twice")
	}
	if _, err := msg.RefundTask(ctx, &types.MsgRefundTask{Actor: requester, TaskId: posted.TaskId}); err == nil {
		t.Fatal("refunded after payment")
	}
	if bank.burned != 200 || bank.accounts[worker] != 700 || bank.accounts[monitorA] != 50 || bank.accounts[monitorB] != 50 || bank.module != 3*MinBond {
		t.Fatalf("incorrect settlement ledger: burn=%d worker=%d monitors=%d/%d module=%d", bank.burned, bank.accounts[worker], bank.accounts[monitorA], bank.accounts[monitorB], bank.module)
	}
	if keeper.getReserved(ctx, worker) != 0 {
		t.Fatal("worker bond reservation was not released")
	}
}

func TestBothInvalidDisputeNeverRewardsChallenger(t *testing.T) {
	key := storetypes.NewKVStoreKey(ModuleName)
	ctx := testutil.DefaultContextWithKeys(map[string]*storetypes.KVStoreKey{ModuleName: key}, nil, nil).WithBlockHeight(50)
	requester, worker, challenger := account(5), account(6), account(7)
	bank := &memoryBank{accounts: map[string]uint64{requester: 0, worker: 0, challenger: 0},
		module: MinBond + 1000 + 10}
	keeper := NewKeeper(key, bank)
	keeper.setBond(ctx, worker, MinBond)
	keeper.setReserved(ctx, worker, MinBond)
	task := Task{Requester: requester, Worker: worker, Challenger: challenger,
		MaxFee: 1000, ReservedBond: MinBond, ChallengeBond: 10, Status: "challenged"}
	if err := (msgServer{keeper}).resolveChallenge(ctx, &task, vm.BothInvalid); err != nil {
		t.Fatal(err)
	}
	if task.Status != "refunded" || bank.accounts[requester] != 1000 || bank.accounts[challenger] != 0 ||
		bank.burned != 100010 || bank.module != 900000 || keeper.GetBond(ctx, worker) != 900000 || keeper.getReserved(ctx, worker) != 0 {
		t.Fatalf("both-invalid accounting violated: task=%+v bank=%+v", task, bank)
	}
}

func TestLightweightRefundDoesNotLockEscrow(t *testing.T) {
	task := Task{Mode: "lightweight", Status: "pending", ChallengeEnd: 100,
		Attesters: []string{"monitor-a", "monitor-b"}}
	if refundable(task, 120) {
		t.Fatal("refund before timeout")
	}
	if !refundable(task, 121) {
		t.Fatal("lightweight escrow must remain refundable even with two attestations")
	}
	if payable(task, 121) {
		t.Fatal("lightweight task paid without verified receipt and usage tariff")
	}
	task.Status = "refunded"
	if refundable(task, 122) || payable(task, 122) {
		t.Fatal("refunded task could pay or refund twice")
	}
}

func TestVerifiableFinalityAndFeeConservation(t *testing.T) {
	task := Task{Mode: "verifiable", Status: "pending", ChallengeEnd: 100,
		Attesters: []string{"monitor-a", "monitor-b"}}
	if !payable(task, 101) || refundable(task, 121) {
		t.Fatal("attested task must finalize only once")
	}
	task.Status = "settled"
	if payable(task, 102) || refundable(task, 122) {
		t.Fatal("settled task could pay or refund twice")
	}
	for _, fee := range []uint64{1, 19, 20, 21, 1_000_001} {
		burn, monitor, worker := feeSplit(fee)
		if burn+2*monitor+worker != fee {
			t.Fatalf("fee %d not conserved", fee)
		}
		if fee == 1_000_001 && (burn != 200_000 || monitor != 50_000 || worker != 700_001) {
			t.Fatal("expected 20/10/70 split with rounding dust to worker")
		}
	}
}
