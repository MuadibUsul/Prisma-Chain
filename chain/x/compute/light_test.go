package compute

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"

	storetypes "cosmossdk.io/store/types"
	"github.com/cosmos/cosmos-sdk/testutil"
	"prismachain/chain/x/compute/types"
)

func testNetworkKeyProof(t *testing.T, chainID, account string, key []byte, seed byte) []byte {
	t.Helper()
	private := ed25519.NewKeyFromSeed(bytes.Repeat([]byte{seed}, ed25519.SeedSize))
	if !bytes.Equal(private.Public().(ed25519.PublicKey), key) {
		t.Fatal("test network key does not match deterministic receipt seed")
	}
	payload, err := canonicalJSON(map[string]any{
		"chain_id": chainID, "worker": account, "network_public_key": hex.EncodeToString(key),
	})
	if err != nil {
		t.Fatal(err)
	}
	return ed25519.Sign(private, append([]byte("prisma:network-key-binding:v1\n"), payload...))
}

func TestBondWorkerRequiresNetworkKeyPossession(t *testing.T) {
	worker, attacker := account(91), account(92)
	key := storetypes.NewKVStoreKey(ModuleName)
	ctx := testutil.DefaultContextWithKeys(map[string]*storetypes.KVStoreKey{ModuleName: key}, nil, nil).
		WithBlockHeight(1).WithChainID("prisma-test-1")
	bank := &memoryBank{accounts: map[string]uint64{worker: MinBond + 2, attacker: MinBond}}
	keeper := NewKeeper(key, bank)
	msg := keeper.MsgServer()
	publicKey := ed25519.NewKeyFromSeed(bytes.Repeat([]byte{3}, ed25519.SeedSize)).Public().(ed25519.PublicKey)
	for _, invalid := range [][]byte{
		nil,
		testNetworkKeyProof(t, ctx.ChainID(), attacker, publicKey, 3),
		testNetworkKeyProof(t, "another-chain", worker, publicKey, 3),
	} {
		if _, err := msg.BondWorker(ctx, &types.MsgBondWorker{Worker: worker, Amount: MinBond,
			NetworkPublicKey: publicKey, NetworkKeyProof: invalid}); err == nil {
			t.Fatal("network key registered without an account-bound proof")
		}
	}
	if keeper.GetBond(ctx, worker) != 0 || keeper.store(ctx).Has(networkKey(worker)) {
		t.Fatal("rejected key registration changed bond or network identity")
	}
	proof := testNetworkKeyProof(t, ctx.ChainID(), worker, publicKey, 3)
	if _, err := msg.BondWorker(ctx, &types.MsgBondWorker{Worker: worker, Amount: MinBond,
		NetworkPublicKey: publicKey, NetworkKeyProof: proof}); err != nil {
		t.Fatal(err)
	}
	if len(keeper.store(ctx).Get(networkProofKey(worker))) != 1 {
		t.Fatal("valid network key was not marked proven")
	}
	if _, err := msg.BondWorker(ctx, &types.MsgBondWorker{Worker: worker, Amount: 1}); err != nil {
		t.Fatal("bond top-up without re-sending the network key should work:", err)
	}
	keeper.store(ctx).Delete(networkProofKey(worker)) // Simulate a pre-upgrade registration.
	queried, err := keeper.QueryServer().Worker(ctx, &types.QueryWorkerRequest{Worker: worker})
	if err != nil || len(queried.NetworkPublicKey) != 0 {
		t.Fatal("unproven legacy network key remained visible to authorizers")
	}
	if _, err := msg.BondWorker(ctx, &types.MsgBondWorker{Worker: worker, Amount: 1,
		NetworkPublicKey: publicKey, NetworkKeyProof: proof}); err != nil {
		t.Fatal("same-key proof refresh failed:", err)
	}
	queried, err = keeper.QueryServer().Worker(ctx, &types.QueryWorkerRequest{Worker: worker})
	if err != nil || !bytes.Equal(queried.NetworkPublicKey, publicKey) {
		t.Fatal("proved legacy key was not restored to authorizer queries")
	}
}

func TestLightweightSignedReceiptMeteredSettlement(t *testing.T) {
	fixtureBytes, err := os.ReadFile("../../../network/examples/contract.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Announcement struct {
			Capability struct {
				PublicKey string `json:"public_key"`
			} `json:"capability"`
		} `json:"announcement"`
		DeliveryReceipt  map[string]any `json:"delivery_receipt"`
		GatewayPublicKey string         `json:"gateway_public_key"`
	}
	decoder := json.NewDecoder(bytes.NewReader(fixtureBytes))
	decoder.UseNumber()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatal(err)
	}
	receiptJSON, err := canonicalJSON(fixture.DeliveryReceipt)
	if err != nil {
		t.Fatal(err)
	}
	var receipt lightReceipt
	if err := json.Unmarshal(receiptJSON, &receipt); err != nil {
		t.Fatal(err)
	}
	workerKey, _ := base64.StdEncoding.DecodeString(fixture.Announcement.Capability.PublicKey)
	gatewayKey, _ := base64.StdEncoding.DecodeString(fixture.GatewayPublicKey)
	requester, worker, gateway, monitorA, monitorB := account(71), account(72), account(73), account(74), account(75)
	key := storetypes.NewKVStoreKey(ModuleName)
	ctx := testutil.DefaultContextWithKeys(map[string]*storetypes.KVStoreKey{ModuleName: key}, nil, nil).
		WithBlockHeight(1).WithChainID("prisma-test-1")
	bank := &memoryBank{accounts: map[string]uint64{
		requester: 10_000, worker: MinBond, gateway: MinBond,
		monitorA: MinBond, monitorB: MinBond,
	}}
	keeper := NewKeeper(key, bank)
	msg := keeper.MsgServer()
	weights, _ := hex.DecodeString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	tokenizer, _ := hex.DecodeString("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
	image, _ := hex.DecodeString("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
	if _, err := msg.RegisterModel(ctx, &types.MsgRegisterModel{Owner: requester, ModelId: receipt.ModelID,
		SpecVersion: receipt.SpecVersion, Mode: "lightweight", ImageDigest: image,
		TokenizerDigest: tokenizer, WeightsDigest: weights}); err != nil {
		t.Fatal(err)
	}
	for _, bonded := range []struct {
		address string
		key     []byte
		seed    byte
	}{{worker, workerKey, 2}, {gateway, gatewayKey, 1}, {monitorA, nil, 0}, {monitorB, nil, 0}} {
		var proof []byte
		if bonded.key != nil {
			proof = testNetworkKeyProof(t, ctx.ChainID(), bonded.address, bonded.key, bonded.seed)
		}
		if _, err := msg.BondWorker(ctx, &types.MsgBondWorker{Worker: bonded.address,
			Amount: MinBond, NetworkPublicKey: bonded.key, NetworkKeyProof: proof}); err != nil {
			t.Fatal(err)
		}
	}
	inputDigest, _ := hex.DecodeString(receipt.InputCommitment)
	posted, err := msg.PostTask(ctx, &types.MsgPostTask{Requester: requester, Mode: "lightweight",
		ModelId: receipt.ModelID, SpecVersion: receipt.SpecVersion, InputCommitment: inputDigest,
		DataRef: "encrypted:example", MaxFee: 10_000, Deadline: 100, PrivacyTier: "tier0_relative"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := msg.AcceptTask(ctx, &types.MsgAcceptTask{Worker: worker, TaskId: posted.TaskId}); err != nil {
		t.Fatal(err)
	}
	outputDigest, _ := hex.DecodeString(receipt.OutputCommitment)
	hash := sha256.Sum256(receiptJSON)
	result := &types.MsgSubmitResult{Worker: worker, TaskId: posted.TaskId,
		OutputDigest: outputDigest, OutputTokens: receipt.OutputTokens,
		ReceiptDigest: hash[:], ReceiptJson: receiptJSON}
	ctx = ctx.WithBlockHeight(2)
	withoutBytes := *result
	withoutBytes.ReceiptJson = nil
	if _, err := msg.SubmitResult(ctx, &withoutBytes); err == nil {
		t.Fatal("receipt digest alone authorized lightweight result")
	}
	wrongOutput := *result
	wrongOutput.OutputDigest = bytes.Repeat([]byte{7}, 32)
	if _, err := msg.SubmitResult(ctx, &wrongOutput); err == nil {
		t.Fatal("mismatched output commitment was accepted")
	}
	badDigest := *result
	badDigest.ReceiptDigest = bytes.Repeat([]byte{7}, 32)
	if _, err := msg.SubmitResult(ctx, &badDigest); err == nil {
		t.Fatal("mismatched receipt digest was accepted")
	}
	badSignature := make(map[string]any, len(fixture.DeliveryReceipt))
	for key, value := range fixture.DeliveryReceipt {
		badSignature[key] = value
	}
	badSignature["worker_signature"] = "AAAA"
	badJSON, _ := canonicalJSON(badSignature)
	badHash := sha256.Sum256(badJSON)
	badSigned := *result
	badSigned.ReceiptJson, badSigned.ReceiptDigest = badJSON, badHash[:]
	if _, err := msg.SubmitResult(ctx, &badSigned); err == nil {
		t.Fatal("invalid worker signature was accepted")
	}
	nonCanonical := *result
	nonCanonical.ReceiptJson = append(append([]byte(nil), receiptJSON...), ' ')
	nonCanonicalHash := sha256.Sum256(nonCanonical.ReceiptJson)
	nonCanonical.ReceiptDigest = nonCanonicalHash[:]
	if _, err := msg.SubmitResult(ctx, &nonCanonical); err == nil {
		t.Fatal("noncanonical signed receipt was accepted")
	}
	if _, err := msg.SubmitResult(ctx, result); err != nil {
		t.Fatal(err)
	}
	task, _ := keeper.GetTask(ctx, posted.TaskId)
	if task.Status != "pending" || task.ChargedFee != 3_000 || task.Gateway != gateway ||
		!bytes.Equal(task.ReceiptJSON, receiptJSON) {
		t.Fatalf("signed metered result not recorded: %+v", task)
	}
	if _, err := msg.AttestResult(ctx, &types.MsgAttestResult{Monitor: gateway, TaskId: posted.TaskId}); err == nil {
		t.Fatal("receipt gateway also acted as independent monitor")
	}
	for _, monitor := range []string{monitorA, monitorB} {
		if _, err := msg.AttestResult(ctx, &types.MsgAttestResult{Monitor: monitor, TaskId: posted.TaskId}); err != nil {
			t.Fatal(err)
		}
	}
	ctx = ctx.WithBlockHeight(23)
	if _, err := msg.FinalizeTask(ctx, &types.MsgFinalizeTask{Actor: requester, TaskId: posted.TaskId}); err != nil {
		t.Fatal(err)
	}
	task, _ = keeper.GetTask(ctx, posted.TaskId)
	if task.Status != "settled" || task.ReservedBond != 0 || bank.burned != 600 ||
		bank.accounts[requester] != 7_000 || bank.accounts[worker] != 2_100 ||
		bank.accounts[monitorA] != 150 || bank.accounts[monitorB] != 150 || bank.module != 4*MinBond {
		t.Fatalf("metered settlement did not conserve escrow: task=%+v bank=%+v", task, bank)
	}
	if _, err := msg.FinalizeTask(ctx, &types.MsgFinalizeTask{Actor: requester, TaskId: posted.TaskId}); err == nil {
		t.Fatal("lightweight task paid twice")
	}
}
