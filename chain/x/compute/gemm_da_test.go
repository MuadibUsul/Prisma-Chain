package compute

// DA_REPLICA_V1 adversarial tests: registry, attestation validation,
// quorum gating, sampling challenges with objective timeout, DoS bounds
// and the availability-failure refund path.

import (
	"encoding/json"
	"testing"

	"prismachain/chain/x/compute/types"
	"prismachain/compute/gemmv1"
)

func daHarness(t *testing.T, seed uint32) (*gemmHarness, uint64) {
	t.Helper()
	h := newGemmHarness(t, 1)
	const m, n, k = 16, 16, 16
	taskID, a, b, c := h.postGEMM(m, n, k, seed)
	h.acceptAndSubmit(taskID, a, b, c, m, n, k, false)
	return h, taskID
}

func (h *gemmHarness) rawAttestation(taskID uint64, provider string, keys *gemmKeys, mutate func(*gemmv1.DAAttestation)) []byte {
	h.t.Helper()
	task, err := h.keeper.GetGEMMTask(h.ctx, taskID)
	if err != nil {
		h.t.Fatal(err)
	}
	attestation := gemmv1.DAAttestation{
		ProtocolVersion: gemmv1.DAProtocolVersion,
		TaskID:          task.ProtocolTaskID, AssignmentID: task.AssignmentID,
		OutputRoot: task.OutputRoot, ProviderAccount: []byte(provider),
		ProviderPubKey: keys.networkPub, OutputBytes: task.M * task.N * 4,
		AvailableUntilHeight: daAttestationUntil(task),
		AttestedHeight:       uint64(h.ctx.BlockHeight()),
	}
	if mutate != nil {
		mutate(&attestation)
	}
	if err := gemmv1.SignDAAttestation(&attestation, keys.networkPriv); err != nil {
		h.t.Fatal(err)
	}
	raw, err := json.Marshal(&attestation)
	if err != nil {
		h.t.Fatal(err)
	}
	return raw
}

func (h *gemmHarness) submitAttestation(taskID uint64, provider string, raw []byte) error {
	_, err := h.msg.SubmitDAAttestation(h.ctx, &types.MsgSubmitDAAttestation{
		Provider: provider, GemmTaskId: taskID, AttestationJson: raw,
	})
	return err
}

// TestDARejectsWrongBlobAndIdentities covers DoD E2 items 1-3, 15-17.
func TestDARejectsWrongBlobAndIdentities(t *testing.T) {
	h, taskID := daHarness(t, 91)
	h.registerDAProvider(h.provA, h.provAKeys)

	// Wrong output_root blob claim is rejected.
	raw := h.rawAttestation(taskID, h.provA, h.provAKeys, func(a *gemmv1.DAAttestation) {
		a.OutputRoot = append([]byte(nil), a.OutputRoot...)
		a.OutputRoot[0] ^= 0xff
	})
	if err := h.submitAttestation(taskID, h.provA, raw); err == nil {
		t.Fatal("wrong output_root attestation accepted")
	}
	// Wrong byte count is rejected.
	raw = h.rawAttestation(taskID, h.provA, h.provAKeys, func(a *gemmv1.DAAttestation) {
		a.OutputBytes++
	})
	if err := h.submitAttestation(taskID, h.provA, raw); err == nil {
		t.Fatal("wrong output_bytes attestation accepted")
	}
	// A signature by a key that is not the provider's bonded key is rejected.
	impostor := newGemmKeys(99)
	raw = h.rawAttestation(taskID, h.provA, impostor, nil)
	if err := h.submitAttestation(taskID, h.provA, raw); err == nil {
		t.Fatal("unbonded key attestation accepted")
	}
	// The worker cannot be its own DA provider.
	h.registerDAProvider(h.worker, h.workKeys)
	raw = h.rawAttestation(taskID, h.worker, h.workKeys, nil)
	if err := h.submitAttestation(taskID, h.worker, raw); err == nil {
		t.Fatal("worker self-attestation accepted")
	}
	// A valid attestation is accepted.
	raw = h.rawAttestation(taskID, h.provA, h.provAKeys, nil)
	if err := h.submitAttestation(taskID, h.provA, raw); err != nil {
		t.Fatal(err)
	}
	// A duplicate attestation cannot reduce available_until.
	raw = h.rawAttestation(taskID, h.provA, h.provAKeys, func(a *gemmv1.DAAttestation) {
		a.AvailableUntilHeight--
	})
	if err := h.submitAttestation(taskID, h.provA, raw); err == nil {
		t.Fatal("retrograde available_until accepted")
	}
}

// TestDAQuorumGateAndFailureRefund covers DoD E2 items 4-5, 12-14.
func TestDAQuorumGateAndFailureRefund(t *testing.T) {
	// Scenario 1: quorum gating and success.
	h, taskID := daHarness(t, 92)
	h.registerDAProvider(h.provA, h.provAKeys)
	h.registerDAProvider(h.provB, h.provBKeys)
	for i, monitor := range []string{h.monitorA, h.monitorB} {
		h.bondWorker(monitor, newGemmKeys(byte(70+i)))
		if _, err := h.msg.AttestGEMMTask(h.ctx, &types.MsgAttestGEMMTask{Monitor: monitor, GemmTaskId: taskID}); err != nil {
			t.Fatal(err)
		}
	}
	task, _ := h.keeper.GetGEMMTask(h.ctx, taskID)
	if err := h.submitAttestation(taskID, h.provA, h.rawAttestation(taskID, h.provA, h.provAKeys, nil)); err != nil {
		t.Fatal(err)
	}
	h.advanceHeight(int64(task.ChallengeEnd) + 1)
	if _, err := h.msg.FinalizeGEMM(h.ctx, &types.MsgFinalizeGEMM{Actor: h.requester, GemmTaskId: taskID}); err == nil {
		t.Fatal("finalize accepted with 1/2 DA attestations")
	}
	if err := h.submitAttestation(taskID, h.provB, h.rawAttestation(taskID, h.provB, h.provBKeys, nil)); err != nil {
		t.Fatal(err)
	}
	if _, err := h.msg.FinalizeGEMM(h.ctx, &types.MsgFinalizeGEMM{Actor: h.requester, GemmTaskId: taskID}); err != nil {
		t.Fatalf("finalize with quorum failed: %v", err)
	}

	// Scenario 2: quorum never restored -> objective failure + refund.
	h2, taskID2 := daHarness(t, 93)
	h2.registerDAProvider(h2.provA, h2.provAKeys)
	h2.registerDAProvider(h2.provB, h2.provBKeys)
	if err := h2.submitAttestation(taskID2, h2.provA, h2.rawAttestation(taskID2, h2.provA, h2.provAKeys, nil)); err != nil {
		t.Fatal(err)
	}
	task2, _ := h2.keeper.GetGEMMTask(h2.ctx, taskID2)
	h2.advanceHeight(int64(task2.ResultSubmittedHeight+task2.ChallengeWindow+DAWindowBlocks) + 1)
	if _, err := h2.msg.FailGEMMAvailability(h2.ctx, &types.MsgFailGEMMAvailability{Actor: h2.requester, GemmTaskId: taskID2}); err != nil {
		t.Fatalf("availability failure refund rejected: %v", err)
	}
	refunded, _ := h2.keeper.GetGEMMTask(h2.ctx, taskID2)
	if refunded.Status != GEMMStatusRefunded || refunded.ReceiptID != nil || refunded.ReservedBond != 0 {
		t.Fatalf("availability failure did not refund cleanly: %+v", refunded)
	}

	// Scenario 3: an independent replacement provider restores the quorum.
	h3, taskID3 := daHarness(t, 94)
	h3.registerDAProvider(h3.provA, h3.provAKeys)
	h3.registerDAProvider(h3.provC, h3.provCKeys)
	if err := h3.submitAttestation(taskID3, h3.provA, h3.rawAttestation(taskID3, h3.provA, h3.provAKeys, nil)); err != nil {
		t.Fatal(err)
	}
	if err := h3.submitAttestation(taskID3, h3.provC, h3.rawAttestation(taskID3, h3.provC, h3.provCKeys, nil)); err != nil {
		t.Fatal(err)
	}
	status, valid, _ := msgServer{h3.keeper}.gemmDAStatus(h3.ctx, mustTask(t, h3, taskID3))
	if status != DAStatusReady || valid != 2 {
		t.Fatalf("replacement quorum not ready: %s %d", status, valid)
	}
}

// TestDAChallengeResponseAndTimeout covers DoD E2 items 9-11, 18.
func TestDAChallengeResponseAndTimeout(t *testing.T) {
	h, taskID := daHarness(t, 95)
	h.registerDAProvider(h.provA, h.provAKeys)
	h.registerDAProvider(h.provB, h.provBKeys)
	if err := h.submitAttestation(taskID, h.provA, h.rawAttestation(taskID, h.provA, h.provAKeys, nil)); err != nil {
		t.Fatal(err)
	}
	if err := h.submitAttestation(taskID, h.provB, h.rawAttestation(taskID, h.provB, h.provBKeys, nil)); err != nil {
		t.Fatal(err)
	}
	task, _ := h.keeper.GetGEMMTask(h.ctx, taskID)
	open, err := h.msg.OpenGEMMDAChallenge(h.ctx, &types.MsgOpenGEMMDAChallenge{
		Challenger: h.challenger, GemmTaskId: taskID, Provider: h.provA,
		Nonce: []byte("nonce-1"), Bond: DAPenaltyUprsm,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Deterministic, bounded derivation.
	derivedI, derivedJ, err := gemmv1.DAChallengeTile(task.ProtocolTaskID, []byte(h.provA),
		[]byte(h.challenger), []byte("nonce-1"), uint64(open.TileI), uint32(task.M/8), uint32(task.N/8))
	if err != nil {
		t.Fatal(err)
	}
	if derivedI != open.TileI || derivedJ != open.TileJ {
		t.Fatalf("tile derivation not deterministic: %d,%d vs %d,%d", derivedI, derivedJ, open.TileI, open.TileJ)
	}
	rowsC := uint32((task.M + 7) / 8)
	if open.TileI >= rowsC {
		t.Fatalf("derived tile out of range: %d", open.TileI)
	}
	// One open challenge per provider: a second is a DoS and is rejected.
	if _, err := h.msg.OpenGEMMDAChallenge(h.ctx, &types.MsgOpenGEMMDAChallenge{
		Challenger: h.requester, GemmTaskId: taskID, Provider: h.provA,
		Nonce: []byte("nonce-2"), Bond: DAPenaltyUprsm,
	}); err == nil {
		t.Fatal("second concurrent DA challenge accepted")
	}
	// A mismatched bond is rejected.
	if _, err := h.msg.OpenGEMMDAChallenge(h.ctx, &types.MsgOpenGEMMDAChallenge{
		Challenger: h.requester, GemmTaskId: taskID, Provider: h.provB,
		Nonce: []byte("nonce-3"), Bond: 1,
	}); err == nil {
		t.Fatal("mismatched DA bond accepted")
	}

	// A wrong tile is rejected.
	if _, err := h.msg.RespondGEMMDAChallenge(h.ctx, &types.MsgRespondGEMMDAChallenge{
		Provider: h.provA, ChallengeId: open.ChallengeId,
		Tile: make([]byte, 256), ProofCount: 4, ProofIndex: 0,
	}); err == nil {
		t.Fatal("wrong tile accepted")
	}
	// An oversized proof is rejected before hashing.
	if _, err := h.msg.RespondGEMMDAChallenge(h.ctx, &types.MsgRespondGEMMDAChallenge{
		Provider: h.provA, ChallengeId: open.ChallengeId,
		Tile: make([]byte, 256), ProofCount: 4, ProofIndex: 0,
		ProofSiblings: [][]byte{make([]byte, 32), make([]byte, 32), make([]byte, 32), make([]byte, 32), make([]byte, 32)},
	}); err == nil {
		t.Fatal("oversized DA proof accepted")
	}

	// The correct tile with a bounded proof passes.
	levels, err := gemmv1.BuildLevels(gemmOutputLeaves(h, taskID, h.lastTiles))
	if err != nil {
		t.Fatal(err)
	}
	index := open.TileI*rowsC + open.TileJ
	proof, err := gemmv1.ProveLeaf(levels, index)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.msg.RespondGEMMDAChallenge(h.ctx, &types.MsgRespondGEMMDAChallenge{
		Provider: h.provA, ChallengeId: open.ChallengeId,
		Tile:          h.lastTiles[index].CanonicalBytes(),
		ProofSiblings: proof.Siblings, ProofIndex: proof.Index, ProofCount: proof.Count,
	}); err != nil {
		t.Fatalf("valid tile response rejected: %v", err)
	}

	// Timeout path: an unanswered challenge is objectively failed on chain.
	open2, err := h.msg.OpenGEMMDAChallenge(h.ctx, &types.MsgOpenGEMMDAChallenge{
		Challenger: h.requester, GemmTaskId: taskID, Provider: h.provB,
		Nonce: []byte("nonce-4"), Bond: DAPenaltyUprsm,
	})
	if err != nil {
		t.Fatal(err)
	}
	providerBondBefore := h.keeper.GetBond(h.ctx, h.provB)
	requesterBalBefore := h.bank.accounts[h.requester]
	h.advanceHeight(int64(open2.Deadline) + 1)
	if _, err := h.msg.TimeoutGEMMDAChallenge(h.ctx, &types.MsgTimeoutGEMMDAChallenge{
		Actor: h.requester, ChallengeId: open2.ChallengeId,
	}); err != nil {
		t.Fatalf("DA timeout rejected: %v", err)
	}
	if h.keeper.GetBond(h.ctx, h.provB) != providerBondBefore-DAPenaltyUprsm {
		t.Fatal("provider bond not slashed by the fixed penalty")
	}
	// The requester recovers the challenge bond AND receives the provider's
	// penalty as the objective reward.
	if h.bank.accounts[h.requester] != requesterBalBefore+2*DAPenaltyUprsm {
		t.Fatalf("challenger recovery wrong: got delta %d, want %d",
			h.bank.accounts[h.requester]-requesterBalBefore, 2*DAPenaltyUprsm)
	}
	// The same challenge cannot be timed out twice.
	if _, err := h.msg.TimeoutGEMMDAChallenge(h.ctx, &types.MsgTimeoutGEMMDAChallenge{
		Actor: h.requester, ChallengeId: open2.ChallengeId,
	}); err == nil {
		t.Fatal("duplicate DA timeout accepted")
	}
}

// TestDAFinalizeBlockedByOpenChallenge verifies that an open DA challenge
// keeps the task from finalizing even with the quorum reached.
func TestDAFinalizeBlockedByOpenChallenge(t *testing.T) {
	h, taskID := daHarness(t, 96)
	h.registerDAProvider(h.provA, h.provAKeys)
	h.registerDAProvider(h.provB, h.provBKeys)
	for i, monitor := range []string{h.monitorA, h.monitorB} {
		h.bondWorker(monitor, newGemmKeys(byte(85+i)))
		if _, err := h.msg.AttestGEMMTask(h.ctx, &types.MsgAttestGEMMTask{Monitor: monitor, GemmTaskId: taskID}); err != nil {
			t.Fatal(err)
		}
	}
	if err := h.submitAttestation(taskID, h.provA, h.rawAttestation(taskID, h.provA, h.provAKeys, nil)); err != nil {
		t.Fatal(err)
	}
	if err := h.submitAttestation(taskID, h.provB, h.rawAttestation(taskID, h.provB, h.provBKeys, nil)); err != nil {
		t.Fatal(err)
	}
	task, _ := h.keeper.GetGEMMTask(h.ctx, taskID)
	h.advanceHeight(int64(task.ChallengeEnd) + 1)
	if _, err := h.msg.OpenGEMMDAChallenge(h.ctx, &types.MsgOpenGEMMDAChallenge{
		Challenger: h.challenger, GemmTaskId: taskID, Provider: h.provA,
		Nonce: []byte("nonce-x"), Bond: DAPenaltyUprsm,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := h.msg.FinalizeGEMM(h.ctx, &types.MsgFinalizeGEMM{Actor: h.requester, GemmTaskId: taskID}); err == nil {
		t.Fatal("finalize accepted with an open DA challenge")
	}
}
