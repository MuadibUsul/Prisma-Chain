package compute

// GEMM finalization, Verified Work Receipt creation and the verified-work
// ledger. Receipts are canonical gemmv1 objects hashed with the protocol's
// canonical CBOR; only keeper state transitions can create them, and a
// receipt ID can exist only once.

import (
	"context"
	"errors"
	"fmt"
	"strconv"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"prismachain/chain/x/compute/types"
	"prismachain/compute/gemmv1"
)

// gemmFetchNothing is the honest Phase D availability stance: the chain
// cannot fetch off-chain URLs, so output_data_ref is availability metadata
// only. Data-availability enforcement is an open gate, not solved here.
const gemmDataAvailabilityNote = "output_data_ref is off-chain availability metadata; strong data-availability enforcement remains an open gate"

// AbortGEMMTask refunds an expired, never-submitted task. This is the only
// way escrow leaves a posted/assigned GEMM task short of fraud, so it is
// strictly limited to the requester after the acceptance window.
func (s msgServer) AbortGEMMTask(ctx context.Context, msg *types.MsgAbortGEMMTask) (*types.MsgAbortGEMMTaskResponse, error) {
	s.consumeGEMMGas(ctx, "abort base", GasGEMMTxBase)
	if _, err := addr(msg.Actor); err != nil {
		return nil, err
	}
	task, err := s.GetGEMMTask(ctx, msg.GemmTaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if msg.Actor != task.Requester ||
		(task.Status != GEMMStatusPosted && task.Status != GEMMStatusAssigned) ||
		height <= task.IssuedHeight+task.ChallengeWindow {
		return nil, errors.New("gemm task is not abortable")
	}
	requester := mustAddr(task.Requester)
	if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, requester, amount(task.MaxFee)); err != nil {
		return nil, err
	}
	if err := s.releaseGEMMReservation(ctx, &task); err != nil {
		return nil, err
	}
	task.Status = GEMMStatusRefunded
	return &types.MsgAbortGEMMTaskResponse{}, s.setGEMMTask(ctx, task)
}

// FinalizeGEMM pays a worker whose challenge window closed without an
// unresolved challenge, exactly once, and mints the Verified Work Receipt.
// Settlement economics mirror the verifiable-task fee split.
func (s msgServer) FinalizeGEMM(ctx context.Context, msg *types.MsgFinalizeGEMM) (*types.MsgFinalizeGEMMResponse, error) {
	s.consumeGEMMGas(ctx, "finalize base", GasGEMMTxBase)
	if _, err := addr(msg.Actor); err != nil {
		return nil, err
	}
	task, err := s.GetGEMMTask(ctx, msg.GemmTaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if task.Status != GEMMStatusResultSubmitted || height <= task.ChallengeEnd ||
		len(task.Attesters) != 2 {
		return nil, errors.New("gemm task not finalizable")
	}
	if task.OutputRoot == nil || task.AssignmentID == nil {
		return nil, errors.New("gemm task result state is incomplete")
	}
	// Rebuild the canonical ResultCommit from chain state; the worker
	// signature was verified at submission and is re-verified here so the
	// receipt is built over exactly the committed object.
	// The rebuilt commit must use the worker's declared completed epoch,
	// or the signature (which covers it) will not verify.
	rc := task.gemmResultCommit(task.OutputRoot, task.ResultCompletedEpoch)
	rc.WorkerSignature = append([]byte(nil), task.ResultCommitSignature...)
	if !gemmv1.VerifyResultCommitSignature(rc) {
		return nil, errors.New("gemm ResultCommit signature no longer verifies")
	}
	assignment := task.gemmAssignment()
	verificationMode := gemmv1.ModeOptimisticUnchallenged
	var transcriptDigest []byte
	if task.SurvivedChallenge {
		verificationMode = gemmv1.ModeChallengedWorkerWon
		transcriptDigest = task.DisputeTranscriptDigest
	}
	receipt, err := gemmv1.BuildVerifiedWorkReceipt(task.mustDescriptor(), assignment, rc,
		verificationMode, height, []byte("gemm:"+strconv.FormatUint(task.ID, 10)), transcriptDigest)
	if err != nil {
		return nil, err
	}
	receiptID, err := receipt.ReceiptID()
	if err != nil {
		return nil, err
	}
	if len(s.store(ctx).Get(gemmReceiptKey(receiptID))) != 0 {
		return nil, errors.New("gemm receipt already exists")
	}

	// Existing verifiable-task settlement semantics: burn 20%, each of the
	// two monitors 5%, worker the remainder.
	burn, monitorShare, workerShare := feeSplit(task.MaxFee)
	if burn > 0 {
		if err := s.bank.BurnCoins(ctx, ModuleName, amount(burn)); err != nil {
			return nil, err
		}
	}
	for _, a := range task.Attesters {
		if monitorShare > 0 {
			if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, mustAddr(a), amount(monitorShare)); err != nil {
				return nil, err
			}
		}
	}
	if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, mustAddr(task.Worker), amount(workerShare)); err != nil {
		return nil, err
	}
	if err := s.releaseGEMMReservation(ctx, &task); err != nil {
		return nil, err
	}

	receiptJSON, err := jsonMarshalReceipt(receipt)
	if err != nil {
		return nil, err
	}
	store := s.store(ctx)
	store.Set(gemmReceiptKey(receiptID), receiptJSON)
	s.addGEMMVerifiedWork(ctx, task.Worker, task.M*task.N*task.K)
	task.ReceiptID = receiptID
	task.Status = GEMMStatusFinalized
	if err := s.setGEMMTask(ctx, task); err != nil {
		return nil, err
	}
	s.consumeGEMMGas(ctx, "gemm receipt write", GasPerGEMMStoreWrite*2)
	_ = gemmDataAvailabilityNote
	return &types.MsgFinalizeGEMMResponse{ReceiptId: receiptID}, nil
}

// jsonMarshalReceipt renders the receipt as canonical JSON metadata for
// queries. The receipt ID always comes from the gemmv1 canonical CBOR
// hashing, never from this JSON.
func jsonMarshalReceipt(receipt *gemmv1.VerifiedWorkReceipt) ([]byte, error) {
	id, err := receipt.ReceiptID()
	if err != nil {
		return nil, err
	}
	return []byte(fmt.Sprintf(`{"receipt_id":"%x","task_id":"%x","assignment_id":"%x","worker_pubkey":"%x","operator":%q,"canonical_mac_count":%d,"output_root":"%x","verification_mode":%q,"finalized_epoch":%d,"settlement_reference":"%x","dispute_transcript_digest":"%x"}`,
		id, receipt.TaskID, receipt.AssignmentID, receipt.WorkerPubKey, receipt.Operator,
		receipt.CanonicalMACCount, receipt.OutputRoot, receipt.VerificationMode,
		receipt.FinalizedEpoch, receipt.SettlementReference, receipt.DisputeTranscriptDigest)), nil
}
