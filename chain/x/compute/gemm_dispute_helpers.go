package compute

// Shared helpers for the GEMM dispute state machine.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"prismachain/compute/gemmv1"
)

func mustAddr(raw string) sdk.AccAddress {
	a, err := addr(raw)
	if err != nil {
		panic(fmt.Sprintf("stored address is invalid: %v", err))
	}
	return a
}

func root32(b []byte) [32]byte {
	var out [32]byte
	copy(out[:], b)
	return out
}

func outcomeName(o gemmv1.Outcome) string {
	switch o {
	case gemmv1.WorkerWins:
		return "worker_wins"
	case gemmv1.ChallengerWins:
		return "challenger_wins"
	case gemmv1.BothInvalid:
		return "both_invalid"
	default:
		return "pending"
	}
}

func outputProofOf(p gemmv1.MerkleProof) gemmv1.OutputTileProof {
	return gemmv1.OutputTileProof{Index: p.Index, Count: p.Count, Siblings: p.Siblings}
}

func gemmInputProofFromMsg(matrixID byte, row, col, cols uint32, leafCount uint32, siblings [][]byte, count uint32) (gemmv1.InputTileProof, error) {
	if count != leafCount || cols == 0 || uint64(row)*uint64(cols)+uint64(col) >= uint64(leafCount) {
		return gemmv1.InputTileProof{}, errGEMMProofShape
	}
	depth := gemmv1.DepthFor(leafCount)
	if len(siblings) != depth {
		return gemmv1.InputTileProof{}, errGEMMProofShape
	}
	proof := gemmv1.InputTileProof{
		MatrixID: uint32(matrixID), TileRow: row, TileCol: col, TileCols: cols, Count: count,
		Siblings: make([][]byte, len(siblings)),
	}
	for i, s := range siblings {
		if len(s) != 32 {
			return gemmv1.InputTileProof{}, errGEMMProofShape
		}
		proof.Siblings[i] = append([]byte(nil), s...)
	}
	return proof, nil
}

// gemmDisputeConfig rebuilds the dispute context from chain state.
func gemmDisputeConfig(task *GEMMTask) gemmv1.DisputeConfig {
	rootA := root32(task.MatrixARoot)
	rootB := root32(task.MatrixBRoot)
	return gemmv1.DisputeConfig{
		Task:        task.mustDescriptor(),
		Assignment:  task.gemmAssignment(),
		TaskID:      task.ProtocolTaskID,
		TileI:       uint32(task.ActiveTileI),
		TileJ:       uint32(task.ActiveTileJ),
		MatrixARoot: rootA,
		MatrixBRoot: rootB,
		RoundPeriod: ChallengeRoundBlocks,
	}
}

// claimsFromRecord decodes the two locked TraceCommits and binds them to
// the dispute claims.
func claimsFromRecord(task *GEMMTask, record *GEMMDisputeRecord) (gemmv1.TraceClaim, gemmv1.TraceClaim, error) {
	var workerTC, challengerTC gemmv1.TraceCommit
	if err := json.Unmarshal(record.WorkerTrace, &workerTC); err != nil {
		return gemmv1.TraceClaim{}, gemmv1.TraceClaim{}, err
	}
	if err := json.Unmarshal(record.ChallengerTrace, &challengerTC); err != nil {
		return gemmv1.TraceClaim{}, gemmv1.TraceClaim{}, err
	}
	if !gemmv1.VerifyTraceCommitSignature(&workerTC, task.WorkerProtocolPubKey) {
		return gemmv1.TraceClaim{}, gemmv1.TraceClaim{}, errors.New("worker trace signature invalid")
	}
	if !gemmv1.VerifyTraceCommitSignature(&challengerTC, record.ChallengerPubKey) {
		return gemmv1.TraceClaim{}, gemmv1.TraceClaim{}, errors.New("challenger trace signature invalid")
	}
	worker := gemmv1.TraceClaim{Party: gemmv1.Worker, TraceRoot: root32(workerTC.TraceRoot),
		InitialState: workerTC.InitialState, InitialProof: workerTC.InitialProof,
		FinalState: workerTC.FinalState, FinalProof: workerTC.FinalProof}
	challenger := gemmv1.TraceClaim{Party: gemmv1.Challenger, TraceRoot: root32(challengerTC.TraceRoot),
		InitialState: challengerTC.InitialState, InitialProof: challengerTC.InitialProof,
		FinalState: challengerTC.FinalState, FinalProof: challengerTC.FinalProof}
	return worker, challenger, nil
}

// keptLowFlag reports whether the last completed bisection round moved the
// interval low (states equal) or high (states differ).
func keptLowFlag(dispute *gemmv1.GEMMDispute, status gemmv1.DisputeStatus) uint64 {
	if dispute.LastRoundKeptLow {
		return 1
	}
	return 0
}

// transcriptStates returns the party states recorded for a completed
// bisection round.
func transcriptStates(dispute *gemmv1.GEMMDispute) ([]byte, []byte) {
	low := dispute.LowState()
	lowBytes := low.CanonicalBytes()
	high := dispute.HighState(gemmv1.Challenger)
	highBytes := high.CanonicalBytes()
	return lowBytes, highBytes
}

// expectedStateOf recomputes the arbiter's expected state for the
// transcript: the same single 512-MAC micro-step the dispute performed.
func expectedStateOf(dispute *gemmv1.GEMMDispute, aTile, bTile [64]int8) []byte {
	return dispute.ExpectedArbitrationState(aTile, bTile).CanonicalBytes()
}

// resolveGEMMChallenge applies a finished dispute outcome to the task and
// its escrow/bond economics, mirroring the bounded-VM challenge rules.
func (s msgServer) resolveGEMMChallenge(ctx context.Context, task *GEMMTask, record *GEMMDisputeRecord, outcome gemmv1.Outcome) error {
	if outcome == gemmv1.Pending {
		return s.setGEMMDispute(ctx, *record)
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	challenger := mustAddr(record.Challenger)
	if outcome == gemmv1.WorkerWins {
		// Burning the losing challenger's bond prevents a worker-controlled
		// challenger from recovering the cost through the worker account.
		if err := s.bank.BurnCoins(ctx, ModuleName, amount(record.Bond)); err != nil {
			return err
		}
		task.SurvivedChallenge = true
		task.DisputeTranscriptDigest = append([]byte(nil), record.TranscriptDigest...)
		if len(task.QueuedGEMMChallenges) > 0 {
			next := task.QueuedGEMMChallenges[0]
			task.QueuedGEMMChallenges = task.QueuedGEMMChallenges[1:]
			newRecord := GEMMDisputeRecord{
				TaskID: task.ID, Challenger: next.Challenger,
				ChallengerPubKey: next.ChallengerPubKey, Bond: next.Bond,
				WorkerTile: next.WorkerTile, ChallengerTile: next.ChallengerTile,
				TranscriptDigest: record.TranscriptDigest, Status: GEMMDisputeOpen,
				TraceDeadline: height + ChallengeRoundBlocks,
			}
			task.ActiveTileI, task.ActiveTileJ = next.TileI, next.TileJ
			if err := s.setGEMMDispute(ctx, newRecord); err != nil {
				return err
			}
			task.Status = GEMMStatusChallenged
			return nil
		}
		task.Status = GEMMStatusResultSubmitted
		task.ChallengeEnd = height + task.ChallengeWindow
		s.deleteGEMMDispute(ctx, task.ID)
		return nil
	}
	// ChallengerWins or BothInvalid: refund unused queued bonds, refund the
	// requester escrow and release the worker reservation.
	for _, queued := range task.QueuedGEMMChallenges {
		if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, mustAddr(queued.Challenger), amount(queued.Bond)); err != nil {
			return err
		}
	}
	task.QueuedGEMMChallenges = nil
	requester := mustAddr(task.Requester)
	if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, requester, amount(task.MaxFee)); err != nil {
		return err
	}
	penaltyCap := task.ReservedBond
	if err := s.releaseGEMMReservation(ctx, task); err != nil {
		return err
	}
	if outcome == gemmv1.ChallengerWins {
		if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, challenger, amount(record.Bond)); err != nil {
			return err
		}
	} else if err := s.bank.BurnCoins(ctx, ModuleName, amount(record.Bond)); err != nil {
		return err
	}
	bond := s.GetBond(ctx, task.Worker)
	slash := bond / 10
	if slash > penaltyCap {
		slash = penaltyCap
	}
	if slash > 0 {
		if outcome == gemmv1.ChallengerWins {
			if err := s.bank.SendCoinsFromModuleToAccount(ctx, ModuleName, challenger, amount(slash)); err != nil {
				return err
			}
		} else if err := s.bank.BurnCoins(ctx, ModuleName, amount(slash)); err != nil {
			return err
		}
		s.setBond(ctx, task.Worker, bond-slash)
	}
	if outcome == gemmv1.ChallengerWins {
		task.Status = GEMMStatusFraud
	} else {
		task.Status = GEMMStatusRefunded
	}
	s.deleteGEMMDispute(ctx, task.ID)
	return nil
}

// releaseGEMMReservation mirrors releaseReservation for GEMM tasks.
func (s msgServer) releaseGEMMReservation(ctx context.Context, task *GEMMTask) error {
	if task.ReservedBond == 0 {
		return nil
	}
	reserved := s.getReserved(ctx, task.Worker)
	if reserved < task.ReservedBond {
		return errors.New("worker bond reservation invariant broken")
	}
	s.setReserved(ctx, task.Worker, reserved-task.ReservedBond)
	task.ReservedBond = 0
	return nil
}
