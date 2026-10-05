package compute

// GEMM challenge admission, trace locking, bisection, timeout and
// arbitration. The dispute core is the prismachain/compute/gemmv1 state
// machine, persisted as a canonical GMD1 snapshot so bisection survives
// node restarts. Every consensus transition appends to a versioned
// transcript hash chain; the final digest is what a Verified Work Receipt
// carries.

import (
	"context"
	"encoding/json"
	"errors"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"prismachain/chain/x/compute/types"
	"prismachain/compute/gemmv1"
)

var (
	errGEMMProofShape      = errors.New("gemm: merkle proof size does not match the task shape")
	errGEMMNotDisputeParty = errors.New("gemm: actor is not a dispute party")
	errGEMMWrongPhase      = errors.New("gemm: dispute is in the wrong phase")
	errGEMMTileMembership  = errors.New("gemm: worker output tile is not in the committed output_root")
	errGEMMTraceDeadline   = errors.New("gemm: trace lock deadline expired")
)

// gemmLeafCountFor returns the output-tree leaf count and the expected
// proof depth for the task shape, so oversized proofs are rejected before
// any hashing (DoS guard).
func gemmLeafCountFor(task *GEMMTask) (uint32, int) {
	rowsC := uint32((task.M + gemmv1.TileSize - 1) / gemmv1.TileSize)
	colsC := uint32((task.N + gemmv1.TileSize - 1) / gemmv1.TileSize)
	count := rowsC * colsC
	return count, gemmv1.DepthFor(count)
}

// gemmProofFromMsg builds a protocol MerkleProof after checking the
// sibling count against the expected tree depth.
func gemmProofFromMsg(siblings [][]byte, index, count, expectedDepth int) (gemmv1.MerkleProof, error) {
	if count <= 0 || index >= count || len(siblings) != expectedDepth {
		return gemmv1.MerkleProof{}, errGEMMProofShape
	}
	proof := gemmv1.MerkleProof{Index: uint32(index), Count: uint32(count)}
	proof.Siblings = make([][]byte, len(siblings))
	for i, s := range siblings {
		if len(s) != 32 {
			return gemmv1.MerkleProof{}, errGEMMProofShape
		}
		proof.Siblings[i] = append([]byte(nil), s...)
	}
	return proof, nil
}

// transcriptStep advances the dispute transcript hash chain.
func transcriptStep(prev []byte, tag string, event any) ([]byte, error) {
	var p [32]byte
	copy(p[:], prev)
	digest, err := gemmv1.TranscriptStep(p, tag, event)
	if err != nil {
		return nil, err
	}
	return digest[:], nil
}

// OpenGEMMChallenge admits one bonded challenger, locks its bond on chain,
// verifies the worker tile membership in the committed output_root and
// records the canonical ChallengeOpen. This is a disputed claim, not a
// fraud proof.
func (s msgServer) OpenGEMMChallenge(ctx context.Context, msg *types.MsgOpenGEMMChallenge) (*types.MsgOpenGEMMChallengeResponse, error) {
	s.consumeGEMMGas(ctx, "challenge base", GasGEMMTxBase)
	task, err := s.GetGEMMTask(ctx, msg.GemmTaskId)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if task.Status != GEMMStatusResultSubmitted && task.Status != GEMMStatusChallenged {
		return nil, errors.New("gemm challenge unavailable")
	}
	if task.Status == GEMMStatusResultSubmitted && height > task.ChallengeEnd {
		return nil, errors.New("gemm challenge window closed")
	}
	networkKey, err := s.bondedNetworkKey(ctx, msg.Challenger)
	if err != nil {
		return nil, err
	}
	if msg.Challenger == task.Worker || msg.Challenger == task.Requester {
		return nil, errors.New("challenger is not independent")
	}
	// Rebuild the canonical ChallengeOpen with the CHAIN-derived bond; the
	// challenger signature must verify over exactly this object.
	bond := gemmBondFor(task.MaxFee)
	if msg.ChallengeBond != bond {
		return nil, errors.New("challenge bond does not match the schedule")
	}
	co := &gemmv1.ChallengeOpen{
		ProtocolVersion:      GEMMProtocolVersion,
		TaskID:               task.ProtocolTaskID,
		AssignmentID:         task.AssignmentID,
		ChallengerPubKey:     networkKey,
		WorkerPubKey:         task.WorkerProtocolPubKey,
		DisputedTileI:        msg.DisputedTileI,
		DisputedTileJ:        msg.DisputedTileJ,
		WorkerOutputTile:     append([]byte(nil), msg.WorkerOutputTile...),
		ChallengerOutputTile: append([]byte(nil), msg.ChallengerOutputTile...),
		ChallengeBond:        bond,
		OpenedEpoch:          msg.OpenedEpoch,
	}
	co.ChallengerSignature = append([]byte(nil), msg.ChallengerSignature...)
	// The signed epoch must be a plausible, already-past epoch: not from
	// the future and not before the result it disputes.
	if msg.OpenedEpoch > height || msg.OpenedEpoch < task.ResultSubmittedHeight {
		return nil, errors.New("gemm challenge opened epoch is out of range")
	}
	if err := gemmv1.ValidateChallengeOpen(task.mustDescriptor(), co); err != nil {
		return nil, err
	}
	// Worker tile membership in the committed output_root, with the proof
	// shape bounded before hashing.
	leafCount, depth := gemmLeafCountFor(&task)
	if msg.WorkerProofCount != leafCount {
		return nil, errGEMMProofShape
	}
	proof, err := gemmProofFromMsg(msg.WorkerProofSiblings, int(msg.WorkerProofIndex),
		int(msg.WorkerProofCount), depth)
	if err != nil {
		return nil, err
	}
	co.WorkerOutputProof = proof
	s.gasForProofs(ctx, "output tile proof", 1, len(proof.Siblings))
	s.gasForStates(ctx, "output tiles", 2*256)
	leaf := gemmv1.LeafOutputTile(task.ProtocolTaskID, task.AssignmentID,
		uint32(co.DisputedTileI), uint32(co.DisputedTileJ), co.WorkerOutputTile)
	if !gemmv1.VerifyLeafInclusion((root32(task.OutputRoot)), leaf, proof) {
		return nil, errGEMMTileMembership
	}
	if !gemmv1.VerifyChallengeOpenSignature(co) {
		return nil, errors.New("gemm ChallengeOpen signature is invalid")
	}
	// Decide the admission path BEFORE moving any funds, so a rejected
	// duplicate or saturated queue never locks a second bond.
	transcript, err := transcriptStep(task.DisputeTranscriptDigest, gemmv1.TranscriptTagChallengeOpened,
		gemmv1.TranscriptChallengeOpened{
			TaskID: task.ProtocolTaskID, AssignmentID: task.AssignmentID,
			TileI: uint32(co.DisputedTileI), TileJ: uint32(co.DisputedTileJ),
			ChallengerPubKey: networkKey, Bond: bond,
		})
	if err != nil {
		return nil, err
	}
	if task.Status == GEMMStatusChallenged {
		record, err := s.GetGEMMDispute(ctx, task.ID)
		if err != nil {
			return nil, err
		}
		if record.Challenger == msg.Challenger || len(task.QueuedGEMMChallenges) >= MaxQueuedChallenges {
			return nil, errors.New("challenger already active or queue full")
		}
		for _, queued := range task.QueuedGEMMChallenges {
			if queued.Challenger == msg.Challenger {
				return nil, errors.New("challenger already queued")
			}
		}
		if err := s.bank.SendCoinsFromAccountToModule(ctx, mustAddr(msg.Challenger), ModuleName, amount(bond)); err != nil {
			return nil, err
		}
		task.QueuedGEMMChallenges = append(task.QueuedGEMMChallenges, GEMMQueuedChallenge{
			Challenger: msg.Challenger, ChallengerPubKey: append([]byte(nil), networkKey...),
			Bond: bond, TileI: co.DisputedTileI, TileJ: co.DisputedTileJ,
			WorkerTile: co.WorkerOutputTile, ChallengerTile: co.ChallengerOutputTile,
		})
		if err := s.setGEMMTask(ctx, task); err != nil {
			return nil, err
		}
		return &types.MsgOpenGEMMChallengeResponse{}, nil
	}
	record := GEMMDisputeRecord{
		TaskID: task.ID, Challenger: msg.Challenger,
		ChallengerPubKey: append([]byte(nil), networkKey...), Bond: bond,
		WorkerTile: co.WorkerOutputTile, ChallengerTile: co.ChallengerOutputTile,
		TranscriptDigest: transcript, Status: GEMMDisputeOpen,
		TraceDeadline: height + ChallengeRoundBlocks,
	}
	if err := s.bank.SendCoinsFromAccountToModule(ctx, mustAddr(msg.Challenger), ModuleName, amount(bond)); err != nil {
		return nil, err
	}
	task.ActiveTileI, task.ActiveTileJ = co.DisputedTileI, co.DisputedTileJ
	task.Status = GEMMStatusChallenged
	if task.ChallengeEnd < height+ChallengeBlocks {
		task.ChallengeEnd = height + ChallengeBlocks
	}
	if err := s.setGEMMDispute(ctx, record); err != nil {
		return nil, err
	}
	return &types.MsgOpenGEMMChallengeResponse{ChallengeEnd: task.ChallengeEnd}, s.setGEMMTask(ctx, task)
}

// CommitGEMMTrace locks one party's on-demand tile trace. Only after both
// parties lock does the dispute core start, persisted as a GMD1 snapshot.
func (s msgServer) CommitGEMMTrace(ctx context.Context, msg *types.MsgCommitGEMMTrace) (*types.MsgCommitGEMMTraceResponse, error) {
	s.consumeGEMMGas(ctx, "trace base", GasGEMMTxBase)
	task, err := s.GetGEMMTask(ctx, msg.GemmTaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != GEMMStatusChallenged {
		return nil, errGEMMWrongPhase
	}
	record, err := s.GetGEMMDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	if record.Status != GEMMDisputeOpen {
		return nil, errGEMMWrongPhase
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if height > record.TraceDeadline {
		return nil, errGEMMTraceDeadline
	}
	var party gemmv1.Party
	var pubKey []byte
	switch msg.Actor {
	case task.Worker:
		party, pubKey = gemmv1.Worker, task.WorkerProtocolPubKey
	case record.Challenger:
		party, pubKey = gemmv1.Challenger, record.ChallengerPubKey
	default:
		return nil, errGEMMNotDisputeParty
	}
	if (party == gemmv1.Worker && len(record.WorkerTrace) != 0) ||
		(party == gemmv1.Challenger && len(record.ChallengerTrace) != 0) {
		return nil, errors.New("gemm trace already locked for this party")
	}
	rSteps := uint32(gemmv1.RSteps(task.K))
	traceLeafCount := rSteps + 1
	traceDepth := gemmv1.DepthFor(traceLeafCount)
	initialProof, err := gemmProofFromMsg(msg.InitialProofSiblings, int(msg.InitialProofIndex),
		int(msg.InitialProofCount), traceDepth)
	if err != nil {
		return nil, err
	}
	finalProof, err := gemmProofFromMsg(msg.FinalProofSiblings, int(msg.FinalProofIndex),
		int(msg.FinalProofCount), traceDepth)
	if err != nil {
		return nil, err
	}
	s.gasForProofs(ctx, "trace endpoint proofs", 2, traceDepth)
	s.gasForStates(ctx, "trace states", 2*256)
	tc := &gemmv1.TraceCommit{
		ProtocolVersion: GEMMProtocolVersion,
		TaskID:          task.ProtocolTaskID,
		AssignmentID:    task.AssignmentID,
		DisputedTileI:   task.ActiveTileI,
		DisputedTileJ:   task.ActiveTileJ,
		Party:           uint64(party),
		TraceRoot:       append([]byte(nil), msg.TraceRoot...),
		InitialState:    append([]byte(nil), msg.InitialState...),
		InitialProof:    outputProofOf(initialProof),
		FinalState:      append([]byte(nil), msg.FinalState...),
		FinalProof:      outputProofOf(finalProof),
		LockedEpoch:     msg.LockedEpoch,
		Signature:       append([]byte(nil), msg.Signature...),
	}
	if !digest32(msg.TraceRoot) || len(msg.InitialState) != 256 || len(msg.FinalState) != 256 {
		return nil, errors.New("gemm trace commit has invalid sizes")
	}
	if !gemmv1.VerifyTraceCommitSignature(tc, pubKey) {
		return nil, errors.New("gemm TraceCommit signature is invalid")
	}
	if err := gemmv1.VerifyTraceCommit(task.mustDescriptor(), task.AssignmentID, tc); err != nil {
		return nil, err
	}
	// The locked trace must end at the party's claimed output tile.
	expectedFinal := record.WorkerTile
	if party == gemmv1.Challenger {
		expectedFinal = record.ChallengerTile
	}
	if !equalBytesMsg(tc.FinalState, expectedFinal) {
		return nil, errors.New("gemm locked trace S_R does not match the party's claimed tile")
	}
	if msg.LockedEpoch < task.ResultSubmittedHeight {
		return nil, errors.New("gemm trace locked before the result existed")
	}
	tcCBOR, err := json.Marshal(tc)
	if err != nil {
		return nil, err
	}
	transcript, err := transcriptStep(record.TranscriptDigest, gemmv1.TranscriptTagTraceLocked,
		gemmv1.TranscriptTraceLocked{Party: uint64(party), TraceRoot: tc.TraceRoot})
	if err != nil {
		return nil, err
	}
	record.TranscriptDigest = transcript
	if party == gemmv1.Worker {
		record.WorkerTrace = tcCBOR
	} else {
		record.ChallengerTrace = tcCBOR
	}
	if len(record.WorkerTrace) == 0 || len(record.ChallengerTrace) == 0 {
		return &types.MsgCommitGEMMTraceResponse{}, s.setGEMMDispute(ctx, record)
	}
	// Both traces locked: start the persisted dispute.
	workerClaim, challengerClaim, err := claimsFromRecord(&task, &record)
	if err != nil {
		return nil, err
	}
	dispute, err := gemmv1.NewGEMMDispute(gemmDisputeConfig(&task, &record), workerClaim, challengerClaim, height)
	if err != nil {
		return nil, err
	}
	snapshot, err := dispute.SnapshotV1()
	if err != nil {
		return nil, err
	}
	record.Snapshot = snapshot
	record.Status = GEMMDisputeBisection
	if err := s.setGEMMDispute(ctx, record); err != nil {
		return nil, err
	}
	s.consumeGEMMGas(ctx, "gemm dispute write", GasPerGEMMStoreWrite)
	return &types.MsgCommitGEMMTraceResponse{}, nil
}

// SubmitGEMMMidState advances the persisted bisection by one round. The
// midpoint is derived from chain state; a stale or future step is rejected.
func (s msgServer) SubmitGEMMMidState(ctx context.Context, msg *types.MsgSubmitGEMMMidState) (*types.MsgSubmitGEMMMidStateResponse, error) {
	s.consumeGEMMGas(ctx, "mid base", GasGEMMTxBase)
	task, err := s.GetGEMMTask(ctx, msg.GemmTaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != GEMMStatusChallenged {
		return nil, errGEMMWrongPhase
	}
	record, err := s.GetGEMMDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	if record.Status != GEMMDisputeBisection || len(record.Snapshot) == 0 {
		return nil, errGEMMWrongPhase
	}
	dispute, err := gemmv1.RestoreGEMMDispute(record.Snapshot, gemmDisputeConfig(&task, &record))
	if err != nil {
		return nil, err
	}
	var party gemmv1.Party
	switch msg.Actor {
	case task.Worker:
		party = gemmv1.Worker
	case record.Challenger:
		party = gemmv1.Challenger
	default:
		return nil, errGEMMNotDisputeParty
	}
	rSteps := uint32(gemmv1.RSteps(task.K))
	traceDepth := gemmv1.DepthFor(rSteps + 1)
	proof, err := gemmProofFromMsg(msg.ProofSiblings, int(msg.ProofIndex), int(msg.ProofCount), traceDepth)
	if err != nil {
		return nil, err
	}
	s.gasForProofs(ctx, "midpoint proof", 1, len(proof.Siblings))
	s.gasForStates(ctx, "midpoint state", 256)
	if len(msg.State) != 256 {
		return nil, errors.New("gemm midpoint state must be 256 bytes")
	}
	outcome, err := dispute.SubmitMid(party, msg.State, proof, uint64(sdk.UnwrapSDKContext(ctx).BlockHeight()))
	if err != nil {
		return nil, err
	}
	status := dispute.Status()
	if status.WorkerSubmitted && status.ChallengerSubmitted {
		workerState, challengerState := transcriptStates(dispute)
		transcript, err := transcriptStep(record.TranscriptDigest, gemmv1.TranscriptTagBisection,
			gemmv1.TranscriptBisectionResolved{
				Step: status.Mid, KeptLow: keptLowFlag(dispute, status),
				WorkerState: workerState, ChallengerState: challengerState,
			})
		if err != nil {
			return nil, err
		}
		record.TranscriptDigest = transcript
	}
	if status.ArbitrationReady {
		record.Status = GEMMDisputeArbReady
	}
	snapshot, err := dispute.SnapshotV1()
	if err != nil {
		return nil, err
	}
	record.Snapshot = snapshot
	if err := s.resolveGEMMChallenge(ctx, &task, &record, outcome); err != nil {
		return nil, err
	}
	return &types.MsgSubmitGEMMMidStateResponse{}, s.setGEMMTask(ctx, task)
}

// TimeoutGEMM applies the persisted round deadline. A dispute ready for
// arbitration cannot time out.
func (s msgServer) TimeoutGEMM(ctx context.Context, msg *types.MsgTimeoutGEMM) (*types.MsgTimeoutGEMMResponse, error) {
	s.consumeGEMMGas(ctx, "timeout base", GasGEMMTxBase)
	if _, err := addr(msg.Actor); err != nil {
		return nil, err
	}
	task, err := s.GetGEMMTask(ctx, msg.GemmTaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != GEMMStatusChallenged {
		return nil, errGEMMWrongPhase
	}
	record, err := s.GetGEMMDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	var outcome gemmv1.Outcome
	if record.Status == GEMMDisputeOpen {
		// Trace-lock phase: the party that failed to lock loses.
		if height <= record.TraceDeadline {
			return nil, errors.New("gemm trace deadline has not expired")
		}
		switch {
		case len(record.WorkerTrace) != 0 && len(record.ChallengerTrace) == 0:
			outcome = gemmv1.WorkerWins
		case len(record.WorkerTrace) == 0 && len(record.ChallengerTrace) != 0:
			outcome = gemmv1.ChallengerWins
		default:
			outcome = gemmv1.BothInvalid
		}
	} else {
		dispute, err := gemmv1.RestoreGEMMDispute(record.Snapshot, gemmDisputeConfig(&task, &record))
		if err != nil {
			return nil, err
		}
		outcome, err = dispute.Timeout(height)
		if err != nil {
			return nil, err
		}
		snapshot, err := dispute.SnapshotV1()
		if err != nil {
			return nil, err
		}
		record.Snapshot = snapshot
	}
	if outcome != gemmv1.Pending {
		transcript, err := transcriptStep(record.TranscriptDigest, gemmv1.TranscriptTagTimeout,
			gemmv1.TranscriptTimeout{Outcome: uint64(outcome)})
		if err != nil {
			return nil, err
		}
		record.TranscriptDigest = transcript
	}
	if err := s.resolveGEMMChallenge(ctx, &task, &record, outcome); err != nil {
		return nil, err
	}
	return &types.MsgTimeoutGEMMResponse{Outcome: outcomeName(outcome)}, s.setGEMMTask(ctx, task)
}

// ArbitrateGEMM resolves the final disputed transition with exactly one
// 8x8x8 micro-step (512 canonical MACs). It is permissionless: the result
// is fully determined by chain state plus the witness tiles, which are
// verified against the committed matrix roots.
func (s msgServer) ArbitrateGEMM(ctx context.Context, msg *types.MsgArbitrateGEMM) (*types.MsgArbitrateGEMMResponse, error) {
	s.consumeGEMMGas(ctx, "arbitration base", GasGEMMTxBase+GasGEMMArbitration)
	task, err := s.GetGEMMTask(ctx, msg.GemmTaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != GEMMStatusChallenged {
		return nil, errGEMMWrongPhase
	}
	record, err := s.GetGEMMDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	if record.Status != GEMMDisputeArbReady || len(record.Snapshot) == 0 {
		return nil, errGEMMWrongPhase
	}
	dispute, err := gemmv1.RestoreGEMMDispute(record.Snapshot, gemmDisputeConfig(&task, &record))
	if err != nil {
		return nil, err
	}
	if !dispute.Status().ArbitrationReady {
		return nil, errGEMMWrongPhase
	}
	if len(msg.ATile) != 64 || len(msg.BTile) != 64 {
		return nil, errors.New("gemm arbitration tiles must be 64 bytes")
	}
	step := dispute.Status().Low
	rowsA := uint32((task.M + gemmv1.TileSize - 1) / gemmv1.TileSize)
	colsA := uint32((task.K + gemmv1.TileSize - 1) / gemmv1.TileSize)
	rowsB := colsA
	colsB := uint32((task.N + gemmv1.TileSize - 1) / gemmv1.TileSize)
	aProof, err := gemmInputProofFromMsg(gemmv1.MatrixIDA, msg.AProofTileRow, msg.AProofTileCol,
		msg.AProofTileCols, rowsA*colsA, msg.AProofSiblings, msg.AProofCount)
	if err != nil {
		return nil, err
	}
	bProof, err := gemmInputProofFromMsg(gemmv1.MatrixIDB, msg.BProofTileRow, msg.BProofTileCol,
		msg.BProofTileCols, rowsB*colsB, msg.BProofSiblings, msg.BProofCount)
	if err != nil {
		return nil, err
	}
	if aProof.TileRow != dispute.TileI() || aProof.TileCol != step ||
		bProof.TileRow != step || bProof.TileCol != dispute.TileJ() {
		return nil, errors.New("gemm arbitration witness is for another step")
	}
	s.gasForProofs(ctx, "arbitration input proofs", 2, len(aProof.Siblings)+len(bProof.Siblings))
	var aTile, bTile [64]int8
	for i := 0; i < 64; i++ {
		aTile[i] = int8(msg.ATile[i])
		bTile[i] = int8(msg.BTile[i])
	}
	outcome, err := dispute.Arbitrate(aTile, bTile, aProof, bProof, uint64(sdk.UnwrapSDKContext(ctx).BlockHeight()))
	if err != nil {
		return nil, err
	}
	expected := expectedStateOf(dispute, aTile, bTile)
	transcript, err := transcriptStep(record.TranscriptDigest, gemmv1.TranscriptTagArbitration,
		gemmv1.TranscriptArbitration{Step: step, Outcome: uint64(outcome), Expected: expected})
	if err != nil {
		return nil, err
	}
	record.TranscriptDigest = transcript
	if err := s.resolveGEMMChallenge(ctx, &task, &record, outcome); err != nil {
		return nil, err
	}
	return &types.MsgArbitrateGEMMResponse{Outcome: outcomeName(outcome)}, s.setGEMMTask(ctx, task)
}
