package compute

// WIDE_GEMM_DISPUTE_V1 on-chain path (roadmap A2-02..A2-08).  Opened only
// when the graph bisection localizes the first divergent node to a
// GEMM_A13W10_I64_V1 node; the chain performs bounded evidence checks and
// exactly one 512-MAC recomputation, then settles through the same
// resolveGraphOutcome economics as every other graph dispute.

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"prismachain/chain/x/compute/types"
	"prismachain/compute/canonical"
)

// Wide dispute statuses.
const (
	WideStatusClaiming = "claiming"
	WideStatusBisection = "bisection"
	WideStatusArbReady  = "wide_arb_ready"
	WideStatusResolved  = "resolved"
)

// WideDisputeRecord persists one wide dispute (restart-safe: the whole
// canonical state machine is stored as JSON).
type WideDisputeRecord struct {
	TaskID   uint64 `json:"task_id"`
	NodeID   uint32 `json:"node_id"`
	TileI    uint32 `json:"tile_i"`
	TileJ    uint32 `json:"tile_j"`
	M        int    `json:"m"`
	N        int    `json:"n"`
	K        int    `json:"k"`
	TransposeB bool `json:"transpose_b"`

	WorkerClaim     []byte `json:"worker_claim,omitempty"`
	ChallengerClaim []byte `json:"challenger_claim,omitempty"`
	DisputeJSON     []byte `json:"dispute_json,omitempty"`

	ClaimDeadline uint64 `json:"claim_deadline"`
	Status        string `json:"status"`
	Outcome       string `json:"outcome,omitempty"`
}

func wideDisputeKey(taskID uint64) []byte {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], taskID)
	return append([]byte("widedispute:"), b[:]...)
}

func (k Keeper) GetWideDispute(ctx context.Context, taskID uint64) (WideDisputeRecord, error) {
	raw := k.store(ctx).Get(wideDisputeKey(taskID))
	if len(raw) == 0 {
		return WideDisputeRecord{}, errors.New("wide dispute not found")
	}
	var rec WideDisputeRecord
	if err := json.Unmarshal(raw, &rec); err != nil {
		return WideDisputeRecord{}, err
	}
	return rec, nil
}

func (s msgServer) setWideDispute(ctx context.Context, rec WideDisputeRecord) error {
	raw, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	s.store(ctx).Set(wideDisputeKey(rec.TaskID), raw)
	return nil
}

func (s msgServer) deleteWideDispute(ctx context.Context, taskID uint64) {
	s.store(ctx).Delete(wideDisputeKey(taskID))
}

// wideGEMMGeometry derives (M, N, K, transposeB) from the node inputs.
func wideGEMMGeometry(g *canonical.GraphDescriptorV2, node canonical.GraphNodeV2) (int, int, int, bool, error) {
	descOf := func(ref canonical.TensorRef) canonical.TensorDescriptorV2 {
		if ref.Kind == 0 {
			return g.Inputs[ref.Index].Desc
		}
		return g.Nodes[ref.Index].Output
	}
	a := descOf(node.Inputs[0])
	b := descOf(node.Inputs[1])
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return 0, 0, 0, false, errors.New("wide GEMM operands must be matrices")
	}
	m, k := int(a.Shape[0]), int(a.Shape[1])
	trans := node.Params.Get("transpose_b", 0) == 1
	n := int(b.Shape[1])
	if trans {
		n = int(b.Shape[0])
	}
	return m, n, k, trans, nil
}

// OpenWideGEMMDispute (A2-02): only the graph dispute's challenger may
// open it, only on the first divergent node, and only when that node is a
// wide GEMM.  The tile must be inside the output geometry.
func (s msgServer) OpenWideGEMMDispute(ctx context.Context, msg *types.MsgOpenWideGEMMDispute) (*types.MsgOpenWideGEMMDisputeResponse, error) {
	s.consumeGraphGas(ctx, "wide open base", GasGraphTxBase)
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != GraphStatusChallenged {
		return nil, errGraphWrongPhase
	}
	record, err := s.GetGraphDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	if record.Status != GraphDisputeArbReady || len(record.Snapshot) == 0 {
		return nil, errors.New("the graph bisection must complete before a wide dispute can open")
	}
	if msg.Challenger != record.Challenger {
		return nil, errors.New("only the graph challenger may open the wide dispute")
	}
	if msg.ChallengeBond != record.Bond {
		return nil, errors.New("wide dispute bond must match the locked graph challenge bond")
	}
	graph, err := graphDescriptorV2(&task)
	if err != nil {
		return nil, err
	}
	dispute, err := canonical.RestoreGraphDisputeV2(record.Snapshot, graph, ChallengeRoundBlocks)
	if err != nil {
		return nil, err
	}
	nodeID, err := dispute.FirstDivergentNode()
	if err != nil {
		return nil, err
	}
	if msg.NodeId != nodeID {
		return nil, errors.New("wide dispute node must be the first divergent node")
	}
	node := graph.Nodes[nodeID]
	if node.OperatorID != canonical.OpGEMMWideA13W10 {
		return nil, errors.New("first divergent node is not a wide GEMM; use the typed node arbitration")
	}
	m, n, k, trans, err := wideGEMMGeometry(graph, node)
	if err != nil {
		return nil, err
	}
	// The chunk-extraction rule requires the operand rows to contain
	// whole 8-column windows (true for every real profile shape).
	if k%canonical.WideKStep != 0 || (!trans && n%canonical.WideKStep != 0) {
		return nil, errors.New("wide dispute requires K (and non-transposed N) divisible by 8")
	}
	if int(msg.TileI) >= (m+canonical.WideTileN-1)/canonical.WideTileN ||
		int(msg.TileJ) >= (n+canonical.WideTileN-1)/canonical.WideTileN {
		return nil, errors.New("tile out of range")
	}
	if _, err := s.GetWideDispute(ctx, task.ID); err == nil {
		return nil, errors.New("a wide dispute is already open for this task")
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	rec := WideDisputeRecord{
		TaskID: task.ID, NodeID: nodeID, TileI: msg.TileI, TileJ: msg.TileJ,
		M: m, N: n, K: k, TransposeB: trans,
		ClaimDeadline: height + ChallengeRoundBlocks,
		Status:        WideStatusClaiming,
	}
	if err := s.setWideDispute(ctx, rec); err != nil {
		return nil, err
	}
	record.TranscriptDigest, err = graphTranscriptStep(record.TranscriptDigest, "wide_dispute_opened",
		map[string]any{"node": nodeID, "tile_i": msg.TileI, "tile_j": msg.TileJ, "k": k})
	if err != nil {
		return nil, err
	}
	if err := s.setGraphDispute(ctx, record); err != nil {
		return nil, err
	}
	return &types.MsgOpenWideGEMMDisputeResponse{Deadline: rec.ClaimDeadline}, nil
}

// WideTraceClaim (A2-04): locks one party's partial-state trace.
func (s msgServer) WideTraceClaim(ctx context.Context, msg *types.MsgWideTraceClaim) (*types.MsgWideTraceClaimResponse, error) {
	s.consumeGraphGas(ctx, "wide claim base", GasGraphTxBase)
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != GraphStatusChallenged {
		return nil, errGraphWrongPhase
	}
	record, err := s.GetGraphDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	wd, err := s.GetWideDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	if wd.Status != WideStatusClaiming {
		return nil, errGraphWrongPhase
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if height > wd.ClaimDeadline {
		return nil, errors.New("wide trace claim window has closed")
	}
	if msg.Party != task.Worker && msg.Party != record.Challenger {
		return nil, errors.New("only the assigned worker or the challenger may claim")
	}
	initial, err := canonical.WideStateFromBytes(msg.InitialState)
	if err != nil {
		return nil, err
	}
	final, err := canonical.WideStateFromBytes(msg.FinalState)
	if err != nil {
		return nil, err
	}
	root, err := root32Of(msg.TraceRoot)
	if err != nil {
		return nil, err
	}
	initialProof, err := hashesFrom(msg.InitialProof)
	if err != nil {
		return nil, err
	}
	finalProof, err := hashesFrom(msg.FinalProof)
	if err != nil {
		return nil, err
	}
	claim := canonical.WideTraceClaimV1{Root: root, InitialState: initial,
		InitialProof: initialProof, FinalState: final, FinalProof: finalProof}
	claimJSON, err := json.Marshal(claim)
	if err != nil {
		return nil, err
	}
	if msg.Party == task.Worker {
		if len(wd.WorkerClaim) != 0 {
			return nil, errors.New("worker wide trace already claimed")
		}
		wd.WorkerClaim = claimJSON
	} else {
		if len(wd.ChallengerClaim) != 0 {
			return nil, errors.New("challenger wide trace already claimed")
		}
		wd.ChallengerClaim = claimJSON
	}
	s.consumeGraphGas(ctx, "wide claim hashing",
		GasGraphHash*uint64(len(initialProof)+len(finalProof)+4))
	if len(wd.WorkerClaim) != 0 && len(wd.ChallengerClaim) != 0 {
		var workerClaim, challengerClaim canonical.WideTraceClaimV1
		if err := json.Unmarshal(wd.WorkerClaim, &workerClaim); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(wd.ChallengerClaim, &challengerClaim); err != nil {
			return nil, err
		}
		steps := uint32(canonical.WideStepCount(wd.K))
		d, err := canonical.NewWideDisputeV1(steps, ChallengeRoundBlocks,
			int(wd.TileI), int(wd.TileJ), workerClaim, challengerClaim, height)
		if err != nil {
			return nil, err
		}
		stateJSON, err := d.MarshalState()
		if err != nil {
			return nil, err
		}
		wd.DisputeJSON = stateJSON
		wd.Status = WideStatusBisection
		if d.ArbReady {
			wd.Status = WideStatusArbReady
		}
		record.TranscriptDigest, err = graphTranscriptStep(record.TranscriptDigest, "wide_claims_locked",
			map[string]any{"tile_i": wd.TileI, "tile_j": wd.TileJ})
		if err != nil {
			return nil, err
		}
		if err := s.setGraphDispute(ctx, record); err != nil {
			return nil, err
		}
	}
	if err := s.setWideDispute(ctx, wd); err != nil {
		return nil, err
	}
	return &types.MsgWideTraceClaimResponse{}, nil
}

// WideMidPoint (A2-05): bisects over K-steps on the block-height clock.
func (s msgServer) WideMidPoint(ctx context.Context, msg *types.MsgWideMidPoint) (*types.MsgWideMidPointResponse, error) {
	s.consumeGraphGas(ctx, "wide midpoint base", GasGraphTxBase)
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != GraphStatusChallenged {
		return nil, errGraphWrongPhase
	}
	record, err := s.GetGraphDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	wd, err := s.GetWideDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	if wd.Status != WideStatusBisection || len(wd.DisputeJSON) == 0 {
		return nil, errGraphWrongPhase
	}
	height := uint64(sdk.UnwrapSDKContext(ctx).BlockHeight())
	if msg.Epoch > height+2 {
		return nil, errors.New("midpoint epoch is ahead of the chain clock")
	}
	var party canonical.Party
	switch msg.Party {
	case task.Worker:
		party = canonical.Worker
	case record.Challenger:
		party = canonical.Challenger
	default:
		return nil, errors.New("only the assigned worker or the challenger may submit midpoints")
	}
	d, err := canonical.UnmarshalWideDisputeV1(wd.DisputeJSON)
	if err != nil {
		return nil, err
	}
	state, err := canonical.WideStateFromBytes(msg.State)
	if err != nil {
		return nil, err
	}
	proof, err := hashesFrom(msg.Proof)
	if err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "wide midpoint proofs",
		GasGraphProofSibling*uint64(len(proof))+GasGraphHash)
	if _, err := d.SubmitMid(party, state, proof, msg.Epoch); err != nil {
		return nil, err
	}
	stateJSON, err := d.MarshalState()
	if err != nil {
		return nil, err
	}
	wd.DisputeJSON = stateJSON
	if d.ArbReady {
		wd.Status = WideStatusArbReady
	}
	if err := s.setWideDispute(ctx, wd); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "dispute store", GasGraphStore+GasGraphHash*uint64(len(stateJSON)/256+1))
	return &types.MsgWideMidPointResponse{}, nil
}

// ArbitrateWide512 (A2-06) + Graph bridge (A2-07): extracts the 8x8
// operand tiles from type-proven chunks, recomputes exactly 512 MAC in
// exact int64, and settles through resolveGraphOutcome.
func (s msgServer) ArbitrateWide512(ctx context.Context, msg *types.MsgArbitrateWide512) (*types.MsgArbitrateWide512Response, error) {
	s.consumeGraphGas(ctx, "wide arbitration base", GasGraphTxBase)
	task, err := s.GetGraphTask(ctx, msg.GraphTaskId)
	if err != nil {
		return nil, err
	}
	if task.Status != GraphStatusChallenged {
		return nil, errGraphWrongPhase
	}
	record, err := s.GetGraphDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	wd, err := s.GetWideDispute(ctx, task.ID)
	if err != nil {
		return nil, err
	}
	if wd.Status != WideStatusArbReady || len(wd.DisputeJSON) == 0 {
		return nil, errGraphWrongPhase
	}
	graph, err := graphDescriptorV2(&task)
	if err != nil {
		return nil, err
	}
	node := graph.Nodes[wd.NodeID]
	d, err := canonical.UnmarshalWideDisputeV1(wd.DisputeJSON)
	if err != nil {
		return nil, err
	}
	step, err := d.FirstDivergentStep()
	if err != nil {
		return nil, err
	}
	lowState, err := d.LowStateValue()
	if err != nil {
		return nil, err
	}
	workerHigh, challengerHigh, err := d.HighStates()
	if err != nil {
		return nil, err
	}
	// the submitted next states must equal the locked ones (the locked
	// values are what the parties signed against their trace roots)
	workerNext, err := canonical.WideStateFromBytes(msg.WorkerNextState)
	if err != nil {
		return nil, err
	}
	challengerNext, err := canonical.WideStateFromBytes(msg.ChallengerNextState)
	if err != nil {
		return nil, err
	}
	if workerNext != workerHigh || challengerNext != challengerHigh {
		return nil, errors.New("submitted next states do not match the locked trace endpoints")
	}
	workerRoot, err := hashOfHexLocal(d.Roots[0])
	if err != nil {
		return nil, err
	}
	challengerRoot, err := hashOfHexLocal(d.Roots[1])
	if err != nil {
		return nil, err
	}
	workerNextProof, err := hashesFrom(msg.WorkerNextProof)
	if err != nil {
		return nil, err
	}
	challengerNextProof, err := hashesFrom(msg.ChallengerNextProof)
	if err != nil {
		return nil, err
	}
	if !canonical.VerifyLeafInclusion(workerRoot,
		canonical.WideStateLeafV1(int(wd.TileI), int(wd.TileJ), int(d.High), workerHigh),
		canonical.MerkleProof{Index: d.High, Count: d.Steps, Siblings: workerNextProof}) {
		return nil, errors.New("worker next-state proof does not match the locked trace root")
	}
	if !canonical.VerifyLeafInclusion(challengerRoot,
		canonical.WideStateLeafV1(int(wd.TileI), int(wd.TileJ), int(d.High), challengerHigh),
		canonical.MerkleProof{Index: d.High, Count: d.Steps, Siblings: challengerNextProof}) {
		return nil, errors.New("challenger next-state proof does not match the locked trace root")
	}
	// operand evidence: the tiled part of the A rows followed by the tiled
	// part of the W rows; rows beyond the operand extent are canonical zero
	// padding and carry no evidence.
	aRowsN := wd.M - int(wd.TileI)*canonical.WideTileN
	if aRowsN > canonical.WideTileN {
		aRowsN = canonical.WideTileN
	}
	wRowsN := canonical.WideTileN // K-tile rows always exist (K % 8 == 0)
	if wd.TransposeB {
		wRowsN = wd.N - int(wd.TileJ)*canonical.WideTileN
		if wRowsN > canonical.WideTileN {
			wRowsN = canonical.WideTileN
		}
	}
	if aRowsN <= 0 || wRowsN <= 0 || len(msg.Evidence) != aRowsN+wRowsN {
		return nil, errors.New("wide arbitration evidence count does not match the tile geometry")
	}
	graphDispute, err := canonical.RestoreGraphDisputeV2(record.Snapshot, graph, ChallengeRoundBlocks)
	if err != nil {
		return nil, err
	}
	inputStateRoot := graphDispute.LowStateRoot()
	aRef := node.Inputs[0]
	wRef := node.Inputs[1]
	// The operand kinds decide which descriptor table is legal to index:
	// a real-block GEMM reads its activations from an upstream node output
	// (kind 1), so never touch graph.Inputs with that index.
	var aDesc, wDesc canonical.TensorDescriptorV2
	if aRef.Kind == 1 {
		aDesc = graph.Nodes[aRef.Index].Output
	} else {
		aDesc = graph.Inputs[aRef.Index].Desc
	}
	if wRef.Kind == 1 {
		wDesc = graph.Nodes[wRef.Index].Output
	} else {
		wDesc = graph.Inputs[wRef.Index].Desc
	}
	decodeRows := func(ref canonical.TensorRef, desc canonical.TensorDescriptorV2,
		rows []*types.GraphChunkEvidence, rowElems int, rowStart int) (canonical.WideTileState, error) {
		var tile canonical.WideTileState
		bpe, err := desc.BytesPerElem()
		if err != nil {
			return tile, err
		}
		for i := 0; i < canonical.WideTileN; i++ {
			if i >= len(rows) {
				break // canonical zero padding beyond the operand extent
			}
			ev := rows[i]
			if (canonical.TensorRef{Kind: uint8(ev.RefKind), Index: ev.RefIndex}) != ref {
				return tile, errors.New("operand evidence references the wrong tensor")
			}
			root, err := root32Of(ev.Root)
			if err != nil {
				return tile, err
			}
			var descJSON canonical.TensorDescriptorV2
			if err := json.Unmarshal(ev.DescJson, &descJSON); err != nil {
				return tile, errors.New("malformed evidence descriptor")
			}
			if !sameDescV2Local(descJSON, desc) {
				return tile, errors.New("operand evidence descriptor mismatch")
			}
			if ev.ChunkIndex != uint32((i*rowElems+rowStart)/canonical.ChunkElems) {
				return tile, errors.New("operand evidence chunk index does not match the geometry")
			}
			if !canonical.VerifyChunkV2(root, desc, ev.ChunkIndex, ev.Count, ev.Chunk, mustHashes(ev.Proof)) {
				return tile, errors.New("operand evidence chunk does not match its tensor root")
			}
			if err := verifyGraphEvidenceV2State(graph, wd.NodeID, inputStateRoot, ref, root, ev.StateProof); err != nil {
				return tile, err
			}
			s.consumeGraphGas(ctx, "wide evidence",
				GasGraphStateLeaf+GasGraphProofSibling*uint64(len(ev.StateProof)))
			// extract the 8 K-column values at offset (row*rowElems+rowStart)%64
			pos := (i*rowElems + rowStart) % canonical.ChunkElems
			if pos+canonical.WideKStep > canonical.ChunkElems {
				return tile, errors.New("K-window crosses a chunk boundary")
			}
			for d := 0; d < canonical.WideKStep; d++ {
				off := (pos + d) * bpe
				if off+bpe > len(ev.Chunk) {
					return tile, errors.New("operand evidence truncated")
				}
				tile[i*canonical.WideTileN+d] = decodeBE16(ev.Chunk[off:off+bpe])
			}
		}
		return tile, nil
	}
	// A rows: row r = tile_i*8 + i, K-window starts at step*8 within the row
	aTile, err := decodeRows(aRef, aDesc, msg.Evidence[:aRowsN], wd.K, int(step)*canonical.WideKStep)
	if err != nil {
		return nil, err
	}
	wRows := msg.Evidence[aRowsN:]
	var wTile canonical.WideTileState
	if wd.TransposeB {
		// W (N, K): evidence row j is n = tile_j*8 + j, K-window at step*8;
		// WideTileStep expects wTile[d][j], i.e. exactly that layout.
		wTile, err = decodeRows(wRef, wDesc, wRows, wd.K, int(step)*canonical.WideKStep)
	} else {
		// W (K, N): evidence row d is k = step*8 + d, N-window at tile_j*8,
		// giving wTile[d][j] directly.
		wTile, err = decodeRows(wRef, wDesc, wRows, wd.N, int(wd.TileJ)*canonical.WideTileN)
	}
	if err != nil {
		return nil, err
	}
	wideOutcome, _, err := canonical.WideArbitrate(int(wd.TileI), int(wd.TileJ), int(step),
		lowState, aTile, wTile, workerHigh, challengerHigh, true, true)
	if err != nil {
		// neither party's locked state matches the exact 512-MAC
		// recomputation: no certainty, no settlement (never a slash).
		return nil, err
	}
	outcome := wideOutcomeToGraphOutcome(wideOutcome)
	wd.Outcome = canonicalOutcomeName(outcome)
	wd.Status = WideStatusResolved
	if err := s.setWideDispute(ctx, wd); err != nil {
		return nil, err
	}
	record.TranscriptDigest, err = graphTranscriptStep(record.TranscriptDigest, "wide_512_arbitration",
		map[string]any{"node": wd.NodeID, "step": step, "outcome": uint64(outcome),
			"mac": canonical.Wide512MAC})
	if err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "wide arbiter witness", GasGraphArbiterUnit*canonical.Wide512MAC)
	if _, err := s.resolveGraphOutcome(ctx, &task, &record, outcome); err != nil {
		return nil, err
	}
	s.deleteWideDispute(ctx, task.ID)
	return &types.MsgArbitrateWide512Response{Outcome: canonicalOutcomeName(outcome)}, nil
}

// wideOutcomeToGraphOutcome maps the wide arbiter verdict into the shared
// settlement vocabulary.
func wideOutcomeToGraphOutcome(o canonical.WideOutcome) canonical.Outcome {
	switch o {
	case canonical.WideWorkerWins:
		return canonical.WorkerWins
	case canonical.WideChallengerWins:
		return canonical.ChallengerWins
	default:
		return canonical.BothInvalid
	}
}

func sameDescV2Local(a, b canonical.TensorDescriptorV2) bool {
	if a.Dtype != b.Dtype || a.Layout != b.Layout || len(a.Shape) != len(b.Shape) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	return true
}

func mustHashes(raw [][]byte) []canonical.Hash {
	out := make([]canonical.Hash, len(raw))
	for i, b := range raw {
		if len(b) == 32 {
			copy(out[i][:], b)
		}
	}
	return out
}

// decodeBE16 decodes a two's complement big-endian 2/4/8-byte value.
func decodeBE16(b []byte) int64 {
	var v int64
	for _, x := range b {
		v = v<<8 | int64(x)
	}
	width := len(b) * 8
	if width < 64 {
		shift := uint(64 - width)
		v = v << shift >> shift
	}
	return v
}

// hashOfHexLocal decodes a 32-byte hex hash (the wide state machine stores
// roots as hex JSON strings).
func hashOfHexLocal(s string) (canonical.Hash, error) {
	var h canonical.Hash
	b, err := hex.DecodeString(s)
	if err != nil || len(b) != 32 {
		return h, errors.New("malformed hex hash")
	}
	copy(h[:], b)
	return h, nil
}
