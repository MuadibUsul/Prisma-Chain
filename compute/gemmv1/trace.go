package gemmv1

// On-demand tile trace. Under v0.1.1 a trace is created ONLY for one
// disputed output tile after a valid ChallengeOpen. It is never built for
// the whole output matrix, so normal-path storage stays bounded by the
// output commitment itself.

import "errors"

// TileTraceArtifacts is one party's local trace for a disputed tile.
type TileTraceArtifacts struct {
	States []State // S0..SR, length R+1
	Levels [][]Hash
	Root   Hash
}

// BuildTileTrace generates S0..SR for C_tile[i][j] where
// S0 is the zero matrix and S(r+1) = Sr + A_tile[i][r] x B_tile[r][j].
// S_R equals the honest output tile for that position.
func BuildTileTrace(a, b []int8, m, n, k uint64, taskID, assignmentID []byte, tileI, tileJ uint32) (*TileTraceArtifacts, error) {
	if err := CheckInputSizes(m, n, k, a, b); err != nil {
		return nil, err
	}
	rSteps := RSteps(k)
	states := make([]State, rSteps+1)
	s := State{}
	states[0] = s
	for r := uint64(0); r < rSteps; r++ {
		aTile := ExtractATile(a, m, k, uint64(tileI), r)
		bTile := ExtractBTile(b, k, n, r, uint64(tileJ))
		s = MicroStep(&s, aTile, bTile)
		states[r+1] = s
	}
	levels, err := buildLevels(traceLeaves(taskID, assignmentID, tileI, tileJ, states))
	if err != nil {
		return nil, err
	}
	return &TileTraceArtifacts{
		States: states,
		Levels: levels,
		Root:   levels[len(levels)-1][0],
	}, nil
}

func traceLeaves(taskID, assignmentID []byte, tileI, tileJ uint32, states []State) []Hash {
	leaves := make([]Hash, len(states))
	for step, s := range states {
		leaves[step] = LeafTraceState(taskID, assignmentID, tileI, tileJ, uint32(step), s.CanonicalBytes())
	}
	return leaves
}

// TraceRoot returns the trace root over S0..SR.
func (art *TileTraceArtifacts) TraceRoot() Hash { return art.Root }

// BuildTraceCommit produces the signed wire object that locks a party's
// trace. After both parties lock, the trace cannot change.
func BuildTraceCommit(t *TaskDescriptor, assignmentID []byte, party Party, tileI, tileJ uint32, art *TileTraceArtifacts, lockedEpoch uint64) (*TraceCommit, error) {
	taskID, err := t.TaskID()
	if err != nil {
		return nil, err
	}
	rSteps := uint32(RSteps(t.K))
	if len(art.States) != int(rSteps)+1 {
		return nil, errors.New("gemmv1: trace length does not match R")
	}
	initialProof, err := ProveLeaf(art.Levels, 0)
	if err != nil {
		return nil, err
	}
	finalProof, err := ProveLeaf(art.Levels, rSteps)
	if err != nil {
		return nil, err
	}
	return &TraceCommit{
		ProtocolVersion: ProtocolVersion,
		TaskID:          taskID,
		AssignmentID:    append([]byte(nil), assignmentID...),
		DisputedTileI:   uint64(tileI),
		DisputedTileJ:   uint64(tileJ),
		Party:           uint64(party),
		TraceRoot:       append([]byte(nil), art.Root[:]...),
		InitialState:    art.States[0].CanonicalBytes(),
		InitialProof:    initialProof,
		FinalState:      art.States[rSteps].CanonicalBytes(),
		FinalProof:      finalProof,
		LockedEpoch:     lockedEpoch,
	}, nil
}

// VerifyTraceCommit is the coordinator-side check that a locked trace is
// internally consistent: S0 is the zero matrix, both endpoints carry valid
// inclusion proofs against the committed root, and coordinates match.
func VerifyTraceCommit(t *TaskDescriptor, assignmentID []byte, tc *TraceCommit) error {
	if tc.ProtocolVersion != ProtocolVersion {
		return errVersion
	}
	taskID, err := t.TaskID()
	if err != nil {
		return err
	}
	if !equalBytes(tc.TaskID, taskID) || !equalBytes(tc.AssignmentID, assignmentID) {
		return errors.New("gemmv1: trace commit is not bound to this task/assignment")
	}
	if tc.DisputedTileI >= tileRows(t.M) || tc.DisputedTileJ >= tileCols(t.N) {
		return errBadChallengeTile
	}
	if tc.Party != uint64(Worker) && tc.Party != uint64(Challenger) {
		return errors.New("gemmv1: trace commit party must be worker or challenger")
	}
	var root Hash
	if err := checkHashField("trace_root", tc.TraceRoot); err != nil {
		return err
	}
	copy(root[:], tc.TraceRoot)
	initial, err := StateFromCanonical(tc.InitialState)
	if err != nil {
		return err
	}
	if !initial.IsZero() {
		return errors.New("gemmv1: S0 of a locked trace must be the zero matrix")
	}
	final, err := StateFromCanonical(tc.FinalState)
	if err != nil {
		return err
	}
	rSteps := uint32(RSteps(t.K))
	leafInitial := LeafTraceState(taskID, assignmentID, uint32(tc.DisputedTileI), uint32(tc.DisputedTileJ), 0, initial.CanonicalBytes())
	leafFinal := LeafTraceState(taskID, assignmentID, uint32(tc.DisputedTileI), uint32(tc.DisputedTileJ), rSteps, final.CanonicalBytes())
	if !VerifyLeafInclusion(root, leafInitial, tc.InitialProof) {
		return errors.New("gemmv1: invalid S0 inclusion proof")
	}
	if !VerifyLeafInclusion(root, leafFinal, tc.FinalProof) {
		return errors.New("gemmv1: invalid S_R inclusion proof")
	}
	return nil
}
