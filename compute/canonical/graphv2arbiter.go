package canonical

// A1-07: V2 typed cheap-operator arbitration.  Every claim and every
// evidence blob is verified against TensorRootV2 under the V2 domains;
// the operator recompute is bounded exactly like the V1 arbiter.
//
// Wide GEMM nodes never reach this path: they are adjudicated by
// WIDE_GEMM_DISPUTE_V1 (A2).  A requant node is recomputed element-wise
// (its input may be an INT64_ACCUM chunk); the Q12.20 operators share the
// frozen V1 value-window arithmetic through a typed view.

import (
	"errors"
	"fmt"
)

// ChunkEvidenceV2 couples one typed input chunk with the identity and
// membership proof that binds it to the committed V2 input state.
type ChunkEvidenceV2 struct {
	Ref        TensorRef
	Desc       TensorDescriptorV2
	Root       Hash
	StateProof []byte
	ChunkIndex uint32
	Count      uint32
	Bytes      []byte
	Proof      []Hash
}

// StateProofV2 derives the V2 state root and one tensor's membership
// proof from the live tensor roots (used by watchers and the chain to
// assemble/verify arbiter evidence).
func StateProofV2(live map[TensorRef]Hash, ref TensorRef) (root Hash, index, count uint32, siblings []Hash, err error) {
	if len(live) == 0 {
		return Hash{}, 0, 0, nil, errors.New("canonical/v2: empty state")
	}
	ids := make([]uint64, 0, len(live))
	byID := map[uint64]TensorRef{}
	for r := range live {
		id := uint64(r.Kind)<<32 | uint64(r.Index)
		ids = append(ids, id)
		byID[id] = r
	}
	sortUint64s(ids)
	leaves := make([]Hash, len(ids))
	wantID := uint64(ref.Kind)<<32 | uint64(ref.Index)
	found := -1
	for i, id := range ids {
		r := byID[id]
		leaves[i] = StateLeafV2(r.Kind, r.Index, live[r])
		if id == wantID {
			found = i
		}
	}
	if found < 0 {
		return Hash{}, 0, 0, nil, errors.New("canonical/v2: tensor not part of the state")
	}
	levels, err := buildLevels(leaves)
	if err != nil {
		return Hash{}, 0, 0, nil, err
	}
	siblings, err = proveLeaf(levels, found)
	if err != nil {
		return Hash{}, 0, 0, nil, err
	}
	return levels[len(levels)-1][0], uint32(found), uint32(len(ids)), siblings, nil
}

func decodeBEInt(b []byte) int64 {
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

func encodeBEInt(v int64, width int) []byte {
	out := make([]byte, width)
	for i := width - 1; i >= 0; i-- {
		out[i] = byte(v)
		v >>= 8
	}
	return out
}

// recomputeChunkV2 is the bounded V2 operator arbiter.
func recomputeChunkV2(node GraphNodeV2, evidence []ChunkEvidenceV2, ropeTable *RopeConstants,
	index uint32) ([]byte, error) {
	switch node.OperatorID {
	case OpRequantizeWideV1:
		if len(evidence) != 1 {
			return nil, errors.New("canonical/v2: requant needs one input chunk")
		}
		in := evidence[0]
		inBpe, err := in.Desc.BytesPerElem()
		if err != nil {
			return nil, err
		}
		outBpe, err := node.Output.BytesPerElem()
		if err != nil {
			return nil, err
		}
		if len(in.Bytes) != ChunkElems*inBpe {
			return nil, errors.New("canonical/v2: requant input chunk width mismatch")
		}
		mult := node.Params.Get("mult", 0)
		shift := uint(node.Params.Get("shift", 0))
		lo := node.Params.Get("clamp_lo", MinFx)
		hi := node.Params.Get("clamp_hi", MaxFx)
		out := make([]byte, ChunkElems*outBpe)
		for j := 0; j < ChunkElems; j++ {
			vin := decodeBEInt(in.Bytes[j*inBpe : (j+1)*inBpe])
			prod := vin * mult
			if vin != 0 && prod/vin != mult {
				return nil, fmt.Errorf("canonical/v2: requant intermediate overflow at %d", j)
			}
			res := clampInt64(RShiftRoundEven64(prod, shift), lo, hi)
			copy(out[j*outBpe:(j+1)*outBpe], encodeBEInt(res, outBpe))
		}
		return out, nil
	default:
		// Q12.20 operators: the value window arithmetic is exactly the
		// frozen V1 arbiter's; only the typed container differs.
		for _, ev := range evidence {
			if ev.Desc.Dtype != DtypeV2Q12_20 {
				return nil, fmt.Errorf("canonical/v2: operator %s needs Q12.20 evidence", node.OperatorID)
			}
		}
		v1node := GraphNode{NodeID: node.NodeID, OperatorID: node.OperatorID, Version: node.Version,
			Output: NewDesc(DtypeQ12_20, node.Output.Shape...), Params: node.Params}
		v1ev := make([]ChunkEvidence, len(evidence))
		for i, ev := range evidence {
			v1ev[i] = ChunkEvidence{Ref: ev.Ref, Desc: NewDesc(DtypeQ12_20, ev.Desc.Shape...),
				Bytes: ev.Bytes}
		}
		return recomputeChunk(v1node, v1ev, ropeTable, index)
	}
}

// ArbitrateNodeChunkV2 adjudicates the disputed chunk of one V2 node from
// typed evidence.  Bounded exactly like the V1 arbiter; the chain never
// recomputes the node or the block.
func ArbitrateNodeChunkV2(graph *GraphDescriptorV2, node GraphNodeV2, inputStateRoot Hash,
	worker, challenger NodeChunkClaim, inputEvidence []ChunkEvidenceV2,
	ropeTable *RopeConstants) (Outcome, error) {
	if err := graph.ValidateV2(); err != nil {
		return Pending, err
	}
	if node.OperatorID == OpGEMMWideA13W10 {
		return Pending, errors.New("canonical/v2: wide GEMM nodes are adjudicated by WIDE_GEMM_DISPUTE_V1")
	}
	if _, err := LookupV2(node.OperatorID); err != nil {
		return Pending, err
	}
	outBpe, err := node.Output.BytesPerElem()
	if err != nil {
		return Pending, err
	}
	elems, _ := node.Output.Elems()
	count := uint32((elems + ChunkElems - 1) / ChunkElems)
	if count == 0 {
		count = 1
	}
	if len(worker.ChunkBytes) != ChunkElems*outBpe || len(challenger.ChunkBytes) != ChunkElems*outBpe {
		return Pending, errors.New("canonical/v2: chunk width does not match the output dtype")
	}
	if !VerifyChunkV2(worker.OutRoot, node.Output, worker.ChunkIndex, count, worker.ChunkBytes, worker.Proof) {
		return Pending, errBadChunkProof
	}
	if !VerifyChunkV2(challenger.OutRoot, node.Output, challenger.ChunkIndex, count, challenger.ChunkBytes, challenger.Proof) {
		return Pending, errBadChunkProof
	}
	if worker.ChunkIndex != challenger.ChunkIndex {
		return Pending, errors.New("canonical/v2: parties must dispute the same chunk position")
	}
	if equalBytes(worker.ChunkBytes, challenger.ChunkBytes) {
		return Pending, errSameChunk
	}
	for _, ev := range inputEvidence {
		if !VerifyChunkV2(ev.Root, ev.Desc, ev.ChunkIndex, ev.Count, ev.Bytes, ev.Proof) {
			return Pending, errors.New("canonical/v2: input evidence does not match its tensor root")
		}
	}
	expected, err := recomputeChunkV2(node, inputEvidence, ropeTable, worker.ChunkIndex)
	if err != nil {
		return Pending, err
	}
	workerOK := equalBytes(expected, worker.ChunkBytes)
	challengerOK := equalBytes(expected, challenger.ChunkBytes)
	switch {
	case workerOK:
		return WorkerWins, nil
	case challengerOK:
		return ChallengerWins, nil
	default:
		return BothInvalid, nil
	}
}
