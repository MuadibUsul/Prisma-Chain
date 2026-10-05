package canonical

// A1-07: V2 typed arbitration tests over a real requant node (the widest
// cheap-op family: 126 of the 310 nodes) and one Q12.20 operator.

import (
	"bytes"
	"testing"
)

func requantNodeV2() GraphNodeV2 {
	return GraphNodeV2{
		NodeID: 1, OperatorID: OpRequantizeWideV1, Version: OpRequantVersionWide,
		Inputs: []TensorRef{{Kind: 0, Index: 0}},
		Output: NewDescV2(DtypeV2A13, 2, 3),
		Params: ParamList{
			{Key: "clamp_hi", Value: A13Max}, {Key: "clamp_lo", Value: A13Min},
			{Key: "mult", Value: 1 << 20}, {Key: "out_dtype", Value: int64(DtypeV2A13)},
			{Key: "shift", Value: 20},
		},
	}
}

func TestArbitrateRequantV2(t *testing.T) {
	node := requantNodeV2()
	// input: an INT64_ACCUM tensor (the real GEMM accumulator shape)
	inData := []int64{4, -4, 1 << 20, -(1 << 20), 7, -7}
	inT := mustTensorV2(t, NewDescV2(DtypeV2Int64Accum, 2, 3), inData)
	inRoot, err := inT.TensorRootV2()
	if err != nil {
		t.Fatal(err)
	}
	// honest output (mult=2^20, shift=20 -> identity, clamped to A13)
	outData := []int64{4, -4, 4095, -4096, 7, -7}
	outT := mustTensorV2(t, node.Output, outData)
	outRoot, err := outT.TensorRootV2()
	if err != nil {
		t.Fatal(err)
	}
	outChunk, outProof, err := outT.ChunkProofV2(0)
	if err != nil {
		t.Fatal(err)
	}
	// fraudulent output: one element altered
	badData := append([]int64(nil), outData...)
	badData[2]--  // in-range corruption (4095 -> 4094)
	badT := mustTensorV2(t, node.Output, badData)
	badRoot, err := badT.TensorRootV2()
	if err != nil {
		t.Fatal(err)
	}
	badChunk, badProof, err := badT.ChunkProofV2(0)
	if err != nil {
		t.Fatal(err)
	}
	inChunk, inProof, err := inT.ChunkProofV2(0)
	if err != nil {
		t.Fatal(err)
	}
	evidence := []ChunkEvidenceV2{{
		Ref: TensorRef{Kind: 0, Index: 0}, Desc: inT.Desc, Root: inRoot,
		ChunkIndex: 0, Count: 1, Bytes: inChunk, Proof: inProof,
	}}
	claims := func(wRoot, cRoot Hash, wChunk, cChunk []byte, wP, cP []Hash) (NodeChunkClaim, NodeChunkClaim) {
		return NodeChunkClaim{OutRoot: wRoot, ChunkIndex: 0, ChunkBytes: wChunk, Proof: wP},
			NodeChunkClaim{OutRoot: cRoot, ChunkIndex: 0, ChunkBytes: cChunk, Proof: cP}
	}
	worker, challenger := claims(badRoot, outRoot, badChunk, outChunk, badProof, outProof)
	outcome, err := ArbitrateNodeChunkV2(smallWideGraph(t), node, Hash{},
		worker, challenger, evidence, nil)
	if err != nil {
		t.Fatal(err)
	}
	if outcome != ChallengerWins {
		t.Fatalf("outcome %v, want ChallengerWins", outcome)
	}
	// roles swapped: the worker holding the honest chunk wins
	worker, challenger = claims(outRoot, badRoot, outChunk, badChunk, outProof, badProof)
	outcome, err = ArbitrateNodeChunkV2(smallWideGraph(t), node, Hash{},
		worker, challenger, evidence, nil)
	if err != nil || outcome != WorkerWins {
		t.Fatalf("outcome %v err %v, want WorkerWins", outcome, err)
	}
	// both wrong: BothInvalid
	worker, challenger = claims(badRoot, badRoot, badChunk, badChunk, badProof, badProof)
	if _, err := ArbitrateNodeChunkV2(smallWideGraph(t), node, Hash{},
		worker, challenger, evidence, nil); err == nil {
		t.Fatal("identical chunks accepted")
	}
	// corrupt evidence chunk is refused
	tampered := append([]byte(nil), inChunk...)
	tampered[0] ^= 1
	badEvidence := []ChunkEvidenceV2{{
		Ref: TensorRef{Kind: 0, Index: 0}, Desc: inT.Desc, Root: inRoot,
		ChunkIndex: 0, Count: 1, Bytes: tampered, Proof: inProof,
	}}
	worker, challenger = claims(badRoot, outRoot, badChunk, outChunk, badProof, outProof)
	if _, err := ArbitrateNodeChunkV2(smallWideGraph(t), node, Hash{},
		worker, challenger, badEvidence, nil); err == nil {
		t.Fatal("tampered evidence accepted")
	}
	// wide GEMM nodes are refused here (they belong to the wide dispute path)
	gemmNode := GraphNodeV2{NodeID: 0, OperatorID: OpGEMMWideA13W10, Version: OpGEMMVersionWide,
		Inputs: []TensorRef{{Kind: 0, Index: 0}, {Kind: 0, Index: 1}},
		Output: NewDescV2(DtypeV2Int64Accum, 2, 2), Params: ParamList{}}
	worker, challenger = claims(badRoot, outRoot, badChunk, outChunk, badProof, outProof)
	if _, err := ArbitrateNodeChunkV2(smallWideGraph(t), gemmNode, Hash{},
		worker, challenger, evidence, nil); err == nil {
		t.Fatal("wide GEMM node accepted by the cheap-op arbiter")
	}
}

func TestArbitrateAddV2SharesV1Arithmetic(t *testing.T) {
	// a Q12.20 ADD node recomputes through the frozen V1 value window
	node := GraphNodeV2{NodeID: 0, OperatorID: OpAddFixedV1, Version: VersionFxFusion,
		Inputs: []TensorRef{{Kind: 0, Index: 0}, {Kind: 0, Index: 1}},
		Output: NewDescV2(DtypeV2Q12_20, 70), Params: ParamList{}}
	aData := make([]int64, 70)
	bData := make([]int64, 70)
	for i := range aData {
		aData[i] = int64(i) * 1000
		bData[i] = -int64(i) * 999
	}
	aT := mustTensorV2(t, NewDescV2(DtypeV2Q12_20, 70), aData)
	bT := mustTensorV2(t, NewDescV2(DtypeV2Q12_20, 70), bData)
	outData := make([]int64, 70)
	for i := range outData {
		outData[i] = clampInt64(aData[i]+bData[i], MinFx, MaxFx)
	}
	outT := mustTensorV2(t, node.Output, outData)
	outRoot, _ := outT.TensorRootV2()
	aRoot, _ := aT.TensorRootV2()
	bRoot, _ := bT.TensorRootV2()
	chunk, proof, err := outT.ChunkProofV2(1)
	if err != nil {
		t.Fatal(err)
	}
	bad := append([]byte(nil), chunk...)
	bad[3] ^= 0x40
	badT := mustTensorV2(t, node.Output, func() []int64 {
		d := append([]int64(nil), outData...)
		d[64+3] ^= 0x40
		return d
	}())
	badRoot, _ := badT.TensorRootV2()
	badChunk, badProof, _ := badT.ChunkProofV2(1)
	ev := func(tensor *TensorV2, root Hash, idx uint32) ChunkEvidenceV2 {
		c, p, err := tensor.ChunkProofV2(int(idx))
		if err != nil {
			t.Fatal(err)
		}
		return ChunkEvidenceV2{Ref: TensorRef{Kind: 0, Index: 0}, Desc: tensor.Desc,
			Root: root, ChunkIndex: idx, Count: uint32(tensor.ChunkCount()), Bytes: c, Proof: p}
	}
	evidence := []ChunkEvidenceV2{
		ev(aT, aRoot, 1),
		ev(bT, bRoot, 1),
	}
	_ = proof
	outcome, err := ArbitrateNodeChunkV2(smallWideGraph(t), node, Hash{},
		NodeChunkClaim{OutRoot: outRoot, ChunkIndex: 1, ChunkBytes: chunk, Proof: proof},
		NodeChunkClaim{OutRoot: badRoot, ChunkIndex: 1, ChunkBytes: badChunk, Proof: badProof},
		evidence, nil)
	if err != nil {
		t.Fatal(err)
	}
	if outcome != WorkerWins {
		t.Fatalf("outcome %v, want WorkerWins (honest first claim)", outcome)
	}
	if bytes.Equal(chunk, badChunk) {
		t.Fatal("test chunks collided")
	}
}

