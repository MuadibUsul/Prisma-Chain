package canonical

// TransformerBlockV1 tests: macro expansion structure, deterministic
// execution, independent verification of the transpose-B GEMM form,
// per-node fraud localization on the real block graph, and bounded
// arbitration including the GEMM full-operand path.

import (
	"testing"

	"prismachain/compute/gemmv1"
)

func miniBlockConfig() BlockConfig {
	return BlockConfig{Seq: 16, DModel: 128, Heads: 4, HeadDim: 32, MLPHidden: 256}
}

func mediumBlockConfig() BlockConfig {
	return BlockConfig{Seq: 64, DModel: 512, Heads: 8, HeadDim: 64, MLPHidden: 2048}
}

// testQuantPlan is the plumbing test's static quantization: 5 fractional
// bits (int8 = q12.20 >> 15, range about +/-4.0) with the dequant
// multipliers that inversion implies. Real deployments use calibration
// scales; the arithmetic contract is identical.
func testQuantPlan(t *testing.T, cfg BlockConfig) BlockQuant {
	t.Helper()
	scale, err := AttentionScaleFx(cfg.HeadDim)
	if err != nil {
		t.Fatal(err)
	}
	narrow := RequantSpec{Mult: 1, Shift: 15, Lo: -127, Hi: 127, OutInt8: true}
	fx := func(mult, shift int64) RequantSpec {
		return RequantSpec{Mult: mult, Shift: shift, Lo: MinFx, Hi: MaxFx}
	}
	return BlockQuant{
		NormEpsFx:       10,
		Norm1ToInt8:     narrow,
		Norm2ToInt8:     narrow,
		QKAccumToFx:     fx(1024, 0),
		QKRopeToInt8:    narrow,
		VAccumToInt8:    RequantSpec{Mult: 1, Shift: 5, Lo: -127, Hi: 127, OutInt8: true},
		ScoresAccumToFx: fx(1024*scale, 20),
		SoftmaxToInt8:   RequantSpec{Mult: 1, Shift: 15, Lo: 0, Hi: 127, OutInt8: true},
		CtxAccumToInt8:  RequantSpec{Mult: 1, Shift: 5, Lo: -127, Hi: 127, OutInt8: true},
		ProjAccumToFx:   fx(1024, 0),
		GateAccumToFx:   fx(1024, 0),
		UpAccumToFx:     fx(1024, 0),
		MlpMulToInt8:    narrow,
		DownAccumToFx:   fx(1024, 0),
	}
}

type lcg struct{ s uint64 }

func (r *lcg) next() uint64 {
	r.s = r.s*6364136223846793005 + 1442695040888963407
	return r.s
}

func (r *lcg) inRange(lo, hi int64) int64 {
	return lo + int64(r.next()>>33)%(hi-lo+1)
}

func randomTensor(r *lcg, desc TensorDescriptor, lo, hi int64) Tensor {
	elems, _ := desc.Elems()
	data := make([]int32, elems)
	for i := range data {
		data[i] = int32(r.inRange(lo, hi))
	}
	return Tensor{Desc: desc, Data: data}
}

type blockFixture struct {
	cfg    BlockConfig
	graph  *GraphDescriptor
	inputs map[uint32]Tensor
	tables map[uint32]*RopeConstants
	exec   *GraphExecution
}

func buildBlockFixture(t *testing.T, cfg BlockConfig) *blockFixture {
	t.Helper()
	if err := cfg.Validate(); err != nil {
		t.Fatal(err)
	}
	r := &lcg{s: 0xC0FFEE1234567}
	d, hd, mlp := int64(cfg.DModel), int64(cfg.HeadDim), int64(cfg.MLPHidden)

	x := randomTensor(r, NewDesc(DtypeQ12_20, int64(cfg.Seq), d), -(1 << 18), 1<<18)
	n1 := randomTensor(r, NewDesc(DtypeQ12_20, d), 1<<20, 1<<20) // weight 1.0
	n2 := randomTensor(r, NewDesc(DtypeQ12_20, d), 1<<20, 1<<20)
	ropeConst, ropeTensor, _ := buildRopeTable(t, cfg.Seq, cfg.HeadDim/2)

	inputs := map[uint32]Tensor{0: x, 1: n1, 2: n2, 3: ropeTensor}
	var set BlockInputSet
	set.X = GraphInput{Desc: x.Desc, Root: mustRoot(t, x)}
	set.Norm1 = GraphInput{Desc: n1.Desc, Root: mustRoot(t, n1)}
	set.Norm2 = GraphInput{Desc: n2.Desc, Root: mustRoot(t, n2)}
	set.RopeTable = GraphInput{Desc: ropeTensor.Desc, Root: mustRoot(t, ropeTensor)}

	idx := uint32(4)
	addWeight := func(name string, shape ...int64) GraphInput {
		desc := NewDesc(DtypeInt8, shape...)
		w := randomTensor(r, desc, -16, 16)
		inputs[idx] = w
		idx++
		return GraphInput{Desc: desc, Root: mustRoot(t, w)}
	}
	for h := 0; h < cfg.Heads; h++ {
		q := addWeight("wq", d, hd)
		k := addWeight("wk", d, hd)
		v := addWeight("wv", d, hd)
		o := addWeight("wo", hd, d)
		set.WQ = append(set.WQ, q)
		set.WK = append(set.WK, k)
		set.WV = append(set.WV, v)
		set.WO = append(set.WO, o)
	}
	set.WGate = addWeight("wgate", d, mlp)
	set.WUp = addWeight("wup", d, mlp)
	set.WDown = addWeight("wdown", mlp, d)

	graph, err := BuildTransformerBlockV1(cfg, set, testQuantPlan(t, cfg))
	if err != nil {
		t.Fatal(err)
	}
	tables := map[uint32]*RopeConstants{3: ropeConst}
	exec, err := ExecuteGraph(graph, inputs, tables)
	if err != nil {
		t.Fatal(err)
	}
	return &blockFixture{cfg: cfg, graph: graph, inputs: inputs, tables: tables, exec: exec}
}

func mustRoot(t *testing.T, tens Tensor) []byte {
	t.Helper()
	root, err := tens.TensorRoot()
	if err != nil {
		t.Fatal(err)
	}
	return root[:]
}

// nodeCounts counts nodes per operator for structure assertions.
func nodeCounts(graph *GraphDescriptor) map[string]int {
	counts := map[string]int{}
	for _, n := range graph.Nodes {
		counts[n.OperatorID]++
	}
	return counts
}

func expectedNodeCount(cfg BlockConfig) int {
	// attn: norm + req (2) + per head 18 + head adds (H-1) + residual (1)
	// mlp: norm + req + gate GEMM/REQ/SILU + up GEMM/REQ + MUL + REQ +
	//      down GEMM/REQ + residual ADD (12)
	return 2 + cfg.Heads*18 + (cfg.Heads - 1) + 1 + 12
}

func TestTransformerBlockMiniStructure(t *testing.T) {
	cfg := miniBlockConfig()
	fix := buildBlockFixture(t, cfg)
	graph := fix.graph

	if graph.Spec != SpecTransformerBlockV1 {
		t.Fatalf("spec = %q", graph.Spec)
	}
	wantNodes := expectedNodeCount(cfg)
	if len(graph.Nodes) != wantNodes {
		t.Fatalf("node count = %d, want %d", len(graph.Nodes), wantNodes)
	}
	if wantInputs := 4 + 4*cfg.Heads + 3; len(graph.Inputs) != wantInputs {
		t.Fatalf("input count = %d, want %d", len(graph.Inputs), wantInputs)
	}
	if len(graph.Outputs) != 1 {
		t.Fatalf("output count = %d", len(graph.Outputs))
	}
	counts := nodeCounts(graph)
	wantGEMM := 6*cfg.Heads + 3 // QKV, scores, ctx, proj per head + gate/up/down
	if counts[OpGEMMInt8V1] != wantGEMM {
		t.Fatalf("GEMM nodes = %d, want %d", counts[OpGEMMInt8V1], wantGEMM)
	}
	if counts[OpRoPEFixedV1] != 2*cfg.Heads {
		t.Fatalf("ROPE nodes = %d, want %d", counts[OpRoPEFixedV1], 2*cfg.Heads)
	}

	// GraphID is deterministic and binds every committed root.
	id1, err := graph.GraphID()
	if err != nil {
		t.Fatal(err)
	}
	id2, _ := fix.graph.GraphID()
	if id1 != id2 {
		t.Fatal("GraphID is not deterministic")
	}
}

func TestTransformerBlockGraphIDBindsRoots(t *testing.T) {
	cfg := miniBlockConfig()
	r := &lcg{s: 7}
	mk := func() BlockInputSet {
		r = &lcg{s: 7}
		desc := func(dtype Dtype, shape ...int64) TensorDescriptor { return NewDesc(dtype, shape...) }
		tens := func(desc TensorDescriptor, lo, hi int64) GraphInput {
			w := randomTensor(r, desc, lo, hi)
			root, err := w.TensorRoot()
			if err != nil {
				t.Fatal(err)
			}
			return GraphInput{Desc: desc, Root: root[:]}
		}
		d, hd, mlp := int64(cfg.DModel), int64(cfg.HeadDim), int64(cfg.MLPHidden)
		set := BlockInputSet{
			X:         tens(desc(DtypeQ12_20, int64(cfg.Seq), d), -(1 << 18), 1<<18),
			Norm1:     tens(desc(DtypeQ12_20, d), 1<<20, 1<<20),
			Norm2:     tens(desc(DtypeQ12_20, d), 1<<20, 1<<20),
			RopeTable: tens(desc(DtypeQ12_20, int64(cfg.Seq)*hd), 0, 0),
			WGate:     tens(desc(DtypeInt8, d, mlp), -16, 16),
			WUp:       tens(desc(DtypeInt8, d, mlp), -16, 16),
			WDown:     tens(desc(DtypeInt8, mlp, d), -16, 16),
		}
		for h := 0; h < cfg.Heads; h++ {
			set.WQ = append(set.WQ, tens(desc(DtypeInt8, d, hd), -16, 16))
			set.WK = append(set.WK, tens(desc(DtypeInt8, d, hd), -16, 16))
			set.WV = append(set.WV, tens(desc(DtypeInt8, d, hd), -16, 16))
			set.WO = append(set.WO, tens(desc(DtypeInt8, hd, d), -16, 16))
		}
		_ = r
		return set
	}
	set1 := mk()
	g1, err := BuildTransformerBlockV1(cfg, set1, testQuantPlan(t, cfg))
	if err != nil {
		t.Fatal(err)
	}
	id1, _ := g1.GraphID()

	// Same roots again: identical GraphID.
	g2, err := BuildTransformerBlockV1(cfg, mk(), testQuantPlan(t, cfg))
	if err != nil {
		t.Fatal(err)
	}
	id2, _ := g2.GraphID()
	if id1 != id2 {
		t.Fatal("same inputs produced different GraphIDs")
	}

	// Flip one committed weight root: the identity must change.
	set3 := mk()
	set3.WGate.Root = append([]byte(nil), set3.WGate.Root...)
	set3.WGate.Root[0] ^= 0x01
	g3, err := BuildTransformerBlockV1(cfg, set3, testQuantPlan(t, cfg))
	if err != nil {
		t.Fatal(err)
	}
	id3, _ := g3.GraphID()
	if id1 == id3 {
		t.Fatal("GraphID did not change with a tampered weight root")
	}
}

func TestTransformerBlockMiniExecution(t *testing.T) {
	cfg := miniBlockConfig()
	fix := buildBlockFixture(t, cfg)
	exec := fix.exec

	// Deterministic replay.
	again, err := ExecuteGraph(fix.graph, fix.inputs, fix.tables)
	if err != nil {
		t.Fatal(err)
	}
	if len(exec.Outputs) != 1 || exec.Outputs[0] != again.Outputs[0] {
		t.Fatal("block execution is not deterministic")
	}
	if len(exec.Trail) != len(fix.graph.Nodes)+1 {
		t.Fatalf("trail length = %d", len(exec.Trail))
	}

	// Work vector: GEMM MACs exactly as derived from the static shapes.
	perHead := int64(cfg.Seq)*int64(cfg.HeadDim)*int64(cfg.DModel)*3 +
		int64(cfg.Seq)*int64(cfg.Seq)*int64(cfg.HeadDim)*2 +
		int64(cfg.Seq)*int64(cfg.DModel)*int64(cfg.HeadDim)
	mlp := int64(cfg.Seq)*int64(cfg.DModel)*int64(cfg.MLPHidden)*2 +
		int64(cfg.Seq)*int64(cfg.MLPHidden)*int64(cfg.DModel)
	wantMAC := perHead*int64(cfg.Heads) + mlp
	if got := exec.WorkVector.Get("GEMM_MAC", -1); got != wantMAC {
		t.Fatalf("GEMM_MAC = %d, want %d", got, wantMAC)
	}
	if exec.WorkVector.Get("SOFTMAX_EXP", -1) != int64(cfg.Seq*cfg.Seq*cfg.Heads) {
		t.Fatalf("SOFTMAX_EXP = %d", exec.WorkVector.Get("SOFTMAX_EXP", -1))
	}

	// The transpose-B GEMM form must equal the direct dot-product
	// definition C[i,j] = sum_d A[i,d] * B[j,d].
	checked := 0
	for _, node := range fix.graph.Nodes {
		if node.OperatorID != OpGEMMInt8V1 || !TransposeB(node.Params) {
			continue
		}
		a := exec.Tensors[node.Inputs[0]]
		b := exec.Tensors[node.Inputs[1]]
		out := exec.Tensors[TensorRef{Kind: 1, Index: node.NodeID}]
		m := int(node.Output.Shape[0])
		n := int(node.Output.Shape[1])
		k := int(a.Desc.Shape[1])
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var sum int64
				for d := 0; d < k; d++ {
					sum += int64(a.Data[i*k+d]) * int64(b.Data[j*k+d])
				}
				if int32(sum) != out.Data[i*n+j] {
					t.Fatalf("transpose-B GEMM mismatch at (%d,%d)", i, j)
				}
			}
		}
		checked++
	}
	if checked != cfg.Heads {
		t.Fatalf("checked %d transpose-B GEMM nodes, want %d", checked, cfg.Heads)
	}
}

// bisectToNode runs the dispute session to arbitration readiness and
// returns the localized first divergent node.
func bisectToNode(t *testing.T, graph *GraphDescriptor, workerTrail, honestTrail []Hash) uint32 {
	t.Helper()
	graphID, err := graph.GraphID()
	if err != nil {
		t.Fatal(err)
	}
	cfg := GraphDisputeConfig{Graph: graph, GraphID: graphID, RoundPeriod: 50}
	dispute, err := NewGraphDispute(cfg, trailClaim(t, graphID, workerTrail), trailClaim(t, graphID, honestTrail), 1000)
	if err != nil {
		t.Fatal(err)
	}
	rounds := 0
	for !dispute.ArbReady() {
		mid := dispute.low + (dispute.high-dispute.low)/2
		wLevels, _ := TrailLevels(graphID, workerTrail)
		cLevels, _ := TrailLevels(graphID, honestTrail)
		wp, err := proveLeaf(wLevels, int(mid))
		if err != nil {
			t.Fatal(err)
		}
		cp, err := proveLeaf(cLevels, int(mid))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := dispute.SubmitMid(Worker, workerTrail[mid], wp, uint64(1001+2*rounds)); err != nil {
			t.Fatal(err)
		}
		if _, err := dispute.SubmitMid(Challenger, honestTrail[mid], cp, uint64(1002+2*rounds)); err != nil {
			t.Fatal(err)
		}
		rounds++
		if rounds > 16 {
			t.Fatal("bisection did not converge")
		}
	}
	node, err := dispute.FirstDivergentNode()
	if err != nil {
		t.Fatal(err)
	}
	return node
}

func firstNodeWith(t *testing.T, graph *GraphDescriptor, opID string) uint32 {
	t.Helper()
	for _, node := range graph.Nodes {
		if node.OperatorID == opID {
			return node.NodeID
		}
	}
	t.Fatalf("no %s node in the block graph", opID)
	return 0
}

// TestTransformerBlockFraudLocalization: corrupting one node of the real
// block graph must localize to exactly that node for the nonlinear,
// row and elementwise stages.
func TestTransformerBlockFraudLocalization(t *testing.T) {
	fix := buildBlockFixture(t, miniBlockConfig())
	targets := []uint32{
		firstNodeWith(t, fix.graph, OpSoftmaxFixedV1),
		firstNodeWith(t, fix.graph, OpRMSNormFixedV1),
		firstNodeWith(t, fix.graph, OpRequantizeV1),
		firstNodeWith(t, fix.graph, OpAddFixedV1),
	}
	for _, want := range targets {
		fraudTrail := fraudExecution(t, fix.graph, fix.inputs, fix.tables, want)
		got := bisectToNode(t, fix.graph, fraudTrail, fix.exec.Trail)
		if got != want {
			t.Fatalf("fraud at node %d (%s) localized to node %d",
				want, fix.graph.Nodes[want].OperatorID, got)
		}
	}
}

// fullOperandChunks renders every chunk of one tensor as evidence bound
// to its committed root.
func fullOperandChunks(t *testing.T, ref TensorRef, tensors map[TensorRef]Tensor) []ChunkEvidence {
	t.Helper()
	tens := tensors[ref]
	root, err := tens.TensorRoot()
	if err != nil {
		t.Fatal(err)
	}
	count := uint32(tens.ChunkCount())
	evidence := make([]ChunkEvidence, 0, count)
	for i := uint32(0); i < count; i++ {
		chunk, err := tens.chunkBytes(int(i))
		if err != nil {
			t.Fatal(err)
		}
		proof, err := tens.ChunkProof(int(i))
		if err != nil {
			t.Fatal(err)
		}
		evidence = append(evidence, ChunkEvidence{
			Ref: ref, Desc: tens.Desc, Root: root, ChunkIndex: i, Count: count,
			Bytes: chunk, Proof: proof,
		})
	}
	return evidence
}

func claimFor(t *testing.T, tens Tensor, corrupt bool) NodeChunkClaim {
	t.Helper()
	data := append([]int32(nil), tens.Data...)
	if corrupt {
		data[0]++
	}
	liar := Tensor{Desc: tens.Desc, Data: data}
	root, err := liar.TensorRoot()
	if err != nil {
		t.Fatal(err)
	}
	chunk, err := liar.chunkBytes(0)
	if err != nil {
		t.Fatal(err)
	}
	proof, err := liar.ChunkProof(0)
	if err != nil {
		t.Fatal(err)
	}
	return NodeChunkClaim{OutRoot: root, ChunkIndex: 0, ChunkBytes: chunk, Proof: proof}
}

// TestTransformerBlockGEMMNodeArbitration: a GEMM node dispute resolves
// through the bounded full-operand arbiter against the committed input
// state, with the correct party winning in both directions.
func TestTransformerBlockGEMMNodeArbitration(t *testing.T) {
	fix := buildBlockFixture(t, miniBlockConfig())
	gemmNode := firstNodeWith(t, fix.graph, OpGEMMInt8V1)

	fraudTrail := fraudExecution(t, fix.graph, fix.inputs, fix.tables, gemmNode)
	localized := bisectToNode(t, fix.graph, fraudTrail, fix.exec.Trail)
	if localized != gemmNode {
		t.Fatalf("GEMM fraud localized to node %d, want %d", localized, gemmNode)
	}
	node := fix.graph.Nodes[gemmNode]
	outRef := TensorRef{Kind: 1, Index: gemmNode}
	honestOut := fix.exec.Tensors[outRef]
	inputStateRoot := fix.exec.Trail[gemmNode] // state committed before this node

	evidence := append(
		fullOperandChunks(t, node.Inputs[0], fix.exec.Tensors),
		fullOperandChunks(t, node.Inputs[1], fix.exec.Tensors)...,
	)
	// The lying worker: corrupted output chunk. Challenger: the honest chunk.
	workerClaim := claimFor(t, honestOut, true)
	challengerClaim := claimFor(t, honestOut, false)
	verdict, err := ArbitrateNodeChunk(fix.graph, node, inputStateRoot, workerClaim, challengerClaim, evidence, nil)
	if err != nil {
		t.Fatal(err)
	}
	if verdict != ChallengerWins {
		t.Fatalf("lying GEMM worker verdict = %v, want ChallengerWins", verdict)
	}
	// Reverse: honest worker, lying challenger.
	verdict, err = ArbitrateNodeChunk(fix.graph, node, inputStateRoot, challengerClaim, workerClaim, evidence, nil)
	if err != nil {
		t.Fatal(err)
	}
	if verdict != WorkerWins {
		t.Fatalf("honest GEMM worker verdict = %v, want WorkerWins", verdict)
	}
	// Incomplete operand evidence must be rejected.
	if _, err := ArbitrateNodeChunk(fix.graph, node, inputStateRoot, workerClaim, challengerClaim, evidence[:len(evidence)-1], nil); err == nil {
		t.Fatal("incomplete GEMM evidence was accepted")
	}
}

// TestTransformerBlockBoundedArbitration: a softmax node of the block
// graph adjudicates with one bounded row of evidence (v0.1.1 MaxSafeK
// stays untouched; witness sizes stay small).
func TestTransformerBlockBoundedArbitration(t *testing.T) {
	fix := buildBlockFixture(t, miniBlockConfig())
	softNode := firstNodeWith(t, fix.graph, OpSoftmaxFixedV1)
	fraudTrail := fraudExecution(t, fix.graph, fix.inputs, fix.tables, softNode)
	if got := bisectToNode(t, fix.graph, fraudTrail, fix.exec.Trail); got != softNode {
		t.Fatalf("softmax fraud localized to %d", got)
	}
	node := fix.graph.Nodes[softNode]
	outRef := TensorRef{Kind: 1, Index: softNode}
	honestOut := fix.exec.Tensors[outRef]

	// Evidence: the disputed row as chunk blobs, bound to the input tensor.
	rowRef := node.Inputs[0]
	rowTens := fix.exec.Tensors[rowRef]
	hidden := int(node.Output.Shape[len(node.Output.Shape)-1])
	base := 0 // chunk 0 -> first row (hidden = Seq = 16, row spans first 16 elems)
	_ = base
	chunksPerRow := hidden / ChunkElems
	if chunksPerRow == 0 {
		chunksPerRow = 1
	}
	root, err := rowTens.TensorRoot()
	if err != nil {
		t.Fatal(err)
	}
	count := uint32(rowTens.ChunkCount())
	var evidence []ChunkEvidence
	for i := uint32(0); i < uint32(chunksPerRow); i++ {
		chunk, err := rowTens.chunkBytes(int(i))
		if err != nil {
			t.Fatal(err)
		}
		proof, err := rowTens.ChunkProof(int(i))
		if err != nil {
			t.Fatal(err)
		}
		evidence = append(evidence, ChunkEvidence{
			Ref: rowRef, Desc: rowTens.Desc, Root: root, ChunkIndex: i, Count: count,
			Bytes: chunk, Proof: proof,
		})
	}
	verdict, err := ArbitrateNodeChunk(fix.graph, node, fix.exec.Trail[softNode],
		claimFor(t, honestOut, true), claimFor(t, honestOut, false), evidence, nil)
	if err != nil {
		t.Fatal(err)
	}
	if verdict != ChallengerWins {
		t.Fatalf("softmax verdict = %v", verdict)
	}
}

func TestAttentionScaleFx(t *testing.T) {
	cases := []struct {
		headDim int
		want    int64
	}{
		{2, 741455}, // 2^20 / sqrt(2)
		{32, 185363},
		{64, 131072},
		{128, 92681}, // 2^20 / sqrt(128)
	}
	for _, c := range cases {
		got, err := AttentionScaleFx(c.headDim)
		if err != nil {
			t.Fatal(err)
		}
		if got != c.want {
			t.Fatalf("AttentionScaleFx(%d) = %d, want %d", c.headDim, got, c.want)
		}
	}
	if _, err := AttentionScaleFx(0); err == nil {
		t.Fatal("zero head_dim accepted")
	}
}

// TestRequantizeInt8Mode: the int8 output mode clamps, validates ranges
// and reports the int8 dtype.
func TestRequantizeInt8Mode(t *testing.T) {
	op, err := Lookup(OpRequantizeV1)
	if err != nil {
		t.Fatal(err)
	}
	in := Tensor{Desc: NewDesc(DtypeInt32Accum, 3), Data: []int32{1000, -5000, 200000000}}
	params := RequantSpec{Mult: 1, Shift: 5, Lo: -127, Hi: 127, OutInt8: true}.Params()
	out, err := op.Execute([]Tensor{in}, params)
	if err != nil {
		t.Fatal(err)
	}
	if Dtype(out.Desc.Dtype) != DtypeInt8 {
		t.Fatalf("output dtype = %d, want int8", out.Desc.Dtype)
	}
	want := []int32{31, -127, 127} // 1000>>5=31 (ties), -5000>>5=-156 -> clamp, huge -> clamp
	for i := range want {
		if out.Data[i] != want[i] {
			t.Fatalf("int8 requantize[%d] = %d, want %d", i, out.Data[i], want[i])
		}
	}
	if _, err := op.OutputSpec([]TensorDescriptor{in.Desc}, RequantSpec{Mult: 1, Shift: 5, Lo: -200, Hi: 127, OutInt8: true}.Params()); err == nil {
		t.Fatal("out-of-range int8 clamp accepted")
	}
	if _, err := op.OutputSpec([]TensorDescriptor{in.Desc}, ParamList{{Key: "out_dtype", Value: 2}, {Key: "mult", Value: 1}, {Key: "shift", Value: 1}, {Key: "clamp_lo", Value: -1}, {Key: "clamp_hi", Value: 1}}); err == nil {
		t.Fatal("invalid out_dtype accepted")
	}
}

// TestTransformerBlockMedium runs the medium block end to end (skipped
// in -short mode).
func TestTransformerBlockMedium(t *testing.T) {
	if testing.Short() {
		t.Skip("medium block execution is a long test")
	}
	cfg := mediumBlockConfig()
	fix := buildBlockFixture(t, cfg)
	if len(fix.graph.Nodes) != expectedNodeCount(cfg) {
		t.Fatalf("node count = %d", len(fix.graph.Nodes))
	}
	again, err := ExecuteGraph(fix.graph, fix.inputs, fix.tables)
	if err != nil {
		t.Fatal(err)
	}
	if fix.exec.Outputs[0] != again.Outputs[0] {
		t.Fatal("medium block execution is not deterministic")
	}
	if fix.exec.WorkVector.Get("GEMM_MAC", 0) <= 0 {
		t.Fatal("medium block work vector is empty")
	}
	_ = gemmv1.MaxSafeK // admission reused, not modified
}
