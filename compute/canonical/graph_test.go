package canonical

// Graph-level tests: honest execution, fraud localization to the first
// divergent node for every operator, false-challenge rejection, bounded
// arbitration verdicts and malformed-evidence rejection.

import (
	"math"
	"testing"
)

func powFloat(a, b float64) float64 { return math.Pow(a, b) }
func cosFloat(x float64) float64    { return math.Cos(x) }
func sinFloat(x float64) float64    { return math.Sin(x) }
func roundFxTest(x float64) int32   { return int32(math.Round(x * float64(int64(1)<<FracBits))) }

const (
	gSeq    = 16
	gHidden = 128
)

// buildMiniGraph builds a graph that touches every canonical operator:
//
//	0 RMSNORM(x, w)      1 SILU           2 MUL (gating)
//	3 ADD (residual)     4 REQUANTIZE     5 ROPE
//	6 SOFTMAX
//
// inputs: 0 = x [seq,hidden], 1 = w_norm [hidden], 2 = rope constants
func buildMiniGraph(t *testing.T) (*GraphDescriptor, map[uint32]Tensor, map[uint32]*RopeConstants, []Hash) {
	t.Helper()
	xDesc := NewDesc(DtypeQ12_20, gSeq, gHidden)
	wDesc := NewDesc(DtypeQ12_20, gHidden)
	x := Tensor{Desc: xDesc, Data: make([]int32, gSeq*gHidden)}
	for i := range x.Data {
		x.Data[i] = int32(((i*97)%400 - 200) * 1049) // 约 [-0.2,0.2] in Q12.20
	}
	w := Tensor{Desc: wDesc, Data: make([]int32, gHidden)}
	for i := range w.Data {
		w.Data[i] = int32(one - int64(i%5)*(one/8))
	}
	constants, tableTensor, tableRoot := buildRopeTable(t, gSeq, gHidden/2)
	accumDesc := NewDesc(DtypeInt32Accum, gSeq, gHidden)
	accum := Tensor{Desc: accumDesc, Data: make([]int32, gSeq*gHidden)}
	for i := range accum.Data {
		accum.Data[i] = int32((i*31)%2000 - 1000) // raw accumulator magnitudes
	}
	accumRoot, err0 := accum.TensorRoot()
	if err0 != nil {
		t.Fatal(err0)
	}

	xRoot, err := x.TensorRoot()
	if err != nil {
		t.Fatal(err)
	}
	wRoot, err := w.TensorRoot()
	if err != nil {
		t.Fatal(err)
	}
	graph := &GraphDescriptor{
		ProtocolVersion: GraphProtocolVersion,
		Spec:            "MINI_BLOCK_TEST",
		Inputs: []GraphInput{
			{Name: "x", Desc: xDesc, Root: xRoot[:]},
			{Name: "w_norm", Desc: wDesc, Root: wRoot[:]},
			{Name: "rope_table", Desc: tableTensor.Desc, Root: tableRoot[:]},
			{Name: "accum", Desc: accumDesc, Root: accumRoot[:]},
		},
	}
	add := func(id uint32, op string, ins []TensorRef, out TensorDescriptor, params ParamList) {
		graph.Nodes = append(graph.Nodes, GraphNode{
			NodeID: id, OperatorID: op, Version: VersionFxFusion,
			Inputs: ins, Output: out, Params: params,
		})
	}
	ref := func(kind uint8, index uint32) TensorRef { return TensorRef{Kind: kind, Index: index} }
	add(0, OpRMSNormFixedV1, []TensorRef{ref(0, 0), ref(0, 1)}, xDesc,
		ParamList{{Key: "chunk", Value: normChunk}, {Key: "eps_fx", Value: 10}})
	add(1, OpSiLUFixedV1, []TensorRef{ref(1, 0)}, xDesc, nil)
	add(2, OpMulFixedV1, []TensorRef{ref(1, 0), ref(1, 1)}, xDesc, nil)
	add(3, OpAddFixedV1, []TensorRef{ref(1, 2), ref(0, 0)}, xDesc, nil)
	add(4, OpRequantizeV1, []TensorRef{ref(0, 3)}, xDesc,
		ParamList{{Key: "clamp_hi", Value: int64(MaxFx)}, {Key: "clamp_lo", Value: int64(MinFx)},
			{Key: "mult", Value: 1 << 10}, {Key: "shift", Value: 10}})
	add(5, OpRoPEFixedV1, []TensorRef{ref(1, 4), ref(0, 2)}, xDesc,
		ParamList{{Key: "half_dim", Value: gHidden / 2}, {Key: "max_pos", Value: gSeq}})
	add(6, OpSoftmaxFixedV1, []TensorRef{ref(1, 5)}, xDesc,
		ParamList{{Key: "chunk", Value: normChunk}})
	graph.Outputs = []TensorRef{ref(1, 6), ref(1, 3)}
	if err := graph.Validate(); err != nil {
		t.Fatal(err)
	}
	inputs := map[uint32]Tensor{0: x, 1: w, 2: tableTensor, 3: accum}
	tables := map[uint32]*RopeConstants{2: constants}
	roots := []Hash{xRoot, wRoot, tableRoot, accumRoot}
	return graph, inputs, tables, roots
}

func buildRopeTable(t *testing.T, positions, pairs int) (*RopeConstants, Tensor, Hash) {
	t.Helper()
	table := make([]int32, positions*pairs*2)
	for pos := 0; pos < positions; pos++ {
		for pair := 0; pair < pairs; pair++ {
			cosV, sinV := ropeConstantsQuick(pos, pair, pairs)
			table[(pos*pairs+pair)*2] = cosV
			table[(pos*pairs+pair)*2+1] = sinV
		}
	}
	constants, err := NewRopeConstants(table, pairs)
	if err != nil {
		t.Fatal(err)
	}
	tensor := Tensor{Desc: NewDesc(DtypeQ12_20, int64(len(table))), Data: table}
	root, err := tensor.TensorRoot()
	if err != nil {
		t.Fatal(err)
	}
	return constants, tensor, root
}

func TestGraphHonestExecutionDeterministic(t *testing.T) {
	graph, inputs, tables, _ := buildMiniGraph(t)
	exec1, err := ExecuteGraph(graph, inputs, tables)
	if err != nil {
		t.Fatal(err)
	}
	exec2, err := ExecuteGraph(graph, inputs, tables)
	if err != nil {
		t.Fatal(err)
	}
	if len(exec1.Trail) != len(graph.Nodes)+1 {
		t.Fatalf("trail length %d", len(exec1.Trail))
	}
	for i := range exec1.Outputs {
		if exec1.Outputs[i] != exec2.Outputs[i] {
			t.Fatal("graph execution is not deterministic")
		}
	}
	if len(exec1.WorkVector) == 0 {
		t.Fatal("work vector is empty")
	}
	// graph id binds parameters
	id1, _ := graph.GraphID()
	mutated := *graph
	mutated.Nodes = append([]GraphNode(nil), graph.Nodes...)
	mutated.Nodes[0].Params = ParamList{{Key: "chunk", Value: normChunk}, {Key: "eps_fx", Value: 11}}
	id2, _ := mutated.GraphID()
	if id1 == id2 {
		t.Fatal("graph id does not change with parameters")
	}
}

// fraudExecution rebuilds a trail where node `fraudNode`'s output tensor
// is corrupted (one element in chunk 0) before the state root is taken.
func fraudExecution(t *testing.T, graph *GraphDescriptor, inputs map[uint32]Tensor,
	tables map[uint32]*RopeConstants, fraudNode uint32) []Hash {
	t.Helper()
	exec, err := ExecuteGraph(graph, inputs, tables)
	if err != nil {
		t.Fatal(err)
	}
	// Simulate the lying worker: corrupt the node's stored output tensor
	// and recompute every later state root from it. We rebuild the trail by
	// replaying with a hook.
	tensors := map[TensorRef]Tensor{}
	for ref, t2 := range exec.Tensors {
		tensors[ref] = t2
	}
	ref := TensorRef{Kind: 1, Index: fraudNode}
	corrupted := tensors[ref]
	corrupted.Data = append([]int32(nil), corrupted.Data...)
	corrupted.Data[0] += 1
	tensors[ref] = corrupted
	var trail []Hash
	// recompute from the corrupted node onward: honest prefix, corrupted tail
	for k := uint32(0); k <= fraudNode; k++ {
		trail = append(trail, exec.Trail[k])
	}
	// rebuild the live-set state after the corrupted node
	live := map[TensorRef]Tensor{}
	for _, in := range graph.Inputs {
		_ = in
	}
	live = map[TensorRef]Tensor{}
	for key := range exec.Tensors {
		if key.Kind == 0 || key.Index <= fraudNode {
			live[key] = tensors[key] // tensors holds the corrupted node output
		}
	}
	state, err := stateRootFor(live)
	if err != nil {
		t.Fatal(err)
	}
	trail = append(trail, state)
	for k := fraudNode + 1; k < uint32(len(graph.Nodes)); k++ {
		live[TensorRef{Kind: 1, Index: k}] = exec.Tensors[TensorRef{Kind: 1, Index: k}]
		state, err = stateRootFor(live)
		if err != nil {
			t.Fatal(err)
		}
		trail = append(trail, state)
	}
	return trail
}

func trailClaim(t *testing.T, graphID Hash, trail []Hash) TrailClaim {
	t.Helper()
	levels, err := TrailLevels(graphID, trail)
	if err != nil {
		t.Fatal(err)
	}
	last := uint32(len(trail) - 1)
	pi, err := proveLeaf(levels, 0)
	if err != nil {
		t.Fatal(err)
	}
	pf, err := proveLeaf(levels, int(last))
	if err != nil {
		t.Fatal(err)
	}
	return TrailClaim{
		Root: levels[len(levels)-1][0], InitialRoot: trail[0], InitialProof: pi,
		FinalRoot: trail[last], FinalProof: pf,
	}
}

// TestGraphFraudLocalization: a corrupted node output must localize to
// the FIRST divergent node for every operator position in the graph.
func TestGraphFraudLocalization(t *testing.T) {
	graph, inputs, tables, _ := buildMiniGraph(t)
	exec, err := ExecuteGraph(graph, inputs, tables)
	if err != nil {
		t.Fatal(err)
	}
	graphID, _ := graph.GraphID()
	honestClaim := trailClaim(t, graphID, exec.Trail)

	for fraudNode := uint32(0); fraudNode < uint32(len(graph.Nodes)); fraudNode++ {
		fraudTrail := fraudExecution(t, graph, inputs, tables, fraudNode)
		fraudClaim := trailClaim(t, graphID, fraudTrail)
		if fraudClaim.FinalRoot == honestClaim.FinalRoot {
			t.Fatalf("fraud at node %d did not change the final state root", fraudNode)
		}
		cfg := GraphDisputeConfig{Graph: graph, GraphID: graphID, RoundPeriod: 50}
		// the lying worker holds the fraud trail; the challenger the honest one
		dispute, err := NewGraphDispute(cfg, fraudClaim, honestClaim, 1000)
		if err != nil {
			t.Fatal(err)
		}
		rounds := 0
		for dispute.ArbReady() == false {
			mid := dispute.low + (dispute.high-dispute.low)/2
			wLevels, _ := TrailLevels(graphID, fraudTrail)
			cLevels, _ := TrailLevels(graphID, exec.Trail)
			wp, _ := proveLeaf(wLevels, int(mid))
			cp, _ := proveLeaf(cLevels, int(mid))
			if _, err := dispute.SubmitMid(Worker, fraudTrail[mid], wp, uint64(1001+2*rounds)); err != nil {
				t.Fatal(err)
			}
			if _, err := dispute.SubmitMid(Challenger, exec.Trail[mid], cp, uint64(1002+2*rounds)); err != nil {
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
		if node != fraudNode {
			t.Fatalf("fraud at node %d localized to node %d", fraudNode, node)
		}
	}
}

// TestGraphFalseChallenge: identical executions must not open a dispute.
func TestGraphFalseChallenge(t *testing.T) {
	graph, inputs, tables, _ := buildMiniGraph(t)
	exec, err := ExecuteGraph(graph, inputs, tables)
	if err != nil {
		t.Fatal(err)
	}
	graphID, _ := graph.GraphID()
	claim := trailClaim(t, graphID, exec.Trail)
	cfg := GraphDisputeConfig{Graph: graph, GraphID: graphID, RoundPeriod: 50}
	if _, err := NewGraphDispute(cfg, claim, claim, 1000); err == nil {
		t.Fatal("identical trails must not open a dispute")
	}
}

// TestBoundedArbitration per operator: correct party wins, liar loses.
func TestBoundedArbitration(t *testing.T) {
	graph, inputs, tables, _ := buildMiniGraph(t)
	exec, err := ExecuteGraph(graph, inputs, tables)
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range graph.Nodes {
		out := exec.Tensors[TensorRef{Kind: 1, Index: node.NodeID}]
		// The lying worker commits a corrupted output tensor: its chunk,
		// its root and its proof all belong to the corrupted tensor.
		corrupted := out
		corrupted.Data = append([]int32(nil), out.Data...)
		corrupted.Data[0] += 1
		lieLevels, _, err := corrupted.TensorTree()
		if err != nil {
			t.Fatal(err)
		}
		honestLevels, _, err := out.TensorTree()
		if err != nil {
			t.Fatal(err)
		}
		liarChunk, _ := corrupted.chunkBytes(0)
		liarProof, _ := proveLeaf(lieLevels, 0)
		liarRoot, _ := corrupted.TensorRoot()
		chunk, _ := out.chunkBytes(0)
		proof, _ := proveLeaf(honestLevels, 0)
		root, _ := out.TensorRoot()
		worker := NodeChunkClaim{OutRoot: liarRoot, ChunkIndex: 0, ChunkBytes: liarChunk, Proof: liarProof}
		challenger := NodeChunkClaim{OutRoot: root, ChunkIndex: 0, ChunkBytes: chunk, Proof: proof}
		evidence, table := arbitrationEvidence(t, graph, inputs, tables, exec, node)
		outcome, err := ArbitrateNodeChunk(graph, node, exec.Trail[node.NodeID], worker, challenger, evidence, table)
		if err != nil {
			t.Fatalf("node %d arbitration error: %v", node.NodeID, err)
		}
		if outcome != ChallengerWins {
			t.Fatalf("node %d (%s): lying worker not beaten, outcome %d", node.NodeID, node.OperatorID, outcome)
		}
		// reverse roles: honest worker vs lying challenger
		outcome, err = ArbitrateNodeChunk(graph, node, exec.Trail[node.NodeID], challenger, worker, evidence, table)
		if err != nil {
			t.Fatal(err)
		}
		if outcome != WorkerWins {
			t.Fatalf("node %d (%s): honest worker lost, outcome %d", node.NodeID, node.OperatorID, outcome)
		}
	}
}

// arbitrationEvidence assembles the bounded input chunks a node's arbiter
// consumes, with proofs against the committed input state.
func arbitrationEvidence(t *testing.T, graph *GraphDescriptor, inputs map[uint32]Tensor,
	tables map[uint32]*RopeConstants, exec *GraphExecution, node GraphNode) ([]ChunkEvidence, *RopeConstants) {
	t.Helper()
	var evidence []ChunkEvidence
	var table *RopeConstants
	// appendChunks attaches the chunks covering element range [start,end)
	// of one tensor, with proofs against that tensor's committed root.
	appendChunks := func(ref TensorRef, start, end int) {
		tt := exec.Tensors[ref]
		levels, _, err := tt.TensorTree()
		if err != nil {
			t.Fatal(err)
		}
		elems, _ := tt.Desc.Elems()
		count := uint32((elems + ChunkElems - 1) / ChunkElems)
		root, _ := tt.TensorRoot()
		for c := start / ChunkElems; c <= (end-1)/ChunkElems; c++ {
			chunk, _ := tt.chunkBytes(c)
			proof, _ := proveLeaf(levels, c)
			evidence = append(evidence, ChunkEvidence{Ref: ref, Desc: tt.Desc, Root: root,
				ChunkIndex: uint32(c), Count: count, Bytes: chunk, Proof: proof})
		}
	}
	hidden := int(node.Output.Shape[len(node.Output.Shape)-1])
	row := 0 // disputed chunk is always 0 in these tests
	switch node.OperatorID {
	case OpRMSNormFixedV1:
		appendChunks(node.Inputs[0], row*hidden, (row+1)*hidden)
		appendChunks(node.Inputs[1], 0, hidden)
	case OpSoftmaxFixedV1:
		appendChunks(node.Inputs[0], row*hidden, (row+1)*hidden)
	case OpRoPEFixedV1:
		start := row * hidden
		appendChunks(node.Inputs[0], start, start+ChunkElems)
		table = tables[node.Inputs[1].Index]
	default:
		for _, ref := range node.Inputs {
			base := row * hidden
			appendChunks(ref, base, base+ChunkElems)
		}
	}
	return evidence, table
}

// ropeConstantsQuick generates test RoPE constants with float math; the
// pinned production table is generated by the (future) converter with the
// same rounding rule.
func ropeConstantsQuick(pos, pair, pairs int) (int32, int32) {
	theta := float64(pos) / powFloat(10000, 2*float64(pair)/float64(2*pairs))
	return roundFxTest(cosFloat(theta)), roundFxTest(sinFloat(theta))
}
