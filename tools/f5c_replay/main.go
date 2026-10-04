// F.5C replay tool: executes the formal QWEN3_BLOCK_PROFILE_V2 graph with
// the Go reference executor and prints GraphIDV2, the work vector and every
// node's TensorRootV2.  Cross-language equivalence: the roots must equal the
// Python mirror's roots byte-for-byte (tools/f5c_compare_replay.py).
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"prismachain/compute/canonical"
)

func main() {
	descPath := flag.String("desc", "testdata/f5c_qwen3_block_v2.json", "GraphDescriptorV2 JSON")
	inputsPath := flag.String("inputs", "testdata/f5c_v2_inputs/inputs.bin", "packed input tensors")
	outPath := flag.String("out", "testdata/f5c_v2_replay_go.json", "output JSON")
	flag.Parse()

	raw, err := os.ReadFile(*descPath)
	must(err)
	var g canonical.GraphDescriptorV2
	must(json.Unmarshal(raw, &g))
	must(g.ValidateV2())
	graphID, err := g.GraphIDV2()
	must(err)

	bin, err := os.ReadFile(*inputsPath)
	must(err)
	if string(bin[:8]) != "F5CV2IN1" {
		panic("bad inputs magic")
	}
	count := int(binary.LittleEndian.Uint32(bin[8:12]))
	off := 12
	inputs := map[uint32]*canonical.TensorV2{}
	for i := 0; i < count; i++ {
		n := int(binary.LittleEndian.Uint64(bin[off : off+8]))
		off += 8
		data := make([]int64, n)
		for j := 0; j < n; j++ {
			data[j] = int64(binary.LittleEndian.Uint64(bin[off+j*8 : off+j*8+8]))
		}
		off += n * 8
		t, err := canonical.NewTensorV2(g.Inputs[i].Desc, data)
		must(err)
		inputs[uint32(i)] = t
	}

	exec, err := canonical.ExecuteGraphV2(&g, inputs)
	must(err)

	nodeRoots := make([]string, len(g.Nodes))
	for i := range g.Nodes {
		t := exec.Tensors[canonical.TensorRef{Kind: 1, Index: uint32(i)}]
		root, err := t.TensorRootV2()
		must(err)
		nodeRoots[i] = fmt.Sprintf("%x", root[:])
	}
	outputs := make([]string, len(exec.Outputs))
	for i, r := range exec.Outputs {
		outputs[i] = fmt.Sprintf("%x", r[:])
	}
	work := make([][2]any, 0, len(exec.WorkVector))
	for _, p := range exec.WorkVector {
		work = append(work, [2]any{p.Key, p.Value})
	}
	doc := map[string]any{
		"graph_id_v2":  fmt.Sprintf("%x", graphID[:]),
		"work_vector":  work,
		"node_roots_v2": nodeRoots,
		"output_roots": outputs,
		"nodes":        len(g.Nodes),
		"inputs":       len(g.Inputs),
	}
	enc, err := json.MarshalIndent(doc, "", " ")
	must(err)
	must(os.WriteFile(*outPath, append(enc, '\n'), 0o644))
	fmt.Printf("GraphIDV2: %x\nnodes: %d inputs: %d\nwritten: %s\n",
		graphID[:], len(g.Nodes), len(g.Inputs), filepath.Base(*outPath))
	_ = math.MaxInt64
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "fatal:", err)
		os.Exit(1)
	}
}
