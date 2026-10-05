// f5c-genlegacy freezes the V1 protocol corpus into a golden fixture:
// V1 TensorRoots, the frozen V1 GraphID, a fixed V1 commit preimage hash
// and V1 receipt id, plus the frozen V2 counterparts.  Rerunning the tool
// on a correct tree reproduces the file byte-for-byte; the regression test
// (compute/canonical/legacy_golden_test.go) recomputes every value.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"prismachain/compute/canonical"
)

func h(s string) []byte { b, _ := hex.DecodeString(s); return b }

func repeat(hexByte string, n int) []byte {
	out := make([]byte, n)
	for i := range out {
		out[i] = h(hexByte)[0]
	}
	return out
}

func main() {
	repo := "."
	raw, err := os.ReadFile(filepath.Join(repo, "compute", "canonical", "testdata", "canonical_vectors.json"))
	must(err)
	var vectors struct {
		TensorRoots []struct {
			Name string `json:"name"`
			Desc struct {
				Dtype  uint8   `json:"dtype"`
				Layout uint8   `json:"layout"`
				Shape  []int64 `json:"shape"`
			} `json:"desc"`
			Data []int32 `json:"data"`
			Root string  `json:"root"`
		} `json:"tensor_roots"`
		Graph struct {
			GraphID string          `json:"graph_id"`
			DescRaw json.RawMessage `json:"descriptor"`
		} `json:"graph"`
	}
	must(json.Unmarshal(raw, &vectors))

	// --- V1 tensor roots (recomputed from the fixture data) ------------------
	type rootEntry struct {
		Name string `json:"name"`
		Root string `json:"root"`
	}
	var roots []rootEntry
	for _, tr := range vectors.TensorRoots {
		desc := canonical.NewDesc(canonical.Dtype(tr.Desc.Dtype), tr.Desc.Shape...)
		root, err := canonical.TensorRootOf(desc, tr.Data)
		must(err)
		if hex.EncodeToString(root[:]) != tr.Root {
			panic("fixture root mismatch for " + tr.Name)
		}
		roots = append(roots, rootEntry{Name: tr.Name, Root: tr.Root})
	}

	// --- V1 graph id ----------------------------------------------------------
	// the fixture encodes roots as hex strings and Params as [[k, v], ...]
	// pairs; parse manually (same path as the cross-language test) and sort
	// the parameter lists canonically.
	jmap := func(v any) map[string]any { return v.(map[string]any) }
	jarr := func(v any) []any { return v.([]any) }
	jstr := func(v any) string { return v.(string) }
	jnum := func(v any) int64 { return int64(v.(float64)) }
	var g canonical.GraphDescriptor
	var descAny any
	must(json.Unmarshal(vectors.Graph.DescRaw, &descAny))
	descMap := jmap(descAny)
	g.ProtocolVersion = jstr(descMap["protocol_version"])
	g.Spec = jstr(descMap["spec"])
	for _, inAny := range jarr(descMap["inputs"]) {
		in := jmap(inAny)
		d := jmap(in["desc"])
		shapeParts := jarr(d["shape"])
		shape := make([]int64, len(shapeParts))
		for i, sp := range shapeParts {
			shape[i] = jnum(sp)
		}
		rootBytes := h(jstr(in["root"]))
		g.Inputs = append(g.Inputs, canonical.GraphInput{
			Name: jstr(in["name"]),
			Desc: canonical.NewDesc(canonical.Dtype(jnum(d["dtype"])), shape...),
			Root: rootBytes,
		})
	}
	for _, nodeAny := range jarr(descMap["nodes"]) {
		n := jmap(nodeAny)
		var refs []canonical.TensorRef
		for _, refAny := range jarr(n["inputs"]) {
			r := jmap(refAny)
			refs = append(refs, canonical.TensorRef{Kind: uint8(jnum(r["kind"])), Index: uint32(jnum(r["index"]))})
		}
		var params canonical.ParamList
		for _, prAny := range jarr(n["params"]) {
			pr := jarr(prAny)
			params = append(params, canonical.Pair{Key: jstr(pr[0]), Value: jnum(pr[1])})
		}
		od := jmap(n["output"])
		oshapeParts := jarr(od["shape"])
		oshape := make([]int64, len(oshapeParts))
		for i, sp := range oshapeParts {
			oshape[i] = jnum(sp)
		}
		g.Nodes = append(g.Nodes, canonical.GraphNode{
			NodeID:     uint32(jnum(n["node_id"])),
			OperatorID: jstr(n["operator_id"]),
			Version:    jstr(n["operator_version"]),
			Inputs:     refs,
			Output:     canonical.NewDesc(canonical.Dtype(jnum(od["dtype"])), oshape...),
			Params:     params.Canonical(),
		})
	}
	for _, refAny := range jarr(descMap["outputs"]) {
		r := jmap(refAny)
		g.Outputs = append(g.Outputs, canonical.TensorRef{Kind: uint8(jnum(r["kind"])), Index: uint32(jnum(r["index"]))})
	}
	graphID, err := g.GraphID()
	must(err)
	if hex.EncodeToString(graphID[:]) != vectors.Graph.GraphID {
		panic("graph id mismatch")
	}

	// --- fixed V1 commit preimage hash ---------------------------------------
	rcV1 := &canonical.GraphResultCommit{
		ProtocolVersion: canonical.GraphResultCommitVersion,
		GraphID:         repeat("11", 32),
		TaskRef:         []byte("legacy-golden-task"),
		AssignmentRef:   repeat("22", 32),
		WorkerPubKey:    repeat("33", 32),
		FinalOutputRoot: repeat("44", 32),
		OutputRoots:     [][]byte{repeat("55", 32), repeat("66", 32)},
		CompletedEpoch:  7,
	}
	encV1, err := canonical.EncodeCanonical(rcV1)
	must(err)
	preV1 := sha256.Sum256(append([]byte(canonical.DomainSig), encV1...))

	// --- fixed V1 receipt id --------------------------------------------------
	work := canonical.WorkVector{}.Add("ADD_ELEMENT", 11).Add("GEMM_MAC", 22)
	receiptV1, err := canonical.BuildVerifiedGraphWorkReceiptV1(&g, []byte("legacy-golden-task"),
		repeat("22", 32), repeat("33", 32), []canonical.Hash{{0x44}}, work, canonical.ModeOptimisticUnchallenged,
		9, []byte("settle"), nil)
	must(err)
	v1ReceiptID, err := receiptV1.ReceiptID()
	must(err)

	// --- frozen V2 counterparts ----------------------------------------------
	a13 := canonical.NewDescV2(canonical.DtypeV2A13, 3)
	d := []int64{-4096, -1, 4095}
	v2Root, err := canonical.TensorRootV2Of(a13, d)
	must(err)

	rcV3 := &canonical.GraphResultCommitV3{
		ProtocolVersion:          canonical.GraphResultCommitV3Version,
		GraphID:                  repeat("77", 32),
		ArithmeticID:             canonical.ArithmeticProfileA13W10,
		PolicyID:                 canonical.PolicyIDA13W10,
		TaskRef:                  []byte("legacy-golden-task"),
		AssignmentRef:            repeat("22", 32),
		WorkerPubKey:             repeat("33", 32),
		NodeOutputManifestRootV2: repeat("88", 32),
		FinalOutputRoot:          repeat("44", 32),
		OutputRoots:              [][]byte{repeat("55", 32)},
		CompletedEpoch:           7,
	}
	encV3, err := canonical.EncodeCanonical(rcV3)
	must(err)
	preV3 := sha256.Sum256(append([]byte(canonical.DomainSigV3), encV3...))

	doc := map[string]any{
		"version": "f5c-legacy-golden-v1",
		"v1": map[string]any{
			"tensor_roots":      roots,
			"graph_id":          vectors.Graph.GraphID,
			"commit_preimage":   hex.EncodeToString(preV1[:]),
			"receipt_id":        hex.EncodeToString(v1ReceiptID[:]),
			"commit_version":    canonical.GraphResultCommitVersion,
			"receipt_domain":    "PRISMA_GRAPH_RECEIPT_V1\\x00",
			"signature_domain":  "PRISMA_CANONICAL_SIG_V1\\x00",
		},
		"v2": map[string]any{
			"a13_probe_root":   hex.EncodeToString(v2Root[:]),
			"commit_preimage":  hex.EncodeToString(preV3[:]),
			"commit_version":   canonical.GraphResultCommitV3Version,
			"signature_domain": "PRISMA_CANONICAL_SIG_V3\\x00",
			"graph_id_v2":      "8fb86087697e277602bb4b2088ba48eb4172c21d9c4f83914582f2cbf00d4def",
			"policy_id":        canonical.PolicyIDA13W10,
		},
		"note": "V1 values must reproduce byte-for-byte on every future tree; " +
			"V2 values freeze the wide-integer family at its introduction.",
	}
	out, err := json.MarshalIndent(doc, "", " ")
	must(err)
	dest := filepath.Join(repo, "testdata", "f5c_legacy_golden.json")
	must(os.WriteFile(dest, append(out, '\n'), 0o644))
	fmt.Println("written:", dest)
	fmt.Println("v1 graph_id:", vectors.Graph.GraphID)
	fmt.Println("v1 receipt:", hex.EncodeToString(v1ReceiptID[:]))
	fmt.Println("v2 a13 root:", hex.EncodeToString(v2Root[:]))
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "fatal:", err)
		os.Exit(1)
	}
}
