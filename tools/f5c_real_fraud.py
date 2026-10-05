"""F.5C real fraud artifact generators (roadmap A6-03/A6-05, reused by the
watcher tests): produce a self-consistent fraudulent bundle — one node
output altered, every downstream node recomputed from it, manifest and
final roots recomputed, so ONLY the mathematics is wrong.

    python tools/f5c_real_fraud.py --kind gemm --node 2 --out-prefix /tmp/f5c_g
    python tools/f5c_real_fraud.py --kind rope --node 138 --out-prefix /tmp/f5c_r
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import canonical_ref as R  # noqa: E402
from f5c_bundle_v2 import build_bundle, graph_id_v2_of  # noqa: E402
from f5c_exec_v2 import execute_v2, load_inputs_from_bundle  # noqa: E402


def task_meta(graph_doc: dict, outputs: dict, case: str) -> dict:
    sample = next(iter(outputs.values()))
    _ = sample
    nodes = graph_doc["nodes"]
    return {
        "case": case,
        "graph_id_v2": graph_id_v2_of(graph_doc).hex(),
        "policy_id": graph_doc["arithmetic"]["policy_id"],
        "task_ref": "f5c-e2e-task",
    }


def build(graph_doc: dict, inputs: dict, outputs: dict) -> bytes:
    entries = []
    input_entries = []
    for i, entry in enumerate(graph_doc["inputs"]):
        input_entries.append({"name": entry["name"], "desc": entry["desc"], "data": inputs[i]})
    for node in graph_doc["nodes"]:
        nid = int(node["node_id"])
        entries.append({"node_id": nid, "desc": node["output"], "data": outputs[nid]})
    return build_bundle(graph_doc, input_entries, entries)


def finish_task_meta(graph_doc: dict, outputs: dict, case: str) -> dict:
    meta = task_meta(graph_doc, outputs, case)
    raw = build(graph_doc, _MAIN_INPUTS, outputs)
    from f5c_bundle_v2 import decode_bundle
    header = decode_bundle(raw)["header"]
    _ = raw
    meta["manifest_root_v2"] = header["manifest_root_v2"]
    meta["final_output_root"] = header["final_output_root"]
    meta["output_roots"] = [header["node_outputs"][-1]["root"]]
    return meta


_MAIN_INPUTS: dict = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["gemm", "rope"], required=True)
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--delta", type=int, default=1)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    graph_doc = json.loads((REPO / "testdata" / "f5c_qwen3_block_v2.json")
                           .read_text(encoding="utf-8"))
    inputs = load_inputs_from_bundle(graph_doc, REPO / "testdata" / "f5b2_block_bundle.npz")
    global _MAIN_INPUTS
    _MAIN_INPUTS = inputs

    target = graph_doc["nodes"][args.node]
    want = "GEMM_A13W10_I64_V1" if args.kind == "gemm" else "ROPE_FIXED_V1"
    if target["operator_id"] != want:
        raise SystemExit(f"node {args.node} is {target['operator_id']}, not {want}")

    honest = execute_v2(graph_doc, inputs)
    fraud = execute_v2(graph_doc, inputs, tamper=(args.node, args.index, args.delta))

    honest_raw = build(graph_doc, inputs, honest)
    fraud_raw = build(graph_doc, inputs, fraud)
    honest_meta = finish_task_meta(graph_doc, honest, f"honest")
    fraud_meta = finish_task_meta(graph_doc, fraud, f"{args.kind}_fraud_node{args.node}")

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.with_name(prefix.name + "_honest.bin")).write_bytes(honest_raw)
    (prefix.with_name(prefix.name + "_fraud.bin")).write_bytes(fraud_raw)
    (prefix.with_name(prefix.name + "_honest_task.json")).write_text(
        json.dumps(honest_meta, indent=1) + "\n", encoding="utf-8")
    (prefix.with_name(prefix.name + "_fraud_task.json")).write_text(
        json.dumps(fraud_meta, indent=1) + "\n", encoding="utf-8")
    # the fraud differs in exactly the injected node's subtree
    diff_nodes = [int(n["node_id"]) for n in graph_doc["nodes"]
                  if not np.array_equal(honest[int(n["node_id"])], fraud[int(n["node_id"])])]
    print(json.dumps({"kind": args.kind, "node": args.node,
                      "differing_nodes": diff_nodes[:5],
                      "differing_count": len(diff_nodes),
                      "honest_manifest": honest_meta["manifest_root_v2"][:16],
                      "fraud_manifest": fraud_meta["manifest_root_v2"][:16]}, indent=1))


if __name__ == "__main__":
    main()
