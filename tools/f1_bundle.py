"""GraphVerificationBundleV1 for the real Qwen3 block (Phase F.1).

A bundle is what a DA provider stores and what the permissionless watcher
consumes — nothing else:

    graph.json          canonical descriptor (Go-cased transport JSON)
    index.json          input/node-output file names, tensor roots, node
                        roots, manifest root, final output root, graph id
    tensors/*.npy       every committed tensor (inputs + node outputs)

`write_bundle` can corrupt a chosen node's output (fraud injection); the
manifest root is then committed OVER the corrupted outputs, exactly like
a lying worker would do, so the watcher must catch it semantically.

    python tools/f1_bundle.py --make        # honest + fraud bundles
    python tools/f1_bundle.py --watch       # run the watcher on each
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import canonical_ref as R  # noqa: E402
import f1_canonical_numpy as N  # noqa: E402
from convert_qwen3_block import (  # noqa: E402
    Conformer, build_converted_graph, calibrate, instrumented_forward, load_token_cases,
    mse_optimal_scale, to_fx, K_HEAD_FACTOR, CAL_MARGIN,
)

BUNDLES = REPO / "models" / "qwen3-0.6b-base-canonical" / "layer0" / "bundles"
RESULTS = REPO / "docs" / "phase-f1-watcher-results.json"


def build_graph(conf: Conformer):
    cal = load_token_cases("qwen3_f1_calibration")
    maxima, samples = calibrate(conf, cal, collect_samples=True)
    chosen = {}
    for key, vals in samples.items():
        step_fx, pct, mse = mse_optimal_scale(vals)
        if key.startswith("k_h"):
            step_fx = max(1, int(round(step_fx * K_HEAD_FACTOR)))
        chosen[key] = {"step_fx": step_fx, "percentile": pct, "mse": mse,
                       "clipped_fraction": 0.0, "calibration_max_abs": maxima.get(key, 0.0)}
    chosen["p"] = {"step_fx": max(1, int(np.ceil(CAL_MARGIN / 127 * (1 << 20)))),
                   "percentile": 100.0, "mse": 0.0, "clipped_fraction": 0.0,
                   "calibration_max_abs": 1.0}
    snake, go, plan, tdata, graph_id = build_converted_graph(conf, 16, {"chosen": chosen, "p": 1.0})
    return snake, go, tdata, graph_id


def write_bundle(conf: Conformer, out_dir: Path, case_index: int, corrupt_node: int | None,
                 corrupt_kind: str = "gemm") -> dict:
    snake, go, tdata, _ = build_graph(conf)
    case = load_token_cases("qwen3_f1_eval")[case_index]
    hidden = conf.embedding(case["token_ids"])
    inputs = dict(tdata)
    inputs[0] = to_fx(hidden).astype(np.int32)
    x_root = N.tensor_root_np(snake["inputs"][0]["desc"], inputs[0])
    # the descriptor with THIS case's committed input root (bytes form for
    # hashing; the Go-cased transport copy gets the same bytes)
    patched = dict(snake)
    patched["inputs"] = [dict(i) for i in snake["inputs"]]
    patched["inputs"][0] = {**patched["inputs"][0], "root": x_root}
    graph_id = R.graph_id(patched)
    go = json.loads(json.dumps(go, default=lambda b: b.hex()))
    go["Inputs"][0]["Root"] = x_root.hex()
    sc = json.loads(json.dumps(patched, default=lambda b: b.hex()))
    for gi in sc["inputs"]:
        if isinstance(gi["root"], str):
            gi["root"] = bytes.fromhex(gi["root"])
    out = N.execute_graph_np(sc, inputs, rope_tables={2: tdata[2]})
    tensors = out["tensors"]
    roots = dict(out["roots"])

    injected = None
    if corrupt_node is not None:
        ref = (1, corrupt_node)
        corrupted = tensors[ref].astype(np.int64).copy()
        if corrupt_kind == "gemm":
            corrupted[0, 0] += 3
        else:  # rope / cheap operator
            corrupted[0, 0] += 5
        tensors[ref] = corrupted.astype(np.int32)
        desc = sc["nodes"][corrupt_node]["output"]
        roots[ref] = N.tensor_root_np(desc, tensors[ref])
        injected = {"node": corrupt_node, "operator": sc["nodes"][corrupt_node]["operator_id"],
                    "kind": corrupt_kind,
                    "detail": "one output element altered after execution; the manifest "
                              "below is committed OVER the corrupted output"}

    node_roots = [roots[(1, i)] for i in range(len(sc["nodes"]))]
    manifest_root, _ = R.build_node_output_manifest(graph_id, patched["nodes"], node_roots)
    outputs = [roots[(r["kind"], r["index"])] for r in sc["outputs"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tensors").mkdir(exist_ok=True)
    (out_dir / "graph.json").write_text(json.dumps(go, default=lambda b: b.hex()), encoding="utf-8")

    index = {"inputs": [], "node_outputs": [], "node_roots": [h.hex() for h in node_roots],
             "manifest_root": manifest_root.hex(),
             "final_output_root": R.final_output_root(outputs).hex(),
             "graph_id": graph_id.hex()}
    for i, ginput in enumerate(sc["inputs"]):
        name = f"inputs_{i:02d}.npy"
        np.save(out_dir / "tensors" / name, inputs[i])
        index["inputs"].append({"index": i, "file": f"tensors/{name}",
                                "root": ginput["root"].hex() if isinstance(ginput["root"], bytes)
                                else ginput["root"]})
    for i in range(len(sc["nodes"])):
        name = f"node_{i:03d}.npy"
        np.save(out_dir / "tensors" / name, tensors[(1, i)])
        index["node_outputs"].append({"node": i, "file": f"tensors/{name}",
                                      "root": roots[(1, i)].hex()})
    (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    expected = {"graph_id": graph_id.hex(), "manifest_root": manifest_root.hex(),
                "final_output_root": R.final_output_root(outputs).hex()}
    (out_dir / "expected.json").write_text(json.dumps(expected, indent=1), encoding="utf-8")
    return {"bundle": str(out_dir), "injected": injected, "expected": expected}


def watch(bundle: Path, timeout_s: int = 1800) -> dict:
    started = time.time()
    proc = subprocess.run([sys.executable, "-u", str(REPO / "tools" / "graph_watcher.py"),
                           "--bundle", str(bundle), "--expected", str(bundle / "expected.json")],
                          capture_output=True, text=True, timeout=timeout_s)
    out = proc.stdout
    payload = json.loads(out[out.index("{"):]) if "{" in out else {"status": "NO_OUTPUT",
                                                                   "stderr": proc.stderr[-400:]}
    payload["wall_seconds_process"] = time.time() - started
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    conf = Conformer(0)

    if args.make or not args.watch:
        print("building bundles...")
        info = {}
        info["honest"] = write_bundle(conf, BUNDLES / "honest", 0, None)
        # locate the first and a late GEMM node of the block
        snake, go, tdata, gid = build_graph(conf)
        gemm_nodes = [n["node_id"] for n in snake["nodes"] if n["operator_id"] == R.OP_GEMM]
        rope_nodes = [n["node_id"] for n in snake["nodes"] if n["operator_id"] == R.OP_ROPE]
        print("gemm nodes:", len(gemm_nodes), "rope nodes:", len(rope_nodes))
        info["gemm_fraud"] = write_bundle(conf, BUNDLES / "gemm_fraud", 0, gemm_nodes[20], "gemm")
        info["rope_fraud"] = write_bundle(conf, BUNDLES / "rope_fraud", 0, rope_nodes[3], "rope")
        (BUNDLES / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print("bundles written under", BUNDLES)

    if args.watch:
        report = {"phase": "F.1", "block": "Qwen3-0.6B-Base layer 0, seq 16",
                  "note": "watcher runs read ONLY the bundle; each run is a fresh process "
                          "(restart-equivalent)",
                  "runs": {}}
        for name in ("honest", "gemm_fraud", "rope_fraud"):
            first = watch(BUNDLES / name)
            second = watch(BUNDLES / name)
            report["runs"][name] = {
                "first": first,
                "restart_reproduced": (first.get("status") == second.get("status") and
                                       first.get("first_suspected_node") == second.get("first_suspected_node")),
                "second_status": second.get("status"),
            }
            print(name, "->", first.get("status"),
                  "node:", first.get("first_suspected_node"), first.get("operator_id", ""))
        RESULTS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("written:", RESULTS)


if __name__ == "__main__":
    main()
