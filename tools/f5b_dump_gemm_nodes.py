"""EXPERIMENTAL / NON-PROTOCOL — dump the 83 real GEMM nodes for GPU replay.

Runs the frozen F.5A A13W10 CPU pipeline on one heldout case, captures at
every GEMM node the exact operator inputs (A13 container + W10
container bytes as int16, transpose flag) and the expected int64
accumulator BEFORE requantization, and writes them to
testdata/f5b_gemm_nodes.npz for the GPU runner's node-by-node exactness
check (section 73 of the F.5B spec).

    python tools/f5b_dump_gemm_nodes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

ONE = 1 << 20


def main() -> None:
    import f5a_joint_precision as F5A
    from convert_qwen3_block import Conformer, build_converted_graph

    conf = Conformer(0)
    cases, samples, head_samples = F5A.collect_calibration(conf)
    plan = F5A.build_plan(conf, samples, head_samples, 13, 10)
    stats = {"acc_abs_max": 0, "fx_saturations": 0, "gemm_dump": []}
    held = json.loads((REPO / "testdata" / "qwen3_f5a_heldout.json")
                      .read_text(encoding="utf-8"))["cases"]
    hidden = conf.embedding(held[0]["token_ids"])
    F5A.run_block_bits(conf, hidden, plan, stats)
    shots = stats["gemm_dump"]

    # the pipeline executes GEMMs in the same order as the graph's GEMM
    # node ids; map index -> node_id for the GPU replay metadata
    chosen = {k: {"step_fx": v["step_fx"], "percentile": v["percentile"], "mse": v["mse"],
                  "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
              for k, v in plan["sites"].items() if k not in ("k_heads", "q_heads", "p")}
    chosen["p"] = {"step_fx": plan["sites"]["p"]["step_fx"], "percentile": 100.0, "mse": 0.0,
                   "clipped_fraction": 0.0, "calibration_max_abs": 1.0}
    for i, st in enumerate(plan["sites"]["k_heads"]):
        chosen[f"k_h{i}"] = {"step_fx": st, "percentile": 100.0, "mse": 0.0,
                             "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
    for i, st in enumerate(plan["sites"]["q_heads"]):
        chosen[f"q_h{i}"] = {"step_fx": st, "percentile": 100.0, "mse": 0.0,
                             "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
    snake, _go, _plan, _tdata, _gid = build_converted_graph(conf, 16, {"chosen": chosen, "p": 1.0})
    gemm_ids = [n["node_id"] for n in snake["nodes"] if n["operator_id"] == "GEMM_INT8_V1"]
    assert len(gemm_ids) == len(shots), (len(gemm_ids), len(shots))
    print("captured GEMM nodes:", len(shots))

    out = {}
    meta = []
    for node_id, info in zip(gemm_ids, shots):
        out[f"a_{node_id}"] = info["a"].astype(np.int16)
        out[f"w_{node_id}"] = info["w"].astype(np.int16)
        out[f"c_{node_id}"] = info["c"].astype(np.int64)
        meta.append({"node_id": node_id, "M": int(info["a"].shape[0]),
                     "N": int(info["w"].shape[0]), "K": int(info["a"].shape[1]),
                     "transpose_b": bool(info["transpose_b"])})
    np.savez_compressed(REPO / "testdata" / "f5b_gemm_nodes.npz", **out)
    (REPO / "testdata" / "f5b_gemm_nodes_meta.json").write_text(
        json.dumps({"case_index": 0, "nodes": meta}, indent=1) + "\n", encoding="utf-8")
    print("written testdata/f5b_gemm_nodes.npz with", len(meta), "nodes")


if __name__ == "__main__":
    main()
