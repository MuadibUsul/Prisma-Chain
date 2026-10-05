"""F.5C shared V2 graph executor (research/tooling).

executes a GraphDescriptorV2 document (the wire JSON form) over its input
arrays with the FROZEN f1_canonical_numpy kernels, optionally injecting a
self-consistent tamper (one element of one GEMM output changed, every
downstream node recomputed from the corrupted tensor).

This is the single execution path used by the bundle builder, the fraud
artifact generator and the watcher's independent recomputation — worker and
watcher therefore share the frozen semantics, never a hand-copied kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import f1_canonical_numpy as N  # noqa: E402
import canonical_ref as R  # noqa: E402


def execute_v2(graph_doc: dict, inputs: dict, tamper: tuple | None = None) -> dict:
    """Returns {node_id: np.ndarray}.  tamper = (node_id, flat_index, delta)
    injects the corruption into that node's output before executing the
    downstream nodes from it."""
    tensors: dict[tuple[int, int], np.ndarray] = {}
    for i, entry in enumerate(graph_doc["inputs"]):
        tensors[(0, i)] = np.asarray(inputs[i], dtype=np.int64).reshape(entry["desc"]["shape"])
    out: dict[int, np.ndarray] = {}
    for node in graph_doc["nodes"]:
        ins = [tensors[(r["kind"], r["index"])] for r in node["inputs"]]
        op = node["operator_id"]
        params = {p["key"]: p["value"] for p in node["params"]}
        shape = tuple(int(d) for d in node["output"]["shape"])
        if op == "GEMM_A13W10_I64_V1":
            a = ins[0].astype(np.int64)
            w = ins[1].astype(np.int64)
            trans = params.get("transpose_b", 0) == 1
            t = (a @ w.T) if trans else (a @ w)
        elif op == "REQUANTIZE_WIDE_V1":
            t = np.array(R.requant_wide(ins[0].reshape(-1).tolist(), int(params["mult"]),
                                        int(params["shift"]), int(params["clamp_lo"]),
                                        int(params["clamp_hi"]), int(params["out_dtype"])),
                         dtype=np.int64)
        elif op == "ADD_FIXED_V1":
            t = N.add_fx(ins[0], ins[1]).astype(np.int64)
        elif op == "MUL_FIXED_V1":
            t = N.mul_fx(ins[0], ins[1]).astype(np.int64)
        elif op == "RMSNORM_FIXED_V1":
            t = N.op_rmsnorm(ins[0].reshape(shape), ins[1], int(params["eps_fx"])).astype(np.int64)
        elif op == "ROPE_FIXED_V1":
            t = N.op_rope_adjacent(ins[0].astype(np.int32).reshape(shape),
                                   np.asarray(ins[1]), int(params["half_dim"])).astype(np.int64)
        elif op == "SILU_FIXED_V1":
            t = N.op_silu(ins[0].astype(np.int32).reshape(shape)).astype(np.int64)
        elif op == "SOFTMAX_FIXED_V1":
            t = N.op_softmax_rows(ins[0].astype(np.int32).reshape(shape)).astype(np.int64)
        else:
            raise ValueError(f"unknown operator {op}")
        t = np.asarray(t, dtype=np.int64).reshape(shape)
        if tamper is not None and int(node["node_id"]) == tamper[0]:
            flat = t.reshape(-1).copy()
            flat[tamper[1]] += tamper[2]
            t = flat.reshape(shape)
        tensors[(1, int(node["node_id"]))] = t
        out[int(node["node_id"])] = t
    return out


def load_inputs_from_bundle(graph_doc: dict, bundle_npz: Path) -> dict:
    """The F.5B.2 const bundle keyed by const id, ordered by the V2 doc."""
    data = np.load(bundle_npz)
    inputs = {}
    for i, entry in enumerate(graph_doc["inputs"]):
        cid = int(entry["name"].split("_")[1])
        inputs[i] = data[f"const_{cid}"].reshape(entry["desc"]["shape"])
    return inputs
