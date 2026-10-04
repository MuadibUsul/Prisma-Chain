"""F.5B.2 — full-block GPU executor (research-only).

Interprets testdata/f5b2_block_program.json over testdata/
f5b2_block_bundle.npz entirely on the GPU: every node dispatches a CUDA
kernel (the 7 canonical op kernels + the F.5B.1 fused A13W10 m16n8k32
GEMM); all intermediates stay GPU-resident; weights and constants are
uploaded once up front.  CPU does control flow and (after execution)
root/hash comparison only — no CPU arithmetic fallback exists on any
path (UNSUPPORTED_GPU_OPERATOR aborts instead).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import f5b2_ext  # noqa: E402


class UnsupportedGpuOperator(RuntimeError):
    pass


class GpuBlockExecutor:
    def __init__(self, program: dict, bundle_path: Path, device: str = "cuda"):
        self.ext = f5b2_ext.build()
        self.program = program
        self.device = device
        self.gpu_nodes_executed = 0
        self.cpu_arithmetic_nodes = 0
        self.unsupported_nodes = 0
        self.consts: dict[int, torch.Tensor] = {}
        for c in program["consts"]:
            arr = np.load(bundle_path)[f"const_{c['id']}"]
            self.consts[c["id"]] = torch.tensor(arr, dtype=torch.int64, device=device)
        self.gemm_packs: dict[int, tuple] = {}

    def _gemm_pack(self, const_idx: int, transpose_b: bool) -> tuple:
        key = (const_idx, bool(transpose_b))
        if key not in self.gemm_packs:
            w = self.consts[const_idx]
            w_nk = (w.contiguous() if transpose_b else w.t().contiguous())
            self.gemm_packs[key] = f5b2_ext.prepack_w10(w_nk)
        return self.gemm_packs[key]

    def run(self) -> tuple[list[torch.Tensor], dict]:
        tensors: list[torch.Tensor | None] = [None] * len(self.program["nodes"])
        t_total = time.perf_counter()
        gemm_ms = 0.0
        other_ms = 0.0
        for n in self.program["nodes"]:
            ins = [self.consts[r["index"]] if r["kind"] == "const" else tensors[r["index"]]
                   for r in n["inputs"]]
            op, params, shape = n["op"], n["params"], tuple(n["shape"])
            t0 = time.perf_counter()
            if op == "GEMM":
                trans = bool(params.get("transpose_b", False))
                a = ins[0].to(torch.int16).contiguous()
                assert int(a.min()) >= -4096 and int(a.max()) <= 4095, \
                    f"node {n['seq']}: A13 container range violated"
                w0, w1, ws = self._gemm_pack(n["inputs"][1]["index"], trans)
                N, K = int(self.consts[n["inputs"][1]["index"]].shape[1] if not trans
                           else self.consts[n["inputs"][1]["index"]].shape[0]), int(a.shape[1])
                out = self.ext.wide_gemm(a, w0, w1, ws, N, K)
                gemm_ms += (time.perf_counter() - t0) * 1e3
            elif op == "REQUANTIZE":
                out = self.ext.op_requant(ins[0], params["mult"], params["shift"],
                                          params["lo"], params["hi"])
                other_ms += (time.perf_counter() - t0) * 1e3
            elif op == "ADD":
                out = self.ext.op_add(ins[0], ins[1])
                other_ms += (time.perf_counter() - t0) * 1e3
            elif op == "MUL":
                out = self.ext.op_mul(ins[0], ins[1])
                other_ms += (time.perf_counter() - t0) * 1e3
            elif op == "RMSNORM":
                out = self.ext.op_rmsnorm(ins[0].reshape(shape), ins[1], params["eps_fx"])
                other_ms += (time.perf_counter() - t0) * 1e3
            elif op == "ROPE":
                out = self.ext.op_rope(ins[0].reshape(shape), ins[1], params["half_dim"])
                other_ms += (time.perf_counter() - t0) * 1e3
            elif op == "SILU":
                out = self.ext.op_silu(ins[0].reshape(shape))
                other_ms += (time.perf_counter() - t0) * 1e3
            elif op == "SOFTMAX":
                out = self.ext.op_softmax(ins[0].reshape(shape))
                other_ms += (time.perf_counter() - t0) * 1e3
            else:
                self.unsupported_nodes += 1
                raise UnsupportedGpuOperator(
                    f"UNSUPPORTED_GPU_OPERATOR: {op} at node {n['seq']} (no CPU fallback)")
            self.gpu_nodes_executed += 1
            tensors[n["seq"]] = out.reshape(shape)
        wall_ms = (time.perf_counter() - t_total) * 1e3
        audit = {"gpu_nodes_executed": self.gpu_nodes_executed,
                 "cpu_arithmetic_nodes": self.cpu_arithmetic_nodes,
                 "unsupported_nodes": self.unsupported_nodes,
                 "wall_total_ms": round(wall_ms, 3),
                 "wall_gemm_ms": round(gemm_ms, 3),
                 "wall_non_gemm_ms": round(other_ms, 3),
                 "note": "wall-clock host timings include dispatch; recorded per "
                         "spec section 55/88 with no acceptance gate"}
        return tensors, audit


def execute_and_compare(device: str = "cuda") -> dict:
    program = json.loads((REPO / "testdata" / "f5b2_block_program.json")
                         .read_text(encoding="utf-8"))
    manifest = np.load(REPO / "testdata" / "f5b2_cpu_manifest.npz")
    roots = json.loads((REPO / "testdata" / "f5b2_cpu_roots.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(REPO / "tools"))
    from f5b2_dump_block_program import wide_node_root

    ex = GpuBlockExecutor(program, REPO / "testdata" / "f5b2_block_bundle.npz", device)
    tensors, audit = ex.run()

    from collections import Counter
    per_op = {}
    node_rows = []
    all_exact = True
    for n in program["nodes"]:
        got = tensors[n["seq"]].detach().cpu().numpy().astype(np.int64)
        want = manifest[f"out_{n['seq']}"].reshape(got.shape)
        mism = int((got != want).sum())
        first = None if mism == 0 else np.argwhere(got != want)[0].tolist()
        root = wide_node_root(got)
        root_ok = root == roots["nodes"][n["seq"]]["root"]
        op = n["op"]
        st = per_op.setdefault(op, {"nodes": 0, "exact": 0, "mismatch": 0, "root_mismatch": 0})
        st["nodes"] += 1
        if mism == 0 and root_ok:
            st["exact"] += 1
        else:
            all_exact = False
            if mism:
                st["mismatch"] += 1
            if not root_ok:
                st["root_mismatch"] += 1
        node_rows.append({"seq": n["seq"], "node_id": n["node_id"], "op": op,
                          "exact": mism == 0 and root_ok, "mismatches": mism,
                          "first_mismatch": first, "root_ok": root_ok})
    final = tensors[-1].detach().cpu().numpy().astype(np.int64)
    final_exact = bool(np.array_equal(final, np.asarray(roots["final_output"], dtype=np.int64)))
    final_root_ok = wide_node_root(final) == roots["final_root"]
    return {"nodes_total": len(program["nodes"]),
            "nodes_exact": sum(1 for r in node_rows if r["exact"]),
            "roots_exact": sum(1 for r in node_rows if r["root_ok"]),
            "final_exact": final_exact, "final_root_ok": final_root_ok,
            "per_operator": per_op, "all_exact": bool(all_exact),
            "graph_id": program["graph_id"], "policy_id": program["policy_id"],
            "audit": audit, "first_bad_nodes": [r for r in node_rows if not r["exact"]][:10]}
