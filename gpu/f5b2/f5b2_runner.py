"""F.5B.2 — full-block GPU exactness gate runner (research-only).

Run on one GPU pod at a time (cost discipline: GPU1 first, GPU2 only if
GPU1 is fully exact):

    python gpu/f5b2/f5b2_runner.py --device cuda --out docs/phase-f5b2-gpu-results.current.json

Order (spec section 48): environment probe -> extension build -> operator
vectors (canonical + adversarial) -> wide GEMM vectors -> 83-node GEMM
regression -> full 310-node block (bytes + wide roots + final) -> audits
-> result JSON.  Any operator-vector FAIL aborts before the full block.

Cross-GPU compare (on the CPU machine):

    python gpu/f5b2/f5b2_runner.py --compare <gpuA.json> <gpuB.json>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(HERE), str(REPO / "gpu" / "f5b1"), str(REPO / "gpu" / "f5b")):
    if p not in sys.path:
        sys.path.insert(0, p)

import f5b2_block  # noqa: E402
import f5b2_ext  # noqa: E402
from f5b_gpu_runner import cuda_probe, env_capture  # noqa: E402

FLOAT_TOKENS = ["float(", "float32", "float64", "half(", "bfloat16",
                ".to(torch.float", "expf", "sqrtf", "rsqrtf", " sin(", " cos("]


def float_audit() -> dict:
    files = ["gpu/f5b2/f5b2_kernels.cu", "gpu/f5b1/f5b1_kernels.cu",
             "gpu/f5b2/f5b2_ext.py", "gpu/f5b2/f5b2_block.py"]
    hits = {}
    for rel in files:
        src = (REPO / rel).read_text(encoding="utf-8")
        hits[rel] = {t: src.count(t) for t in FLOAT_TOKENS if src.count(t)}
    flat = {t: sum(h.get(t, 0) for h in hits.values()) for t in FLOAT_TOKENS}
    return {"files": hits, "total_hits": flat,
            "canonical_float_ops": 0,
            "note": "hits are listed verbatim; the canonical path contains no "
                    "floating-point arithmetic (integer kernels + int8 "
                    "tensor-core MMA only)"}


def gemm_regression(device: str) -> dict:
    """83 real GEMM nodes (F.5B evidence) through the fused kernel."""
    import numpy as np

    ext = f5b2_ext.build()
    data = np.load(REPO / "testdata" / "f5b_gemm_nodes.npz")
    meta = json.loads((REPO / "testdata" / "f5b_gemm_nodes_meta.json")
                      .read_text(encoding="utf-8"))["nodes"]
    exact = 0
    bad = []
    for m in meta:
        nid = m["node_id"]
        a = torch.tensor(data[f"a_{nid}"], dtype=torch.int16, device=device)
        w = torch.tensor(data[f"w_{nid}"].astype(np.int64), device=device)
        expected = torch.tensor(data[f"c_{nid}"], dtype=torch.int64, device=device)
        trans = bool(m["transpose_b"])
        w_nk = (w.contiguous() if trans else w.t().contiguous())
        N, K = int(w_nk.shape[0]), int(w_nk.shape[1])
        w0, w1, ws = f5b2_ext.prepack_w10(w_nk)
        got = ext.wide_gemm(a, w0, w1, ws, N, K)
        if torch.equal(got, expected):
            exact += 1
        else:
            bad.append({"node_id": nid, "mismatches": int((got != expected).sum().item())})
    return {"nodes": len(meta), "exact": exact, "all_exact": exact == len(meta),
            "bad": bad[:10]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
        return

    device = args.device
    report = {"phase": "F.5B.2",
              "environment": env_capture(torch.device(device)),
              "cuda_probe": cuda_probe(torch.device(device)),
              "float_audit": float_audit()}
    out = Path(args.out) if args.out else REPO / "docs" / "phase-f5b2-gpu-results.json"

    if device != "cuda":
        report["note"] = "CPU dry-run: structure validation only; kernels NOT TESTED"
        out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print("CPU dry-run written:", out)
        return

    f5b2_ext.build()
    report["canonical_vectors"] = f5b2_ext.run_canonical_vectors(device)
    report["adversarial_vectors"] = f5b2_ext.run_adversarial_vectors(device)
    report["wide_gemm_vectors"] = f5b2_ext.run_wide_gemm_vectors(device)
    op_gate_ok = (report["canonical_vectors"]["all_exact"]
                  and report["adversarial_vectors"]["all_exact"]
                  and report["wide_gemm_vectors"]["all_exact"])
    report["operator_gate_pass"] = bool(op_gate_ok)
    print("operator gate:", "PASS" if op_gate_ok else "FAIL")
    for sec in ("canonical_vectors", "adversarial_vectors", "wide_gemm_vectors"):
        bad = [r for r in report[sec]["results"] if not r.get("ok")]
        for b in bad[:5]:
            print(f"  FAIL {sec}: {b}")
    if not op_gate_ok:
        report["abort"] = "operator vector gate failed — full block not run (section 33)"
        out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print("ABORTED. written:", out)
        sys.exit(2)

    report["gemm_regression"] = gemm_regression(device)
    print("GEMM regression:", report["gemm_regression"]["exact"], "/",
          report["gemm_regression"]["nodes"])
    if not report["gemm_regression"]["all_exact"]:
        report["abort"] = "GEMM regression failed — full block not run"
        out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print("ABORTED. written:", out)
        sys.exit(2)

    block = f5b2_block.execute_and_compare(device)
    report["full_block"] = block
    print("full block:", block["nodes_exact"], "/", block["nodes_total"], "nodes exact;",
          block["roots_exact"], "roots exact; final_exact =", block["final_exact"],
          "final_root_ok =", block["final_root_ok"])
    for r in block["first_bad_nodes"][:5]:
        print("  BAD:", r)
    print("audit:", json.dumps(block["audit"]))
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("written:", out)


def compare(path_a: str, path_b: str) -> None:
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))

    def blk(doc):
        return doc.get("full_block", {})

    ba, bb = blk(a), blk(b)
    gates_a = (ba.get("all_exact") and ba.get("nodes_exact") == ba.get("nodes_total")
               and a.get("operator_gate_pass") and a.get("gemm_regression", {}).get("all_exact")
               and ba.get("audit", {}).get("cpu_arithmetic_nodes") == 0
               and ba.get("audit", {}).get("unsupported_nodes") == 0)
    gates_b = (bb.get("all_exact") and bb.get("nodes_exact") == bb.get("nodes_total")
               and b.get("operator_gate_pass") and b.get("gemm_regression", {}).get("all_exact")
               and bb.get("audit", {}).get("cpu_arithmetic_nodes") == 0
               and bb.get("audit", {}).get("unsupported_nodes") == 0)
    same_graph = (ba.get("graph_id") == bb.get("graph_id"))
    cross = {
        "phase": "F.5B.2",
        "gpu_a": a.get("environment", {}).get("gpu_name"),
        "gpu_b": b.get("environment", {}).get("gpu_name"),
        "cross_arch": (a.get("environment", {}).get("compute_capability")
                       != b.get("environment", {}).get("compute_capability")),
        "graph_id": ba.get("graph_id"), "same_graph_id": same_graph,
        "nodes_exact": {"a": ba.get("nodes_exact"), "b": bb.get("nodes_exact")},
        "roots_exact": {"a": ba.get("roots_exact"), "b": bb.get("roots_exact")},
        "final_exact": {"a": ba.get("final_exact"), "b": bb.get("final_exact")},
        "operator_gate": {"a": a.get("operator_gate_pass"), "b": b.get("operator_gate_pass")},
        "gemm_regression": {"a": a.get("gemm_regression", {}).get("all_exact"),
                            "b": b.get("gemm_regression", {}).get("all_exact")},
        "fallback_audit": {"a": ba.get("audit", {}).get("cpu_arithmetic_nodes"),
                           "b": bb.get("audit", {}).get("cpu_arithmetic_nodes")},
        "latency": {"a": ba.get("audit", {}).get("wall_total_ms"),
                    "b": bb.get("audit", {}).get("wall_total_ms")},
        "gpu1_gates_pass": bool(gates_a), "gpu2_gates_pass": bool(gates_b),
    }
    if not (gates_a and gates_b):
        cross["final"] = ("FULL_BLOCK_CROSS_GPU_FAIL" if (gates_a or gates_b)
                          else "FULL_BLOCK_FAIL")
    elif not cross["cross_arch"]:
        cross["final"] = "INCONCLUSIVE_SAME_ARCHITECTURE"
    else:
        cross["final"] = "FUSED_GPU_FEASIBLE_A13W10"
    print(json.dumps(cross, indent=1))
    out = REPO / "docs" / "phase-f5b2-cross-gpu.json"
    out.write_text(json.dumps(cross, indent=1) + "\n", encoding="utf-8")
    print("written:", out)


if __name__ == "__main__":
    main()
