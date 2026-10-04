"""EXPERIMENTAL / NON-PROTOCOL — F.5B GPU runner (A13W10 exactness + bench).

Run on each GPU pod:

    python gpu/f5b/f5b_gpu_runner.py --device cuda --out docs/phase-f5b-gpu-results.<gpu>.json

Then compare the two pods' result files on the CPU machine:

    python gpu/f5b/f5b_gpu_runner.py --compare docs/phase-f5b-gpu-results.A.json docs/phase-f5b-gpu-results.B.json

Checks performed per GPU:
  * wide vectors: CPU direct == CPU radix == GPU (every backend);
  * all 83 real GEMM nodes (testdata/f5b_gemm_nodes.npz): every backend
    bit-identical to the CPU int64 accumulator, with per-node timing;
  * lightweight real-shape benchmark: native int8 tensor-core baseline vs
    SIMT / Karatsuba3 / DP2A (median/p95/min over >=100 timed iterations
    with CUDA events), plus the weighted 83-node schedule ratio;
  * environment capture (GPU, compute capability, driver, CUDA, torch);
  * float audit of the canonical backend source.

A CPU dry-run (--device cpu) validates the harness logic; a `_cpudry`
result is never used for the GPU verdict.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gpu" / "f5b"))

from f5b_gpu_backends import (  # noqa: E402
    Dp2aWSplit2, SimtInt64Reference, TcKaratsuba3, split_radix128, source_float_audit,
)

PERF_GATE = 6.0
MAJOR_SHAPE_FRACTION = 0.05
MAJOR_SHAPE_MAX_RATIO = 10.0


def env_capture(device: torch.device) -> dict:
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        env.update({
            "gpu_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "total_memory_gb": round(props.total_memory / (1 << 30), 1),
            "device_index": device.index,
        })
        try:
            import subprocess
            env["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            env["nvidia_smi"] = f"unavailable: {exc}"
    return env


def timed(fn, device: torch.device, warmup: int = 20, iters: int = 100) -> dict:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    else:
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {"median_ms": statistics.median(times), "p95_ms": times[int(len(times) * 0.95)],
            "min_ms": times[0], "iterations": iters, "warmup": warmup}


def run_vectors(device: torch.device) -> dict:
    vectors = json.loads((REPO / "testdata" / "f5b_wide_gemm_vectors.json")
                         .read_text(encoding="utf-8"))["vectors"]
    results = []
    for v in vectors:
        a = torch.tensor(v["a"], dtype=torch.int64, device=device).reshape(v["a_shape"])
        w = torch.tensor(v["w"], dtype=torch.int64, device=device).reshape(v["w_shape"])
        expected = torch.tensor(v["cpu_direct"], dtype=torch.int64,
                                device=device).reshape(v["cpu_direct_shape"])
        entry = {"name": v["name"], "backends": {}}
        for backend_name, backend in (("GPU_SIMT_INT64_REFERENCE", SimtInt64Reference()),
                                      ("GPU_TC_KARATSUBA3", TcKaratsuba3()),
                                      ("GPU_DP2A_W_SPLIT2", Dp2aWSplit2())):
            try:
                got = backend.run(a, w, v["transpose_b"])
                entry["backends"][backend_name] = {
                    "exact": bool(torch.equal(got, expected)),
                    "mismatches": int((got != expected).sum().item()),
                }
            except Exception as exc:  # noqa: BLE001  explicit, never silent
                entry["backends"][backend_name] = {
                    "exact": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        results.append(entry)
    return {"vectors": results,
            "all_exact": all(b["exact"] for r in results for b in r["backends"].values()),
            "note": "GPU outputs compared elementwise against the CPU int64 oracle"}


def run_nodes(device: torch.device) -> dict:
    data = np.load(REPO / "testdata" / "f5b_gemm_nodes.npz")
    meta = json.loads((REPO / "testdata" / "f5b_gemm_nodes_meta.json")
                      .read_text(encoding="utf-8"))["nodes"]
    per_backend = {"GPU_SIMT_INT64_REFERENCE": {"exact": 0, "mismatch": 0, "nodes": []},
                   "GPU_TC_KARATSUBA3": {"exact": 0, "mismatch": 0, "nodes": []},
                   "GPU_DP2A_W_SPLIT2": {"exact": 0, "mismatch": 0, "nodes": []}}
    for m in meta:
        nid = m["node_id"]
        a = torch.tensor(data[f"a_{nid}"].astype(np.int64), device=device)
        w = torch.tensor(data[f"w_{nid}"].astype(np.int64), device=device)
        expected = torch.tensor(data[f"c_{nid}"], device=device)
        entry = {"node_id": nid, "M": m["M"], "N": m["N"], "K": m["K"],
                 "transpose_b": m["transpose_b"]}
        for name, backend in (("GPU_SIMT_INT64_REFERENCE", SimtInt64Reference()),
                              ("GPU_TC_KARATSUBA3", TcKaratsuba3()),
                              ("GPU_DP2A_W_SPLIT2", Dp2aWSplit2())):
            try:
                got = backend.run(a, w, m["transpose_b"])
                mm = int((got != expected).sum().item())
                per_backend[name]["exact" if mm == 0 else "mismatch"] += 1
                t = timed(lambda: backend.run(a, w, m["transpose_b"]), device)
                per_backend[name]["nodes"].append({**entry, "median_ms": t["median_ms"]})
            except Exception as exc:  # noqa: BLE001
                per_backend[name]["mismatch"] += 1
                per_backend[name]["nodes"].append({**entry,
                                                   "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
    summary = {k: {"exact_nodes": v["exact"], "mismatch_nodes": v["mismatch"]}
               for k, v in per_backend.items()}
    return {"summary": summary, "detail": per_backend,
            "all_exact": all(v["mismatch"] == 0 for v in per_backend.values())}


def run_benchmark(device: torch.device) -> dict:
    shapes_doc = json.loads((REPO / "docs" / "phase-f5b-gemm-shapes.json")
                            .read_text(encoding="utf-8"))
    freq = {}
    for row in shapes_doc["nodes"]:
        key = (row["M"], row["N"], row["K"], row["transpose_b"])
        freq[key] = freq.get(key, 0) + 1
    total_mac = sum(r["logical_mac"] for r in shapes_doc["nodes"])
    results = {}
    for (m, n, k, trans), count in sorted(freq.items(), key=lambda kv: -kv[1]):
        g = torch.Generator(device="cpu").manual_seed(m * 1000003 + n * 10007 + k)
        a13 = torch.randint(-4096, 4096, (m, k), generator=g).to(device)
        w10 = torch.randint(-512, 512, (n, k) if trans else (k, n), generator=g).to(device)
        a8 = torch.randint(-128, 128, (m, k), generator=g).to(torch.int8).to(device)
        w8 = (torch.randint(-128, 128, (n, k), generator=g).to(torch.int8).to(device)
              if trans else torch.randint(-128, 128, (k, n), generator=g).to(torch.int8).to(device))
        entry = {"M": m, "N": n, "K": k, "transpose_b": trans, "schedule_frequency": count,
                 "logical_mac": m * n * k,
                 "mac_fraction": (m * n * k * count) / total_mac}
        # native int8 baseline on the same shapes/launch environment
        packed = TcKaratsuba3().prepack_weights(w10, trans)
        packed2 = Dp2aWSplit2().prepack_weights(w10, trans)
        def try_timed(fn):
            try:
                return timed(fn, device)
            except Exception as exc:  # noqa: BLE001
                return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        entry["native_int8"] = try_timed(lambda: torch._int_mm(a8, w8.T if trans else w8))
        entry["karatsuba3"] = try_timed(lambda: TcKaratsuba3().run(a13, w10, trans, packed))
        entry["simt_int64"] = try_timed(lambda: SimtInt64Reference().run(a13, w10, trans))
        entry["dp2a_w_split2"] = try_timed(lambda: Dp2aWSplit2().run(a13, w10, trans, packed2))
        if "error" in entry["native_int8"] or "error" in entry["karatsuba3"]:
            entry["karatsuba3_ratio"] = None
            entry["dp2a_ratio"] = None
            entry["simt_ratio"] = None
            results[f"{m}x{n}x{k}{'T' if trans else ''}"] = entry
            continue
        entry["native_int8_median_ms"] = entry["native_int8"]["median_ms"]
        entry["karatsuba3_ratio"] = entry["karatsuba3"]["median_ms"] / max(entry["native_int8_median_ms"], 1e-9)
        entry["dp2a_ratio"] = entry["dp2a_w_split2"]["median_ms"] / max(entry["native_int8_median_ms"], 1e-9)
        entry["simt_ratio"] = entry["simt_int64"]["median_ms"] / max(entry["native_int8_median_ms"], 1e-9)
        entry["karatsuba3_ratio"] = entry["karatsuba3"]["median_ms"] / max(entry["native_int8_median_ms"], 1e-9)
        entry["dp2a_ratio"] = entry["dp2a_w_split2"]["median_ms"] / max(entry["native_int8_median_ms"], 1e-9)
        entry["simt_ratio"] = entry["simt_int64"]["median_ms"] / max(entry["native_int8_median_ms"], 1e-9)
        results[f"{m}x{n}x{k}{'T' if trans else ''}"] = entry
    # weighted schedule ratio (KARATSUBA3 vs native int8)
    timing_rows = [e for e in results.values() if e.get("karatsuba3_ratio") is not None]
    wide_total = sum(e["karatsuba3"]["median_ms"] * e["schedule_frequency"] for e in timing_rows)
    native_total = sum(e["native_int8"]["median_ms"] * e["schedule_frequency"] for e in timing_rows)
    major_check = [{"shape": name, "ratio": e["karatsuba3_ratio"]}
                   for name, e in results.items()
                   if e["mac_fraction"] >= MAJOR_SHAPE_FRACTION and e.get("karatsuba3_ratio") is not None]
    return {
        "shapes": results,
        "weighted_schedule": {
            "karatsuba3_total_ms": wide_total, "native_int8_total_ms": native_total,
            "ratio": wide_total / max(native_total, 1e-9)},
        "major_shape_ratios": major_check,
        "performance_pass": bool(wide_total / max(native_total, 1e-9) <= PERF_GATE
                                 and all(m["ratio"] <= MAJOR_SHAPE_MAX_RATIO for m in major_check)),
        "gates": {"schedule_ratio_max": PERF_GATE, "major_shape_ratio_max": MAJOR_SHAPE_MAX_RATIO},
        "physical_mac_note": "Karatsuba3 performs ~3x physical int8 MAC per logical MAC; "
                             "protocol work accounting remains 1x logical (F.5B section 23/24)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    parser.add_argument("--compare", nargs=2, default=None,
                        help="two per-GPU result files; prints the cross-GPU verdict")
    args = parser.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        b = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        cross_nodes = "NOT TESTED"
        if a.get("nodes", {}).get("all_exact") and b.get("nodes", {}).get("all_exact"):
            cross_nodes = "both GPUs exact vs CPU; cross-GPU equality follows from bit-exactness"
        verdict = {
            "gpu_a": a.get("environment", {}).get("gpu_name"),
            "gpu_b": b.get("environment", {}).get("gpu_name"),
            "cross_arch": (a.get("environment", {}).get("compute_capability")
                           != b.get("environment", {}).get("compute_capability")),
            "nodes_exact": {"a": a.get("nodes", {}).get("all_exact"),
                            "b": b.get("nodes", {}).get("all_exact")},
            "vectors_exact": {"a": a.get("vectors", {}).get("all_exact"),
                              "b": b.get("vectors", {}).get("all_exact")},
            "performance_pass": {"a": a.get("benchmark", {}).get("performance_pass"),
                                 "b": b.get("benchmark", {}).get("performance_pass")},
            "cross_gpu": cross_nodes,
        }
        ok = (verdict["nodes_exact"]["a"] and verdict["nodes_exact"]["b"]
              and verdict["vectors_exact"]["a"] and verdict["vectors_exact"]["b"]
              and verdict["gpu_a"] != verdict["gpu_b"])
        if not ok:
            verdict["final"] = "GPU_NOT_FEASIBLE_OR_INCONCLUSIVE"
        elif verdict["performance_pass"]["a"] and verdict["performance_pass"]["b"]:
            verdict["final"] = "GPU_FEASIBLE_A13W10"
        else:
            verdict["final"] = "GPU_CORRECT_BUT_NOT_PRACTICAL"
        print(json.dumps(verdict, indent=1))
        out = REPO / "docs" / "phase-f5b-final-verdict.json"
        out.write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8")
        print("written:", out)
        return

    device = torch.device(args.device)
    report = {"backend_used": "TC_RADIX128_KARATSUBA3 (primary), SIMT + DP2A alternatives",
              "environment": env_capture(device),
              "float_audit": source_float_audit()}
    report["vectors"] = run_vectors(device)
    report["nodes"] = run_nodes(device)
    report["benchmark"] = run_benchmark(device)
    report["full_block_gpu"] = "NOT TESTED (gated on GEMM performance PASS, F.5B section 62)"
    out = Path(args.out) if args.out else REPO / "docs" / "phase-f5b-gpu-results.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("vectors all exact:", report["vectors"]["all_exact"])
    print("nodes:", report["nodes"]["summary"])
    print("schedule ratio:", report["benchmark"]["weighted_schedule"]["ratio"])
    print("performance_pass:", report["benchmark"]["performance_pass"])
    print("written:", out)


if __name__ == "__main__":
    main()
