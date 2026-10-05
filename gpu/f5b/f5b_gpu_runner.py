"""EXPERIMENTAL / NON-PROTOCOL — F.5B GPU runner (A13W10 exactness + bench).

Run on each GPU pod:

    python gpu/f5b/f5b_gpu_runner.py --device cuda --out docs/phase-f5b-gpu-results.<gpu>.json

Then compare the two pods' result files on the CPU machine:

    python gpu/f5b/f5b_gpu_runner.py --compare docs/phase-f5b-gpu-results.A.json docs/phase-f5b-gpu-results.B.json

Checks performed per GPU:
  * platform probe: int64 floor-division and the torch._int_mm minimal-M
    constraint (verbatim evidence, no silent padding assumptions);
  * wide vectors: CPU direct == CPU radix == GPU;
  * all 83 real GEMM nodes (testdata/f5b_gemm_nodes.npz): every executable
    backend bit-identical to the CPU int64 accumulator, with per-node timing;
  * lightweight real-shape benchmark: native int8 tensor-core baseline vs
    Karatsuba3 / Schoolbook4 (median/p95/min over >=100 timed iterations
    with CUDA events), plus the weighted 83-node schedule ratio.  The
    performance gate requires exactness AND every shape measured AND
    positive totals AND the ratio limits — it cannot pass vacuously;
  * environment capture (GPU, compute capability, driver, CUDA, torch);
  * float audit of the canonical backend source.

Direct int64 / int16 tensor-core-free backends (SIMT reference, DP2A) are
CPU-executable only: CUDA has no generic integer matmul primitive, and the
captured `addmm_cuda ... not implemented for 'Long'` error is recorded
verbatim in the results as platform evidence.  GPU_SCHOOLBOOK4 (independent
merge formula from Karatsuba3) fills the on-hardware reference role.

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
    ALL_BACKEND_NAMES, CUDA_BACKENDS, Dp2aWSplit2, Schoolbook4,
    SimtInt64Reference, TcKaratsuba3, pad_role, source_float_audit,
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


def cuda_probe(device: torch.device) -> dict:
    """Verbatim platform facts the padding scheme relies on."""
    if device.type != "cuda":
        return {"note": "CPU dry-run; CUDA probes skipped"}
    res: dict = {}
    x = torch.tensor([[-(1 << 12), -1, 0, (1 << 12) - 1]], dtype=torch.int64, device=device)
    got = torch.div(x + 64, 128, rounding_mode="floor")
    want = torch.tensor([[(v + 64) // 128 for v in [-4096, -1, 0, 4095]]],
                        dtype=torch.int64, device=device)
    res["floor_div_int64_ok"] = bool(torch.equal(got, want))
    probe = {}
    for m_try in (16, 17, 24, 32):
        try:
            a = torch.ones((m_try, 16), dtype=torch.int8, device=device)
            b = torch.ones((16, 16), dtype=torch.int8, device=device)
            torch._int_mm(a, b)
            probe[str(m_try)] = "ok"
        except Exception as exc:  # noqa: BLE001  evidence, not a failure
            probe[str(m_try)] = f"{type(exc).__name__}: {str(exc)[:160]}"
    res["int_mm_m_rows_probe"] = probe
    res["padding_rule"] = "M -> max(32, align16); K -> max(16, align16); N -> max(16, align16)"
    res["padding_rule_justification"] = (
        "torch._int_mm on CUDA requires left-operand M > 16 (see probe); "
        "zero padding is mathematically inert so exactness is preserved")
    return res


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


def _backend_summary(per_backend: dict) -> dict:
    summary = {}
    for name, v in per_backend.items():
        summary[name] = {"exact_nodes": v["exact"], "mismatch_nodes": v["mismatch"],
                         "error_nodes": v["error"], "executable": v["error"] == 0}
        if v["first_error"]:
            summary[name]["first_error"] = v["first_error"]
    return summary


def run_vectors(device: torch.device) -> dict:
    vectors = json.loads((REPO / "testdata" / "f5b_wide_gemm_vectors.json")
                         .read_text(encoding="utf-8"))["vectors"]
    results = []
    per_backend = {n: {"exact": 0, "mismatch": 0, "error": 0, "first_error": None}
                   for n in ALL_BACKEND_NAMES}
    for v in vectors:
        a = torch.tensor(v["a"], dtype=torch.int64, device=device).reshape(v["a_shape"])
        w = torch.tensor(v["w"], dtype=torch.int64, device=device).reshape(v["w_shape"])
        expected = torch.tensor(v["cpu_direct"], dtype=torch.int64,
                                device=device).reshape(v["cpu_direct_shape"])
        entry = {"name": v["name"], "backends": {}}
        for name in ALL_BACKEND_NAMES:
            backend = {"GPU_TC_KARATSUBA3": TcKaratsuba3, "GPU_SCHOOLBOOK4": Schoolbook4,
                       "GPU_SIMT_INT64_REFERENCE": SimtInt64Reference,
                       "GPU_DP2A_W_SPLIT2": Dp2aWSplit2}[name]()
            try:
                got = backend.run(a, w, v["transpose_b"])
                mm = int((got != expected).sum().item())
                entry["backends"][name] = {"exact": mm == 0, "mismatches": mm}
                per_backend[name]["exact" if mm == 0 else "mismatch"] += 1
            except Exception as exc:  # noqa: BLE001  explicit, never silent
                msg = f"{type(exc).__name__}: {str(exc)[:200]}"
                entry["backends"][name] = {"exact": False, "error": msg}
                per_backend[name]["error"] += 1
                per_backend[name]["first_error"] = per_backend[name]["first_error"] or msg
        results.append(entry)
    summary = _backend_summary(per_backend)
    executable = [n for n in ALL_BACKEND_NAMES if summary[n]["executable"]]
    required_ok = all(summary[n]["executable"] and summary[n]["exact_nodes"] == len(vectors)
                      and summary[n]["error_nodes"] == 0 for n in CUDA_BACKENDS)
    return {"vectors": results, "summary": summary, "executable_backends": executable,
            "required_cuda_backends_exact": bool(required_ok),
            "all_exact": all(summary[n]["mismatch_nodes"] == 0
                             and summary[n]["exact_nodes"] == len(vectors)
                             for n in ALL_BACKEND_NAMES),
            "note": "GPU outputs compared elementwise against the CPU int64 oracle"}


def run_nodes(device: torch.device) -> dict:
    data = np.load(REPO / "testdata" / "f5b_gemm_nodes.npz")
    meta = json.loads((REPO / "testdata" / "f5b_gemm_nodes_meta.json")
                      .read_text(encoding="utf-8"))["nodes"]
    total = len(meta)
    per_backend = {n: {"exact": 0, "mismatch": 0, "error": 0,
                       "first_error": None, "nodes": []}
                   for n in ALL_BACKEND_NAMES}
    for m in meta:
        nid = m["node_id"]
        a = torch.tensor(data[f"a_{nid}"].astype(np.int64), device=device)
        w = torch.tensor(data[f"w_{nid}"].astype(np.int64), device=device)
        expected = torch.tensor(data[f"c_{nid}"], device=device)
        entry = {"node_id": nid, "M": m["M"], "N": m["N"], "K": m["K"],
                 "transpose_b": m["transpose_b"]}
        for name in ALL_BACKEND_NAMES:
            backend = {"GPU_TC_KARATSUBA3": TcKaratsuba3, "GPU_SCHOOLBOOK4": Schoolbook4,
                       "GPU_SIMT_INT64_REFERENCE": SimtInt64Reference,
                       "GPU_DP2A_W_SPLIT2": Dp2aWSplit2}[name]()
            try:
                got = backend.run(a, w, m["transpose_b"])
                mm = int((got != expected).sum().item())
                per_backend[name]["exact" if mm == 0 else "mismatch"] += 1
                t = timed(lambda: backend.run(a, w, m["transpose_b"]), device)
                per_backend[name]["nodes"].append({**entry, "median_ms": t["median_ms"]})
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {str(exc)[:200]}"
                per_backend[name]["error"] += 1
                per_backend[name]["first_error"] = per_backend[name]["first_error"] or msg
                per_backend[name]["nodes"].append({**entry, "error": msg})
    summary = _backend_summary(per_backend)
    executable = [n for n in ALL_BACKEND_NAMES if summary[n]["executable"]]
    required_ok = all(summary[n]["executable"] and summary[n]["mismatch_nodes"] == 0
                      and summary[n]["exact_nodes"] == total for n in CUDA_BACKENDS)
    return {"summary": summary, "detail": per_backend, "executable_backends": executable,
            "required_cuda_backends_exact": bool(required_ok),
            "all_exact": all(summary[n]["mismatch_nodes"] == 0
                             and summary[n]["exact_nodes"] == total
                             for n in ALL_BACKEND_NAMES)}


def run_benchmark(device: torch.device, exact_ok: bool) -> dict:
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
        # identical role-based zero padding for the wide paths and the native
        # baseline (torch._int_mm requires left-operand M > 16 on CUDA)
        m_pad, k_pad, n_pad = pad_role(m, k, n)

        def pad_copy(t: torch.Tensor, rows: int, cols: int, dtype) -> torch.Tensor:
            out = torch.zeros((rows, cols), dtype=dtype, device=device)
            out[: t.shape[0], : t.shape[1]] = t
            return out

        a13p = pad_copy(a13, m_pad, k_pad, torch.int64)
        w10p = (pad_copy(w10, n_pad, k_pad, torch.int64) if trans
                else pad_copy(w10, k_pad, n_pad, torch.int64))
        a8p = pad_copy(torch.randint(-128, 128, (m, k), generator=g), m_pad, k_pad, torch.int8)
        w8p = (pad_copy(torch.randint(-128, 128, (n, k), generator=g), n_pad, k_pad, torch.int8)
               if trans else
               pad_copy(torch.randint(-128, 128, (k, n), generator=g), k_pad, n_pad, torch.int8))
        base_b = w8p.T.contiguous() if trans else w8p
        packed = TcKaratsuba3().prepack_weights(w10p, trans)
        entry = {"M": m, "N": n, "K": k, "transpose_b": trans, "schedule_frequency": count,
                 "logical_mac": m * n * k,
                 "mac_fraction": (m * n * k * count) / total_mac,
                 "padded_shapes": {"m": m_pad, "k": k_pad, "n": n_pad}}

        def try_timed(fn):
            try:
                return timed(fn, device)
            except Exception as exc:  # noqa: BLE001
                return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}

        entry["native_int8"] = try_timed(lambda: torch._int_mm(a8p, base_b))
        entry["karatsuba3"] = try_timed(lambda: TcKaratsuba3().run(a13p, w10p, trans, packed))
        entry["schoolbook4"] = try_timed(lambda: Schoolbook4().run(a13p, w10p, trans, packed))
        entry["simt_int64"] = try_timed(lambda: SimtInt64Reference().run(a13p, w10p, trans))
        entry["dp2a_w_split2"] = try_timed(lambda: Dp2aWSplit2().run(a13p, w10p, trans))
        for wide in ("karatsuba3", "schoolbook4", "simt_int64", "dp2a_w_split2"):
            if "error" not in entry[wide] and "error" not in entry["native_int8"]:
                entry[f"{wide}_ratio"] = (entry[wide]["median_ms"]
                                          / max(entry["native_int8"]["median_ms"], 1e-9))
        results[f"{m}x{n}x{k}{'T' if trans else ''}"] = entry

    measured = [e for e in results.values()
                if "error" not in e["native_int8"] and "error" not in e["karatsuba3"]
                and "error" not in e["schoolbook4"]]
    measured_ok = len(measured) == len(results) and len(results) > 0
    wide_total = sum(e["karatsuba3"]["median_ms"] * e["schedule_frequency"] for e in measured)
    schoolbook_total = sum(e["schoolbook4"]["median_ms"] * e["schedule_frequency"]
                           for e in measured)
    native_total = sum(e["native_int8"]["median_ms"] * e["schedule_frequency"] for e in measured)
    ratio = wide_total / native_total if native_total > 0 else None
    major_check = [{"shape": name, "ratio": e["karatsuba3_ratio"]}
                   for name, e in results.items()
                   if e["mac_fraction"] >= MAJOR_SHAPE_FRACTION
                   and e.get("karatsuba3_ratio") is not None]
    ratio_ok = bool(measured_ok and native_total > 0 and wide_total > 0
                    and ratio is not None and ratio <= PERF_GATE)
    major_ok = bool(measured_ok and len(major_check) > 0
                    and all(m["ratio"] <= MAJOR_SHAPE_MAX_RATIO for m in major_check))
    return {
        "shapes": results,
        "weighted_schedule": {
            "karatsuba3_total_ms": wide_total, "schoolbook4_total_ms": schoolbook_total,
            "native_int8_total_ms": native_total, "ratio": ratio,
            "schoolbook4_ratio": (schoolbook_total / native_total
                                  if native_total > 0 else None)},
        "major_shape_ratios": major_check,
        "performance_pass": bool(exact_ok and ratio_ok and major_ok),
        "performance_pass_components": {
            "exactness_ok": bool(exact_ok), "all_shapes_measured": bool(measured_ok),
            "ratio_ok": ratio_ok, "major_shapes_ok": major_ok,
            "note": "the gate cannot pass vacuously: exactness, measured shapes "
                    "and positive totals are all required"},
        "gates": {"schedule_ratio_max": PERF_GATE, "major_shape_ratio_max": MAJOR_SHAPE_MAX_RATIO,
                  "major_shape_fraction": MAJOR_SHAPE_FRACTION},
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

        def req(doc, section):
            return bool(doc.get(section, {}).get("required_cuda_backends_exact"))

        verdict = {
            "gpu_a": a.get("environment", {}).get("gpu_name"),
            "gpu_b": b.get("environment", {}).get("gpu_name"),
            "cross_arch": (a.get("environment", {}).get("compute_capability")
                           != b.get("environment", {}).get("compute_capability")),
            "nodes_required_exact": {"a": req(a, "nodes"), "b": req(b, "nodes")},
            "vectors_required_exact": {"a": req(a, "vectors"), "b": req(b, "vectors")},
            "executable_backends": {"a": a.get("nodes", {}).get("executable_backends"),
                                    "b": b.get("nodes", {}).get("executable_backends")},
            "platform_limited_backends": {
                name: {
                    "a": a.get("nodes", {}).get("summary", {}).get(name, {}).get("first_error"),
                    "b": b.get("nodes", {}).get("summary", {}).get(name, {}).get("first_error"),
                } for name in ("GPU_SIMT_INT64_REFERENCE", "GPU_DP2A_W_SPLIT2")
            },
            "performance_pass": {"a": a.get("benchmark", {}).get("performance_pass"),
                                 "b": b.get("benchmark", {}).get("performance_pass")},
        }
        exact_both = (verdict["nodes_required_exact"]["a"] and verdict["nodes_required_exact"]["b"]
                      and verdict["vectors_required_exact"]["a"]
                      and verdict["vectors_required_exact"]["b"])
        distinct_models = bool(verdict["gpu_a"] and verdict["gpu_b"]
                               and verdict["gpu_a"] != verdict["gpu_b"])
        verdict["cross_gpu"] = (
            "both GPUs bit-exact vs the CPU oracle on all 83 nodes and the vectors; "
            "cross-GPU equality follows from bit-exactness" if exact_both
            else "NOT ESTABLISHED")
        if not (exact_both and distinct_models):
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
    report = {"backend_used": "GPU_TC_KARATSUBA3 (primary) + GPU_SCHOOLBOOK4 (independent "
                              "merge, on-hardware reference)",
              "environment": env_capture(device),
              "cuda_probe": cuda_probe(device),
              "float_audit": source_float_audit()}
    report["vectors"] = run_vectors(device)
    report["nodes"] = run_nodes(device)
    exact_ok = bool(report["vectors"]["required_cuda_backends_exact"]
                    and report["nodes"]["required_cuda_backends_exact"])
    report["benchmark"] = run_benchmark(device, exact_ok)
    report["full_block_gpu"] = "NOT TESTED (gated on GEMM performance PASS, F.5B section 62)"
    out = Path(args.out) if args.out else REPO / "docs" / "phase-f5b-gpu-results.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("vectors required exact:", report["vectors"]["required_cuda_backends_exact"])
    print("nodes required exact:", report["nodes"]["required_cuda_backends_exact"])
    print("nodes summary:", json.dumps(report["nodes"]["summary"]))
    print("schedule ratio:", report["benchmark"]["weighted_schedule"]["ratio"])
    print("performance_pass:", report["benchmark"]["performance_pass"])
    print("written:", out)


if __name__ == "__main__":
    main()
