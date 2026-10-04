"""F.5B.1 — fused wide-GEMM optimization gate runner (research-only).

Run on one GPU pod at a time (cost discipline: GPU1 first; GPU2 only if
GPU1 passes):

    python gpu/f5b1/f5b1_runner.py --device cuda --out docs/phase-f5b1-gpu-results.<gpu>.json

Then compare the two per-GPU files on the CPU machine:

    python gpu/f5b1/f5b1_runner.py --compare <fileA> <fileB>

Gates (FROZEN, unchanged from F.5B): weighted 83-node schedule ratio vs the
fastest valid native INT8 baseline <= 6.0x; every major shape (>= 5% logical
GEMM MAC) <= 10.0x.  Correctness: 83/83 nodes + all wide vectors
bit-identical to the CPU int64 oracle, per candidate.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "gpu" / "f5b"))

import f5b1_ext  # noqa: E402
from f5b1_stages import (  # noqa: E402
    CudaGraphKaratsuba3, CustomNativeInt8M16, F5BEagerKaratsuba3,
    NativePaddedInt8, StageB5Launch, StageCFusedMMA, w_to_nk,
)
from f5b_gpu_runner import (  # noqa: E402  frozen gates + shared diagnostics
    MAJOR_SHAPE_FRACTION, MAJOR_SHAPE_MAX_RATIO, PERF_GATE, cuda_probe, env_capture,
)

LADDER_ORDER = ("STAGE_A_CUDA_GRAPH_K3", "STAGE_B_5LAUNCH_K3", "STAGE_C_TRUE_FUSED_MMA")
WIDE_CANDIDATES = ("F5B_EAGER_KARATSUBA3",) + LADDER_ORDER
NATIVE_CANDIDATES = ("NATIVE_PADDED_INT8", "CUSTOM_NATIVE_INT8_M16")


def timed_stats(fn, warmup: int = 50, iters: int = 200) -> dict:
    """CUDA-event timing with the spec's statistics; bumps to 500 iterations
    once if the measured variance is high."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    def run_iters(n: int) -> list[float]:
        times = []
        for _ in range(n):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        return times

    times = run_iters(iters)
    med = statistics.median(times)
    sd = statistics.pstdev(times)
    used = iters
    if med > 0 and sd / med > 0.05 and iters < 500:
        times = run_iters(500)
        med = statistics.median(times)
        sd = statistics.pstdev(times)
        used = 500
    times.sort()
    return {"median_ms": med, "p50_ms": statistics.median(times),
            "p95_ms": times[int(len(times) * 0.95)], "min_ms": times[0],
            "max_ms": times[-1], "stddev_ms": sd, "iterations": used, "warmup": warmup}


def timed_batched(fn, n_runs: int = 100, warmup: int = 20) -> float:
    """One timed region of n_runs consecutive logical GEMMs, divided out."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_runs):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_runs


def count_gpu_ops(fn, n_runs: int = 5) -> dict:
    """Measured GPU operations per logical GEMM via torch profiler."""
    try:
        from torch.profiler import ProfilerActivity, profile

        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n_runs):
                fn()
            torch.cuda.synchronize()
        total = 0
        kernels = 0
        for evt in prof.events():
            if evt.device_type == torch.autograd.DeviceType.CUDA:
                total += 1
                name = evt.name
                if not (name.startswith("Memcpy") or name.startswith("Memset")
                        or name.startswith("memcpy") or name.startswith("memset")):
                    kernels += 1
        return {"gpu_ops_per_logical_gemm": total / n_runs,
                "kernels_per_logical_gemm": kernels / n_runs, "runs_profiled": n_runs}
    except Exception as exc:  # noqa: BLE001  profiler availability is environment
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def load_nodes():
    data = np.load(REPO / "testdata" / "f5b_gemm_nodes.npz")
    meta = json.loads((REPO / "testdata" / "f5b_gemm_nodes_meta.json")
                      .read_text(encoding="utf-8"))["nodes"]
    return data, meta


def load_vectors():
    return json.loads((REPO / "testdata" / "f5b_wide_gemm_vectors.json")
                      .read_text(encoding="utf-8"))["vectors"]


def build_candidates_for_node(a_np, w_np, transpose_b):
    """All wide candidates for one node (weights prepacked per node)."""
    a16 = torch.tensor(a_np, dtype=torch.int16).cuda()
    w_nk = w_to_nk(torch.tensor(w_np, dtype=torch.long), transpose_b)
    cands = {
        "F5B_EAGER_KARATSUBA3": F5BEagerKaratsuba3(w_nk),
        "STAGE_A_CUDA_GRAPH_K3": CudaGraphKaratsuba3(a16, w_nk),
        "STAGE_B_5LAUNCH_K3": StageB5Launch(w_nk),
        "STAGE_C_TRUE_FUSED_MMA": StageCFusedMMA(w_nk),
    }
    return a16, cands


def run_correctness(device: torch.device) -> dict:
    data, meta = load_nodes()
    vectors = load_vectors()
    node_stat = {name: {"exact": 0, "mismatch": 0, "error": 0, "first_error": None}
                 for name in WIDE_CANDIDATES}
    vec_stat = {name: {"exact": 0, "mismatch": 0, "error": 0, "first_error": None}
                for name in WIDE_CANDIDATES}
    stage_a_vs_eager = {"match": 0, "differ": 0, "error": 0, "not_captured": 0}
    vec_detail: list[dict] = []
    for m in meta:
        nid = m["node_id"]
        expected = torch.tensor(data[f"c_{nid}"], dtype=torch.long).cuda()
        a_np, w_np = data[f"a_{nid}"], data[f"w_{nid}"]
        N, K = int(expected.shape[1]), int(a_np.shape[1])
        # orientation self-check: trans=True -> w is (N,K); trans=False -> (K,N)
        w_shape = tuple(int(x) for x in w_np.shape)
        if (m["transpose_b"] and w_shape != (N, K)) or (not m["transpose_b"] and w_shape != (K, N)):
            raise RuntimeError(f"node {nid} weight orientation inconsistent: {w_shape} "
                               f"N={N} K={K} trans={m['transpose_b']}")
        a16, cands = build_candidates_for_node(a_np, w_np, m["transpose_b"])
        for name, cand in cands.items():
            try:
                got = cand.run(a16)
                mm = int((got != expected).sum().item())
                node_stat[name]["exact" if mm == 0 else "mismatch"] += 1
            except Exception as exc:  # noqa: BLE001  explicit
                msg = f"{type(exc).__name__}: {str(exc)[:200]}"
                node_stat[name]["error"] += 1
                node_stat[name]["first_error"] = node_stat[name]["first_error"] or msg
        # Stage A vs eager: direct equality check (both also vs the oracle)
        sa = cands["STAGE_A_CUDA_GRAPH_K3"]
        if sa.g is None:
            stage_a_vs_eager["not_captured"] += 1
        else:
            try:
                eq = bool(torch.equal(cands["F5B_EAGER_KARATSUBA3"].run(a16), sa.run(a16)))
                stage_a_vs_eager["match" if eq else "differ"] += 1
            except Exception:  # noqa: BLE001
                stage_a_vs_eager["error"] += 1
    for v in vectors:
        a16 = torch.tensor(v["a"], dtype=torch.long).reshape(v["a_shape"])
        expected = torch.tensor(v["cpu_direct"], dtype=torch.long).reshape(
            v["cpu_direct_shape"]).cuda()
        w_nk = w_to_nk(torch.tensor(v["w"], dtype=torch.long).reshape(v["w_shape"]),
                       v["transpose_b"])
        cands = {
            "F5B_EAGER_KARATSUBA3": F5BEagerKaratsuba3(w_nk),
            "STAGE_A_CUDA_GRAPH_K3": CudaGraphKaratsuba3(a16.to(torch.int16).cuda(), w_nk),
            "STAGE_B_5LAUNCH_K3": StageB5Launch(w_nk),
            "STAGE_C_TRUE_FUSED_MMA": StageCFusedMMA(w_nk),
        }
        detail = {"name": v["name"], "backends": {}}
        for name, cand in cands.items():
            try:
                got = cand.run(a16.to(torch.int16).cuda())
                mm = int((got != expected).sum().item())
                vec_stat[name]["exact" if mm == 0 else "mismatch"] += 1
                detail["backends"][name] = {"exact": mm == 0, "mismatches": mm}
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {str(exc)[:200]}"
                vec_stat[name]["error"] += 1
                vec_stat[name]["first_error"] = vec_stat[name]["first_error"] or msg
                detail["backends"][name] = {"exact": False, "error": msg}
        vec_detail.append(detail)
    return {"nodes": node_stat, "vectors": vec_stat, "vectors_detail": vec_detail,
            "stage_a_vs_eager_nodes": stage_a_vs_eager,
            "nodes_total": len(meta), "vectors_total": len(vectors)}


def run_launch_profile() -> dict:
    """Measured kernels per logical GEMM (torch profiler), for two shapes."""
    profile_shapes = [(16, 128, 1024), (16, 3072, 1024)]
    out = {}
    for (m, n, k) in profile_shapes:
        g = torch.Generator(device="cpu").manual_seed(m * 1000003 + n * 10007 + k)
        a13 = torch.randint(-4096, 4096, (m, k), generator=g).to(torch.int64)
        w10 = torch.randint(-512, 512, (n, k), generator=g).to(torch.int64).cuda()
        a8 = torch.randint(-128, 128, (m, k), generator=g).to(torch.int8).cuda()
        w8 = torch.randint(-128, 128, (n, k), generator=g).to(torch.int8).cuda()
        a16 = a13.to(torch.int16).cuda()
        entry = {}
        natives = {
            "NATIVE_PADDED_INT8": NativePaddedInt8(w8),
            "CUSTOM_NATIVE_INT8_M16": CustomNativeInt8M16(w8),
        }
        for name, cand in natives.items():
            entry[name] = count_gpu_ops(lambda c=cand: c.run(a8))
        wides = {
            "F5B_EAGER_KARATSUBA3": F5BEagerKaratsuba3(w10),
            "STAGE_A_CUDA_GRAPH_K3": CudaGraphKaratsuba3(a16, w10),
            "STAGE_B_5LAUNCH_K3": StageB5Launch(w10),
            "STAGE_C_TRUE_FUSED_MMA": StageCFusedMMA(w10),
        }
        for name, cand in wides.items():
            entry[name] = count_gpu_ops(lambda c=cand: c.run(a16))
        out[f"{m}x{n}x{k}"] = entry
    return out


def run_benchmark(device: torch.device, correctness: dict) -> dict:
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
        a13_cpu = torch.randint(-4096, 4096, (m, k), generator=g).to(torch.int64)
        w10_cpu = torch.randint(-512, 512, (n, k), generator=g).to(torch.int64)
        a8 = torch.randint(-128, 128, (m, k), generator=g).to(torch.int8).cuda()
        w8 = torch.randint(-128, 128, (n, k), generator=g).to(torch.int8).cuda()
        a16 = a13_cpu.to(torch.int16).cuda()
        w_nk = w10_cpu.cuda()
        expected = a13_cpu @ w10_cpu.T

        cands = {
            "NATIVE_PADDED_INT8": (NativePaddedInt8(w8), a8),
            "CUSTOM_NATIVE_INT8_M16": (CustomNativeInt8M16(w8), a8),
            "F5B_EAGER_KARATSUBA3": (F5BEagerKaratsuba3(w_nk), a16),
            "STAGE_A_CUDA_GRAPH_K3": (CudaGraphKaratsuba3(a16, w_nk), a16),
            "STAGE_B_5LAUNCH_K3": (StageB5Launch(w_nk), a16),
            "STAGE_C_TRUE_FUSED_MMA": (StageCFusedMMA(w_nk), a16),
        }
        entry = {"M": m, "N": n, "K": k, "transpose_b": trans,
                 "schedule_frequency": count, "logical_mac": m * n * k,
                 "mac_fraction": (m * n * k * count) / total_mac}
        for name, (cand, inp) in cands.items():
            try:
                cand.run(inp)  # warm build paths (graph capture etc.)
                stats = timed_stats(lambda c=cand, i=inp: c.run(i))
                stats["batched_median_ms"] = timed_batched(lambda c=cand, i=inp: c.run(i))
                # batched-work verification (no silently eliminated work)
                out_b = cand.run(inp)
                if name in WIDE_CANDIDATES:
                    stats["batched_output_exact"] = bool(torch.equal(out_b.cpu(), expected))
                else:
                    ref8 = a8.to(torch.int64).cpu() @ w8.to(torch.int64).cpu().T
                    stats["batched_output_exact"] = bool(
                        torch.equal(out_b.cpu().to(torch.int64), ref8))
                entry[name] = stats
            except Exception as exc:  # noqa: BLE001
                entry[name] = {"error": f"{type(exc).__name__}: {str(exc)[:250]}"}
        # fastest valid native baseline = per-shape min of the two natives
        nat = [entry[c]["median_ms"] for c in NATIVE_CANDIDATES
               if "median_ms" in entry[c]]
        entry["native_fastest_median_ms"] = min(nat) if nat else None
        results[f"{m}x{n}x{k}{'T' if trans else ''}"] = entry

    weighted = {}
    for cand in WIDE_CANDIDATES:
        rows = [e for e in results.values()
                if "median_ms" in e.get(cand, {}) and e["native_fastest_median_ms"]]
        if len(rows) != len(results) or not rows:
            weighted[cand] = {"error": "not all shapes measured"}
            continue
        wide_total = sum(e[cand]["median_ms"] * e["schedule_frequency"] for e in rows)
        native_total = sum(e["native_fastest_median_ms"] * e["schedule_frequency"] for e in rows)
        major = [{"shape": name, "ratio": e[cand]["median_ms"] / e["native_fastest_median_ms"]}
                 for name, e in results.items()
                 if e["mac_fraction"] >= MAJOR_SHAPE_FRACTION and "median_ms" in e.get(cand, {})]
        weighted[cand] = {
            "wide_total_ms": wide_total, "native_fastest_total_ms": native_total,
            "ratio": wide_total / native_total if native_total > 0 else None,
            "major_shape_ratios": major,
            "major_ok": bool(major) and all(x["ratio"] <= MAJOR_SHAPE_MAX_RATIO for x in major),
        }

    ladder = {"order": list(LADDER_ORDER), "results": {}, "winner": None}
    for stage in LADDER_ORDER:
        w = weighted.get(stage, {})
        node_ok = (correctness["nodes"][stage]["exact"] == correctness["nodes_total"]
                   and correctness["nodes"][stage]["error"] == 0
                   and correctness["nodes"][stage]["mismatch"] == 0)
        vec_ok = (correctness["vectors"][stage]["exact"] == correctness["vectors_total"]
                  and correctness["vectors"][stage]["error"] == 0
                  and correctness["vectors"][stage]["mismatch"] == 0)
        ratio_ok = bool(w.get("ratio") is not None and w["ratio"] <= PERF_GATE)
        passed = bool(node_ok and vec_ok and ratio_ok and w.get("major_ok"))
        ladder["results"][stage] = {
            "exact": bool(node_ok and vec_ok), "weighted_ratio": w.get("ratio"),
            "major_ok": bool(w.get("major_ok")), "ratio_ok": ratio_ok, "pass": passed}
        if passed and ladder["winner"] is None:
            ladder["winner"] = stage
    return {"shapes": results, "weighted_schedule": weighted, "ladder": ladder,
            "gates": {"schedule_ratio_max": PERF_GATE, "major_shape_ratio_max": MAJOR_SHAPE_MAX_RATIO,
                      "major_shape_fraction": MAJOR_SHAPE_FRACTION,
                      "native_denominator_rule": "fastest valid native INT8 per shape "
                                                 "(min of the two baselines), spec section 11"},
            "physical_mac_note": "the wide path performs 3 physical int8 MMA streams per logical "
                                 "MAC; protocol work accounting remains 1x logical (F.5B section 46)"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    parser.add_argument("--compare", nargs=2, default=None)
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
        return

    device = torch.device(args.device)
    report: dict = {
        "phase": "F.5B.1", "environment": env_capture(device),
        "cuda_probe": cuda_probe(device),
        "full_block_gpu": "NOT STARTED (gated on GEMM performance PASS, spec section 81)",
    }
    if device.type != "cuda":
        report["extension"] = {"built": False, "note": "CPU dry-run: kernel extension and "
                                                       "GPU candidates are NOT TESTED"}
        report["correctness"] = {"note": "NOT TESTED on CPU"}
        report["launch_profile"] = {"note": "NOT TESTED on CPU"}
        report["benchmark"] = {"note": "NOT TESTED on CPU"}
        out = Path(args.out) if args.out else REPO / "docs" / "phase-f5b1-gpu-results.cpudry.json"
        out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print("CPU dry-run: structure validation only; written:", out)
        return

    t0 = time.time()
    f5b1_ext.build()
    report["extension"] = {"built": True, "build_seconds": round(time.time() - t0, 1)}
    report["selftest"] = f5b1_ext.selftest()
    if not report["selftest"]["ok"]:
        out = Path(args.out) if args.out else REPO / "docs" / "phase-f5b1-gpu-results.json"
        report["abort"] = "kernel self-test failed — no evidence produced"
        out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print("SELFTEST FAILED — aborting. written:", out)
        sys.exit(2)
    print("selftest OK")
    report["correctness"] = run_correctness(device)
    print("correctness:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "first_error"}
                                      for k, v in report["correctness"]["nodes"].items()}))
    report["launch_profile"] = run_launch_profile()
    report["benchmark"] = run_benchmark(device, report["correctness"])
    for cand, w in report["benchmark"]["weighted_schedule"].items():
        print(f"weighted {cand}: ratio={w.get('ratio')} major_ok={w.get('major_ok')}")
    print("ladder winner:", report["benchmark"]["ladder"]["winner"])
    out = Path(args.out) if args.out else REPO / "docs" / "phase-f5b1-gpu-results.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("written:", out)


def compare(path_a: str, path_b: str) -> None:
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))

    def ladder(doc):
        return doc.get("benchmark", {}).get("ladder", {})

    la, lb = ladder(a), ladder(b)
    verdict = {
        "phase": "F.5B.1",
        "gpu_a": a.get("environment", {}).get("gpu_name"),
        "gpu_b": b.get("environment", {}).get("gpu_name"),
        "cross_arch": (a.get("environment", {}).get("compute_capability")
                       != b.get("environment", {}).get("compute_capability")),
        "winner_a": la.get("winner"), "winner_b": lb.get("winner"),
        "weighted_ratio_a": {k: v.get("ratio") for k, v in
                             a.get("benchmark", {}).get("weighted_schedule", {}).items()},
        "weighted_ratio_b": {k: v.get("ratio") for k, v in
                             b.get("benchmark", {}).get("weighted_schedule", {}).items()},
        "ladder_a": la.get("results"), "ladder_b": lb.get("results"),
        "native_denominator_rule": "per-shape fastest valid native INT8 baseline "
                                   "(old padded torch baseline vs custom M=16 native)",
    }
    pass_a, pass_b = la.get("winner") is not None, lb.get("winner") is not None
    if not (verdict["gpu_a"] and verdict["gpu_b"] and verdict["gpu_a"] != verdict["gpu_b"]):
        verdict["final"] = "INCONCLUSIVE_NO_SECOND_GPU"
    elif pass_a and pass_b:
        verdict["final"] = ("FUSED_GPU_FEASIBLE_A13W10 if the full 310-node block also runs "
                            "bit-exact on both GPUs (docs/phase-f5b1-real-block-gpu.json); "
                            "until then: FUSION_GEMM_PASS_BLOCK_PENDING")
        verdict["gemm_pass_both"] = True
    elif pass_a or pass_b:
        verdict["final"] = ("ARCH_DEPENDENT_PERFORMANCE (one GPU passes the frozen gates, "
                            "the other does not; F.5C must not start)")
        verdict["gemm_pass_both"] = False
    else:
        exact_a = all(r.get("exact") for r in (la.get("results") or {}).values())
        exact_b = all(r.get("exact") for r in (lb.get("results") or {}).values())
        verdict["final"] = ("FUSION_CORRECT_BUT_NOT_PRACTICAL" if (exact_a and exact_b)
                            else "FUSION_NOT_FEASIBLE")
        verdict["gemm_pass_both"] = False
    print(json.dumps(verdict, indent=1))
    out = REPO / "docs" / "phase-f5b1-final-verdict.json"
    out.write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8")
    print("written:", out)


if __name__ == "__main__":
    main()
