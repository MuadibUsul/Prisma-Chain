"""EXPERIMENTAL / NON-PROTOCOL — F.5B radix-128 decomposition proof (CPU).

Proves, with exhaustive and adversarial tests, that the balanced
radix-128 decomposition executes the frozen A13W10 logical GEMM exactly:

    A = A0 + 128*A1   (A0 in [-64,63], A1 in [-32,32])
    W = W0 + 128*W1   (W0 in [-64,63], W1 in [-4,4])

    schoolbook:  C = C00 + 128*C01 + 128*C10 + 16384*C11
    karatsuba3:  Cross = (A0+A1)(W0+W1) - C00 - C11,  C = C00 + 128*Cross + 16384*C11

with every physical GEMM an s8 x s8 -> s32 shape and the final merge in
exact int64. Also emits the real 83-node shape manifest, the theoretical
per-node requant int64 bounds, and the frozen wide vector file.

    python tools/f5b_radix128.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

A_LO, A_HI = -(1 << 12), (1 << 12) - 1      # A13 signed logical range
W_LO, W_HI = -(1 << 9), (1 << 9) - 1        # W10 signed logical range
INT32_MAX = (1 << 31) - 1
INT64_MAX = (1 << 63) - 1


def split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Balanced radix-128 split with explicit mathematical floor division:
    q = floor((x + 64) / 128); lo = x - 128*q; hi = q."""
    q = (x.astype(np.int64) + 64) // 128     # numpy // is floor for negatives
    lo = x.astype(np.int64) - 128 * q
    return lo, q


def direct_gemm(a: np.ndarray, w: np.ndarray, transpose_b: bool = False) -> np.ndarray:
    a64 = a.astype(np.int64)
    w64 = w.astype(np.int64)
    return a64 @ w64.T if transpose_b else a64 @ w64


def schoolbook(a: np.ndarray, w: np.ndarray, transpose_b: bool = False) -> tuple[np.ndarray, dict]:
    a0, a1 = split(a)
    w0, w1 = split(w)
    mm = (lambda x, y: x @ y.T) if transpose_b else (lambda x, y: x @ y)
    c00 = mm(a0, w0)
    c01 = mm(a0, w1)
    c10 = mm(a1, w0)
    c11 = mm(a1, w1)
    c = c00 + 128 * c01 + 128 * c10 + 16384 * c11
    return c, {"c00": c00, "c01": c01, "c10": c10, "c11": c11}


def karatsuba3(a: np.ndarray, w: np.ndarray, transpose_b: bool = False) -> tuple[np.ndarray, dict]:
    a0, a1 = split(a)
    w0, w1 = split(w)
    mm = (lambda x, y: x @ y.T) if transpose_b else (lambda x, y: x @ y)
    c00 = mm(a0, w0)
    c11 = mm(a1, w1)
    csum = mm(a0 + a1, w0 + w1)
    cross = csum - c00 - c11
    c = c00 + 128 * cross + 16384 * c11
    return c, {"c00": c00, "csum": csum, "c11": c11, "cross": cross}


def exhaustive_split_proof() -> dict:
    a_vals = np.arange(A_LO, A_HI + 1, dtype=np.int64)
    w_vals = np.arange(W_LO, W_HI + 1, dtype=np.int64)
    a0, a1 = split(a_vals)
    w0, w1 = split(w_vals)
    ok = bool(
        np.array_equal(a0 + 128 * a1, a_vals)
        and np.array_equal(w0 + 128 * w1, w_vals)
        and a0.min() >= -64 and a0.max() <= 63
        and a1.min() >= -32 and a1.max() <= 32
        and w0.min() >= -64 and w0.max() <= 63
        and w1.min() >= -4 and w1.max() <= 4
    )
    return {
        "a_values_tested": int(a_vals.size), "w_values_tested": int(w_vals.size),
        "a0_range": [int(a0.min()), int(a0.max())], "a1_range": [int(a1.min()), int(a1.max())],
        "w0_range": [int(w0.min()), int(w0.max())], "w1_range": [int(w1.min()), int(w1.max())],
        "a0_plus_a1_range": [int((a0 + a1).min()), int((a0 + a1).max())],
        "w0_plus_w1_range": [int((w0 + w1).min()), int((w0 + w1).max())],
        "reconstruct_exact": ok,
    }


def bounds_for_k(k: int) -> dict:
    c00 = 64 * 64 * k
    c11 = 32 * 4 * k
    csum = 96 * 68 * k
    cross = c00 + csum + c11
    final = ((1 << 12) * (1 << 9)) * k
    return {"C00": c00, "C11": c11, "Csum": csum,
            "Cross_int32_subtraction_bound": cross,
            "final_int64_bound": final,
            "all_int32_safe": max(c00, c11, csum, cross) <= INT32_MAX,
            "final_int64_safe": final <= INT64_MAX}


def adversarial_patterns(m: int, n: int, k: int) -> dict:
    pats = {}
    pats["all_max"] = (np.full((m, k), A_HI, np.int64), np.full((k, n), W_HI, np.int64))
    pats["all_min"] = (np.full((m, k), A_LO, np.int64), np.full((k, n), W_LO, np.int64))
    pats["a_max_w_min"] = (np.full((m, k), A_HI, np.int64), np.full((k, n), W_LO, np.int64))
    pats["a_min_w_max"] = (np.full((m, k), A_LO, np.int64), np.full((k, n), W_HI, np.int64))
    alt = np.empty((m, k), np.int64)
    alt[:] = A_HI
    alt[:, 1::2] = A_LO
    pats["alternating"] = (alt, np.full((k, n), W_HI, np.int64))
    pats["zeros"] = (np.zeros((m, k), np.int64), np.zeros((k, n), np.int64))
    one_rows = np.zeros((m, k), np.int64)
    one_rows[0, :] = A_HI
    pats["one_hot_rows"] = (one_rows, np.full((k, n), W_HI, np.int64))
    rng = np.random.default_rng(7)
    pats["random"] = (rng.integers(A_LO, A_HI + 1, (m, k)),
                      rng.integers(W_LO, W_HI + 1, (k, n)))
    return pats


def verify_shapes(shapes: list[tuple[int, int, int]], patterns_per_shape: bool = True) -> dict:
    out = {}
    for (m, n, k) in shapes:
        entry = {"m": m, "n": n, "k": k, "patterns": {}}
        pats = adversarial_patterns(m, n, k)
        for name, (a, w) in pats.items():
            if not patterns_per_shape and name != "random":
                continue
            direct = direct_gemm(a, w)
            sb, sb_parts = schoolbook(a, w)
            kz, kz_parts = karatsuba3(a, w)
            sb_ok = bool(np.array_equal(sb, direct))
            kz_ok = bool(np.array_equal(kz, direct))
            entry["patterns"][name] = {
                "schoolbook_exact": sb_ok, "karatsuba3_exact": kz_ok,
                "direct_max_abs": int(np.max(np.abs(direct))) if direct.size else 0,
                "c00_max_abs": int(np.max(np.abs(sb_parts["c00"]))) if direct.size else 0,
                "csum_max_abs": int(np.max(np.abs(kz_parts["csum"]))) if direct.size else 0,
            }
            assert sb_ok and kz_ok, f"{name} {m}x{n}x{k} decomposition mismatch"
        entry["bounds"] = bounds_for_k(k)
        out[f"{m}x{n}x{k}"] = entry
    return out


def real_graph_shapes() -> tuple[list[dict], list[str]]:
    from convert_qwen3_block import Conformer, build_converted_graph, to_fx
    import f5a_joint_precision as F5A
    conf = Conformer(0)
    cases, samples, head_samples = F5A.collect_calibration(conf)
    plan = F5A.build_plan(conf, samples, head_samples, 13, 10)
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
    snake, _go, _plan, tdata, _gid = build_converted_graph(conf, 16, {"chosen": chosen, "p": 1.0})
    rows = []
    for node in snake["nodes"]:
        if node["operator_id"] != "GEMM_INT8_V1":
            continue
        a_ref, b_ref = node["inputs"]
        a_desc = (tdata[a_ref["index"]].shape if a_ref["kind"] == 0
                  else snake["nodes"][a_ref["index"]]["output"]["shape"])
        b_desc = (tdata[b_ref["index"]].shape if b_ref["kind"] == 0
                  else snake["nodes"][b_ref["index"]]["output"]["shape"])
        params = dict((k, v) for k, v in node["params"])
        trans = params.get("transpose_b", 0) == 1
        m, k = int(a_desc[0]), int(a_desc[1])
        n = int(b_desc[0]) if trans else int(b_desc[1])
        rows.append({"node_id": node["node_id"], "M": m, "N": n, "K": k,
                     "transpose_b": bool(trans), "logical_mac": m * n * k})
    dedup = sorted({f"{r['M']}x{r['N']}x{r['K']}{'T' if r['transpose_b'] else ''}" for r in rows})
    return rows, dedup


def theoretical_requant_bounds() -> dict:
    """Per-GEMM-link worst-case requant int64 check (section 39)."""
    pol = json.loads((REPO / "docs" / "phase-f5a-candidate-policies.json")
                     .read_text(encoding="utf-8"))["candidates"]["A13W10"]
    sites, weights = pol["sites"], pol["weights"]
    rows, _dedup = real_graph_shapes()
    links = []
    worst_acc_any = 0
    worst_bits = 0
    # h -> qkv (accum_fx), h2 -> gate/up, hm -> down, scores, v, ctx accum
    mult_defs = [
        ("qkv_accum_fx", sites["h"]["step_fx"] * max(weights["wq"] + weights["wk"] + weights["wv"])),
        ("scores", max(1, round(max(sites["q_heads"]) * max(sites["k_heads"]) * 92681 / (1 << 20)))),
        ("v_accum_i8", max(1, round(sites["h"]["step_fx"] * max(weights["wv"]) / sites["v"]["step_fx"]))),
        ("ctx_accum_i8", max(1, round(sites["p"]["step_fx"] * sites["v"]["step_fx"] / sites["ctx"]["step_fx"]))),
        ("oproj_accum_fx", sites["ctx"]["step_fx"] * max(weights["wo"])),
        ("gateup_accum_fx", sites["h2"]["step_fx"] * max([weights["wg"], weights["wu"]])),
        ("down_accum_fx", sites["hm"]["step_fx"] * weights["wd"]),
    ]
    k_by_family = {
        "qkv_accum_fx": 1024, "scores": 128, "v_accum_i8": 1024,
        "ctx_accum_i8": 16, "oproj_accum_fx": 128,
        "gateup_accum_fx": 1024, "down_accum_fx": 3072,
    }
    all_safe = True
    for name, mult in mult_defs:
        acc_bound = k_by_family[name] * (1 << 12) * (1 << 9)
        product = acc_bound * int(mult)
        bits = product.bit_length()
        safe = product <= INT64_MAX
        all_safe &= safe
        worst_acc_any = max(worst_acc_any, acc_bound)
        worst_bits = max(worst_bits, bits)
        links.append({"link": name, "k": k_by_family[name], "worst_acc": acc_bound,
                      "multiplier": int(mult), "worst_product": product,
                      "product_bits": bits, "int64_safe": bool(safe)})
    return {"links": links, "all_theoretical_requant_int64_safe": all_safe,
            "worst_acc": worst_acc_any, "worst_product_bits": worst_bits,
            "verdict": "THEORETICAL_REQUANT_INT64_SAFE" if all_safe else "REQUIRES_WIDE_REQUANT",
            "note": "worst_acc = K * 2^12 * 2^9 (full signed magnitudes), never observed data"}


def wide_vectors() -> dict:
    rng = np.random.default_rng(11)
    vectors = []
    def add(name, a, w, trans=False):
        direct = direct_gemm(a, w, trans)
        kz, _ = karatsuba3(a, w, trans)
        sb, _ = schoolbook(a, w, trans)
        assert np.array_equal(kz, direct) and np.array_equal(sb, direct), name
        vectors.append({
            "name": name,
            "a": a.reshape(-1).tolist(), "a_shape": list(a.shape),
            "w": w.reshape(-1).tolist(), "w_shape": list(w.shape),
            "transpose_b": trans,
            "cpu_direct": direct.reshape(-1).tolist(),
            "cpu_direct_shape": list(direct.shape),
            "cpu_schoolbook_exact": True,
            "cpu_radix128_karatsuba3_exact": True,
            "gpu": "NOT TESTED",
        })
    add("boundary_all_max", np.full((2, 4), A_HI, np.int64), np.full((4, 3), W_HI, np.int64))
    add("boundary_all_min", np.full((2, 4), A_LO, np.int64), np.full((4, 3), W_LO, np.int64))
    add("small_random", rng.integers(A_LO, A_HI + 1, (3, 8)), rng.integers(W_LO, W_HI + 1, (8, 5)))
    add("transpose_b", rng.integers(A_LO, A_HI + 1, (3, 8)), rng.integers(W_LO, W_HI + 1, (3, 8)), True)
    add("k1", np.array([[A_HI, A_LO]], np.int64), np.array([[W_LO], [W_HI]], np.int64))
    return {"protocol_version": "F5B_WIDE_VECTORS_V1", "a_bits": 13, "w_bits": 10,
            "note": "CPU direct + radix-128 Karatsuba3 must match elementwise; GPU column is NOT TESTED",
            "vectors": vectors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--shapes-only", action="store_true")
    args = parser.parse_args()

    if args.shapes_only:
        rows, dedup = real_graph_shapes()
        (REPO / "docs" / "phase-f5b-gemm-shapes.json").write_text(
            json.dumps({"gemm_nodes": len(rows), "nodes": rows,
                        "deduplicated_shapes": dedup}, indent=1) + "\n", encoding="utf-8")
        print("shapes written:", len(rows), "dedup:", dedup)
        return

    proof = {"split_proof": exhaustive_split_proof()}
    assert proof["split_proof"]["reconstruct_exact"], "exhaustive split proof failed"
    print("exhaustive split proof OK:",
          proof["split_proof"]["a_values_tested"], "+", proof["split_proof"]["w_values_tested"])

    # adversarial + real shapes. K = 3072 uses a reduced (M,N) to keep the
    # CPU proof fast; the decomposition is elementwise so this is exact.
    shapes = [(2, 3, 1), (4, 4, 8), (4, 4, 16), (8, 8, 128), (16, 128, 128),
              (16, 128, 1024), (16, 1024, 128), (2, 2, 1024), (16, 1024, 3072), (2, 2, 3072)]
    proof["shape_tests"] = verify_shapes(shapes)
    print("shape tests OK:", len(shapes), "shapes x adversarial patterns")

    proof["bounds_table"] = {f"K{k}": bounds_for_k(k) for k in (1, 8, 16, 128, 1024, 3072)}
    assert all(v["all_int32_safe"] and v["final_int64_safe"] for v in proof["bounds_table"].values())
    proof["requant"] = theoretical_requant_bounds()
    rows, dedup = real_graph_shapes()
    proof["real_graph"] = {"gemm_nodes": len(rows), "k_max": max(r["K"] for r in rows),
                           "deduplicated_shapes": dedup,
                           "all_nodes_within_bounds": bool(
                               max(r["K"] for r in rows) <= 3072)}
    (REPO / "docs" / "phase-f5b-radix128-proof.json").write_text(
        json.dumps(proof, indent=1) + "\n", encoding="utf-8")
    (REPO / "docs" / "phase-f5b-gemm-shapes.json").write_text(
        json.dumps({"gemm_nodes": len(rows), "nodes": rows,
                    "deduplicated_shapes": dedup}, indent=1) + "\n", encoding="utf-8")
    wv = wide_vectors()
    (REPO / "testdata" / "f5b_wide_gemm_vectors.json").write_text(
        json.dumps(wv, indent=1) + "\n", encoding="utf-8")
    print("proof written; requant verdict:", proof["requant"]["verdict"],
          "| worst product bits:", proof["requant"]["worst_product_bits"])
    print(f"wide vectors written: {len(wv['vectors'])} vectors")


if __name__ == "__main__":
    main()
