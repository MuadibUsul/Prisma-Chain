"""EXPERIMENTAL / NON-PROTOCOL — F.3A activation bit-depth study.

Answers one question: with W8 weights and corrected-F.1 static scale
granularity (per-head q/k steps; per-tensor elsewhere; NO groupwise),
what is the MINIMUM signed activation bit depth b in {8..13} such that
the pinned Qwen3-0.6B layer 0 passes BOTH predeclared gates
(worst cosine >= 0.995 AND worst max_abs <= 0.05), while every GEMM keeps
worst-case signed int32 accumulator safety.

The executor is the single-scale integer pipeline (SLICE-less), with the
quantizer clamp set to Qmax(b) = 2^(b-1)-1 and per-b recalibration by
clipping-aware quantization MSE on the F3A calibration split. All
quantizers are integer mult/shift with ties-to-even; int32 accumulator
range is ASSERTED per GEMM (never wrapped).

Anchors: A8 must reproduce the converted graph (with the corrected
scores multiplier) bit-for-bit; otherwise the run stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import f1_canonical_numpy as N  # noqa: E402
from convert_qwen3_block import Conformer, instrumented_forward, to_fx  # noqa: E402

# F.3A calibration decision (calibration-set criterion only): the F.1
# link-level k factor (0.25) was tuned under the buggy scores scale and is
# pure clipping bias once the scales are correct; the MSE-optimal step
# (factor 1.0) is the principled calibration choice for every candidate.
K_HEAD_FACTOR = 1.0
from f2a_groupwise_search import wscale  # noqa: E402

ONE = 1 << 20
MIN_FX, MAX_FX = -(1 << 31), (1 << 31) - 1
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1


def gemm64(a, b, transpose_b=False):
    """Exact signed GEMM with an int64 accumulator (research semantics).

    The F.5A study widens ONLY the accumulator; operands stay at their
    logical bit depths (stored in int16 research containers). The int64
    range is ASSERTED, never wrapped.
    """
    a64 = np.asarray(a).astype(np.int64)
    b64 = np.asarray(b).astype(np.int64)
    acc = a64 @ b64.T if transpose_b else a64 @ b64
    lo, hi = int(acc.min()), int(acc.max())
    assert INT64_MIN <= lo and hi <= INT64_MAX, "int64 accumulator overflow"
    return acc
ATTN_SCALE_FX = 92681
BITS = [8, 9, 10, 11, 12, 13]
K_MAX_BLOCK = 3072


def qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def max_safe_k(bits: int) -> int:
    return (2**31 - 1) // ((1 << (bits - 1)) * 128)


def max_safe_k_joint(a_bits: int, w_bits: int) -> int:
    """Worst-case signed int32 bound: (2^(a-1) * 2^(w-1)) * K <= 2^31-1."""
    return (2**31 - 1) // ((1 << (a_bits - 1)) * (1 << (w_bits - 1)))


def mse_step_bits(vals: np.ndarray, bits: int) -> tuple[int, float, float]:
    """Clipping-aware MSE-optimal static step for signed b-bit levels."""
    qm = qmax(bits)
    absvals = np.abs(np.asarray(vals, dtype=np.float64).reshape(-1))
    best = None
    for pct in (90.0, 95.0, 97.5, 99.0, 99.5, 99.8, 99.9, 99.95, 99.99, 100.0):
        edge = float(np.percentile(absvals, pct)) * 1.02
        step_fx = max(1, int(np.ceil(edge * ONE / qm)))
        step_real = step_fx / ONE
        counts = np.clip(np.rint(absvals / step_real), 0, qm)
        err = counts * step_real - absvals
        mse = float(np.mean(err * err))
        if best is None or mse < best[2]:
            best = (step_fx, pct, mse)
    return best


def quant_bits(v_fx: np.ndarray, step: int, bits: int, lo: int | None = None) -> np.ndarray:
    qm = qmax(bits)
    if lo is None:
        lo = -qm
    mult = max(1, int(round((1 << 30) / step)))
    return np.clip(N.rshift_round_even(v_fx.astype(np.int64) * np.int64(mult), 30), lo, qm)


def collect_calibration(conf: Conformer):
    cases = json.loads((REPO / "testdata" / "qwen3_f3a_calibration.json")
                       .read_text(encoding="utf-8"))["cases"]
    samples = {k: [] for k in ("h", "q", "k", "v", "ctx", "h2", "hm")}
    head_samples = {k: [] for k in ("q", "k")}
    for case in cases:
        hidden = conf.embedding(case["token_ids"])
        _, _, raw = instrumented_forward(conf, hidden, collect=True)
        for key in samples:
            flat = np.abs(raw[key]).reshape(-1)
            flat = flat[:: max(1, flat.size // 100_000)]
            samples[key].append(flat)
        for key in head_samples:
            arr = raw[key]
            for h in range(arr.shape[1]):
                head_samples[key].append(np.abs(arr[:, h, :]).reshape(-1))
    return cases, samples, head_samples


def build_plan(conf: Conformer, samples, head_samples, bits: int, w_bits: int = 8) -> dict:
    """One candidate's frozen scales for a given (activation, weight) bit pair.

    Activation policy depends ONLY on the activation bits (shared across all
    weight widths for the same a). Weight steps follow the frozen
    slice-maximum rule with QmaxW = 2^(w-1)-1 (weight = 8 reproduces the
    corrected F.1/F.3A weights exactly).
    """
    plan = {"bits": bits, "w_bits": w_bits, "qmax": qmax(bits),
            "qmax_w": qmax(w_bits), "sites": {},
            "theoretical_max_safe_k": max_safe_k_joint(bits, w_bits)}
    for key in ("h", "v", "ctx", "h2", "hm", "q"):
        step, pct, mse = mse_step_bits(np.concatenate(samples[key]), bits)
        plan["sites"][key] = {"step_fx": int(step), "percentile": pct, "mse": mse}
    # global k fallback (the graph builder reads it eagerly even when the
    # per-head keys are present)
    k_global, k_pct, k_mse = mse_step_bits(np.concatenate(samples["k"]), bits)
    plan["sites"]["k"] = {"step_fx": int(k_global), "percentile": k_pct, "mse": k_mse}
    # k: per-head steps with the link-level factor (calibration only)
    k_steps = []
    for h in range(8):
        step, pct, mse = mse_step_bits(head_samples["k"][h], bits)
        k_steps.append(max(1, int(round(step * K_HEAD_FACTOR))))
    plan["sites"]["k_heads"] = k_steps
    # q: per-head (F.1-corrected shape)
    q_steps = []
    for h in range(16):
        step, pct, mse = mse_step_bits(head_samples["q"][h], bits)
        q_steps.append(int(step))
    plan["sites"]["q_heads"] = q_steps
    plan["sites"]["p"] = {"step_fx": max(1, int(np.ceil(1.02 / qmax(bits) * ONE)))}
    plan["weights"] = {"wq": [wscale_w(conf.q_slice(h), w_bits) for h in range(16)],
                       "wk": [wscale_w(conf.k_slice(h), w_bits) for h in range(8)],
                       "wv": [wscale_w(conf.v_slice(h), w_bits) for h in range(8)],
                       "wo": [wscale_w(conf.o_slice(h), w_bits) for h in range(16)],
                       "wg": wscale_w(conf.gate_slice(), w_bits),
                       "wu": wscale_w(conf.up_slice(), w_bits),
                       "wd": wscale_w(conf.down_slice(), w_bits)}
    return plan


def wscale_w(w: np.ndarray, w_bits: int) -> int:
    """Frozen weight rule: slice maximum -> symmetric integer scale for
    QmaxW = 2^(w-1)-1 (identical to the corrected F.1/F.3A rule at w=8)."""
    return max(1, int(np.ceil(np.max(np.abs(w)) * ONE / qmax(w_bits))))


def run_block_bits(conf: Conformer, hidden: np.ndarray, plan: dict, stats: dict) -> np.ndarray:
    """Single-scale integer pipeline at the candidate's activation bits."""
    bits = plan["bits"]
    seq = hidden.shape[0]
    heads, kv, hd = 16, 8, 128
    half = hd // 2
    causal = np.triu(np.full((seq, seq), -(64 << 20), dtype=np.int64), k=1)
    S = plan["sites"]
    W = plan["weights"]

    def gated(acc, mult, shift=20, lo=None, hi=None):
        lo_v, hi_v = int(acc.min()), int(acc.max())
        stats["acc_abs_max"] = max(stats.get("acc_abs_max", 0), max(abs(lo_v), abs(hi_v)))
        assert INT64_MIN <= lo_v and hi_v <= INT64_MAX, "int64 accumulator overflow"
        return lo_v, hi_v

    def requant_acc(acc, mult, shift=20):
        gated(acc, mult, shift)
        m = int(np.max(np.abs(acc))) if acc.size else 0
        bits = m.bit_length() + int(mult).bit_length()
        if "requant_links" in stats:
            stats["requant_links"].append({"max_abs_acc": m, "mult": int(mult),
                                           "shift": int(shift), "product_bits": bits})
        if bits <= 62:
            return np.clip(N.rshift_round_even(acc.astype(np.int64) * np.int64(mult), shift),
                           MIN_FX, MAX_FX)
        # exact ties-to-even bigint path (research executor only; the F.5A
        # report records exactly when this is required)
        s = int(shift)
        out = []
        for v in acc.reshape(-1):
            p = int(v) * int(mult)
            q, r = p >> s, p & ((1 << s) - 1)
            half = 1 << (s - 1)
            q += int(r > half or (r == half and (q & 1) == 1))
            out.append(q)
        big = np.array(out, dtype=object).reshape(acc.shape)
        return np.clip(np.vectorize(int, otypes=[np.int64])(big), MIN_FX, MAX_FX)

    wqmax = qmax(int(plan.get("w_bits", 8)))

    def wc(w, step):
        return np.clip(np.rint(w * ONE / step), -wqmax, wqmax)

    x = np.rint(hidden * ONE).astype(np.int64)
    h = N.op_rmsnorm(x, np.rint(conf.weights["input_layernorm.weight"] * ONE).astype(np.int64), 1)
    h8 = quant_bits(h.astype(np.int64), S["h"]["step_fx"], bits)

    inv = conf.cfg["rope_theta"] ** (-2.0 * np.arange(half) / hd)
    ang = np.outer(np.arange(seq), inv)
    table = np.stack([np.rint(np.cos(ang) * ONE), np.rint(np.sin(ang) * ONE)], axis=-1) \
        .reshape(-1).astype(np.int64)

    def rope(v):
        return N.op_rope_adjacent(v.astype(np.int32), table, half).astype(np.int64)

    qn = np.rint(conf.q_norm() * ONE).astype(np.int64)
    kn = np.rint(conf.k_norm() * ONE).astype(np.int64)
    q8s, k8s, v8s = [], [], []
    for qh in range(heads):
        acc = gemm64(h8, wc(conf.q_slice(qh), W["wq"][qh]), False).astype(np.int64)
        qv = requant_acc(acc, S["h"]["step_fx"] * W["wq"][qh])
        qv = N.op_rmsnorm(qv, qn, 1).astype(np.int64)
        qv = rope(qv)
        q8s.append(quant_bits(qv, S["q_heads"][qh], bits))
    for kh in range(kv):
        acc = gemm64(h8, wc(conf.k_slice(kh), W["wk"][kh]), False).astype(np.int64)
        kvx = requant_acc(acc, S["h"]["step_fx"] * W["wk"][kh])
        kvx = N.op_rmsnorm(kvx, kn, 1).astype(np.int64)
        kvx = rope(kvx)
        k8s.append(quant_bits(kvx, S["k_heads"][kh], bits))
        acc = gemm64(h8, wc(conf.v_slice(kh), W["wv"][kh]), False).astype(np.int64)
        gated(acc, 0)
        mult_v = max(1, int(round(S["h"]["step_fx"] * W["wv"][kh] / S["v"]["step_fx"])))
        v8s.append(np.clip(N.rshift_round_even(acc * np.int64(mult_v), 20), -qmax(bits), qmax(bits)))

    attn = np.zeros((seq, 1024), dtype=np.int64)
    for qh in range(heads):
        kvh = qh // 2
        acc = gemm64(q8s[qh], k8s[kvh], True).astype(np.int64)
        mult = max(1, int(round(S["q_heads"][qh] * S["k_heads"][kvh] * ATTN_SCALE_FX / ONE)))
        sfx = np.clip(N.rshift_round_even(acc * np.int64(mult), 20), MIN_FX, MAX_FX)
        gated(acc, mult)
        probs = N.op_softmax_rows(np.clip(sfx + causal, MIN_FX, MAX_FX).astype(np.int32)).astype(np.int64)
        p8 = quant_bits(probs, S["p"]["step_fx"], bits, lo=0)
        acc = gemm64(p8, v8s[kvh], False).astype(np.int64)
        gated(acc, 0)
        mult_c = max(1, int(round(S["p"]["step_fx"] * S["v"]["step_fx"] / S["ctx"]["step_fx"])))
        c8 = np.clip(N.rshift_round_even(acc * np.int64(mult_c), 20), -qmax(bits), qmax(bits))
        acc = gemm64(c8, wc(conf.o_slice(qh), W["wo"][qh]), False).astype(np.int64)
        ofx = requant_acc(acc, S["ctx"]["step_fx"] * W["wo"][qh])
        raw = attn + ofx
        stats["fx_saturations"] += int(np.count_nonzero((raw < MIN_FX) | (raw > MAX_FX)))
        attn = np.clip(raw, MIN_FX, MAX_FX)

    x1 = np.clip(x + attn, MIN_FX, MAX_FX)
    h2 = N.op_rmsnorm(x1.astype(np.int32),
                      np.rint(conf.weights["post_attention_layernorm.weight"] * ONE).astype(np.int64), 1)
    h2q = quant_bits(h2.astype(np.int64), S["h2"]["step_fx"], bits)
    acc = gemm64(h2q, wc(conf.gate_slice(), W["wg"]), False).astype(np.int64)
    gate = requant_acc(acc, S["h2"]["step_fx"] * W["wg"])
    acc = gemm64(h2q, wc(conf.up_slice(), W["wu"]), False).astype(np.int64)
    upv = requant_acc(acc, S["h2"]["step_fx"] * W["wu"])
    hm = N.op_mul(N.op_silu(gate.astype(np.int32)), upv.astype(np.int32)).astype(np.int64)
    hmq = quant_bits(hm, S["hm"]["step_fx"], bits)
    acc = gemm64(hmq, wc(conf.down_slice(), W["wd"]), False).astype(np.int64)
    down = requant_acc(acc, S["hm"]["step_fx"] * W["wd"])
    return np.clip(x1 + down, MIN_FX, MAX_FX)


# --- stages -----------------------------------------------------------------


def load_split(name: str):
    path = REPO / "testdata" / f"qwen3_f5a_{name}.json"
    if name == "heldout" and "--unlock-heldout" not in sys.argv:
        raise SystemExit("heldout is locked; pass --unlock-heldout")
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def evaluate(conf, plan, cases, max_cases=None):
    worst_cos, worst_abs, mean_abs, errs = 1.0, 0.0, 0.0, []
    n = 0
    for case in cases[: max_cases or len(cases)]:
        hidden = conf.embedding(case["token_ids"])
        ref, _, _ = instrumented_forward(conf, hidden, collect=True)
        y = run_block_bits(conf, hidden, plan, {"acc_abs_max": 0, "fx_saturations": 0})
        yf = y.astype(np.float64) / ONE
        cos = float(np.dot(yf.reshape(-1), ref.reshape(-1)) /
                    (np.linalg.norm(yf) * np.linalg.norm(ref)))
        err = np.abs(yf - ref)
        worst_cos = min(worst_cos, cos)
        worst_abs = max(worst_abs, float(err.max()))
        mean_abs += float(err.max())
        errs.append(err.reshape(-1))
        n += 1
    all_err = np.concatenate(errs)
    return {"worst_cosine": worst_cos, "worst_max_abs": worst_abs,
            "mean_max_abs": mean_abs / max(n, 1), "cases": n,
            "count_gt_005": int((all_err > 0.05).sum()),
            "count_gt_01": int((all_err > 0.1).sum()),
            "count_gt_02": int((all_err > 0.2).sum()),
            "p95_elem": float(np.percentile(all_err, 95)),
            "p99_elem": float(np.percentile(all_err, 99)),
            "histogram": {label: int(((all_err > lo) & (all_err <= hi)).sum())
                          for label, lo, hi in (
                              ("<=0.01", -1, 0.01), ("0.01-0.025", 0.01, 0.025),
                              ("0.025-0.05", 0.025, 0.05), ("0.05-0.1", 0.05, 0.1),
                              ("0.1-0.2", 0.1, 0.2), (">0.2", 0.2, 1e9))}}


def max_safe_k64(a_bits: int, w_bits: int) -> int:
    """Worst-case signed int64 bound: 2^(a+w-2) * K <= 2^63-1."""
    return ((1 << 63) - 1) // (1 << (a_bits + w_bits - 2))


def pair_family():
    """The 15 pairs excluded by int32 safety in F.4A (a+w from 22 to 26)."""
    pairs = []
    for a in range(8, 14):
        for w in range(8, 14):
            if a + w >= 22:
                pairs.append((a, w))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    for flag in ("anchor", "safety", "calibrate", "selection", "freeze", "heldout"):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--unlock-heldout", action="store_true")
    args = parser.parse_args()

    conf = Conformer(0)
    cases, samples, head_samples = collect_calibration(conf)
    pairs = pair_family()
    assert len(pairs) == 15, pairs
    print(f"calibration collected ({len(cases)} cases); family: {len(pairs)} pairs")

    if args.safety:
        from convert_qwen3_block import build_converted_graph
        j8 = build_plan(conf, samples, head_samples, 8, 8)
        chosen = {k: {"step_fx": v["step_fx"], "percentile": v["percentile"], "mse": v["mse"],
                      "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
                  for k, v in j8["sites"].items() if k not in ("k_heads", "q_heads", "p")}
        chosen["p"] = {"step_fx": j8["sites"]["p"]["step_fx"], "percentile": 100.0, "mse": 0.0,
                       "clipped_fraction": 0.0, "calibration_max_abs": 1.0}
        for i, st in enumerate(j8["sites"]["k_heads"]):
            chosen[f"k_h{i}"] = {"step_fx": st, "percentile": 100.0, "mse": 0.0,
                                 "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
        for i, st in enumerate(j8["sites"]["q_heads"]):
            chosen[f"q_h{i}"] = {"step_fx": st, "percentile": 100.0, "mse": 0.0,
                                 "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
        snake, _go, _plan, tdata, _gid = build_converted_graph(conf, 16, {"chosen": chosen, "p": 1.0})
        ks = []
        for node in snake["nodes"]:
            if node["operator_id"] != "GEMM_INT8_V1":
                continue
            a_ref = node["inputs"][0]
            kdim = (tdata[a_ref["index"]].shape[1] if a_ref["kind"] == 0
                    else snake["nodes"][a_ref["index"]]["output"]["shape"][1])
            ks.append(int(kdim))
        k_max = max(ks)
        table = {}
        for a, w in pairs:
            k64 = max_safe_k64(a, w)
            table[f"A{a}W{w}"] = {
                "a_bits": a, "w_bits": w,
                "a_max_magnitude": 1 << (a - 1), "w_max_magnitude": 1 << (w - 1),
                "worst_product": 1 << (a + w - 2),
                "max_safe_k32": max_safe_k_joint(a, w),
                "max_safe_k64": k64,
                "real_graph_kmax": k_max,
                "int32_safe": bool(max_safe_k_joint(a, w) >= k_max),
                "int64_safe": bool(k64 >= k_max),
                "why_new": ("excluded by int32 admission (MaxSafeK32 < 3072); "
                            "safe under the proposed int64 accumulator")}
        doc = {"formula": "max_safe_k64(a,w) = floor((2^63-1) / 2^(a+w-2))",
               "real_graph": {"gemm_nodes": len(ks), "k_max": k_max,
                              "all_k_values": sorted(set(ks))},
               "candidates": table,
               "note": "admission uses the THEORETICAL bound; observed ranges are recorded separately"}
        (REPO / "docs" / "phase-f5a-accumulator-safety.json").write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8")

        # Freivalds worst-case bounds (r in {0,1}^N; K=N=K_max upper bound)
        N = k_max
        fre = {"r_domain": "{0,1}^N", "K_upper": N, "N_upper": N, "candidates": {}}
        for a, w in pairs:
            x_bits = (N * (1 << (w - 1))).bit_length()
            y_bits = (N * (1 << (a - 1)) * N * (1 << (w - 1))).bit_length()
            z_bits = (N * N * (1 << (a + w - 2))).bit_length()
            fre["candidates"][f"A{a}W{w}"] = {
                "x_bits": x_bits, "y_bits": y_bits, "z_bits": z_bits,
                "required_signed_bits": max(x_bits, y_bits, z_bits) + 1,
                "int64_safe": max(x_bits, y_bits, z_bits) + 1 <= 63,
                "verdict": ("FREIVALDS_INT64_SAFE"
                            if max(x_bits, y_bits, z_bits) + 1 <= 63
                            else "FREIVALDS_REQUIRES_WIDER_INT")}
        (REPO / "docs" / "phase-f5a-freivalds-bounds.json").write_text(
            json.dumps(fre, indent=1) + "\n", encoding="utf-8")
        print(f"safety + freivalds bounds written; K_max={k_max}; "
              f"int64-safe all: {all(v['int64_safe'] for v in table.values())}; "
              f"freivalds int64-safe all: {all(v['int64_safe'] for v in fre['candidates'].values())}")

    if args.anchor:
        import f4a_joint_precision as F4A
        f4a_sel = json.loads((REPO / "testdata" / "qwen3_f4a_selection.json")
                             .read_text(encoding="utf-8"))["cases"][:4]
        for name, (a, w) in (("A12W9", (12, 9)), ("A13W8", (13, 8))):
            plan = build_plan(conf, samples, head_samples, a, w)
            f4a_plan = F4A.build_plan(conf, samples, head_samples, a, w)
            stats = {"acc_abs_max": 0, "fx_saturations": 0}
            for ci, case in enumerate(f4a_sel):
                hidden = conf.embedding(case["token_ids"])
                y_new = run_block_bits(conf, hidden, plan, stats)
                y_old = F4A.run_block_bits(conf, hidden, f4a_plan, {"acc_abs_max": 0, "fx_saturations": 0})
                delta = int(np.max(np.abs(y_new.astype(np.int64) - y_old.astype(np.int64))))
                assert delta == 0, f"anchor {name} case {ci}: int64 executor differs from F.4A (delta {delta})"
            print(f"anchor {name}: int64 executor BIT-IDENTICAL to the F.4A executor on 4 historical cases")
        f3a_heldout = json.loads((REPO / "testdata" / "qwen3_f3a_heldout.json")
                                 .read_text(encoding="utf-8"))["cases"][:8]
        plan_13_10 = build_plan(conf, samples, head_samples, 13, 10)
        res = evaluate_with_stats(conf, plan_13_10, f3a_heldout,
                                  {"acc_abs_max": 0, "fx_saturations": 0}, {})
        print(f"anchor diagnostic (UNSAFE_DIAGNOSTIC_ONLY in F.4A; new candidate here): A13W10 "
              f"cos={res['worst_cosine']:.5f} abs={res['worst_max_abs']:.4f} "
              f"(historical ~0.99993-0.99995 / 0.034-0.049)")
        assert 0.9998 <= res["worst_cosine"] <= 1.0 and res["worst_max_abs"] <= 0.06, res

    if args.calibrate:
        doc = {"model": "Qwen/Qwen3-0.6B-Base", "revision": "57ca99e94acb83175495aa2c6b6b0cc498170924",
               "layer": 0, "seq": 16, "accumulator_width": 64,
               "ranking": ["minimize a_bits + w_bits", "then minimize W_bits", "then minimize A_bits"],
               "candidates": {}}
        for a, w in pairs:
            plan = build_plan(conf, samples, head_samples, a, w)
            blob = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
            plan["policy_id"] = hashlib.sha256(blob).hexdigest()
            doc["candidates"][f"A{a}W{w}"] = plan
        (REPO / "docs" / "phase-f5a-candidate-policies.json").write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print("15 candidate policies written")

    if args.selection:
        doc = json.loads((REPO / "docs" / "phase-f5a-candidate-policies.json")
                         .read_text(encoding="utf-8"))
        selection = load_split("selection")
        ref_cache = {}
        results = {}
        requant = {}
        names = sorted(doc["candidates"], key=lambda n: (int(n[1:n.index("W")]) + int(n.split("W")[1]),
                                                         int(n.split("W")[1]), int(n[1:n.index("W")])))
        for name in names:
            plan = doc["candidates"][name]
            stats = {"acc_abs_max": 0, "fx_saturations": 0, "requant_links": []}
            res = evaluate_with_stats(conf, plan, selection, stats, ref_cache)
            v = "PASS" if res["worst_cosine"] >= 0.995 and res["worst_max_abs"] <= 0.05 else "FAIL"
            worst_bits = max([l["product_bits"] for l in stats["requant_links"]] or [0])
            results[name] = {**res, "verdict": v, "policy_id": plan["policy_id"],
                             "max_safe_k64": plan["theoretical_max_safe_k"],
                             "acc_abs_max_observed": stats["acc_abs_max"],
                             "requant_product_bits_max": worst_bits}
            requant[name] = {
                "max_acc_bits": max([l["max_abs_acc"].bit_length() for l in stats["requant_links"]] or [0]),
                "largest_multiplier": max([l["mult"] for l in stats["requant_links"]] or [0]),
                "max_product_bits": worst_bits,
                "int64_safe": worst_bits <= 63,
                "classification": ("REQUANT_INT64_SAFE" if worst_bits <= 63
                                   else "REQUIRES_WIDE_MULTIPLY"),
                "links": len(stats["requant_links"])}
            print(f"selection {name:7s}: cos={res['worst_cosine']:.5f} abs={res['worst_max_abs']:.4f} "
                  f">0.05={res['count_gt_005']:5d} req_bits={worst_bits} -> {v}")
        (REPO / "docs" / "phase-f5a-selection-results.json").write_text(
            json.dumps(results, indent=1) + "\n", encoding="utf-8")
        (REPO / "docs" / "phase-f5a-requant-bounds.json").write_text(
            json.dumps({"note": "observed accumulator magnitudes and requant multiplier products "
                                "over the F5A selection split; classification per section 21",
                        "cumulative": requant}, indent=1) + "\n", encoding="utf-8")

    if args.freeze:
        doc = json.loads((REPO / "docs" / "phase-f5a-candidate-policies.json")
                         .read_text(encoding="utf-8"))
        sel = json.loads((REPO / "docs" / "phase-f5a-selection-results.json")
                         .read_text(encoding="utf-8"))
        requant = json.loads((REPO / "docs" / "phase-f5a-requant-bounds.json")
                             .read_text(encoding="utf-8"))["cumulative"]
        family = {"family": [f"A{a}W{w}" for a, w in pairs],
                  "ranking": doc["ranking"], "accumulator_width": 64, "candidates": {}}
        for name, plan in doc["candidates"].items():
            family["candidates"][name] = {
                "a_bits": plan["bits"], "w_bits": plan["w_bits"],
                "policy_id": plan["policy_id"],
                "max_safe_k64": plan["theoretical_max_safe_k"],
                "safe": bool(plan["theoretical_max_safe_k"] >= 3072),
                "requant": requant[name]["classification"],
                "selection": {k: sel[name][k] for k in
                              ("worst_cosine", "worst_max_abs", "verdict")}}
        blob = json.dumps(family["candidates"], sort_keys=True, separators=(",", ":")).encode()
        family["family_sha256"] = hashlib.sha256(blob).hexdigest()
        (REPO / "docs" / "phase-f5a-frozen-family.json").write_text(
            json.dumps(family, indent=1) + "\n", encoding="utf-8")
        print("frozen family sha256:", family["family_sha256"][:16])

    if args.heldout:
        fam = json.loads((REPO / "docs" / "phase-f5a-frozen-family.json")
                         .read_text(encoding="utf-8"))
        pol = json.loads((REPO / "docs" / "phase-f5a-candidate-policies.json")
                         .read_text(encoding="utf-8"))
        heldout = load_split("heldout")
        ref_cache = {}
        results = {}
        passers = []
        for name in fam["family"]:
            entry = fam["candidates"][name]
            plan = pol["candidates"][name]
            payload = {"bits": plan["bits"], "w_bits": plan["w_bits"],
                       "sites": plan["sites"], "weights": plan["weights"]}
            stats = {"acc_abs_max": 0, "fx_saturations": 0}
            res = evaluate_with_stats(conf, payload, heldout, stats, ref_cache)
            v = "PASS" if (res["worst_cosine"] >= 0.995 and res["worst_max_abs"] <= 0.05
                           and res["count_gt_005"] == 0 and entry["safe"]) else "FAIL"
            results[name] = {**res, "verdict": v, "policy_id": entry["policy_id"],
                             "a_bits": entry["a_bits"], "w_bits": entry["w_bits"],
                             "max_safe_k64": entry["max_safe_k64"],
                             "requant": entry["requant"],
                             "acc_abs_max_observed": stats["acc_abs_max"],
                             "fx_saturations": stats["fx_saturations"]}
            if v == "PASS":
                passers.append((entry["a_bits"] + entry["w_bits"], entry["w_bits"],
                                entry["a_bits"], name))
            print(f"heldout {name:7s}: cos={res['worst_cosine']:.5f} abs={res['worst_max_abs']:.4f} "
                  f">0.05={res['count_gt_005']} acc_max={stats['acc_abs_max']} -> {v}")
        if passers:
            passers.sort()
            winner = passers[0][3]
            verdict = f"FEASIBLE_INT64_{winner}"
        else:
            winner = None
            verdict = "NOT_FEASIBLE_WITHIN_A13_W13_INT64_FRONTIER"
        doc = {"verdict": verdict, "winner": winner,
               "all_passers_by_ranking": [p[3] for p in sorted(passers)],
               "thresholds": {"worst_cosine": 0.995, "worst_max_abs": 0.05,
                              "count_gt_005_must_be": 0},
               "proceed_to_f5b": "YES" if winner else "NO",
               "results": results,
               "histograms": {k: v["histogram"] for k, v in results.items()}}
        (REPO / "docs" / "phase-f5a-heldout-results.json").write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        (REPO / "docs" / "phase-f5a-error-histograms.json").write_text(
            json.dumps(doc["histograms"], indent=1) + "\n", encoding="utf-8")
        print("VERDICT:", verdict)


def evaluate_with_stats(conf, plan, cases, stats, ref_cache=None):
    worst_cos, worst_abs, mean_abs, errs = 1.0, 0.0, 0.0, []
    for ci, case in enumerate(cases):
        if ref_cache is not None and (ci, "h") in ref_cache:
            hidden, ref = ref_cache[(ci, "h")], ref_cache[(ci, "ref")]
        else:
            hidden = conf.embedding(case["token_ids"])
            ref, _, _ = instrumented_forward(conf, hidden, collect=True)
            if ref_cache is not None:
                ref_cache[(ci, "h")] = hidden
                ref_cache[(ci, "ref")] = ref
        y = run_block_bits(conf, hidden, plan, stats)
        yf = y.astype(np.float64) / ONE
        cos = float(np.dot(yf.reshape(-1), ref.reshape(-1)) /
                    (np.linalg.norm(yf) * np.linalg.norm(ref)))
        err = np.abs(yf - ref)
        worst_cos = min(worst_cos, cos)
        worst_abs = max(worst_abs, float(err.max()))
        mean_abs += float(err.max())
        errs.append(err.reshape(-1))
    all_err = np.concatenate(errs)
    return {"worst_cosine": worst_cos, "worst_max_abs": worst_abs,
            "mean_max_abs": mean_abs / len(cases), "cases": len(cases),
            "count_gt_0025": int((all_err > 0.025).sum()),
            "count_gt_005": int((all_err > 0.05).sum()),
            "count_gt_01": int((all_err > 0.1).sum()),
            "count_gt_02": int((all_err > 0.2).sum()),
            "p95_elem": float(np.percentile(all_err, 95)),
            "p99_elem": float(np.percentile(all_err, 99)),
            "histogram": {label: int(((all_err > lo) & (all_err <= hi)).sum())
                          for label, lo, hi in (
                              ("<=0.01", -1, 0.01), ("0.01-0.025", 0.01, 0.025),
                              ("0.025-0.05", 0.025, 0.05), ("0.05-0.1", 0.05, 0.1),
                              ("0.1-0.2", 0.1, 0.2), (">0.2", 0.2, 1e9))}}


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
