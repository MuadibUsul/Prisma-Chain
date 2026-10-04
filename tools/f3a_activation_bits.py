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
from convert_qwen3_block import Conformer, instrumented_forward, to_fx, K_HEAD_FACTOR  # noqa: E402
from f2a_groupwise_search import wscale  # noqa: E402

ONE = 1 << 20
MIN_FX, MAX_FX = -(1 << 31), (1 << 31) - 1
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
ATTN_SCALE_FX = 92681
BITS = [8, 9, 10, 11, 12, 13]
K_MAX_BLOCK = 3072


def qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def max_safe_k(bits: int) -> int:
    return (2**31 - 1) // ((1 << (bits - 1)) * 128)


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


def build_plan(conf: Conformer, samples, head_samples, bits: int) -> dict:
    """One candidate's frozen scales for a given activation bit depth."""
    plan = {"bits": bits, "qmax": qmax(bits), "sites": {}, "theoretical_max_safe_k": max_safe_k(bits)}
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
    plan["weights"] = {"wq": [wscale(conf.q_slice(h)) for h in range(16)],
                       "wk": [wscale(conf.k_slice(h)) for h in range(8)],
                       "wv": [wscale(conf.v_slice(h)) for h in range(8)],
                       "wo": [wscale(conf.o_slice(h)) for h in range(16)],
                       "wg": wscale(conf.gate_slice()), "wu": wscale(conf.up_slice()),
                       "wd": wscale(conf.down_slice())}
    return plan


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
        stats["acc_abs_max"] = max(stats["acc_abs_max"], int(np.max(np.abs(acc))))
        assert INT32_MIN <= int(acc.min()) and int(acc.max()) <= INT32_MAX, "int32 accumulator overflow"
        return int(acc.min()), int(acc.max())

    def requant_acc(acc, mult, shift=20):
        gated(acc, mult, shift)
        return np.clip(N.rshift_round_even(acc.astype(np.int64) * np.int64(mult), shift), MIN_FX, MAX_FX)

    def wc(w, step):
        return np.clip(np.rint(w * ONE / step), -127, 127).astype(np.int8)

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
        acc = N.op_gemm(h8, wc(conf.q_slice(qh), W["wq"][qh]), False).astype(np.int64)
        qv = requant_acc(acc, S["h"]["step_fx"] * W["wq"][qh])
        qv = N.op_rmsnorm(qv, qn, 1).astype(np.int64)
        qv = rope(qv)
        q8s.append(quant_bits(qv, S["q_heads"][qh], bits))
    for kh in range(kv):
        acc = N.op_gemm(h8, wc(conf.k_slice(kh), W["wk"][kh]), False).astype(np.int64)
        kvx = requant_acc(acc, S["h"]["step_fx"] * W["wk"][kh])
        kvx = N.op_rmsnorm(kvx, kn, 1).astype(np.int64)
        kvx = rope(kvx)
        k8s.append(quant_bits(kvx, S["k_heads"][kh], bits))
        acc = N.op_gemm(h8, wc(conf.v_slice(kh), W["wv"][kh]), False).astype(np.int64)
        gated(acc, 0)
        mult_v = max(1, int(round(S["h"]["step_fx"] * W["wv"][kh] / S["v"]["step_fx"])))
        v8s.append(np.clip(N.rshift_round_even(acc * np.int64(mult_v), 20), -qmax(bits), qmax(bits)))

    attn = np.zeros((seq, 1024), dtype=np.int64)
    for qh in range(heads):
        kvh = qh // 2
        acc = N.op_gemm(q8s[qh], k8s[kvh], True).astype(np.int64)
        mult = max(1, int(round(S["q_heads"][qh] * S["k_heads"][kvh] * ATTN_SCALE_FX / ONE)))
        sfx = np.clip(N.rshift_round_even(acc * np.int64(mult), 20), MIN_FX, MAX_FX)
        gated(acc, mult)
        probs = N.op_softmax_rows(np.clip(sfx + causal, MIN_FX, MAX_FX).astype(np.int32)).astype(np.int64)
        p8 = quant_bits(probs, S["p"]["step_fx"], bits, lo=0)
        acc = N.op_gemm(p8, v8s[kvh], False).astype(np.int64)
        gated(acc, 0)
        mult_c = max(1, int(round(S["p"]["step_fx"] * S["v"]["step_fx"] / S["ctx"]["step_fx"])))
        c8 = np.clip(N.rshift_round_even(acc * np.int64(mult_c), 20), -qmax(bits), qmax(bits))
        acc = N.op_gemm(c8, wc(conf.o_slice(qh), W["wo"][qh]), False).astype(np.int64)
        ofx = requant_acc(acc, S["ctx"]["step_fx"] * W["wo"][qh])
        raw = attn + ofx
        stats["fx_saturations"] += int(np.count_nonzero((raw < MIN_FX) | (raw > MAX_FX)))
        attn = np.clip(raw, MIN_FX, MAX_FX)

    x1 = np.clip(x + attn, MIN_FX, MAX_FX)
    h2 = N.op_rmsnorm(x1.astype(np.int32),
                      np.rint(conf.weights["post_attention_layernorm.weight"] * ONE).astype(np.int64), 1)
    h2q = quant_bits(h2.astype(np.int64), S["h2"]["step_fx"], bits)
    acc = N.op_gemm(h2q, wc(conf.gate_slice(), W["wg"]), False).astype(np.int64)
    gate = requant_acc(acc, S["h2"]["step_fx"] * W["wg"])
    acc = N.op_gemm(h2q, wc(conf.up_slice(), W["wu"]), False).astype(np.int64)
    upv = requant_acc(acc, S["h2"]["step_fx"] * W["wu"])
    hm = N.op_mul(N.op_silu(gate.astype(np.int32)), upv.astype(np.int32)).astype(np.int64)
    hmq = quant_bits(hm, S["hm"]["step_fx"], bits)
    acc = N.op_gemm(hmq, wc(conf.down_slice(), W["wd"]), False).astype(np.int64)
    down = requant_acc(acc, S["hm"]["step_fx"] * W["wd"])
    return np.clip(x1 + down, MIN_FX, MAX_FX)


# --- stages -----------------------------------------------------------------


def load_split(name: str):
    path = REPO / "testdata" / f"qwen3_f3a_{name}.json"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    for flag in ("anchor", "safety", "calibrate", "selection", "freeze", "heldout"):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--unlock-heldout", action="store_true")
    args = parser.parse_args()

    conf = Conformer(0)
    cases, samples, head_samples = collect_calibration(conf)
    print(f"calibration collected ({len(cases)} cases)")

    if args.safety:
        from convert_qwen3_block import build_converted_graph
        plan8 = build_plan(conf, samples, head_samples, 8)
        chosen = {k: {"step_fx": v["step_fx"], "percentile": v["percentile"], "mse": v["mse"],
                      "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
                  for k, v in plan8["sites"].items() if k not in ("k_heads", "q_heads", "p")}
        chosen["p"] = {"step_fx": plan8["sites"]["p"]["step_fx"], "percentile": 100.0, "mse": 0.0,
                       "clipped_fraction": 0.0, "calibration_max_abs": 1.0}
        for i, s in enumerate(plan8["sites"]["k_heads"]):
            chosen[f"k_h{i}"] = {"step_fx": s, "percentile": 100.0, "mse": 0.0,
                                 "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
        for i, s in enumerate(plan8["sites"]["q_heads"]):
            chosen[f"q_h{i}"] = {"step_fx": s, "percentile": 100.0, "mse": 0.0,
                                 "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
        snake, go, plan, tdata, gid = build_converted_graph(conf, 16, {"chosen": chosen, "p": 1.0})
        rows = []
        for node in snake["nodes"]:
            if node["operator_id"] != "GEMM_INT8_V1":
                continue
            a = node["inputs"][0]
            kdim = (tdata[a["index"]].shape[1] if a["kind"] == 0
                    else snake["nodes"][a["index"]]["output"]["shape"][1])
            row = {"node_id": node["node_id"], "K": int(kdim)}
            for b in BITS:
                row[f"A{b}_max_safe_k"] = max_safe_k(b)
                row[f"A{b}_safe"] = bool(kdim <= max_safe_k(b))
            rows.append(row)
        doc = {"real_graph_K_max": int(max(r["K"] for r in rows)),
               "theoretical": {f"A{b}": max_safe_k(b) for b in BITS},
               "note": ("admission must use the theoretical bound; observed ranges are "
                        "recorded separately and never substitute for it"),
               "nodes": rows}
        (REPO / "docs" / "phase-f3a-accumulator-safety.json").write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"safety written; K_max={doc['real_graph_K_max']}; "
              f"A13 safe={all(r['A13_safe'] for r in rows)}; "
              f"A14 would be safe={K_MAX_BLOCK <= max_safe_k(14)}")

    if args.anchor:
        # A8 must reproduce the converted graph bit-for-bit
        import json as _json
        from convert_qwen3_block import build_converted_graph
        plan8 = build_plan(conf, samples, head_samples, 8)
        chosen = {k: {"step_fx": v["step_fx"], "percentile": v["percentile"], "mse": v["mse"],
                      "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
                  for k, v in plan8["sites"].items() if k not in ("k_heads", "q_heads", "p")}
        chosen["p"] = {"step_fx": plan8["sites"]["p"]["step_fx"], "percentile": 100.0, "mse": 0.0,
                       "clipped_fraction": 0.0, "calibration_max_abs": 1.0}
        for i, s in enumerate(plan8["sites"]["k_heads"]):
            chosen[f"k_h{i}"] = {"step_fx": s, "percentile": 100.0, "mse": 0.0,
                                 "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
        for i, s in enumerate(plan8["sites"]["q_heads"]):
            chosen[f"q_h{i}"] = {"step_fx": s, "percentile": 100.0, "mse": 0.0,
                                 "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
        snake, go, plan, tdata, gid = build_converted_graph(conf, 16, {"chosen": chosen, "p": 1.0})
        hidden = conf.embedding(cases[0]["token_ids"])
        inputs = dict(tdata)
        inputs[0] = to_fx(hidden).astype(np.int32)
        sc = _json.loads(_json.dumps(snake, default=lambda b: b.hex()))
        sc["inputs"][0]["root"] = N.tensor_root_np(snake["inputs"][0]["desc"], inputs[0])
        for gi in sc["inputs"][1:]:
            gi["root"] = bytes.fromhex(gi["root"])
        out = N.execute_graph_np(sc, inputs, rope_tables={2: tdata[2]})
        graph_y = out["tensors"][(1, snake["outputs"][0]["index"])].astype(np.int64)
        y = run_block_bits(conf, hidden, plan8, {"acc_abs_max": 0, "fx_saturations": 0})
        delta = int(np.max(np.abs(graph_y - y)))
        print(f"anchor: A8 executor vs converted graph max |delta| = {delta}")
        assert delta == 0, "A8 does not reproduce the corrected baseline"

    if args.calibrate:
        doc = {"family": [f"W8A{b}" for b in BITS], "candidates": {}}
        for b in BITS:
            plan = build_plan(conf, samples, head_samples, b)
            blob = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
            plan["policy_id"] = hashlib.sha256(blob).hexdigest()
            doc["candidates"][f"A{b}"] = plan
            print(f"A{b}: policy_id={plan['policy_id'][:16]} "
                  f"step(h)={plan['sites']['h']['step_fx']/ONE:.5f} "
                  f"p={plan['sites']['p']['step_fx']}")
        (REPO / "docs" / "phase-f3a-candidate-policies.json").write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print("candidate policies written")

    if args.selection:
        doc = json.loads((REPO / "docs" / "phase-f3a-candidate-policies.json")
                         .read_text(encoding="utf-8"))
        selection = load_split("selection")
        results = {}
        for name, plan in doc["candidates"].items():
            res = evaluate(conf, plan, selection)
            verdict = "PASS" if res["worst_cosine"] >= 0.995 and res["worst_max_abs"] <= 0.05 else "FAIL"
            results[name] = {**res, "verdict": verdict, "policy_id": plan["policy_id"]}
            print(f"selection {name}: cos={res['worst_cosine']:.5f} abs={res['worst_max_abs']:.4f} "
                  f">0.05={res['count_gt_005']} -> {verdict}")
        (REPO / "docs" / "phase-f3a-selection-results.json").write_text(
            json.dumps(results, indent=1) + "\n", encoding="utf-8")

    if args.freeze:
        doc = json.loads((REPO / "docs" / "phase-f3a-candidate-policies.json")
                         .read_text(encoding="utf-8"))
        safety = json.loads((REPO / "docs" / "phase-f3a-accumulator-safety.json")
                            .read_text(encoding="utf-8"))
        frozen = {"family": doc["family"], "k_max_real_graph": safety["real_graph_K_max"],
                  "candidates": {}}
        for name, plan in doc["candidates"].items():
            b = plan["bits"]
            frozen["candidates"][name] = {
                "bits": b, "policy_id": plan["policy_id"],
                "theoretical_max_safe_k": max_safe_k(b),
                "all_gemm_nodes_safe": bool(safety["real_graph_K_max"] <= max_safe_k(b)),
                "sites": plan["sites"], "weights": plan["weights"]}
        blob = json.dumps(frozen["candidates"], sort_keys=True, separators=(",", ":")).encode()
        frozen["family_sha256"] = hashlib.sha256(blob).hexdigest()
        (REPO / "docs" / "phase-f3a-frozen-candidates.json").write_text(
            json.dumps(frozen, indent=1) + "\n", encoding="utf-8")
        print("frozen candidates written; family sha256:", frozen["family_sha256"][:16])

    if args.heldout:
        frozen = json.loads((REPO / "docs" / "phase-f3a-frozen-candidates.json")
                            .read_text(encoding="utf-8"))
        heldout = load_split("heldout")
        results = {}
        lowest_pass = None
        for name in ("A8", "A9", "A10", "A11", "A12", "A13"):
            entry = frozen["candidates"][name]
            plan = {"bits": entry["bits"], "qmax": qmax(entry["bits"]),
                    "sites": entry["sites"], "weights": entry["weights"]}
            stats = {"acc_abs_max": 0, "fx_saturations": 0}
            res = evaluate_with_stats(conf, plan, heldout, stats)
            v = "PASS" if (res["worst_cosine"] >= 0.995 and res["worst_max_abs"] <= 0.05
                           and entry["all_gemm_nodes_safe"]) else "FAIL"
            results[name] = {**res, "verdict": v, "policy_id": entry["policy_id"],
                             "acc_abs_max_observed": stats["acc_abs_max"],
                             "fx_saturations": stats["fx_saturations"]}
            if v == "PASS" and lowest_pass is None:
                lowest_pass = name
            print(f"heldout {name}: cos={res['worst_cosine']:.5f} abs={res['worst_max_abs']:.4f} "
                  f">0.05={res['count_gt_005']} acc_max={stats['acc_abs_max']} -> {v}")
        verdict = f"FEASIBLE_{lowest_pass}" if lowest_pass else "NOT_FEASIBLE_THROUGH_A13"
        doc = {"verdict": verdict, "lowest_passing": lowest_pass,
               "thresholds": {"worst_cosine": 0.995, "worst_max_abs": 0.05},
               "proceed_to_f3b": "YES" if lowest_pass else "NO",
               "results": results, "histograms": {k: v["histogram"] for k, v in results.items()},
               "historical_note": "F.2A heldout numbers remain on the F.2A branch only"}
        (REPO / "docs" / "phase-f3a-heldout-results.json").write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        (REPO / "docs" / "phase-f3a-error-histograms.json").write_text(
            json.dumps(doc["histograms"], indent=1) + "\n", encoding="utf-8")
        print("VERDICT:", verdict)


def evaluate_with_stats(conf, plan, cases, stats):
    worst_cos, worst_abs, mean_abs, errs = 1.0, 0.0, 0.0, []
    for case in cases:
        hidden = conf.embedding(case["token_ids"])
        ref, _, _ = instrumented_forward(conf, hidden, collect=True)
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
