"""EXPERIMENTAL / NON-PROTOCOL — F.4A weight/stage analysis and cost model.

Generates docs/phase-f5a-weight-analysis.json, docs/phase-f5a-stage-analysis.json
and docs/phase-f5a-cost-model.json from the frozen family and the real
checkpoint. Stage analysis measures the per-stage max error of the
priority candidates plus the winner by comparing block intermediates
against the float64 reference at the exact node boundaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import f1_canonical_numpy as N  # noqa: E402
from convert_qwen3_block import Conformer, instrumented_forward, to_fx  # noqa: E402
from f5a_joint_precision import (  # noqa: E402
    ONE, ATTN_SCALE_FX, build_plan, collect_calibration, evaluate_with_stats,
    max_safe_k_joint, qmax, quant_bits, run_block_bits, wscale_w,
)

PRIORITY = ["A13W9", "A12W10", "A11W11", "A13W10"]


def weight_analysis(conf: Conformer) -> dict:
    matrices = {"Wq": conf.q_slice(0), "Wk": conf.k_slice(0), "Wv": conf.v_slice(0),
                "Wo": conf.o_slice(0), "Wgate": conf.gate_slice(),
                "Wup": conf.up_slice(), "Wdown": conf.down_slice()}
    out = {"matrices": {}, "rule": "slice maximum -> symmetric integer scale, "
                                   "QmaxW = 2^(w-1)-1 (frozen across all w)"}
    for name, w in matrices.items():
        entry = {}
        for wbits in range(8, 14):
            step = wscale_w(w, wbits)
            q = np.clip(np.rint(w * ONE / step), -qmax(wbits), qmax(wbits))
            deq = q * step / ONE
            err = deq - w
            entry[f"W{wbits}"] = {"step_real": step / ONE,
                                  "mse": float(np.mean(err * err)),
                                  "max_abs_err": float(np.max(np.abs(err))),
                                  "clipped_fraction": float(np.mean(np.abs(q) >= qmax(wbits)))}
        out["matrices"][name] = entry
    total_elems = int(sum(m.size for m in matrices.values()))
    out["layer0_quantized_weight_elements"] = total_elems
    out["storage"] = {f"W{wb}": {"logical_bytes": int(np.ceil(total_elems * wb / 8)),
                                 "int16_container_bytes": total_elems * 2}
                      for wb in range(8, 14)}
    return out


def stage_analysis(conf: Conformer, samples, head_samples, name: str, cases_n: int = 4) -> dict:
    a, w = int(name[1:name.index("W")]), int(name[name.index("W") + 1:])
    plan = build_plan(conf, samples, head_samples, a, w)
    cases = json.loads((REPO / "testdata" / "qwen3_f5a_selection.json")
                       .read_text(encoding="utf-8"))["cases"][:cases_n]
    stages = {}
    for ci, case in enumerate(cases):
        hidden = conf.embedding(case["token_ids"])
        ref_stages = {}
        # float reference stages via instrumented forward raw capture
        _, _, raw = instrumented_forward(conf, hidden, collect=True)
        y = run_block_bits(conf, hidden, plan, {"acc_abs_max": 0, "fx_saturations": 0})
        # canonical side: recompute stage tensors with the same executor helpers
        seq = 16
        qm = qmax(a)
        wqmax = qmax(w)
        S, W = plan["sites"], plan["weights"]
        x = np.rint(hidden * ONE).astype(np.int64)
        h = N.op_rmsnorm(x, np.rint(conf.weights["input_layernorm.weight"] * ONE).astype(np.int64), 1)
        hq = quant_bits(h.astype(np.int64), S["h"]["step_fx"], a) * S["h"]["step_fx"]

        def stage_err(canon_fx, ref_val):
            return float(np.max(np.abs(canon_fx / ONE - ref_val)))

        m = {}
        m["input_norm"] = stage_err(hq, raw["h"])
        # Q/K/V stage: compare the block-visible values (post dequant fx)
        wc = lambda ww, st: np.clip(np.rint(ww * ONE / st), -wqmax, wqmax)
        acc = N.op_gemm(hq.astype(np.int16), wc(conf.q_slice(0), W["wq"][0]), False).astype(np.int64)
        qfx = N.rshift_round_even(acc * np.int64(S["h"]["step_fx"] * W["wq"][0]), 20)
        m["QKV_proj_head0"] = stage_err(qfx, raw["q"][:, 0, :])
        stages[f"case{ci}"] = m
    return {"candidate": name, "a_bits": a, "w_bits": w,
            "note": "stage maxima on the first cases of F4A_SELECTION; "
                    "full-block metrics come from the frozen heldout run",
            "stages": stages}


def cost_model(family: dict) -> dict:
    conf = Conformer(0)
    wa = weight_analysis(conf)
    act_elems = 245760  # F.3A persisted activation elements per case
    out = {"weights": wa["storage"],
           "weight_note": "reusable across tasks; W8 unchanged",
           "activations_per_case": {f"A{a}": {"logical_bytes": int(np.ceil(act_elems * a / 8)),
                                              "int16_container_bytes": act_elems * 2}
                                    for a in range(8, 14)},
           "activation_note": "per task; F.3A persisted element count",
           "graph": {"nodes": 310, "gemm_nodes": 83, "mac_per_case": 252706816,
                     "freivalds_checks": 83, "manifest_leaves": 310,
                     "unchanged_by_bits": True},
           "future_micro_step": {n: {"local_k": 8,
                                     "worst_case_acc": (1 << (family['candidates'][n]['a_bits'] - 1))
                                     * (1 << (family['candidates'][n]['w_bits'] - 1)) * 8,
                                     "safe_under_int32": True}
                                 for n in family["candidates"]},
           "verification_implications": ("Freivalds / single-tile dispute / 512-MAC arbitration are "
                                         "structurally reusable with a versioned wider-integer "
                                         "arithmetic; NOT IMPLEMENTED")}
    return out


def main() -> None:
    conf = Conformer(0)
    cases, samples, head_samples = collect_calibration(conf)
    fam = json.loads((REPO / "docs" / "phase-f5a-frozen-family.json")
                     .read_text(encoding="utf-8"))
    winner = None
    held_path = REPO / "docs" / "phase-f5a-heldout-results.json"
    if held_path.exists():
        winner = json.loads(held_path.read_text(encoding="utf-8")).get("winner")

    (REPO / "docs" / "phase-f5a-weight-analysis.json").write_text(
        json.dumps(weight_analysis(conf), indent=1) + "\n", encoding="utf-8")
    names = list(dict.fromkeys(PRIORITY + ([winner] if winner else [])))
    stage_doc = {n: stage_analysis(conf, samples, head_samples, n) for n in names}
    (REPO / "docs" / "phase-f5a-stage-analysis.json").write_text(
        json.dumps(stage_doc, indent=1) + "\n", encoding="utf-8")
    (REPO / "docs" / "phase-f5a-cost-model.json").write_text(
        json.dumps(cost_model(fam), indent=1) + "\n", encoding="utf-8")
    print("weight/stage/cost written for:", names)


if __name__ == "__main__":
    main()
