"""Quantization-error attribution for QWEN3_BLOCK_PROFILE_V1.

Composed ENTIRELY from the bit-exact numpy operator kernels
(f1_canonical_numpy), plus a per-site switch that skips the int8
quantizer of ONE site at a time. With all quantizers on, the mirror is
asserted equal to the real converted graph; switching one quantizer off
isolates that site's contribution to the final output error.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "tools")
sys.path.insert(0, "compute/canonical/python")

import f1_canonical_numpy as N  # noqa: E402
from convert_qwen3_block import (  # noqa: E402
    Conformer, instrumented_forward, load_token_cases, mse_optimal_scale,
)

ONE = 1 << 20


def q_step_mult(step: int) -> int:
    """The converter's fx->int8 multiplier: round(2^30 / step)."""
    return max(1, int(round((1 << 30) / step)))


def weight_scale(w: np.ndarray) -> int:
    return max(1, int(np.ceil(np.max(np.abs(w)) * ONE / 127)))


def wcounts(w: np.ndarray, step: int) -> np.ndarray:
    return np.clip(np.rint(w * ONE / step), -127, 127).astype(np.int8)


def deq_fx(acc: np.ndarray, mult: int) -> np.ndarray:
    """The converter's accum -> q12.20 dequant: clip(rshift_even(acc*mult, 20))."""
    v = N.rshift_round_even(acc.astype(np.int64) * np.int64(mult), 20)
    return np.clip(v, -(1 << 31), (1 << 31) - 1)


def block(conf: Conformer, hidden: np.ndarray, steps: dict, wsteps: dict, off: set) -> np.ndarray:
    """Canonical pipeline; off = sites whose int8 quantizer is skipped.

    Mirrors the converted graph's exact rounding structure: GEMMs run on
    int8 counts, then ONE explicit dequant REQ (mult/shift identical to
    the converter's), and int8 sites use the fx->int8 or accum->int8
    quantizer the converter chose for that site.
    """
    seq = hidden.shape[0]
    heads, kv, hd = 16, 8, 128
    half = hd // 2
    mask_fx = -(64 << 20)

    def site_fx(name, v, lo=-127, hi=127):
        if name in off:
            return v.astype(np.int64), None
        mult = q_step_mult(steps[name])
        q = np.clip(N.rshift_round_even(v.astype(np.int64) * np.int64(mult), 30), lo, hi)
        return q.astype(np.int64) * steps[name], steps[name]

    def site_acc(name, v, mult, step):
        if name in off:
            return v.astype(np.int64), None
        q = np.clip(N.rshift_round_even(v.astype(np.int64) * np.int64(mult), 20), -127, 127)
        return q.astype(np.int64) * step, step

    # per-slice weight scales, exactly as the converter derives them
    wq_s = [weight_scale(conf.q_slice(h)) for h in range(heads)]
    wk_s = [weight_scale(conf.k_slice(h)) for h in range(kv)]
    wv_s = [weight_scale(conf.v_slice(h)) for h in range(kv)]
    wo_s = [weight_scale(conf.o_slice(h)) for h in range(heads)]

    x = np.rint(hidden * ONE).astype(np.int64)
    h = N.op_rmsnorm(x, np.rint(conf.weights["input_layernorm.weight"] * ONE).astype(np.int64), 1)
    h8, s_h = site_fx("h", h.astype(np.int64))
    qa_h = steps["h"] if "h" not in off else 1

    inv = conf.cfg["rope_theta"] ** (-2.0 * np.arange(half) / hd)
    ang = np.outer(np.arange(seq), inv)
    table = np.stack([np.rint(np.cos(ang) * ONE), np.rint(np.sin(ang) * ONE)], axis=-1)         .reshape(-1).astype(np.int64)

    def rope(v):
        return N.op_rope_adjacent(v.astype(np.int32), table, half).astype(np.int64)

    qn = np.rint(conf.q_norm() * ONE).astype(np.int64)
    kn = np.rint(conf.k_norm() * ONE).astype(np.int64)
    qs, ks, vs = [], [], []
    wq_eff = 1 if "wq" in off else None
    for qh in range(heads):
        ws = wq_s[qh]
        acc = N.op_gemm(h8.astype(np.int8), wcounts(conf.q_slice(qh), ws), False).astype(np.int64)
        qv = deq_fx(acc, qa_h * ws)
        qv = N.op_rmsnorm(qv, qn, 1).astype(np.int64)
        qv = rope(qv)
        qv, _ = site_fx("q", qv)
        qs.append(qv)
    for kh in range(kv):
        ws = wk_s[kh]
        acc = N.op_gemm(h8.astype(np.int8), wcounts(conf.k_slice(kh), ws), False).astype(np.int64)
        kvx = deq_fx(acc, qa_h * ws)
        kvx = N.op_rmsnorm(kvx, kn, 1).astype(np.int64)
        kvx = rope(kvx)
        kvx, _ = site_fx("k", kvx)
        ks.append(kvx)
        ws = wv_s[kh]
        acc = N.op_gemm(h8.astype(np.int8), wcounts(conf.v_slice(kh), ws), False).astype(np.int64)
        mult_v = max(1, int(round(qa_h * ws / steps["v"])))
        vv, _ = site_acc("v", acc, mult_v, steps["v"])
        vs.append(vv)

    f = 92681
    mult_scores = max(1, int(round(steps["q"] * steps["k"] * f / ONE)))
    attn_sum = np.zeros((seq, 1024), dtype=np.int64)
    for qh in range(heads):
        kvh = qh // 2
        sacc = N.op_gemm(qs[qh].astype(np.int8), ks[kvh].astype(np.int8), True).astype(np.int64)
        sfx = np.clip(N.rshift_round_even(sacc * np.int64(mult_scores), 20), -(1 << 31), (1 << 31) - 1)
        sm = sfx + mask_fx
        probs = N.op_softmax_rows(sm.astype(np.int32)).astype(np.int64)
        p8, _ = site_fx("p", probs, 0, 127)
        cacc = N.op_gemm(p8.astype(np.int8), vs[kvh].astype(np.int8), False).astype(np.int64)
        mult_c = max(1, int(round(steps["p"] * steps["v"] / steps["ctx"])))
        c8, _ = site_acc("ctx", cacc, mult_c, steps["ctx"])
        acc = N.op_gemm(c8.astype(np.int8), wcounts(conf.o_slice(qh), wo_s[qh]), False).astype(np.int64)
        attn_sum += deq_fx(acc, steps["ctx"] * wo_s[qh])

    x1 = np.clip(x + attn_sum, -(1 << 31), (1 << 31) - 1)
    h2 = N.op_rmsnorm(x1.astype(np.int32),
                      np.rint(conf.weights["post_attention_layernorm.weight"] * ONE).astype(np.int64), 1)
    h2_8, _ = site_fx("h2", h2.astype(np.int64))
    gate = deq_fx(N.op_gemm(h2_8.astype(np.int8), wcounts(conf.gate_slice(), wsteps["wg"]), False).astype(np.int64),
                  steps["h2"] * wsteps["wg"])
    upv = deq_fx(N.op_gemm(h2_8.astype(np.int8), wcounts(conf.up_slice(), wsteps["wu"]), False).astype(np.int64),
                 steps["h2"] * wsteps["wu"])
    hm = N.op_mul(N.op_silu(gate.astype(np.int32)), upv.astype(np.int32)).astype(np.int64)
    hm8, _ = site_fx("hm", hm)
    down = deq_fx(N.op_gemm(hm8.astype(np.int8), wcounts(conf.down_slice(), wsteps["wd"]), False).astype(np.int64),
                  steps["hm"] * wsteps["wd"])
    return np.clip(x1 + down, -(1 << 31), (1 << 31) - 1)


def main() -> None:
    import json
    conf = Conformer(0)
    cal = load_token_cases("qwen3_f1_calibration")
    sam = {}
    for case in cal:
        hidden = conf.embedding(case["token_ids"])
        _, _, raw = instrumented_forward(conf, hidden, collect=True)
        for key, arr in raw.items():
            flat = np.abs(arr).reshape(-1)
            if flat.size > 200_000:
                flat = flat[:: flat.size // 200_000]
            sam.setdefault(key, []).append(flat)
    steps = {k: mse_optimal_scale(np.concatenate(v))[0] for k, v in sam.items()}
    steps["p"] = max(1, int(np.ceil(1.02 / 127 * ONE)))
    wsteps = {}
    for name, w in (("wq", conf.q_slice(0)), ("wk", conf.k_slice(0)), ("wv", conf.v_slice(0)),
                    ("wo", conf.o_slice(0)), ("wg", conf.gate_slice()), ("wu", conf.up_slice()),
                    ("wd", conf.down_slice())):
        wsteps[name] = weight_scale(w)
    print("site steps :", {k: round(v / ONE, 6) for k, v in steps.items()})
    print("weight steps:", {k: round(v / ONE, 6) for k, v in wsteps.items()})

    cases = load_token_cases("qwen3_f1_eval")[:3]
    refs = [conf.embedding(c["token_ids"]) for c in cases]
    ref_out = [instrumented_forward(conf, r)[0] for r in refs]

    # --- validate the mirror against the real converted graph -----------
    from convert_qwen3_block import build_converted_graph, to_fx
    scales = {"chosen": {k: {"step_fx": steps[k], "percentile": 100.0, "mse": 0.0,
                             "clipped_fraction": 0.0, "calibration_max_abs": 0.0} for k in sam},
              "p": 1.0}
    snake, go, plan, tdata, gid = build_converted_graph(conf, 16, scales)
    hidden0 = refs[0]
    inputs = dict(tdata)
    inputs[0] = to_fx(hidden0).astype(np.int32)
    sc = json.loads(json.dumps(snake, default=lambda b: b.hex()))
    sc["inputs"][0]["root"] = N.tensor_root_np(snake["inputs"][0]["desc"], inputs[0])
    for gi in sc["inputs"][1:]:
        gi["root"] = bytes.fromhex(gi["root"])
    out = N.execute_graph_np(sc, inputs, rope_tables={2: tdata[2]})
    graph_y = out["tensors"][(1, snake["outputs"][0]["index"])].astype(np.int64)
    mirror_y = block(conf, hidden0, steps, wsteps, set())
    delta = int(np.max(np.abs(graph_y - mirror_y)))
    print(f"mirror vs graph: max |delta| = {delta} fx units")
    assert delta == 0, "mirror does not reproduce the converted graph"

    def evaluate(off):
        worst = (1.0, 0.0)
        for hidden, ref in zip(refs, ref_out):
            y = block(conf, hidden, steps, wsteps, off) / ONE
            cos = float(np.dot(y.reshape(-1), ref.reshape(-1)) /
                        (np.linalg.norm(y) * np.linalg.norm(ref)))
            worst = (min(worst[0], cos), max(worst[1], float(np.abs(y - ref).max())))
        return worst

    print("ALL ON   :", evaluate(set()))
    for site in ["h", "q", "k", "v", "ctx", "p", "h2", "hm"]:
        print(f"no {site:4s} :", evaluate({site}))
    for site in ["wq", "wk", "wv", "wo", "wg", "wu", "wd"]:
        print(f"no {site:4s} :", evaluate({site}))
    print("ALL OFF  :", evaluate(set(list(steps) + list(wsteps))))


if __name__ == "__main__":
    main()
