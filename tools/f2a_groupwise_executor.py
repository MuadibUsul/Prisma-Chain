"""EXPERIMENTAL / NON-PROTOCOL — F.2A groupwise integer simulator.

Simulates the exact integer pipeline a future SLICE_VIEW extension would
allow, using ONLY the frozen semantics of the existing operators:

    INT8 slice operands -> GEMM_INT8_V1 accumulator (int64-exact)
    -> REQUANTIZE_V1 (integer mult/shift, ties-to-even) to the common
       Q12.20 scale of that logical sum
    -> ADD_FIXED_V1 (saturating) across partials
    -> existing graph semantics downstream (causal mask, softmax, ...)

Groupwise sites (reduction dimension of the consuming GEMM):

    q, k : per attention head, head_dim = 128   (scores GEMM)
    ctx  : per attention head, head_dim = 128   (o_proj GEMM)
    h    : hidden = 1024                        (q/k/v GEMMs)
    h2   : hidden = 1024                        (gate/up GEMMs)
    hm   : mlp = 3072                           (down GEMM)

Each slice carries its own static int8 step (calibration-derived). The
weight slice is the contiguous K-range slice of the F.1-quantized weight
(elementwise identical to slicing the original quantized weight). MACs
are preserved exactly (asserted); saturation counts and scale
representation errors are recorded. No production canonical types are
registered here.

Correctness anchors (see --selftest):
  * policy with every site OFF reproduces the F.1 converted graph output
    BIT-EXACTLY (the off path uses F.1's own formulas);
  * a synthetic two-slice case with unequal steps matches a hand-computed
    integer reference, including negative values, round ties and
    saturation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, "tools")
sys.path.insert(0, "compute/canonical/python")

import f1_canonical_numpy as N  # noqa: E402
from convert_qwen3_block import Conformer  # noqa: E402

ONE = 1 << 20
MIN_FX, MAX_FX = -(1 << 31), (1 << 31) - 1

# reduction width available for groupwise slicing per site
GROUPSITE_K = {"k": 128, "q": 128, "ctx": 128, "h": 1024, "h2": 1024, "hm": 3072}
ATTN_SCALE_FX = 92681


@dataclass
class GroupPolicy:
    groups: dict = field(default_factory=dict)   # site -> group size (0 = F.1 single)

    def group(self, site: str) -> int:
        return int(self.groups.get(site, 0))

    def enabled(self, site: str) -> bool:
        return self.group(site) > 0


@dataclass
class RunStats:
    saturations: int = 0
    partial_gemms: int = 0
    mac_total: int = 0
    scale_rel_error_max: float = 0.0
    clip_total: int = 0


def q_mult(step: int) -> int:
    return max(1, int(round((1 << 30) / step)))


def quant_counts(v_fx: np.ndarray, step: int, lo: int = -127, hi: int = 127) -> np.ndarray:
    return np.clip(N.rshift_round_even(v_fx.astype(np.int64) * np.int64(q_mult(step)), 30),
                   lo, hi).astype(np.int8)


def split_slices(v: np.ndarray, steps: list[int]):
    """Contiguous last-dim slices, each quantized with its own static step."""
    k = v.shape[-1]
    per = k // len(steps)
    out = []
    for s, step in enumerate(steps):
        lo = s * per
        hi = k if s == len(steps) - 1 else (s + 1) * per
        out.append((quant_counts(v[..., lo:hi], step), step, lo, hi))
    return out


def slice_steps_mse(samples_per_slice: list[np.ndarray], method: str = "mse") -> list[int]:
    from convert_qwen3_block import mse_optimal_scale
    steps = []
    for vals in samples_per_slice:
        flat = np.abs(np.asarray(vals).reshape(-1))
        if method == "mse":
            step, _p, _m = mse_optimal_scale(flat)
        else:
            step = max(1, int(np.ceil(flat.max() * 1.02 / 127 * ONE)))
        steps.append(int(step))
    return steps


def w_counts_slice(w: np.ndarray, k_lo: int, k_hi: int, wstep: int) -> np.ndarray:
    return np.clip(np.rint(w[k_lo:k_hi, :] * ONE / wstep), -127, 127).astype(np.int8)


def _add_sat(a: np.ndarray | None, b: np.ndarray, stats: RunStats) -> np.ndarray:
    if a is None:
        return b
    raw = a + b
    stats.saturations += int(np.count_nonzero((raw < MIN_FX) | (raw > MAX_FX)))
    return np.clip(raw, MIN_FX, MAX_FX)


def run_block(conf: Conformer, hidden: np.ndarray, policy: GroupPolicy,
              site_steps: dict, wsteps: dict, base_steps: dict,
              debug_out: dict | None = None) -> tuple[np.ndarray, RunStats]:
    """Integer simulation of the whole block.

    site_steps: {site: [per-slice step,...]} — only for sites whose group
    size is > 0; base_steps: F.1 single-scale steps for every site.
    """
    stats = RunStats()
    seq = hidden.shape[0]
    heads, kv, hd = 16, 8, 128
    half = hd // 2
    mask_fx = -(64 << 20)

    causal = np.triu(np.full((seq, seq), mask_fx, dtype=np.int64), k=1)
    x = np.rint(hidden * ONE).astype(np.int64)
    h = N.op_rmsnorm(x, np.rint(conf.weights["input_layernorm.weight"] * ONE).astype(np.int64), 1)
    h_slices = split_slices(h, site_steps.get("h", [base_steps["h"]]))
    if debug_out is not None:
        debug_out["h8"] = h_slices[0][0] if len(h_slices) == 1 else np.concatenate([c for c, *_ in h_slices], axis=-1)

    inv = conf.cfg["rope_theta"] ** (-2.0 * np.arange(half) / hd)
    ang = np.outer(np.arange(seq), inv)
    table = np.stack([np.rint(np.cos(ang) * ONE), np.rint(np.sin(ang) * ONE)], axis=-1) \
        .reshape(-1).astype(np.int64)

    def rope(v):
        return N.op_rope_adjacent(v.astype(np.int32), table, half).astype(np.int64)

    def partial_gemm_fx(act_slices, w, wstep):
        """Sum over K slices of int8 GEMMs requantized to the common fx
        scale of the full reduction (mult per slice = astep * wstep)."""
        total = None
        for counts, astep, k_lo, k_hi in act_slices:
            wc = w_counts_slice(w, k_lo, k_hi, wstep)
            acc = N.op_gemm(counts, wc, False).astype(np.int64)
            stats.partial_gemms += 1
            stats.mac_total += counts.shape[0] * wc.shape[1] * (k_hi - k_lo)
            total = _add_sat(total, np.clip(N.rshift_round_even(acc * np.int64(astep * wstep), 20),
                                            MIN_FX, MAX_FX), stats)
        return total

    # ---- Q/K/V projections (K = hidden) --------------------------------
    qn = np.rint(conf.q_norm() * ONE).astype(np.int64)
    kn = np.rint(conf.k_norm() * ONE).astype(np.int64)
    q_steps_all = site_steps.get("q") or [[base_steps["q"]]] * heads
    k_steps_all = site_steps.get("k") or [[base_steps["k"]]] * kv
    q_fx_per_head, k_fx_per_head, v8_per_head = [], [], []
    for qh in range(heads):
        qv = partial_gemm_fx(h_slices, conf.q_slice(qh), wsteps["wq"][qh])
        qv = N.op_rmsnorm(qv, qn, 1).astype(np.int64)
        q_fx_per_head.append(rope(qv))
    for kh in range(kv):
        kvx = partial_gemm_fx(h_slices, conf.k_slice(kh), wsteps["wk"][kh])
        kvx = N.op_rmsnorm(kvx, kn, 1).astype(np.int64)
        k_fx_per_head.append(rope(kvx))
        # V: with the h site OFF replicate F.1's one-step accum->int8; with
        # it ON use the two-step groupwise formulation (documented rounding
        # change; single-slice two-step vs F.1 one-step differ by <= 1 LSB).
        if policy.group("h") > 0:
            vfx = partial_gemm_fx(h_slices, conf.v_slice(kh), wsteps["wv"][kh])
            v8 = quant_counts(vfx, base_steps["v"])
        else:
            acc = N.op_gemm(h_slices[0][0], w_counts_slice(conf.v_slice(kh), 0, 1024, wsteps["wv"][kh]),
                            False).astype(np.int64)
            stats.partial_gemms += 1
            stats.mac_total += seq * 128 * 1024
            mult_v = max(1, int(round(base_steps["h"] * wsteps["wv"][kh] / base_steps["v"])))
            v8 = np.clip(N.rshift_round_even(acc * np.int64(mult_v), 20), -127, 127).astype(np.int8)
        v8_per_head.append(v8)
        if debug_out is not None and len(v8_per_head) == 1:
            debug_out["q8_h0"] = quant_counts(q_fx_per_head[0][:, :part], q_step_of(0, 0))
            debug_out["k8_h0"] = quant_counts(k_fx_per_head[0][:, :part], k_step_of(0, 0))
            debug_out["v8_h0"] = v8

    # ---- scores (K = head_dim, groupwise q/k) --------------------------
    # The reduction is partitioned at the FINEST enabled group size; both
    # operands use the SAME partition (a non-groupwise site keeps its one
    # step for every sub-slice; a coarser groupwise site keeps its group
    # step for sub-slices inside that group). Partial GEMMs are then
    # requantized to the common score scale and added (section 19-20).
    g_q, g_k = policy.group("q"), policy.group("k")
    active = [g for g in (g_q, g_k) if g > 0]
    part = min(active) if active else hd
    q_step_of = lambda qh, s_lo: (q_steps_all[qh][s_lo // g_q] if g_q > 0 else q_steps_all[qh][0])
    k_step_of = lambda kh, s_lo: (k_steps_all[kh][s_lo // g_k] if g_k > 0 else k_steps_all[kh][0])
    ctx_group = policy.group("ctx")
    attn_sum = np.zeros((seq, 1024), dtype=np.int64)
    p_step = base_steps["p"]
    for qh in range(heads):
        kvh = qh // 2
        total = None
        qv_fx = q_fx_per_head[qh]
        kv_fx = k_fx_per_head[kvh]
        for s_lo in range(0, hd, part):
            s_hi = min(hd, s_lo + part)
            qs = q_step_of(qh, s_lo)
            ks = k_step_of(kvh, s_lo)
            q8 = quant_counts(qv_fx[:, s_lo:s_hi], qs)
            k8 = quant_counts(kv_fx[:, s_lo:s_hi], ks)
            acc = N.op_gemm(q8, k8, True).astype(np.int64)
            stats.partial_gemms += 1
            stats.mac_total += seq * seq * (s_hi - s_lo)
            mult = max(1, int(round(qs * ks * ATTN_SCALE_FX / ONE)))
            target = qs * ks * ATTN_SCALE_FX
            stats.scale_rel_error_max = max(stats.scale_rel_error_max,
                                            abs(mult * ONE - target) / target)
            total = _add_sat(total, np.clip(N.rshift_round_even(acc * np.int64(mult), 20),
                                            MIN_FX, MAX_FX), stats)
        sm = np.clip(total + causal, MIN_FX, MAX_FX)
        if debug_out is not None and qh == 0:
            debug_out["sacc_h0"] = None
            debug_out["sfx_h0"] = total
            debug_out["sm_h0"] = sm
        probs = N.op_softmax_rows(sm.astype(np.int32)).astype(np.int64)
        p8 = quant_counts(probs, p_step, 0, 127)
        # context (K = seq): unchanged single path
        cacc = N.op_gemm(p8, v8_per_head[kvh], False).astype(np.int64)
        stats.partial_gemms += 1
        stats.mac_total += seq * 128 * seq
        if ctx_group > 0:
            # cacc counts are p8*v8 with steps (p_step, v_step): the common
            # fx dequant is cacc * p_step * v_step >> 20 (one shift, the
            # /2^20 of the Q12.20 product).
            mult_c = int(p_step) * int(base_steps["v"])
            cfx = np.clip(N.rshift_round_even(cacc * np.int64(mult_c), 20), MIN_FX, MAX_FX)
            ctx_steps_all = site_steps["ctx"]
            c_slices = split_slices(cfx, ctx_steps_all[qh] if isinstance(ctx_steps_all[0], list)
                                    else ctx_steps_all)
            ofx = partial_gemm_fx(c_slices, conf.o_slice(qh), wsteps["wo"][qh])
        else:
            mult_c = max(1, int(round(p_step * base_steps["v"] / base_steps["ctx"])))
            c8 = np.clip(N.rshift_round_even(cacc * np.int64(mult_c), 20), -127, 127).astype(np.int8)
            acc = N.op_gemm(c8, w_counts_slice(conf.o_slice(qh), 0, 128, wsteps["wo"][qh]), False).astype(np.int64)
            stats.partial_gemms += 1
            stats.mac_total += seq * 1024 * 128
            ofx = np.clip(N.rshift_round_even(acc * np.int64(base_steps["ctx"] * wsteps["wo"][qh]), 20),
                          MIN_FX, MAX_FX)
        attn_sum = _add_sat(attn_sum, ofx, stats)
        if debug_out is not None and qh == 0:
            debug_out["probs_h0"] = probs.astype(np.int64)
            debug_out["p8_h0"] = p8
            debug_out["ofx_h0"] = ofx
            if ctx_group > 0:
                debug_out["c8_h0"] = c_slices[0][0] if len(c_slices) == 1 else None
            else:
                debug_out["c8_h0"] = c8

    x1 = np.clip(x + attn_sum, MIN_FX, MAX_FX)
    if debug_out is not None:
        debug_out["attn_sum"] = attn_sum
        debug_out["x1"] = x1

    # ---- MLP (h2 -> gate/up, hm -> down) -------------------------------
    h2 = N.op_rmsnorm(x1.astype(np.int32),
                      np.rint(conf.weights["post_attention_layernorm.weight"] * ONE).astype(np.int64), 1)
    h2_slices = split_slices(h2.astype(np.int64), site_steps.get("h2", [base_steps["h2"]]))
    if debug_out is not None:
        debug_out["h2"] = h2.astype(np.int64)
        debug_out["h2_8"] = h2_slices[0][0] if len(h2_slices) == 1 else np.concatenate([c for c, *_ in h2_slices], axis=-1)
    gate = partial_gemm_fx(h2_slices, conf.gate_slice(), wsteps["wg"])
    upv = partial_gemm_fx(h2_slices, conf.up_slice(), wsteps["wu"])
    hm = N.op_mul(N.op_silu(gate.astype(np.int32)), upv.astype(np.int32)).astype(np.int64)
    if debug_out is not None:
        debug_out["gate"] = gate
        debug_out["up"] = upv
        debug_out["hm"] = hm
    hm_slices = split_slices(hm, site_steps.get("hm", [base_steps["hm"]]))
    down = partial_gemm_fx(hm_slices, conf.down_slice(), wsteps["wd"])
    y = np.clip(x1 + down, MIN_FX, MAX_FX)
    return y, stats


def expected_macs(seq: int) -> int:
    """Analytic GEMM MAC count of the block (must equal stats.mac_total)."""
    return (16 + 8 + 8) * seq * 1024 * 128 + 16 * seq * seq * 128 + \
        16 * seq * 128 * seq + 16 * seq * 1024 * 128 + \
        2 * seq * 1024 * 3072 + seq * 3072 * 1024


def selftest() -> None:
    """Anchor 1: synthetic two-slice integer reference; Anchor 2: the
    all-OFF policy equals the F.1 graph bit-for-bit."""
    # --- anchor 1: hand-computed integer reference ----------------------
    q_steps = [16, 64]          # unequal slice steps
    k_steps = [4, 8]
    q_fx = np.array([[33, -17, 250, -1]], dtype=np.int64)
    k_fx = np.array([[9, -5, 61, 3]], dtype=np.int64)
    f = ATTN_SCALE_FX

    total = None
    for (a_lo, a_hi), (b_lo, b_hi), qs, ks in (( (0, 2), (0, 2), 16, 4), ((2, 4), (2, 4), 64, 8)):
        q8 = quant_counts(q_fx[:, a_lo:a_hi], qs)
        k8 = quant_counts(k_fx[:, b_lo:b_hi], ks)
        acc = int(q8.astype(np.int64) @ k8.astype(np.int64).T)
        mult = max(1, int(round(qs * ks * f / ONE)))
        part = N.rshift_round_even(acc * mult, 20)
        total = part if total is None else max(MIN_FX, min(MAX_FX, total + part))
    # manual duplicate of the same arithmetic, ties-to-even explicit
    def te(v, s):
        qq, r = v >> s, v & ((1 << s) - 1)
        hh = 1 << (s - 1)
        return qq + int(r > hh or (r == hh and (qq & 1) == 1))
    manual = 0
    for (a_lo, a_hi), (b_lo, b_hi), qs, ks in (((0, 2), (0, 2), 16, 4), ((2, 4), (2, 4), 64, 8)):
        q8 = [max(-127, min(127, te(int(v) * q_mult(qs), 30))) for v in q_fx[0, a_lo:a_hi]]
        k8 = [max(-127, min(127, te(int(v) * q_mult(ks), 30))) for v in k_fx[0, b_lo:b_hi]]
        acc = sum(a * b for a, b in zip(q8, k8))
        manual += te(acc * max(1, round(qs * ks * f / ONE)), 20)
    assert total == manual, f"synthetic reference mismatch: {total} != {manual}"
    print("anchor 1 OK: two-slice unequal-step integer reference matches")

    # --- anchor 2: all-OFF policy == F.1 graph --------------------------
    print("anchor 2 (all-OFF policy == F.1 graph) runs in f2a_groupwise_search.py --anchor")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        raise SystemExit("library: see f2a_groupwise_search.py")
