"""QWEN3_BLOCK_PROFILE_V1 converter: pinned Qwen3-0.6B-Base layer 0 into
CANONICAL_GRAPH_V1 with static W8A8 quantization.

    python tools/convert_qwen3_block.py --check-references   # A vs B first
    python tools/convert_qwen3_block.py --convert            # build artifacts
    python tools/convert_qwen3_block.py --gate               # accuracy gate

Design decisions (all documented in docs/transformer-block-v1.md and the
quantization plan):

* Qwen3 attention is Q/K-normalized: q_norm/k_norm are RMSNORM nodes over
  the head dimension, applied before RoPE — expressed with the EXISTING
  operators, no new operator.
* GQA (16 Q heads / 8 KV heads) is static wiring: each K/V chain node is
  referenced by exactly two Q-head branches.
* HF rotate-half RoPE is mapped onto canonical ADJACENT-pair rotation by
  the permutation perm[2j]=j, perm[2j+1]=d/2+j applied to the Q/K weight
  slices and the q/k norm weights. Scores are permutation-invariant, so
  the permutation is exact.
* The causal mask is a committed constant tensor added with ADD_FIXED_V1;
  masked entries use MASK_FX = -(64 << 20), far below the canonical
  exp underflow threshold.
* The attention scale is folded into the scores REQUANTIZE parameters
  (AttentionScaleFx, integer only).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import canonical_ref as R  # noqa: E402
import f1_canonical_numpy as N  # noqa: E402
from qwen3_reference import (  # noqa: E402
    _load_all, load_layer_weights, load_qwen3_block_config, qwen_rope_permutation,
)

import os as _os
MODEL_DIR = Path(_os.environ.get("PRISMA_MODEL_DIR", str(REPO / "models" / "qwen3-0.6b-base")))
OUT_DIR = REPO / "models" / "qwen3-0.6b-base-canonical" / "layer0"
TESTDATA = REPO / "testdata"

FRAC_BITS = 20
MASK_FX = -(64 << FRAC_BITS)
CAL_MARGIN = 1.02
EPS_FX = max(1, round(1e-6 * (1 << 20)))

# Link-level tuning discovered by the sensitivity experiment
# (tools/f1_attribution.py + the per-head scan): the K-site MSE optimum
# under-resolves the many small-magnitude channels because the k_norm
# weight tail (max ~96) dominates the step; a finer step measured better
# end to end at 0.25x. Calibration-set only; the eval set is untouched.
K_HEAD_FACTOR = 0.25


def to_fx(x: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=np.float64) * (1 << 20)).astype(np.int64)


def calib_scale(max_abs_fx: float | int) -> int:
    """int8 step in fx units from a calibration maximum (with margin)."""
    return max(1, int(math.ceil(float(max_abs_fx) * CAL_MARGIN / 127.0)))


def mse_optimal_scale(samples: np.ndarray) -> tuple[int, float, float]:
    """Pick the int8 fx step minimizing MEASURED quantization MSE over the
    calibration distribution (clamping at +-127 counts included).

    Returns (step_fx, chosen_percentile, mse_real). Uses ONLY calibration
    samples; the evaluation set is never consulted.
    """
    absvals = np.abs(samples.astype(np.float64))
    best = None
    for pct in (90.0, 95.0, 97.5, 99.0, 99.5, 99.8, 99.9, 99.95, 99.99, 100.0):
        edge = float(np.percentile(absvals, pct)) * CAL_MARGIN
        step_real = max(edge / 127.0, 1e-12)
        step_fx = max(1, int(math.ceil(step_real * (1 << 20))))
        step_real = step_fx / (1 << 20)
        counts = np.clip(np.rint(absvals / step_real), 0, 127)
        err = counts * step_real - absvals
        mse = float(np.mean(err * err))
        if best is None or mse < best[2]:
            best = (step_fx, pct, mse)
    return best


def quantize_int8(w: np.ndarray, qw: int) -> np.ndarray:
    """int8 = round(w_fx / qw) with w_fx = w * 2^20 (qw is an fx step)."""
    q = np.rint(w.astype(np.float64) * (1 << 20) / qw).astype(np.int64)
    return np.clip(q, -127, 127).astype(np.int8)


def weight_scale(w: np.ndarray) -> int:
    m = float(np.max(np.abs(w)))
    return max(1, int(math.ceil(m * (1 << 20) / 127.0)))


class Conformer:
    """Deterministic source of the block's canonical input/constant tensors."""

    def __init__(self, layer: int = 0):
        self.cfg = load_qwen3_block_config()
        self.layer = layer
        self.weights = load_layer_weights(layer)
        self.perm = qwen_rope_permutation(self.cfg["head_dim"])

    def embedding(self, token_ids) -> np.ndarray:
        import torch
        embed = _load_all()["model.embed_tokens.weight"].to(torch.float64).numpy()
        return embed[np.asarray(token_ids)].astype(np.float64)

    def rope_table(self, seq: int) -> np.ndarray:
        head_dim = self.cfg["head_dim"]
        half = head_dim // 2
        inv_freq = self.cfg["rope_theta"] ** (-2.0 * np.arange(half) / head_dim)
        angles = np.outer(np.arange(seq), inv_freq)
        cos = np.rint(np.cos(angles) * (1 << 20)).astype(np.int64)
        sin = np.rint(np.sin(angles) * (1 << 20)).astype(np.int64)
        table = np.stack([cos, sin], axis=-1).reshape(-1)  # [pos][pair](cos,sin)
        return table.astype(np.int64)

    def mask(self, seq: int) -> np.ndarray:
        m = np.zeros((seq, seq), dtype=np.int64)
        m[np.triu_indices(seq, k=1)] = MASK_FX
        return m

    # --- per-head weight slices -------------------------------------------

    def q_slice(self, h: int):
        # Head coords are the ROWS of the [head_dim, hidden] slice; the
        # rotate-half -> adjacent-pair permutation applies to those rows.
        hd = self.cfg["head_dim"]
        w = self.weights["self_attn.q_proj.weight"][h * hd:(h + 1) * hd, :]
        return w[self.perm, :].T  # [hidden, head_dim]

    def k_slice(self, h: int):
        hd = self.cfg["head_dim"]
        w = self.weights["self_attn.k_proj.weight"][h * hd:(h + 1) * hd, :]
        return w[self.perm, :].T

    def v_slice(self, h: int):
        hd = self.cfg["head_dim"]
        return self.weights["self_attn.v_proj.weight"][h * hd:(h + 1) * hd, :].T

    def o_slice(self, h: int):
        hd = self.cfg["head_dim"]
        w = self.weights["self_attn.o_proj.weight"][:, h * hd:(h + 1) * hd]  # [d, hd]
        return w.T  # [hd, d]

    def q_norm(self):
        return np.asarray(self.weights["self_attn.q_norm.weight"])[self.perm]

    def k_norm(self):
        return np.asarray(self.weights["self_attn.k_norm.weight"])[self.perm]

    def gate_slice(self):
        return self.weights["mlp.gate_proj.weight"].T

    def up_slice(self):
        return self.weights["mlp.up_proj.weight"].T

    def down_slice(self):
        return self.weights["mlp.down_proj.weight"].T


# --- instrumented float64 forward (calibration) ----------------------------


def instrumented_forward(conf: Conformer, hidden: np.ndarray, collect: bool = False):
    """Reference B order with site maxima recorded (float64)."""
    cfg = conf.cfg
    seq = hidden.shape[0]
    heads, kv_heads, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = cfg["rms_norm_eps"]

    def rms_norm(x, w):
        var = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float64)
        return (x / np.sqrt(var + eps)) * w

    def rope_adjacent(x):
        # Canonical ROPE on the permuted head layout: adjacent pairs
        # (y_2j, y_2j+1) hold HF's (x_j, x_j+d/2) with the same frequency.
        half = hd // 2
        inv = cfg["rope_theta"] ** (-2.0 * np.arange(half) / hd)
        ang = np.outer(np.arange(seq), inv)[:, None, :]
        c, s = np.cos(ang), np.sin(ang)
        even, odd = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = even * c - odd * s
        out[..., 1::2] = even * s + odd * c
        return out

    maxima = {}
    raw = {}
    def record(name, val):
        m = float(np.max(np.abs(val)))
        maxima[name] = max(maxima.get(name, 0.0), m)
        if collect:
            raw[name] = np.asarray(val, dtype=np.float64)

    h = rms_norm(hidden, conf.weights["input_layernorm.weight"].astype(np.float64))
    record("h", h)

    q = np.stack([h @ conf.q_slice(i) for i in range(heads)], axis=1)      # [seq,heads,hd]
    q = q / np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + eps) * conf.q_norm()
    q = rope_adjacent(q)
    record("q", q)

    k = np.stack([h @ conf.k_slice(i) for i in range(kv_heads)], axis=1)
    k = k / np.sqrt(np.mean(k * k, axis=-1, keepdims=True) + eps) * conf.k_norm()
    k = rope_adjacent(k)
    record("k", k)

    v = np.stack([h @ conf.v_slice(i) for i in range(kv_heads)], axis=1)
    record("v", v)

    scale = 1.0 / math.sqrt(hd)
    causal = np.triu(np.full((seq, seq), -np.inf), k=1)
    ctx = np.empty_like(q)
    group = heads // kv_heads
    for qh in range(heads):
        kvh = qh // group
        scores = (q[:, qh, :] @ k[:, kvh, :].T) * scale + causal
        mx = scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores - mx)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        ctx[:, qh, :] = probs @ v[:, kvh, :]
    record("ctx", ctx)

    attn = sum(ctx[:, i, :] @ conf.o_slice(i) for i in range(heads))
    x1 = hidden + attn

    h2 = rms_norm(x1, conf.weights["post_attention_layernorm.weight"].astype(np.float64))
    record("h2", h2)
    gate = h2 @ conf.gate_slice()
    up = h2 @ conf.up_slice()
    hm = (gate / (1.0 + np.exp(-gate))) * up
    record("hm", hm)
    y = x1 + hm @ conf.down_slice()
    if collect:
        return y, maxima, raw
    return y, maxima


def calibrate(conf: Conformer, cases, collect_samples: bool = False):
    """Record site maxima (and, optionally, subsampled |values| for the
    MSE-optimal per-site int8 step selection)."""
    maxima = {}
    samples = {}
    for case in cases:
        hidden = conf.embedding(case["token_ids"])
        _, m, raw = instrumented_forward(conf, hidden, collect=True)
        for key, value in m.items():
            maxima[key] = max(maxima.get(key, 0.0), value)
        if collect_samples:
            budget = 400_000
            for key, arr in raw.items():
                flat = np.abs(arr).reshape(-1)
                if flat.size > budget:
                    stride = max(1, flat.size // budget)
                    flat = flat[::stride]
                samples.setdefault(key, []).append(flat)
            # per-head samples for the head-indexed sites
            for key in ("q", "k", "v", "ctx"):
                arr = raw.get(key)
                if arr is None:
                    continue
                # [seq, heads, head_dim] -> per head
                for h in range(arr.shape[1]):
                    flat = np.abs(arr[:, h, :]).reshape(-1)
                    samples.setdefault(f"{key}_h{h}", []).append(flat)
                    hk = f"{key}_h{h}"
                    maxima[hk] = max(maxima.get(hk, 0.0), float(flat.max()))
                    # per-channel maxima (for operand balancing)
                    per_ch = np.abs(arr[:, h, :]).max(axis=0)
                    samples.setdefault(f"chan_{key}_h{h}", []).append(per_ch)
    if collect_samples:
        samples = {k: np.concatenate(v) for k, v in samples.items()}
        return maxima, samples
    return maxima


# --- graph construction -----------------------------------------------------


def build_converted_graph(conf: Conformer, seq: int, scales: dict):
    """Returns (snake descriptor, go descriptor, quant plan, input tensors)."""
    cfg = conf.cfg
    d, hd = cfg["hidden_size"], cfg["head_dim"]
    heads, kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    group = heads // kv_heads
    mlp = cfg["intermediate_size"]
    attn_scale = R.attention_scale_fx(hd)

    inputs = []          # (name, desc snake, np data)
    go_inputs = []
    tensor_data = {}

    def add_input(name, desc, data, dtype_fx=True):
        if dtype_fx:
            desc = {"dtype": R.DTYPE_Q12_20, "layout": 1, "shape": list(data.shape)}
        else:
            desc = {"dtype": R.DTYPE_INT8, "layout": 1, "shape": list(data.shape)}
        data = np.asarray(data, dtype=np.int32)
        root = N.tensor_root_np(desc, data)
        idx = len(inputs)
        inputs.append({"name": name, "desc": desc, "root": root})
        go_inputs.append({"Name": name, "Desc": {"Dtype": desc["dtype"], "Layout": 1,
                                                 "Shape": desc["shape"]}, "Root": root})
        tensor_data[idx] = data
        return idx

    mask = conf.mask(seq).astype(np.int32)
    table = conf.rope_table(seq).astype(np.int32)
    x_idx = add_input("x", None, np.zeros((seq, d), dtype=np.int32))  # placeholder, replaced later
    mask_idx = add_input("causal_mask", None, mask)
    table_idx = add_input("rope_table", None, table)

    quant_plan = {"method": "static W8A8: per-site int8 activation steps chosen by "
                            "minimizing measured quantization MSE over the calibration "
                            "distribution (quantile grid, clamping included); per-slice "
                            "int8 weight scales from the checkpoint slice maximum",
                  "sites": {}, "nodes": {}}

    chosen = scales.get("chosen")
    if chosen:
        qa = {key: chosen[key]["step_fx"] for key in chosen}
        quant_plan["sites"] = {
            key: {"step_fx": chosen[key]["step_fx"],
                  "percentile": chosen[key]["percentile"],
                  "mse": chosen[key]["mse"],
                  "clipped_fraction": chosen[key]["clipped_fraction"],
                  "calibration_max_abs": chosen[key]["calibration_max_abs"]}
            for key in chosen
        }
    else:
        qa = {key: calib_scale(value * (1 << 20)) for key, value in scales.items()}
        quant_plan["sites"] = {key: {"calibration_max_abs": scales[key],
                                     "step_fx": qa[key]} for key in scales if key != "p"}
    # softmax probabilities are bounded by one: fixed precision scale
    qa["p"] = max(1, int(math.ceil(CAL_MARGIN / 127.0 * (1 << 20))))
    quant_plan["sites"]["p"] = {"step_fx": qa["p"], "calibration_max_abs": 1.0}

    # weight inputs
    wq_idx, wk_idx, wv_idx, wo_idx = [], [], [], []
    qw_q, qw_k, qw_v, qw_o = [], [], [], []
    for h in range(heads):
        w = conf.q_slice(h); qw = weight_scale(w)
        wq_idx.append(add_input(f"wq_h{h}", None, quantize_int8(w, qw).astype(np.int32), dtype_fx=False)); qw_q.append(qw)
        w = conf.o_slice(h); qw = weight_scale(w)
        wo_idx.append(add_input(f"wo_h{h}", None, quantize_int8(w, qw).astype(np.int32), dtype_fx=False)); qw_o.append(qw)
    for h in range(kv_heads):
        w = conf.k_slice(h); qw = weight_scale(w)
        wk_idx.append(add_input(f"wk_h{h}", None, quantize_int8(w, qw).astype(np.int32), dtype_fx=False)); qw_k.append(qw)
        w = conf.v_slice(h); qw = weight_scale(w)
        wv_idx.append(add_input(f"wv_h{h}", None, quantize_int8(w, qw).astype(np.int32), dtype_fx=False)); qw_v.append(qw)

    def norm_input(name, array):
        return add_input(name, None, to_fx(array).astype(np.int32))

    attn_norm_idx = norm_input("attn_norm_w", conf.weights["input_layernorm.weight"])
    mlp_norm_idx = norm_input("mlp_norm_w", conf.weights["post_attention_layernorm.weight"])
    qn_idx = norm_input("q_norm_w", conf.q_norm())
    kn_idx = norm_input("k_norm_w", conf.k_norm())
    wg_idx = add_input("w_gate", None, quantize_int8(conf.gate_slice(),
                                                     weight_scale(conf.gate_slice())).astype(np.int32), dtype_fx=False)
    qw_gate = weight_scale(conf.gate_slice())
    wu_idx = add_input("w_up", None, quantize_int8(conf.up_slice(),
                                                   weight_scale(conf.up_slice())).astype(np.int32), dtype_fx=False)
    qw_up = weight_scale(conf.up_slice())
    wd_idx = add_input("w_down", None, quantize_int8(conf.down_slice(),
                                                     weight_scale(conf.down_slice())).astype(np.int32), dtype_fx=False)
    qw_down = weight_scale(conf.down_slice())

    nodes = []
    site_nodes = {}
    def req_node(ref, params, desc, site):
        ref_out = node(R.OP_REQUANTIZE, [ref], params, desc)
        site_nodes.setdefault(site, []).append(ref_out["Index"])
        return ref_out

    def node(op, refs, params, desc):
        nodes.append({"node_id": len(nodes), "operator_id": op,
                      "operator_version": R.VERSION_GEMM if op == R.OP_GEMM else R.VERSION_FXFUSION,
                      "inputs": refs, "output": desc, "params": params})
        return {"Kind": 1, "Index": len(nodes) - 1}

    def inref(i):
        return {"Kind": 0, "Index": i}

    def req_params(mult, shift, lo, hi, out_int8=False):
        params = [["clamp_hi", hi], ["clamp_lo", lo], ["mult", mult], ["shift", shift]]
        if out_int8:
            params.append(["out_dtype", 1])
        return sorted(params)

    accum_fx = lambda qa_a, qw: (qa_a * qw, 20)
    accum_i8 = lambda qa_a, qw, qout: (max(1, int(round(qa_a * qw / qout))), 20)
    fx_i8 = lambda qout: (int(round((1 << 30) / qout)), 30)

    desc_i8 = lambda shape: {"dtype": R.DTYPE_INT8, "layout": 1, "shape": list(shape)}
    desc_fx = lambda shape: {"dtype": R.DTYPE_Q12_20, "layout": 1, "shape": list(shape)}
    desc_acc = lambda shape: {"dtype": R.DTYPE_INT32_ACCUM, "layout": 1, "shape": list(shape)}

    # operand balancing: per-channel factors t_j applied to Q and 1/t_j to
    # K (exact: (q t)(k / t) = q k), shrinking both int8 ranges. The
    # exponent gamma comes from the calibration data only.
    balance = scales.get("balance", {})
    gamma = balance.get("gamma", 0.0)

    def balance_mul(ref, factors, name):
        const = np.rint(np.asarray(factors, dtype=np.float64)[None, :]
                        * np.ones((seq, 1)) * (1 << 20)).astype(np.int32)
        idx = add_input(name, None, const)
        return node(R.OP_MUL, [ref, inref(idx)], [], desc_fx((seq, hd)))

    # --- attention ---
    h = node(R.OP_RMSNORM, [inref(x_idx), inref(attn_norm_idx)], [["eps_fx", EPS_FX]],
             desc_fx((seq, d)))
    h8 = req_node(h, req_params(*fx_i8(qa["h"]), -127, 127, out_int8=True), desc_i8((seq, d)), "h")

    q_chains = []
    for qh in range(heads):
        acc = node(R.OP_GEMM, [h8, inref(wq_idx[qh])], [], desc_acc((seq, hd)))
        qfx = req_node(acc, req_params(*accum_fx(qa["h"], qw_q[qh]), R.MIN_FX, R.MAX_FX),
                       desc_fx((seq, hd)), "q_pre_norm")
        qn = node(R.OP_RMSNORM, [qfx, inref(qn_idx)], [["eps_fx", EPS_FX]], desc_fx((seq, hd)))
        qr = node(R.OP_ROPE, [qn, inref(table_idx)],
                  sorted([["half_dim", hd // 2], ["max_pos", seq]]), desc_fx((seq, hd)))
        qb = qr
        if gamma > 0 and balance:
            t = np.asarray(balance["t_h" + str(qh)], dtype=np.float64)
            qb = balance_mul(qr, t, f"qbal_h{qh}")
        q8 = req_node(qb, req_params(*fx_i8(qa.get(f"q_h{qh}", qa["q"])), -127, 127, out_int8=True),
                      desc_i8((seq, hd)), "q")
        q_chains.append(q8)

    k_chains, v_chains = [], []
    for kh in range(kv_heads):
        acc = node(R.OP_GEMM, [h8, inref(wk_idx[kh])], [], desc_acc((seq, hd)))
        kfx = req_node(acc, req_params(*accum_fx(qa["h"], qw_k[kh]), R.MIN_FX, R.MAX_FX),
                       desc_fx((seq, hd)), "k_pre_norm")
        kn = node(R.OP_RMSNORM, [kfx, inref(kn_idx)], [["eps_fx", EPS_FX]], desc_fx((seq, hd)))
        kr = node(R.OP_ROPE, [kn, inref(table_idx)],
                  sorted([["half_dim", hd // 2], ["max_pos", seq]]), desc_fx((seq, hd)))
        kb = kr
        if gamma > 0 and balance:
            t = np.asarray(balance["t_h" + str(kh)], dtype=np.float64)
            kb = balance_mul(kr, 1.0 / t, f"kbal_h{kh}")
        k8 = req_node(kb, req_params(*fx_i8(qa.get(f"k_h{kh}", qa["k"])), -127, 127, out_int8=True),
                      desc_i8((seq, hd)), "k")
        k_chains.append(k8)

        vacc = node(R.OP_GEMM, [h8, inref(wv_idx[kh])], [], desc_acc((seq, hd)))
        v8 = req_node(vacc, req_params(*accum_i8(qa["h"], qw_v[kh], qa.get(f"v_h{kh}", qa["v"])),
                                       -127, 127, out_int8=True), desc_i8((seq, hd)), "v")
        v_chains.append(v8)

    head_outs = []
    for qh in range(heads):
        kvh = qh // group
        sacc = node(R.OP_GEMM, [q_chains[qh], k_chains[kvh]], [["transpose_b", 1]], desc_acc((seq, seq)))
        mult_scores = max(1, int(round(qa["q"] * qa["k"] * attn_scale / (1 << 20))))
        sfx = req_node(sacc, req_params(mult_scores, 20, R.MIN_FX, R.MAX_FX),
                       desc_fx((seq, seq)), "scores")
        masked = node(R.OP_ADD, [sfx, inref(mask_idx)], [], desc_fx((seq, seq)))
        probs = node(R.OP_SOFTMAX, [masked], [], desc_fx((seq, seq)))
        p8 = req_node(probs, req_params(*fx_i8(qa["p"]), 0, 127, out_int8=True), desc_i8((seq, seq)), "p")
        cacc = node(R.OP_GEMM, [p8, v_chains[kvh]], [], desc_acc((seq, hd)))
        c8 = req_node(cacc, req_params(*accum_i8(qa["p"], qa.get(f"v_h{kvh}", qa["v"]),
                                                 qa.get(f"ctx_h{qh}", qa["ctx"])),
                                       -127, 127, out_int8=True), desc_i8((seq, hd)), "ctx")
        oacc = node(R.OP_GEMM, [c8, inref(wo_idx[qh])], [], desc_acc((seq, d)))
        ofx = req_node(oacc, req_params(*accum_fx(qa.get(f"ctx_h{qh}", qa["ctx"]), qw_o[qh]),
                                        R.MIN_FX, R.MAX_FX), desc_fx((seq, d)), "o_proj")
        head_outs.append(ofx)

    head_sum = head_outs[0]
    for qh in range(1, heads):
        head_sum = node(R.OP_ADD, [head_sum, head_outs[qh]], [], desc_fx((seq, d)))
    x1 = node(R.OP_ADD, [inref(x_idx), head_sum], [], desc_fx((seq, d)))

    h2 = node(R.OP_RMSNORM, [x1, inref(mlp_norm_idx)], [["eps_fx", EPS_FX]], desc_fx((seq, d)))
    h2_8 = req_node(h2, req_params(*fx_i8(qa["h2"]), -127, 127, out_int8=True), desc_i8((seq, d)), "h2")
    gacc = node(R.OP_GEMM, [h2_8, inref(wg_idx)], [], desc_acc((seq, mlp)))
    gfx = req_node(gacc, req_params(*accum_fx(qa["h2"], qw_gate), R.MIN_FX, R.MAX_FX),
                   desc_fx((seq, mlp)), "gate")
    gs = node(R.OP_SILU, [gfx], [], desc_fx((seq, mlp)))
    uacc = node(R.OP_GEMM, [h2_8, inref(wu_idx)], [], desc_acc((seq, mlp)))
    ufx = req_node(uacc, req_params(*accum_fx(qa["h2"], qw_up), R.MIN_FX, R.MAX_FX),
                   desc_fx((seq, mlp)), "up")
    hm = node(R.OP_MUL, [gs, ufx], [], desc_fx((seq, mlp)))
    hm8 = req_node(hm, req_params(*fx_i8(qa["hm"]), -127, 127, out_int8=True), desc_i8((seq, mlp)), "hm")
    dacc = node(R.OP_GEMM, [hm8, inref(wd_idx)], [], desc_acc((seq, d)))
    dfx = req_node(dacc, req_params(*accum_fx(qa["hm"], qw_down), R.MIN_FX, R.MAX_FX),
                   desc_fx((seq, d)), "down")
    y = node(R.OP_ADD, [x1, dfx], [], desc_fx((seq, d)))

    quant_plan["site_nodes"] = {k: sorted(v) for k, v in site_nodes.items()}

    # per-node REQUANTIZE parameters into the quant plan (spec section 27)
    for n in nodes:
        if n["operator_id"] == R.OP_REQUANTIZE:
            quant_plan["nodes"][str(n["node_id"])] = {"operator": n["operator_id"],
                                                      "params": dict(n["params"])}

    snake_nodes = []
    for n in nodes:
        snake_nodes.append({
            "node_id": n["node_id"], "operator_id": n["operator_id"],
            "operator_version": n["operator_version"],
            "inputs": [{"kind": r["Kind"], "index": r["Index"]} for r in n["inputs"]],
            "output": n["output"], "params": n["params"],
        })
    snake = {
        "protocol_version": R.GRAPH_PROTOCOL_VERSION, "spec": "QWEN3_BLOCK_PROFILE_V1",
        "inputs": inputs, "nodes": snake_nodes,
        "outputs": [{"kind": 1, "index": y["Index"]}],
    }
    go = {
        "ProtocolVersion": R.GRAPH_PROTOCOL_VERSION, "Spec": "QWEN3_BLOCK_PROFILE_V1",
        "Inputs": go_inputs,
        "Nodes": [{
            "NodeID": n["node_id"], "OperatorID": n["operator_id"], "Version": n["operator_version"],
            "Inputs": n["inputs"],
            "Output": {"Dtype": n["output"]["dtype"], "Layout": 1,
                       "Shape": list(n["output"]["shape"])},
            "Params": [{"Key": k, "Value": v} for k, v in n["params"]],
        } for n in nodes],
        "Outputs": [{"Kind": 1, "Index": y["Index"]}],
    }
    graph_id = R.graph_id(snake)
    return snake, go, quant_plan, tensor_data, graph_id


def balanced_steps(samples, gamma: float, q_extra: float = 1.0, k_extra: float = 0.25):
    """Per-head q/k steps after operand balancing.

    t_j = (max_k_j / max_q_j)^(gamma/2); the balanced channel maxima are
    max_q_j * t_j = max_k_j / t_j; the step is the ~99.99th percentile of
    the balanced distribution (approximated by the channel maxima), with
    the measured link-level factors applied.
    """
    steps = {}
    for h in range(16):
        qmax = np.max(np.stack(samples[f"chan_q_h{h}"]), axis=0)  # [128]
        if f"chan_k_h{h // 2}" in samples:
            kmax = np.max(np.stack(samples[f"chan_k_h{h // 2}"]), axis=0)
        else:
            kmax = qmax
        t = np.power(np.maximum(kmax, 1e-9) / np.maximum(qmax, 1e-9), gamma / 2.0)
        q_bal = np.percentile(qmax * t, 99.99)
        k_bal = np.percentile(kmax / t, 99.99)
        steps[f"q_h{h}"] = max(1, int(np.ceil(q_bal * CAL_MARGIN * q_extra / 127 * (1 << 20))))
        if h < 8:
            steps[f"k_h{h}"] = max(1, int(np.ceil(k_bal * CAL_MARGIN * k_extra / 127 * (1 << 20))))
        steps[f"t_h{h}"] = t
    if gamma > 0:
        for h in range(16):
            if h < 8:
                steps[f"t_h{h}"] = steps[f"t_h{h}"]
    return steps


def load_token_cases(name: str):
    return json.loads((TESTDATA / f"{name}.json").read_text(encoding="utf-8"))["cases"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-references", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--seq", type=int, default=16)
    args = parser.parse_args()
    if not (args.check_references or args.convert or args.gate):
        parser.error("choose at least one stage")

    conf = Conformer(0)
    if args.check_references:
        from qwen3_reference import official_block_output, block_forward
        cases = load_token_cases("qwen3_f1_eval")[:3]
        worst = {"cosine": 1.0, "max_abs": 0.0}
        for case in cases:
            hidden = conf.embedding(case["token_ids"])
            ref_b, _ = instrumented_forward(conf, hidden)
            perm_b = block_forward(hidden, conf.weights, conf.cfg, dtype=np.float64,
                                   rotate="adjacent", permute_qk=True)
            perm_err = float(np.max(np.abs(perm_b - ref_b)))
            ref_a = official_block_output(hidden.astype(np.float32), 0)
            diff = ref_a - ref_b
            cosine = float(np.dot(ref_a.reshape(-1), ref_b.reshape(-1)) /
                           (np.linalg.norm(ref_a) * np.linalg.norm(ref_b)))
            worst["cosine"] = min(worst["cosine"], cosine)
            worst["max_abs"] = max(worst["max_abs"], float(np.max(np.abs(diff))))
            print(f"case: A-vs-B max_abs={float(np.max(np.abs(diff))):.3e} cosine={cosine:.8f} "
                  f"| perm-vs-half={perm_err:.3e}")
        ok = worst["cosine"] >= 0.9999 and worst["max_abs"] <= 1e-3
        print(f"reference agreement: worst cosine={worst['cosine']:.8f} worst max_abs={worst['max_abs']:.3e} "
              f"-> {'OK' if ok else 'FAILED'}")
        if not ok:
            sys.exit(3)

    if args.convert or args.gate:
        cal_cases = load_token_cases("qwen3_f1_calibration")
        maxima, samples = calibrate(conf, cal_cases, collect_samples=True)
        print("calibration maxima:", {k: round(v, 4) for k, v in maxima.items()})
        chosen = {}
        for key, vals in samples.items():
            step_fx, pct, mse = mse_optimal_scale(vals)
            if key.startswith("k_h"):
                step_fx = max(1, int(round(step_fx * K_HEAD_FACTOR)))
            chosen[key] = {
                "step_fx": step_fx,
                "percentile": pct,
                "mse": mse,
                "step_real": step_fx / (1 << 20),
                "calibration_max_abs": maxima.get(key, float(np.max(vals))),
                "clipped_fraction": float(np.mean(np.abs(vals) > step_fx / (1 << 20) * 127)),
            }
            if not key.startswith("chan_"):
                print(f"site {key:4s}: step={step_fx/(1<<20):.6f} pct={pct} mse={mse:.3e} "
                      f"clip={chosen[key]['clipped_fraction']*100:.3f}% "
                      f"(max {chosen[key]['calibration_max_abs']:.2f})")
        scales = {"chosen": chosen, "p": 1.0}
        seq = args.seq
        snake, go, plan, tensor_data, graph_id = build_converted_graph(conf, seq, scales)
        print(f"graph: {len(go['Nodes'])} nodes, {len(go['Inputs'])} inputs, graph_id={graph_id.hex()}")
        out_dir = OUT_DIR / f"seq{seq}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "graph.json").write_text(json.dumps(go, default=lambda b: b.hex()), encoding="utf-8")
        (out_dir / "quant_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
        (out_dir / "graph_id.txt").write_text(graph_id.hex() + "\n", encoding="utf-8")
        np.save(out_dir / "tensor_data.npy", np.array(
            [tensor_data[i] for i in sorted(tensor_data)], dtype=object), allow_pickle=True)
        print("artifacts written:", out_dir)

        if args.gate:
            report = run_gate(conf, snake, tensor_data, graph_id, seq)
            path = REPO / "docs" / "phase-f1-real-model-accuracy.json"
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("accuracy report:", path)
            print(f"worst cosine={report['worst_cosine']:.8f} worst max_abs={report['worst_max_abs']:.3e} "
                  f"-> {report['verdict']}")
            if report["verdict"] != "PASS":
                sys.exit(4)


def run_gate(conf: Conformer, snake: dict, tensor_data: dict, graph_id, seq: int) -> dict:
    eval_cases = load_token_cases("qwen3_f1_eval")
    per_case = []
    worst_cos, worst_abs = 1.0, 0.0
    for i, case in enumerate(eval_cases):
        hidden = conf.embedding(case["token_ids"])[:seq]
        ref, _ = instrumented_forward(conf, hidden)
        inputs = dict(tensor_data)
        x_desc = snake["inputs"][0]["desc"]
        inputs[0] = to_fx(hidden).astype(np.int32)
        # rebuild the committed root of x for this case
        x_root = N.tensor_root_np(x_desc, inputs[0])
        snake_case = json.loads(json.dumps(snake, default=lambda b: b.hex()))
        snake_case["inputs"][0]["root"] = x_root
        for ginput in snake_case["inputs"][1:]:
            ginput["root"] = bytes.fromhex(ginput["root"]) if isinstance(ginput["root"], str) else ginput["root"]
        rope_table = tensor_data[2]
        out = N.execute_graph_np(snake_case, inputs, rope_tables={2: rope_table})
        y_fx = out["tensors"][(1, snake["outputs"][0]["index"])].astype(np.float64)
        y = y_fx / (1 << 20)
        diff = y - ref
        cosine = float(np.dot(y.reshape(-1), ref.reshape(-1)) /
                       (np.linalg.norm(y) * np.linalg.norm(ref)))
        case_abs = float(np.max(np.abs(diff)))
        per_case.append({"case": i, "cosine": cosine, "max_abs": case_abs,
                         "mean_abs": float(np.mean(np.abs(diff)))})
        worst_cos = min(worst_cos, cosine)
        worst_abs = max(worst_abs, case_abs)
    verdict = "PASS" if worst_cos >= 0.995 and worst_abs <= 0.05 else "FAIL"
    return {
        "model": "Qwen/Qwen3-0.6B-Base",
        "revision": "57ca99e94acb83175495aa2c6b6b0cc498170924",
        "layer": 0, "sequence_length": seq,
        "calibration_set": "testdata/qwen3_f1_calibration.json",
        "eval_set": "testdata/qwen3_f1_eval.json",
        "graph_id": graph_id.hex(),
        "thresholds": {"worst_cosine": 0.995, "worst_max_abs": 0.05},
        "per_case": per_case,
        "worst_cosine": worst_cos, "worst_max_abs": worst_abs,
        "verdict": verdict,
    }


if __name__ == "__main__":
    main()
