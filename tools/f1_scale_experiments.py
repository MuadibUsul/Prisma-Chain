"""Ablation harness for QWEN3_BLOCK_PROFILE_V1 quantization layouts.

All variants are algebraically EXACT refactorings expressible with the
existing operators (no protocol change):

  * the attention dot product is invariant under the per-pair rotation,
    so per-channel fixed factors can be moved between the Q and K sides;
  * norm weights can be folded into the following weight matrix columns
    (converter-side slicing), making the normalized activations unit
    scale;
  * a per-channel fixed factor can be applied with MUL against a
    committed constant tensor.

Each variant is evaluated on a few EVAL cases for orientation only; the
official gate always runs on the full held-out set through the converter.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import canonical_ref as R
import f1_canonical_numpy as N
from convert_qwen3_block import (
    Conformer, build_converted_graph, instrumented_forward, load_token_cases,
    mse_optimal_scale, to_fx,
)


def make_steps(samples, keys):
    return {k: mse_optimal_scale(samples[k])[0] for k in keys}


def run_variant(conf, samples, cases, variant: str):
    """variant in {v0_original, v1_kfold_all, v3_normsplit}."""
    cfg = conf.cfg
    eps = cfg["rms_norm_eps"]

    def rms(x, w):
        var = np.mean(x * x, axis=-1, keepdims=True)
        return x / np.sqrt(var + eps) * w

    worst_cos, worst_abs = 1.0, 0.0
    for case in cases:
        hidden = conf.embedding(case["token_ids"])
        ref, _, _ = instrumented_forward(conf, hidden, collect=True)
        seq = hidden.shape[0]
        x = to_fx(hidden).astype(np.int64)

        def fx_mul(a, b):
            return N.mul_fx(a, b)

        # --- canonical pipeline with the variant's layout --------------
        h_w = conf.weights["input_layernorm.weight"].astype(np.float64)
        wq_all = [conf.q_slice(i) for i in range(16)]
        wk_all = [conf.k_slice(i) for i in range(8)]
        wv_all = [conf.v_slice(i) for i in range(8)]
        qnorm, knorm = conf.q_norm(), conf.k_norm()

        if variant == "h_fold":
            # fold the attn-norm weight into the GEMM columns; h is unit
            for arr in wq_all + wk_all + wv_all:
                arr *= h_w[None, :]
            h_weight = np.ones_like(h_w)
        else:
            h_weight = h_w

        h_fx = N.op_rmsnorm(x.reshape(seq, -1), to_fx(h_weight).astype(np.int32),
                            max(1, round(1e-6 * (1 << 20)))).astype(np.int64)

        def quantize(v_fx, step):
            q = N.rshift_round_even(v_fx * np.int64(1 << 30), 30)
            del q
            return np.clip(N.rshift_round_even(v_fx * np.int64(round((1 << 30) / step)), 30),
                           -127, 127).astype(np.int64)

        # per-head steps from the variant's distributions
        steps = variant_steps(conf, samples, variant)

        rop = make_rope(conf, seq)
        scores = np.zeros((seq, seq), dtype=np.int64)
        q8s, k8s = [], []
        for qh in range(16):
            wq = wq_all[qh]
            qv = h_fx @ to_fx(wq) if False else None
            qv = N.op_gemm(h_fx.astype(np.int32),
                           np.clip(np.rint(wq * (1 << 20) / np.max(np.abs(wq)) * 127 / (1 << 20) * 0 + 0), 0, 1).astype(np.int8), False) if False else None
        raise SystemExit("unused")

    return worst_cos, worst_abs


def variant_steps(conf, samples, variant):
    raise SystemExit("unused")


def make_rope(conf, seq):
    raise SystemExit("unused")


if __name__ == "__main__":
    raise SystemExit("scaffold only — see convert_qwen3_block.py")
