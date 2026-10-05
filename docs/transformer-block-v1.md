# Verifiable Transformer block v1 (TRANSFORMER_BLOCK_V1)

`BuildTransformerBlockV1` expands a pre-norm Transformer block into a
static canonical graph. Attention and the SwiGLU MLP are graph MACROS:
they expand into the canonical operator set — there is no black-box
"attention operator" and no second protocol architecture.

```
x1 = x + Σ_h Wo_h · (softmax(scale · (RoPE(Q_h)·RoPE(K_h)ᵀ)) · V_h)
out = x1 + Wdown · (SiLU(Wgate·norm(x1)) ⊙ (Wup·norm(x1)))
```

## How the macro expands (and why)

* Heads are separate weight slices chosen by the converter
  (`Wq_h`, `Wk_h`, `Wv_h`, `Wo_h` are committed graph inputs), so no
  runtime slicing or views of activations exist.
* Head concatenation is the mathematically equivalent sum over per-head
  projections `Σ_h O_h·Wo_h` of the same linear map.
* `scores = Q·Kᵀ` uses the GEMM node's `transpose_b` form.
* The attention scale is folded into the scores' REQUANTIZE parameters
  (`AttentionScaleFx`, integer-only `floor(2²⁰/√d)`).
* Activations are statically quantized: fixed per-stage scales chosen at
  conversion time (classic W8A8). Dynamic per-input scales are NOT part
  of v1; this is a documented limitation, not an accident. Quantization
  is exercised by the values, not by float code: every REQUANTIZE node
  carries its committed `mult/shift/clamps`.
* Norm weights, the RoPE table and all quantized weights are committed
  input roots inside `GraphID`.

The builder's own order is the canonical one; the chain re-derives every
output descriptor from the operator contracts, so a hand-written graph
with mismatched declarations is rejected structurally.

## Sizes that are exercised

| Block | seq | d_model | heads | head_dim | mlp | nodes | GEMM MACs | status |
|---|---|---|---|---|---|---|---|---|
| mini | 16 | 128 | 4 | 32 | 256 | 90 | 2,686,976 | executed (Go tests, devnet E2E) |
| medium | 64 | 512 | 8 | 64 | 2048 | 166 | 272,629,760 | executed (Go test, ~13 s scalar) |

Measured single-threaded reference speeds (docs/phase-f-benchmark-results.json):
mini ≈ 0.41 s/run (6.6M MAC/s), medium ≈ 13.0 s/run (21M MAC/s). These are
scalar reference numbers for protocol verification, not a performance
claim; a GPU backend is NOT TESTED and must reproduce the bit-exact
vectors before it can be admitted.

## Accuracy gate

Thresholds were predeclared in docs/canonical-math-v1.md before any
benchmark: elementwise exact; RMSNorm max-abs ≤ 5e-3 and cosine ≥ 0.9999;
SiLU absolute ≤ 5e-3 (|x|≤1) or relative ≤ 1e-2 (|x|>1); Softmax
absolute ≤ 5e-2 and cosine ≥ 0.99; RoPE absolute ≤ 1e-4; whole block
cosine ≥ 0.995 and absolute ≤ 5e-2 against a float reference. The
numeric-design simulation behind those thresholds is
docs/canonical-math-v1-analysis.json (Q12.20 measured block-level cosine
0.997 against float64; Q8.24 was excluded for dynamic range).

The end-to-end block accuracy against a real float model is part of the
converter milestone below; the *arithmetic* is fully specified and
cross-language bit-exact today.

## Conversion plan (REAL_MODEL_BLOCK = NOT TESTED)

The converter is not built in this milestone, so `REAL_MODEL_BLOCK` is
NOT TESTED and the phase status is PARTIAL by the predeclared rule. The
plan the code already supports:

1. pick a pinned open model revision and one Transformer block;
2. export per-head `Wq/Wk/Wv/Wo` and `Wgate/Wup/Wdown` as int8 with a
   calibration pass choosing the static scales;
3. generate the pinned RoPE table with high-precision math into Q12.20;
4. emit the descriptor, quant plan and committed roots;
5. run the accuracy gate against the float block on fixed seeds, then
   cross-GPU bit-exactness (two GPUs) before any throughput number.

Until steps 1-5 run, this document claims protocol-level verification of
a canonical quantized Transformer block — never "we verify a model".
