# Phase F.1 report: real model closure — REAL_MODEL_BLOCK = FAIL (precision)

Branch `protocol/transformer-phase-f1-real-model` (worktree from
`origin/protocol/transformer-phase-f` @ 8c0ce61, 15 commits behind the
Phase E ref at start). Baseline suites (canonical Go, gemmv1 + verify,
chain 38 tests, Python mirror) were green before any change.

## Status summary (no blurring)

```
Pinned real checkpoint (manifest, hashes)     PASS
Exact float reference vs official Qwen3       PASS  (cosine 1.00000000, max_abs 1.6e-6)
Rotate-half -> adjacent-pair permutation      PASS  (0.0 difference)
Theoretical mapping (q/k norm, GQA, mask)     PASS  (existing operators only)
REAL_MODEL_BLOCK accuracy gate                FAIL  (worst cosine 0.957 vs 0.995;
                                                      worst max_abs 1.91 vs 0.05)
V2 commitment + manifest (protocol)           PASS  (canonical + chain + tests)
GPU_BACKEND                                   NOT TESTED (no GPU pods available)
GRAPH_WATCHER_E2E                             NOT TESTED (offline pieces exist)
GRAPH_DA_INTEGRATION                          NOT TESTED
GRAPH->GEMM 512-MAC bridge                    NOT TESTED
4-validator real-block E2E                    NOT TESTED
```

By the predeclared rules (Phase F.1 sections 28/98): **Phase F remains
PARTIAL.** No threshold was changed, no frozen arithmetic touched, no
operator added, no result dressed up.

## What was actually done

1. **Pinned checkpoint.** `Qwen/Qwen3-0.6B-Base` at revision
   `57ca99e94acb83175495aa2c6b6b0cc498170924`, downloaded and
   hash-verified (`model.safetensors` sha256 `cd2a5120…23eba` equals the
   blob etag), license apache-2.0, full file manifest with per-file
   SHA256 in `docs/phase-f1-model-manifest.json`. No `trust_remote_code`,
   no floating revision. The real config matches the expected structure
   exactly (hidden 1024, intermediate 3072, 16 Q heads / 8 KV heads,
   head_dim 128, eps 1e-6, rope_theta 1e6, attention_bias false) — no
   discrepancy to record.

2. **Reference validation before any quantization** (section 24). The
   hand-written float64 layer reference (Reference B) agrees with the
   official transformers implementation (Reference A) to cosine
   1.00000000 and max_abs 1.6e-6 on real tokenizer inputs; the
   rotate-half→adjacent-pair conversion used by the canonical ROPE is
   exact (identical values, difference 0.0). The mapping therefore uses
   ONLY existing operators: `RMSNORM_FIXED_V1` for q_norm/k_norm, static
   GQA wiring (one K/V chain referenced by two Q-head branches), a
   committed causal-mask constant added with `ADD_FIXED_V1` (mask value
   `-(64 << 20)`, far below the canonical exp underflow), the attention
   scale folded into the scores `REQUANTIZE`, and per-head weight slices
   straight from the checkpoint bytes.

3. **Converter** (`tools/convert_qwen3_block.py`, profile
   `QWEN3_BLOCK_PROFILE_V1`): real weights → 310-node
   `CANONICAL_GRAPH_V1` (58 committed inputs), GraphID
   `6ad46fde…ced0a0` for the tuned configuration, static W8A8 quant plan
   (`docs/phase-f1-quantization-plan.json` beside the artifacts) with
   per-site steps chosen by measured quantization MSE on a FROZEN
   calibration set (16 sequences), evaluation on a disjoint 16-sequence
   held-out set. Token ids are frozen in `testdata/qwen3_f1_*.json`.

4. **Accuracy gate: FAIL.** Worst-case over the held-out set:
   cosine 0.957, max_abs 1.91 (thresholds 0.995 / 0.05, predeclared and
   untouched). Full per-case numbers:
   `docs/phase-f1-real-model-accuracy.json`.

## Where the error comes from (located, as required)

The gate's error is **activation quantization**, not the arithmetic:

* The canonical math is exact: Go/Python cross-language vectors are
  bit-exact and a bit-exact numpy executor (validated on 85k samples,
  every operator vector and the full graph) reproduces the converted
  graph byte for byte.
* Weight quantization is NOT the bottleneck: MSE-optimal weight scales
  change the gate by < 0.001.
* The Q/K path dominates: scaling only the k site's step 4× drops the
  cosine to 0.64; the k tail comes from the pinned `k_norm` weight
  (max 96.5, 0.6B checkpoint), which forces one static step over a
  ~100× channel range. Per-head MSE-optimal k steps × 0.25 (calibration
  only) lift the gate from 0.951 to 0.964.
* Everything else (h, h2, ctx, gate, up, down, hm, o_proj) costs roughly
  0.89-0.93 cosine when its step is doubled — the error is distributed
  across all activation sites, ~0.035 cosine in total after tuning.
* Two exact refactorings were tested and measured WORSE: per-channel
  operand balancing (q·t vs k/t; 0.83 at full balance) and balanced+
  tuned combinations.

## The expressivity limit and the proposed (NOT implemented) extension

With the frozen operator set there is exactly ONE static scale per
tensor: the head coupling RMSNorm sits between the weight slicing and
the quantization, so per-channel/per-block activation steps are not
expressible, and heavy-tailed channels cannot be isolated. Closing the
gap needs a versioned protocol extension — per the Phase F.1 rules it is
PROPOSED ONLY, never applied here:

* a `REQUANTIZE` form with per-slice steps (still integer, still
  static), or
* an explicit canonical slice/view tensor op that lets a head be split
  into blocks with independent committed scales.

A follow-up phase should decide this with the accuracy evidence in hand;
the predeclared thresholds stay binding.

## Data list

* `docs/phase-f1-model-manifest.json` — repository, revision, license,
  per-file SHA256, tool versions.
* `docs/phase-f1-real-model-accuracy.json` — per-case metrics, worst
  values, thresholds, verdict, attribution, measured variants.
* `models/qwen3-0.6b-base-canonical/layer0/seq16/` (gitignored
  artifacts) — graph.json, graph_id, quant plan, tensor data.
* `testdata/qwen3_f1_calibration.json`, `testdata/qwen3_f1_eval.json`.
* Tools: `tools/f1_model_manifest.py`, `tools/qwen3_reference.py`,
  `tools/f1_make_token_sets.py`, `tools/convert_qwen3_block.py`,
  `tools/f1_canonical_numpy.py` + `tools/f1_numpy_check.py` (bit-exact
  executor proof), `tools/f1_attribution.py` (sensitivity harness).
* Protocol (this phase, additive): node-output manifest, GraphResultCommitV2,
  V2 receipt, `MsgSubmitGraphResultV2`, admission bounds and gas policy for
  a 310-node graph; all covered by the existing chain suite (green).

## Honest statement (worded per section 101)

Prisma converts one exact pinned Qwen3 Transformer block into
CANONICAL_GRAPH_V1 and executes it under its canonical quantized
arithmetic with bit-exact cross-language semantics — and the block does
NOT pass the predeclared accuracy gate, so no claim is made that the
real model block is verified at the required fidelity. Phase F remains
PARTIAL.
