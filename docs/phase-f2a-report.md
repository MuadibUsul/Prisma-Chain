# Phase F.2A report — Quantization Feasibility Gate

Branch `research/transformer-phase-f2a-quant` (from
`origin/protocol/transformer-phase-f1-real-model` @ 1072bc6). Research
round only: no protocol, chain, proto, GPU or devnet change (verify with
`git diff origin/protocol/transformer-phase-f1-real-model -- chain
compute` — empty).

## Verdict

```text
NOT FEASIBLE
CONTIGUOUS_GROUPWISE_INT8_NOT_SUFFICIENT
```

The cosine half of the predeclared gate IS reachable with static
contiguous per-slice INT8 (0.9954 on tuning at 3.7x nodes), but the
max_abs half (≤ 0.05) is structurally out of reach for every contiguous
grouping tested (128/64/32/16/8): the best measured worst-case is 0.24
(tuning) / 0.31 (heldout), driven by a BROAD precision floor (806 of
16384 elements above 0.05), not by a lone outlier. The gate is an AND,
so the answer is NOT FEASIBLE, and per the round's rules the search was
stopped instead of refining below group 8.

## A real F.1 defect found by this round (recorded, not hidden)

The F.2A bit-exactness anchor exposed a genuine bug in the F.1
converter: the scores dequant multiplier used the GLOBAL q/k steps while
the operands were quantized with PER-HEAD steps, scaling the attention
logits by the step ratio (4x too small under the k×0.25 policy). All F.1
accuracy numbers were measured through this bug. With the multiplier
derived from the actual per-head steps, the same F.1 policy measures
cosine 0.9933 / max_abs 0.423 on the F.1 evaluation set (previously
0.957 / 1.91). The corrected policy is the F.2A baseline.

## Correctness of the experimental simulator

* Anchor 1: a two-slice unequal-step case matches a hand-computed integer
  reference (ties-to-even, saturation, negatives included).
* Anchor 2: the all-OFF policy reproduces the converted graph
  BIT-IDENTICALLY (causal-mask tensor, every REQUANTIZE, MAC count).
* MAC invariant asserted at run time: Σ_s M·N·K_s = M·N·K; no saturation
  occurred in any candidate (saturations = 0 everywhere).

## Search results (tuning set; 8 cases during search, full 32 for the
frozen candidates; heldout untouched until freeze)

Baseline (corrected F.1 policy): cosine 0.99253, max_abs 0.279.

Marginals (one site at a time):
* k: grouping HURTS (g128 0.9607 → g8 0.9927); the per-head fine step
  with clipping already beats contiguous grouping — the F.1 k×0.25
  factor is re-validated as legitimate (0.9925 vs 0.9607 with plain
  steps).
* q: +0.0008 (g16/32); h: +0.0011 (g32, but 10.7x nodes); h2: +0.0004;
  hm: +0.0018 (g16); ctx: initially catastrophic due to a simulator
  double-shift bug (fixed; then +0.0005).
* No single site moves max_abs below 0.24.

Guided combinations (all recorded in docs/phase-f2a-search-results.json
plus this report): the best-on-tuning candidate is
**q32 + hm16 + h2_128** — tuning cosine 0.99539 / max_abs 0.240,
3.68x nodes, 4.05x GEMMs (under the 5x soft targets; MAC count
preserved).

## Frozen policy and fresh heldout (single run, after freeze)

PolicyID `f4d63575269a718f882e6df6d0433e013a47a68511ba40295b21d658c724368d`
(docs/phase-f2a-frozen-policy.json). Fresh heldout: 64 new sequences,
never used for selection.

```text
worst cosine  = 0.994210   (required >= 0.995)  FAIL
worst max_abs = 0.313896   (required <= 0.05)   FAIL
mean max_abs  = 0.202844 ; saturations 0
```

The tuning estimate did not generalize (0.9954 → 0.9942): exactly the
"tuning is not final" rule. Per section 85 the result is FAIL.

## Ablation of the frozen policy (tuning)

Full frozen: 0.99539 / 0.240 (tuning). Without q: 0.99426 / 0.300;
without hm: 0.99354 / 0.366; without h2: 0.99469 / 0.313 — hm contributes
most, then q, then h2; no site is load-bearing enough to change the
verdict.

## Final report questions

**Q1. Can static contiguous per-slice activation quantization close the
real Qwen3 accuracy gap without changing CANONICAL_MATH_V1?** NO.
cosine: YES in principle (0.9954 tuning / 0.9942 heldout are close but
short); max_abs: NO (0.24-0.31 vs 0.05, broad-based). The AND gate fails.

**Q2. Minimum slice policy that passes the ORIGINAL gate?** None of the
tested policies passes. Best: q group 32, hm group 16, h2 group 128 (all
contiguous, static; every group ≥ 8).

**Q3. Does the policy preserve the original GEMM MAC count?** YES —
asserted at run time (Σ M·N·K_s = M·N·K; identical per-case MAC total).

**Q4. Complexity increase (frozen policy):** nodes 310 → 1141 (3.68x);
GEMM nodes 83 → 336 (4.05x); REQUANTIZE 126 → 379; ADD +253; manifest
leaves 310 → 1142; Freivalds checks 83 → 336 (≈24.6k extra verification
vector MACs per round); dispute depth +2 bits; bundle/DA bytes grow
proportionally to node outputs (estimate; no bundle built this round).
All inside the round's soft targets (nodes ≤ 5x, checks ≤ 10x) — the
policy is affordable; it simply does not pass.

**Q5. Is the improvement primarily from fixing Q/K heavy-tail
quantization?** NO — after the F.1 scale bug was fixed, k grouping HURTS;
the gains are distributed over q/hm/h2 (and h). The heavy-tail story was
confounded by the logits-scale bug; with correct logits the tail is
already handled by the fine per-head step with clipping.

**Q6. Fresh held-out set not used for selection?** YES — 64 sequences
generated and frozen before the search; heldout loaded once, after the
policy freeze (explicit `--unlock-heldout`).

**Q7. Expressibility with SLICE_VIEW + existing REQUANTIZE_V1 /
GEMM_INT8_V1 / ADD_FIXED_V1 only?** YES — the simulator implements
exactly that operator set (integer mult/shift, ties-to-even, saturating
adds); no additional arithmetic was needed or assumed.

**Q8. Should Prisma proceed to F.2B protocol implementation?** NO.
Implementing SLICE_VIEW/partial GEMM would add ~3.7x graph complexity
and still fail the predeclared max_abs gate. The report's conclusion:
the current INT8 canonical architecture requires a deeper redesign
before real-model support (per section 92); the next architecture
discussion should start from the measured floor (max_abs ~0.24 from
8-bit activation precision at |x| ≤ 5 with a broad error tail) and the
option space outside contiguous static INT8.

## Data list (§88)

* baseline (corrected F.1 policy, tuning): cosine 0.99253, max_abs
  0.2786 — full per-candidate table in docs/phase-f2a-search-results.json
* best tuning: 0.99539 / 0.240 (q32+hm16+h2_128)
* frozen PolicyID: `f4d635…368d`
* fresh heldout: worst cosine 0.994210, worst max_abs 0.313896 (FAIL)
* K quant MSE: single-scale per head in
  docs/phase-f2a-outlier-analysis.json (`k_single_scale_mse`); groupwise
  steps for g16 recorded there too — grouping does not beat the fine
  single step, which is why k was NOT made groupwise
* nodes: 310 → 1142 (3.68x); GEMMs 83 → 336 (4.05x); REQ 126 → 379
* GEMM MAC: unchanged (invariant asserted)
* Freivalds checks: 83 → 336; manifest leaves 310 → 1141; dispute depth
  +2 bits; DA bytes: proportional growth (estimate)
* saturations: 0 in every candidate
* scale approximation error: max 6.7e-13 relative (scores mult rounding)
* worst-error record (§58): token 13 / channel 35, |ref| 4.046, error
  0.240; 806/16384 elements above 0.05, 26 above 0.1, 2 above 0.2

## Honest statement

Phase F.1 showed single-scale static W8A8 is insufficient for the pinned
real Qwen3 block. Phase F.2A shows that static CONTIGUOUS groupwise W8A8
(with a mathematically exact partial-GEMM + common-scale requantization
structure that preserves MACs and stays inside the deterministic
verification architecture) closes the cosine gate's margin but does not
close the max_abs gate — the predeclared AND condition fails, so F.2B
must NOT start. No thresholds were changed, no frozen arithmetic was
touched, no dynamic quantization was used, and no protocol code was
modified.
