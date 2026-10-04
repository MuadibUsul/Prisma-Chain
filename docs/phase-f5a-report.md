# Phase F.5A report — Wide-Accumulator Joint Precision Feasibility

# FEASIBLE_INT64_A13W10

A **numerically feasible** candidate exists once ONLY the canonical GEMM
accumulator is widened to signed int64: **A13W10** (activation 13-bit,
weight 10-bit, logical values stored in int16 research containers) passes
the unchanged predeclared gate on a fresh heldout set with
`count(error > 0.05) = 0`. This is a research result under a versioned
wide-integer arithmetic — **NOT IMPLEMENTED** in the protocol; zero
protocol diff on this branch; GPU_BACKEND = NOT TESTED.

Branch `research/transformer-phase-f5a-int64-frontier` (from F.4A).
`git diff origin/research/transformer-phase-f4a-joint-precision -- chain
compute proto` is empty. No threshold changed; CanonicalMathV1 untouched;
no groupwise/SLICE_VIEW, no rotation/SmoothQuant, no dynamic scaling, no
mixed per-site precision, no A14/W14, no int64 protocol code.

## Fresh heldout (64 cases, single unlocked run on the frozen 15-family)

| candidate | sum | worst cos | worst max_abs | >0.05 | verdict |
|---|---|---|---|---|---|
| A13W9 | 22 | 0.999841 | 0.05177 | 5 | FAIL |
| A12W10 | 22 | 0.999736 | 0.08074 | 12 | FAIL |
| A11W11 | 22 | 0.999461 | 0.12185 | 405 | FAIL |
| A10W12 | 22 | 0.997170 | 0.41390 | 15988 | FAIL |
| A9W13 | 22 | 0.986380 | 0.48948 | 139617 | FAIL |
| A12W11 | 23 | 0.999800 | 0.07082 | 75 | FAIL |
| **A13W10** | **23** | **0.999912** | **0.03893** | **0** | **PASS** |
| A11W12 | 23 | 0.999530 | 0.10897 | 226 | FAIL |
| A10W13 | 23 | 0.997057 | 0.42848 | 16012 | FAIL |
| A13W11 | 24 | 0.999957 | 0.04532 | 0 | PASS |
| A12W12 | 24 | 0.999786 | 0.07564 | 76 | FAIL |
| A11W13 | 24 | 0.999477 | 0.12956 | 380 | FAIL |
| A13W12 | 25 | 0.999951 | 0.04754 | 0 | PASS |
| A12W13 | 25 | 0.999796 | 0.11391 | 76 | FAIL |
| A13W13 | 26 | 0.999947 | 0.03471 | 0 | PASS |

Ranking (predeclared: min a+w → min W → min A): passers are
A13W10(23), A13W11(24), A13W12(25), A13W13(26) → **winner A13W10**
(unique at sum 23, and the lowest weight bits among passers).
Selection results (64-case subset) tracked the heldout table closely.

## Exactly how far beyond the int32-safe frontier the block needs to go

* **sum 22 fails**: the +1-bit frontier falls just short (best A13W9 at
  0.0518 with 5 elements above 0.05; A12W10 0.0807). The earlier
  "approximately +2 operand bits" estimate is now confirmed on a fresh
  heldout rather than asserted (§49-56): the int32-safe budget a+w ≤ 21
  is insufficient, a+w = 23 suffices, a+w = 22 does not.
* The winning pair also matches the F.3A diagnostic direction: the
  remaining constraint was the accumulator width, not the model mapping.

## Arithmetic feasibility classification of the winner

* **GEMM accumulation**: worst product 2^21, K_max 3072 →
  `MaxSafeK64(13,10) = 4,398,046,511,103 ≫ 3072` —
  INT64_ACC_SAFE for all 83 real GEMM nodes by the theoretical bound;
  observed |acc|max ≈ 1.6e7 (≈1.7e-12 of int64), zero overflows.
* **Requantization (accumulator → Q12.20)**: every GEMM→REQUANTIZE link
  measured (`docs/phase-f5a-requant-bounds.json`): the largest exact
  product `|acc| × mult` needs ≤ 47 signed bits for every candidate →
  **REQUANT_INT64_SAFE**. (A wide-shift multiplier variant was probed to
  rule out a multiplier-representation floor: it changes accuracy by
  < 0.001 and pushes the requant intermediate to 69 bits, i.e. beyond
  int64 — the legacy shift-20 multipliers are both sufficient and
  int64-clean. Recorded, not used.)
* **Freivalds cheap detection**: worst-case bounds with r ∈ {0,1}^N stay
  inside signed int64 for all 15 candidates
  (`docs/phase-f5a-freivalds-bounds.json`; required ≤ 49 bits) with large
  margins.
* **512-MAC micro-step**: worst local accumulator 8·2^(a+w−2) ≤ 2^27 ≪
  2^63 → the single-tile dispute stays expressible with the wider
  integer arithmetic.
* All of the above are structural/arithmetic statements about a future
  versioned arithmetic; a wider-activation implementation is
  **NOT IMPLEMENTED** and GPU feasibility is an open gate (existing
  INT8→INT32 tensor-core paths cannot run A13W10 as-is).

## Anchors (run before the fresh sweep)

* The int64 executor is **bit-identical** to the F.4A int32 executor for
  the int32-safe candidates A12W9 and A13W8 (widening is neutral where
  no overflow exists).
* The historical A13W10 diagnostic reproduces (0.99993 / 0.0492) on the
  F.3A data.
* Verdict-computation correction (disclosed): the frozen family recorded
  its `safe` field with the INT32 formula, so the automatically printed
  verdict was NOT_FEASIBLE despite four numerically passing rows. No
  PolicyID, scale, clamp or bit width changed after the freeze; the
  corrected verdict recomputes PASS/FAIL from the stored heldout metrics
  plus the F.5A int64 safety verified in
  `docs/phase-f5a-accumulator-safety.json`, and both the printed and the
  corrected verdicts are preserved in
  `docs/phase-f5a-heldout-results.json`.

## Structure and cost (unchanged by operand bits)

* Graph: nodes 310, GEMM nodes 83, MAC 252,706,816 per case, manifest
  leaves 310, Freivalds checks 83 — all identical to the frozen block.
* Weights: 9,961,472 quantized elements (reusable across tasks); W10
  logical bytes scale 8→10 bits vs W8; int16 research containers.
* Activations: 245,760 persisted elements per case; A13 logical 399 KB /
  492 KB int16 per case; the int64 accumulator is kernel-internal
  (`docs/phase-f5a-cost-model.json` distinguishes streaming vs
  materialized temporary bytes; a materialized [16,3072] int64 tile
  would be 393 KB of temporary bandwidth).
* Verification implications: same shapes, wider arithmetic; the dispute
  and receipt structure is untouched and **NOT IMPLEMENTED**.

## Lineage (kept strictly separate)

| Round | Setting | Fresh-heldout result |
|---|---|---|
| F.2A | groupwise W8A8 | NOT FEASIBLE |
| F.3A | W8A8..W8A13 | NOT_FEASIBLE_THROUGH_A13 |
| F.4A | 21 int32-safe pairs | NOT_FEASIBLE_WITHIN_INT32_AW_FRONTIER |
| F.5A | 15 int64-only pairs | **FEASIBLE_INT64_A13W10** |

## Report questions

**Q1.** Does widening the accumulator beyond int32 unlock a passing joint
precision pair? **YES** (on the fresh heldout).

**Q2.** Minimum total precision budget a+w that passes? **23**
(A13W10); 22 fails.

**Q3.** Selected passing pair? **A13W10** (predeclared ranking).

**Q4.** Fresh heldout metrics of the winner: worst cosine **0.999912**,
worst max_abs **0.03893**, count(>0.05) **0**.

**Q5.** Worst-case int64 GEMM accumulation safety on all 83 real nodes?
**YES** (theoretical bound 4.4e12 ≫ K_max 3072).

**Q6.** Can accumulator→Q12.20 requantization run with int64
intermediates only? **YES — REQUANT_INT64_SAFE** (≤ 47 bits observed).

**Q7.** Can exact Freivalds detection stay within int64? **YES**
(FREIVALDS_INT64_SAFE for all 15; ≤ 49 required signed bits) — NOT
IMPLEMENTED.

**Q8.** Does the candidate change nodes / GEMMs / MAC? **NO**
(310 / 83 / 252,706,816).

**Q9.** +1, +2, or more bits beyond the int32-safe budget? **+2 total
operand bits** (sum 22 fails, sum 23 passes) — measured on fresh data.

**Q10.** Proceed to wide-integer protocol/GPU feasibility
implementation? **YES**, with the attached requirements: versioned
wide-integer GEMM arithmetic (operands logical A13/W10, exact int64
accumulator), requantization that stays int64, a versioned Freivalds and
512-MAC arbitration under the same arithmetic, and an explicit GPU
implementation gate (GPU_BACKEND = NOT TESTED). Phrasing per §91: this is
NUMERICALLY FEASIBLE UNDER A VERSIONED WIDE-INTEGER ARITHMETIC — not
"Prisma now supports A13W10".

## Data list

docs/phase-f5a-{data-manifest, accumulator-safety, candidate-policies,
frozen-family, selection-results, heldout-results, error-histograms,
weight-analysis, stage-analysis, requant-bounds, freivalds-bounds,
cost-model}.json plus this report; fresh disjoint data in
testdata/qwen3_f5a_*.json (overlap = 0 against F.1/F.2A/F.3A/F.4A).

## Honest statement

The gate, the reference, the model, the scale granularity, the frozen
operators and the chain were not touched; no candidate was retuned after
the heldout was opened; the verdict-flag correction is disclosed above
and changed no policy; wide-integer protocol code and GPU work are NOT
IMPLEMENTED. The stop rule is met: plain-number-format research ends
here (highest-weight-bits winner A13W10 at sum 23), and the next round is
the F.5B wide-integer implementation feasibility question (GPU kernel
strategy, wide requantization, Freivalds and 512-MAC dispute arithmetic).
