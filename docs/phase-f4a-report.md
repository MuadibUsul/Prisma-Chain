# Phase F.4A report — Joint Precision Frontier

# NOT_FEASIBLE_WITHIN_INT32_AW_FRONTIER

`PROCEED_TO_F4B = NO`. Branch `research/transformer-phase-f4a-joint-precision`
(from F.3A). Research-only: zero protocol diff
(`git diff origin/research/transformer-phase-f3a-activation-bits -- chain
compute proto` is empty), no GPU, no chain, no devnet, no new dtype or
operator. W8..W13 weights and A8..A13 activations share ONE frozen
quantization procedure each (per-slice maximum weight scales with
QmaxW=2^(w-1)-1; calibration-MSE activation steps per A, shared across W);
scales stay per-head q/k and per-tensor elsewhere; no groupwise, no
rotation, no dynamic scaling.

## The answer (fresh heldout, 64 cases, single unlocked run after freezing;
family sha256 97aceb4b542ea7ec…)

| candidate | worst cos | worst max_abs | elems >0.05 | verdict |
|---|---|---|---|---|
| A8W8 | 0.96649 | 1.1897 | 311158 | FAIL |
| A8W13 | 0.96531 | 1.1306 | 317368 | FAIL |
| A9W8 | 0.98727 | 0.5815 | 117752 | FAIL |
| A9W12 | 0.98729 | 0.6228 | 109221 | FAIL |
| A10W8 | 0.99510 | 0.3238 | 29356 | FAIL |
| A10W10 | 0.99551 | 0.2857 | 30590 | FAIL |
| A10W11 | 0.99571 | 0.2940 | 28840 | FAIL |
| A11W8 | 0.99851 | 0.1774 | 1971 | FAIL |
| A11W9 | 0.99897 | 0.1438 | 1041 | FAIL |
| A11W10 | 0.99909 | 0.1744 | 770 | FAIL |
| A12W8 | 0.99938 | 0.0928 | 153 | FAIL |
| **A12W9** | **0.99970** | **0.0887** | **13** | FAIL (best) |
| A13W8 | 0.99947 | 0.0859 | 87 | FAIL |

Selection (32-case subset) matched the heldout table closely — no
overfitting — and every safe candidate FAILS the AND gate. The best
frontier point (A12W9) passes cosine by a wide margin and misses max_abs
by 1.8x with 13 elements above 0.05 (a passing candidate requires that
count to be exactly 0).

## Why the whole frontier fails (measured structure)

Two opposing walls inside the safe budget a+w ≤ 21:

* **Activation-limited wall**: holding W fixed, max_abs improves with a
  but saturates at W8 (A11W8 0.177 → A12W8 0.093 → A13W8 0.086), and rail
  weight bits buy almost nothing when activation is coarse (A8W8 1.190 →
  A8W13 1.131; A10W8 0.324 → A10W11 0.294).
* **Weight-limited wall**: at fixed a, W9 helps (A12W9 0.0887 vs A12W8
  0.0928; A11W9 0.1438 vs A11W8 0.1774) but W10+ add nothing (A11W10
  0.1744; A10W11 0.294) — consistent with F.3A's diagnosis once the
  activation side is already near its own floor.

The one combination that ever measured numerically passing — **A13W10
(0.99993-0.99995 / 0.034-0.049 across rounds)** — needs a+w = 23, i.e.
`MaxSafeK(13,10) = 1023 < 3072`: **UNSAFE_DIAGNOSTIC_ONLY**, never a
candidate. In other words the worst-case signed-int32 accumulation
architecture caps the precision budget about two bits short of what the
pinned real block needs.

## Accumulator safety (theoretical, not observed)

* `MaxSafeK(a,w) = floor((2^31-1) / 2^(a+w-2))` implemented and verified;
  real graph: 83 GEMM nodes, K_max = 3072 → the pre-registered 21 pairs
  with a+w ≤ 21 are all safe (`docs/phase-f4a-accumulator-safety.json`).
* Safety grid (✓ safe, X unsafe at K=3072): the frontier row a+w=21 is
  (A13W8)(A12W9)(A11W10)(A10W11)(A9W12)(A8W13); everything above the
  diagonal is excluded before measurement, including A13W10/A12W10/A13W9.
* Observed accumulator extrema stay far inside int32 (A13W8
  |acc|max ≈ 1.59e7, 136x headroom) and zero int32 overflows/saturations
  occurred — recorded, but admission uses the theoretical bound only
  (§18/§74).
* Future micro-step check: the 8×8×8 / 512-MAC arbiter with the frontier
  worst case (2^12 × 2^9 × 8) is 1.7e7 ≪ 2^31 — structurally safe for any
  of the 21 pairs; the standalone-task admission would need a versioned
  MaxSafeK(a,w), NOT IMPLEMENTED.

## F.3A diagnosis: independently confirmed (Q6)

Yes. In the frozen heldout, with activation fixed at A12: W8 → W9 lowers
max_abs 0.0928 → 0.0887 and the >0.05 count 153 → 13; with A11: W8 → W9
0.1774 → 0.1438. Weight precision genuinely moves the floor, but the
SAFE budget cannot supply enough of it alongside enough activation bits.
(The unsafe diagnostic's 0.034 remains the measured evidence that the
needed weight precision exists — just not inside the int32 contract.)

## Cost model and unchanged structure (Q7/Q8)

* Graph: nodes 310, GEMM nodes 83, GEMM MAC 252,706,816 per case,
  Freivalds checks 83, manifest leaves 310 — unchanged by any bit pair
  (verified against the real converted graph; no topology change).
* Weights (layer 0, 9,961,472 quantized elements; per-matrix MSE, max quantization error, step and clipped fraction for W8..W13): logical bytes scale with W bits;
  int16 containers double.  Weights are reusable across tasks.
* Activations: per-task persisted elements unchanged (245,760); logical
  bytes scale with A bits; int16 containers 2x.
* Logical storage is NOT reported as one misleading total: weights
  (reusable, distribution cost) and activations (per task) are separate.
* Wider integer arithmetic is a structural statement only: Freivalds /
  single-tile dispute / 512-MAC arbitration appear reusable with a
  versioned arithmetic — **NOT IMPLEMENTED**.

## Lineage (kept strictly separate)

| Round | Setting | Fresh-heldout result |
|---|---|---|
| F.2A | contiguous groupwise W8A8 | 0.994210 / 0.313896 NOT FEASIBLE |
| F.3A | W8A8..W8A13 | A13 0.99941 / 0.0906 NOT_FEASIBLE_THROUGH_A13 |
| F.3A diagnostic | A13W10 (UNSAFE) | 0.99993-0.99995 / 0.034-0.049 |
| F.4A | 21 safe (a,w) pairs | best A12W9 0.99970 / 0.0887 — NOT_FEASIBLE_WITHIN_INT32_AW_FRONTIER |

## Report questions

**Q1.** Does joint activation/weight precision close the gap under
CANONICAL_MATH_V1 + signed int32 accumulation? **NO.**

**Q2.** Lowest-cost passing safe candidate? **NONE** (predeclared ranking
min-W then min-A had no passer to apply to; the best-quality point A12W9
fails anyway).

**Q3.** Fresh-heldout gate values? Best candidate A12W9: worst cosine
0.99970 (≥0.995 ✓), worst max_abs 0.0887 (≤0.05 ✗), count >0.05 = 13
(must be 0).

**Q4.** Worst-case int32 accumulator safety for every real GEMM node?
YES for all 21 safe candidates by the theoretical bound (K_max 3072); the
numerically-attractive A13W10 is UNSAFE and excluded.

**Q5.** MaxSafeK(a,w) table: see `docs/phase-f4a-accumulator-safety.json`
(A13W8 4095, A12W9 4095, A11W10 4095, A10W11 4095, A9W12 4095, A8W13 4095;
A13W10 1023 excluded; full 21-row table in the file).

**Q6.** F.3A weight-floor diagnosis independently confirmed? **YES**
(see the W8→W9→W10 deltas above).

**Q7.** Does a passing candidate change nodes/GEMMs/MAC? No candidate
passes; for all candidates the measured graph values are unchanged
(310 / 83 / 252,706,816).

**Q8.** Logical and practical storage costs? Weights vs activations
reported separately in `docs/phase-f4a-cost-model.json`; activation bytes
scale with A, weight bytes with W, int16 containers 2x.

**Q9.** Can the future arithmetic structurally retain Freivalds +
single-tile dispute + 512-MAC arbitration? YES structurally (bounded
local K=8, worst-case local accumulator 1.7e7 ≪ 2^31) — **NOT
IMPLEMENTED**.

**Q10.** Proceed to protocol implementation? **NO** — no
FEASIBLE_AxWy exists within the int32 frontier.

## Data list (§86)

corrected A8 baseline and all 21 rows: docs/phase-f4a-selection-results.json
and docs/phase-f4a-heldout-results.json; MaxSafeK table + 83-node scan:
docs/phase-f4a-accumulator-safety.json; PolicyIDs (21):
docs/phase-f4a-candidate-policies.json; frozen family + sha256:
docs/phase-f4a-frozen-family.json; histograms:
docs/phase-f4a-error-histograms.json and docs/phase-f4a-search-results.json;
weight MSE/step per matrix per W:
docs/phase-f4a-weight-analysis.json; stage analysis (priority candidates):
docs/phase-f4a-stage-analysis.json; cost:
docs/phase-f4a-cost-model.json; fresh disjoint data (overlap = 0):
docs/phase-f4a-data-manifest.json; anchors: A8W8 bit-identical to the
converted graph, A13W8 ≈ F.3A, A13W10 ≈ F.3A diagnostic.

## Honest statement

No threshold was changed; CanonicalMathV1, the frozen operators, the
chain and the verifier were not touched; unsafe combinations were never
candidates; "logical bits in an int16 research container" is not an
INT16 model; numerical feasibility is not GPU or protocol support. The
measured boundary for the next architecture discussion: within
worst-case-safe int32 accumulation the joint integer budget is about two
bits short of the pinned block's requirement — the next round may
consider int64 accumulation or different numeric representations
(rotation etc.) with the same measurement discipline.
