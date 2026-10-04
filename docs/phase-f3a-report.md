# Phase F.3A report — Activation Bit-Depth Feasibility Gate

# NOT_FEASIBLE_THROUGH_A13

Branch `research/transformer-phase-f3a-activation-bits` (from F.2A).
Research-only round: no protocol/chain/proto change, no GPU, no devnet.
W8 weights frozen; the single experimental variable is the signed static
activation bit depth (global b in {8..13}); scale granularity stays at
the corrected-F.1 shape (per-head q/k; per-tensor elsewhere; NO
groupwise); CanonicalMathV1 and the predeclared AND gate untouched.

## The answer in one table (fresh heldout, 64 cases, single unlocked run
after freezing; selection numbers in docs/phase-f3a-selection-results.json)

| Bits | MaxSafeK | worst cos | worst max_abs | elems >0.05 | elems >0.1 | verdict |
|---|---|---|---|---|---|---|
| A8  | 131071 | 0.97366 | 0.9853 | 299063 | — | FAIL |
| A9  | 65535  | 0.98965 | 0.4364 | 115039 | — | FAIL |
| A10 | 32767  | 0.99662 | 0.2441 | 25806 | — | FAIL |
| A11 | 16383  | 0.99880 | 0.1631 | 1460 | — | FAIL |
| A12 | 8191   | 0.99935 | 0.0854 | 188 | 0 | FAIL |
| A13 | 4095   | 0.99941 | 0.0906 | 190 | 0 | FAIL |

cosine crosses the gate at A10; **max_abs never does**: it halves per bit
(0.985 → 0.436 → 0.244 → 0.163 → 0.085) and then PLATEAUS at ~0.085-0.091
for A12/A13. Verdict per the predeclared AND condition:
`NOT_FEASIBLE_THROUGH_A13`, `PROCEED_TO_F3B = NO`.

## Where the plateau comes from (measured, not hypothesized)

* A research-only diagnostic (NOT a candidate: weight levels raised to
  W10 with the same coverage) makes **A13 pass cleanly: cosine 0.99995,
  max_abs 0.0344, ZERO elements above 0.05**. The residual floor at
  A12/A13 is therefore the **frozen INT8 weight quantization**, not
  activation resolution. Weights are explicitly out of scope this round
  (the single variable is activation bits), so the honest conclusion is
  that activation precision above ~A10/A11 buys nothing further and the
  AND gate cannot be closed by activation bits alone.
* A calibration artifact was found and removed first: the F.1-era
  `k x0.25` link-level factor was tuned under the buggy scores scale and
  acted as pure clipping bias; the principled calibration choice is the
  MSE-optimal step (factor 1.0). With it, the family scales cleanly
  (without it, every candidate sat at max_abs ~0.27 regardless of bits —
  a false plateau).

## Accumulator safety (§10/§12/§37/§38/§39)

* `MaxSafeK(b) = floor((2^31-1) / (2^(b-1) * 128))` verified:
  A8 131071, A9 65535, A10 32767, A11 16383, A12 8191, A13 4095.
* Real converted graph scan (all 83 GEMM nodes): K_max = 3072 →
  all nodes SAFE for A8-A13 (`docs/phase-f3a-accumulator-safety.json`);
  A14 (MaxSafeK 2047) would be UNSAFE — A13 is the natural ceiling of the
  int32-accumulator architecture, as the round's rules anticipated.
* Every accumulator ASSERTED within signed int32 (never wrapped); the
  observed activation-range extrema are recorded per candidate
  (A13: |acc|max ≈ 1.56e7, ~0.73% of int32 max — 136x headroom), but
  admission must use the THEORETICAL bound, never the observed range.
* Saturations: 0 (activation clamps and Q12.20 adds) in every candidate.

## Protocol-cost profile (§40-48)

* Graph nodes 310 and GEMM nodes 83 UNCHANGED; GEMM MAC 252,706,816 per
  case UNCHANGED (the same graph runs; only the activation dtype widens).
* Freivalds checks 83 unchanged; the 512-MAC micro-step stays safe
  (K=8 < MaxSafeK for every candidate) but a versioned wider-activation
  GEMM arithmetic would be required — NOT IMPLEMENTED.
* Storage (persisted activation tensors, 245,760 elements per case):
  A8 246 KB → A13 logical 399 KB (1.62x) or int16 container 492 KB (2x).
  Weights unchanged (W8). DA growth is only this activation factor.
* Early-exit note: per §85 the table is the whole comparison; no further
  bits investigated.

## Lineage (kept separate on purpose, §87)

| Round | Setting | Result |
|---|---|---|
| F.1 original | single-scale W8A8 | 0.957 / 1.91 — CONTAINED A SCORES-SCALE BUG |
| F.2A corrected | single-scale W8A8 | ≈0.9933 / 0.423 (F.1 eval set) |
| F.2A groupwise W8A8 | frozen policy, fresh heldout | 0.994210 / 0.313896 — NOT FEASIBLE |
| F.3A | W8A8..W8A13 | A13: 0.99941 / 0.0906 — **NOT_FEASIBLE_THROUGH_A13** |
| F.3A diagnostic only | A13 + W10 weights | 0.99995 / 0.0344 → points the next architecture discussion at WEIGHT precision |

## Report questions

**Q1. Does increasing activation precision beyond INT8 close the
real-model gap without changing CANONICAL_MATH_V1?** NO — it closes the
cosine half (from A10) but not max_abs (plateau ~0.09 at A13).

**Q2. Minimum activation bit width passing the AND gate?** NONE of
{A9..A13}.

**Q3. Does a passing candidate preserve worst-case int32 accumulator
safety on every GEMM node?** No candidate passes; A8-A13 are all
accumulator-SAFE by the theoretical bound (and A14 would not be).

**Q4. MaxSafeK table?** A8 131071, A9 65535, A10 32767, A11 16383,
A12 8191, A13 4095 (verified against the code formula; A13 is the
highest width that keeps K_max=3072 safe).

**Q5. Does the F.2A error floor shrink with bit depth?** YES until
~A11/A12 (element counts above 0.05: 299k → 115k → 26k → 1.5k → 188 →
190), then it saturates at the W8-weight floor (diagnostic evidence
above).

**Q6. Do wider activations require more graph nodes or MACs?** NO —
nodes 310, GEMMs 83, MACs 252,706,816 all unchanged.

**Q7. Practical int16-container cost?** persisted activation bytes 2x A8
(1.62x if logically bit-packed at A13); bundle/DA growth limited to that
factor; weights unchanged.

**Q7b/Q8. Can Freivalds + single-tile bisection + 512-MAC arbitration be
reused with only versioned wider integer arithmetic?** Structurally YES
(same shapes, bounded K=8 micro-step, wider exact integer products), but
this is a claim about a future versioned implementation only — NOT
IMPLEMENTED.

**Q9. Proceed to a protocol implementation round?** **NO** (verdict
NOT_FEASIBLE_THROUGH_A13). The next architecture discussion should start
from the measured boundary: activation bits alone cannot clear the gate;
the residual floor is INT8 weight quantization (W10 diagnostic passes);
a future round could evaluate weight precision or other numeric
architectures with the same measurement discipline.

## Data list (§86)

* corrected A8 baseline and A8-A13 tables: docs/phase-f3a-selection-results.json,
  docs/phase-f3a-heldout-results.json (verdict + per-case aggregates)
* MaxSafeK + real-graph K scan: docs/phase-f3a-accumulator-safety.json
* candidate scales/PolicyIDs: docs/phase-f3a-candidate-policies.json,
  frozen family sha256 e650c2d735af380c…: docs/phase-f3a-frozen-candidates.json
* error histograms (six bins per candidate): docs/phase-f3a-error-histograms.json
* per-site quant MSE / step / clipping per bit: docs/phase-f3a-site-analysis.json
* storage/DA/cost: docs/phase-f3a-cost-model.json
* fresh disjoint data (overlap = 0 enforced): docs/phase-f3a-data-manifest.json
  and testdata/qwen3_f3a_{calibration,selection,heldout}.json
* anchors: A8 executor == converted graph bit-for-bit; a synthetic
  integer reference for the quantizer; MAC invariant asserted.

## Honest statements

No threshold was changed; CanonicalMathV1, the frozen operators, weights,
the chain and the verifier were not touched; the A13-in-int16 diagnostic
is not an INT16 model claim; the wide-activation verification story is a
structural observation, not an implementation. The measured answer to
"is the remaining fidelity gap fundamentally an 8-bit activation
resolution problem?" is **no: it is an INT8 weight-quantization floor
once activations are ≥~A12**.
