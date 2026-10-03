# Prisma GEMM Verification v0.1.2 — Report

Branch `protocol/gemm-v0.1.2-freivalds`, frozen v0.1.1 baseline tag
`v0.1.1-phase0` (commit `5334766`). Protocol:
`docs/gemm-verification-v0.1.2.md`. Benchmark machine: local Windows x64,
Go 1.19.3, CPU exact-int64 paths only. Every status is PASS, FAIL or
NOT TESTED.

## Commits

- `830c108` Fix MaxSafeK to the true int32 worst-case bound
- `5f96a8a` Add GEMM v0.1.2 Freivalds verification layer
- `038cd42` Add Python Freivalds mirror and cross-language vectors
- `e4546fe` Add fast verification CLI, v0.1.2 benchmark ladder and fast E2E
- (final commit) docs + gate

## MaxSafeK result

The old bound `MaxInt32 / (127*127) = 133162` was wrong: the largest
|int8 x int8| product is (-128) x (-128) = 16384, so the true int32-safe
bound is `(2^31-1)/16384 = 131071`. Fixed, with the adversarial test:
K=131071 with all -128 x -128 inputs accumulates exactly 2147467264
without wrapping and is admitted; K=131072 is rejected at admission.

## Changed files

- `compute/gemmv1/types.go` (MaxSafeK fix), `compute/gemmv1/task_test.go`
- `compute/gemmv1/verify/`: profile, randomness (CSPRNG bound to an
  immutable commit; seeded sources development-only), Freivalds scalar +
  batch (independent rounds), residual, row locator, availability
  descriptor with DATA_UNAVAILABLE / OUTPUT_DATA_COMMITMENT_MISMATCH
- `compute/gemmv1/python/gemmv1/`: freivalds.py, row_locator.py,
  availability.py, gen_freivalds_vectors.py, e2e.py fast path
- `compute/gemmv1/testdata/freivalds_vectors.json` (cross-language H)
- `cmd/prisma-gemm/`: verify-fast, bench-verify
- `compute/gemmv1/python/tests/`: test_freivalds.py, test_e2e_fast.py
- `docs/`: this report, gemm-verification-v0.1.2.md, delivery gate,
  gemm-v0.1.2-benchmark-results.json

## Test results (all PASS unless noted)

- Baseline: full Go suite + Python suite green before changes; nothing
  from v0.1.1 removed or weakened.
- Test A honest C passes (scalar + batch, CSPRNG and deterministic
  streams): PASS.
- Test B single-element fraud: detected, localized to row 5 / col 9 /
  tile (0,1), v0.1.1 dispute ends ChallengerWins: PASS.
- Test C whole-tile fraud: full fast chain to ChallengerWins: PASS.
- Test D false challenge on honest C: rejected (localization finds no
  differing tile; honest C passes detection): PASS.
- Test E data unavailable: cannot finalize optimistic; refund route:
  PASS.
- Test F commitment mismatch: OUTPUT_DATA_COMMITMENT_MISMATCH, rejected:
  PASS.
- Test G randomness timing: challenge randomness cannot be constructed
  without an immutable commit: PASS.
- Test H cross-language: Go and Python reproduce the generated vectors
  (R stream, X/Y/Z per round, residual rows, localization)
  bit-for-bit: PASS.
- Test I multiple errors across rows/columns: one fraud path found and
  proven: PASS.
- Test J MaxSafeK boundary: 131071 admitted (worst case exact), 131072
  rejected: PASS.
- Batch vs scalar agreement: PASS (after fixing a real batch bug where
  all rounds reused one vector).
- E2E fast-honest / fast-fraud with instrumentation: the fast path makes
  ZERO calls to the full reference GEMM while detecting fraud and winning
  the deterministic dispute: PASS. The v0.1.1 full-verification path
  still works unchanged: PASS.

## False-accept simulation (empirical, deterministic trial seeds)

| rounds | trials | accepts | empirical | theoretical bound |
| --- | --- | --- | --- | --- |
| 1 | 4000 | 1988 | 0.497 | 0.5 |
| 2 | 4000 | 1030 | 0.2575 | 0.25 |
| 4 | 4000 | 244 | 0.061 | 0.0625 |
| 8 | 20000 | 81 | 0.00405 | 0.00391 |
| 16 | 50000 | 1 | 0.00002 | 1.53e-05 |

Observed rates are implementation evidence, not a mathematical proof; the
2^-rounds bound remains the guarantee.

## Benchmark ladder (CPU exact int64, one machine)

Full data: `docs/gemm-v0.1.2-benchmark-results.json`.

| Shape | Full recompute | Freivalds 8r (ratio) | 40r (ratio) | Row+tile localization | Dispute | Total fraud path (40r) |
| --- | --- | --- | --- | --- | --- | --- |
| 512^3 | 89 ms | 7 ms (0.079) | 38 ms (0.433) | 2.0 ms | 8.0 ms (6r) | 0.546 |
| 1024^3 | 723 ms | 32 ms (0.044) | 159 ms (0.220) | 6.0 ms | 30.6 ms (7r) | 0.271 |
| 2048^3 | 5756 ms | 114 ms (0.020) | 767 ms (0.133) | 25.5 ms | 135.0 ms (8r) | 0.161 |
| 4096^3 | 58718 ms | 471 ms (0.0080) | 3446 ms (0.0587) | 114.1 ms | 502.1 ms (9r) | 0.0692 |

- DetectionRatio at 4096^3: 0.0080 (8 rounds) to 0.0587 (40 rounds).
- TotalFraudPathRatio at 4096^3: 0.0692 at 40 rounds, 0.0169 at 8 rounds;
  both far below the 0.10 target. Fixed dispute costs dominate at small
  shapes, which is expected and honest.
- On-demand trace per disputed tile at 4096^3: 2 x (513 x 256) = 262 KB
  total for both parties; bisection 9 rounds.
- Peak memory at 4096^3: about 6.5 GB (matrices, C and tile structures of
  the reference path).
- GPU verifier: NOT TESTED this round (pods released); a GPU path must be
  exact integer arithmetic, never FP32/TF32.

## Known limitations

- The challenger still downloads the full committed C (O(n^2) data) and
  rebuilds its Merkle tree; availability of C within the challenge window
  is a new protocol requirement enforced via the refund route.
- Detection rounds are CSPRNG challenger-local; a production coordinator
  policy for rounds and bonds is a Phase D decision.
- The row/tile localization assumes the Freivalds residual row contains a
  real error; if the residual row happens to verify clean, the challenger
  must fall back to another residual row or more rounds (handled by
  LocalizeFromRows returning found=false, not by a false challenge).
- GPU acceleration of the verifier is unmeasured.

## Next blocker

Phase D: chain integration of ChallengeOpen, dispute state with bonds and
deadlines, and VWR settlement. The verification primitive it needs is now
cheap enough to be watcher-friendly.

## Final answers (§32)

### Q1 — Can the challenger detect an incorrect GEMM without recomputing it?

**YES.** Evidence: the fast path detects single-element, whole-tile and
multi-error fraud at 512^3-4096^3 with zero calls to the full reference
GEMM (instrumented, `test_e2e_fast`), with measured detection cost of
0.8%-5.9% of full recomputation at 4096^3 for 8-40 rounds, and an
empirical false-accept rate matching the 2^-rounds bound.

### Q2 — After probabilistic detection, can Prisma deterministically locate and prove an exact invalid micro-step?

**YES.** Evidence: bad row -> exact O(KN) row recomputation -> bad
column -> bad tile -> the unchanged v0.1.1 ChallengeOpen, on-demand tile
trace, bisection and 512-MAC arbitration end with ChallengerWins and no
worker receipt, on the CPU library, the CLI and the instrumented E2E
(including the earlier two-GPU RunPod run for the underlying dispute).

### Q3 — Is TotalFraudPathRatio < 0.10 for 4096^3?

**YES — 0.0692 measured** at 40 rounds (3446 ms Freivalds + 114 ms
row/tile localization + 502 ms dispute against a 58718 ms full
recomputation), and 0.0169 at 8 rounds. Real benchmark numbers on one
machine, not complexity estimates.
