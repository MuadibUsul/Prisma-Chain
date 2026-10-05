# Prisma GEMM Verification v0.1.2

This document records the implemented v0.1.2 verification layer: Freivalds
probabilistic detection plus exact bad-row/bad-tile localization, reusing
the v0.1.1 deterministic tile dispute unchanged. Implementation:
`compute/gemmv1/verify/` (Go) and `compute/gemmv1/python/gemmv1/`
(`freivalds.py`, `row_locator.py`, `availability.py`); CLI
`prisma-gemm verify-fast`; E2E fast-honest / fast-fraud.

## What Freivalds is — and is not

**Freivalds detects a likely incorrect result; the deterministic tile
dispute proves the exact invalid computation step.**

- Detection (this document): probabilistic, O(MK + NK + MN) per round,
  runs on the challenger without ever recomputing A x B. A mismatch is a
  reason to open a challenge, never a slashing proof.
- Adjudication (v0.1.1, unchanged): a specific bad tile, an on-demand
  trace, O(log K) bisection and a 512-MAC micro-step arbitration. This is
  the sole final evidence, and the arbiter still never recomputes the task.

No worker is slashed because of a Freivalds mismatch, and a Freivalds
mismatch alone never settles a task.

## FREIVALDS_BINARY_V1

For A: int8[M,K], B: int8[K,N], C: int32[M,N] and a binary vector
r in {0,1}^N:

```
x = B x r      (K, int64)
y = A x x      (M, int64)
z = C x r      (M, int64)
mismatch  <=>  y != z
```

q rounds use q independent vectors. For a fixed incorrect C each
independent round detects with probability >= 1/2, so
`theoretical_false_accept_upper_bound = 2^-rounds`. This bounds the
DETECTION layer only; it is not the overall attack success probability of
the system, which is governed by the deterministic dispute that a
fraudulent worker must also lose.

### Arithmetic model

All arithmetic is exact signed int64 with no wraparound. The bounds, proved
before execution by `Profile.AdmitFor`:

```
|x_t| <= 128 * N
|y_i| <= 128 * 128 * N * K = 16384 * N * K
|z_i| <= N * (2^31 - 1)            (C is canonical int32)
|y_i - z_i| <= 16384*N*K + N*(2^31-1) < 2^62
```

Shapes violating the bound are rejected at verification-profile admission;
there is no float fallback, no FP32/FP64/TF32, and no silent overflow in
either implementation. Python asserts the int64 range explicitly on every
accumulated element.

### Randomness

- Challenge randomness is produced only after the worker's ResultCommit
  and output_root are immutable (`NewChallengeRandomness` requires the
  committed binding and refuses empty ones).
- Production sources are CSPRNG (`crypto/rand` / `secrets`).
- Seeded sources exist only for tests and cross-language vectors and are
  marked development-only; they can never back a production challenge.
- A worker cannot see the challenge randomness before its output_root is
  locked, so it cannot tailor an output to pass.

### Batch mode

q rounds run as one batch: R in {0,1}^{N x q}, X = B x R, Y = A x X,
Z = C x R. Rounds are INDEPENDENT vectors (a batch bug that reused one
vector across rounds was found by the tests and fixed on both sides).
Batch and scalar must agree exactly; this is tested.

## Output data availability

Freivalds verifies the matrix C the challenger received, so the received C
must first be proven to be the worker's committed output:

1. The v0.1.1 `ResultCommit` is frozen; availability is a separate
   versioned object, `OutputAvailabilityDescriptor` (version `0.1.2`,
   domain `PRISMA_GEMM_OUTPUT_AVAILABILITY_V1`), carrying `output_root`,
   `output_bytes` and `output_data_ref`.
2. Before verification the challenger rebuilds the output Merkle tree over
   the received C and compares with `output_root`. Mismatch:
   `OUTPUT_DATA_COMMITMENT_MISMATCH` — the data is not the committed
   output and nothing proceeds on top of it.
3. If the worker cannot serve C in the challenge window: `DATA_UNAVAILABLE`
   — the task MUST NOT finalize as `optimistic_unchallenged`; it routes to
   the refund path. Bond penalties remain a Phase D decision.

## Bad-row localization and tile evidence

After a mismatch (D = Y - Z):

1. Pick a row with D[row,*] != 0 — that output row contains at least one
   wrong element with overwhelming probability for that round.
2. Recompute exactly that row: `expected_row = A[row,:] x B`, O(KN),
   exact int32 under the task admission bound. `ReferenceRowGEMM` never
   touches the rest of C.
3. Compare columns to find a bad column; map to the 8x8 tile
   (row/8, col/8).
4. The challenger tile presented to `ChallengeOpen` is built from 8 exact
   row recomputations (8 x O(KN)) — never from the worker's data and never
   from a full GEMM.

From there the v0.1.1 flow takes over unchanged: ChallengeOpen validation,
on-demand single-tile trace, O(log K) bisection, 8x8x8 arbitration with
512 canonical MACs, WorkerWins / ChallengerWins / BothInvalid.

## False-accept expectation

The empirical false-accept simulation (deterministic per-trial seeds,
single-element fraud) matches the theory:

| rounds | trials | accepts | empirical | theoretical bound |
| --- | --- | --- | --- | --- |
| 1 | 4000 | 1988 | 0.497 | 0.5 |
| 2 | 4000 | 1030 | 0.2575 | 0.25 |
| 4 | 4000 | 244 | 0.061 | 0.0625 |
| 8 | 20000 | 81 | 0.00405 | 0.00391 |
| 16 | 50000 | 1 | 0.00002 | 1.53e-05 |

Observed rates are evidence about the implementation, not a replacement
for the bound.

## GPU path

The v0.1.2 GPU verifier was NOT run this round (the two GPU pods from
Phase C are released). The exact-int64 CPU path is the reference; a GPU
accelerator must use an exact integer path (int64 matmul or int32-safe
chunking with exact accumulation) and is forbidden from FP32/TF32
approximations. When two pods are next available, the priority is the
4096^3 fast-verification benchmark, not a repeat of the v0.1.1 bit-exact
proof.
