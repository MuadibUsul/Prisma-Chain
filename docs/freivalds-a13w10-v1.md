# FREIVALDS_A13W10_I64_V1 — detection-only verification of wide GEMMs

```
r in {0,1}^N   (fresh randomness per round, per GEMM node)
x = W r        exact int64
y = A x        exact int64
z = C r        exact int64
check y == z
```

- **Detection only.**  A mismatch NEVER slashes by itself; it triggers the
  deterministic dispute path (graph bisection -> typed node arbitration ->
  wide 512-MAC arbiter for GEMM nodes).  Slashing requires a
  deterministic fraud proof.
- **Rounds.** Production round count is 40 (per-GEMM false accept <=
  2^-40); with the 83 real GEMM nodes the union bound is
  `83 * 2^-40 < 2^-33.6`.  The 2-round bundle-test parameter used in F.1
  is explicitly NOT a production security parameter.
- **Post-commit randomness.**  The worker's commitments (CommitV3 with the
  manifest root) are locked on chain BEFORE any randomness exists.  The
  watcher derives each round's bits from
  `SHA256("PRISMA_FREIVALDS_RANDOM_V1\0" || manifest_root || task_ref ||
  u32be(round) || nonce || CSPRNG entropy)` stretched to N bits —
  `H(manifest_root)` alone is forbidden as the only source, and a worker
  cannot predict the bits before committing.  A deterministic
  `--test-seed` mode exists and is labelled TEST ONLY in every artifact.
- **Operand coverage.** A13 x W10 and A13 x A13 (attention inner
  products), `transpose_b` 0/1.  Intermediates are asserted to stay
  inside signed int64: the per-shape theoretical bound
  (`a-1 + ceil(log2 k) + (w-1) + ceil(log2 n) + 1` signed bits) is
  recomputed from the admitted shape for every node; all six real shapes
  fit with margin (measured <= 46 bits, F.5A).
- **Cost.** The check is matrix-vector work — never a full GEMM
  recomputation.  The watcher instruments `full_gemm_calls = 0`, which the
  evidence suite asserts (83 GEMMs x 40 rounds in ~1.1 s on the real
  39.6 MB bundle; 227 cheap nodes recomputed exactly in ~3.2 s).
- **Localization (post-detection, off-chain).** The residual exposes the
  mismatching rows directly; the watcher then recomputes exactly ONE
  disputed row to find the bad column and maps it to the canonical 8x8
  output tile that the on-chain wide dispute will adjudicate.
