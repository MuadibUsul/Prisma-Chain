# Phase F.5B report — Exact Wide-Integer GPU Feasibility (A13W10)

# GPU_BACKEND = NOT TESTED

**Current maximum claimable state.** The entire CPU-side program is
complete and PASSES: the radix-128 decomposition proof, the 83-note
GEMM evidence dump, the three exact GPU backends, and a full CPU
dry-run that exercises the runner end to end (all vectors exact, all
83 nodes exact on all three backends, schedule ratio 3.31). The GPU
verdict itself — bit-exactness on real NVIDIA hardware, cross-GPU
agreement, and the performance gate — **requires two GPU runs that
have not happened yet**. Until then this phase claims exactly
GPU_BACKEND = NOT TESTED; the CPU dry-run is harness validation only
and is excluded from any verdict (file name `docs/phase-f5b-harness-cpudry.json`,
environment field marks it a DRY RUN).

Branch `research/transformer-phase-f5b-gpu-feasibility` (from F.5A
step 10, 959b701). `git diff origin/research/transformer-phase-f5a-int64-frontier
-- chain compute proto` is empty: **zero protocol diff**. No
threshold was changed; CanonicalMathV1 untouched; no wide-integer
protocol code; no full-block GPU work (gated).

## Where F.5B stands

| item | status |
|---|---|
| F.5A evidence normalization (mechanically derived, hashed) | **DONE** — `docs/phase-f5a-verdict-final.json` |
| Balanced radix-128 split proof (exhaustive A13/W10 values) | **PASS** — CPU |
| Schoolbook + Karatsuba3 exactness vs direct int64 oracle | **PASS** — CPU, 10 shapes × adversarial patterns |
| Theoretical requant int64 safety (all requant links) | **PASS** — THEORETICAL_REQUANT_INT64_SAFE, worst product 55 bits |
| Real 83-node GEMM evidence dump (inputs + CPU-correct outputs) | **DONE** — `testdata/f5b_gemm_nodes.npz` |
| Three exact GPU backends (SIMT ref, TC Karatsuba3, DP2A W-split2) | **WRITTEN**, no float in canonical path (source audit); runtime audit pending |
| Harness end-to-end validation | **PASS** — CPU dry-run, recorded as harness validation only |
| Bit-exactness on real NVIDIA GPUs | **NOT TESTED** — requires hardware |
| Cross-GPU agreement (≥2 different models / compute capabilities) | **NOT TESTED** |
| Performance gates (≤6× weighted schedule; >5%-MAC shapes ≤10×) | **NOT TESTED** on GPU |
| Full 310-node block on GPU | **NOT STARTED** — gated on GEMM performance PASS (§62) |

## 1. F.5A evidence normalization (commit 1)

`docs/phase-f5a-verdict-final.json` is derived mechanically (script,
not hand-typed) from four frozen F.5A artifacts, each recorded with its
SHA256: winner **A13W10**, worst cosine **0.9999123540730438**, worst
max_abs **0.038933493885028536**, count(>0.05)=0, MaxSafeK32 1023,
MaxSafeK64 4398046511103, int64 accumulation / requant / Freivalds
flags true, PolicyID `eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0`,
family_sha256 4b0ab6d5e0c69402, final verdict PASS. The F.5A report got
an addendum pointing at the normalization; the original heldout file is
preserved unmodified and the stale `max_safe_k64` field in it is
explicitly superseded.

## 2. Radix-128 decomposition — CPU proof (PASS)

Balanced split emitted on hardware as signed int8 tiles:
`q = floor((x + 64) / 128)`, `lo = x − 128q`.

- **Exhaustive split proof**: every A13 value (8192) and every W10
  value (1024) splits with exact reconstruction and digit ranges
  A0∈[−64,63], A1∈[−32,32], W0∈[−64,63], W1∈[−4,4]; sums
  A0+A1∈[−96,95], W0+W1∈[−68,67]. All products fit int8×int8→int32.
- **Exactness**: schoolbook (C00, C01, C10, C11) and **Karatsuba3**
  (Cross = Csum − C00 − C11, C = C00 + 128·Cross + 16384·C11) are
  bit-identical to the direct int64 oracle on all shapes, including
  the real ones with K = 16, 128, 1024, 3072, and on adversarial
  patterns (all-max, all-min, mixed extremes, alternating, zeros,
  one-hot rows, random). Partial bounds
  C00 ≤ 12,582,912, C11 ≤ 393,216, Csum ≤ 20,054,016 — int32-safe;
  final merge in int64.
- **Requant**: per-link theoretical analysis over full signed
  magnitudes gives **THEORETICAL_REQUANT_INT64_SAFE**, worst product
  55 bits ≪ 62 (the assertion budget in the hardware requant).
- 83 real GEMM node shapes extracted from the converted graph
  (`docs/phase-f5b-gemm-shapes.json`); wide vectors frozen in
  `testdata/f5b_wide_gemm_vectors.json` (boundary all-max, all-min,
  small random, transpose_b, k=1).

## 3. Real 83-node evidence (DONE)

`tools/f5b_dump_gemm_nodes.py` reruns the frozen F.5A executor on the
heldout case with a GEMM dump hook and maps the 83 pipeline GEMM calls
one-to-one onto graph GEMM node ids (assert 83 == 83). The dump
(`testdata/f5b_gemm_nodes.npz`, with meta json) stores, per node: the
exact int16 activation tile, int8/int16 weight tile, transpose flag,
and the CPU-correct int64 output. The hook in `tools/f5a_joint_precision.py`
was re-anchored bit-identical to F.4A (A12W9/A13W8 spot checks) before
any evidence was produced.

## 4. GPU backends (WRITTEN; runtime NOT TESTED)

All three produce the exact same int64 result; selected by
benchmarking on the pods:

- `GPU_SIMT_INT64_REFERENCE` — direct int64 integer matmul in chunks
  (ground truth on hardware, no decomposition).
- `GPU_TC_KARATSUBA3` — three s8×s8→s32 tensor-core GEMMs
  (`torch._int_mm`, i.e. cuBLASLt integer path, **no float anywhere**):
  C00 = A0·W0, C11 = A1·W1, Csum = Asum·Wsum, Cross = Csum − C00 − C11,
  then C = C00 + (Cross << 7) + (C11 << 14) in int64. Physical 3× MAC
  (protocol counts logical MAC only).
- `GPU_DP2A_W_SPLIT2` — int16×int8 integer dot (the dp2a style split
  on the weight side; not tensor core).

`requant_wide` executes `round(acc·mult >> shift)` with a hard
`acc_bits + m.bit_length() ≤ 62` assertion (justified by §2's 55-bit
theoretical bound), never a float approximation. `source_float_audit()`
reports **FLOAT_OPS_USED = 0** for the canonical path; a runtime audit
accompanies every GPU run.

## 5. Harness validation (CPU dry-run — NOT evidence)

`python gpu/f5b/f5b_gpu_runner.py --device cpu --out docs/phase-f5b-harness-cpudry.json`:

- vectors: all exact (all three backends vs the direct CPU oracle);
- nodes: 83/83 exact, 0 mismatches, for **each of the three backends**
  (CPU torch._int_mm availability lets the full Karatsuba3 path run);
- weighted 83-node schedule ratio vs native int8 baseline: **3.307**
  (the expected ≈3× physical cost; the gate constant is 6.0);
  `performance_pass: True` — of the harness logic, on CPU, nothing more;
- per-shape ratios, per-node medians, environment capture, error/WARN
  capture: all exercised.

This validates the harness end to end so the pod time is minimal, and
is explicitly **excluded from any GPU verdict**.

## 6. Runbook (what remains; needs the two GPUs)

On each pod (any CUDA PyTorch image, torch ≥ 2.4 for `torch._int_mm`;
**no model download needed** — the 83-node evidence and vectors are in
the repo):

```
git clone --depth 1 -b research/transformer-phase-f5b-gpu-feasibility \
    https://github.com/MuadibUsul/Prisma-Chain.git f5b && cd f5b
python gpu/f5b/f5b_gpu_runner.py --device cuda \
    --out docs/phase-f5b-gpu-results.<gpu>.json
```

Then, on any machine with the two result files (CPU is fine):

```
python gpu/f5b/f5b_gpu_runner.py --compare \
    docs/phase-f5b-gpu-results.<gpuA>.json docs/phase-f5b-gpu-results.<gpuB>.json
```

→ writes `docs/phase-f5b-final-verdict.json` with the cross-GPU
verdict. Verdict selection (predeclared): both GPUs exact (nodes +
vectors) and distinct GPU models required; if both performance gates
pass → **GPU_FEASIBLE_A13W10**; exact but not practical →
**GPU_CORRECT_BUT_NOT_PRACTICAL**; otherwise
**GPU_NOT_FEASIBLE_OR_INCONCLUSIVE**. If only one GPU is ever
available, the verdict is **INCONCLUSIVE_NO_SECOND_GPU** (recorded in
this report, not by the tool).

Two different NVIDIA models (ideally different compute capabilities,
e.g. the RTX 4000 Ada + RTX 2000 Ada pair used in the earlier GEMM
E2E) are required for the cross-architecture claim.

## 7. Report questions

**Q1.** Does a balanced radix-128 int8 decomposition exist for every
A13 and W10 value with exact reconstruction and provable digit
ranges? **YES** (exhaustive, all 8192 + 1024 values).

**Q2.** Are schoolbook and Karatsuba3 bit-identical to the direct
int64 product on the real node shapes and adversarial patterns?
**YES** — CPU proof; on real NVIDIA GPUs **NOT TESTED**.

**Q3.** Are all 83 real GEMM nodes captured as evidence and reproduced
exactly? Captured: **YES** (83/83). Reproduced: **YES on CPU**
(dry-run); on GPUs **NOT TESTED**.

**Q4.** Is requantization intermediate int64-safe on all requant
links? **YES theoretically** — THEORETICAL_REQUANT_INT64_SAFE, worst
product 55 bits; hardware assertion in place; runtime **NOT TESTED**.

**Q5.** Do the implemented GPU backends contain any float operation in
the canonical path? **NO by source audit** (FLOAT_OPS_USED = 0,
`torch._int_mm` integer path only); runtime audit accompanies each GPU
run and is **NOT TESTED**.

**Q6.** Does A13W10 arithmetic execute bit-identically to the CPU
oracle on real NVIDIA GPUs? **NOT TESTED** — harness ready; requires
the two GPU runs.

**Q7.** Do two different GPU models / compute capabilities agree?
**NOT TESTED** — requires the two GPU runs.

**Q8.** Performance: ≤6× over the weighted 83-node schedule and ≤10×
on shapes >5% MAC vs native INT8? **NOT TESTED** on GPU. Harness logic
validated (CPU dry-run ratio 3.31 — not evidence).

**Q9.** Was the full 310-node block executed on GPU? **NO — not
started**, gated on the GEMM performance PASS per the phase
specification.

**Q10.** Final verdict? **GPU_BACKEND = NOT TESTED.** CPU-side program
complete and PASS; the vocabulary is reserved for the two-GPU outcome:
GPU_FEASIBLE_A13W10 / GPU_CORRECT_BUT_NOT_PRACTICAL /
GPU_NOT_FEASIBLE / INCONCLUSIVE_NO_SECOND_GPU. No verdict is issued
without hardware evidence.

## Data list

docs/phase-f5b-{radix128-proof, gemm-shapes, harness-cpudry}.json,
docs/phase-f5a-verdict-final.json, testdata/f5b_{wide_gemm_vectors.json,
gemm_nodes.npz, gemm_nodes_meta.json}, gpu/f5b/f5b_gpu_{backends,runner}.py,
tools/{f5b_radix128.py, f5b_dump_gemm_nodes.py}, f5b_proof.log, this
report. Pending (not created): docs/phase-f5b-gpu-results.<gpu>.json ×2
and docs/phase-f5b-final-verdict.json.

## Honest statement

No GPU result is claimed, fabricated, or extrapolated: every GPU field
above says NOT TESTED and the only executed runs are on CPU, labeled
as harness validation. The gate constants (6.0 / 10.0 / 0.05), the
evidence, the frozen F.5A family and the protocol are untouched
(zero-diff verified against the F.5A branch over chain/compute/proto).
The pods' results will be pasted verbatim into
`docs/phase-f5b-gpu-results.<gpu>.json` exactly as the runner writes
them — no post-editing of numbers.
