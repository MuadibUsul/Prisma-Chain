# Phase F.5B report — Exact Wide-Integer GPU Feasibility (A13W10)

# GPU_CORRECT_BUT_NOT_PRACTICAL

A13W10 canonical arithmetic **executes exactly on real NVIDIA GPUs** — both
frozen tensor-core strategies (Karatsuba3 and Schoolbook4) are bit-identical
to the CPU int64 oracle on **all 83 real GEMM nodes** and all wide vectors,
on **two different GPU architectures** (A40, cc 8.6 / RTX 4000 Ada, cc 8.9).
The **predeclared performance gate FAILS** on both: the weighted 83-node
schedule runs at **10.51×** (A40) and **12.77×** (RTX 4000 Ada) vs the
frozen 6× limit, with major shapes up to **15.18×** vs the frozen 10× limit.
The cost driver is the eager-PyTorch kernel count per GEMM node (activation
split + 3–4 int8 GEMMs + 6-kernel int64 merge ≈ 15–20 launches vs the
native baseline's single `torch._int_mm` call), not the 3× physical MAC of
the decomposition. No threshold was adjusted. The full 310-node GPU block
remains **NOT STARTED** (gated on the performance PASS). Zero protocol
diff; the protocol work accounting stays 1× logical MAC.

Branch `research/transformer-phase-f5b-gpu-feasibility` (steps 1–8, head
`step 8`). `git diff origin/research/transformer-phase-f5a-int64-frontier
-- chain compute proto` is empty. One protocol-adjacent change: none.

## The two GPU runs (round 2, fixed harness)

| | A40 (cc 8.6, Ampere) | RTX 4000 Ada (cc 8.9, Ada) |
|---|---|---|
| torch / CUDA / driver | 2.8.0+cu128 / 12.8 / 580.159.04 | 2.8.0+cu128 / 12.8 / 595.91.07 |
| Karatsuba3 — 83 nodes exact | **83/83** | **83/83** |
| Schoolbook4 — 83 nodes exact | **83/83** | **83/83** |
| wide vectors exact | **yes (5/5)** | **yes (5/5)** |
| SIMT / DP2A direct matmul | platform-limited (below) | platform-limited (below) |
| weighted schedule ratio (K3) | **10.51×** | **12.77×** |
| weighted schedule ratio (SB4) | 11.41× | 13.31× |
| major shapes (≥5% MAC) K3 | 7.26 / 14.15 / 7.63 / 5.24 | 10.03 / 15.18 / 9.92 / 5.80 |
| performance gate (6× / 10×) | **FAIL** | **FAIL** |
| float ops in canonical path | 0 | 0 |

Results downloaded byte-exact over the terminal relay (chunked base64,
md5-verified); `docs/phase-f5b-final-verdict.json` is produced by the
`--compare` step from exactly these two files.

## CUDA platform facts the CPU dry-run could not see (found live, fixed)

The first pod run (also on both GPUs) produced 0/83 everywhere and the
verbatim errors below; they are now first-class evidence, captured in every
result file (`cuda_probe`) and in the per-backend `first_error` fields:

1. **`NotImplementedError: "addmm_cuda" not implemented for 'Long'`** —
   torch has **no generic integer matmul on CUDA**. The direct
   `GPU_SIMT_INT64_REFERENCE` and `GPU_DP2A_W_SPLIT2` paths are
   CPU-executable only (they still run in the CPU proof and dry-run); on
   GPU their cross-check role is filled by `GPU_SCHOOLBOOK4`, an
   independent merge formula (C00 + (C01+C10)<<7 + C11<<14) over the same
   radix-128 split.
2. **`torch._int_mm` requires the left operand M > 16** — probed on both
   GPUs: M=16 → "needs to be greater than 16"; M=17/24/32 → ok. Every real
   block GEMM has M = 16 (seq length), so all tensor-core calls now
   zero-pad **by role** (M → max(32, align16), K/N → max(16, align16)).
   Zero padding is mathematically inert (exactness preserved) and is
   applied identically to the native int8 baseline, so ratios stay fair.
3. **The performance gate could pass vacuously** when every timed shape
   errored (0/0 ratio → "≤ 6×"). The gate now requires exactness on the
   CUDA-executable backends **and** every shape measured **and** positive
   totals **and** the ratio limits. (Integrity fix — a gate that can pass on
   zero measurements is a fake gate.)

Fix commit: `950c6db` (F.5B step 7). The CPU dry-run was re-verified green
after the fix (4 backends × 83 nodes exact, ratio 3.14).

## Why the performance gate fails (measured, not speculated)

Native int8 baseline totals (schedule-weighted): 3.08 ms (A40) / 3.29 ms
(Ada); wide totals: 32.41 ms / 41.97 ms. Per-occurrence times are tiny
(native 0.021–0.130 ms; wide 0.15–0.75 ms), so **neither side is at MAC
throughput**: at M=16-padded-to-32, a single cuBLASLt int8 GEMM already
runs at a small fraction of peak, and the wide path multiplies that by
strategy overhead. Per GEMM node the wide path executes ~15–20 kernels
(activation split ≈ 8 elementwise ops, 3–4 `_int_mm`, merge ≈ 6 int64
elementwise ops) against the baseline's 1 — the ratio tracks kernel count,
not arithmetic. Schoolbook4 (4 GEMMs) is ~5–8% slower than Karatsuba3, so
the failure is **overhead-dominated, not strategy-specific**. Event-based
per-iteration timing adds the same absolute overhead to both sides, which
biases the measured ratio *down*: the true wide/native cost is at least
what is reported.

The obvious path to ≤6× is **fusion** — a single kernel (or compiled graph)
doing split + int8 tensor-core products + int64 merge without materializing
intermediates. That is future work, **NOT attempted and NOT tested here**;
this phase answers the feasibility question for the straightforward
composition of sanctioned primitives, and the answer for that realization
is: correct, but not practical.

## What was proven before the pods (CPU, unchanged from step 6)

Radix-128 split exhaustive proof (all 8192 A13 + 1024 W10 values, exact
reconstruction, digit ranges); schoolbook + Karatsuba3 bit-identical to the
direct int64 oracle on 10 shapes × adversarial patterns (K up to 3072);
int32 safety bounds for C00/C11/Csum/Cross and the int64 final bound;
THEORETICAL per-node requant verdict THEORETICAL_REQUANT_INT64_SAFE (worst
product 55 bits); the real 83-node shape manifest; frozen wide vectors
(CPU columns exact; GPU columns now measured). See commit history, steps
1–6, and `docs/phase-f5b-radix128-proof.json`.

## Report questions

**Q1.** Does A13W10 canonical arithmetic execute bit-identically to the
CPU int64 oracle on real NVIDIA GPUs? **YES** — both CUDA-executable
tensor-core strategies, all 83 nodes, all vectors, on both GPUs.

**Q2.** Two different GPU models / compute capabilities? **YES** — A40
(8.6) and RTX 4000 Ada (8.9); cross-architecture equality follows from
bit-exactness against the same CPU oracle (and each other, by transitivity
of equality on the identical expected values).

**Q3.** Which strategies are CUDA-executable? **Karatsuba3 (3 int8 GEMMs +
int64 merge) and Schoolbook4 (4 int8 GEMMs + independent merge)**;
direct int64/int16 matmul (`SIMT` reference, `DP2A` W-split) is not a
torch CUDA primitive — verbatim error recorded; those run on CPU only.

**Q4.** Requant intermediate int64 safety on hardware? Theoretical bound
55 bits (int64-safe) with a runtime assertion in `requant_wide`; the
requant kernel is deterministic integer code and was exercised on the CPU
paths. **GPU-exercised: NOT TESTED** in this harness run (GEMM arm focus).

**Q5.** Performance vs native INT8: ≤6× weighted, ≤10× major shapes?
**NO** — 10.51× / 12.77× weighted; majors up to 14.15× / 15.18×. The
frozen gate failed on both GPUs and is reported as a failure.

**Q6.** Is the failure strategy-specific? **NO** — Schoolbook4 is within
~5–8% of Karatsuba3; both are dominated by per-node kernel count in eager
PyTorch, not by the 3× physical MAC.

**Q7.** Was any threshold, gate or accepted input adjusted? **NO.** The
only harness changes were integrity/platform fixes (padding for a real
hardware constraint, non-vacuous gate, verbatim error capture).

**Q8.** Was the full 310-node block run on GPU? **NO — not started**
(gated on GEMM performance PASS per the phase specification).

**Q9.** What would plausibly bring the ratio under the gate? **Fusion**
(split + tensor-core + merge in one kernel / compiled graph). Not
attempted; no fusion claim is made.

**Q10.** Final verdict? **GPU_CORRECT_BUT_NOT_PRACTICAL.** The exactness
arm is proven on two architectures; the practicality arm fails the
predeclared 6×/10× gates for the eager realization. F.5C (protocol
integration) remains gated on GPU_FEASIBLE_A13W10 and is therefore **not
started**.

## Data list

docs/phase-f5b-gpu-results.nvidia_a40.json,
docs/phase-f5b-gpu-results.nvidia_rtx_4000_ada_generation.json (both
byte-verified on download), docs/phase-f5b-final-verdict.json,
docs/phase-f5b-{radix128-proof, gemm-shapes, harness-cpudry}.json,
docs/phase-f5a-verdict-final.json, testdata/f5b_{wide_gemm_vectors.json,
gemm_nodes.npz, gemm_nodes_meta.json}, gpu/f5b/f5b_gpu_{backends,runner}.py,
deploy/runpod_f5b/run_f5b_jupyter.py, tools/{f5b_radix128.py,
f5b_dump_gemm_nodes.py}, f5b_proof.log, this report.

## Honest statement

Everything above is measured on the stated hardware with the stated
versions; the per-GPU result files are exactly as the runner wrote them on
the pods (md5-verified in transit, compare output derived mechanically).
The failure of the performance gate is reported as a failure with the
mechanism identified from the data, not softened; the thresholds were not
moved. Two backends (SIMT, DP2A) could not run on CUDA hardware and are
explicitly excluded from the GPU verdict with their verbatim platform
errors preserved. The full-block GPU run and any fusion optimization are
NOT STARTED by the spec's own gating. No GPU result was fabricated,
extrapolated, or edited.
