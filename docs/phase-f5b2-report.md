# Phase F.5B.2 report — Full Canonical GPU Block Exactness Gate

# FUSED_GPU_FEASIBLE_A13W10

The complete real 310-node Qwen3 layer0 canonical block (A13W10 frozen
policy, GraphID `cdad39513b76ab3dac07aed9014bcd6bd794af7f54cee5a17da0d3a62547bfb0`,
PolicyID `eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0`)
executes **bit-exactly on real NVIDIA GPUs**: every one of the 310 node
outputs and every one of the 310 wide validation roots is identical to the
frozen F.5A CPU executor's manifest on **both** an A40 (SM86) and an
RTX 2000 Ada (SM89), the final output bytes and root match on both, the
CPU-arithmetic-fallback count is **0**, and the canonical float-op audit is
**0**.  Together with F.5B.1's GEMM performance PASS (TRUE_FUSED_MMA
weighted 1.17x vs the fastest native INT8 baseline on both architectures,
frozen 6x/10x gates), the predeclared verdict of this phase is:

**FUSED_GPU_FEASIBLE_A13W10** — Phase F's GPU blocker is formally closed;
Phase F.5C (formal wide-integer protocol integration) is now the next
phase per the F.5B.1 stop rule.

Branch `research/transformer-phase-f5b2-full-block-gpu` (from F.5B.1 step
5).  No protocol files touched (`chain/`, `compute/canonical/`,
`compute/gemmv1/`, `proto/` unchanged); all work in `gpu/f5b2/`,
`tools/f5b2_*`, `deploy/runpod_f5b2/`, `testdata/f5b2_*`, `docs/`.

## How the CPU reference was produced (frozen executor, no new arithmetic)

`tools/f5b2_dump_block_program.py` runs the FROZEN F.5A executor
(`run_block_bits`) on F.5A heldout case 0 (real tokenizer input, committed
in `qwen3_f5a_heldout.json`) with PURE-CAPTURE source insertions — no
arithmetic statement is modified, and the hooked module's final output is
asserted bit-identical to the pristine module's.  All 310 canonical
operator invocations are captured (op, params, inputs, output) and mapped
onto the converted real-block graph positionally with a full
310-node operator-sequence assertion (the F.5B GEMM-dump precedent
generalized to all operators).  Three order reconciliations between the
research pipeline and the converted graph are documented and asserted
saturation-free (`fx_saturations == 0`): head-0's add-to-zero is folded
away (ofx is already clamped), the 15 head-attention accumulation ADDs are
deferred to the graph's head_sum chain (same left fold in head order), and
SILU sits after the gate requant in the graph vs before the up GEMM in the
pipeline (elementwise).  The committed artifacts are
`testdata/f5b2_block_program.json` (GraphID + PolicyID + 310 nodes),
`f5b2_block_bundle.npz` (58 consts incl. all W10 weight tensors — pods
need no model), `f5b2_cpu_manifest.npz`, `f5b2_cpu_roots.json` (wide
roots), and `f5b2_adversarial_vectors.json`.  A numpy interpreter of the
committed program reproduces the manifest **310/310 incl. every root**
before any pod time was spent (program self-containment proof).

## The GPU side

`gpu/f5b2/f5b2_kernels.cu` ports every canonical operator to exact
integer CUDA (ties-to-even shifts, saturating fixed-point ops, the frozen
exp with trunc-k/Taylor/saturation branches, the frozen invsqrt with the
pinned 16-entry table and exactly 4 Newton steps, index-order
deterministic reductions, ties-even softmax division), concatenated at
build time with the F.5B.1 fused A13W10 m16n8k32 kernel as the GEMM
backend (TRUE_FUSED_MMA_A13W10 per the F.5B.1 decision; CUDA-graph
replay stays a diagnostic).  Attention k/v chains are dynamic GEMM B
operands packed at run time with the same exact integer split; weight
consts are prepacked once.  `f5b2_block.py` executes all 310 nodes with
GPU-resident intermediates and NO CPU fallback path (unknown operators
abort with UNSUPPORTED_GPU_OPERATOR).

## The two GPU runs (identical artifacts, runner, GraphID, PolicyID)

| gate | A40 (SM86) | RTX 2000 Ada (SM89) |
|---|---|---|
| operator vectors (canonical 11 + adversarial 17 + wide 5) | **PASS** | **PASS** |
| GEMM regression (83 real nodes, fused kernel) | **83/83** | **83/83** |
| full block nodes exact | **310/310** | **310/310** |
| wide roots exact | **310/310** | **310/310** |
| final output bytes + root exact | **yes** | **yes** |
| cpu_arithmetic_nodes / unsupported | **0 / 0** | **0 / 0** |
| canonical float ops (static source audit) | **0** | **0** |
| wall ms (recorded, no gate) | 52.7 (46.3 GEMM) | 50.6 (47.0 GEMM) |

Per-operator exactness (both GPUs identical): RMSNORM 26/26, REQUANTIZE
126/126, GEMM 83/83, ROPE 24/24, ADD 33/33, SOFTMAX 16/16, SILU 1/1,
MUL 1/1.  Result files were downloaded byte-exact (md5-verified) and the
cross-GPU verdict was produced mechanically by `--compare`
(`docs/phase-f5b2-cross-gpu.json`).

## Report questions

**Q1.** Can every operator in the real 310-node block execute on GPU with
deterministic integer arithmetic? **YES** (all 8 operator types, all 310
nodes, both architectures).

**Q2.** Does the GPU implementation exactly match the frozen CPU canonical
executor? **YES** (byte + wide-root equality per node).

**Q3.** How many nodes match? **310 / 310** on both GPUs.

**Q4.** How many (wide) TensorRoots match? **310 / 310** on both GPUs.

**Q5.** Does TRUE_FUSED_MMA_A13W10 remain exact on all 83 real GEMM
nodes? **YES** (regression re-run on both GPUs: 83/83, plus the same
kernel inside the block run).

**Q6.** Are RMSNorm and Softmax deterministic across SM86 and SM89?
**YES** — sequential index-order integer reductions (no atomics; integer
addition is associative), identical results on both architectures
(`docs/phase-f5b2-reduction-proof.json`).

**Q7.** Was any CPU arithmetic fallback used? **NO** (`cpu_arithmetic_
nodes = 0` on both; no fallback path exists in the executor).

**Q8.** Was any floating-point canonical arithmetic used? **NO** (static
source audit zero hits; integer kernels + int8 tensor-core MMA only).

**Q9.** Do CPU, GPU SM89, and GPU SM86 produce identical final bytes and
root? **YES** (final root `1a7b9c7d2d2f361e…` from
`f5b2_cpu_roots.json`; both GPUs byte-equal to the CPU manifest's final
output and root).

**Q10.** Should Prisma proceed to Phase F.5C — Formal Wide-Integer
Protocol Integration? **YES** — the predeclared condition
(`FUSED_GPU_FEASIBLE_A13W10`) is met.

## Data list

docs/phase-f5b2-{gpu-results.nvidia_a40,
gpu-results.nvidia_rtx_2000_ada_generation, cross-gpu, real-block-gpu,
operator-results, reduction-proof, gpu-results.cpudry}.json,
docs/phase-f5b2-report.md, testdata/f5b2_{block_program.json,
block_bundle.npz, cpu_manifest.npz, cpu_roots.json,
adversarial_vectors.json}, gpu/f5b2/{f5b2_kernels.cu, f5b2_ext.py,
f5b2_block.py, f5b2_runner.py}, tools/{f5b2_dump_block_program.py,
f5b2_validate_program.py}, deploy/runpod_f5b2/run_f5b2_jupyter.py.

## Honest statement

The CPU reference is the frozen F.5A executor with observation-only
instrumentation (bit-identity anchored); the GPU verdict compares against
that manifest, not against GPU-written expectations.  Root comparison and
hashing run on the CPU harness after execution, which the spec explicitly
allows (sections 6/31); no operator result ever touched the CPU.  Wall
timings are recorded without a gate per sections 55/89.  The three
pipeline/graph order reconciliations are documented and asserted
saturation-free.  Nothing was retuned to make the block pass: the frozen
arithmetic, thresholds, and graph identity are unchanged, and zero
protocol files were touched.
