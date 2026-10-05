# Canonical operators v1 (CANONICAL_GRAPH_V1)

The operator set of the verifiable Transformer block. Every operator is a
pure integer function with a static contract: `ValidateParams`,
`OutputSpec`, `Work`, `Execute`, `ArbiterBound`. There is no reflection,
no dynamic loading and no floating point in any consensus path; the
registry is fixed at build time in `compute/canonical/operator.go`.

Two implementations exist and must agree byte for byte:

* Go reference: `compute/canonical/` (the protocol definition);
* Python mirror: `compute/canonical/python/canonical_ref.py`, checked
  against the Go code by `compute/canonical/testdata/canonical_vectors.json`
  (`TestCanonicalCrossLanguageVectors` in Go,
  `python/test_canonical_ref.py` on the Python side).

## Operator table

| Operator | Version | Semantics (all Q12.20 unless noted) | Arbiter unit |
|---|---|---|---|
| `GEMM_INT8_V1` | 0.1.1 (frozen) | `C = A·B` or `C = A·Bᵀ` (see below), int8×int8→int32 accumulator, admission `K ≤ MaxSafeK = 131071` | full operand evidence + bounded chunk recompute |
| `ADD_FIXED_V1` | 1.0.0 | saturating `a + b` | 2×64 input elements |
| `MUL_FIXED_V1` | 1.0.0 | `(a·b) >> 20` ties-to-even | 2×64 input elements |
| `REQUANTIZE_V1` | 1.0.0 | `clamp(round_ties_even(x·mult >> shift), lo, hi)`; input accumulator or Q12.20; `out_dtype` 0 = Q12.20, 1 = int8 | 64 input elements |
| `RMSNORM_FIXED_V1` | 1.0.0 | `yᵢ = xᵢ·invsqrt(mean(x²)+eps)·wᵢ`, reduction chunk 16, pinned invsqrt | x rows + weight row (bounded) |
| `ROPE_FIXED_V1` | 1.0.0 | pair rotation with pinned (cos, sin) constants from the committed table | 64 input elements |
| `SILU_FIXED_V1` | 1.0.0 | `x·sigmoid(x)` with the frozen two-branch sigmoid | 64 input elements |
| `SOFTMAX_FIXED_V1` | 1.0.0 | row max → subtract → canonical exp → sum → exact division | disputed rows (bounded) |

There is no `SCALE` operator: the attention scale `1/√head_dim` is folded
into the REQUANTIZE parameters of the scores node by the converter, with
`AttentionScaleFx(head_dim) = floor(2²⁰/√head_dim)` computed in integer
arithmetic (no float anywhere in the pipeline).

## GEMM as a graph node

The frozen v0.1.1 arithmetic is reused through `gemmv1.ReferenceGEMM`;
nothing in `compute/gemmv1` is modified. The graph adapter adds exactly one
thing the static graph needs: the optional `transpose_b` parameter.

* `transpose_b = 0`: `A [M,K] int8`, `B [K,N] int8`, `C = A·B`.
* `transpose_b = 1`: `A [M,K] int8`, `B [N,K] int8`, `C[i,j] = Σ_d A[i,d]·B[j,d]`.

Attention needs `scores = Q·Kᵀ` and a static graph has no view or
transpose operator; the transpose is exact index arithmetic inside the
adapter and the micro-arithmetic stays byte-identical to v0.1.1
(int64 accumulation is bit-identical within the `MaxSafeK` admission).
A test compares the node output against the naive dot-product definition
for every transpose-B node of the block.

GEMM node arbitration recomputes the disputed output chunk from the FULL
committed operands (every operand chunk must be supplied in index order
with chunk proofs and state proofs). The v0.1.1 tile dispute remains the
specialized route for large standalone GEMM tasks; canonical block graphs
declare operand sizes that keep full-operand adjudication affordable, and
GraphGasV1 prices it by MACs.

## Work counters (no universal CWU)

Each operator reports its own per-unit counters (`ADD_ELEMENT`,
`GEMM_MAC`, `REQUANTIZE_ELEMENT`, `RMSNORM_ELEMENT`/`_REDUCTION`,
`ROPE_PAIR`, `SILU_ELEMENT`, `SOFTMAX_ELEMENT`/`_EXP`). There is
deliberately NO universal "canonical work unit": fusion pricing is an open
economic question, so settlement uses the agreed flat fee and the derived
vector only bounds execution gas. `GraphWorkVector(descriptor)` recomputes
the counters purely from the static descriptor, and a test proves it
equals the executed work vector for both block sizes — a worker can never
bias its own pricing.

## Dispute story per operator

A graph dispute (docs/canonical-graph-v1.md) bisects the on-demand state
trail to the FIRST divergent node, then dispatches to the bounded arbiter
of that node's operator:

* elementwise operators (ADD, MUL, REQUANTIZE, SILU, ROPE): one 64-element
  chunk, recomputed from evidence that must be leaves of the committed
  input state;
* row operators (RMSNORM, SOFTMAX): the chunk-aligned row span the
  disputed chunk touches (the block's score rows are shorter than one
  chunk, so a chunk may cover several rows — handled explicitly);
* GEMM nodes: the full-operand path above, chained to the frozen v0.1.1
  micro-arithmetic.

The arbiter verdict is `WorkerWins`, `ChallengerWins` or `BothInvalid`
based on which party's committed chunk equals the recomputation — the
arbiter never recomputes the block.

## Cross-language status

All eight operators plus the GEMM transpose form replay bit-for-bit
between Go and Python in `canonical_vectors.json` (math primitives,
tensor roots, eleven operator vectors, and a full ten-node graph with
`GraphID`, trail, outputs and work vector identical on both sides).
