# GEMM_A13W10_I64_V1 — the wide-integer GEMM

## Mathematics (backend-independent)

```
C[i,j] = sum_{k} int64(A[i,k]) * int64(W[k,j])      (transpose_b = 0)
C[i,j] = sum_{k} int64(A[i,k]) * int64(W[j,k])      (transpose_b = 1)
A in A13 ([-4096,4095]), W in W10 ([-512,511]) or A13 for attention
inner products (QxK^T, PxV), C in INT64_ACCUM (exact signed int64)
```

Admission: `K <= MaxSafeK64(13, wBits)` with
`MaxSafeK64(a,w) = floor((2^63 - 1) / (2^(a-1) * 2^(w-1)))`;
`MaxSafeK64(13,10) = 4398046511103`; real block `K_max = 3072`.  The chain
checks the bound from the descriptor; it never trusts observed data.

Real block evidence: 83 GEMM nodes, `GEMM_A13W10_MAC = 252,706,816`
(logical MAC, counted once regardless of the physical realization).

## Work accounting

`GEMM_A13W10_MAC = M*N*K` from the descriptor (chain-derived).  GPU
decomposition (radix-128 limbs, tensor cores, Karatsuba) is a Worker
implementation detail: it never enters consensus identity, receipts, or
the work vector.

## Executions

- Go reference: `ReferenceWideGEMM` — plain int64 multiply/accumulate.
  The ONLY arbiter semantics.
- Python mirror: `reference_wide_gemm` — bit-identical.
- GPU (proven, F.5B.1/F.5B.2): `TRUE_FUSED_MMA_A13W10` — one m16n8k32 s8
  kernel per logical GEMM on SM86 + SM89, bit-exact on the full 310-node
  block, weighted 1.17x vs the fastest native INT8 baseline.

## Frozen vectors

`testdata/f5b_wide_gemm_vectors.json` replays bit-exactly through Go,
Python and the GPU path; the formal V2 graph replays 310/310 nodes
identical between CPU, A40 and RTX 2000 Ada.

## Dispute (WIDE_GEMM_DISPUTE_V1)

8x8 output tiles; K in 8-wide steps; partial state S_r (64 int64, BE64)
per step; trace root under `PRISMA_WIDE_GEMM_TRACE_V1`; K-step bisection
to the first divergent step; the final on-chain arbiter accepts only the
committed operand chunks (typed, proved), extracts the 8x8 A/W tiles
itself and recomputes exactly one 8x8x8 = 512-MAC step in exact int64.
Worst-case magnitude 8*4096*512 = 16,777,216 — far inside int64.  Per
A2-07 the verdict settles through the standard refund/slash/reward with
zero worker receipt.  Measured final witness (devnet-scale graph): 5,550
bytes, 10 evidence chunks, 512 gas-priced MAC units, 366,347 gas units
for the arbitration transaction (docs/phase-f5c-gas-results.json).
