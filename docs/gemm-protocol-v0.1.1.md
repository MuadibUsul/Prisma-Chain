# Prisma Verifiable GEMM Protocol v0.1.1

This document records the **implemented** GEMM_INT8_V1 protocol: an
independent, versioned verifiable-compute operator next to
`PRISMA_BOUNDED_INT_VM_V1` (the `vm/` package). It does not replace the
whitepaper; the whitepaper will be reconciled with this document manually in
a later editorial pass.

Implementation: `compute/gemmv1/` (Go), `compute/gemmv1/python/gemmv1/`
(Python mirror, GPU adapter, two-node services), CLI `cmd/prisma-gemm/`.

## Scope

GEMM_INT8_V1 makes one computation verifiable between machines that do not
trust each other:

```
Task -> Assign -> INT8 GEMM -> Output Commitment -> (Challenge) ->
On-demand Tile Trace -> Interactive Bisection -> 8x8x8 Micro-step
Arbitration -> Finalize -> Verified Work Receipt
```

Out of scope for v0.1.1: FP16/BF16/FP32/TF32 arithmetic, fast-math or
approximate GEMM, ZKML/FHE/MPC, model parallelism, training, and any change
to consensus or to the existing bounded-integer VM.

## Output-first optimistic commitment

The normal path never commits an execution trace. A worker submits exactly:

- TaskDescriptor acceptance via an Assignment (task_id + worker binding),
- an output Merkle root over 8x8 int32 tiles,
- `canonical_mac_count = M x N x K`.

A full-trace commitment at 4096x4096x4096 with tile size 8 would require
terabytes of intermediate states; v0.1.1 replaces it with an **on-demand**
trace: only after a valid challenge does each party lock a trace, and only
for the one disputed output tile. This is a mandatory protocol property, not
an optimization.

## Arithmetic and canonical layout

- `C = A x B`, A: int8[M,K], B: int8[K,N], C: int32[M,N].
- Every product is int8 x int8 -> int32; the accumulator is signed int32.
- Task admission rejects `K > MaxSafeK = (2^31 - 1) / (127 * 127) = 133162`,
  so no partial sum can overflow and every loop order is bit-exact.
- A/B canonical bytes: consecutive raw signed int8 values, row-major.
- C and trace states: signed two's-complement big-endian int32.
- All hashes are SHA-256 over platform-independent byte strings.

## Tiling

`TILE_SIZE = 8`. Boundary tiles are zero padded in all directions; padding
never enters the canonical work count. `R = ceil(K / 8)` is the number of
K-direction trace steps. The work measure is exactly
`canonical_mac_count = M x N x K` (uint64); 1 CWU (display only) = 2^20
canonical MACs. No GPU model, timing, VRAM or power weighting exists in
v0.1.1; hardware affects only execution time, price and profitability.

## Canonical encoding and hashing

Protocol objects are encoded with a deterministic CBOR subset (RFC 8949 core
deterministic encoding: shortest-form heads, maps sorted bytewise by their
canonical key encodings; floats, tags, booleans and indefinite lengths are
rejected). The Go and Python encoders are independent implementations of the
same rules and must produce identical bytes; this is enforced by
`compute/gemmv1/testdata/cross_language_vectors.json` (protocol test J).

Every hash input starts with a domain string ending in `0x00`:

| Domain | Over |
| --- | --- |
| `PRISMA_GEMM_TASK_V1` | CanonicalCBOR(TaskDescriptor) |
| `PRISMA_GEMM_ASSIGNMENT_V1` | CanonicalCBOR(Assignment) |
| `PRISMA_GEMM_INPUT_TILE_V1` | matrix_id \|\| tile_row \|\| tile_col \|\| tile bytes |
| `PRISMA_GEMM_OUTPUT_TILE_V1` | task_id \|\| assignment_id \|\| tile_i \|\| tile_j \|\| tile bytes |
| `PRISMA_GEMM_TRACE_STATE_V1` | task_id \|\| assignment_id \|\| tile_i \|\| tile_j \|\| step \|\| state bytes |
| `PRISMA_GEMM_VWR_V1` | CanonicalCBOR(VerifiedWorkReceipt) |
| `PRISMA_GEMM_SIG_V1` | signatures over canonical objects (extension) |

Task ID = SHA256(`PRISMA_GEMM_TASK_V1\0` || CanonicalCBOR(TaskDescriptor));
receipt ID is analogous. Coordinates and steps in leaf preimages use
fixed-width big-endian uint32 so concatenation is unambiguous. The existing
prismavm v1 hashing format is untouched; GEMM v1 has its own version space.

## Merkle trees

Binary Merkle trees over domain-separated leaves. Internal nodes are
SHA256(left || right) over two fixed 32-byte digests. Canonical odd-node
behavior: the last node of an odd level pairs with a copy of itself; a
single leaf is its own root. Proofs bind the leaf position with
`(index, count)`; a proof with a wrong index, count or sibling list is
rejected. Rebuilding a tree from the same leaves always yields the same
root (tested across sizes 1..17 leaves and in the cross-language vectors).

A and B use distinct matrix identifiers (`0x01`, `0x02`) in their leaves.
Output-tile leaves bind task_id and assignment_id, so a tile proof cannot be
replayed against another task or another worker's commit.

## Task, assignment and result

- `TaskDescriptor` (see `types.go`): fixed `operator = GEMM_INT8_V1`,
  `arithmetic_spec = INT8_INT32_V1`, `tile_size = 8`; requester key and
  nonce; M, N, K; both matrix roots; challenge window; price cap and asset.
- `Assignment` binds worker to task; `assignment_id` is derived from it and
  appears in every ResultCommit, trace leaf and receipt (replay protection).
- `ResultCommit` is the entire normal-path submission: protocol version,
  task_id, assignment_id, worker key, output_root, canonical_mac_count,
  completed_epoch, Ed25519 signature. It intentionally contains **no**
  execution-trace root. Receipts are only built over validly signed commits.

## Challenge, on-demand trace, bisection, arbitration

1. A challenger recomputes A x B (full verification is allowed in v0.1.1).
   If one output tile differs, it opens `ChallengeOpen` with the worker tile
   plus an inclusion proof against the worker's committed output_root. The
   coordinator rejects the challenge unless the proof verifies and the two
   tiles differ.
2. Both parties then lock a trace for the disputed tile only: S0 (zero
   matrix) .. S(R), each state an 8x8 int32 tile, committed by
   `TraceCommit` (trace root + endpoint proofs). Locking is immutable.
   The worker's S(R) must equal its committed tile; the challenger's S(R)
   must equal its canonical tile.
3. Interactive bisection over [0, R] mirrors `vm/dispute.go`: each round both
   parties submit state(mid) with an inclusion proof against their own
   locked root; equal states move `low`, differing states move `high`.
   `O(log2 R)` rounds. Every round has a response deadline; worker silence
   loses the round, challenger silence loses the round, both silent is
   `BothInvalid` and maps to the existing refund path.
4. Micro-step arbitration at the final disputed transition: the arbiter
   verifies A_tile[i][t] against matrix_a_root and B_tile[t][j] against
   matrix_b_root, computes `Expected = S_t + A_tile x B_tile` — exactly 512
   canonical MACs — and compares. Outcomes: `WorkerWins`,
   `ChallengerWins`, `BothInvalid`. The arbiter never re-executes the task.

Unlike `vm/monitor.go`, the v1 game cannot reach the "correct final output
but different interior root" corner: the normal path commits no full-trace
root at all, and a tile dispute always pins both endpoints of the bisection
to outputs that provably differ.

## Verified Work Receipt

Only a FINALIZED task yields a `VerifiedWorkReceipt`: protocol version,
task/assignment ids, worker key, operator, canonical_mac_count, output_root,
`verification_mode` (`optimistic_unchallenged` or `challenged_worker_won`),
finalized epoch, optional settlement reference and optional dispute
transcript digest (empty when unchallenged). A worker that loses a dispute
receives no receipt. `full execution_root` is not a field.

## GPU backend

`compute/gemmv1/python/gemmv1/gpu.py` accepts only a true
INT8 x INT8 -> INT32 CUDA kernel (`torch._int_mm`). Inputs are padded to
kernel-required multiples with zeros, which adds only zero products. If no
such kernel exists the environment reports `GPU_BACKEND_UNSUPPORTED`; the
code never falls back to floating point. GPU output must equal the Go CPU
reference bit-for-bit (numpy int64 matmul over int8 inputs provides an exact
independent oracle on GPU nodes; the K admission bound keeps int64
accumulation overflow-free).

## Two-node E2E topology

```
Coordinator
      |
 +----+----+
 v         v
Pod A     Pod B
Worker    Challenger
```

The worker service executes tasks, optionally corrupts one output tile
through a DEVELOPMENT-ONLY flag, and serves on-demand traces and bisection
midpoints. The challenger recomputes independently, locks traces, bisects
and arbitrates over HTTP. `deploy/runpod_gemm/` drives this on two RunPod
GPU nodes; every endpoint and dimension is read from environment variables
(`PRISMA_WORKER_URL`, `PRISMA_GEMM_M/N/K/SEED`, ...), and no credentials are
printed or committed.

## Chain integration order

Phase A (library + tests) and Phase B (CLI + local E2E) are done; Phase C is
the two-GPU RunPod E2E; only Phase D touches the chain module, by adding a
GEMM task spec, ResultCommit, ChallengeOpen, GEMM dispute state and VWR
handling next to the existing bounded-VM task flow. The 20% burn / 5% + 5%
monitor / remainder worker settlement of whitepaper v0.2 section 8 stays
authoritative; `settlement_reference` carries the chain pointer then.
