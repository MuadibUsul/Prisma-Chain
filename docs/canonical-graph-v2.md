# CANONICAL_GRAPH_V2 — the wide-integer graph family

Version `CANONICAL_GRAPH_V2/1.0.0`.  Additive: every V1 domain, object and
proof stays frozen; the graph dispatch is explicit through the
descriptor's own `protocol_version` (the chain never reinterprets a V1
envelope).

## Descriptor and GraphID

`GraphDescriptorV2 = { protocol_version, spec, arithmetic:
A13W10_I64_PROFILE_V1, inputs: [{name, descV2, root}], nodes:
[{node_id, operator_id, operator_version, inputs:[TensorRef], output:
descV2, params:[{key,value}]}], outputs:[TensorRef] }`.

- `GraphIDV2 = SHA256("PRISMA_CANONICAL_GRAPH_V2\0" || CBOR(descriptor))`.
  The arithmetic profile (hence PolicyID) is bound into the identity; the
  GPU backend never appears in any consensus object.
- Wire form: JSON with base64 `[]byte` fields; the canonical-CBOR view
  keeps raw bytes.  Both languages sort map keys by encoded bytes.
- Node ids are dense and ordered; forward references are rejected;
  operator versions must match the registry.

## Operators

Frozen V1 cheap operators reused by ID and version (quantized values in
Q12.20): ADD_FIXED_V1, MUL_FIXED_V1, RMSNORM_FIXED_V1, ROPE_FIXED_V1,
SILU_FIXED_V1, SOFTMAX_FIXED_V1.  New versioned operators:

- `GEMM_A13W10_I64_V1`: `C[i,j] = sum_k int64(A[i,k]) * int64(W[k,j])`,
  exact signed int64 accumulation; `transpose_b` 0/1; the LEFT operand is
  A13 and the RIGHT operand is W10 (model weights) or A13 (attention
  inner products — the realized block has 32 such nodes).  Admission:
  `K <= MaxSafeK64(13, wBits)` with `MaxSafeK64(13,10) = 4398046511103`.
  Work counter `GEMM_A13W10_MAC = M*N*K` (never 3x physical GPU streams).
- `REQUANTIZE_WIDE_V1`: `v = clamp(round_ties_even(input*mult >> shift),
  lo, hi)`, exact integer, target dtype Q12.20 or A13; int64-intermediate
  overflow is an admission failure, never a silent widening.

## Execution

`ExecuteGraphV2` validates the graph, verifies every input against its
committed root, executes nodes in order and produces the state trail.  The
chain NEVER executes the graph; the reference executor (Go) is mirrored
bit-for-bit by the Python implementation, and the Worker may use the
proven GPU backend (F.5B.1/F.5B.2).

## State and trail

- `StateLeafV2(kind,index,root) = SHA256("PRISMA_CANONICAL_GRAPH_STATE_V2\0"
  || u32be(kind) || u32be(index) || root)`; the state root is the Merkle
  root over sorted live tensor ids.
- `TrailLeafV2(graphID,step,stateRoot) = SHA256(
  "PRISMA_CANONICAL_GRAPH_TRACE_V2\0" || graphID || u32be(step) ||
  stateRoot)`; `TrailRootV2` / `TrailProofV2` / `VerifyTrailProofV2`
  under the same domain.  V1 material never verifies under these.

## Commitments

- `NodeOutputManifestV2`: leaf = `SHA256("PRISMA_GRAPH_NODE_MANIFEST_V2\0"
  || graphIDV2 || u32be(node_id) || operator_id || operator_version ||
  CBOR(descV2))` combined with the node's output TensorRootV2; root over
  the leaves in node order.  Built before ANY verification randomness
  exists; a Watcher's randomness is post-commit (CSPRNG + context +
  nonce), never `H(manifest_root)` alone.
- `GraphResultCommitV3` (version 3.0.0): graph id, arithmetic id, policy
  id, task/assignment refs, worker pubkey, manifest root V2, output roots,
  final output root, completed epoch, Ed25519 signature under its own
  signing domain (`PRISMA_CANONICAL_SIG_V3`).  A V1/V2-domain signature
  can never validate V3 bytes.
- `VerifiedGraphWorkReceiptV3`: binds protocol version, GraphIDV2,
  arithmetic profile + PolicyID, refs, worker key, manifest root, output
  roots, chain-derived work vector, verification mode, epochs, settlement
  and DA references, optional dispute digest — under its own receipt
  domain.  Exactly one receipt on honest finalization; zero on fraud or
  availability failure.  Any exact backend earns the same receipt.

## Disputes

- Graph bisection (`GraphDisputeV2`): both parties lock their V2 trails
  (shared initial state, differing final states); midpoint states are
  proved against the locked trail roots; the interval bisects to the first
  divergent node; round clock is block-height-bound.  Snapshot V2 is
  JSON, restart-safe, and persists an in-flight round's median.
- Typed cheap-op arbitration (`ArbitrateNodeChunkV2`): both chunks and all
  operand evidence verify against TensorRootV2; the requant operator is
  recomputed element-wise (including INT64_ACCUM input chunks) and the
  Q12.20 operators share the frozen bounded V1 arbiter arithmetic; wide
  GEMM nodes are refused here.
- `WIDE_GEMM_DISPUTE_V1`: opened only on the first divergent GEMM node;
  8x8 tiles; partial-state chain `S_{r+1} = S_r + A_tile*W_tile` (exactly
  512 logical MAC per step) with BE64 leaves under their own trace
  domain; K-step bisection; the final arbiter recomputes exactly one
  512-MAC step from type-proven operand chunks (tile values extracted by
  the chain from the proven chunks, never trusted from the caller).
  Settlement flows through the same economics; a fraudulent worker
  receives no receipt.  Freivalds mismatch is DETECTION ONLY and never
  slashes by itself.
