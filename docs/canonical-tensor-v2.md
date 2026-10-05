# CANONICAL_TENSOR_V2 — typed commitments for wide integers

Version `CANONICAL_TENSOR_V2/1.0.0`.  Additive to CANONICAL_GRAPH_V1:
V1 descriptors, roots and proofs are untouched and can never decode or
verify under V2 domains.

## Logical dtypes

| dtype | value | logical range | container | canonical element encoding |
|---|---|---|---|---|
| Q12.20 | 1 | signed int32 | 32 bits | BE32 two's complement |
| A13 | 129 | [-4096, 4095] | 16 bits | BE16 two's complement |
| W10 | 130 | [-512, 511] | 16 bits | BE16 two's complement |
| INT64_ACCUM | 131 | signed int64 | 64 bits | BE64 two's complement |

Logical width != storage width: A13 is 13 logical bits in a 16-bit
container, W10 is 10 logical bits in a 16-bit container.  Values outside
the logical range are REJECTED at construction/admission — the bytes can
never be reinterpreted as another dtype.  Two's complement has exactly one
zero.

## Descriptor

`TensorDescriptorV2 = {dtype, layout (1 = row-major), shape}`.  Its
canonical CBOR encoding is hashed into every leaf and the root, binding
the bytes to the dtype.

## Chunking and roots

- 64 logical elements per chunk; the final chunk is zero-padded with the
  dtype's canonical zero encoding (all-zero bytes).
- `leaf(i) = SHA256("PRISMA_CANONICAL_TENSOR_V2\0" || CBOR(desc) ||
  u32be(i) || chunk_bytes)`
- Merkle tree over the leaves with the repo rule (odd node pairs with
  itself).
- `TensorRootV2 = SHA256("PRISMA_CANONICAL_TENSOR_ROOT_V2\0" ||
  CBOR(desc) || merkle_root)`

Domains are distinct from V1: the same logical values produce different
roots across versions, and a V1 leaf/sibling material can never fold to a
V2 root (pinned by the legacy golden test).

## Proofs and verification

`VerifyChunkV2(rootV2, descV2, index, count, chunk, siblings)`:
width-checks the chunk against the dtype, rebuilds the leaf, folds the
inclusion proof back to the chunk merkle root and lifts it to the
TensorRootV2 domain.  Wrong dtype, wrong width, tampered bytes and V1
material are all rejected.  The Verifier never re-parses a descriptor from
untrusted bytes.

## Golden vectors

`testdata/canonical_graph_v2_state_vectors.json` (state roots) and the
legacy golden file pin cross-language equality (Go == Python) for
encodings, leafs, merkle folds and roots.
