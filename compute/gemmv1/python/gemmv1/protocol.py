"""Domain separation, protocol hashing and object builders for GEMM v1.

Mirrors compute/gemmv1/hash.go and the object builders. All protocol
objects are plain Python dicts with the exact snake_case keys of the Go
`gemm` struct tags; encode_canonical makes them byte-identical.
"""

import hashlib

from .canonical_cbor import encode_canonical

ProtocolVersion = "0.1.1"
Operator = "GEMM_INT8_V1"
ArithmeticSpec = "INT8_INT32_V1"

DomainTask = b"PRISMA_GEMM_TASK_V1\x00"
DomainAssignment = b"PRISMA_GEMM_ASSIGNMENT_V1\x00"
DomainInputTile = b"PRISMA_GEMM_INPUT_TILE_V1\x00"
DomainOutputTile = b"PRISMA_GEMM_OUTPUT_TILE_V1\x00"
DomainTraceState = b"PRISMA_GEMM_TRACE_STATE_V1\x00"
DomainVWR = b"PRISMA_GEMM_VWR_V1\x00"
DomainSig = b"PRISMA_GEMM_SIG_V1\x00"
DomainTestGen = b"PRISMA_GEMM_TESTGEN_V1\x00"

MATRIX_ID_A = 0x01
MATRIX_ID_B = 0x02


def _sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.digest()


def _u32be(v: int) -> bytes:
    return v.to_bytes(4, "big")


def task_hash(domain: bytes, obj: dict) -> bytes:
    """Domain || CanonicalCBOR(obj)."""
    return _sha256(domain, encode_canonical(obj))


def task_id(descriptor: dict) -> bytes:
    return task_hash(DomainTask, descriptor)


def assignment_id(assignment: dict) -> bytes:
    return task_hash(DomainAssignment, assignment)


def leaf_input_tile(matrix_id: int, tile_row: int, tile_col: int, tile_bytes: bytes) -> bytes:
    return _sha256(DomainInputTile, bytes([matrix_id]), _u32be(tile_row), _u32be(tile_col), tile_bytes)


def leaf_output_tile(task: bytes, assignment: bytes, tile_i: int, tile_j: int, tile_bytes: bytes) -> bytes:
    return _sha256(DomainOutputTile, task, assignment, _u32be(tile_i), _u32be(tile_j), tile_bytes)


def leaf_trace_state(task: bytes, assignment: bytes, tile_i: int, tile_j: int, step: int, state_bytes: bytes) -> bytes:
    return _sha256(DomainTraceState, task, assignment, _u32be(tile_i), _u32be(tile_j), _u32be(step), state_bytes)


def merkle_root(leaves) -> bytes:
    """Binary Merkle root; an odd last node pairs with itself, a single
    leaf is its own root. Mirrors compute/gemmv1/merkle.go."""
    if not leaves:
        raise ValueError("Merkle tree needs at least one leaf")
    level = list(leaves)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(_sha256(left, right))
        level = nxt
    return level[0]


def build_matrix_roots(a, b, m: int, n: int, k: int):
    """Commit A and B through their zero-padded 8x8 tile grids."""
    from .tensors import extract_a_tile, extract_b_tile, INT8_TILE_SIZE

    rows_a = (m + 7) // 8
    cols_a = (k + 7) // 8
    rows_b = cols_a
    cols_b = (n + 7) // 8
    leaves_a = [
        leaf_input_tile(MATRIX_ID_A, i, r, bytes(v & 0xFF for v in extract_a_tile(a, m, k, i, r)))
        for i in range(rows_a)
        for r in range(cols_a)
    ]
    leaves_b = [
        leaf_input_tile(MATRIX_ID_B, r, j, bytes(v & 0xFF for v in extract_b_tile(b, k, n, r, j)))
        for r in range(rows_b)
        for j in range(cols_b)
    ]
    return merkle_root(leaves_a), merkle_root(leaves_b), cols_a, cols_b, rows_a, cols_b


def receipt_id(receipt: dict) -> bytes:
    return task_hash(DomainVWR, receipt)


# DA_REPLICA_V1 constants (mirror compute/gemmv1/da.go).
DAProtocolVersion = "DA_REPLICA_V1"
DomainDAChallengeV1 = b"PRISMA_GEMM_DA_CHALLENGE_V1\x00"


def da_challenge_tile(task_id: bytes, provider_account: bytes, challenger_account: bytes,
                      nonce: bytes, opened_height: int, rows_c: int, cols_c: int):
    """Deterministic sampling-tile derivation mirroring DAChallengeTile."""
    import hashlib

    h = hashlib.sha256()
    h.update(DomainDAChallengeV1)
    h.update(task_id)
    h.update(provider_account)
    h.update(challenger_account)
    h.update(nonce)
    h.update(opened_height.to_bytes(8, "big"))
    digest = h.digest()
    total = rows_c * cols_c
    index = int.from_bytes(digest[:8], "big") % total
    return index // cols_c, index % cols_c
