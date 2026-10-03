"""Canonical tensor layout and reference GEMM mirroring compute/gemmv1.

A and B are raw signed int8 bytes row-major; C and trace states are signed
two's-complement big-endian int32. The reference GEMM is the protocol
oracle, not a performance path.
"""

import struct

TILE_SIZE = 8
INT8_TILE_SIZE = TILE_SIZE * TILE_SIZE
INT_TILE_COUNT = TILE_SIZE * TILE_SIZE
INT_TILE_BYTES = INT_TILE_COUNT * 4


def int32s_to_canonical(vals) -> bytes:
    return b"".join(struct.pack(">i", int(v)) for v in vals)


def canonical_to_int32s(raw: bytes):
    if len(raw) % 4 != 0:
        raise ValueError("canonical int32 input must be a multiple of 4 bytes")
    return [v[0] for v in struct.iter_unpack(">i", raw)]


def extract_a_tile(a, m: int, k: int, i: int, r: int):
    """Zero-padded 8x8 A tile A_tile[i][r] (i over M, r over K)."""
    tile = [0] * INT8_TILE_SIZE
    for row in range(TILE_SIZE):
        gi = i * TILE_SIZE + row
        if gi >= m:
            break
        for col in range(TILE_SIZE):
            gk = r * TILE_SIZE + col
            if gk >= k:
                break
            tile[row * TILE_SIZE + col] = a[gi * k + gk]
    return tile


def extract_b_tile(b, k: int, n: int, r: int, j: int):
    """Zero-padded 8x8 B tile B_tile[r][j] (r over K, j over N)."""
    tile = [0] * INT8_TILE_SIZE
    for row in range(TILE_SIZE):
        gk = r * TILE_SIZE + row
        if gk >= k:
            break
        for col in range(TILE_SIZE):
            gn = j * TILE_SIZE + col
            if gn >= n:
                break
            tile[row * TILE_SIZE + col] = b[gk * n + gn]
    return tile


def reference_gemm(a, b, m: int, n: int, k: int):
    """C = A x B with int8 inputs and a signed int32 accumulator."""
    c = [0] * (m * n)
    for i in range(m):
        ai = a[i * k:(i + 1) * k]
        base = i * n
        for t in range(k):
            av = ai[t]
            if av == 0:
                continue
            bt = b[t * n:(t + 1) * n]
            for j in range(n):
                c[base + j] += av * bt[j]
    return c


def output_tiles(c, m: int, n: int):
    """Slice C into zero-padded row-major 8x8 int32 tiles (flat lists)."""
    rows_c = (m + TILE_SIZE - 1) // TILE_SIZE
    cols_c = (n + TILE_SIZE - 1) // TILE_SIZE
    tiles = []
    for i in range(rows_c):
        for j in range(cols_c):
            tile = [0] * INT_TILE_COUNT
            for row in range(TILE_SIZE):
                gi = i * TILE_SIZE + row
                if gi >= m:
                    break
                for col in range(TILE_SIZE):
                    gj = j * TILE_SIZE + col
                    if gj >= n:
                        break
                    tile[row * TILE_SIZE + col] = c[gi * n + gj]
            tiles.append(tile)
    return tiles
