"""Exact bad-row -> bad-column -> bad-tile localization, mirroring
compute/gemmv1/verify/row_locator.go (Go). The Freivalds mismatch only
GUIDES this search; the resulting tile evidence feeds the deterministic
v0.1.1 dispute."""

from .tensors import INT_TILE_COUNT


def reference_row_gemm(a, row: int, b, m: int, n: int, k: int):
    """C[row,:] = A[row,:] x B, exact int32 under the MaxSafeK bound."""
    if row < 0 or row >= m:
        raise ValueError("row out of range")
    if len(a) != m * k or len(b) != k * n:
        raise ValueError("input shape mismatch")
    out = [0] * n
    for t in range(k):
        av = a[row * k + t]
        if av == 0:
            continue
        bt = b[t * n:(t + 1) * n]
        for j in range(n):
            out[j] += av * bt[j]
    return out


def locate_bad_columns(worker_row, expected_row):
    return [j for j in range(len(worker_row)) if worker_row[j] != expected_row[j]]


def tile_for_element(row: int, col: int):
    return row // 8, col // 8


def localize_from_rows(a, b, c, m: int, n: int, k: int, rows):
    """Returns (localization|None, found). localization has row, columns,
    tile_i, tile_j, expected_row."""
    if not rows:
        raise ValueError("no residual rows to localize")
    for row in rows:
        expected = reference_row_gemm(a, row, b, m, n, k)
        worker_row = c[row * n:(row + 1) * n]
        cols = locate_bad_columns(worker_row, expected)
        if not cols:
            continue
        tile_i, tile_j = tile_for_element(row, cols[0])
        return {
            "row": row,
            "columns": cols,
            "tile_i": tile_i,
            "tile_j": tile_j,
            "expected_row": expected,
        }, True
    return None, False


assert INT_TILE_COUNT == 64
