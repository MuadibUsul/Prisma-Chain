"""Deterministic test-matrix generator mirroring GenTestMatrix in
compute/gemmv1/helpers.go."""

import hashlib

from .protocol import DomainTestGen


def gen_test_matrix(matrix_id: int, seed: int, length: int):
    """Value i = int8(SHA256(DomainTestGen || u32be(seed) || id || u64be(i))[0])."""
    prefix = DomainTestGen + seed.to_bytes(4, "big") + bytes([matrix_id])
    out = []
    for i in range(length):
        block = hashlib.sha256(prefix + i.to_bytes(8, "big")).digest()
        out.append(block[0] - 256 if block[0] > 127 else block[0])
    return out
