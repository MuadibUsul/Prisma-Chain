"""Merkle inclusion proofs mirroring compute/gemmv1/merkle.go.

Odd last node pairs with itself; a single leaf is its own root. Proofs bind
the leaf position via (index, count)."""

from .protocol import _sha256


def build_levels(leaves):
    if not leaves:
        raise ValueError("Merkle tree needs at least one leaf")
    level = list(leaves)
    levels = [level]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(_sha256(left, right))
        levels.append(nxt)
        level = nxt
    return levels


def root_of(leaves) -> bytes:
    levels = build_levels(leaves)
    return levels[-1][0]


def prove(levels, index: int):
    """Return (index, count, siblings) for the leaf at position index."""
    count = len(levels[0])
    if index >= count:
        raise ValueError("leaf index out of range")
    siblings = []
    idx = index
    for level in levels[:-1]:
        if idx % 2 == 0:
            sib = idx + 1
            if sib >= len(level):
                sib = idx
        else:
            sib = idx - 1
        siblings.append(level[sib])
        idx //= 2
    return index, count, siblings


def verify_inclusion(root: bytes, leaf: bytes, index: int, count: int, siblings) -> bool:
    if count == 0 or index >= count or len(siblings) != _depth(count):
        return False
    h = leaf
    idx, cnt = index, count
    for sib in siblings:
        if idx % 2 == 0:
            right = sib if idx + 1 < cnt else h
            h = _sha256(h, right)
        else:
            h = _sha256(sib, h)
        idx //= 2
        cnt = (cnt + 1) // 2
    return h == root


def _depth(count: int) -> int:
    depth = 0
    while count > 1:
        count = (count + 1) // 2
        depth += 1
    return depth
