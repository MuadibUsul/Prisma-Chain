"""GraphIDV2 verification — an independent mirror of the frozen canonical
encoder (``compute/canonical/python/canonical_ref.py``) plus the V2 graph
domain.

The worker must be able to verify a downloaded graph profile *by itself*: a
worker host ships the worker package and the binaries, not the research tree.
So this module re-implements the canonical CBOR encoder and the graph-id
domain, and ``tests/test_graphid.py`` cross-checks it against the
authoritative module and the frozen GraphIDV2 of ``testdata/f5c_qwen3_block_v2.json``
for every run — drift between the mirror and the frozen protocol fails CI.

Frozen rules reproduced exactly:

    encode_canonical : canonical CBOR (head/major bytes, maps sorted by
                       encoded key bytes; bool/None are refused)
    hash_bytes       : sha256 over the concatenated parts
    GraphIDV2        : sha256(DOMAIN_GRAPH_V2 || encode_canonical(graph))
                       where inputs[].root is base64 on the wire and raw
                       bytes in the canonical view
"""

from __future__ import annotations

import base64
import hashlib
import json

DOMAIN_GRAPH_V2 = b"PRISMA_CANONICAL_GRAPH_V2\x00"


class CanonicalError(ValueError):
    """The value cannot be encoded canonically."""


def _head(major: int, val: int) -> bytes:
    m = major << 5
    if val < 24:
        return bytes([m | val])
    if val < 0x100:
        return bytes([m | 24, val])
    if val < 0x10000:
        return bytes([m | 25]) + val.to_bytes(2, "big")
    if val < 0x100000000:
        return bytes([m | 26]) + val.to_bytes(4, "big")
    return bytes([m | 27]) + val.to_bytes(8, "big")


def _encode(value) -> bytes:
    if isinstance(value, bool) or value is None:
        raise CanonicalError(f"unsupported canonical CBOR type: {type(value)!r}")
    if isinstance(value, int):
        if value >= 0:
            if value > 2**64 - 1:
                raise CanonicalError("uint64 overflow")
            return _head(0, value)
        if value < -(2**63):
            raise CanonicalError("int64 underflow")
        return _head(1, -value - 1)
    if isinstance(value, (bytes, bytearray)):
        return _head(2, len(value)) + bytes(value)
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return _head(3, len(raw)) + raw
    if isinstance(value, (list, tuple)):
        out = [_head(4, len(value))]
        for item in value:
            out.append(_encode(item))
        return b"".join(out)
    if isinstance(value, dict):
        pairs = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalError("map keys must be text strings")
            pairs.append((_encode(key), _encode(item)))
        pairs.sort(key=lambda kv: kv[0])
        out = [_head(5, len(pairs))]
        for enc_key, enc_val in pairs:
            out.append(enc_key)
            out.append(enc_val)
        return b"".join(out)
    raise CanonicalError(f"unsupported canonical CBOR type: {type(value)!r}")


def encode_canonical(value) -> bytes:
    return _encode(value)


def hash_bytes(*parts: bytes) -> bytes:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.digest()


def canon_graph_doc(doc: dict) -> dict:
    """Canonical view of a wire GraphDescriptorV2: base64 input roots as bytes."""
    out = json.loads(json.dumps(doc))
    for entry in out.get("inputs", []):
        entry["root"] = base64.b64decode(entry["root"])
    return out


def graph_id_v2_of(doc: dict) -> bytes:
    return hash_bytes(DOMAIN_GRAPH_V2, encode_canonical(canon_graph_doc(doc)))


def policy_id_of(doc: dict) -> str:
    """The descriptor arithmetic policy id recorded in the graph document."""
    try:
        return str(doc["arithmetic"]["policy_id"])
    except (KeyError, TypeError) as exc:
        raise CanonicalError("graph document has no arithmetic.policy_id") from exc
