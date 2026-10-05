"""Canonical CBOR encoder mirroring compute/gemmv1/encoding.go.

Restricted to the GEMM v1 protocol subset: unsigned and negative integers,
byte strings, text strings, lists and maps with text-string keys. Map keys
are sorted bytewise by their canonical encodings (RFC 8949 section 4.2.1),
which for the short ASCII keys used by this protocol coincides with
length-then-bytewise ordering. Floats, booleans and None are rejected.
"""

import struct


class CanonicalCBORSError(ValueError):
    pass


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
        raise CanonicalCBORSError(f"type not supported by canonical CBOR encoding: {type(value)!r}")
    if isinstance(value, int):
        if value >= 0:
            if value > 2**64 - 1:
                raise CanonicalCBORSError("uint64 overflow")
            return _head(0, value)
        if value < -(2**63):
            raise CanonicalCBORSError("int64 underflow")
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
                raise CanonicalCBORSError("map keys must be text strings")
            pairs.append((_encode(key), _encode(item)))
        pairs.sort(key=lambda kv: kv[0])
        out = [_head(5, len(pairs))]
        for enc_key, enc_val in pairs:
            out.append(enc_key)
            out.append(enc_val)
        return b"".join(out)
    raise CanonicalCBORSError(f"type not supported by canonical CBOR encoding: {type(value)!r}")


def encode_canonical(value) -> bytes:
    """Return the RFC 8949 core-deterministic CBOR encoding of value."""
    return _encode(value)


__all__ = ["encode_canonical", "CanonicalCBORSError"]
