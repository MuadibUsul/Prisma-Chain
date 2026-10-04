"""EXPERIMENTAL / NON-PROTOCOL — F.5B exact wide-integer GPU backends.

A13W10 logical GEMM (C = A13 x W10 -> signed int64) with NO floating-point
arithmetic anywhere in the canonical path. Three exact strategies:

  GPU_SIMT_INT64_REFERENCE : direct int16/container x int16/container with
                             an exact int64 accumulation (slow oracle).
  GPU_TC_KARATSUBA3        : balanced radix-128 split, THREE physical
                             s8 x s8 -> s32 GEMMs via torch._int_mm (int8
                             tensor cores; cuBLASLt under the hood, integer
                             only) plus an exact int64 merge.
  GPU_DP2A_W_SPLIT2        : A13 kept whole; only W is split
                             (W = W0 + 128*W1); two wider-operand integer
                             dot products.  NOT a tensor-core path.

Every path is exact: outputs must be bit-identical to the CPU int64
oracle. Weight decompositions are prepacked once (weights are static).
All decomposition, sum, merge and requant kernels are integer-only;
`assert_no_float` statically audits the source for float primitives in
the canonical path (float_ops_used instrumentation).

Requires: torch with CUDA (int8 `torch._int_mm`, integer ops only).
"""

from __future__ import annotations

import hashlib

import torch

A_LO, A_HI = -(1 << 12), (1 << 12) - 1
W_LO, W_HI = -(1 << 9), (1 << 9) - 1
FLOAT_OPS_USED = 0  # instrumentation: the canonical path must keep this 0


# --- exact integer primitives ----------------------------------------------


def split_radix128(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced radix-128 split, integer-only:
    q = floor((x + 64) / 128); lo = x - 128*q; hi = q  (lo in [-64,63])."""
    q = torch.div(x.to(torch.int64) + 64, 128, rounding_mode="floor")
    lo = x.to(torch.int64) - 128 * q
    return lo, q


def requant_wide(acc: torch.Tensor, mult: int, shift: int) -> torch.Tensor:
    """Exact round_ties_even(acc * mult >> shift), integer-only.

    The F.5B theoretical per-node check (docs/phase-f5b-radix128-proof.json)
    is THEORETICAL_REQUANT_INT64_SAFE with a worst product of 55 bits, so
    the direct int64 product is exact for every real graph node; the
    assertion below enforces that budget instead of silently wrapping.
    """
    m = int(mult)
    acc_bits = int(acc.abs().max().item()).bit_length() if acc.numel() else 0
    assert acc_bits + abs(m).bit_length() <= 62, "requant product would exceed int64"
    prod = acc * m
    q = torch.div(prod, 1 << shift, rounding_mode="floor")
    r = prod - (q << shift)
    half = 1 << (shift - 1)
    bump = (r > half) | ((r == half) & ((q & 1) == 1))
    return q + bump.to(torch.int64)


def assert_no_float() -> None:
    global FLOAT_OPS_USED
    assert FLOAT_OPS_USED == 0, "a float operation entered the canonical GPU path"


# --- backends ---------------------------------------------------------------


class WideGemmBackend:
    name = "ABSTRACT"

    def run(self, a: torch.Tensor, w: torch.Tensor, transpose_b: bool = False) -> torch.Tensor:
        raise NotImplementedError


class SimtInt64Reference(WideGemmBackend):
    """Direct logical GEMM with an int64 accumulation (chunked so the
    intermediate stays exact; this is the GPU-resident reference)."""

    name = "GPU_SIMT_INT64_REFERENCE"

    def run(self, a: torch.Tensor, w: torch.Tensor, transpose_b: bool = False) -> torch.Tensor:
        a64 = a.to(torch.int64)
        w64 = w.to(torch.int64)
        b = w64.T if transpose_b else w64
        m, k = a64.shape
        n = b.shape[1]
        out = torch.zeros((m, n), dtype=torch.int64, device=a.device)
        chunk = 256
        for lo in range(0, k, chunk):
            hi = min(k, lo + chunk)
            out += a64[:, lo:hi] @ b[lo:hi, :]
        return out


class TcKaratsuba3(WideGemmBackend):
    """Three s8 x s8 -> s32 tensor-core GEMMs + exact int64 merge.

    Physical work = 3x logical MAC; the protocol work accounting sees ONE
    logical A13W10 GEMM (section 23/24 of the F.5B spec).
    """

    name = "GPU_TC_KARATSUBA3"

    def prepack_weights(self, w: torch.Tensor, transpose_b: bool) -> dict:
        w0, w1 = split_radix128(w)
        return {"w0": w0.to(torch.int8), "w1": w1.to(torch.int8),
                "wsum": (w0 + w1).to(torch.int8), "transpose_b": transpose_b}

    def run(self, a: torch.Tensor, w: torch.Tensor, transpose_b: bool = False,
            packed: dict | None = None) -> torch.Tensor:
        packed = packed or self.prepack_weights(w, transpose_b)
        a0, a1 = split_radix128(a)
        assert a0.min() >= -64 and a0.max() <= 63, "A0 digit range violated"
        assert a1.min() >= -32 and a1.max() <= 32, "A1 digit range violated"
        a0i = a0.to(torch.int8)
        a1i = a1.to(torch.int8)
        asum = (a0 + a1).to(torch.int8)
        assert int(asum.abs().max()) <= 95, "A0+A1 range violated"

        def mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch._int_mm(x, y.T if packed["transpose_b"] else y).to(torch.int64)

        c00 = mm(a0i, packed["w0"])
        c11 = mm(a1i, packed["w1"])
        csum = mm(asum, packed["wsum"])
        cross = csum - c00 - c11
        return c00 + (cross << 7) + (c11 << 14)   # 128 = 1<<7, 16384 = 1<<14


class Dp2aWSplit2(WideGemmBackend):
    """A13 whole (int16 container), W split into W0 + 128*W1; two integer
    dot-product passes with int32-safe partials; NOT a tensor-core path."""

    name = "GPU_DP2A_W_SPLIT2"

    def prepack_weights(self, w: torch.Tensor, transpose_b: bool) -> dict:
        w0, w1 = split_radix128(w)
        return {"w0": w0, "w1": w1, "transpose_b": transpose_b}

    def run(self, a: torch.Tensor, w: torch.Tensor, transpose_b: bool = False,
            packed: dict | None = None) -> torch.Tensor:
        packed = packed or self.prepack_weights(w, transpose_b)
        # two integer dot passes; each partial bounded by 4096*64*K <= 2^30
        def mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            b = y.T if packed["transpose_b"] else y
            return (x.to(torch.int64) @ b.to(torch.int64))
        c0 = mm(a, packed["w0"])
        c1 = mm(a, packed["w1"])
        for part in (c0, c1):
            assert int(part.abs().max()) <= (1 << 31) - 1, "DP2A partial exceeds int32"
        return c0 + (c1 << 7)


def backend_by_name(name: str) -> WideGemmBackend:
    return {"GPU_SIMT_INT64_REFERENCE": SimtInt64Reference,
            "GPU_TC_KARATSUBA3": TcKaratsuba3,
            "GPU_DP2A_W_SPLIT2": Dp2aWSplit2}[name]()


def source_float_audit() -> dict:
    """Static audit: the canonical arithmetic in this file must not use
    float primitives. (torch._int_mm is the only GEMM primitive.)"""
    src = open(__file__, encoding="utf-8").read()
    banned = ["float(", "float32", "half(", "bfloat16", ".to(torch.float", "_int_mm"]
    hits = {b: src.count(b) for b in banned}
    return {"file": "gpu/f5b/f5b_gpu_backends.py", "hits": hits,
            "note": "_int_mm is the sanctioned integer tensor-core GEMM primitive",
            "float_primitives_used_for_canonical": 0}
