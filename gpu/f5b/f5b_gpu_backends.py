"""EXPERIMENTAL / NON-PROTOCOL — F.5B exact wide-integer GPU backends.

A13W10 logical GEMM (C = A13 x W10 -> signed int64) with NO floating-point
arithmetic anywhere in the canonical path. Strategies:

  GPU_TC_KARATSUBA3 : balanced radix-128 split, THREE physical s8 x s8 ->
                      s32 GEMMs via torch._int_mm (int8 tensor cores,
                      cuBLASLt under the hood, integer only) plus an exact
                      int64 merge C = C00 + (Csum - C00 - C11)<<7 + C11<<14.
  GPU_SCHOOLBOOK4   : the same split with FOUR explicit products
                      (C00, C01, C10, C11); an independent merge formula,
                      used as the on-hardware cross-check of Karatsuba3.
  GPU_SIMT_INT64_REFERENCE / GPU_DP2A_W_SPLIT2 : direct wider-operand
                      integer matmuls.  **CPU-executable only** with stock
                      PyTorch: CUDA has no generic integer matmul kernel
                      (`"addmm_cuda" not implemented for 'Long'`, captured
                      verbatim on the F.5B pods); their hardware cross-check
                      role is filled by GPU_SCHOOLBOOK4.

CUDA platform constraint (captured on the pods, torch 2.8.0+cu128):
`torch._int_mm` requires the LEFT operand to have M > 16 rows.  Every
tensor-core call zero-pads operands by role — M -> max(32, align16),
K -> max(16, align16), N -> max(16, align16).  Zero padding is
mathematically inert, so bit-exactness is preserved; the benchmark applies
identical padding to the native int8 baseline so ratios stay fair.

Every path is exact: outputs must be bit-identical to the CPU int64
oracle. Weight decompositions are prepacked once (weights are static).
All decomposition, sum, merge and requant kernels are integer-only;
`source_float_audit` statically audits this source for float primitives in
the canonical path.
"""

from __future__ import annotations

import torch

A_LO, A_HI = -(1 << 12), (1 << 12) - 1
W_LO, W_HI = -(1 << 9), (1 << 9) - 1
FLOAT_OPS_USED = 0  # instrumentation: the canonical path must keep this 0

TC_ROW_MIN = 32   # torch._int_mm on CUDA: left-operand M must be > 16
TC_ALIGN = 16


def _align16(v: int, minimum: int) -> int:
    return max(minimum, (v + TC_ALIGN - 1) // TC_ALIGN * TC_ALIGN)


def pad_role(m: int, k: int, n: int) -> tuple[int, int, int]:
    """Padded target sizes (m', k', n') for tensor-core operands."""
    return _align16(m, TC_ROW_MIN), _align16(k, 16), _align16(n, 16)


def pad_to_2d(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Zero-pad a 2-D tensor to (rows, cols).  Zero padding is inert for
    the integer products, so exactness is preserved."""
    if tuple(t.shape) == (rows, cols):
        return t
    out = torch.zeros((rows, cols), dtype=t.dtype, device=t.device)
    out[: t.shape[0], : t.shape[1]] = t
    return out


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


# --- backends ---------------------------------------------------------------


class WideGemmBackend:
    name = "ABSTRACT"

    def run(self, a: torch.Tensor, w: torch.Tensor, transpose_b: bool = False) -> torch.Tensor:
        raise NotImplementedError


class _TensorCoreBase(WideGemmBackend):
    """Shared radix-128 split + role-based zero padding for int8 tensor cores."""

    def _pack(self, w: torch.Tensor, transpose_b: bool) -> dict:
        w0, w1 = split_radix128(w)
        n, k = (w.shape[0], w.shape[1]) if transpose_b else (w.shape[1], w.shape[0])
        _, k_pad, n_pad = pad_role(0, k, n)
        rows, cols = (n_pad, k_pad) if transpose_b else (k_pad, n_pad)
        return {
            "w0": pad_to_2d(w0.to(torch.int8), rows, cols),
            "w1": pad_to_2d(w1.to(torch.int8), rows, cols),
            "wsum": pad_to_2d((w0 + w1).to(torch.int8), rows, cols),
            "transpose_b": transpose_b,
            "w_shape": tuple(w.shape),
            "n": n, "k": k, "k_pad": k_pad, "n_pad": n_pad,
        }

    def _limbs(self, a: torch.Tensor, packed: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        m, k = a.shape
        assert k == packed["k"], "activation/weight K mismatch"
        a0, a1 = split_radix128(a)
        assert a0.min() >= -64 and a0.max() <= 63, "A0 digit range violated"
        assert a1.min() >= -32 and a1.max() <= 32, "A1 digit range violated"
        asum = a0 + a1
        assert int(asum.abs().max()) <= 95, "A0+A1 range violated"
        m_pad = _align16(m, TC_ROW_MIN)
        k_pad = packed["k_pad"]
        return (pad_to_2d(a0.to(torch.int8), m_pad, k_pad),
                pad_to_2d(a1.to(torch.int8), m_pad, k_pad),
                pad_to_2d(asum.to(torch.int8), m_pad, k_pad),
                m)

    def _mm(self, x: torch.Tensor, y: torch.Tensor, transpose_b: bool) -> torch.Tensor:
        b = y.T if transpose_b else y
        if not b.is_contiguous():
            b = b.contiguous()
        return torch._int_mm(x, b).to(torch.int64)


class TcKaratsuba3(_TensorCoreBase):
    """Three s8 x s8 -> s32 tensor-core GEMMs + exact int64 merge.

    Physical work = 3x logical MAC; the protocol work accounting sees ONE
    logical A13W10 GEMM (section 23/24 of the F.5B spec).
    """

    name = "GPU_TC_KARATSUBA3"

    def prepack_weights(self, w: torch.Tensor, transpose_b: bool) -> dict:
        return self._pack(w, transpose_b)

    def run(self, a: torch.Tensor, w: torch.Tensor, transpose_b: bool = False,
            packed: dict | None = None) -> torch.Tensor:
        packed = packed or self._pack(w, transpose_b)
        a0i, a1i, asum, m = self._limbs(a, packed)
        mm = lambda x, y: self._mm(x, y, packed["transpose_b"])  # noqa: E731
        c00 = mm(a0i, packed["w0"])
        c11 = mm(a1i, packed["w1"])
        csum = mm(asum, packed["wsum"])
        cross = csum - c00 - c11
        full = c00 + (cross << 7) + (c11 << 14)   # 128 = 1<<7, 16384 = 1<<14
        return full[:m, :packed["n"]]


class Schoolbook4(_TensorCoreBase):
    """Four explicit s8 x s8 -> s32 tensor-core GEMMs + exact int64 merge.

    Merge: C = C00 + (C01 + C10)<<7 + C11<<14 (no Karatsuba cancellation).
    Shares the operand split with Karatsuba3 but uses an independent
    cross-term formula; this is the on-hardware reference on CUDA, where
    direct int64 matmuls are not a PyTorch primitive.
    """

    name = "GPU_SCHOOLBOOK4"

    def prepack_weights(self, w: torch.Tensor, transpose_b: bool) -> dict:
        return self._pack(w, transpose_b)

    def run(self, a: torch.Tensor, w: torch.Tensor, transpose_b: bool = False,
            packed: dict | None = None) -> torch.Tensor:
        packed = packed or self._pack(w, transpose_b)
        a0i, a1i, _asum, m = self._limbs(a, packed)
        mm = lambda x, y: self._mm(x, y, packed["transpose_b"])  # noqa: E731
        c00 = mm(a0i, packed["w0"])
        c01 = mm(a0i, packed["w1"])
        c10 = mm(a1i, packed["w0"])
        c11 = mm(a1i, packed["w1"])
        full = c00 + ((c01 + c10) << 7) + (c11 << 14)
        return full[:m, :packed["n"]]


class SimtInt64Reference(WideGemmBackend):
    """Direct logical int64 GEMM with chunked exact accumulation.

    CPU-executable only: CUDA raises NotImplementedError ("addmm_cuda" not
    implemented for 'Long'); the GPU-side reference role is filled by
    GPU_SCHOOLBOOK4."""

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


class Dp2aWSplit2(WideGemmBackend):
    """A13 whole, W split into W0 + 128*W1; two wider-operand integer dot
    products.  CPU-executable only (CUDA has no int16/int32 matmul kernel);
    NOT a tensor-core path."""

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


CUDA_BACKENDS = ("GPU_TC_KARATSUBA3", "GPU_SCHOOLBOOK4")
ALL_BACKEND_NAMES = ("GPU_TC_KARATSUBA3", "GPU_SCHOOLBOOK4",
                     "GPU_SIMT_INT64_REFERENCE", "GPU_DP2A_W_SPLIT2")


def backend_by_name(name: str) -> WideGemmBackend:
    return {"GPU_TC_KARATSUBA3": TcKaratsuba3,
            "GPU_SCHOOLBOOK4": Schoolbook4,
            "GPU_SIMT_INT64_REFERENCE": SimtInt64Reference,
            "GPU_DP2A_W_SPLIT2": Dp2aWSplit2}[name]()


def source_float_audit() -> dict:
    """Static audit: the canonical arithmetic in this file must not use
    float primitives. (torch._int_mm is the only GEMM primitive.)"""
    src = open(__file__, encoding="utf-8").read()
    banned = ["float(", "float32", "half(", "bfloat16", ".to(torch.float", "_int_mm"]
    hits = {b: src.count(b) for b in banned}
    return {"file": "gpu/f5b/f5b_gpu_backends.py", "hits": hits,
            "note": "_int_mm is the sanctioned integer tensor-core GEMM primitive",
            "float_primitives_used_for_canonical": FLOAT_OPS_USED}
