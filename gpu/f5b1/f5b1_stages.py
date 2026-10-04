"""F.5B.1 — candidate GPU backends for the fusion ladder (research-only).

Ladder (per spec sections 13-28):
  STAGE_A_CUDA_GRAPH_K3   : CUDA-graph capture of the F.5B eager Karatsuba3
                            pipeline (no arithmetic change).
  STAGE_B_5LAUNCH_K3      : 1 fused split kernel + 3x torch._int_mm +
                            1 fused int64 merge kernel  = 5 launches.
  STAGE_C_TRUE_FUSED_MMA  : one custom m16n8k32 kernel per logical GEMM
                            (in-kernel split, 3 tensor-core accumulator
                            streams, int64 epilogue, single output write).
  CUSTOM_NATIVE_INT8_M16  : native s8 x s8 -> s32 baseline on the same
                            framework (tile, launch, layouts), M = 16 native.
  NATIVE_PADDED_INT8      : historical F.5B baseline (torch._int_mm, M->32).
  F5B_EAGER_KARATSUBA3    : historical F.5B wide backend (unchanged import).

Correctness is judged against the CPU int64 oracle by the runner; these
classes only produce outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "gpu" / "f5b"))
sys.path.insert(0, str(HERE))

import f5b1_ext  # noqa: E402
from f5b_gpu_backends import TcKaratsuba3, pad_role  # noqa: E402


def w_to_nk(w: torch.Tensor, transpose_b: bool, device="cuda") -> torch.Tensor:
    """Node weight -> logical (N, K) int64 (prepack-time only). `device=None`
    keeps the tensor where it is (used by the local CPU check)."""
    w = w.to(torch.int64)
    if device is not None:
        w = w.to(device)
    return w.T.contiguous() if not transpose_b else w.contiguous()


class F5BEagerKaratsuba3:
    """Historical F.5B eager backend (padded M=32, ~15-20 kernels/GEMM)."""

    name = "F5B_EAGER_KARATSUBA3"

    def __init__(self, w_nk: torch.Tensor):
        self.impl = TcKaratsuba3()
        self.w = w_nk
        self.packed = self.impl.prepack_weights(self.w, transpose_b=True)

    def run(self, a16: torch.Tensor) -> torch.Tensor:
        return self.impl.run(a16.to(torch.int64), self.w, True, self.packed)


class CudaGraphKaratsuba3:
    """Stage A: identical arithmetic to the eager F.5B backend, captured as a
    CUDA graph per shape and replayed (diagnostic: how much of the cost is
    host/driver launch submission?)."""

    name = "STAGE_A_CUDA_GRAPH_K3"

    def __init__(self, a16_example: torch.Tensor, w_nk: torch.Tensor):
        self.impl = TcKaratsuba3()
        self.w = w_nk
        self.packed = self.impl.prepack_weights(self.w, transpose_b=True)
        self.a_static = torch.empty_like(a16_example).cuda()
        self.capture_error: str | None = None
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.impl.run(self.a_static, self.w, True, self.packed)
            torch.cuda.current_stream().wait_stream(side)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                self.out = self.impl.run(self.a_static, self.w, True, self.packed)
        except Exception as exc:  # noqa: BLE001  explicit capture failure
            self.capture_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            self.g = None

    def run(self, a16: torch.Tensor) -> torch.Tensor:
        if self.g is None:
            raise RuntimeError(f"graph capture failed: {self.capture_error}")
        self.a_static.copy_(a16)
        self.g.replay()
        return self.out


class StageB5Launch:
    """Stage B: fused split kernel + 3 torch._int_mm + fused merge kernel.

    The M=16 -> 32 pad of the torch._int_mm boundary remains a *library*
    constraint here (documented, unchanged from F.5B); the point of Stage B
    is only the removal of the pointwise launch cloud around the GEMMs.
    """

    name = "STAGE_B_5LAUNCH_K3"

    def __init__(self, w_nk: torch.Tensor):
        self.ext = f5b1_ext.build()
        self.w = w_nk
        self.N, self.K = int(w_nk.shape[0]), int(w_nk.shape[1])
        self.Kpad = max(32, (self.K + 31) // 32 * 32)
        w0, w1, ws = f5b1_ext.prepack_w10(w_nk)
        # (Kpad, N) contiguous B operands for torch._int_mm
        self.b0 = w0[:, : self.N].t().contiguous()
        self.b1 = w1[:, : self.N].t().contiguous()
        self.bs = ws[:, : self.N].t().contiguous()
        dev = "cuda"
        self.a_pad = torch.zeros((16, self.Kpad), dtype=torch.int16, device=dev)
        self.A0 = torch.zeros((32, self.Kpad), dtype=torch.int8, device=dev)
        self.A1 = torch.zeros((32, self.Kpad), dtype=torch.int8, device=dev)
        self.AS = torch.zeros((32, self.Kpad), dtype=torch.int8, device=dev)

    def run(self, a16: torch.Tensor) -> torch.Tensor:
        self.a_pad[:, : self.K].copy_(a16)
        self.ext.split_a13(self.a_pad, self.A0, self.A1, self.AS)
        c00 = torch._int_mm(self.A0, self.b0)
        c11 = torch._int_mm(self.A1, self.b1)
        cs = torch._int_mm(self.AS, self.bs)
        return self.ext.merge_k3(c00, c11, cs)


class StageCFusedMMA:
    """Stage C: one custom m16n8k32 kernel per logical GEMM, M=16 native."""

    name = "STAGE_C_TRUE_FUSED_MMA"

    def __init__(self, w_nk: torch.Tensor):
        self.ext = f5b1_ext.build()
        self.w = w_nk
        self.N, self.K = int(w_nk.shape[0]), int(w_nk.shape[1])
        self.w0, self.w1, self.ws = f5b1_ext.prepack_w10(w_nk)

    def run(self, a16: torch.Tensor) -> torch.Tensor:
        a16 = a16.contiguous().to(torch.int16)
        return self.ext.wide_gemm(a16, self.w0, self.w1, self.ws, self.N, self.K)


class CustomNativeInt8M16:
    """Custom native s8 x s8 -> s32 baseline: same tile/framework as Stage C,
    M = 16 native (no padding), no decomposition."""

    name = "CUSTOM_NATIVE_INT8_M16"

    def __init__(self, w8_nk: torch.Tensor):
        self.ext = f5b1_ext.build()
        self.w8 = w8_nk
        self.N, self.K = int(w8_nk.shape[0]), int(w8_nk.shape[1])
        self.wp = f5b1_ext.prepack_w8(w8_nk)

    def run(self, a8: torch.Tensor) -> torch.Tensor:
        return self.ext.native_gemm(a8.contiguous().to(torch.int8), self.wp, self.N, self.K)


class NativePaddedInt8:
    """Historical F.5B baseline: torch._int_mm with role-based M->32 zero
    padding (kept for the record; the final gate uses the fastest native)."""

    name = "NATIVE_PADDED_INT8"

    def __init__(self, w8_nk: torch.Tensor):
        self.w8 = w8_nk
        self.N, self.K = int(w8_nk.shape[0]), int(w8_nk.shape[1])
        self.m_pad, self.k_pad, self.n_pad = pad_role(16, self.K, self.N)

    def _pad(self, t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        out = torch.zeros((rows, cols), dtype=t.dtype, device=t.device)
        out[: t.shape[0], : t.shape[1]] = t
        return out

    def run(self, a8: torch.Tensor) -> torch.Tensor:
        ap = self._pad(a8, self.m_pad, self.k_pad)
        wp = self._pad(self.w8, self.n_pad, self.k_pad)   # (Npad, Kpad)
        b = wp.t().contiguous()                            # (Kpad, Npad)
        return torch._int_mm(ap, b)[:16, : self.N]
