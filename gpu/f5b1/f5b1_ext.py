"""F.5B.1 — torch wrapper around the exact integer CUDA kernels (research-only).

Builds gpu/f5b1/f5b1_kernels.cu via torch.utils.cpp_extension.load_inline
(nvcc on the pod; one binary compiled for sm_86 and sm_89).  All arithmetic
is integer-only; no float anywhere in the canonical path.

Kernel API (all exact):
  wide_gemm(A13_i16, W0_i8, W1_i8, WS_i8, N, K) -> int64 (16, N)
  native_gemm(A8_i8, W8_i8, N, K)               -> int32 (16, N)
  split_a13(A13_i16, A0, A1, AS)   (Stage B helper; writes rows 0..15)
  merge_k3(C00_i32, C11_i32, CS_i32) -> int64 (16, N)   (Stage B helper)

Prepack (untimed, at load): W stored as int8 (Npad, Kpad) row-major, k
contiguous — the "col-major B" operand of mma.row.col — with Npad/Kpad
zero-padded to multiples of 32 (mathematically inert).
"""

from __future__ import annotations

from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CU_SRC_PATH = HERE / "f5b1_kernels.cu"

CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>

extern "C" void f5b1_launch_wide(const void*, const int8_t*, const int8_t*,
                                 const int8_t*, void*, int, int, cudaStream_t);
extern "C" void f5b1_launch_native(const void*, const int8_t*, void*, int, int,
                                   cudaStream_t);
extern "C" void f5b1_launch_split(const int16_t*, int8_t*, int8_t*, int8_t*,
                                  int, cudaStream_t);
extern "C" void f5b1_launch_merge(const int32_t*, const int32_t*, const int32_t*,
                                  int64_t*, int, cudaStream_t);

torch::Tensor wide_gemm(torch::Tensor A, torch::Tensor W0, torch::Tensor W1,
                        torch::Tensor WS, int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kInt16 && A.is_contiguous());
  TORCH_CHECK(W0.is_cuda() && W0.dtype() == torch::kInt8 && W0.is_contiguous());
  const at::cuda::CUDAGuard guard(A.device());
  auto C = torch::empty({16, N}, torch::dtype(torch::kInt64).device(A.device()));
  f5b1_launch_wide(A.data_ptr(), W0.data_ptr<int8_t>(), W1.data_ptr<int8_t>(),
                   WS.data_ptr<int8_t>(), C.data_ptr(), (int)N, (int)K,
                   at::cuda::getCurrentCUDAStream());
  return C;
}

torch::Tensor native_gemm(torch::Tensor A, torch::Tensor W, int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kInt8 && A.is_contiguous());
  TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kInt8 && W.is_contiguous());
  const at::cuda::CUDAGuard guard(A.device());
  auto C = torch::empty({16, N}, torch::dtype(torch::kInt32).device(A.device()));
  f5b1_launch_native(A.data_ptr(), W.data_ptr<int8_t>(), C.data_ptr(), (int)N,
                     (int)K, at::cuda::getCurrentCUDAStream());
  return C;
}

void split_a13(torch::Tensor A, torch::Tensor A0, torch::Tensor A1, torch::Tensor AS) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kInt16 && A.is_contiguous());
  const at::cuda::CUDAGuard guard(A.device());
  f5b1_launch_split(A.data_ptr<int16_t>(), A0.data_ptr<int8_t>(),
                    A1.data_ptr<int8_t>(), AS.data_ptr<int8_t>(),
                    (int)A.size(1), at::cuda::getCurrentCUDAStream());
}

torch::Tensor merge_k3(torch::Tensor C00, torch::Tensor C11, torch::Tensor CS) {
  TORCH_CHECK(C00.is_cuda() && C00.dtype() == torch::kInt32 && C00.is_contiguous());
  const at::cuda::CUDAGuard guard(C00.device());
  auto C = torch::empty({16, C00.size(1)}, torch::dtype(torch::kInt64).device(C00.device()));
  f5b1_launch_merge(C00.data_ptr<int32_t>(), C11.data_ptr<int32_t>(),
                    CS.data_ptr<int32_t>(), C.data_ptr<int64_t>(),
                    (int)C00.size(1), at::cuda::getCurrentCUDAStream());
  return C;
}
"""

_ext = None


def build(verbose: bool = False):
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline

        cuda_src = CU_SRC_PATH.read_text(encoding="utf-8")
        _ext = load_inline(
            name="f5b1_kernels_ext",
            cpp_sources=CPP_SRC,
            cuda_sources=cuda_src,
            functions=["wide_gemm", "native_gemm", "split_a13", "merge_k3"],
            extra_cuda_cflags=[
                "-O3",
                "--generate-code=arch=compute_86,code=sm_86",
                "--generate-code=arch=compute_89,code=sm_89",
            ],
            verbose=verbose,
        )
    return _ext


def available() -> bool:
    try:
        build()
        return True
    except Exception:  # noqa: BLE001  (CPU machines: no nvcc/CUDA)
        return False


# --- prepack (untimed; weights are static) ---------------------------------


def _pad_nk(t: torch.Tensor, N: int, K: int) -> torch.Tensor:
    Npad = (N + 31) // 32 * 32
    Kpad = (K + 31) // 32 * 32
    out = torch.zeros((Npad, Kpad), dtype=torch.int8, device=t.device)
    out[:N, :K] = t.to(torch.int8)
    return out


def prepack_w10(w_nk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Logical W10 (N, K) int64 -> padded int8 (W0, W1, Wsum) via the frozen
    balanced radix-128 split (identical code path to the F.5B backends)."""
    import sys
    sys.path.insert(0, str(REPO / "gpu" / "f5b"))
    from f5b_gpu_backends import split_radix128  # noqa: PLC0415

    N, K = w_nk.shape
    w0, w1 = split_radix128(w_nk)
    return (_pad_nk(w0, N, K), _pad_nk(w1, N, K), _pad_nk(w0 + w1, N, K))


def prepack_w8(w_nk: torch.Tensor) -> torch.Tensor:
    N, K = w_nk.shape
    return _pad_nk(w_nk, N, K)


# --- self-test (run on the pod before any evidence is produced) -------------


def selftest() -> dict:
    """Exactness probes for the kernels themselves, including one-hot
    layout diagnostics (mma fragment maps).  Any failure here aborts the
    harness before it produces evidence."""
    ext = build()
    dev = "cuda"
    rng = torch.Generator(device="cpu").manual_seed(20261005)
    out: dict = {"cases": [], "ok": True}

    # split kernel: int16 -> (A0, A1, AS), exact reconstruction
    a13 = torch.randint(-4096, 4096, (16, 96), generator=rng).to(torch.int16).to(dev)
    A32 = torch.zeros((32, 96), dtype=torch.int8, device=dev)
    A0 = torch.zeros_like(A32)
    A1 = torch.zeros_like(A32)
    AS = torch.zeros_like(A32)
    ext.split_a13(a13, A0, A1, AS)
    rec = A0[:16].to(torch.int64) + 128 * A1[:16].to(torch.int64)
    ok = torch.equal(rec, a13.to(torch.int64)) and torch.equal(
        AS[:16].to(torch.int64), A0[:16].to(torch.int64) + A1[:16].to(torch.int64))
    out["cases"].append({"name": "split_exact", "ok": bool(ok)})
    out["ok"] &= bool(ok)

    # native kernel: random exactness + one-hot layout probe
    for (N, K) in [(16, 32), (16, 16), (128, 16), (64, 96), (128, 128)]:
        a8 = torch.randint(-127, 128, (16, K), generator=rng).to(torch.int8).to(dev)
        w8 = torch.randint(-127, 128, (N, K), generator=rng).to(torch.int8)
        C = ext.native_gemm(a8, prepack_w8(w8).to(dev), N, K)
        ref = a8.to(torch.int64).cpu() @ w8.to(torch.int64).T
        ok = torch.equal(C.cpu().to(torch.int64), ref)
        out["cases"].append({"name": f"native_exact_{N}x{K}", "ok": bool(ok),
                             "first_mismatch": None if ok else _first_mismatch(C.cpu(), ref)})
        out["ok"] &= bool(ok)

    # one-hot probes (diagnosable layout failures)
    for (i, j) in [(0, 0), (5, 3), (15, 31), (7, 17)]:
        a8 = torch.zeros((16, 32), dtype=torch.int8, device=dev)
        a8[i, j] = 1
        w8 = torch.arange(16 * 32, dtype=torch.int64).reshape(16, 32) % 97 - 48
        C = ext.native_gemm(a8, prepack_w8(w8.to(torch.int8)).to(dev), 16, 32)
        ref = a8.to(torch.int64).cpu() @ w8.T
        ok = torch.equal(C.cpu().to(torch.int64), ref)
        out["cases"].append({"name": f"native_onehot_{i}_{j}", "ok": bool(ok),
                             "first_mismatch": None if ok else _first_mismatch(C.cpu(), ref)})
        out["ok"] &= bool(ok)

    # wide kernel: random + adversarial extremes
    cases = [
        ("wide_random", lambda N, K: (
            torch.randint(-4096, 4096, (16, K), generator=rng).to(dev),
            torch.randint(-512, 512, (N, K), generator=rng).to(dev))),
        ("wide_all_max", lambda N, K: (
            torch.full((16, K), 4095, dtype=torch.int64, device=dev),
            torch.full((N, K), 511, dtype=torch.int64, device=dev))),
        ("wide_all_min", lambda N, K: (
            torch.full((16, K), -4096, dtype=torch.int64, device=dev),
            torch.full((N, K), -512, dtype=torch.int64, device=dev))),
        ("wide_mixed", lambda N, K: (
            (torch.arange(16 * K, device=dev).reshape(16, K) % 2) * 8191 - 4096,
            (torch.arange(N * K, device=dev).reshape(N, K) % 2) * 1023 - 512)),
    ]
    for name, gen in cases:
        for (N, K) in [(16, 32), (128, 16), (128, 1024), (1024, 128)]:
            a13, w10 = gen(N, K)
            a16 = a13.to(torch.int16).contiguous()
            w0, w1, ws = prepack_w10(w10.contiguous())
            C = ext.wide_gemm(a16, w0, w1, ws, N, K)
            ref = a13.cpu() @ w10.cpu().T
            ok = torch.equal(C.cpu(), ref)
            out["cases"].append({"name": f"{name}_{N}x{K}", "ok": bool(ok),
                                 "first_mismatch": None if ok else _first_mismatch(C.cpu(), ref)})
            out["ok"] &= bool(ok)

    # merge kernel
    N = 64
    c00 = torch.randint(-12582912, 12582913, (16, N), generator=rng).to(torch.int32).to(dev)
    c11 = torch.randint(-393216, 393217, (16, N), generator=rng).to(torch.int32).to(dev)
    cs = torch.randint(-20054016, 20054017, (16, N), generator=rng).to(torch.int32).to(dev)
    C = ext.merge_k3(c00, c11, cs)
    ref = c00.cpu().to(torch.int64) + ((cs.cpu().to(torch.int64) - c00.cpu().to(torch.int64)
                                        - c11.cpu().to(torch.int64)) << 7) + (c11.cpu().to(torch.int64) << 14)
    ok = torch.equal(C.cpu(), ref)
    out["cases"].append({"name": "merge_exact", "ok": bool(ok)})
    out["ok"] &= bool(ok)
    return out


def _first_mismatch(got: torch.Tensor, ref: torch.Tensor) -> dict | None:
    bad = (got != ref)
    if not bool(bad.any()):
        return None
    idx = bad.nonzero()[0].tolist()
    return {"index": idx, "got": int(got[tuple(idx)]), "expected": int(ref[tuple(idx)]),
            "mismatch_count": int(bad.sum())}
