"""F.5B.2 — torch extension wrapper for the canonical op kernels.

Builds gpu/f5b2/f5b2_kernels.cu CONCATENATED with gpu/f5b1/f5b1_kernels.cu
(the fused A13W10 m16n8k32 GEMM kernels), so one binary provides every
operator the full-block executor needs.  Integer arithmetic only.

Op semantics are faithful ports of tools/f1_canonical_numpy.py:
  requant(x, mult, shift, lo, hi) : rte64(x*mult >> shift) clamp -> int64
  add(a, b)                       : saturating int64 add (int32 range)
  mul(a, b)                       : MulFx (rte20 product, saturate)
  rmsnorm(x [R,H], w [H], eps)    : index-order sum of (x*x)>>20, mean,
                                    frozen invsqrt, MulFx x inv, MulFx w
  rope(x [P,H], table, pairs)     : adjacent-pair table RoPE
  silu(x)                         : MulFx(x, sigmoid(x)), frozen exp
  softmax(x [R,C])                : row max, saturating sub, frozen exp,
                                    index-order sum, ties-even division
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "gpu" / "f5b1"))
sys.path.insert(0, str(HERE))

CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>

extern "C" void f5b2_launch_requant(const int64_t*, int64_t*, long long, int,
                                    long long, long long, long long, cudaStream_t);
extern "C" void f5b2_launch_add(const int64_t*, const int64_t*, int64_t*, long long,
                                cudaStream_t);
extern "C" void f5b2_launch_mul(const int64_t*, const int64_t*, int64_t*, long long,
                                cudaStream_t);
extern "C" void f5b2_launch_rmsnorm(const int64_t*, const int64_t*, int64_t*, int,
                                    long long, long long, cudaStream_t);
extern "C" void f5b2_launch_rope(const int64_t*, const int64_t*, int64_t*, int,
                                 long long, int, cudaStream_t);
extern "C" void f5b2_launch_silu(const int64_t*, int64_t*, long long, cudaStream_t);
extern "C" void f5b2_launch_softmax(const int64_t*, int64_t*, int, long long,
                                    cudaStream_t);
extern "C" void f5b1_launch_wide(const void*, const int8_t*, const int8_t*,
                                 const int8_t*, void*, int, int, cudaStream_t);

torch::Tensor op_requant(torch::Tensor x, int64_t mult, int64_t shift, int64_t lo,
                         int64_t hi) {
  const at::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  long long total = x.numel();
  f5b2_launch_requant(x.data_ptr<int64_t>(), out.data_ptr<int64_t>(),
                      (long long)mult, (int)shift, (long long)lo, (long long)hi,
                      total, at::cuda::getCurrentCUDAStream());
  return out;
}

torch::Tensor op_add(torch::Tensor a, torch::Tensor b) {
  const at::cuda::CUDAGuard guard(a.device());
  auto out = torch::empty_like(a);
  f5b2_launch_add(a.data_ptr<int64_t>(), b.data_ptr<int64_t>(),
                  out.data_ptr<int64_t>(), a.numel(),
                  at::cuda::getCurrentCUDAStream());
  return out;
}

torch::Tensor op_mul(torch::Tensor a, torch::Tensor b) {
  const at::cuda::CUDAGuard guard(a.device());
  auto out = torch::empty_like(a);
  f5b2_launch_mul(a.data_ptr<int64_t>(), b.data_ptr<int64_t>(),
                  out.data_ptr<int64_t>(), a.numel(),
                  at::cuda::getCurrentCUDAStream());
  return out;
}

torch::Tensor op_rmsnorm(torch::Tensor x, torch::Tensor w, int64_t eps_fx) {
  const at::cuda::CUDAGuard guard(x.device());
  long long rows = x.size(0);
  int hidden = (int)x.size(1);
  auto out = torch::empty_like(x);
  f5b2_launch_rmsnorm(x.data_ptr<int64_t>(), w.data_ptr<int64_t>(),
                      out.data_ptr<int64_t>(), hidden, (long long)eps_fx, rows,
                      at::cuda::getCurrentCUDAStream());
  return out;
}

torch::Tensor op_rope(torch::Tensor x, torch::Tensor table, int64_t pairs) {
  const at::cuda::CUDAGuard guard(x.device());
  long long total_pairs = x.numel() / 2;
  int hidden = (int)x.size(1);
  auto out = torch::empty_like(x);
  f5b2_launch_rope(x.data_ptr<int64_t>(), table.data_ptr<int64_t>(),
                   out.data_ptr<int64_t>(), (int)pairs, total_pairs, hidden,
                   at::cuda::getCurrentCUDAStream());
  return out;
}

torch::Tensor op_silu(torch::Tensor x) {
  const at::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  f5b2_launch_silu(x.data_ptr<int64_t>(), out.data_ptr<int64_t>(), x.numel(),
                   at::cuda::getCurrentCUDAStream());
  return out;
}

torch::Tensor op_softmax(torch::Tensor x) {
  const at::cuda::CUDAGuard guard(x.device());
  long long rows = x.size(0);
  int cols = (int)x.size(1);
  auto out = torch::empty_like(x);
  f5b2_launch_softmax(x.data_ptr<int64_t>(), out.data_ptr<int64_t>(), cols, rows,
                      at::cuda::getCurrentCUDAStream());
  return out;
}

torch::Tensor wide_gemm(torch::Tensor A, torch::Tensor W0, torch::Tensor W1,
                        torch::Tensor WS, int64_t N, int64_t K) {
  TORCH_CHECK(A.dtype() == torch::kInt16 && A.is_contiguous());
  const at::cuda::CUDAGuard guard(A.device());
  auto C = torch::empty({16, N}, torch::dtype(torch::kInt64).device(A.device()));
  f5b1_launch_wide(A.data_ptr(), W0.data_ptr<int8_t>(), W1.data_ptr<int8_t>(),
                   WS.data_ptr<int8_t>(), C.data_ptr(), (int)N, (int)K,
                   at::cuda::getCurrentCUDAStream());
  return C;
}
"""

_ext = None


def build(verbose: bool = False):
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline

        cuda_src = ((HERE / "f5b2_kernels.cu").read_text(encoding="utf-8") + "\n"
                    + (REPO / "gpu" / "f5b1" / "f5b1_kernels.cu").read_text(encoding="utf-8"))
        _ext = load_inline(
            name="f5b2_kernels_ext",
            cpp_sources=CPP_SRC,
            cuda_sources=cuda_src,
            functions=["op_requant", "op_add", "op_mul", "op_rmsnorm", "op_rope",
                       "op_silu", "op_softmax", "wide_gemm"],
            extra_cuda_cflags=["-O3",
                               "--generate-code=arch=compute_86,code=sm_86",
                               "--generate-code=arch=compute_89,code=sm_89"],
            verbose=verbose,
        )
    return _ext


# --- prepack (untimed; weights are static) -----------------------------------


def _pad_nk(t: torch.Tensor, N: int, K: int) -> torch.Tensor:
    Npad = (N + 31) // 32 * 32
    Kpad = (K + 31) // 32 * 32
    out = torch.zeros((Npad, Kpad), dtype=torch.int8, device=t.device)
    out[:N, :K] = t.to(torch.int8)
    return out


def prepack_w10(w_nk: torch.Tensor):
    """Logical W10 (N, K) int64 -> padded int8 (W0, W1, Wsum)."""
    from f5b_gpu_backends import split_radix128

    N, K = w_nk.shape
    w0, w1 = split_radix128(w_nk)
    return (_pad_nk(w0, N, K), _pad_nk(w1, N, K), _pad_nk(w0 + w1, N, K))


# --- operator gate ------------------------------------------------------------


def _vec_in(spec: dict, device: str) -> torch.Tensor:
    data = spec["data"]
    shape = spec["desc"]["shape"]
    return torch.tensor(data, dtype=torch.int64, device=device).reshape(shape)


def _params(p) -> dict:
    if isinstance(p, dict):
        return p
    return {k: v for k, v in p}


def run_canonical_vectors(device: str) -> dict:
    """Replay compute/canonical/testdata/canonical_vectors.json through the
    GPU kernels (GEMM_INT8_V1 cases run through the fused A13W10 kernel as a
    superset — same exact products)."""
    doc = json.loads((REPO / "compute" / "canonical" / "testdata" / "canonical_vectors.json")
                     .read_text(encoding="utf-8"))
    ext = build()
    results = []
    all_ok = True
    for entry in doc["operators"]:
        op = entry["operator_id"]
        name = entry["name"]
        params = _params(entry.get("params", {}))
        ins = [_vec_in(i, device) for i in entry["inputs"]]
        expected = torch.tensor(entry["output_data"], dtype=torch.int64,
                                device=device).reshape(entry["output_desc"]["shape"])
        try:
            if op == "ADD_FIXED_V1":
                got = ext.op_add(ins[0], ins[1])
            elif op == "MUL_FIXED_V1":
                got = ext.op_mul(ins[0], ins[1])
            elif op == "REQUANTIZE_V1":
                got = ext.op_requant(ins[0], params["mult"], params["shift"],
                                     params.get("clamp_lo", -(1 << 31)),
                                     params.get("clamp_hi", (1 << 31) - 1))
            elif op == "RMSNORM_FIXED_V1":
                got = ext.op_rmsnorm(ins[0], ins[1], params["eps_fx"])
            elif op == "ROPE_FIXED_V1":
                got = ext.op_rope(ins[0], ins[1], params["half_dim"])
            elif op == "SILU_FIXED_V1":
                got = ext.op_silu(ins[0])
            elif op == "SOFTMAX_FIXED_V1":
                got = ext.op_softmax(ins[0])
            elif op == "GEMM_INT8_V1":
                trans = params.get("transpose_b", 0) == 1
                m = int(ins[0].shape[0])
                a = ins[0].to(torch.int16).contiguous()
                if m < 16:  # fused kernel is M=16-native; zero-pad short rows
                    a16 = torch.zeros((16, int(ins[0].shape[1])), dtype=torch.int16,
                                      device=device)
                    a16[:m] = a
                    a = a16
                w = ins[1].to(torch.int64)
                w_nk = w if trans else w.t().contiguous()
                N, K = int(w_nk.shape[0]), int(w_nk.shape[1])
                w0, w1, ws = prepack_w10(w_nk)
                got = ext.wide_gemm(a, w0, w1, ws, N, K)
                if m < 16:
                    got = got[:m]
            else:
                raise ValueError(f"unknown operator {op}")
            ok = bool(torch.equal(got, expected))
        except Exception as exc:  # noqa: BLE001  explicit
            got = None
            ok = False
            results.append({"op": op, "name": name, "ok": False,
                            "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            all_ok = False
            continue
        all_ok &= ok
        results.append({"op": op, "name": name, "ok": ok,
                        "mismatches": 0 if ok else int((got != expected).sum().item())})
    return {"results": results, "all_exact": bool(all_ok), "source": "canonical_vectors.json"}


def run_adversarial_vectors(device: str) -> dict:
    doc = json.loads((REPO / "testdata" / "f5b2_adversarial_vectors.json")
                     .read_text(encoding="utf-8"))
    ext = build()
    results = []
    all_ok = True
    for v in doc["vectors"]:
        ins = [torch.tensor(i["data"], dtype=torch.int64, device=device).reshape(i["shape"])
               for i in v["inputs"]]
        expected = torch.tensor(v["expected"], dtype=torch.int64, device=device)
        p = v["params"]
        try:
            if v["op"] == "RMSNORM":
                got = ext.op_rmsnorm(ins[0], ins[1], p["eps_fx"])
            elif v["op"] == "REQUANTIZE":
                got = ext.op_requant(ins[0], p["mult"], p["shift"], p["lo"], p["hi"])
            elif v["op"] == "SOFTMAX":
                got = ext.op_softmax(ins[0])
            elif v["op"] == "SILU":
                got = ext.op_silu(ins[0])
            elif v["op"] == "MUL":
                got = ext.op_mul(ins[0], ins[1])
            elif v["op"] == "ADD":
                got = ext.op_add(ins[0], ins[1])
            else:
                raise ValueError(v["op"])
            ok = bool(torch.equal(got.reshape(expected.shape), expected))
            results.append({"op": v["op"], "name": v["name"], "ok": ok,
                            "mismatches": 0 if ok else
                            int((got.reshape(expected.shape) != expected).sum().item())})
        except Exception as exc:  # noqa: BLE001
            results.append({"op": v["op"], "name": v["name"], "ok": False,
                            "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            ok = False
        all_ok &= ok
    return {"results": results, "all_exact": bool(all_ok),
            "source": "f5b2_adversarial_vectors.json"}


def run_wide_gemm_vectors(device: str) -> dict:
    doc = json.loads((REPO / "testdata" / "f5b_wide_gemm_vectors.json")
                     .read_text(encoding="utf-8"))["vectors"]
    ext = build()
    results = []
    all_ok = True
    for v in doc:
        a = torch.tensor(v["a"], dtype=torch.int64, device=device).reshape(v["a_shape"])
        w = torch.tensor(v["w"], dtype=torch.int64, device=device).reshape(v["w_shape"])
        expected = torch.tensor(v["cpu_direct"], dtype=torch.int64,
                                device=device).reshape(v["cpu_direct_shape"])
        try:
            m = int(a.shape[0])
            a16 = a.to(torch.int16)
            if m < 16:  # the fused kernel is M=16-native; pad short rows
                a16 = torch.zeros((16, int(a.shape[1])), dtype=torch.int16, device=device)
                a16[:m] = a.to(torch.int16)
            w_nk = (w.contiguous() if v["transpose_b"] else w.t().contiguous())
            N, K = int(w_nk.shape[0]), int(w_nk.shape[1])
            w0, w1, ws = prepack_w10(w_nk)
            got = ext.wide_gemm(a16, w0, w1, ws, N, K)
            if m < 16:
                got = got[:m]
            ok = bool(torch.equal(got, expected))
            results.append({"name": v["name"], "ok": ok,
                            "mismatches": 0 if ok else int((got != expected).sum().item())})
        except Exception as exc:  # noqa: BLE001
            results.append({"name": v["name"], "ok": False,
                            "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            ok = False
        all_ok &= ok
    return {"results": results, "all_exact": bool(all_ok),
            "source": "f5b_wide_gemm_vectors.json (A13W10 GEMM gate)"}
