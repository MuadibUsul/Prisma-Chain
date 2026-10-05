"""GPU backend for GEMM_INT8_V1.

The only accepted GPU path is a true INT8 x INT8 -> INT32 kernel
(torch._int_mm). If such a kernel is not available the backend reports
GPU_BACKEND_UNSUPPORTED; it never falls back to floating point, because a
float path cannot honor the bit-exactness requirement.

Inputs and outputs use the canonical layout: A/B as raw int8 values
(Python ints in [-128, 127]), C as flat signed int32 values. Dimensions are
padded to multiples of 16 for torch._int_mm kernel constraints; zero
padding adds only zero products, so the real M x N output stays bit-exact.
"""

Backend = None
try:  # torch is optional; the E2E also runs on the pure-Python path.
    import torch  # type: ignore

    Backend = "torch"
except Exception:  # pragma: no cover - torch not installed
    torch = None  # type: ignore


def backend_status() -> dict:
    if torch is None:
        return {"backend": "GPU_BACKEND_UNSUPPORTED", "reason": "torch not installed"}
    if not torch.cuda.is_available():
        return {"backend": "GPU_BACKEND_UNSUPPORTED", "reason": "no CUDA device"}
    if not hasattr(torch, "_int_mm"):
        return {"backend": "GPU_BACKEND_UNSUPPORTED", "reason": "torch._int_mm unavailable"}
    name = torch.cuda.get_device_name(0)
    return {"backend": "torch_int_mm_cuda", "device": name}


def gpu_gemm(a, b, m: int, n: int, k: int):
    """INT8 GEMM on the GPU. Returns a flat list of int32 values, or raises
    GEMMBackendUnsupported when no exact kernel exists."""
    if torch is None or not torch.cuda.is_available() or not hasattr(torch, "_int_mm"):
        raise GEMMBackendUnsupported(backend_status()["reason"])

    def pad(dim):
        return (dim + 15) // 16 * 16

    mp, kp, np_ = pad(m), pad(k), pad(n)
    a_t = torch.zeros((mp, kp), dtype=torch.int8)
    b_t = torch.zeros((kp, np_), dtype=torch.int8)
    for i in range(m):
        row = a[i * k:(i + 1) * k]
        a_t[i, :k] = torch.tensor(row, dtype=torch.int8)
    for t in range(k):
        row = b[t * n:(t + 1) * n]
        b_t[t, :n] = torch.tensor(row, dtype=torch.int8)
    a_dev, b_dev = a_t.cuda(), b_t.cuda()
    c_dev = torch._int_mm(a_dev, b_dev)
    torch.cuda.synchronize()
    c = c_dev.cpu().tolist()
    out = [0] * (m * n)
    for i in range(m):
        row = c[i][:n]
        out[i * n:(i + 1) * n] = row
    return out


class GEMMBackendUnsupported(RuntimeError):
    """Raised when no bit-exact INT8->INT32 GPU kernel is available."""


__all__ = ["backend_status", "gpu_gemm", "GEMMBackendUnsupported", "Backend"]
