"""Execution backend interface, error taxonomy and selection (B2-06).

Contract first, hardware second:

- **The GPU backend is not a consensus identity.** Consensus is over the
  canonical tensor roots and the frozen arithmetic; a GPU is an accelerator
  whose outputs must match the frozen reference bit-for-bit. A worker on a
  different *proven* device must produce identical roots.
- **No silent fallback.** An unsupported compute capability, a missing
  kernel build, an out-of-memory condition and a device error are all typed
  refusals; nothing here may approximate, and no float path exists.

What runs where:

```text
FusedMMA13W10Backend   SM86/SM89 + the frozen kernel sources (gpu/f5b1,
                       gpu/f5b2) loaded through torch's inline extension.
                       NOT EXERCISED IN THIS ENVIRONMENT: the development
                       host is SM75 — GPU_REAL_SMOKE = BLOCKED (recorded).
TestBackend            deterministic, arithmetic-free placeholder used by
                       unit tests and CI. It is explicitly NOT a reference
                       and refuses the bit-exactness gate.
```

`select_backend` is the only supported way to obtain a backend: it takes the
B2-02 capability report and refuses anything that cannot be bit-exact.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from .graphid import hash_bytes

# Proven device targets (F.5B.1/F.5B.2): the only capabilities allowed to run
# the frozen kernels.
PROVEN_CAPABILITIES = ("8.6", "8.9")
REFERENCE_BACKEND = "TRUE_FUSED_MMA_A13W10"

DOMAIN_NODE_OUTPUT = b"PRISMA_WORKER_TEST_NODE_OUTPUT_V1\x00"


class BackendError(Exception):
    """Base class for execution-backend failures."""


class BackendUnavailable(BackendError):
    """No usable backend on this host (capability, kernels, environment)."""


class UnsupportedShapeError(BackendError):
    """A typed refusal: the request is outside the frozen rules."""


class OutOfMemoryError(BackendError):
    """Device memory exhausted: fail the task, keep the worker alive."""


class DeviceError(BackendError):
    """A device/driver error: explicit, with a reset policy on the caller side."""


class DeterminismError(BackendError):
    """Outputs did not match the expected deterministic reference."""


@dataclass
class ExecutionMetrics:
    nodes: int = 0
    ops: dict = field(default_factory=dict)
    wall_ms: float = 0.0
    memory_high_water_bytes: int = 0
    errors: dict = field(default_factory=dict)

    def record_op(self, kind: str) -> None:
        self.nodes += 1
        self.ops[kind] = self.ops.get(kind, 0) + 1

    def record_error(self, name: str) -> None:
        self.errors[name] = self.errors.get(name, 0) + 1

    def to_dict(self) -> dict:
        return {"nodes": self.nodes, "ops": dict(self.ops), "wall_ms": round(self.wall_ms, 3),
                "memory_high_water_bytes": self.memory_high_water_bytes, "errors": dict(self.errors)}


class ExecutionBackend(Protocol):
    name: str
    is_reference: bool

    def load(self, graph_document: dict, *, weights_dir: Optional[pathlib.Path] = None) -> None:
        ...

    def prepack(self) -> None:
        ...

    def execute(self, inputs: dict) -> dict:
        ...

    def close(self) -> None:
        ...


def map_backend_exception(exc: BaseException) -> BackendError:
    """Translate a backend/SDK exception into the typed taxonomy.

    The mapping is textual on purpose: it must work without torch installed
    (unit tests pass synthetic exceptions) and must never swallow an unknown
    failure into a generic success path.
    """
    message = str(exc)
    lowered = message.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return OutOfMemoryError(message)
    if "cuda error" in lowered or "device-side assert" in lowered or "cublas" in lowered:
        return DeviceError(message)
    if "unsupported" in lowered or "shape" in lowered:
        return UnsupportedShapeError(message)
    return BackendError(message)


# --- the deterministic test backend (NOT a reference) ----------------------

class TestBackend:
    """Deterministic, arithmetic-free placeholder: no GPU, no torch, no floats.

    Node outputs are derived from a domain-separated hash of (node id, op,
    input roots) so repeated runs are byte-identical. It exists so the job
    lifecycle, journal, commit plumbing and CI can be exercised end to end
    before any GPU is involved — it is **not** a reference implementation,
    it is never used unless explicitly allowed, and it refuses the
    bit-exactness gate.
    """

    name = "test-backend"
    is_reference = False

    def __init__(self, metrics: Optional[ExecutionMetrics] = None):
        self.metrics = metrics or ExecutionMetrics()
        self.document: Optional[dict] = None
        self._loaded = False
        self._prepacked = False

    def load(self, graph_document: dict, *, weights_dir: Optional[pathlib.Path] = None) -> None:
        if not isinstance(graph_document, dict) or "nodes" not in graph_document:
            raise UnsupportedShapeError("graph document must carry a nodes list")
        self.document = graph_document
        self._loaded = True

    def prepack(self) -> None:
        if not self._loaded:
            raise BackendError("load() must be called before prepack()")
        self._prepacked = True

    def execute(self, inputs: dict) -> dict:
        if not (self._loaded and self._prepacked):
            raise BackendError("backend is not loaded/prepacked")
        started = time.monotonic()
        outputs: dict = {}
        for node in self.document.get("nodes", []):
            node_id = str(node.get("id", ""))
            kind = str(node.get("op", "unknown"))
            payload = json.dumps({"id": node_id, "op": kind, "inputs": inputs},
                                 sort_keys=True, separators=(",", ":")).encode()
            outputs[node_id] = hash_bytes(DOMAIN_NODE_OUTPUT, payload).hex()
            self.metrics.record_op(kind)
        self.metrics.wall_ms += (time.monotonic() - started) * 1000
        return outputs

    def close(self) -> None:
        self.document = None
        self._loaded = self._prepacked = False

    def bit_exact_gate(self) -> None:
        raise BackendUnavailable(
            "the test backend cannot pass the bit-exactness gate; run the frozen "
            f"{REFERENCE_BACKEND} backend on SM86/SM89 (GPU_REAL_SMOKE is required)")


# --- the frozen GPU backend (loaded, not invented) -------------------------

class FusedMMA13W10Backend:
    """Wraps the frozen GPU kernel sources through torch's inline extension.

    The kernel sources stay in the frozen tree (``gpu/f5b1/f5b1_kernels.cu``,
    ``gpu/f5b2/f5b2_kernels.cu``) and are loaded at runtime; this class never
    re-implements arithmetic. If the sources, torch or a CUDA device are
    missing, the backend reports ``BackendUnavailable`` explicitly.
    """

    name = REFERENCE_BACKEND
    is_reference = True

    def __init__(self, frozen_gpu_path: pathlib.Path, metrics: Optional[ExecutionMetrics] = None,
                 capability: str = ""):
        self.frozen_gpu_path = pathlib.Path(frozen_gpu_path)
        self.metrics = metrics or ExecutionMetrics()
        self.capability = capability
        self._extension = None

    def _torch(self):
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise BackendUnavailable(
                "torch is not installed; the release image ships the CUDA build") from exc
        if not torch.cuda.is_available():
            raise BackendUnavailable("torch reports no CUDA device; refusing to run")
        return torch

    def load(self, graph_document: dict, *, weights_dir: Optional[pathlib.Path] = None) -> None:
        # Static checks first: the error must not depend on the environment
        # (torch present or not) — a worker without the shipped kernels cannot
        # execute, full stop.
        sources = [self.frozen_gpu_path / "f5b1" / "f5b1_kernels.cu",
                   self.frozen_gpu_path / "f5b2" / "f5b2_kernels.cu"]
        missing = [str(p) for p in sources if not p.exists()]
        if missing:
            raise BackendUnavailable(
                "frozen kernel sources are not present: " + ", ".join(missing) +
                " — the release ships them; a worker without them cannot execute")
        torch = self._torch()
        from torch.utils.cpp_extension import load_inline  # type: ignore

        cuda_sources = "\n".join(p.read_text(encoding="utf-8") for p in sources)
        cpp = (self.frozen_gpu_path / "f5b2" / "f5b2_ext.py")
        if not cpp.exists():
            raise BackendUnavailable("f5b2_ext.py (the extension host) is not present")
        try:
            self._extension = load_inline(
                name="prisma_worker_f5b2_kernels_ext",
                cpp_sources=_extract_cpp(cpp),
                cuda_sources=cuda_sources,
                functions=["op_requant", "op_add", "op_mul", "op_rmsnorm", "op_rope",
                           "op_silu", "op_softmax", "wide_gemm"],
                extra_cuda_cflags=["-O3",
                                   f"--generate-code=arch=compute_{self.capability.replace('.', '')},"
                                   f"code=sm_{self.capability.replace('.', '')}"],
            )
        except Exception as exc:  # noqa: BLE001 - re-typed below
            raise map_backend_exception(exc) from exc
        _ = torch

    def prepack(self) -> None:
        if self._extension is None:
            raise BackendError("load() must be called before prepack()")

    def execute(self, inputs: dict) -> dict:
        if self._extension is None:
            raise BackendError("backend is not loaded")
        raise BackendUnavailable(
            "the frozen kernel wiring (graph walk, packing, root computation) is completed in the "
            "GPU-enabled environment; this host cannot exercise it (SM75 < SM86) — "
            "GPU_REAL_SMOKE = BLOCKED")

    def close(self) -> None:
        self._extension = None


def _extract_cpp(ext_host: pathlib.Path) -> str:
    """The C++ shim lives inside f5b2_ext.py as a string literal."""
    text = ext_host.read_text(encoding="utf-8")
    start = text.find('CPP_SRC = """')
    if start == -1:
        raise BackendUnavailable(f"{ext_host} does not define CPP_SRC")
    body = text[start + len('CPP_SRC = """'):]
    end = body.find('"""')
    if end == -1:
        raise BackendUnavailable(f"{ext_host}: unterminated CPP_SRC")
    return body[:end]


# --- selection -------------------------------------------------------------

def select_backend(capability_report, *, frozen_gpu_path: Optional[pathlib.Path] = None,
                   allow_test_backend: bool = False,
                   metrics: Optional[ExecutionMetrics] = None) -> ExecutionBackend:
    """The only supported way to obtain a backend.

    ``capability_report`` is a ``prisma_worker.gpu.CapabilityReport`` (B2-02).
    Any path other than a proven capability plus the frozen kernels is a
    typed refusal; the deterministic test backend requires an explicit flag
    and is recorded as non-reference.
    """
    capability = getattr(capability_report, "compute_capability", "")
    supported = bool(getattr(capability_report, "backend_supported", False))
    if capability in PROVEN_CAPABILITIES and supported:
        if frozen_gpu_path is None:
            raise BackendUnavailable(
                "the frozen kernel sources must be provided (--frozen-gpu-path); "
                "the release ships them next to the worker")
        return FusedMMA13W10Backend(frozen_gpu_path, metrics=metrics, capability=capability)
    if allow_test_backend:
        return TestBackend(metrics=metrics)
    reason = getattr(capability_report, "reason", "capability not proven")
    raise BackendUnavailable(
        f"refusing to execute on compute capability {capability!r}: {reason}. "
        "Pass --allow-test-backend only for lifecycle plumbing tests; it cannot pass the "
        "bit-exactness gate.")


def self_check_determinism(backend: ExecutionBackend, document: dict, inputs: dict,
                           *, runs: int = 2) -> dict:
    """Execute twice and require byte-identical outputs (B2-06 self-check hook)."""
    results = []
    for _ in range(runs):
        backend.load(document)
        backend.prepack()
        results.append({k: str(v) for k, v in backend.execute(inputs).items()})
    first = results[0]
    for other in results[1:]:
        if other != first:
            raise DeterminismError(f"outputs differ between runs: {set(first.items()) ^ set(other.items())}")
    return first


def compare_with_reference(actual: dict, reference: dict) -> None:
    """Bit-exactness gate: any difference is a hard failure, never a tolerance."""
    if set(actual) != set(reference):
        raise DeterminismError(
            f"node set mismatch: only in actual {sorted(set(actual) - set(reference))}, "
            f"only in reference {sorted(set(reference) - set(actual))}")
    for key in sorted(actual):
        if actual[key] != reference[key]:
            raise DeterminismError(f"node {key}: actual {actual[key]} != reference {reference[key]}")
