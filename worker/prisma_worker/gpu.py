"""GPU probe + capability report (B2-02).

The worker executes the frozen canonical graph with the
``TRUE_FUSED_MMA_A13W10`` backend. That backend is bit-exact only on the
compute capabilities proven in Phase F (F.5B.1/F.5B.2): **SM86** (A40) and
**SM89** (RTX 2000 Ada). Anything else is reported as
``CAPABILITY_UNSUPPORTED`` — there is no CPU fallback and no silent
degradation, by frozen rule.

The probe never guesses: if ``nvidia-smi`` is missing or does not report a
compute capability, it fails with an explicit error instead of inventing a
value.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable

BACKEND = "TRUE_FUSED_MMA_A13W10"
CANONICAL_PROFILE = "canonical-graph-v2/a13w10 (Qwen3-0.6B layer0 v2)"

# Compute capability -> the hardware that produced the frozen evidence.
PROVEN_TARGETS: dict[str, str] = {
    "8.6": "NVIDIA A40 (F.5B.2 bit-exact evidence)",
    "8.9": "NVIDIA RTX 2000 Ada Generation (F.5B.2 bit-exact evidence)",
}

_NVIDIA_SMI_QUERY = "name,compute_cap,memory.total,driver_version"


class ProbeError(Exception):
    """The probe could not produce a trustworthy report."""


@dataclass
class CapabilityReport:
    gpu_model: str
    compute_capability: str
    vram_bytes: int
    driver_version: str
    cuda_runtime: str | None
    backend: str
    backend_supported: bool
    status: str
    reason: str
    supported_profiles: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gpu_model": self.gpu_model,
            "compute_capability": self.compute_capability,
            "vram_bytes": self.vram_bytes,
            "driver_version": self.driver_version,
            "cuda_runtime": self.cuda_runtime,
            "backend": self.backend,
            "backend_supported": self.backend_supported,
            "status": self.status,
            "reason": self.reason,
            "supported_profiles": self.supported_profiles,
            "notes": self.notes,
        }


def _default_runner(nvidia_smi: str) -> str:
    proc = subprocess.run(
        [nvidia_smi, f"--query-gpu={_NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20)
    if proc.returncode != 0:
        raise ProbeError(f"nvidia-smi failed: {proc.stderr.strip() or proc.returncode}")
    return proc.stdout


def _detect_cuda_runtime() -> tuple[str | None, list[str]]:
    notes: list[str] = []
    try:
        import torch  # type: ignore

        runtime = getattr(torch.version, "cuda", None)
        if runtime:
            notes.append(f"torch {torch.__version__} reports CUDA runtime {runtime}")
        else:
            notes.append(f"torch {torch.__version__} is a CPU-only build; "
                         "the worker needs the CUDA build with the frozen kernels")
        return runtime, notes
    except ImportError:
        notes.append("torch is not installed in this environment; the worker image ships it "
                     "(the frozen kernels are compiled for SM86/SM89 at runtime)")
        return None, notes


def probe(*, nvidia_smi: str | None = None,
          runner: Callable[[str], str] | None = None,
          cuda_runtime: str | None = None) -> CapabilityReport:
    """Probe the local GPU and return the capability report.

    ``runner`` and ``cuda_runtime`` are injectable for tests; production
    calls use nvidia-smi and detect the torch runtime.
    """
    binary = nvidia_smi or shutil.which("nvidia-smi")
    if not binary and runner is None:
        raise ProbeError("nvidia-smi not found: this host has no NVIDIA GPU (or no driver); "
                         "the worker cannot run — there is no CPU fallback")
    nvidia_smi = binary or "nvidia-smi"
    output = (runner or _default_runner)(nvidia_smi)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ProbeError("nvidia-smi returned no GPU rows; refusing to guess a capability")
    if len(lines) > 1:
        # Multi-GPU hosts: the worker binds one device; report the first and
        # note the others rather than silently picking one.
        pass
    fields = [part.strip() for part in lines[0].split(",")]
    if len(fields) != 4:
        raise ProbeError(f"unexpected nvidia-smi row: {lines[0]!r}")
    model, capability, memory_mib, driver = fields
    if not capability or "." not in capability:
        raise ProbeError(f"nvidia-smi did not report a compute capability ({capability!r}); "
                         "a driver upgrade is required for this query")
    try:
        vram_bytes = int(float(memory_mib)) * 1024 * 1024
    except ValueError as exc:
        raise ProbeError(f"unexpected VRAM value {memory_mib!r}") from exc

    notes: list[str] = []
    if len(lines) > 1:
        notes.append(f"{len(lines)} GPUs detected; this report covers the first ({model}); "
                     "bind the intended device explicitly")
    if cuda_runtime is None:
        cuda_runtime, cuda_notes = _detect_cuda_runtime()
        notes.extend(cuda_notes)

    if capability in PROVEN_TARGETS:
        return CapabilityReport(
            gpu_model=model, compute_capability=capability, vram_bytes=vram_bytes,
            driver_version=driver, cuda_runtime=cuda_runtime, backend=BACKEND,
            backend_supported=True, status="CAPABILITY_SUPPORTED",
            reason=f"compute capability {capability} is a proven target: {PROVEN_TARGETS[capability]}",
            supported_profiles=[CANONICAL_PROFILE], notes=notes)
    return CapabilityReport(
        gpu_model=model, compute_capability=capability, vram_bytes=vram_bytes,
        driver_version=driver, cuda_runtime=cuda_runtime, backend=BACKEND,
        backend_supported=False, status="CAPABILITY_UNSUPPORTED",
        reason=(f"compute capability {capability} is not a proven target "
                f"({', '.join(sorted(PROVEN_TARGETS))} only); the frozen kernels cannot execute "
                "bit-exactly here and no CPU fallback exists"),
        supported_profiles=[], notes=notes)


def format_report(report: CapabilityReport, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(report.to_dict(), indent=1)
    lines = [
        f"status            {report.status}",
        f"gpu               {report.gpu_model} (compute capability {report.compute_capability})",
        f"vram              {report.vram_bytes / (1024 ** 3):.1f} GiB",
        f"driver            {report.driver_version}",
        f"cuda runtime      {report.cuda_runtime or 'not detected'}",
        f"backend           {report.backend}",
        f"supported         {'yes' if report.backend_supported else 'no'}",
        f"reason            {report.reason}",
    ]
    if report.supported_profiles:
        lines.append(f"profiles          {', '.join(report.supported_profiles)}")
    for note in report.notes:
        lines.append(f"note              {note}")
    return "\n".join(lines)
