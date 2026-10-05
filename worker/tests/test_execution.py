"""B2-06: backend selection, error taxonomy, determinism — with the GPU gate
explicitly recorded as blocked on this host (SM75 < SM86)."""

from __future__ import annotations

import pathlib

import pytest

from prisma_worker import execution, gpu

DOCUMENT = {"nodes": [{"id": "n0", "op": "GEMM"}, {"id": "n1", "op": "REQUANT"},
                      {"id": "n2", "op": "ADD"}]}


def report(capability: str, supported: bool | None = None) -> gpu.CapabilityReport:
    supported = capability in execution.PROVEN_CAPABILITIES if supported is None else supported
    return gpu.CapabilityReport(
        gpu_model=f"device-{capability}", compute_capability=capability, vram_bytes=1,
        driver_version="1", cuda_runtime="12.8", backend=execution.REFERENCE_BACKEND,
        backend_supported=supported, status="CAPABILITY_SUPPORTED" if supported else "CAPABILITY_UNSUPPORTED",
        reason="proven" if supported else f"compute capability {capability} is not a proven target")


def test_proven_capability_selects_the_frozen_backend():
    backend = execution.select_backend(report("8.9"), frozen_gpu_path=pathlib.Path("gpu"))
    assert isinstance(backend, execution.FusedMMA13W10Backend)
    assert backend.is_reference and backend.name == execution.REFERENCE_BACKEND


def test_proven_capability_without_kernel_path_refuses():
    with pytest.raises(execution.BackendUnavailable, match="frozen kernel sources"):
        execution.select_backend(report("8.6"))


def test_unproven_capability_refuses_and_names_the_reason():
    with pytest.raises(execution.BackendUnavailable) as excinfo:
        execution.select_backend(report("7.5"))
    assert "7.5" in str(excinfo.value) and "not a proven target" in str(excinfo.value)


def test_test_backend_requires_an_explicit_flag():
    with pytest.raises(execution.BackendUnavailable):
        execution.select_backend(report("7.5"))
    backend = execution.select_backend(report("7.5"), allow_test_backend=True)
    assert isinstance(backend, execution.TestBackend) and backend.is_reference is False


def test_missing_kernels_on_a_proven_device_is_explicit(tmp_path):
    backend = execution.FusedMMA13W10Backend(tmp_path, capability="8.6")
    with pytest.raises(execution.BackendUnavailable, match="frozen kernel sources"):
        backend.load(DOCUMENT)


def test_test_backend_is_deterministic_and_counts_nodes():
    metrics = execution.ExecutionMetrics()
    backend = execution.TestBackend(metrics=metrics)
    outputs = execution.self_check_determinism(backend, DOCUMENT, {"x": [1, 2, 3]})
    assert set(outputs) == {"n0", "n1", "n2"}
    assert metrics.nodes == 3 * 2 and metrics.ops == {"GEMM": 2, "REQUANT": 2, "ADD": 2}
    assert "wall_ms" in metrics.to_dict()


def test_metrics_measure_a_real_workload():
    """A workload large enough that the wall clock resolves it."""
    document = {"nodes": [{"id": f"n{i}", "op": "ADD"} for i in range(5000)]}
    metrics = execution.ExecutionMetrics()
    backend = execution.TestBackend(metrics=metrics)
    backend.load(document)
    backend.prepack()
    backend.execute({"x": [1]})
    assert metrics.nodes == 5000 and metrics.wall_ms > 0


def test_test_backend_refuses_the_bit_exactness_gate():
    with pytest.raises(execution.BackendUnavailable, match="cannot pass the bit-exactness gate"):
        execution.TestBackend().bit_exact_gate()


def test_compare_with_reference_is_bit_exact_not_tolerant():
    execution.compare_with_reference({"a": "01"}, {"a": "01"})
    with pytest.raises(execution.DeterminismError, match="node a"):
        execution.compare_with_reference({"a": "01"}, {"a": "02"})
    with pytest.raises(execution.DeterminismError, match="node set mismatch"):
        execution.compare_with_reference({"a": "01"}, {"b": "01"})


def test_backend_requires_load_and_prepack_before_execute():
    backend = execution.TestBackend()
    with pytest.raises(execution.BackendError, match="not loaded"):
        backend.execute({})
    backend.load(DOCUMENT)
    with pytest.raises(execution.BackendError, match="not loaded"):
        backend.execute({})
    backend.prepack()
    assert backend.execute({}) != {}


def test_unsupported_graph_is_a_typed_refusal():
    with pytest.raises(execution.UnsupportedShapeError):
        execution.TestBackend().load({"not_nodes": []})


def test_exception_taxonomy_maps_oom_and_device_errors():
    assert isinstance(execution.map_backend_exception(RuntimeError("CUDA out of memory. Tried to allocate")),
                      execution.OutOfMemoryError)
    assert isinstance(execution.map_backend_exception(RuntimeError("CUDA error: an illegal memory access")),
                      execution.DeviceError)
    assert isinstance(execution.map_backend_exception(RuntimeError("unsupported shape 3x5")),
                      execution.UnsupportedShapeError)
    generic = execution.map_backend_exception(RuntimeError("something else entirely"))
    assert type(generic) is execution.BackendError


def test_metrics_record_errors():
    metrics = execution.ExecutionMetrics()
    metrics.record_error("OutOfMemoryError")
    metrics.record_error("OutOfMemoryError")
    assert metrics.to_dict()["errors"] == {"OutOfMemoryError": 2}
