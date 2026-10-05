"""B2-02: the GPU probe must report honestly and never invent a capability."""

from __future__ import annotations

import json

import pytest

from prisma_worker import gpu

A40 = "NVIDIA A40, 8.6, 46068, 550.54.15\n"
RTX_2000_ADA = "NVIDIA RTX 2000 Ada Generation, 8.9, 16380, 550.54.15\n"
RTX_2060_SUPER = "NVIDIA GeForce RTX 2060 SUPER, 7.5, 8192, 616.92\n"


def test_proven_target_is_supported():
    report = gpu.probe(runner=lambda _: A40, cuda_runtime="12.8")
    assert report.status == "CAPABILITY_SUPPORTED"
    assert report.backend_supported and report.backend == "TRUE_FUSED_MMA_A13W10"
    assert report.supported_profiles and "canonical-graph-v2" in report.supported_profiles[0]
    assert report.vram_bytes == 46068 * 1024 * 1024
    assert report.cuda_runtime == "12.8"


def test_second_proven_target_is_supported():
    report = gpu.probe(runner=lambda _: RTX_2000_ADA, cuda_runtime="12.8")
    assert report.status == "CAPABILITY_SUPPORTED"


def test_unproven_capability_is_explicitly_unsupported():
    report = gpu.probe(runner=lambda _: RTX_2060_SUPER, cuda_runtime="12.8")
    assert report.status == "CAPABILITY_UNSUPPORTED"
    assert report.backend_supported is False
    assert report.supported_profiles == []
    assert "8.6" in report.reason and "8.9" in report.reason
    assert "no CPU fallback" in report.reason


def test_missing_nvidia_smi_is_an_explicit_error(monkeypatch):
    monkeypatch.setattr(gpu.shutil, "which", lambda _: None)
    with pytest.raises(gpu.ProbeError) as excinfo:
        gpu.probe()
    assert "no CPU fallback" in str(excinfo.value)


def test_malformed_rows_are_refused():
    with pytest.raises(gpu.ProbeError):
        gpu.probe(runner=lambda _: "")
    with pytest.raises(gpu.ProbeError):
        gpu.probe(runner=lambda _: "NVIDIA A40, 8.6\n")
    with pytest.raises(gpu.ProbeError):
        gpu.probe(runner=lambda _: "NVIDIA A40, N/A, 46068, 550\n")
    with pytest.raises(gpu.ProbeError):
        gpu.probe(runner=lambda _: "NVIDIA A40, 8.6, not-a-number, 550\n")


def test_multi_gpu_reports_first_with_a_note():
    report = gpu.probe(runner=lambda _: A40 + RTX_2060_SUPER, cuda_runtime="12.8")
    assert report.gpu_model.startswith("NVIDIA A40")
    assert any("2 GPUs detected" in note for note in report.notes)


def test_report_serialises_and_formats():
    report = gpu.probe(runner=lambda _: A40, cuda_runtime="12.8")
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["compute_capability"] == "8.6"
    text = gpu.format_report(report)
    assert "CAPABILITY_SUPPORTED" in text and "TRUE_FUSED_MMA_A13W10" in text


def test_cli_gpu_probe_paths(monkeypatch, capsys):
    """The CLI must reach the probe without identity flags (regression guard)."""
    from prisma_worker import cli

    monkeypatch.setattr(gpu, "probe", lambda **_kw: gpu.CapabilityReport(
        gpu_model="NVIDIA A40", compute_capability="8.6", vram_bytes=1, driver_version="550",
        cuda_runtime="12.8", backend=gpu.BACKEND, backend_supported=True,
        status="CAPABILITY_SUPPORTED", reason="proven", supported_profiles=[gpu.CANONICAL_PROFILE]))
    assert cli.main(["gpu", "probe", "--json"]) == 0
    assert '"CAPABILITY_SUPPORTED"' in capsys.readouterr().out

    monkeypatch.setattr(gpu, "probe", lambda **_kw: gpu.CapabilityReport(
        gpu_model="NVIDIA GeForce RTX 2060 SUPER", compute_capability="7.5", vram_bytes=1,
        driver_version="616", cuda_runtime="12.8", backend=gpu.BACKEND, backend_supported=False,
        status="CAPABILITY_UNSUPPORTED", reason="not proven"))
    assert cli.main(["gpu", "probe"]) == 2  # distinct exit code for unsupported

    def boom(**_kw):
        raise gpu.ProbeError("nvidia-smi not found")

    monkeypatch.setattr(gpu, "probe", boom)
    assert cli.main(["gpu", "probe"]) == 1
