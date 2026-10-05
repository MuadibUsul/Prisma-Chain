"""B3-01: the watcher's FrozenVerifier drives the REAL frozen WatcherV2 over
real bundles (the honest + GEMM-fraud pair produced by the frozen tool)."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
for extra in ("watcher", "worker", "network", "compute/canonical/python",
              "compute/gemmv1/python", "tools"):
    sys.path.insert(0, str(REPO / extra))

from prisma_watcher.sources import FrozenVerifier  # noqa: E402

TMP = REPO / "testdata" / "f5c_tmp"


@pytest.fixture(scope="module")
def fixtures():
    honest = TMP / "f5c_g_honest.bin"
    fraud = TMP / "f5c_g_fraud.bin"
    if not (honest.exists() and fraud.exists()):
        pytest.skip("run tools/f5c_real_fraud.py --kind gemm --node 145 "
                    "--out-prefix testdata/f5c_tmp/f5c_g to produce the fixtures")
    task = json.loads((TMP / "f5c_g_honest_task.json").read_text(encoding="utf-8"))
    fraud_task = json.loads((TMP / "f5c_g_fraud_task.json").read_text(encoding="utf-8"))
    return honest, task, fraud, fraud_task


def test_honest_bundle_verifies_clean(fixtures):
    honest, task, _, _ = fixtures
    result = FrozenVerifier(test_seed=7).verify(honest, task)
    assert result["verdict"] == "clean", result["detail"][:400]
    assert "pass" in result["detail"].lower()


def test_fraud_bundle_is_detected(fixtures):
    _, _, fraud, fraud_task = fixtures
    result = FrozenVerifier(test_seed=7).verify(fraud, fraud_task)
    assert result["verdict"] == "fraud", result["detail"][:400]
    report = result["report"]
    assert report["verdict"] in ("fraud_detected", "fraud")


def test_tampered_bundle_against_an_honest_task_is_fraud(fixtures):
    honest, task, fraud, _ = fixtures
    result = FrozenVerifier(test_seed=7).verify(fraud, task)
    assert result["verdict"] == "fraud", result["detail"][:400]
