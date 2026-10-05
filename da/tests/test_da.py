"""B4-01: the DA daemon verifies before storing, re-verifies before serving,
signs frozen attestations, and survives restarts."""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import urllib.request

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
for extra in ("da", "worker", "network", "compute/canonical/python", "compute/gemmv1/python"):
    sys.path.insert(0, str(REPO / extra))

from prisma_da.config import ChainConfig, DaemonConfig  # noqa: E402
from prisma_da.daemon import Daemon  # noqa: E402
from prisma_da.frozen import frozen  # noqa: E402
from prisma_da.storage import ArtifactIndex, IntegrityError, RootMismatch, StorageError  # noqa: E402
from prisma_worker.da import verify_attestation  # noqa: E402
from prisma_worker.identity import WorkerIdentity  # noqa: E402

M, N = 2, 2
TASK_ID = 5
TASK_ID32 = "11" * 32
ASSIGNMENT = "22" * 32
VALUES = [1, -2, 3, 4]


def blob_and_root() -> tuple[bytes, str]:
    values = list(VALUES)
    blob = b"".join(int(v).to_bytes(4, "big", signed=True) for v in values)
    tiles = frozen.tensors.output_tiles(values, M, N)
    cols_c = (N + 7) // 8
    leaves = [frozen.protocol.leaf_output_tile(bytes.fromhex(TASK_ID32), bytes.fromhex(ASSIGNMENT),
                                               i // cols_c, i % cols_c,
                                               frozen.tensors.int32s_to_canonical(tile))
              for i, tile in enumerate(tiles)]
    return blob, frozen.protocol.merkle_root(leaves).hex()


@pytest.fixture()
def daemon(tmp_path):
    config = DaemonConfig(account="prsm1provider", listen_host="127.0.0.1", listen_port=0,
                          data_dir=str(tmp_path / "data"), keystore="",
                          chain=ChainConfig(rpc_urls=["http://127.0.0.1:1"], chain_id="prisma-test-1"))
    identity = WorkerIdentity(protocol_seed=bytes(range(32)), account_scalar=bytes([0x11] * 32))
    return Daemon(config, identity, log=lambda _m: None)


def store_payload(root: str | None = None, blob: bytes | None = None) -> dict:
    blob = blob if blob is not None else blob_and_root()[0]
    root = root if root is not None else blob_and_root()[1]
    return {"task_id": TASK_ID, "c_hex": blob.hex(), "output_root": root, "m": M, "n": N,
            "task_id32": TASK_ID32, "assignment_id": ASSIGNMENT}


def test_store_verifies_then_serves(daemon):
    code, payload = daemon.handle("POST", "/store", store_payload())
    assert code == 200 and payload["stored"] is True
    code, health = daemon.handle("GET", "/health", None)
    assert code == 200 and TASK_ID in health["tasks"]
    assert health["protocol"] == "DA_REPLICA_V1"
    code, served = daemon.handle("GET", f"/v1/da/{TASK_ID}/output", None)
    assert code == 200 and bytes.fromhex(served["c_hex"]) == blob_and_root()[0]
    assert daemon.counters["store_ok"] == 1 and daemon.counters["output_served"] == 1


def test_store_refuses_a_wrong_root(daemon):
    code, payload = daemon.handle("POST", "/store", store_payload(root="00" * 32))
    assert code == 400 and "OUTPUT_ROOT_MISMATCH" in payload["error"]
    assert daemon.counters["store_rejected"] == 1
    assert daemon.handle("GET", "/health", None)[1]["tasks"] == []


def test_store_refuses_a_wrong_length(daemon):
    code, payload = daemon.handle("POST", "/store", store_payload(blob=b"\x00" * 8))
    assert code == 400 and "does not match M*N*4" in payload["error"]


def test_store_refuses_missing_fields(daemon):
    code, payload = daemon.handle("POST", "/store", {"task_id": TASK_ID})
    assert code == 400 and "missing fields" in payload["error"]


def test_attestation_verifies_with_the_worker_verifier(daemon):
    daemon.handle("POST", "/store", store_payload())
    code, payload = daemon.handle("POST", "/attest", {
        "task_id": TASK_ID, "assignment_id": ASSIGNMENT,
        "available_until": 9000, "attested_height": 100})
    assert code == 200
    wire = payload["attestation"]
    canonical = verify_attestation(wire, expected={
        "provider_public": daemon.identity.protocol_public, "task_id": TASK_ID,
        "assignment_id_hex": ASSIGNMENT, "output_root_hex": blob_and_root()[1],
        "output_bytes": M * N * 4, "min_available_until": 9000})
    assert canonical["attested_height"] == 100
    assert daemon.counters["attestations"] == 1


def test_attestation_refused_when_not_stored(daemon):
    code, payload = daemon.handle("POST", "/attest", {
        "task_id": 99, "assignment_id": ASSIGNMENT, "available_until": 9000, "attested_height": 1})
    assert code == 404 and "not stored" in payload["error"]


def test_restart_recovery_rescans_and_serves(tmp_path):
    root = tmp_path / "data"
    index = ArtifactIndex(root)
    index.store(task_id=TASK_ID, c_hex=blob_and_root()[0].hex(), output_root_hex=blob_and_root()[1],
                m=M, n=N, task_id32_hex=TASK_ID32, assignment_id_hex=ASSIGNMENT)
    fresh = ArtifactIndex(root)                      # a restarted daemon
    report = fresh.rescan()
    assert report["scanned"] == 1 and report["serving"] == 1 and report["bad"] == []
    assert fresh.output(TASK_ID) == blob_and_root()[0]


def test_corrupted_artifact_is_quarantined_and_never_served(tmp_path):
    index = ArtifactIndex(tmp_path / "data")
    index.store(task_id=TASK_ID, c_hex=blob_and_root()[0].hex(), output_root_hex=blob_and_root()[1],
                m=M, n=N, task_id32_hex=TASK_ID32, assignment_id_hex=ASSIGNMENT)
    (index.task_dir(TASK_ID) / "output.bin").write_bytes(b"\xff" * 16)   # on-disk corruption
    with pytest.raises(IntegrityError):
        index.output(TASK_ID)
    assert list(index.root.glob("quarantine-*")), "the bad artifact must be moved aside"
    fresh = ArtifactIndex(index.root)
    report = fresh.rescan()
    assert report["serving"] == 0, "a corrupted artifact must never be in the serving set"
    assert report["bad"] == [], "it was already quarantined by the read path"


def test_startup_rescan_quarantines_corruption_it_finds(tmp_path):
    root = tmp_path / "data"
    index = ArtifactIndex(root)
    index.store(task_id=TASK_ID, c_hex=blob_and_root()[0].hex(), output_root_hex=blob_and_root()[1],
                m=M, n=N, task_id32_hex=TASK_ID32, assignment_id_hex=ASSIGNMENT)
    (index.task_dir(TASK_ID) / "output.bin").write_bytes(b"\xff" * 16)
    fresh = ArtifactIndex(root)          # a restarted daemon, without reading first
    report = fresh.rescan()
    assert report["serving"] == 0 and len(report["bad"]) == 1
    assert report["bad"][0]["reason"] == "content hash mismatch"
    assert list(root.glob("quarantine-*")), "the rescan must move the bad artifact aside"


def test_unknown_route_is_explicit(daemon):
    assert daemon.handle("GET", "/nope", None)[0] == 404
    assert daemon.handle("GET", f"/v1/da/{TASK_ID}/tile/x/0", None)[0] == 400
    code, payload = daemon.handle("GET", f"/v1/da/{TASK_ID}/tile/0/0", None)
    assert code == 404 and payload["error"] == "DATA_UNAVAILABLE"


def test_tile_proof_verifies_with_the_frozen_verifier(daemon):
    daemon.handle("POST", "/store", store_payload())
    code, payload = daemon.handle("GET", f"/v1/da/{TASK_ID}/tile/0/0", None)
    assert code == 200
    proof = payload["proof"]
    assert frozen.merkle.verify_inclusion(bytes.fromhex(payload["output_root"]),
                                          bytes.fromhex(payload["leaf"]),
                                          proof["index"], proof["count"],
                                          [bytes.fromhex(s) for s in proof["siblings"]])
    assert len(bytes.fromhex(payload["tile"])) == 8 * 8 * 4      # one 8x8 int32 tile
    assert daemon.counters["tiles_served"] == 1


def test_tile_out_of_range_is_data_unavailable(daemon):
    daemon.handle("POST", "/store", store_payload())
    assert daemon.handle("GET", f"/v1/da/{TASK_ID}/tile/9/9", None)[0] == 404


def test_metrics_count_everything(daemon):
    daemon.handle("POST", "/store", store_payload())
    daemon.handle("GET", f"/v1/da/{TASK_ID}/output", None)
    daemon.handle("GET", "/health", None)
    metrics = daemon.handle("GET", "/metrics", None)[1]
    assert metrics["requests"] >= 3 and metrics["store_ok"] == 1
    assert metrics["index"]["artifacts"] == 1 and metrics["index"]["bytes"] == M * N * 4


def test_http_surface_and_graceful_shutdown(daemon):
    daemon.config.listen_port = 0
    started: dict = {}
    daemon.serve(ready=lambda health: started.update(health), run_forever=False)
    try:
        port = daemon.server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response:
            health = json.loads(response.read().decode())
        assert health["provider"] == "prsm1provider"
        assert started["protocol"] == "DA_REPLICA_V1"
    finally:
        daemon.shutdown(drain_seconds=1.0)
    assert daemon._shutting_down is True
    with pytest.raises(Exception):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
