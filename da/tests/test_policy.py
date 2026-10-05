"""B4-02: quota, TTL, safe GC and the integrity scan."""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
for extra in ("da", "worker", "network", "compute/canonical/python", "compute/gemmv1/python"):
    sys.path.insert(0, str(REPO / extra))

from prisma_da.policy import (  # noqa: E402
    QuotaExceeded, RetentionPolicy, enforce_quota, gc, integrity_scan, ttl_for)
from prisma_da.storage import ArtifactIndex  # noqa: E402
from prisma_da.frozen import frozen  # noqa: E402

M, N = 2, 2
TASK_ID = 9
TASK_ID32 = "33" * 32
ASSIGNMENT = "44" * 32
VALUES = [5, 6, 7, 8]


def blob_and_root(values=VALUES):
    blob = b"".join(int(v).to_bytes(4, "big", signed=True) for v in values)
    tiles = frozen.tensors.output_tiles(list(values), M, N)
    cols_c = (N + 7) // 8
    leaves = [frozen.protocol.leaf_output_tile(bytes.fromhex(TASK_ID32), bytes.fromhex(ASSIGNMENT),
                                               i // cols_c, i % cols_c,
                                               frozen.tensors.int32s_to_canonical(tile))
              for i, tile in enumerate(tiles)]
    return blob, frozen.protocol.merkle_root(leaves).hex()


def stored(tmp_path, task_id=TASK_ID, values=VALUES) -> ArtifactIndex:
    index = ArtifactIndex(tmp_path / "data")
    blob, root = blob_and_root(values)
    index.store(task_id=task_id, c_hex=blob.hex(), output_root_hex=root, m=M, n=N,
                task_id32_hex=TASK_ID32, assignment_id_hex=ASSIGNMENT)
    return index


def test_quota_rejects_before_writing(tmp_path):
    index = stored(tmp_path)
    with pytest.raises(QuotaExceeded, match="quota exceeded"):
        enforce_quota(index, 1024, quota_bytes=index.entries[TASK_ID].bytes + 512)
    enforce_quota(index, 512, quota_bytes=index.entries[TASK_ID].bytes + 512)     # exactly fits
    enforce_quota(index, 10 ** 6, quota_bytes=0)                                  # unlimited


def test_gc_never_deletes_without_a_ttl(tmp_path):
    index = stored(tmp_path)
    report = gc(index, RetentionPolicy(ttl_seconds=0), now_ms=10 ** 15)
    assert report["deleted"] == [] and "ttl 0" in report["kept"][0]["reason"]
    assert index.output(TASK_ID) is not None


def test_gc_respects_ttl_and_live_challenges(tmp_path):
    index = stored(tmp_path)
    original = index.entries[TASK_ID].stored_at_ms      # capture before mutating
    index.entries[TASK_ID].stored_at_ms = original - 5_000_000
    policy = RetentionPolicy(ttl_seconds=3600)
    live = gc(index, policy, live_challenges=[TASK_ID], now_ms=original)
    assert live["deleted"] == [] and "live challenge" in live["kept"][0]["reason"]
    dry = gc(index, policy, now_ms=original, dry_run=True)
    assert dry["deleted"] and dry["deleted"][0]["dry_run"] is True
    assert index.output(TASK_ID) is not None, "a dry run must not delete anything"
    done = gc(index, policy, now_ms=original)
    assert len(done["deleted"]) == 1 and done["freed_bytes"] == M * N * 4
    assert index.output(TASK_ID) is None and not index.task_dir(TASK_ID).exists()


def test_gc_keeps_a_fresh_artifact(tmp_path):
    index = stored(tmp_path)
    policy = RetentionPolicy(ttl_seconds=60)
    report = gc(index, policy, now_ms=index.entries[TASK_ID].stored_at_ms + 1_000)
    assert report["deleted"] == [] and "ttl" in report["kept"][0]["reason"]


def test_per_class_ttl_overrides_the_default(tmp_path):
    assert ttl_for(RetentionPolicy(ttl_seconds=10, class_ttl_seconds={"challenge_relevant": 99}),
                   {"artifact_class": "challenge_relevant"}) == 99
    assert ttl_for(RetentionPolicy(ttl_seconds=10), {"artifact_class": "output"}) == 10


def test_integrity_scan_quarantines_and_reports(tmp_path):
    index = stored(tmp_path)
    healthy = integrity_scan(index)
    assert healthy == {"checked": 1, "bad": [], "healthy": 1}
    (index.task_dir(TASK_ID) / "output.bin").write_bytes(b"\x01" * 16)
    report = integrity_scan(index)
    assert report["checked"] == 1 and len(report["bad"]) == 1 and report["healthy"] == 0
    assert list(index.root.glob("quarantine-*"))


def test_daemon_rejects_an_upload_over_quota(tmp_path):
    from prisma_da.config import ChainConfig, DaemonConfig
    from prisma_da.daemon import Daemon
    from prisma_worker.identity import WorkerIdentity

    config = DaemonConfig(account="prsm1provider", data_dir=str(tmp_path / "data"),
                          quota_bytes=8, chain=ChainConfig(chain_id="c"))
    identity = WorkerIdentity(protocol_seed=bytes(range(32)), account_scalar=bytes([0x11] * 32))
    daemon = Daemon(config, identity, log=lambda _m: None)
    blob, root = blob_and_root()
    code, payload = daemon.handle("POST", "/store", {
        "task_id": TASK_ID, "c_hex": blob.hex(), "output_root": root, "m": M, "n": N,
        "task_id32": TASK_ID32, "assignment_id": ASSIGNMENT})
    assert code == 413 and "quota exceeded" in payload["error"]
    assert daemon.counters["quota_rejected"] == 1
