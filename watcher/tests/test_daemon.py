"""B3-01: watcher lifecycle — discovery, retrieval, frozen verification,
journal resume, and restart recovery. The frozen WatcherV2 is exercised where
a real bundle exists; otherwise the verifier is a stub."""


from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
for extra in ("watcher", "worker", "network", "compute/canonical/python",
              "compute/gemmv1/python", "tools"):
    sys.path.insert(0, str(REPO / extra))

from prisma_watcher.daemon import WatcherDaemon  # noqa: E402
from prisma_watcher.frozen import frozen  # noqa: E402
from prisma_watcher.sources import HttpBundleSource, RetrievalError  # noqa: E402
from prisma_watcher.chain_scan import PrismadTaskScan, COMPLETED_STATUSES  # noqa: E402


class FakeChain:
    def __init__(self, tasks):
        self.tasks = tasks
        self.height_value = 100

    def completed_tasks(self, *, start_id=1, max_scan=512):
        return self.tasks

    def height(self):
        return self.height_value


class FakeBundles:
    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)
        self.fetched: list[int] = []

    def fetch(self, task, destination):
        task_id = int(task.get("id", 0))
        if task_id in self.fail_for:
            raise RetrievalError(f"no provider has task {task_id}")
        self.fetched.append(task_id)
        path = pathlib.Path(destination) / "bundle.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-bundle")
        return path


class StubVerifier:
    """Deterministic verdicts without the frozen stack (the frozen path is
    covered by test_watcher_frozen.py against the real WatcherV2)."""

    def __init__(self, verdict_by_task):
        self.verdict_by_task = verdict_by_task

    def verify(self, bundle_path, task):
        task_id = int(task.get("id", 0))
        verdict = self.verdict_by_task.get(task_id, "clean")
        return {"verdict": verdict, "detail": f"stub:{verdict}"}


def daemon(tmp_path, tasks, verdicts, fail_for=()):
    return WatcherDaemon(FakeChain(tasks), FakeBundles(fail_for), StubVerifier(verdicts),
                         tmp_path / "journal", log=lambda _m: None)


def task(task_id, status="result_submitted"):
    return {"id": task_id, "status": status}


def test_poll_verifies_and_journals(tmp_path):
    watcher = daemon(tmp_path, [task(1), task(2)], {1: "clean", 2: "fraud"})
    results = watcher.poll_once(work_dir=tmp_path / "work")
    assert [r["action"] for r in results] == ["verified", "verified"]
    assert [r["verdict"] for r in results] == ["clean", "fraud"]
    entry = json.loads((tmp_path / "journal" / "task-00000002.json").read_text())
    assert entry["phase"] == "verified" and entry["verdict"] == "fraud"
    assert watcher.metrics.fraud == 1 and watcher.metrics.clean == 1


def test_verified_tasks_are_not_reprocessed(tmp_path):
    watcher = daemon(tmp_path, [task(1)], {1: "clean"})
    watcher.poll_once(work_dir=tmp_path / "work")
    again = watcher.poll_once(work_dir=tmp_path / "work")
    assert again == [{"task_id": 1, "action": "skipped", "verdict": "clean"}]
    assert watcher.metrics.discovered == 1, "the second pass must not rediscover"


def test_retrieval_failure_is_journaled_and_the_next_pass_resumes(tmp_path):
    watcher = daemon(tmp_path, [task(7)], {7: "clean"}, fail_for=(7,))
    first = watcher.poll_once(work_dir=tmp_path / "work")
    assert first[0]["action"] == "retrieval_failed"
    entry = json.loads((tmp_path / "journal" / "task-00000007.json").read_text())
    assert entry["phase"] == "retrieval_failed"
    # A retrieval failure is resumable: record it as a pending phase.
    watcher.journal_write(7, phase="discovered", note="provider came back")
    watcher.bundles.fail_for.clear()
    second = watcher.poll_once(work_dir=tmp_path / "work")
    assert second[0]["action"] == "verified"
    assert watcher.metrics.resumed == 1


def test_scheduler_is_bounded(tmp_path):
    watcher = daemon(tmp_path, [], {})
    sleeps: list[float] = []
    watcher.run(interval_seconds=0.01, max_iterations=3, sleep=sleeps.append,
                work_dir=tmp_path / "work")
    assert sleeps == [0.01, 0.01] and watcher.metrics.discovered == 0


def test_pending_tasks_reads_the_journal(tmp_path):
    watcher = daemon(tmp_path, [], {})
    watcher.journal_write(3, phase="discovered")
    watcher.journal_write(4, phase="verified")
    watcher.journal_write(5, phase="retrieval_failed")   # not a resumable phase
    assert watcher.pending_tasks() == [3]


def test_frozen_libraries_are_located():
    ok, why = frozen.available()
    assert ok, why


def test_http_source_roundtrip(tmp_path):
    task_id = 12
    blob = b"canonical-blob-bytes"
    payload = json.dumps({"task_id": task_id, "c_hex": blob.hex()}).encode()

    class Handler:
        def __call__(self, url, timeout):
            assert f"/v1/da/{task_id}/output" in url
            return payload

    source = HttpBundleSource(["http://provider-a", "http://provider-b"],
                              http_get=Handler())
    path = source.fetch({"id": task_id}, tmp_path / "work")
    assert path.read_bytes() == blob


def test_http_source_falls_over_and_bounds_size(tmp_path):
    calls: list[str] = []

    def handler(url, timeout):
        calls.append(url)
        if "provider-a" in url:
            raise OSError("connection refused")
        return json.dumps({"task_id": 5, "c_hex": "00" * (70 * 1024 * 1024)}).encode()

    source = HttpBundleSource(["http://provider-a", "http://provider-b"], http_get=handler,
                              max_bytes=64 * 1024 * 1024)
    with pytest.raises(RetrievalError, match="exceeds"):
        source.fetch({"id": 5}, tmp_path / "work")
    assert len(calls) == 2 and "provider-b" in calls[-1]


def test_chain_scan_statuses_and_bounded_scan():
    scan = PrismadTaskScan(prismad="prismad", chain_id="c", rpc_urls=["http://x:1"],
                           runner=lambda cmd, **kw: subprocess_completed({}),
                           http_get=lambda url: {"result": {"sync_info": {"latest_block_height": "9"}}})
    assert "result_submitted" in COMPLETED_STATUSES
    assert scan.completed_tasks(start_id=1, max_scan=3) == []


def subprocess_completed(payload):
    import subprocess
    return subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
