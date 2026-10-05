"""B2-04: discovery, compatibility, journal and exactly-once accept."""

from __future__ import annotations

import json
import pathlib

import pytest

from prisma_worker import chain as chain_mod
from prisma_worker import jobs


def task_payload(task_id=7, status="posted", mode="verifiable", model_id="qwen3-0.6b-layer0-v2",
                 spec_version="v1", deadline=1000, worker=""):
    return {"id": task_id, "status": status, "mode": mode, "model_id": model_id,
            "spec_version": spec_version, "requester": "prsm1requester", "worker": worker,
            "deadline": deadline, "reserved_bond": 500, "max_fee": 100}


class FakeClient:
    def __init__(self, tasks: dict[int, dict], height=100, chain_id="prisma-testnet-1"):
        self.tasks = tasks
        self.height = height
        self.chain_id = chain_id
        self.accept_calls: list[int] = []

    def task(self, task_id):
        return self.tasks.get(task_id, {})

    def status(self):
        return {"sync_info": {"latest_block_height": str(self.height)}}

    def accept_task(self, task_id, account_scalar_hex):
        self.accept_calls.append(task_id)
        self.tasks[task_id] = {**self.tasks.get(task_id, task_payload(task_id)),
                               "worker": "prsm1worker", "status": "accepted"}
        return "ACCEPTTX"

    def validate_chain_id(self):
        return self.chain_id


@pytest.fixture()
def journal(tmp_path: pathlib.Path) -> jobs.Journal:
    return jobs.Journal(tmp_path / "journal")


def test_discovery_scans_and_stops_after_a_run_of_misses(journal):
    client = FakeClient({1: task_payload(1), 3: task_payload(3), 4: task_payload(4, status="finalized")})
    found = jobs.discover(client, start_id=1, max_scan=20, max_consecutive_missing=3,
                          log=lambda _m: None)
    assert [t.task_id for t in found] == [1, 3]  # open tasks only, missing ids skipped


def test_discovery_respects_the_scan_bound():
    client = FakeClient({i: task_payload(i) for i in range(1, 50)})
    found = jobs.discover(client, start_id=1, max_scan=5, max_consecutive_missing=10,
                          log=lambda _m: None)
    assert len(found) == 5


def test_compatibility_rejects_wrong_model_mode_and_status():
    ok, _ = jobs.compatibility(jobs.TaskSummary.from_chain(task_payload()))
    assert ok
    for mutate, fragment in (({"model_id": "other-model"}, "model"),
                             ({"mode": "lightweight"}, "mode"),
                             ({"status": "finalized"}, "not open")):
        ok, reason = jobs.compatibility(jobs.TaskSummary.from_chain(task_payload(**mutate)))
        assert not ok and fragment in reason


def test_accept_journals_before_the_transaction(journal):
    client = FakeClient({7: task_payload(7)})
    entry = jobs.accept_task(client, journal, jobs.TaskSummary.from_chain(task_payload(7)),
                             account_scalar_hex="11" * 32, current_height=100, log=lambda _m: None)
    recorded = journal.read(7)
    assert recorded["phase"] == jobs.PHASE_ACCEPTED and recorded["txhash"] == "ACCEPTTX"
    assert [h["phase"] for h in recorded["history"]] == [jobs.PHASE_ACCEPTING, jobs.PHASE_ACCEPTED]
    assert entry["txhash"] == "ACCEPTTX"


def test_duplicate_accept_is_refused_without_a_second_transaction(journal):
    client = FakeClient({7: task_payload(7)})
    task = jobs.TaskSummary.from_chain(task_payload(7))
    jobs.accept_task(client, journal, task, account_scalar_hex="11" * 32,
                     current_height=100, log=lambda _m: None)
    jobs.accept_task(client, journal, task, account_scalar_hex="11" * 32,
                     current_height=100, log=lambda _m: None)
    assert client.accept_calls == [7], "a second accept must never be submitted"


def test_crash_during_accept_reconciles_against_the_chain(journal):
    journal.record(7, jobs.PHASE_ACCEPTING, model_id="qwen3-0.6b-layer0-v2")
    client = FakeClient({7: task_payload(7, worker="prsm1worker", status="accepted")})
    entry = jobs.accept_task(client, journal, jobs.TaskSummary.from_chain(task_payload(7)),
                             account_scalar_hex="11" * 32, current_height=100, log=lambda _m: None)
    assert entry["phase"] == jobs.PHASE_ACCEPTED and entry.get("reconciled") is True
    assert client.accept_calls == []


def test_crash_without_chain_effect_retries(journal):
    journal.record(7, jobs.PHASE_ACCEPTING)
    client = FakeClient({7: task_payload(7)})  # still open, no worker
    jobs.accept_task(client, journal, jobs.TaskSummary.from_chain(task_payload(7)),
                     account_scalar_hex="11" * 32, current_height=100, log=lambda _m: None)
    assert client.accept_calls == [7]


def test_deadline_margin_refuses_before_any_transaction(journal):
    client = FakeClient({7: task_payload(7, deadline=104)})
    with pytest.raises(jobs.JobError, match="deadline"):
        jobs.accept_task(client, journal, jobs.TaskSummary.from_chain(task_payload(7, deadline=104)),
                         account_scalar_hex="11" * 32, current_height=100, safety_margin=5,
                         log=lambda _m: None)
    assert client.accept_calls == []
    assert journal.read(7)["phase"] == jobs.PHASE_ACCEPT_FAILED


def test_rejected_accept_is_journaled_with_the_reason(journal):
    class FailingClient(FakeClient):
        def accept_task(self, task_id, account_scalar_hex):
            raise chain_mod.ChainError("transaction rejected: out of gas")

    client = FailingClient({7: task_payload(7)})
    with pytest.raises(chain_mod.ChainError):
        jobs.accept_task(client, journal, jobs.TaskSummary.from_chain(task_payload(7)),
                         account_scalar_hex="11" * 32, current_height=100, log=lambda _m: None)
    entry = journal.read(7)
    assert entry["phase"] == jobs.PHASE_ACCEPT_FAILED and "out of gas" in entry["reason"]


def test_run_discovery_cycle_accepts_up_to_the_limit(journal):
    client = FakeClient({1: task_payload(1), 2: task_payload(2), 3: task_payload(3)})
    accepted = jobs.run_discovery_cycle(client, journal, account_scalar_hex="11" * 32,
                                        max_scan=10, accept_limit=2, log=lambda _m: None)
    assert [e["task_id"] for e in accepted] == [1, 2]
    assert client.accept_calls == [1, 2]


def test_run_discovery_cycle_skips_incompatible_tasks(journal):
    client = FakeClient({1: task_payload(1, model_id="someone-elses-model"),
                         2: task_payload(2)})
    accepted = jobs.run_discovery_cycle(client, journal, account_scalar_hex="11" * 32,
                                        max_scan=10, accept_limit=1, log=lambda _m: None)
    assert [e["task_id"] for e in accepted] == [2]
    assert journal.read(1) is None  # incompatible tasks are never journaled as accepted


def test_journal_is_atomic_and_readable(tmp_path):
    journal = jobs.Journal(tmp_path / "j")
    journal.record(1, jobs.PHASE_DISCOVERED)
    journal.record(1, jobs.PHASE_EXECUTING, note="later phase")
    entry = journal.read(1)
    assert entry["phase"] == jobs.PHASE_EXECUTING and entry["note"] == "later phase"
    assert [e["task_id"] for e in journal.entries()] == [1]
    assert not list((tmp_path / "j").glob("*.tmp"))


def test_client_task_query_tolerates_missing_ids():
    def runner(cmd, timeout=60.0):
        raise chain_mod.ChainError("query compute task … failed: task 99 not found")

    client = chain_mod.ChainClient(chain_mod.ChainConfig(rpc_urls=["http://x"], chain_id="c"),
                                   runner=runner,
                                   http_get=lambda url: json.dumps({"result": {"node_info": {"network": "c"}}}))
    assert client.task(99) == {}


def test_client_task_query_surfaces_real_errors():
    def runner(cmd, timeout=60.0):
        raise chain_mod.ChainError("connection refused")

    client = chain_mod.ChainClient(chain_mod.ChainConfig(rpc_urls=["http://x"], chain_id="c"),
                                   runner=runner,
                                   http_get=lambda url: json.dumps({"result": {"node_info": {"network": "c"}}}))
    with pytest.raises(chain_mod.ChainError, match="connection refused"):
        client.task(1)


def test_client_task_query_decodes_the_base64_envelope():
    """QueryTaskResponse carries bytes task_json, not a flat object."""
    import base64

    document = task_payload(7)
    wanted = [r"query", "compute", "task", "--task-id", "7"]

    def runner(cmd, timeout=60.0):
        assert cmd[1:5] == ["query", "compute", "task", "--task-id"], cmd
        assert cmd[5] == "7", cmd
        return json.dumps({"task_json": base64.b64encode(json.dumps(document).encode()).decode()})

    client = chain_mod.ChainClient(chain_mod.ChainConfig(rpc_urls=["http://x"], chain_id="c"),
                                   runner=runner,
                                   http_get=lambda url: json.dumps({"result": {"node_info": {"network": "c"}}}))
    task = client.task(7)
    assert task["id"] == 7 and task["model_id"] == "qwen3-0.6b-layer0-v2"
    assert jobs.TaskSummary.from_chain(task).task_id == 7


def test_client_task_query_rejects_a_corrupt_envelope():
    def runner(cmd, timeout=60.0):
        return json.dumps({"task_json": "not-base64!!"})

    client = chain_mod.ChainClient(chain_mod.ChainConfig(rpc_urls=["http://x"], chain_id="c"),
                                   runner=runner,
                                   http_get=lambda url: json.dumps({"result": {"node_info": {"network": "c"}}}))
    with pytest.raises(chain_mod.ChainError, match="not base64 JSON"):
        client.task(1)


def test_client_model_query_decodes_its_envelope():
    import base64

    document = {"id": "m", "version": "v1", "mode": "verifiable"}

    def runner(cmd, timeout=60.0):
        return json.dumps({"model_json": base64.b64encode(json.dumps(document).encode()).decode()})

    client = chain_mod.ChainClient(chain_mod.ChainConfig(rpc_urls=["http://x"], chain_id="c"),
                                   runner=runner,
                                   http_get=lambda url: json.dumps({"result": {"node_info": {"network": "c"}}}))
    assert client.model("m", "v1")["mode"] == "verifiable"
