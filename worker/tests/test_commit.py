"""B2-07: the worker's CommitV3 must be byte-identical to the frozen
construction and must verify under the repository's independent Go verifier."""

from __future__ import annotations

import base64
import json
import pathlib
import sys

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from prisma_worker import commit as commit_mod
from prisma_worker.identity import WorkerIdentity
from prisma_worker.jobs import PHASE_ACCEPTED, PHASE_SUBMITTED, Journal

REPO = pathlib.Path(__file__).resolve().parents[2]
FROZEN_GRAPH = REPO / "testdata" / "f5c_qwen3_block_v2.json"


@pytest.fixture()
def frozen_graph() -> dict:
    return json.loads(FROZEN_GRAPH.read_text(encoding="utf-8"))


def _authoritative_commit(graph_doc: dict, *, task_id: int, assignment_hex: str,
                          worker_pub: bytes, roots: list[str], epoch: int, seed: bytes):
    """Build the same commit with the frozen tool (tools/f5c_commit_v3.py)."""
    sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))
    sys.path.insert(0, str(REPO / "tools"))
    import f5c_commit_v3  # type: ignore

    priv = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
    return f5c_commit_v3.build_signed_commit_v3(graph_doc, task_id, assignment_hex,
                                                worker_pub, roots, epoch, priv)


@pytest.fixture()
def setup(frozen_graph):
    seed = bytes(range(32))
    identity = WorkerIdentity(protocol_seed=seed, account_scalar=bytes([0x11] * 32))
    roots = [f"{i:02x}" * 32 for i in range(1, len(frozen_graph["nodes"]) + 1)]
    assignment_hex = "ab" * 32
    return {"identity": identity, "seed": seed, "roots": roots,
            "assignment_hex": assignment_hex, "task_id": 42, "epoch": 777}


def test_manifest_matches_the_frozen_tool(frozen_graph, setup):
    sys.path.insert(0, str(REPO / "tools"))
    sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))
    import f5c_commit_v3  # type: ignore

    ours = commit_mod.node_output_manifest_v2(frozen_graph, setup["roots"])
    theirs = f5c_commit_v3.node_output_manifest_v2(frozen_graph, setup["roots"])
    assert ours == theirs, "NodeOutputManifestV2 differs from the frozen construction"


def test_commit_and_signature_match_the_frozen_tool_byte_for_byte(frozen_graph, setup):
    unsigned = commit_mod.build_commit_v3(
        frozen_graph, task_id=setup["task_id"], assignment_ref_hex=setup["assignment_hex"],
        worker_public=setup["identity"].protocol_public, node_roots_hex=setup["roots"],
        completed_epoch=setup["epoch"])
    signed = commit_mod.sign_commit_v3(setup["identity"], unsigned)
    theirs, _, _, _ = _authoritative_commit(
        frozen_graph, task_id=setup["task_id"], assignment_hex=setup["assignment_hex"],
        worker_pub=setup["identity"].protocol_public, roots=setup["roots"],
        epoch=setup["epoch"], seed=setup["seed"])
    assert signed == theirs
    assert signed["signature"] == theirs["signature"]
    assert signed["policy_id"] == commit_mod.POLICY_ID_A13W10
    assert signed["arithmetic_id"] == commit_mod.ARITHMETIC_PROFILE_A13W10


def test_signature_covers_the_preimage_with_a_cleared_signature_field(frozen_graph, setup):
    signed = commit_mod.sign_commit_v3(setup["identity"], commit_mod.build_commit_v3(
        frozen_graph, task_id=setup["task_id"], assignment_ref_hex=setup["assignment_hex"],
        worker_public=setup["identity"].protocol_public, node_roots_hex=setup["roots"],
        completed_epoch=setup["epoch"]))
    preimage = commit_mod.commit_v3_preimage(signed)
    verify_key = ed25519.Ed25519PublicKey.from_public_bytes(setup["identity"].protocol_public)
    verify_key.verify(base64.b64decode(signed["signature"]), preimage)
    # A different completed_epoch changes the preimage: the construction is bound.
    other = dict(signed, completed_epoch=setup["epoch"] + 1)
    assert commit_mod.commit_v3_preimage(other) != preimage


def test_go_verifier_accepts_the_commit(frozen_graph, setup):
    signed = commit_mod.sign_commit_v3(setup["identity"], commit_mod.build_commit_v3(
        frozen_graph, task_id=setup["task_id"], assignment_ref_hex=setup["assignment_hex"],
        worker_public=setup["identity"].protocol_public, node_roots_hex=setup["roots"],
        completed_epoch=setup["epoch"]))
    result = commit_mod.go_verify_commit(signed, repo_root=REPO)
    if not result.verified:
        pytest.skip(f"independence check unavailable here: {result.detail}")
    assert "COMMIT OK" in result.detail


def test_go_verifier_rejects_a_tampered_commit(frozen_graph, setup):
    signed = commit_mod.sign_commit_v3(setup["identity"], commit_mod.build_commit_v3(
        frozen_graph, task_id=setup["task_id"], assignment_ref_hex=setup["assignment_hex"],
        worker_public=setup["identity"].protocol_public, node_roots_hex=setup["roots"],
        completed_epoch=setup["epoch"]))
    tampered = dict(signed, completed_epoch=setup["epoch"] + 1)
    try:
        commit_mod.go_verify_commit(tampered, repo_root=REPO)
    except commit_mod.CommitError as exc:
        assert "rejected" in str(exc)
    else:
        if not commit_mod.go_verify_commit(signed, repo_root=REPO).verified:
            pytest.skip("independence check unavailable here")
        pytest.fail("the Go verifier accepted a tampered commit")


def test_binding_refusal_covers_task_assignment_and_worker(frozen_graph, setup):
    signed = commit_mod.sign_commit_v3(setup["identity"], commit_mod.build_commit_v3(
        frozen_graph, task_id=setup["task_id"], assignment_ref_hex=setup["assignment_hex"],
        worker_public=setup["identity"].protocol_public, node_roots_hex=setup["roots"],
        completed_epoch=setup["epoch"]))
    commit_mod.verify_binding(signed, expected_task_id=setup["task_id"],
                              expected_assignment_ref_hex=setup["assignment_hex"],
                              expected_worker_public=setup["identity"].protocol_public)
    with pytest.raises(commit_mod.CommitBindingError, match="binds task"):
        commit_mod.verify_binding(signed, expected_task_id=43,
                                  expected_assignment_ref_hex=setup["assignment_hex"],
                                  expected_worker_public=setup["identity"].protocol_public)
    with pytest.raises(commit_mod.CommitBindingError, match="binds assignment"):
        commit_mod.verify_binding(signed, expected_task_id=setup["task_id"],
                                  expected_assignment_ref_hex="cd" * 32,
                                  expected_worker_public=setup["identity"].protocol_public)
    with pytest.raises(commit_mod.CommitBindingError, match="different worker key"):
        commit_mod.verify_binding(signed, expected_task_id=setup["task_id"],
                                  expected_assignment_ref_hex=setup["assignment_hex"],
                                  expected_worker_public=b"\x00" * 32)


def test_commit_id_is_stable_and_signature_independent(frozen_graph, setup):
    unsigned = commit_mod.build_commit_v3(
        frozen_graph, task_id=setup["task_id"], assignment_ref_hex=setup["assignment_hex"],
        worker_public=setup["identity"].protocol_public, node_roots_hex=setup["roots"],
        completed_epoch=setup["epoch"])
    signed = commit_mod.sign_commit_v3(setup["identity"], unsigned)
    assert commit_mod.commit_id(unsigned) == commit_mod.commit_id(signed)


def test_duplicate_submission_is_impossible(tmp_path, frozen_graph, setup):
    journal = Journal(tmp_path / "journal")
    journal.record(setup["task_id"], PHASE_ACCEPTED, txhash="ACCEPTTX")
    signed = commit_mod.sign_commit_v3(setup["identity"], commit_mod.build_commit_v3(
        frozen_graph, task_id=setup["task_id"], assignment_ref_hex=setup["assignment_hex"],
        worker_public=setup["identity"].protocol_public, node_roots_hex=setup["roots"],
        completed_epoch=setup["epoch"]))
    identifier = commit_mod.guard_duplicate_submit(journal, setup["task_id"], signed)
    assert journal.read(setup["task_id"])["phase"] == PHASE_SUBMITTED
    assert journal.read(setup["task_id"])["commit_id"] == identifier
    with pytest.raises(commit_mod.DuplicateSubmitError):
        commit_mod.guard_duplicate_submit(journal, setup["task_id"], signed)


def test_commit_before_accept_is_refused(tmp_path, frozen_graph, setup):
    journal = Journal(tmp_path / "journal")
    signed = commit_mod.sign_commit_v3(setup["identity"], commit_mod.build_commit_v3(
        frozen_graph, task_id=setup["task_id"], assignment_ref_hex=setup["assignment_hex"],
        worker_public=setup["identity"].protocol_public, node_roots_hex=setup["roots"],
        completed_epoch=setup["epoch"]))
    with pytest.raises(commit_mod.CommitError, match="only built after an accepted job"):
        commit_mod.guard_duplicate_submit(journal, setup["task_id"], signed)


def test_malformed_inputs_are_refused(frozen_graph, setup):
    with pytest.raises(commit_mod.CommitError, match="node roots"):
        commit_mod.build_commit_v3(frozen_graph, task_id=1, assignment_ref_hex="ab" * 32,
                                   worker_public=setup["identity"].protocol_public,
                                   node_roots_hex=setup["roots"][:3], completed_epoch=1)
    with pytest.raises(commit_mod.CommitError, match="32-byte ed25519"):
        commit_mod.build_commit_v3(frozen_graph, task_id=1, assignment_ref_hex="ab" * 32,
                                   worker_public=b"short", node_roots_hex=setup["roots"],
                                   completed_epoch=1)
    with pytest.raises(commit_mod.CommitError, match="assignment reference"):
        commit_mod.build_commit_v3(frozen_graph, task_id=1, assignment_ref_hex="",
                                   worker_public=setup["identity"].protocol_public,
                                   node_roots_hex=setup["roots"], completed_epoch=1)
