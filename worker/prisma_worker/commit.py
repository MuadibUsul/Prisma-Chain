"""GraphResultCommitV3 construction for the worker (B2-07).

The commit is a frozen protocol object: any preimage or serialization
deviation is a protocol violation, so this module is a **line-by-line
mirror** of the frozen construction in ``tools/f5c_commit_v3.py`` +
``compute/canonical/python/canonical_ref.py``:

    TensorRootV2 / node leaves : hash(DOMAIN_MANIFEST_V2, graph_id,
                                     u32be(node_id), operator_id,
                                     operator_version, encode_canonical(output))
                                 then hash(leaf, root_bytes); pairwise Merkle,
                                 the last node duplicated on odd levels
    GraphResultCommitV3        protocol_version "3.0.0", arithmetic A13W10,
                               frozen PolicyID, task/assignment refs, worker
                               pubkey, manifest root, final output root
    signature preimage         DOMAIN_SIG_V3 || encode_canonical(wire view)
                               with the signature field cleared to b""
    final_output_root          hash(DOMAIN_VWR, last_root)

The tests cross-check both the unsigned object and the signature against the
authoritative tool byte for byte, and hand the produced commit to the
repository's independent **Go** verifier (``tools/f5c_verify_commit``), which
uses the chain's own encoder — a Python/Go cross-language gate.

Duplicate protection lives here too: a commit is bound to one assignment and
the journal refuses a second submission for the same task.
"""

from __future__ import annotations

import base64
import pathlib
import struct
from dataclasses import dataclass
from typing import Optional

from .graphid import DOMAIN_GRAPH_V2, encode_canonical, graph_id_v2_of, hash_bytes

# Frozen domains and identities (must equal canonical_ref.py / the Go encoder).
DOMAIN_MANIFEST_V2 = b"PRISMA_GRAPH_NODE_MANIFEST_V2\x00"
DOMAIN_VWR = b"PRISMA_GRAPH_VWR_V1\x00"
DOMAIN_SIG_V3 = b"PRISMA_CANONICAL_SIG_V3\x00"
ARITHMETIC_PROFILE_A13W10 = "A13W10_I64_PROFILE_V1"
POLICY_ID_A13W10 = "eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0"
PROTOCOL_VERSION_V3 = "3.0.0"

_BYTES_FIELDS = ("graph_id", "task_ref", "assignment_ref", "worker_pubkey",
                 "node_output_manifest_root_v2", "final_output_root")
_SIGNATURE_FIELD = "signature"


class CommitError(Exception):
    """Base class for commit construction/verification failures."""


class CommitBindingError(CommitError):
    """The commit does not bind to this task/assignment/worker."""


class DuplicateSubmitError(CommitError):
    """A commit for this task has already been submitted."""


def _u32be(value: int) -> bytes:
    return struct.pack(">I", value)


def final_output_root(outputs: list[bytes]) -> bytes:
    return hash_bytes(DOMAIN_VWR, *outputs)


def node_output_manifest_v2(graph_document: dict, node_roots_hex: list[str]) -> bytes:
    """NodeOutputManifestV2 over all delivered outputs (frozen construction)."""
    graph_id = graph_id_v2_of(graph_document)
    nodes = graph_document["nodes"]
    if len(node_roots_hex) != len(nodes):
        raise CommitError(f"expected {len(nodes)} node roots, got {len(node_roots_hex)}")
    leaves = []
    for node, root_hex in zip(nodes, node_roots_hex):
        leaf = hash_bytes(DOMAIN_MANIFEST_V2, graph_id, _u32be(int(node["node_id"])),
                          str(node["operator_id"]).encode(), str(node["operator_version"]).encode(),
                          encode_canonical(node["output"]))
        leaves.append(hash_bytes(leaf, bytes.fromhex(root_hex)))
    level = leaves
    while len(level) > 1:
        level = [hash_bytes(level[i] + (level[i + 1] if i + 1 < len(level) else level[i]))
                 for i in range(0, len(level), 2)]
    if not level:
        raise CommitError("the graph has no nodes")
    return level[0]


def wire_view(commit: dict) -> dict:
    """Canonical-encoding view: []byte fields raw, signature cleared to b\"\"."""
    out = {}
    for key, value in commit.items():
        if key in _BYTES_FIELDS:
            out[key] = base64.b64decode(value)
        elif key == "output_roots":
            out[key] = [base64.b64decode(x) for x in value]
        elif key == _SIGNATURE_FIELD:
            out[key] = b""
        else:
            out[key] = value
    out[_SIGNATURE_FIELD] = b""  # the struct always encodes the cleared field
    return out


def commit_v3_preimage(commit: dict) -> bytes:
    return DOMAIN_SIG_V3 + encode_canonical(wire_view(commit))


def build_commit_v3(graph_document: dict, *, task_id: int, assignment_ref_hex: str,
                    worker_public: bytes, node_roots_hex: list[str],
                    completed_epoch: int) -> dict:
    """The unsigned GraphResultCommitV3 object (signature field absent)."""
    if not worker_public or len(worker_public) != 32:
        raise CommitError("worker public key must be a 32-byte ed25519 key")
    assignment = bytes.fromhex(assignment_ref_hex)
    if not assignment:
        raise CommitError("assignment reference must not be empty")
    graph_id = graph_id_v2_of(graph_document)
    manifest = node_output_manifest_v2(graph_document, node_roots_hex)
    last_root = bytes.fromhex(node_roots_hex[-1])
    return {
        "protocol_version": PROTOCOL_VERSION_V3,
        "graph_id": base64.b64encode(graph_id).decode(),
        "arithmetic_id": ARITHMETIC_PROFILE_A13W10,
        "policy_id": POLICY_ID_A13W10,
        "task_ref": base64.b64encode(struct.pack(">Q", task_id)).decode(),
        "assignment_ref": base64.b64encode(assignment).decode(),
        "worker_pubkey": base64.b64encode(worker_public).decode(),
        "node_output_manifest_root_v2": base64.b64encode(manifest).decode(),
        "final_output_root": base64.b64encode(final_output_root([last_root])).decode(),
        "output_roots": [base64.b64encode(last_root).decode()],
        "completed_epoch": int(completed_epoch),
    }


def sign_commit_v3(identity, commit: dict) -> dict:
    """Sign the frozen preimage with the worker's ed25519 protocol key."""
    if not hasattr(identity, "sign_raw"):
        raise CommitError("identity must provide an ed25519 raw signer (sign_raw)")
    signed = dict(commit)
    signed[_SIGNATURE_FIELD] = ""
    signed[_SIGNATURE_FIELD] = base64.b64encode(
        identity.sign_raw(commit_v3_preimage(signed))).decode()
    return signed


def verify_binding(commit: dict, *, expected_task_id: int, expected_assignment_ref_hex: str,
                   expected_worker_public: bytes) -> None:
    """Refuse a commit that does not bind to this task/assignment/worker."""
    try:
        task_ref = struct.unpack(">Q", base64.b64decode(commit["task_ref"]))[0]
    except (KeyError, ValueError, struct.error) as exc:
        raise CommitBindingError(f"commit task_ref is not a u64: {exc}") from exc
    if task_ref != expected_task_id:
        raise CommitBindingError(f"commit binds task {task_ref}, expected {expected_task_id}")
    assignment = base64.b64decode(commit.get("assignment_ref", ""))
    if assignment.hex() != expected_assignment_ref_hex:
        raise CommitBindingError(
            f"commit binds assignment {assignment.hex()[:16]}…, expected {expected_assignment_ref_hex[:16]}…")
    worker = base64.b64decode(commit.get("worker_pubkey", ""))
    if worker != expected_worker_public:
        raise CommitBindingError("commit is signed for a different worker key")
    if commit.get("protocol_version") != PROTOCOL_VERSION_V3:
        raise CommitBindingError(f"commit protocol_version {commit.get('protocol_version')!r}")
    if commit.get("policy_id") != POLICY_ID_A13W10:
        raise CommitBindingError("commit does not carry the frozen PolicyID")
    if commit.get("arithmetic_id") != ARITHMETIC_PROFILE_A13W10:
        raise CommitBindingError("commit does not carry the frozen arithmetic profile")


def commit_id(commit: dict) -> str:
    """Deterministic identity of a commit's preimage (duplicate protection)."""
    unsigned = {k: v for k, v in commit.items() if k != _SIGNATURE_FIELD}
    return hash_bytes(DOMAIN_SIG_V3, encode_canonical(wire_view(unsigned))).hex()


def guard_duplicate_submit(journal, task_id: int, commit: dict) -> str:
    """Refuse a second submission for one task; record the submitted commit id.

    The journal is the worker's write-ahead state (B2-04): reaching
    ``submitted`` for a task means a commit left this worker, and the frozen
    chain accepts exactly one commit per assignment.
    """
    from .jobs import PHASE_ACCEPTED, PHASE_EXECUTING, PHASE_SUBMITTED

    entry = journal.read(task_id)
    if not entry:
        raise CommitError(
            f"task {task_id} has no journal entry; a commit is only built after an accepted job")
    if entry.get("phase") == PHASE_SUBMITTED:
        raise DuplicateSubmitError(
            f"task {task_id} already submitted commit {entry.get('commit_id', '?')[:16]}…; refusing")
    if entry.get("phase") not in (PHASE_ACCEPTED, PHASE_EXECUTING):
        raise CommitError(
            f"task {task_id} is {entry.get('phase')!r}; a commit is only built after an accepted job")
    identifier = commit_id(commit)
    journal.record(task_id, PHASE_SUBMITTED, commit_id=identifier,
                   manifest_root=commit.get("node_output_manifest_root_v2", ""),
                   final_output_root=commit.get("final_output_root", ""))
    return identifier


@dataclass
class CrossCheck:
    """Result of handing a commit to the repository's Go verifier."""

    verified: bool
    detail: str


def go_verify_commit(commit: dict, *, repo_root: Optional[pathlib.Path] = None,
                     timeout: float = 300.0) -> CrossCheck:
    """Verify a commit with ``tools/f5c_verify_commit`` (the chain's own encoder).

    Returns ``CrossCheck(False, …)`` when Go is unavailable — the caller
    decides whether to skip; a verification failure raises.
    """
    import json
    import shutil
    import subprocess
    import tempfile

    root = pathlib.Path(repo_root or pathlib.Path(__file__).resolve().parents[2])
    if shutil.which("go") is None:
        return CrossCheck(False, "go toolchain is not available")
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "commit.json"
        path.write_text(json.dumps(commit, indent=1) + "\n", encoding="utf-8")
        (root / "testdata" / "f5c_tmp").mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(["go", "run", "./tools/f5c_verify_commit", str(path)],
                              cwd=root, capture_output=True, text=True, timeout=timeout)
    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0 or "COMMIT OK" not in proc.stdout:
        raise CommitError(f"the independent Go verifier rejected the commit: {output}")
    return CrossCheck(True, output)
