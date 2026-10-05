"""F.5C helper: Python construction of GraphResultCommitV3 + ManifestV2
(must match the Go canonical encoding byte-for-byte; verified against the
chain's ValidateGraphResultCommitV3 in the devnet E2E).
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import canonical_ref as R  # noqa: E402

DOMAIN_SIG_V3 = b"PRISMA_CANONICAL_SIG_V3\x00"


def node_output_manifest_v2(graph_doc: dict, node_roots_hex: list) -> bytes:
    from f5c_bundle_v2 import graph_id_v2_of
    graph_id = graph_id_v2_of(graph_doc)
    leaves = []
    for node, root_hex in zip(graph_doc["nodes"], node_roots_hex):
        leaf = R.hash_bytes(R.DOMAIN_MANIFEST_V2, graph_id,
                            R._u32be(int(node["node_id"])),
                            node["operator_id"].encode(), node["operator_version"].encode(),
                            R.encode_canonical(node["output"]))
        leaves.append(R.hash_bytes(leaf, bytes.fromhex(root_hex)))
    level = leaves
    while len(level) > 1:
        level = [R.hash_bytes(level[i] + (level[i + 1] if i + 1 < len(level) else level[i]))
                 for i in range(0, len(level), 2)]
    return level[0]


def _wire_view(commit: dict) -> dict:
    """The canonical-encoding view: []byte fields as raw bytes, signature
    cleared to b"" exactly like the Go struct encoder."""
    out = {}
    for k, v in commit.items():
        if k in ("graph_id", "task_ref", "assignment_ref", "worker_pubkey",
                 "manifest_root_v2", "node_output_manifest_root_v2",
                 "final_output_root"):
            out[k] = base64.b64decode(v)
        elif k == "output_roots":
            out[k] = [base64.b64decode(x) for x in v]
        elif k == "signature":
            out[k] = b""
        else:
            out[k] = v
    # the Go struct always encodes the (cleared) signature field
    out["signature"] = b""
    return out


def commit_v3_preimage(commit: dict) -> bytes:
    return DOMAIN_SIG_V3 + R.encode_canonical(_wire_view(commit))


def build_signed_commit_v3(graph_doc: dict, task_id: int, assignment_ref_hex: str,
                           worker_pub: bytes, node_roots_hex: list, completed_epoch: int,
                           priv) -> tuple:
    import struct
    from f5c_bundle_v2 import graph_id_v2_of
    graph_id = graph_id_v2_of(graph_doc)
    manifest = node_output_manifest_v2(graph_doc, node_roots_hex)
    final_root = R.final_output_root([bytes.fromhex(node_roots_hex[-1])])
    commit = {
        "protocol_version": "3.0.0",
        "graph_id": base64.b64encode(graph_id).decode(),
        "arithmetic_id": R.ARITHMETIC_PROFILE_A13W10,
        "policy_id": R.POLICY_ID_A13W10,
        "task_ref": base64.b64encode(struct.pack(">Q", task_id)).decode(),
        "assignment_ref": base64.b64encode(bytes.fromhex(assignment_ref_hex)).decode(),
        "worker_pubkey": base64.b64encode(worker_pub).decode(),
        "node_output_manifest_root_v2": base64.b64encode(manifest).decode(),
        "final_output_root": base64.b64encode(final_root).decode(),
        "output_roots": [base64.b64encode(bytes.fromhex(node_roots_hex[-1])).decode()],
        "completed_epoch": int(completed_epoch),
    }
    sig = priv.sign(commit_v3_preimage(commit))
    commit["signature"] = base64.b64encode(sig).decode()
    return commit, graph_id, manifest, final_root
