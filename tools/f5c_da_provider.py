"""F.5C DA provider simulator for GRAPH_VERIFICATION_BUNDLE_V2 (A4-02/A4-04).

A provider only ever proves STORAGE: it verifies the bundle's internal
commitments against the chain-locked task metadata before attesting, and
answers typed chunk challenges from the stored bytes.  It never executes
the Transformer.

    python tools/f5c_da_provider.py attest --bundle B --task T \
        --provider ACCT --key-hex PRIV --until HEIGHT --out ATTESTATION.json
    python tools/f5c_da_provider.py respond --bundle B --task T \
        --node N --chunk C --out RESPONSE.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import canonical_ref as R  # noqa: E402
from f5c_bundle_v2 import decode_bundle, tensor_from_bundle  # noqa: E402
from f5c_watcher_v2 import WatcherV2  # noqa: E402

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except Exception:  # pragma: no cover
    Ed25519PrivateKey = None


def _canonical_attestation(att_wire: dict) -> dict:
    """The canonical-encoding view: []byte fields as raw bytes, exactly the
    Go struct's CBOR form (the signing preimage must match the chain)."""
    return {
        "protocol_version": att_wire["protocol_version"],
        "graph_task_id": int(att_wire["graph_task_id"]),
        "graph_id": base64.b64decode(att_wire["graph_id"]),
        "manifest_root_v2": base64.b64decode(att_wire["manifest_root_v2"]),
        "final_output_root": base64.b64decode(att_wire["final_output_root"]),
        "provider_account": base64.b64decode(att_wire["provider_account"]),
        "provider_pub_key": base64.b64decode(att_wire["provider_pub_key"]),
        "bundle_bytes": int(att_wire["bundle_bytes"]),
        "available_until_height": int(att_wire["available_until_height"]),
        "attested_height": int(att_wire["attested_height"]),
        # the Go struct encodes the (cleared) signature field as an empty
        # byte string; the signing preimage must contain it
        "signature": b"",
    }


def sign_attestation(att: dict, priv_hex: str) -> dict:
    if Ed25519PrivateKey is None:
        raise SystemExit("cryptography package required for provider signing")
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex))
    preimage = b"PRISMA_GRAPH_DA_ATTESTATION_V2\x00" + \
        R.encode_canonical(_canonical_attestation(att))
    att["signature"] = base64.b64encode(private.sign(preimage)).decode()
    return att


def attest(bundle_path: Path, task_path: Path, provider: str, priv_hex: str,
           until: int, out_path: Path) -> dict:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    raw = bundle_path.read_bytes()
    # pre-attestation verification (A4-02): the provider will not attest to
    # bytes that do not match the worker's committed result
    report = WatcherV2(bundle_path, task, test_seed=1).root_phase(raw)
    if not report["pass"]:
        raise SystemExit("refusing to attest: bundle does not match the committed task: "
                         + json.dumps([c for c in report["checks"] if not c["ok"]]))
    header = decode_bundle(raw)["header"]
    pub = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex)).public_key().public_bytes_raw()
    att = {
        "protocol_version": "GRAPH_DA_ATTESTATION_V2/1.0.0",
        "graph_task_id": int(task["graph_task_id"]),
        "graph_id": base64.b64encode(bytes.fromhex(header["graph_id_v2"])).decode(),
        "manifest_root_v2": base64.b64encode(bytes.fromhex(header["manifest_root_v2"])).decode(),
        "final_output_root": base64.b64encode(bytes.fromhex(header["final_output_root"])).decode(),
        "provider_account": base64.b64encode(provider.encode()).decode(),
        "provider_pub_key": base64.b64encode(pub).decode(),
        "bundle_bytes": len(raw),
        "available_until_height": int(until),
        "attested_height": int(task.get("attested_height", 1)),
    }
    att = sign_attestation(att, priv_hex)
    out_path.write_text(json.dumps(att, indent=1) + "\n", encoding="utf-8")
    return {"attested": True, "bytes": len(raw), "out": str(out_path)}


def respond(bundle_path: Path, task_path: Path, node_id: int, chunk_index: int,
            out_path: Path) -> dict:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    raw = bundle_path.read_bytes()
    bundle = decode_bundle(raw)
    header = bundle["header"]
    graph_doc = header["graph_json"]
    node = graph_doc["nodes"][node_id]
    entry = header["node_outputs"][node_id]
    data = tensor_from_bundle(bundle, entry)
    desc = entry["desc"]
    width = {1: 4, 2: 2, 129: 2, 130: 2, 131: 8}[int(desc["dtype"])]
    be = {4: ">i4", 2: ">i2", 8: ">i8"}[width]
    import numpy as np
    flat = np.asarray(data, dtype=np.int64)
    elems = 1
    for d in desc["shape"]:
        elems *= d
    count = max(1, (elems + 63) // 64)
    # canonical chunk bytes (zero padded to 64 elements)
    padded = np.zeros(count * 64, dtype=np.int64)
    padded[:elems] = flat
    blob = padded.astype(be).tobytes()
    chunk = blob[chunk_index * 64 * width:(chunk_index + 1) * 64 * width]
    # chunk inclusion proof against the node's TensorRootV2
    root_hex = entry["root"]
    desc_bytes = R.encode_canonical(desc)
    level = []
    for i in range(count):
        c = blob[i * 64 * width:(i + 1) * 64 * width]
        level.append(R.tensor_v2_leaf(desc_bytes, i, c))
    levels = [level]
    while len(level) > 1:
        level = [R.hash_bytes(level[i] + (level[i + 1] if i + 1 < len(level) else level[i]))
                 for i in range(0, len(level), 2)]
        levels.append(level)
    siblings = []
    idx = chunk_index
    for lvl in levels[:-1]:
        sib = idx ^ 1
        siblings.append((lvl[sib] if sib < len(lvl) else lvl[idx]).hex())
        idx //= 2
    # manifest inclusion proof for the node root
    graph_id = bytes.fromhex(header["graph_id_v2"])
    leaves = []
    for n, e in zip(graph_doc["nodes"], header["node_outputs"]):
        leaf = R.hash_bytes(R.DOMAIN_MANIFEST_V2, graph_id, R._u32be(int(n["node_id"])),
                            n["operator_id"].encode(), n["operator_version"].encode(),
                            R.encode_canonical(e["desc"]))
        leaves.append(R.hash_bytes(leaf, bytes.fromhex(e["root"])))
    levels = [leaves]
    while len(levels[-1]) > 1:
        lvl = levels[-1]
        levels.append([R.hash_bytes(lvl[i] + (lvl[i + 1] if i + 1 < len(lvl) else lvl[i]))
                       for i in range(0, len(lvl), 2)])
    manifest_proof = []
    idx = node_id
    for lvl in levels[:-1]:
        sib = idx ^ 1
        manifest_proof.append((lvl[sib] if sib < len(lvl) else lvl[idx]).hex())
        idx //= 2
    resp = {
        "node_id": node_id,
        "node_root": root_hex,
        "node_desc_json": json.dumps(desc, separators=(",", ":")),
        "manifest_proof": manifest_proof,
        "chunk": base64.b64encode(chunk).decode(),
        "chunk_index": chunk_index,
        "chunk_count": count,
        "chunk_proof": siblings,
        "_task": task.get("case"),
    }
    out_path.write_text(json.dumps(resp, indent=1) + "\n", encoding="utf-8")
    return {"responded": True, "node": node_id, "chunk": chunk_index, "out": str(out_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("attest")
    a.add_argument("--bundle", required=True)
    a.add_argument("--task", required=True)
    a.add_argument("--provider", required=True)
    a.add_argument("--key-hex", required=True)
    a.add_argument("--until", type=int, required=True)
    a.add_argument("--out", required=True)
    r = sub.add_parser("respond")
    r.add_argument("--bundle", required=True)
    r.add_argument("--task", required=True)
    r.add_argument("--node", type=int, required=True)
    r.add_argument("--chunk", type=int, default=0)
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "attest":
        print(json.dumps(attest(Path(args.bundle), Path(args.task), args.provider,
                                args.key_hex, args.until, Path(args.out))))
    else:
        print(json.dumps(respond(Path(args.bundle), Path(args.task), args.node,
                                 args.chunk, Path(args.out))))


if __name__ == "__main__":
    main()
