"""GraphVerificationBundleV2 (roadmap A3-01/A3-02): schema, codec,
artifact hash and builder.

File format (single blob):

    magic  "F5CBNDL2" (8 bytes)
    u32    header length (little-endian)
    header canonical-CBOR of:
        {
          "version": "GRAPH_VERIFICATION_BUNDLE_V2/1.0.0",
          "graph_id_v2": hex, "policy_id": hex,
          "manifest_root_v2": hex, "final_output_root": hex,
          "inputs":       [{name, desc, root, offset, length}],
          "node_outputs": [{node_id, desc, root, offset, length}],
        }
    blobs  all tensor payloads concatenated in offset order, each tensor
           encoded element-wise at its dtype width (big-endian two's
           complement) — the same bytes the TensorRootV2 chunk rule splits
           into 64-element chunks.

artifact_hash = SHA256("PRISMA_GRAPH_BUNDLE_V2\\0"
                       || canonical-CBOR(header) || SHA256(blobs))

The schema deliberately carries NO execution trace and no dispute trail:
those exist only after a challenge.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))
sys.path.insert(0, str(REPO / "tools"))

import canonical_ref as R  # noqa: E402
from f5c_build_v2_graph import fast_tensor_root_v2, fast_validate  # noqa: E402

MAGIC = b"F5CBNDL2"
BUNDLE_VERSION = "GRAPH_VERIFICATION_BUNDLE_V2/1.0.0"
ARTIFACT_DOMAIN = b"PRISMA_GRAPH_BUNDLE_V2\x00"

_WIDTH = {R.DTYPE_V2_Q12_20: 4, R.DTYPE_V2_A13: 2, R.DTYPE_V2_W10: 2,
          R.DTYPE_V2_INT64_ACCUM: 8}
_BE = {4: ">i4", 2: ">i2", 8: ">i8"}


def canon_graph_doc(doc: dict) -> dict:
    """The canonical-encoding view of a wire GraphDescriptorV2: []byte
    fields (input roots) as raw bytes.  All local GraphIDV2 / manifest
    computations MUST use this view; the wire form (base64) is only for
    transport and display."""
    out = json.loads(json.dumps(doc))  # deep copy of the pure-JSON form
    for entry in out["inputs"]:
        entry["root"] = base64.b64decode(entry["root"])
    return out


def graph_id_v2_of(doc: dict) -> bytes:
    return R.hash_bytes(R.DOMAIN_GRAPH_V2, R.encode_canonical(canon_graph_doc(doc)))


def tensor_bytes(desc: dict, data) -> bytes:
    width = _WIDTH[int(desc["dtype"])]
    flat = np.asarray(data, dtype=np.int64).reshape(-1)
    return flat.astype(_BE[width]).tobytes()


def build_bundle(graph_doc: dict, inputs: list, node_outputs: list) -> bytes:
    """inputs/node_outputs: [{name|node_id, desc, data(int64 list/array)}]."""
    header = {
        "version": BUNDLE_VERSION,
        "graph_json": graph_doc,
        "graph_id_v2": graph_id_v2_of(graph_doc).hex(),
        "policy_id": graph_doc["arithmetic"]["policy_id"],
        "manifest_root_v2": None,
        "final_output_root": None,
        "inputs": [],
        "node_outputs": [],
    }
    blobs = bytearray()
    for entry in inputs:
        payload = tensor_bytes(entry["desc"], entry["data"])
        header["inputs"].append({
            "name": entry["name"], "desc": entry["desc"],
            "root": fast_tensor_root_v2(entry["desc"], np.asarray(entry["data"])).hex(),
            "offset": len(blobs), "length": len(payload)})
        blobs += payload
    for entry in node_outputs:
        payload = tensor_bytes(entry["desc"], entry["data"])
        header["node_outputs"].append({
            "node_id": int(entry["node_id"]), "desc": entry["desc"],
            "root": fast_tensor_root_v2(entry["desc"], np.asarray(entry["data"])).hex(),
            "offset": len(blobs), "length": len(payload)})
        blobs += payload
    # manifest + final root straight from the post-order roots
    node_roots = [n["root"] for n in header["node_outputs"]]
    graph_id = bytes.fromhex(header["graph_id_v2"])
    # NodeOutputManifestV2: leaf = H(domain || graph_id || node_id || op ||
    # version || CBOR(desc)) combined with the output root.  The bundle
    # itself only needs to reproduce the ROOT the chain locked; the builder
    # computes it with the same rule as the Go implementation.
    manifest_leaves = []
    for node, out in zip(graph_doc["nodes"], header["node_outputs"]):
        desc_bytes = R.encode_canonical(out["desc"])
        leaf = R.hash_bytes(R.DOMAIN_MANIFEST_V2, graph_id,
                            R._u32be(int(node["node_id"])),
                            node["operator_id"].encode(), node["operator_version"].encode(),
                            desc_bytes)
        manifest_leaves.append(R.hash_bytes(leaf, bytes.fromhex(out["root"])))
    level = manifest_leaves
    while len(level) > 1:
        level = [R.hash_bytes(level[i] + (level[i + 1] if i + 1 < len(level) else level[i]))
                 for i in range(0, len(level), 2)]
    header["manifest_root_v2"] = level[0].hex()
    header["final_output_root"] = R.hash_bytes(R.DOMAIN_VWR,
                                               bytes.fromhex(node_roots[-1])).hex()
    header_bytes = R.encode_canonical(header)
    blob_digest = hashlib.sha256(bytes(blobs)).digest()
    artifact = hashlib.sha256(ARTIFACT_DOMAIN + header_bytes + blob_digest).hexdigest()
    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", len(header_bytes))
    out += header_bytes
    out += blobs
    out += hashlib.sha256(ARTIFACT_DOMAIN + header_bytes + blob_digest).digest()
    _ = artifact
    return bytes(out)


def decode_bundle(raw: bytes) -> dict:
    if raw[:8] != MAGIC:
        raise ValueError("bundle: bad magic")
    hlen = struct.unpack("<I", raw[8:12])[0]
    header = R.decode_canonical(raw[12:12 + hlen])
    blobs = raw[12 + hlen:-32]
    trailer = raw[-32:]
    blob_digest = hashlib.sha256(blobs).digest()
    expected = hashlib.sha256(ARTIFACT_DOMAIN + raw[12:12 + hlen] + blob_digest).digest()
    if trailer != expected:
        raise ValueError("bundle: artifact hash mismatch")
    return {"header": header, "blobs": blobs,
            "artifact_hash": expected.hex()}


def tensor_from_bundle(bundle: dict, entry: dict):
    width = _WIDTH[int(entry["desc"]["dtype"])]
    payload = bundle["blobs"][entry["offset"]:entry["offset"] + entry["length"]]
    if len(payload) != entry["length"]:
        raise ValueError("bundle: truncated payload")
    arr = np.frombuffer(payload, dtype=_BE[width]).astype(np.int64)
    elems = 1
    for d in entry["desc"]["shape"]:
        elems *= d
    if arr.size != elems:
        raise ValueError("bundle: payload/element count mismatch")
    return arr


def size_breakdown(raw: bytes) -> dict:
    b = decode_bundle(raw)
    h = b["header"]
    inputs_bytes = sum(e["length"] for e in h["inputs"])
    outputs_bytes = sum(e["length"] for e in h["node_outputs"])
    total = len(raw)
    return {"total_bytes": total,
            "header_bytes": 12 + len(R.encode_canonical(h)),
            "descriptor_bytes": len(json.dumps(h)),  # informational
            "input_payload_bytes": inputs_bytes,
            "node_output_payload_bytes": outputs_bytes,
            "trailer_bytes": 32}


# --- A3-02: builders --------------------------------------------------------


def build_small_bundle() -> bytes:
    """A tiny two-node ADD + REQUANT V2 graph (self-contained fixture)."""
    add = {"node_id": 0, "operator_id": R.OP_ADD, "operator_version": "1.0.0",
           "inputs": [{"kind": 0, "index": 0}, {"kind": 0, "index": 1}],
           "output": R.tensor_desc_v2(R.DTYPE_V2_Q12_20, (2, 3)), "params": []}
    req = {"node_id": 1, "operator_id": R.OP_REQUANT_VERSION_WIDE.split("/")[0] if False else "REQUANTIZE_WIDE_V1",
           "operator_version": "REQUANTIZE_WIDE_V1/1.0.0",
           "inputs": [{"kind": 1, "index": 0}],
           "output": R.tensor_desc_v2(R.DTYPE_V2_A13, (2, 3)),
           "params": [{"key": "clamp_hi", "value": R.A13_MAX},
                      {"key": "clamp_lo", "value": R.A13_MIN},
                      {"key": "mult", "value": 1 << 20},
                      {"key": "out_dtype", "value": R.DTYPE_V2_A13},
                      {"key": "shift", "value": 20}]}
    a_desc = R.tensor_desc_v2(R.DTYPE_V2_Q12_20, (2, 3))
    b_desc = R.tensor_desc_v2(R.DTYPE_V2_Q12_20, (2, 3))
    a_data = [10, -20, 30, -40, 50, -60]
    b_data = [1, 2, 3, 4, 5, 6]
    a_root = R.tensor_root_v2(a_desc, a_data)
    b_root = R.tensor_root_v2(b_desc, b_data)
    graph_doc = {
        "protocol_version": R.PROTOCOL_VERSION_GRAPH_V2,
        "spec": "TEST_SMALL_BUNDLE_V1",
        "arithmetic": R.arithmetic_profile_a13w10(),
        "inputs": [{"name": "a", "desc": a_desc,
                    "root": base64.b64encode(a_root).decode()},
                   {"name": "b", "desc": b_desc,
                    "root": base64.b64encode(b_root).decode()}],
        "nodes": [add, req],
        "outputs": [{"kind": 1, "index": 1}],
    }
    # execute with f1 kernels (frozen semantics)
    import f1_canonical_numpy as N
    a = np.array(a_data, dtype=np.int64).reshape(2, 3)
    b = np.array(b_data, dtype=np.int64).reshape(2, 3)
    add_out = N.add_fx(a, b).astype(np.int64)
    req_out = np.array(R.requant_wide(add_out.reshape(-1).tolist(), 1 << 20, 20,
                                      R.A13_MIN, R.A13_MAX, R.DTYPE_V2_A13),
                       dtype=np.int64).reshape(2, 3)
    return build_bundle(graph_doc,
                        [{"name": "a", "desc": a_desc, "data": a_data},
                         {"name": "b", "desc": b_desc, "data": b_data}],
                        [{"node_id": 0, "desc": add["output"], "data": add_out},
                         {"node_id": 1, "desc": req["output"], "data": req_out}])


def build_real_qwen_bundle() -> bytes:
    """The formal 310-node QWEN3_BLOCK_PROFILE_V2 bundle, assembled from
    the frozen F.5B.2 execution artifacts (testdata/f5b2_*)."""
    prog = json.loads((REPO / "testdata" / "f5b2_block_program.json").read_text(encoding="utf-8"))
    bundle = np.load(REPO / "testdata" / "f5b2_block_bundle.npz")
    manifest = np.load(REPO / "testdata" / "f5b2_cpu_manifest.npz")
    v2doc = json.loads((REPO / "testdata" / "f5c_qwen3_block_v2.json").read_text(encoding="utf-8"))
    inputs = []
    for c in v2doc["inputs"]:
        cid = int(c["name"].split("_")[1])
        inputs.append({"name": c["name"], "desc": c["desc"],
                       "data": bundle[f"const_{cid}"].reshape(c["desc"]["shape"])})
    outs = []
    for node in v2doc["nodes"]:
        outs.append({"node_id": node["node_id"], "desc": node["output"],
                     "data": manifest[f"out_{node['node_id']}"].reshape(node["output"]["shape"])})
    _ = prog
    return build_bundle(v2doc, inputs, outs)


def main() -> None:
    small = build_small_bundle()
    dest = REPO / "testdata" / "f5c_bundle_v2_small.bin"
    dest.write_bytes(small)
    print("small bundle:", dest, len(small), "bytes")
    print("small breakdown:", json.dumps(size_breakdown(small)))

    real = build_real_qwen_bundle()
    dest_real = REPO / "testdata" / "f5c_bundle_v2_real.bin"
    dest_real.write_bytes(real)
    bd = size_breakdown(real)
    print("real bundle:", len(real), "bytes")
    print("real breakdown:", json.dumps(bd))
    (REPO / "docs" / "phase-f5c-bundle-size.json").write_text(json.dumps({
        "small": size_breakdown(small), "real_qwen_310": bd,
        "note": "real bundle built from the frozen F.5B.2 execution artifacts; "
                "the whole-bundle artifact hash binds the header and the payload digest",
    }, indent=1) + "\n", encoding="utf-8")
    print("written docs/phase-f5c-bundle-size.json")


if __name__ == "__main__":
    main()
