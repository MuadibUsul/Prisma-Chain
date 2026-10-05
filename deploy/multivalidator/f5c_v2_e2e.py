"""F.5C devnet E2E on the four-validator chain (prisma-mv-1), Graph V2.

Honest scenario first (session deliverable): post a small formal
CANONICAL_GRAPH_V2 task -> accept -> worker executes with the frozen
semantics -> CommitV3 -> two typed DA attestations -> finalize (quorum
gate) -> verify exactly one VWR V3 -> permissionless Watcher V2 on the
task's bundle.

    python deploy/multivalidator/f5c_v2_e2e.py [--scenario honest|gemm_fraud]

Writes docs/phase-f5c-e2e-<scenario>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy"))
sys.path.insert(0, str(REPO / "deploy" / "multivalidator"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import f_graph_test as F  # noqa: E402
import gemm_chain_smoke as G  # noqa: E402
import canonical_ref as R  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from f5c_exec_v2 import execute_v2  # noqa: E402
from f5c_bundle_v2 import build_bundle, decode_bundle  # noqa: E402
from f5c_commit_v3 import build_signed_commit_v3  # noqa: E402
from f5c_watcher_v2 import WatcherV2  # noqa: E402

CHAIN_WINDOW = F.CHALLENGE_WINDOW  # the chain's minimum challenge window
MAX_FEE = 2_000_000


def build_small_v2_graph():
    """K=8 two-node V2 block: wide GEMM + wide requant (same geometry as the
    chain unit fixtures)."""
    import base64
    import numpy as np
    a_data = [1, -2, 3, -4, 5, -6, 7, -8, 4095, -4096, 100, -100, 5, 6, -7, 8]
    w_data = [int(i % 17) - 8 for i in range(64)]
    a_desc = R.tensor_desc_v2(R.DTYPE_V2_A13, (2, 8))
    w_desc = R.tensor_desc_v2(R.DTYPE_V2_W10, (8, 8))
    a_root = R.tensor_root_v2(a_desc, a_data)
    w_root = R.tensor_root_v2(w_desc, w_data)
    graph_doc = {
        "protocol_version": R.PROTOCOL_VERSION_GRAPH_V2,
        "spec": "TEST_WIDE_DISPUTE_V1",
        "arithmetic": R.arithmetic_profile_a13w10(),
        "inputs": [
            {"name": "a", "desc": a_desc, "root": base64.b64encode(a_root).decode()},
            {"name": "w", "desc": w_desc, "root": base64.b64encode(w_root).decode()},
        ],
        "nodes": [
            {"node_id": 0, "operator_id": "GEMM_A13W10_I64_V1",
             "operator_version": "GEMM_A13W10_I64_V1/1.0.0",
             "inputs": [{"kind": 0, "index": 0}, {"kind": 0, "index": 1}],
             "output": R.tensor_desc_v2(R.DTYPE_V2_INT64_ACCUM, (2, 8)),
             "params": []},
            {"node_id": 1, "operator_id": "REQUANTIZE_WIDE_V1",
             "operator_version": "REQUANTIZE_WIDE_V1/1.0.0",
             "inputs": [{"kind": 1, "index": 0}],
             "output": R.tensor_desc_v2(R.DTYPE_V2_A13, (2, 8)),
             "params": [{"key": "clamp_hi", "value": R.A13_MAX},
                        {"key": "clamp_lo", "value": R.A13_MIN},
                        {"key": "mult", "value": 1 << 20},
                        {"key": "out_dtype", "value": R.DTYPE_V2_A13},
                        {"key": "shift", "value": 20}]},
        ],
        "outputs": [{"kind": 1, "index": 1}],
    }
    inputs = {0: np.array(a_data, dtype=np.int64).reshape(2, 8),
              1: np.array(w_data, dtype=np.int64).reshape(8, 8)}
    honest = execute_v2(graph_doc, inputs)
    return graph_doc, inputs, honest


def node_roots_hex(graph_doc, outputs):
    roots = []
    for node in graph_doc["nodes"]:
        data = outputs[int(node["node_id"])]
        roots.append(R.tensor_root_v2(node["output"], data.reshape(-1).tolist()).hex())
    return roots


def bundle_for(graph_doc, inputs, outputs) -> bytes:
    input_entries = []
    for i, entry in enumerate(graph_doc["inputs"]):
        input_entries.append({"name": entry["name"], "desc": entry["desc"], "data": inputs[i]})
    out_entries = []
    for node in graph_doc["nodes"]:
        nid = int(node["node_id"])
        out_entries.append({"node_id": nid, "desc": node["output"], "data": outputs[nid]})
    return build_bundle(graph_doc, input_entries, out_entries)


def wait_height(target: int, timeout: int = 240) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        h = F.height()
        if h >= target:
            return h
        time.sleep(2)
    raise SystemExit(f"height did not reach {target} (stuck at {F.height()})")


def run_honest(report: dict) -> None:
    F.with_service("validator-a")
    # requester + worker identities
    requester_name = f"v2req{os.urandom(3).hex()}"
    requester_seed, _ = G.new_ed25519()
    requester = F.ensure_imported(requester_name, requester_seed)
    G.fund(requester, 40_000_000)
    worker_name = f"v2wrk{os.urandom(3).hex()}"
    worker_seed, _ = G.new_ed25519()
    worker = F.ensure_imported(worker_name, worker_seed)
    G.fund(worker, 40_000_000)
    worker_priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(worker_seed))
    worker_pub = worker_priv.public_key().public_bytes_raw()

    # DA providers (two distinct accounts with bonded network keys)
    provs = []
    for tag in ("pa", "pb"):
        name = f"v2{tag}{os.urandom(2).hex()}"
        seed, _ = G.new_ed25519()
        addr = F.ensure_imported(name, seed)
        G.fund(addr, 40_000_000)
        priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed))
        pub = priv.public_key().public_bytes_raw()
        nk_proof = G.network_key_proof(G.key_address(name), seed, pub)
        F.send_tx_big("bond-worker", name, amount=5 * F.MIN_BOND,
                      network_public_key=F.hexb(pub), network_key_proof=F.hexb(nk_proof))
        F.send_tx_big("register-da-provider", name)
        provs.append({"name": name, "priv": priv, "seed": seed})
    # worker bonds with its network key
    worker_nk_proof = G.network_key_proof(worker, worker_seed, worker_pub)
    F.send_tx_big("bond-worker", worker_name, amount=5 * F.MIN_BOND,
                  network_public_key=F.hexb(worker_pub), network_key_proof=F.hexb(worker_nk_proof))

    graph_doc, inputs, honest = build_small_v2_graph()
    roots = node_roots_hex(graph_doc, honest)
    bundle = bundle_for(graph_doc, inputs, honest)
    from f5c_bundle_v2 import graph_id_v2_of
    report["graph_id_v2"] = graph_id_v2_of(graph_doc).hex()
    report["bundle_bytes"] = len(bundle)

    nonce = os.urandom(16)
    key_proof = F.requester_key_proof_graph(requester, requester_seed,
                                            F.requester_pub_bytes(requester_seed), nonce)
    report["start_height"] = F.height()
    F.send_tx_big("post-graph-task", requester_name,
                  requester_protocol_pubkey=F.hexb(F.requester_pub_bytes(requester_seed)),
                  requester_key_proof=F.hexb(key_proof), requester_nonce=F.hexb(nonce),
                  graph_json=F.hexb(json.dumps(graph_doc, separators=(",", ":")).encode()),
                  input_data_ref="dev://f5c-v2", challenge_window=CHAIN_WINDOW,
                  max_price_per_cwu=1000, max_fee=MAX_FEE)
    task_id = F.latest_task_id(report)
    report["task_id"] = task_id
    F.send_tx_big("accept-graph-task", worker_name, graph_task_id=task_id,
                  assignment_nonce=F.hexb(os.urandom(16)))
    task = F.wait_graph_task(task_id, "assigned")
    import base64 as _assignment_b64
    assignment_ref = _assignment_b64.b64decode(task["assignment_ref"])
    assignment_ref_hex = assignment_ref.hex()

    commit, graph_id, manifest, final_root = build_signed_commit_v3(
        graph_doc, task_id, assignment_ref_hex, worker_pub, roots, 42, worker_priv)
    (REPO / "testdata" / "f5c_tmp" / f"e2e_commit_{task_id}.json").write_text(
        json.dumps(commit), encoding="utf-8")
    (REPO / "testdata" / "f5c_tmp" / f"e2e_ctx_{task_id}.json").write_text(json.dumps({
        "task_id": task_id, "assignment_ref_hex": assignment_ref_hex,
        "worker_pub_hex": worker_pub.hex(), "roots": roots,
        "graph_id": graph_id.hex(), "manifest": manifest.hex(),
        "final_root": final_root.hex()}), encoding="utf-8")
    F.send_tx_big("submit-graph-result-v2", worker_name, graph_task_id=task_id,
                  node_output_manifest_root=F.hexb(manifest),
                  output_roots=F.hexb(bytes.fromhex(roots[-1])),
                  final_output_root=F.hexb(final_root),
                  completed_epoch=42,
                  worker_signature=F.hexb(bytes.fromhex(_b64hex(commit["signature"]))))
    task = F.wait_graph_task(task_id, "result_submitted")
    report["result_submitted_height"] = task.get("result_submitted_height")
    report["challenge_end"] = task.get("challenge_end")

    # DA attestations (typed, signed by each provider)
    import base64 as b64mod
    from f5c_da_provider import attest as da_attest
    tmp = REPO / "testdata" / "f5c_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    task_meta_path = tmp / f"task_{task_id}.json"
    header = decode_bundle(bundle)["header"]
    task_meta = {"graph_id_v2": header["graph_id_v2"],
                 "manifest_root_v2": header["manifest_root_v2"],
                 "final_output_root": header["final_output_root"],
                 "policy_id": header["policy_id"], "graph_task_id": task_id}
    task_meta_path.write_text(json.dumps(task_meta), encoding="utf-8")
    bundle_path = tmp / f"bundle_{task_id}.bin"
    bundle_path.write_bytes(bundle)
    att_hashes = []
    h_now = F.height()
    for prov in provs:
        att_path = tmp / f"att_{task_id}_{prov['name']}.json"
        da_attest(bundle_path, task_meta_path, prov["name"], prov["seed"], h_now + 500,
                  att_path, attested_height=h_now)
        raw = att_path.read_bytes()
        F.send_tx_big("submit-graph-da-attestation", prov["name"], graph_task_id=task_id,
                      attestation_json=F.hexb(raw))
        att_hashes.append(json.loads(raw)["signature"])
    report["da_attestations"] = att_hashes

    wait_height(int(task["challenge_end"]) + 1)
    F.send_tx_big("finalize-graph-task", requester_name, graph_task_id=task_id)
    final = F.wait_graph_task(task_id, "finalized")
    report["receipt_id"] = final.get("receipt_id")
    report["final_status"] = final.get("status")

    # permissionless watcher on the committed bundle
    watcher = WatcherV2(bundle_path, task_meta, test_seed=7)
    verdict = watcher.run()
    report["watcher_verdict"] = verdict["verdict"]
    report["watcher_instrumentation"] = verdict["instrumentation"]
    report["four_validators"] = {}
    for v in F.RPCS:
        st = F.rpc(F.RPCS[v], "/status")["sync_info"]
        report["four_validators"][v] = {
            "height": int(st["latest_block_height"]),
            "app_hash": st["latest_app_hash"]}
    hashes = {d["app_hash"] for d in report["four_validators"].values()}
    report["four_validators_converged"] = len(hashes) == 1
    report["status"] = "PASS"


def _b64hex(s: str) -> str:
    import base64
    return base64.b64decode(s).hex()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="honest")
    args = ap.parse_args()
    report = {"scenario": args.scenario, "chain_id": "prisma-mv-1"}
    if args.scenario == "honest":
        run_honest(report)
    else:
        raise SystemExit(f"scenario {args.scenario} not implemented yet")
    out = REPO / "docs" / f"phase-f5c-e2e-{args.scenario}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k in
                      ("status", "task_id", "watcher_verdict", "receipt_id")}, indent=1))
    print("written:", out)


if __name__ == "__main__":
    main()
