"""F.5C A6-06: real ROPE-fraud devnet E2E (four-validator chain).

The worker commits a SELF-CONSISTENT fraudulent result of a three-node
Q12.20 chain (ROPE -> SILU -> MUL): one ROPE output element is altered in
range and every downstream node is recomputed from the corrupted tensor,
then manifest and final roots and a valid CommitV3 signature are produced.
The honest challenger drives the deterministic path on chain: challenge ->
V2 trail claims -> one bisection midpoint -> first divergent node is the
ROPE node -> bounded typed-evidence arbitration (pinned table, committed
input chunk) -> ChallengerWins -> fraud status -> zero VWR.  The
permissionless Watcher V2 must detect and localize the tamper through the
cheap-operator recomputation.

    python deploy/multivalidator/f5c_v2_rope_fraud.py

Writes docs/phase-f5c-e2e-rope-fraud.json.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy"))
sys.path.insert(0, str(REPO / "deploy" / "multivalidator"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import f_graph_test as F  # noqa: E402
import gemm_chain_smoke as G  # noqa: E402
import canonical_ref as R  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from f5c_v2_e2e import bundle_for, node_roots_hex  # noqa: E402
from f5c_bundle_v2 import graph_id_v2_of, decode_bundle  # noqa: E402
from f5c_commit_v3 import build_signed_commit_v3  # noqa: E402
from f5c_exec_v2 import execute_v2  # noqa: E402
from f5c_watcher_v2 import WatcherV2  # noqa: E402

POSITIONS = 8
PAIRS = 4            # hidden = 8 = 2 * PAIRS
# Q12.20 sin/cos table: cos near 1.0 with a small drift, sin spread across
# pairs/positions so the rotation is a real (non-identity) mapping.
TABLE_VALUES = [v for pos in range(POSITIONS) for pair in range(PAIRS)
                for v in (1048576 - 2000 * ((pos * PAIRS + pair) % 5),
                          60000 * (pair + 1) - 150000 + 3000 * pos)]


def funded(name_prefix: str, amount: int):
    name = f"{name_prefix}{os.urandom(3).hex()}"
    seed, _ = G.new_ed25519()
    addr = F.ensure_imported(name, seed)
    G.fund(addr, amount)
    priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed))
    pub = priv.public_key().public_bytes_raw()
    return name, addr, seed, priv, pub


def bonded(name: str, addr: str, seed: str, pub: bytes) -> None:
    proof = G.network_key_proof(addr, seed, pub)
    F.send_tx_big("bond-worker", name, amount=5 * F.MIN_BOND,
                  network_public_key=F.hexb(pub), network_key_proof=F.hexb(proof))


def state_root_v2(roots: dict) -> bytes:
    return R.graph_state_root_v2(roots)


def build_rope_chain_v2_graph():
    """ROPE -> SILU -> MUL chain over Q12.20 tensors (one full chunk each:
    64 elements, so no arbitration-window padding edge cases)."""
    x_data = [((i * 971) % 1201) - 600 for i in range(POSITIONS * PAIRS * 2)]
    x_desc = R.tensor_desc_v2(R.DTYPE_V2_Q12_20, (POSITIONS, PAIRS * 2))
    t_desc = R.tensor_desc_v2(R.DTYPE_V2_Q12_20, (len(TABLE_VALUES),))
    x_root = R.tensor_root_v2(x_desc, x_data)
    t_root = R.tensor_root_v2(t_desc, TABLE_VALUES)
    rope_out = R.tensor_desc_v2(R.DTYPE_V2_Q12_20, (POSITIONS, PAIRS * 2))
    graph_doc = {
        "protocol_version": R.PROTOCOL_VERSION_GRAPH_V2,
        "spec": "TEST_ROPE_FRAUD_V1",
        "arithmetic": R.arithmetic_profile_a13w10(),
        "inputs": [
            {"name": "x", "desc": x_desc, "root": base64.b64encode(x_root).decode()},
            {"name": "rope", "desc": t_desc, "root": base64.b64encode(t_root).decode()},
        ],
        "nodes": [
            {"node_id": 0, "operator_id": R.OP_ROPE,
             "operator_version": "1.0.0",
             "inputs": [{"kind": 0, "index": 0}, {"kind": 0, "index": 1}],
             "output": rope_out,
             "params": [{"key": "half_dim", "value": PAIRS},
                        {"key": "max_pos", "value": POSITIONS}]},
            {"node_id": 1, "operator_id": "SILU_FIXED_V1",
             "operator_version": "1.0.0",
             "inputs": [{"kind": 1, "index": 0}],
             "output": rope_out, "params": []},
            {"node_id": 2, "operator_id": "ADD_FIXED_V1",
             "operator_version": "1.0.0",
             "inputs": [{"kind": 1, "index": 1}, {"kind": 0, "index": 0}],
             "output": rope_out, "params": []},
        ],
        "outputs": [{"kind": 1, "index": 2}],
    }
    inputs = {0: np.array(x_data, dtype=np.int64).reshape(POSITIONS, PAIRS * 2),
              1: np.array(TABLE_VALUES, dtype=np.int64)}
    honest = execute_v2(graph_doc, inputs)
    return graph_doc, inputs, honest


def chunk_of(desc: dict, flat, index: int):
    """Canonical 64-element BE chunk window of one tensor."""
    width = {4: 4, 2: 2, 1: 4, 129: 2, 130: 2, 131: 8}[int(desc["dtype"])]
    be = {4: ">i4", 2: ">i2", 8: ">i8"}[width]
    elems = 1
    for d in desc["shape"]:
        elems *= d
    count = max(1, (elems + 63) // 64)
    padded = np.zeros(count * 64, dtype=np.int64)
    padded[:elems] = np.asarray(flat, dtype=np.int64)
    blob = padded.astype(be).tobytes()
    chunk = blob[index * 64 * width:(index + 1) * 64 * width]
    desc_bytes = R.encode_canonical(desc)
    level = [R.tensor_v2_leaf(desc_bytes, i, blob[i * 64 * width:(i + 1) * 64 * width])
             for i in range(count)]
    levels = [level]
    while len(levels[-1]) > 1:
        lvl = levels[-1]
        levels.append([R.hash_bytes(lvl[i] + (lvl[i + 1] if i + 1 < len(lvl) else lvl[i]))
                       for i in range(0, len(lvl), 2)])
    siblings = []
    idx = index
    for lvl in levels[:-1]:
        sib = idx ^ 1
        siblings.append((lvl[sib] if sib < len(lvl) else lvl[idx]).hex())
        idx //= 2
    return chunk, siblings, count


def main() -> None:
    report = {"scenario": "rope_fraud", "chain_id": "prisma-mv-1"}
    F.with_service("validator-a")

    req_name, req_addr, req_seed, _, _ = funded("rreq", 40_000_000)
    wk_name, wk_addr, wk_seed, wk_priv, wk_pub = funded("rwrk", 40_000_000)
    ch_name, ch_addr, ch_seed, _, ch_pub = funded("rchal", 40_000_000)
    bonded(wk_name, wk_addr, wk_seed, wk_pub)
    bonded(ch_name, ch_addr, ch_seed, ch_pub)
    provs = []
    for tag in ("ra", "rb"):
        p_name, p_addr, p_seed, p_priv, p_pub = funded(f"r{tag}", 40_000_000)
        bonded(p_name, p_addr, p_seed, p_pub)
        F.send_tx_big("register-da-provider", p_name)
        provs.append({"name": p_name, "seed": p_seed})

    graph_doc, inputs, honest = build_rope_chain_v2_graph()
    gid = graph_id_v2_of(graph_doc)
    report["graph_id_v2"] = gid.hex()

    # --- self-consistent fraudulent result: ROPE element (0,0) altered --------
    fraud = execute_v2(graph_doc, inputs, tamper=(0, 0, 1))
    honest_roots = node_roots_hex(graph_doc, honest)
    fraud_roots = node_roots_hex(graph_doc, fraud)
    report["divergent_nodes"] = [i for i, (h, f) in enumerate(zip(honest_roots, fraud_roots))
                                 if h != f]

    nonce = os.urandom(16)
    key_proof = F.requester_key_proof_graph(req_addr, req_seed,
                                            F.requester_pub_bytes(req_seed), nonce)
    F.send_tx_big("post-graph-task", req_name,
                  requester_protocol_pubkey=F.hexb(F.requester_pub_bytes(req_seed)),
                  requester_key_proof=F.hexb(key_proof), requester_nonce=F.hexb(nonce),
                  graph_json=F.hexb(json.dumps(graph_doc, separators=(",", ":")).encode()),
                  input_data_ref="dev://f5c-rope-fraud", challenge_window=F.CHALLENGE_WINDOW,
                  max_price_per_cwu=1000, max_fee=2_000_000)
    task_id = F.latest_task_id(report)
    report["task_id"] = task_id
    F.send_tx_big("accept-graph-task", wk_name, graph_task_id=task_id,
                  assignment_nonce=F.hexb(os.urandom(16)))
    task = F.wait_graph_task(task_id, "assigned")
    assignment_ref_hex = base64.b64decode(task["assignment_ref"]).hex()

    commit, _, manifest, final_root = build_signed_commit_v3(
        graph_doc, task_id, assignment_ref_hex, wk_pub, fraud_roots, 42, wk_priv)
    F.send_tx_big("submit-graph-result-v2", wk_name, graph_task_id=task_id,
                  node_output_manifest_root=F.hexb(manifest),
                  output_roots=F.hexb(bytes.fromhex(fraud_roots[-1])),
                  final_output_root=F.hexb(final_root),
                  completed_epoch=42,
                  worker_signature=F.hexb(base64.b64decode(commit["signature"])))
    task = F.wait_graph_task(task_id, "result_submitted")
    _ = manifest

    # --- DA stores the fraudulent bundle -------------------------------------
    fraud_bundle = bundle_for(graph_doc, inputs, fraud)
    bundle_path = REPO / "testdata" / "f5c_tmp" / f"rope_fraud_bundle_{task_id}.bin"
    bundle_path.write_bytes(fraud_bundle)
    hdr = decode_bundle(fraud_bundle)["header"]
    task_meta = {"graph_id_v2": hdr["graph_id_v2"], "manifest_root_v2": hdr["manifest_root_v2"],
                 "final_output_root": hdr["final_output_root"], "policy_id": hdr["policy_id"],
                 "graph_task_id": task_id}
    meta_path = REPO / "testdata" / "f5c_tmp" / f"rope_fraud_task_{task_id}.json"
    meta_path.write_text(json.dumps(task_meta), encoding="utf-8")
    from f5c_da_provider import attest as da_attest
    h_now = F.height()
    for prov in provs:
        att_path = REPO / "testdata" / "f5c_tmp" / f"rope_fraud_att_{task_id}_{prov['name']}.json"
        da_attest(bundle_path, meta_path, prov["name"], prov["seed"], h_now + 500,
                  att_path, attested_height=h_now)
        F.send_tx_big("submit-graph-da-attestation", prov["name"], graph_task_id=task_id,
                      attestation_json=F.hexb(att_path.read_bytes()))

    # --- challenge ------------------------------------------------------------
    F.send_tx_big("open-graph-challenge", ch_name, graph_task_id=task_id,
                  challenger_output_roots=F.hexb(bytes.fromhex(honest_roots[-1])),
                  challenge_bond=F.MIN_BOND)

    # --- V2 trails ------------------------------------------------------------
    x_root = base64.b64decode(graph_doc["inputs"][0]["root"])
    t_root = R.tensor_root_v2(graph_doc["inputs"][1]["desc"], TABLE_VALUES)
    s0 = state_root_v2({(0, 0): x_root, (0, 1): t_root})

    def trail(node_roots):
        live = {(0, 0): x_root, (0, 1): t_root}
        states = [s0]
        for i, r in enumerate(node_roots):
            live[(1, i)] = bytes.fromhex(r)
            states.append(state_root_v2(dict(live)))
        return states

    honest_trail = trail(honest_roots)
    fraud_trail = trail(fraud_roots)

    def claim(party: str, t: list) -> None:
        p0 = R.trail_proof_v2(gid, t, 0)
        p1 = R.trail_proof_v2(gid, t, len(t) - 1)
        F.send_tx_big("graph-trail-claim", party, graph_task_id=task_id,
                      trail_root=F.hexb(R.trail_root_v2(gid, t)),
                      initial_root=F.hexb(t[0]),
                      initial_proof=[F.hexb(s) for s in p0],
                      final_root=F.hexb(t[-1]),
                      final_proof=[F.hexb(s) for s in p1])

    claim(wk_name, fraud_trail)
    claim(ch_name, honest_trail)

    # --- one bisection midpoint round ----------------------------------------
    rec = F.graph_dispute(task_id)
    snap = json.loads(base64.b64decode(rec["snapshot"]).decode())
    low, high = int(snap["low"]), int(snap["high"])
    report["bisection_interval"] = [low, high]
    mid = low + (high - low) // 2
    for party, t in ((wk_name, fraud_trail), (ch_name, honest_trail)):
        proof = R.trail_proof_v2(gid, t, mid)
        time.sleep(2)
        F.send_tx_big("graph-mid-point", party, graph_task_id=task_id,
                      state_root=F.hexb(t[mid]),
                      proof_siblings=[F.hexb(s) for s in proof],
                      epoch=F.height())
    rec = F.graph_dispute(task_id)
    report["dispute_status_after_midpoint"] = rec.get("status")
    report["dispute_high_after_midpoint"] = json.loads(base64.b64decode(rec["snapshot"]).decode())["high"]

    # --- bounded ROPE arbitration on the first divergent node (node 0) --------
    x_chunk, x_chunk_proof, x_count = chunk_of(graph_doc["inputs"][0]["desc"],
                                               inputs[0].reshape(-1), 0)
    w_fraud_chunk, _, _ = chunk_of(graph_doc["nodes"][0]["output"],
                                   fraud[0].reshape(-1), 0)
    w_honest_chunk, _, _ = chunk_of(graph_doc["nodes"][0]["output"],
                                    honest[0].reshape(-1), 0)
    leaf_t = R.hash_bytes(R.DOMAIN_GRAPH_STATE_V2, R._u32be(0), R._u32be(1), t_root)
    evidence = [{
        "desc_json": F.b64(json.dumps(graph_doc["inputs"][0]["desc"],
                                      separators=(",", ":")).encode()),
        "ref_kind": 0, "ref_index": 0, "root": F.b64(x_root),
        "chunk_index": 0, "count": x_count, "chunk": F.b64(x_chunk),
        "proof": x_chunk_proof, "state_proof": [F.b64(leaf_t)],
    }]
    F.send_tx_big("arbitrate-graph-node", ch_name, graph_task_id=task_id,
                  worker_out_root=fraud_roots[0],
                  worker_chunk_index=0,
                  worker_chunk=F.hexb(w_fraud_chunk),
                  worker_chunk_proof=[],
                  challenger_out_root=honest_roots[0],
                  challenger_chunk_index=0,
                  challenger_chunk=F.hexb(w_honest_chunk),
                  challenger_chunk_proof=[],
                  evidence=[json.dumps(ev, separators=(",", ":")) for ev in evidence],
                  rope_table=TABLE_VALUES)
    final = F.wait_graph_task(task_id, "fraud")
    report["final_status"] = final.get("status")
    report["receipt_id"] = final.get("receipt_id") or ""
    report["zero_vwr"] = not final.get("receipt_id")

    # --- watcher on the fraudulent bundle (cheap-op detection) ---------------
    w_report = WatcherV2(bundle_path, task_meta, test_seed=7).run()
    report["watcher_verdict"] = w_report["verdict"]
    cheap = w_report["math_phase"]["cheap_ops"]
    report["watcher_cheap_mismatches"] = cheap["mismatches"][:2]
    report["watcher_cheap_nodes"] = cheap["nodes"]

    report["four_validators"] = {}
    for v in F.RPCS:
        st = F.rpc(F.RPCS[v], "/status")["sync_info"]
        report["four_validators"][v] = {"height": int(st["latest_block_height"]),
                                        "app_hash": st["latest_app_hash"]}
    hashes = {d["app_hash"] for d in report["four_validators"].values()}
    report["four_validators_converged"] = len(hashes) == 1
    report["status"] = ("PASS" if report["final_status"] == "fraud" and report["zero_vwr"]
                        and report["watcher_verdict"] == "fraud_detected"
                        and report["four_validators_converged"] else "FAIL")
    out = REPO / "docs" / "phase-f5c-e2e-rope-fraud.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in
                      ("status", "task_id", "final_status", "zero_vwr", "watcher_verdict",
                       "dispute_status_after_midpoint", "four_validators_converged")}, indent=1))
    print("written:", out)


if __name__ == "__main__":
    main()
