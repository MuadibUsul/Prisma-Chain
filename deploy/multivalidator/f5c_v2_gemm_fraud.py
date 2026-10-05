"""F.5C A6-04: real GEMM-fraud devnet E2E (four-validator chain).

The worker commits a SELF-CONSISTENT fraudulent result (one wide-GEMM output
element altered in range, the downstream requant recomputed, manifest and
final roots recomputed, valid CommitV3 signature).  The honest challenger
drives the full deterministic path on chain: challenge -> V2 trail claims ->
midpoint -> first divergent node is the GEMM -> wide dispute -> K-trace
claims -> exactly 512-MAC arbitration -> ChallengerWins -> fraud status ->
zero VWR.  The permissionless Watcher V2 is run on the fraudulent bundle
and must detect and localize the fraud.

    python deploy/multivalidator/f5c_v2_gemm_fraud.py

Writes docs/phase-f5c-e2e-gemm-fraud.json.
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
from f5c_v2_e2e import build_small_v2_graph, bundle_for, node_roots_hex, wait_height  # noqa: E402
from f5c_bundle_v2 import graph_id_v2_of  # noqa: E402
from f5c_commit_v3 import build_signed_commit_v3  # noqa: E402
from f5c_watcher_v2 import WatcherV2  # noqa: E402


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


def main() -> None:
    report = {"scenario": "gemm_fraud", "chain_id": "prisma-mv-1"}
    F.with_service("validator-a")

    req_name, req_addr, req_seed, _, _ = funded("v2req", 40_000_000)
    wk_name, wk_addr, wk_seed, wk_priv, wk_pub = funded("v2wrk", 40_000_000)
    ch_name, ch_addr, ch_seed, _, ch_pub = funded("v2chal", 40_000_000)
    bonded(wk_name, wk_addr, wk_seed, wk_pub)
    bonded(ch_name, ch_addr, ch_seed, ch_pub)
    provs = []
    for tag in ("pa", "pb"):
        p_name, p_addr, p_seed, p_priv, p_pub = funded(f"v2{tag}", 40_000_000)
        bonded(p_name, p_addr, p_seed, p_pub)
        F.send_tx_big("register-da-provider", p_name)
        provs.append({"name": p_name, "seed": p_seed})

    graph_doc, inputs, honest = build_small_v2_graph()
    gid = graph_id_v2_of(graph_doc)
    report["graph_id_v2"] = gid.hex()

    # --- self-consistent fraudulent result ----------------------------------
    fraud0 = honest[0].copy()
    fraud0[0, 0] += 1
    fraud1 = np.clip(fraud0, R.A13_MIN, R.A13_MAX)
    fraud = {0: fraud0, 1: fraud1}
    honest_roots = node_roots_hex(graph_doc, honest)
    fraud_roots = node_roots_hex(graph_doc, fraud)

    nonce = os.urandom(16)
    key_proof = F.requester_key_proof_graph(req_addr, req_seed,
                                            F.requester_pub_bytes(req_seed), nonce)
    F.send_tx_big("post-graph-task", req_name,
                  requester_protocol_pubkey=F.hexb(F.requester_pub_bytes(req_seed)),
                  requester_key_proof=F.hexb(key_proof), requester_nonce=F.hexb(nonce),
                  graph_json=F.hexb(json.dumps(graph_doc, separators=(",", ":")).encode()),
                  input_data_ref="dev://f5c-fraud", challenge_window=F.CHALLENGE_WINDOW,
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
    report["fraud_committed_height"] = task.get("result_submitted_height")

    # DA stores the fraudulent bundle (quorum still reachable: the fraud is
    # mathematically self-consistent, so storage attestations succeed)
    fraud_bundle = bundle_for(graph_doc, inputs, fraud)
    bundle_path = REPO / "testdata" / "f5c_tmp" / f"fraud_bundle_{task_id}.bin"
    bundle_path.write_bytes(fraud_bundle)
    h_now = F.height()
    header = json.loads((REPO / "testdata" / "f5c_tmp" / f"task_{task_id}.json").read_text(encoding="utf-8")) \
        if (REPO / "testdata" / "f5c_tmp" / f"task_{task_id}.json").exists() else None
    header = None
    from f5c_bundle_v2 import decode_bundle
    hdr = decode_bundle(fraud_bundle)["header"]
    task_meta = {"graph_id_v2": hdr["graph_id_v2"], "manifest_root_v2": hdr["manifest_root_v2"],
                 "final_output_root": hdr["final_output_root"], "policy_id": hdr["policy_id"],
                 "graph_task_id": task_id}
    meta_path = REPO / "testdata" / "f5c_tmp" / f"fraud_task_{task_id}.json"
    meta_path.write_text(json.dumps(task_meta), encoding="utf-8")
    from f5c_da_provider import attest as da_attest
    for prov in provs:
        att_path = REPO / "testdata" / "f5c_tmp" / f"fraud_att_{task_id}_{prov['name']}.json"
        da_attest(bundle_path, meta_path, prov["name"], prov["seed"], h_now + 500,
                  att_path, attested_height=h_now)
        F.send_tx_big("submit-graph-da-attestation", prov["name"], graph_task_id=task_id,
                      attestation_json=F.hexb(att_path.read_bytes()))
    _ = manifest

    # --- challenge ------------------------------------------------------------
    F.send_tx_big("open-graph-challenge", ch_name, graph_task_id=task_id,
                  challenger_output_roots=F.hexb(bytes.fromhex(honest_roots[-1])),
                  challenge_bond=F.MIN_BOND)

    # --- V2 trails ------------------------------------------------------------
    a_root = R.tensor_root_v2(graph_doc["inputs"][0]["desc"], inputs[0].reshape(-1).tolist())
    w_root = R.tensor_root_v2(graph_doc["inputs"][1]["desc"], inputs[1].reshape(-1).tolist())
    s0 = state_root_v2({(0, 0): a_root, (0, 1): w_root})

    def trail(node0_root_hex, node1_root_hex):
        s1 = state_root_v2({(0, 0): a_root, (0, 1): w_root, (1, 0): bytes.fromhex(node0_root_hex)})
        s2 = state_root_v2({(0, 0): a_root, (0, 1): w_root,
                            (1, 0): bytes.fromhex(node0_root_hex),
                            (1, 1): bytes.fromhex(node1_root_hex)})
        return [s0, s1, s2]

    honest_trail = trail(honest_roots[0], honest_roots[1])
    fraud_trail = trail(fraud_roots[0], fraud_roots[1])

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

    # --- midpoint round (2-node graph -> one round reaches arb_ready) ---------
    def snapshot_interval():
        rec = F.graph_dispute(task_id)
        snap = json.loads(base64.b64decode(rec["snapshot"]).decode())
        return int(snap["low"]), int(snap["high"])

    low, high = snapshot_interval()
    mid = low + (high - low) // 2
    for party, t in ((wk_name, fraud_trail), (ch_name, honest_trail)):
        proof = R.trail_proof_v2(gid, t, mid)
        time.sleep(2)
        F.send_tx_big("graph-mid-point", party, graph_task_id=task_id,
                      state_root=F.hexb(t[mid]),
                      proof_siblings=[F.hexb(s) for s in proof],
                      epoch=F.height())
    rec = F.graph_dispute(task_id)
    report["dispute_status_before_wide"] = rec.get("status")

    # --- wide dispute ---------------------------------------------------------
    F.send_tx_big("open-wide-gemm-dispute", ch_name, graph_task_id=task_id,
                  node_id=0, tile_i=0, tile_j=0, challenge_bond=F.MIN_BOND)

    a_tile = np.zeros((8, 8), dtype=np.int64)
    a_tile[:2, :] = inputs[0]
    w_tile = inputs[1].astype(np.int64)
    honest_s1 = a_tile @ w_tile
    fraud_s1 = honest_s1.copy()
    fraud_s1[0, 0] += 1
    s0_bytes = R.wide_state_bytes([0] * 64)
    honest_s1_bytes = R.wide_state_bytes(honest_s1.reshape(-1).tolist())
    fraud_s1_bytes = R.wide_state_bytes(fraud_s1.reshape(-1).tolist())

    def claim_trace(party: str, s1_bytes: bytes) -> list:
        states = [s0_bytes, s1_bytes]
        root = R.wide_trace_root_v1(0, 0, states)
        p0 = R.wide_trace_proof_v1(0, 0, states, 0)
        p1 = R.wide_trace_proof_v1(0, 0, states, 1)
        F.send_tx_big("wide-trace-claim", party, graph_task_id=task_id,
                      trace_root=F.hexb(root),
                      initial_state=F.hexb(s0_bytes),
                      initial_proof=[F.hexb(s) for s in p0],
                      final_state=F.hexb(s1_bytes),
                      final_proof=[F.hexb(s) for s in p1])
        return p1

    worker_high_proof = claim_trace(wk_name, fraud_s1_bytes)
    challenger_high_proof = claim_trace(ch_name, honest_s1_bytes)

    # --- typed evidence: 8x8 operand tiles proven from committed chunks -------
    leaf_a = R.hash_bytes(R.DOMAIN_GRAPH_STATE_V2, R._u32be(0), R._u32be(0), a_root)
    leaf_w = R.hash_bytes(R.DOMAIN_GRAPH_STATE_V2, R._u32be(0), R._u32be(1), w_root)

    def chunk_of(desc: dict, flat: np.ndarray, index: int):
        width = {4: 4, 2: 2, 129: 2, 130: 2, 131: 8}[int(desc["dtype"])]
        be = {4: ">i4", 2: ">i2", 8: ">i8"}[width]
        elems = 1
        for d in desc["shape"]:
            elems *= d
        count = max(1, (elems + 63) // 64)
        padded = np.zeros(count * 64, dtype=np.int64)
        padded[:elems] = flat
        blob = padded.astype(be).tobytes()
        chunk = blob[index * 64 * width:(index + 1) * 64 * width]
        desc_bytes = R.encode_canonical(desc)
        level = [R.tensor_v2_leaf(desc_bytes, i,
                                  blob[i * 64 * width:(i + 1) * 64 * width])
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

    a_chunk, a_chunk_proof, a_count = chunk_of(graph_doc["inputs"][0]["desc"],
                                               inputs[0].reshape(-1), 0)
    w_chunk, w_chunk_proof, w_count = chunk_of(graph_doc["inputs"][1]["desc"],
                                               inputs[1].reshape(-1), 0)
    a_desc_json = F.b64(json.dumps(graph_doc["inputs"][0]["desc"],
                                   separators=(",", ":")).encode())
    w_desc_json = F.b64(json.dumps(graph_doc["inputs"][1]["desc"],
                                   separators=(",", ":")).encode())
    evidence = []
    for i in range(2):  # A rows 0..1 (M=2); the rest is canonical zero padding
        evidence.append({
            "desc_json": a_desc_json,
            "ref_kind": 0, "ref_index": 0, "root": F.b64(a_root),
            "chunk_index": 0, "count": a_count, "chunk": F.b64(a_chunk),
            "proof": a_chunk_proof, "state_proof": [F.b64(leaf_w)],
        })
    for d in range(8):  # W rows 0..7 (K=8)
        evidence.append({
            "desc_json": w_desc_json,
            "ref_kind": 0, "ref_index": 1, "root": F.b64(w_root),
            "chunk_index": 0, "count": w_count, "chunk": F.b64(w_chunk),
            "proof": w_chunk_proof, "state_proof": [F.b64(leaf_a)],
        })

    F.send_tx_big("arbitrate-wide512", ch_name, graph_task_id=task_id,
                  worker_next_state=F.hexb(fraud_s1_bytes),
                  worker_next_proof=[F.hexb(s) for s in worker_high_proof],
                  challenger_next_state=F.hexb(honest_s1_bytes),
                  challenger_next_proof=[F.hexb(s) for s in challenger_high_proof],
                  evidence=[json.dumps(ev, separators=(",", ":")) for ev in evidence])
    final = F.wait_graph_task(task_id, "fraud")
    report["final_status"] = final.get("status")
    report["receipt_id"] = final.get("receipt_id") or ""
    report["zero_vwr"] = not final.get("receipt_id")

    # --- watcher on the fraudulent bundle (detection + localization) ----------
    w_report = WatcherV2(bundle_path, task_meta, test_seed=7).run()
    report["watcher_verdict"] = w_report["verdict"]
    if w_report["verdict"] == "fraud_detected":
        report["watcher_localization"] = w_report["math_phase"]["freivalds"]["mismatches"][:1]

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
    out = REPO / "docs" / "phase-f5c-e2e-gemm-fraud.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in
                      ("status", "task_id", "final_status", "zero_vwr", "watcher_verdict",
                       "dispute_status_before_wide", "four_validators_converged")}, indent=1))
    print("written:", out)


if __name__ == "__main__":
    main()
