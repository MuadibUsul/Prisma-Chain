"""F.5C A6-09: wide-dispute restart devnet E2E (four-validator chain).

The A6-04 GEMM-fraud dispute is driven to wide-dispute arb-ready (both
K-trace claims locked), then ALL FOUR validators are restarted.  After the
stack reconverges the persisted dispute must be byte-identical (same
locked roots, same high/low, same first divergent K-step) and the resumed
512-MAC arbitration must produce the same verdict: challenger_wins ->
fraud -> zero VWR.  The watcher still localizes the fraud on the
fraudulent bundle.

    python deploy/multivalidator/f5c_v2_wide_restart.py

Writes docs/phase-f5c-e2e-wide-restart.json.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy"))
sys.path.insert(0, str(REPO / "deploy" / "multivalidator"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import numpy as np  # noqa: E402
import f_graph_test as F  # noqa: E402
import gemm_chain_smoke as GS  # noqa: E402
import canonical_ref as R  # noqa: E402
from f5c_bundle_v2 import decode_bundle, graph_id_v2_of  # noqa: E402
from f5c_commit_v3 import build_signed_commit_v3  # noqa: E402
from f5c_watcher_v2 import WatcherV2  # noqa: E402
from f5c_v2_e2e import build_small_v2_graph, bundle_for, node_roots_hex  # noqa: E402
from f5c_v2_gemm_fraud import funded, bonded, state_root_v2  # noqa: E402

VALIDATORS = ["validator-a", "validator-b", "validator-c", "validator-d"]


def wide_dispute_record(task_id: int):
    key = b"widedispute:" + task_id.to_bytes(8, "big")
    return F.store_get(key)


def dispute_fingerprint(rec: dict) -> dict:
    raw = base64.b64decode(rec["dispute_json"])
    doc = json.loads(raw.decode())
    return {"sha256": hashlib.sha256(raw).hexdigest(),
            "steps": doc["steps"], "high": doc["high"], "low": doc["low"],
            "roots": doc["roots"], "arb_ready": doc["arb_ready"],
            "high_s0": doc["high_s0"], "high_s1": doc["high_s1"]}


def restart_validators(report: dict) -> None:
    heights_before = {}
    for v in F.RPCS:
        st = F.rpc(F.RPCS[v], "/status")["sync_info"]
        heights_before[v] = int(st["latest_block_height"])
    report["heights_before_restart"] = heights_before
    t0 = time.time()
    for v in VALIDATORS:
        subprocess.run(["docker", "restart", f"prisma-multivalidator-{v}-1"],
                       check=True, capture_output=True, timeout=180)
    # wait until every validator answers again
    deadline = time.time() + 300
    while time.time() < deadline:
        heights = {}
        ok = True
        for v in F.RPCS:
            try:
                st = F.rpc(F.RPCS[v], "/status")["sync_info"]
                heights[v] = (int(st["latest_block_height"]), st["latest_app_hash"])
            except Exception:
                ok = False
                break
        if ok and len({h for h, _ in heights.values()}) == 1 and \
                len({a for _, a in heights.values()}) == 1:
            break
        time.sleep(3)
    report["restart_seconds"] = round(time.time() - t0, 1)
    # chain must advance again after the restart
    h0 = F.height()
    deadline = time.time() + 120
    while F.height() <= h0 and time.time() < deadline:
        time.sleep(2)
    report["height_advanced_after_restart"] = F.height() > h0


def main() -> None:
    report = {"scenario": "wide_dispute_restart", "chain_id": "prisma-mv-1"}
    F.with_service("validator-a")

    req_name, req_addr, req_seed, _, _ = funded("rreq", 40_000_000)
    wk_name, wk_addr, wk_seed, wk_priv, wk_pub = funded("rwrk", 40_000_000)
    ch_name, ch_addr, ch_seed, _, ch_pub = funded("rchal", 40_000_000)
    bonded(wk_name, wk_addr, wk_seed, wk_pub)
    bonded(ch_name, ch_addr, ch_seed, ch_pub)
    provs = []
    for tag in ("sa", "sb"):
        p_name, p_addr, p_seed, p_priv, p_pub = funded(f"s{tag}", 40_000_000)
        bonded(p_name, p_addr, p_seed, p_pub)
        F.send_tx_big("register-da-provider", p_name)
        provs.append({"name": p_name, "seed": p_seed})

    graph_doc, inputs, honest = build_small_v2_graph()
    gid = graph_id_v2_of(graph_doc)
    report["graph_id_v2"] = gid.hex()
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
                  input_data_ref="dev://f5c-wide-restart", challenge_window=F.CHALLENGE_WINDOW,
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
    F.wait_graph_task(task_id, "result_submitted")
    _ = manifest

    fraud_bundle = bundle_for(graph_doc, inputs, fraud)
    bundle_path = REPO / "testdata" / "f5c_tmp" / f"wide_restart_bundle_{task_id}.bin"
    bundle_path.write_bytes(fraud_bundle)
    hdr = decode_bundle(fraud_bundle)["header"]
    task_meta = {"graph_id_v2": hdr["graph_id_v2"], "manifest_root_v2": hdr["manifest_root_v2"],
                 "final_output_root": hdr["final_output_root"], "policy_id": hdr["policy_id"],
                 "graph_task_id": task_id}
    meta_path = REPO / "testdata" / "f5c_tmp" / f"wide_restart_task_{task_id}.json"
    meta_path.write_text(json.dumps(task_meta), encoding="utf-8")
    from f5c_da_provider import attest as da_attest
    h_now = F.height()
    for prov in provs:
        att_path = REPO / "testdata" / "f5c_tmp" / f"wide_restart_att_{task_id}_{prov['name']}.json"
        da_attest(bundle_path, meta_path, prov["name"], prov["seed"], h_now + 500,
                  att_path, attested_height=h_now)
        F.send_tx_big("submit-graph-da-attestation", prov["name"], graph_task_id=task_id,
                      attestation_json=F.hexb(att_path.read_bytes()))

    F.send_tx_big("open-graph-challenge", ch_name, graph_task_id=task_id,
                  challenger_output_roots=F.hexb(bytes.fromhex(honest_roots[-1])),
                  challenge_bond=F.MIN_BOND)

    a_root = R.tensor_root_v2(graph_doc["inputs"][0]["desc"], inputs[0].reshape(-1).tolist())
    w_root = R.tensor_root_v2(graph_doc["inputs"][1]["desc"], inputs[1].reshape(-1).tolist())
    s0 = state_root_v2({(0, 0): a_root, (0, 1): w_root})

    def trail(node0_hex, node1_hex):
        s1 = state_root_v2({(0, 0): a_root, (0, 1): w_root, (1, 0): bytes.fromhex(node0_hex)})
        s2 = state_root_v2({(0, 0): a_root, (0, 1): w_root,
                            (1, 0): bytes.fromhex(node0_hex),
                            (1, 1): bytes.fromhex(node1_hex)})
        return [s0, s1, s2]

    def claim(party: str, t: list) -> None:
        p0 = R.trail_proof_v2(gid, t, 0)
        p1 = R.trail_proof_v2(gid, t, len(t) - 1)
        F.send_tx_big("graph-trail-claim", party, graph_task_id=task_id,
                      trail_root=F.hexb(R.trail_root_v2(gid, t)),
                      initial_root=F.hexb(t[0]),
                      initial_proof=[F.hexb(s) for s in p0],
                      final_root=F.hexb(t[-1]),
                      final_proof=[F.hexb(s) for s in p1])

    fraud_trail = trail(fraud_roots[0], fraud_roots[1])
    honest_trail = trail(honest_roots[0], honest_roots[1])
    claim(wk_name, fraud_trail)
    claim(ch_name, honest_trail)

    rec = F.graph_dispute(task_id)
    snap = json.loads(base64.b64decode(rec["snapshot"]).decode())
    low, high = int(snap["low"]), int(snap["high"])
    mid = low + (high - low) // 2
    for party, t in ((wk_name, fraud_trail), (ch_name, honest_trail)):
        proof = R.trail_proof_v2(gid, t, mid)
        time.sleep(2)
        F.send_tx_big("graph-mid-point", party, graph_task_id=task_id,
                      state_root=F.hexb(t[mid]),
                      proof_siblings=[F.hexb(s) for s in proof],
                      epoch=F.height())

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
    wd = wide_dispute_record(task_id)
    report["wide_status_before_restart"] = wd.get("status")
    fp_before = dispute_fingerprint(wd)
    report["fingerprint_before_restart"] = fp_before

    # ==== restart every validator under the open dispute =====================
    restart_validators(report)

    # ==== the persisted dispute must be identical after the restart ==========
    wd_after = wide_dispute_record(task_id)
    fp_after = dispute_fingerprint(wd_after)
    report["fingerprint_after_restart"] = fp_after
    report["dispute_state_identical"] = (fp_before["sha256"] == fp_after["sha256"]
                                         and fp_before["high_s0"] == fp_after["high_s0"]
                                         and fp_before["high_s1"] == fp_after["high_s1"])

    # ==== resume: 512-MAC arbitration with the same evidence =================
    leaf_a = R.hash_bytes(R.DOMAIN_GRAPH_STATE_V2, R._u32be(0), R._u32be(0), a_root)
    leaf_w = R.hash_bytes(R.DOMAIN_GRAPH_STATE_V2, R._u32be(0), R._u32be(1), w_root)

    def chunk_of(desc: dict, flat, index: int):
        width = {4: 4, 2: 2, 129: 2, 130: 2, 131: 8}[int(desc["dtype"])]
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

    a_chunk, a_chunk_proof, a_count = chunk_of(graph_doc["inputs"][0]["desc"],
                                               inputs[0].reshape(-1), 0)
    w_chunk, w_chunk_proof, w_count = chunk_of(graph_doc["inputs"][1]["desc"],
                                               inputs[1].reshape(-1), 0)
    a_desc_json = F.b64(json.dumps(graph_doc["inputs"][0]["desc"], separators=(",", ":")).encode())
    w_desc_json = F.b64(json.dumps(graph_doc["inputs"][1]["desc"], separators=(",", ":")).encode())
    evidence = []
    for i in range(2):
        evidence.append({
            "desc_json": a_desc_json, "ref_kind": 0, "ref_index": 0, "root": F.b64(a_root),
            "chunk_index": 0, "count": a_count, "chunk": F.b64(a_chunk),
            "proof": a_chunk_proof, "state_proof": [F.b64(leaf_w)]})
    for d in range(8):
        evidence.append({
            "desc_json": w_desc_json, "ref_kind": 0, "ref_index": 1, "root": F.b64(w_root),
            "chunk_index": 0, "count": w_count, "chunk": F.b64(w_chunk),
            "proof": w_chunk_proof, "state_proof": [F.b64(leaf_a)]})

    ch_balance_before_arb = GS.balance(ch_addr)
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
    # ChallengerWins settles the bond back to the challenger (BothInvalid
    # would burn it); the wide dispute record is deleted on settlement.
    ch_balance_after_arb = GS.balance(ch_addr)
    report["challenger_balance_delta"] = ch_balance_after_arb - ch_balance_before_arb
    report["challenger_wins_settlement"] = ch_balance_after_arb > ch_balance_before_arb
    report["wide_dispute_deleted_on_settlement"] = wide_dispute_record(task_id) is None

    w_report = WatcherV2(bundle_path, task_meta, test_seed=7).run()
    report["watcher_verdict"] = w_report["verdict"]
    report["watcher_localization"] = w_report["math_phase"]["freivalds"]["mismatches"][:1]

    report["four_validators"] = {}
    for v in F.RPCS:
        st = F.rpc(F.RPCS[v], "/status")["sync_info"]
        report["four_validators"][v] = {"height": int(st["latest_block_height"]),
                                        "app_hash": st["latest_app_hash"]}
    hashes = {d["app_hash"] for d in report["four_validators"].values()}
    report["four_validators_converged"] = len(hashes) == 1
    report["status"] = ("PASS" if report["final_status"] == "fraud" and report["zero_vwr"]
                        and report["dispute_state_identical"]
                        and report["height_advanced_after_restart"]
                        and report["challenger_wins_settlement"]
                        and report["watcher_verdict"] == "fraud_detected"
                        and report["four_validators_converged"] else "FAIL")
    out = REPO / "docs" / "phase-f5c-e2e-wide-restart.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in
                      ("status", "task_id", "final_status", "zero_vwr",
                       "dispute_state_identical", "challenger_wins_settlement", "restart_seconds",
                       "four_validators_converged")}, indent=1))
    print("written:", out)


if __name__ == "__main__":
    main()
