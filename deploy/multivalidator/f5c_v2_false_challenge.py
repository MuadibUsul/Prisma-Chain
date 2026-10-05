"""F.5C A6-07: false-challenge devnet E2E (four-validator chain).

The worker commits the HONEST result of the ROPE -> SILU -> ADD chain.  A
challenger opens a challenge with a fabricated counter-claim (one ROPE
output element altered, downstream recomputed).  The deterministic path
(challenge -> V2 trails -> bisection -> bounded typed arbitration) must
return WorkerWins: the task survives (status back to result_submitted),
the challenger's bond is burned, and after the challenge window the worker
finalizes with EXACTLY ONE VerifiedGraphWorkReceiptV3.

    python deploy/multivalidator/f5c_v2_false_challenge.py

Writes docs/phase-f5c-e2e-false-challenge.json.
"""

from __future__ import annotations

import base64
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
from f5c_v2_e2e import bundle_for, node_roots_hex, wait_height  # noqa: E402
from f5c_bundle_v2 import graph_id_v2_of, decode_bundle  # noqa: E402
from f5c_commit_v3 import build_signed_commit_v3  # noqa: E402
from f5c_exec_v2 import execute_v2  # noqa: E402
from f5c_v2_rope_fraud import build_rope_chain_v2_graph, chunk_of, funded, bonded, \
    state_root_v2, TABLE_VALUES as S_TABLE  # noqa: E402


def main() -> None:
    report = {"scenario": "false_challenge", "chain_id": "prisma-mv-1"}
    F.with_service("validator-a")

    req_name, req_addr, req_seed, _, _ = funded("freq", 40_000_000)
    wk_name, wk_addr, wk_seed, wk_priv, wk_pub = funded("fwrk", 40_000_000)
    ch_name, ch_addr, ch_seed, ch_priv, ch_pub = funded("fchal", 40_000_000)
    bonded(wk_name, wk_addr, wk_seed, wk_pub)
    bonded(ch_name, ch_addr, ch_seed, ch_pub)
    provs = []
    for tag in ("fa", "fb"):
        p_name, p_addr, p_seed, p_priv, p_pub = funded(f"f{tag}", 40_000_000)
        bonded(p_name, p_addr, p_seed, p_pub)
        F.send_tx_big("register-da-provider", p_name)
        provs.append({"name": p_name, "seed": p_seed})

    graph_doc, inputs, honest = build_rope_chain_v2_graph()
    gid = graph_id_v2_of(graph_doc)
    report["graph_id_v2"] = gid.hex()
    fraud = execute_v2(graph_doc, inputs, tamper=(0, 0, 1))
    honest_roots = node_roots_hex(graph_doc, honest)
    fraud_roots = node_roots_hex(graph_doc, fraud)

    nonce = os.urandom(16)
    key_proof = F.requester_key_proof_graph(req_addr, req_seed,
                                            F.requester_pub_bytes(req_seed), nonce)
    F.send_tx_big("post-graph-task", req_name,
                  requester_protocol_pubkey=F.hexb(F.requester_pub_bytes(req_seed)),
                  requester_key_proof=F.hexb(key_proof), requester_nonce=F.hexb(nonce),
                  graph_json=F.hexb(json.dumps(graph_doc, separators=(",", ":")).encode()),
                  input_data_ref="dev://f5c-false-challenge", challenge_window=F.CHALLENGE_WINDOW,
                  max_price_per_cwu=1000, max_fee=2_000_000)
    task_id = F.latest_task_id(report)
    report["task_id"] = task_id
    F.send_tx_big("accept-graph-task", wk_name, graph_task_id=task_id,
                  assignment_nonce=F.hexb(os.urandom(16)))
    task = F.wait_graph_task(task_id, "assigned")
    assignment_ref_hex = base64.b64decode(task["assignment_ref"]).hex()

    commit, _, manifest, final_root = build_signed_commit_v3(
        graph_doc, task_id, assignment_ref_hex, wk_pub, honest_roots, 42, wk_priv)
    F.send_tx_big("submit-graph-result-v2", wk_name, graph_task_id=task_id,
                  node_output_manifest_root=F.hexb(manifest),
                  output_roots=F.hexb(bytes.fromhex(honest_roots[-1])),
                  final_output_root=F.hexb(final_root),
                  completed_epoch=42,
                  worker_signature=F.hexb(base64.b64decode(commit["signature"])))
    task = F.wait_graph_task(task_id, "result_submitted")
    _ = manifest

    # honest bundle attested by two DA providers (quorum for the finalize)
    honest_bundle = bundle_for(graph_doc, inputs, honest)
    bundle_path = REPO / "testdata" / "f5c_tmp" / f"false_challenge_bundle_{task_id}.bin"
    bundle_path.write_bytes(honest_bundle)
    hdr = decode_bundle(honest_bundle)["header"]
    task_meta = {"graph_id_v2": hdr["graph_id_v2"], "manifest_root_v2": hdr["manifest_root_v2"],
                 "final_output_root": hdr["final_output_root"], "policy_id": hdr["policy_id"],
                 "graph_task_id": task_id}
    meta_path = REPO / "testdata" / "f5c_tmp" / f"false_challenge_task_{task_id}.json"
    meta_path.write_text(json.dumps(task_meta), encoding="utf-8")
    from f5c_da_provider import attest as da_attest
    h_now = F.height()
    for prov in provs:
        att_path = REPO / "testdata" / "f5c_tmp" / f"false_challenge_att_{task_id}_{prov['name']}.json"
        da_attest(bundle_path, meta_path, prov["name"], prov["seed"], h_now + 500,
                  att_path, attested_height=h_now)
        F.send_tx_big("submit-graph-da-attestation", prov["name"], graph_task_id=task_id,
                      attestation_json=F.hexb(att_path.read_bytes()))

    # --- false challenge: the challenger asserts the fabricated result --------
    ch_balance_before = G.balance(ch_addr)
    report["challenger_balance_before"] = ch_balance_before
    F.send_tx_big("open-graph-challenge", ch_name, graph_task_id=task_id,
                  challenger_output_roots=F.hexb(bytes.fromhex(fraud_roots[-1])),
                  challenge_bond=F.MIN_BOND)

    # --- V2 trails: worker locks the honest trail, challenger the fabricated --
    x_root = base64.b64decode(graph_doc["inputs"][0]["root"])
    t_root = R.tensor_root_v2(graph_doc["inputs"][1]["desc"], S_TABLE)
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

    claim(wk_name, honest_trail)
    claim(ch_name, fraud_trail)

    rec = F.graph_dispute(task_id)
    snap = json.loads(base64.b64decode(rec["snapshot"]).decode())
    low, high = int(snap["low"]), int(snap["high"])
    report["bisection_interval"] = [low, high]
    mid = low + (high - low) // 2
    for party, t in ((wk_name, honest_trail), (ch_name, fraud_trail)):
        proof = R.trail_proof_v2(gid, t, mid)
        time.sleep(2)
        F.send_tx_big("graph-mid-point", party, graph_task_id=task_id,
                      state_root=F.hexb(t[mid]),
                      proof_siblings=[F.hexb(s) for s in proof],
                      epoch=F.height())
    rec = F.graph_dispute(task_id)
    report["dispute_status_after_midpoint"] = rec.get("status")

    # --- arbitration: the worker's chunk is the true computation --------------
    x_chunk, x_chunk_proof, x_count = chunk_of(graph_doc["inputs"][0]["desc"],
                                               inputs[0].reshape(-1), 0)
    w_honest_chunk, _, _ = chunk_of(graph_doc["nodes"][0]["output"],
                                    honest[0].reshape(-1), 0)
    w_fraud_chunk, _, _ = chunk_of(graph_doc["nodes"][0]["output"],
                                   fraud[0].reshape(-1), 0)
    leaf_t = R.hash_bytes(R.DOMAIN_GRAPH_STATE_V2, R._u32be(0), R._u32be(1), t_root)
    evidence = [{
        "desc_json": F.b64(json.dumps(graph_doc["inputs"][0]["desc"],
                                      separators=(",", ":")).encode()),
        "ref_kind": 0, "ref_index": 0, "root": F.b64(x_root),
        "chunk_index": 0, "count": x_count, "chunk": F.b64(x_chunk),
        "proof": x_chunk_proof, "state_proof": [F.b64(leaf_t)],
    }]
    res = F.send_tx_big("arbitrate-graph-node", ch_name, graph_task_id=task_id,
                        worker_out_root=honest_roots[0],
                        worker_chunk_index=0,
                        worker_chunk=F.hexb(w_honest_chunk),
                        worker_chunk_proof=[],
                        challenger_out_root=fraud_roots[0],
                        challenger_chunk_index=0,
                        challenger_chunk=F.hexb(w_fraud_chunk),
                        challenger_chunk_proof=[],
                        evidence=[json.dumps(ev, separators=(",", ":")) for ev in evidence],
                        rope_table=S_TABLE)
    _ = res
    task = F.wait_graph_task(task_id, "result_submitted")
    report["status_after_arbitration"] = task.get("status")
    report["survived_challenge"] = bool(task.get("survived_challenge"))
    ch_balance_after_arb = G.balance(ch_addr)
    report["challenger_balance_after_arbitration"] = ch_balance_after_arb
    report["challenger_bond_burned"] = (ch_balance_before - ch_balance_after_arb) == F.MIN_BOND

    # --- the worker finalizes exactly once after the window -------------------
    wait_height(int(task["challenge_end"]) + 1)
    F.send_tx_big("finalize-graph-task", req_name, graph_task_id=task_id)
    final = F.wait_graph_task(task_id, "finalized")
    report["final_status"] = final.get("status")
    report["receipt_id"] = final.get("receipt_id") or ""
    report["exactly_one_vwr"] = bool(final.get("receipt_id"))
    # a second finalize must be refused (no duplicate receipt)
    second = None
    try:
        F.send_tx_big("finalize-graph-task", req_name, graph_task_id=task_id)
        second = "accepted"
    except RuntimeError as exc:
        second = str(exc)
    report["second_finalize"] = second

    report["four_validators"] = {}
    for v in F.RPCS:
        st = F.rpc(F.RPCS[v], "/status")["sync_info"]
        report["four_validators"][v] = {"height": int(st["latest_block_height"]),
                                        "app_hash": st["latest_app_hash"]}
    hashes = {d["app_hash"] for d in report["four_validators"].values()}
    report["four_validators_converged"] = len(hashes) == 1
    report["status"] = ("PASS" if report["status_after_arbitration"] == "result_submitted"
                        and report["challenger_bond_burned"] and report["exactly_one_vwr"]
                        and second != "accepted"
                        and report["four_validators_converged"] else "FAIL")
    out = REPO / "docs" / "phase-f5c-e2e-false-challenge.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in
                      ("status", "task_id", "status_after_arbitration", "survived_challenge",
                       "challenger_bond_burned", "exactly_one_vwr",
                       "four_validators_converged")}, indent=1))
    print("written:", out)


if __name__ == "__main__":
    main()
