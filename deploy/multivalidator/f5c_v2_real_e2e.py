"""F.5C A6-02/A6-04/A6-06: REAL 310-node Qwen3-block devnet E2E driver.

Runs the roadmap's real-graph scenarios against the four-validator devnet:

    --scenario honest   A6-02: post the real Graph V2 block -> exec -> CommitV3
                        -> 2 DA attestations -> finalize -> exactly one VWR V3
    --scenario gemm     A6-04: self-consistent fraud at the real wide GEMM node
                        145 -> challenge -> V2 trail bisection (9 rounds) ->
                        graph->wide bridge -> K-trace bisection -> 512-MAC
                        arbitration -> challenger_wins, zero VWR
    --scenario rope     A6-06: self-consistent fraud at the real ROPE node 85
                        -> bisection localizes the ROPE node -> bounded
                        table-pinned arbitration -> challenger_wins, zero VWR

Each scenario runs the permissionless Watcher V2 on the committed bundle and
checks four-validator convergence.  Results go to docs/phase-f5c-e2e-<name>.json.

    python deploy/multivalidator/f5c_v2_real_e2e.py --scenario honest
"""

from __future__ import annotations

import argparse
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
from f5c_bundle_v2 import decode_bundle, graph_id_v2_of  # noqa: E402
from f5c_commit_v3 import build_signed_commit_v3  # noqa: E402
from f5c_exec_v2 import execute_v2, load_inputs_from_bundle  # noqa: E402
from f5c_watcher_v2 import WatcherV2  # noqa: E402
from f5c_v2_e2e import wait_height  # noqa: E402
from f5c_v2_rope_fraud import funded, bonded, chunk_of  # noqa: E402

REAL_GRAPH = REPO / "testdata" / "f5c_qwen3_block_v2.json"
REAL_NPZ = REPO / "testdata" / "f5b2_block_bundle.npz"


# --- real block helpers ------------------------------------------------------

def load_real():
    doc = json.loads(REAL_GRAPH.read_text(encoding="utf-8"))
    inputs = load_inputs_from_bundle(doc, REAL_NPZ)
    inputs = {int(k): np.asarray(v, dtype=np.int64) for k, v in inputs.items()}
    return doc, inputs


def input_roots(doc) -> list:
    return [base64.b64decode(e["root"]) for e in doc["inputs"]]


def node_roots(doc, outputs) -> list:
    return [R.tensor_root_v2(n["output"], outputs[int(n["node_id"])].reshape(-1).tolist())
            for n in doc["nodes"]]


def state_trail(roots_in: list, roots_nodes: list) -> list:
    """State roots S_0 .. S_N over the live set (inputs + node outputs)."""
    live = {(0, i): r for i, r in enumerate(roots_in)}
    states = [R.graph_state_root_v2(dict(live))]
    for i, r in enumerate(roots_nodes):
        live[(1, i)] = r
        states.append(R.graph_state_root_v2(dict(live)))
    return states


def node_tensor(doc, inputs, outputs, ref: dict):
    return outputs[int(ref["index"])] if ref["kind"] == 1 else inputs[int(ref["index"])]


def state_proof_v2(live: dict, ref: tuple):
    ids = sorted((int(k) << 32) | int(i) for (k, i) in live)
    leaves = [R.hash_bytes(R.DOMAIN_GRAPH_STATE_V2, R._u32be(tid >> 32),
                           R._u32be(tid & 0xFFFFFFFF), live[(tid >> 32, tid & 0xFFFFFFFF)])
              for tid in ids]
    levels = R.build_levels(leaves)
    index = ids.index((ref[0] << 32) | ref[1])
    return R.prove(levels, index), index, len(leaves)


def evidence_entry(doc, live, ref: dict, tensor, root: bytes, chunk_index: int):
    desc = doc["inputs"][int(ref["index"])]["desc"] if ref["kind"] == 0 \
        else doc["nodes"][int(ref["index"])]["output"]
    chunk, chunk_proof, count = chunk_of(desc, tensor.reshape(-1), chunk_index)
    state_siblings, _, _ = state_proof_v2(live, (int(ref["kind"]), int(ref["index"])))
    return {
        "desc_json": F.b64(json.dumps(desc, separators=(",", ":")).encode()),
        "ref_kind": int(ref["kind"]), "ref_index": int(ref["index"]),
        "root": F.b64(root), "chunk_index": int(chunk_index), "count": int(count),
        "chunk": F.b64(chunk),
        "proof": [F.b64(bytes.fromhex(s)) for s in chunk_proof],
        "state_proof": [F.b64(s) for s in state_siblings],
    }


def wide_geometry(doc, node_id: int):
    node = doc["nodes"][node_id]
    a_ref, w_ref = node["inputs"][0], node["inputs"][1]
    return node, a_ref, w_ref


def run_watcher(bundle_path: Path, task_meta: dict, report: dict) -> None:
    t0 = time.time()
    w = WatcherV2(bundle_path, task_meta, test_seed=7).run()
    report["watcher_verdict"] = w["verdict"]
    report["watcher_seconds"] = round(time.time() - t0, 1)
    fre = w["math_phase"]["freivalds"]
    report["watcher_gemm_mismatches"] = fre["mismatches"][:1]
    report["watcher_cheap_mismatches"] = w["math_phase"]["cheap_ops"]["mismatches"][:1]


def converge_report(report: dict) -> None:
    report["four_validators"] = {}
    for v in F.RPCS:
        st = F.rpc(F.RPCS[v], "/status")["sync_info"]
        report["four_validators"][v] = {"height": int(st["latest_block_height"]),
                                        "app_hash": st["latest_app_hash"]}
    report["four_validators_converged"] = len(
        {d["app_hash"] for d in report["four_validators"].values()}) == 1


def setup_actors(tag: str, n_provs: int = 3):
    req = funded(f"{tag}req", 60_000_000)
    wk = funded(f"{tag}wrk", 60_000_000)
    ch = funded(f"{tag}chal", 60_000_000)
    bonded(wk[0], wk[1], wk[2], wk[4])
    bonded(ch[0], ch[1], ch[2], ch[4])
    provs = []
    for i in range(n_provs):
        p = funded(f"{tag}p{i}", 60_000_000)
        bonded(p[0], p[1], p[2], p[4])
        F.send_tx_big("register-da-provider", p[0])
        provs.append({"name": p[0], "seed": p[2]})
    return req, wk, ch, provs


def post_accept_commit(doc, gid, req, wk, outputs, report, tmp_tag):
    req_name, req_addr, req_seed = req[0], req[1], req[2]
    wk_name, wk_priv, wk_pub = wk[0], wk[3], wk[4]
    nonce = os.urandom(16)
    key_proof = F.requester_key_proof_graph(req_addr, req_seed,
                                            F.requester_pub_bytes(req_seed), nonce)
    graph_json = json.dumps(doc, separators=(",", ":")).encode()
    # the 186KB descriptor exceeds Linux MAX_ARG_STRLEN: travel as a file
    F.send_tx_big_file("post-graph-task", req_name, {"graph_json": graph_json},
                       requester_protocol_pubkey=F.hexb(F.requester_pub_bytes(req_seed)),
                       requester_key_proof=F.hexb(key_proof), requester_nonce=F.hexb(nonce),
                       input_data_ref=f"dev://f5c-real-{tmp_tag}",
                       challenge_window=F.CHALLENGE_WINDOW,
                       max_price_per_cwu=1000, max_fee=4_000_000)
    task_id = F.latest_task_id(report)
    report["task_id"] = task_id
    F.send_tx_big("accept-graph-task", wk_name, graph_task_id=task_id,
                  assignment_nonce=F.hexb(os.urandom(16)))
    task = F.wait_graph_task(task_id, "assigned")
    assignment_ref_hex = base64.b64decode(task["assignment_ref"]).hex()
    roots_hex = [r.hex() for r in node_roots(doc, outputs)]
    commit, _, manifest, final_root = build_signed_commit_v3(
        doc, task_id, assignment_ref_hex, wk_pub, roots_hex, 42, wk_priv)
    F.send_tx_big("submit-graph-result-v2", wk_name, graph_task_id=task_id,
                  node_output_manifest_root=F.hexb(manifest),
                  output_roots=roots_hex[-1],
                  final_output_root=F.hexb(final_root),
                  completed_epoch=42,
                  worker_signature=F.hexb(base64.b64decode(commit["signature"])))
    F.wait_graph_task(task_id, "result_submitted")
    return task_id


def attest_bundle(doc, inputs, outputs, provs, task_id, tmp_tag, report,
                  attest_count: int = 2) -> Path:
    from f5c_bundle_v2 import build_bundle
    in_entries = [{"name": e["name"], "desc": e["desc"], "data": inputs[i]}
                  for i, e in enumerate(doc["inputs"])]
    out_entries = [{"node_id": int(n["node_id"]), "desc": n["output"],
                    "data": outputs[int(n["node_id"])]} for n in doc["nodes"]]
    bundle = build_bundle(doc, in_entries, out_entries)
    bundle_path = REPO / "testdata" / "f5c_tmp" / f"{tmp_tag}_{task_id}.bin"
    bundle_path.write_bytes(bundle)
    hdr = decode_bundle(bundle)["header"]
    task_meta = {"graph_id_v2": hdr["graph_id_v2"], "manifest_root_v2": hdr["manifest_root_v2"],
                 "final_output_root": hdr["final_output_root"], "policy_id": hdr["policy_id"],
                 "graph_task_id": task_id}
    meta_path = REPO / "testdata" / "f5c_tmp" / f"{tmp_tag}_{task_id}_task.json"
    meta_path.write_text(json.dumps(task_meta), encoding="utf-8")
    from f5c_da_provider import attest as da_attest
    h_now = F.height()
    for prov in provs[:attest_count]:
        att_path = REPO / "testdata" / "f5c_tmp" / f"{tmp_tag}_{task_id}_{prov['name']}.json"
        da_attest(bundle_path, meta_path, prov["name"], prov["seed"], h_now + 500,
                  att_path, attested_height=h_now)
        F.send_tx_big("submit-graph-da-attestation", prov["name"], graph_task_id=task_id,
                      attestation_json=F.hexb(att_path.read_bytes()))
    report["da_providers_registered"] = len(provs)
    report["da_offline_providers"] = len(provs) - attest_count
    report["da_attestations"] = attest_count
    report["bundle_bytes"] = len(bundle)
    report["task_meta"] = task_meta
    return bundle_path


def graph_bisection(gid, task_id, wk_name, ch_name, fraud_trail, honest_trail, report):
    """Drive the V2 trail bisection until arb_ready; returns (low, high)."""
    rounds = 0
    while True:
        rec = F.graph_dispute(task_id)
        snap = json.loads(base64.b64decode(rec["snapshot"]).decode())
        if rec["status"] == "arb_ready" or snap.get("arb_ready"):
            report["bisection_rounds"] = rounds
            return int(snap["low"]), int(snap["high"])
        low, high = int(snap["low"]), int(snap["high"])
        mid = low + (high - low) // 2
        for party, t in ((wk_name, fraud_trail), (ch_name, honest_trail)):
            proof = R.trail_proof_v2(gid, t, mid)
            F.send_tx_big("graph-mid-point", party, graph_task_id=task_id,
                          state_root=F.hexb(t[mid]),
                          proof_siblings=[F.hexb(s) for s in proof],
                          epoch=F.height())
        rounds += 1
        if rounds > 16:
            raise RuntimeError("graph bisection did not converge")


def wide_rounds(task_id, wk_name, ch_name, worker_states, challenger_states, report):
    """Drive the K-trace bisection to arb_ready; returns (low, high)."""
    rounds = 0
    while True:
        rec = F.store_get(b"widedispute:" + task_id.to_bytes(8, "big"))
        doc = json.loads(base64.b64decode(rec["dispute_json"]).decode())
        if rec["status"] == "wide_arb_ready" or doc.get("arb_ready"):
            report["wide_rounds"] = rounds
            return int(doc["low"]), int(doc["high"])
        low, high = int(doc["low"]), int(doc["high"])
        mid = low + (high - low) // 2
        for party, states in ((wk_name, worker_states), (ch_name, challenger_states)):
            proof = R.wide_trace_proof_v1(0, 0, states, mid)
            F.send_tx_big("wide-mid-point", party, graph_task_id=task_id,
                          state=F.hexb(states[mid]),
                          proof=[F.hexb(s) for s in proof],
                          epoch=F.height())
        rounds += 1
        if rounds > 16:
            raise RuntimeError("wide bisection did not converge")


# --- scenarios ---------------------------------------------------------------

def scenario_honest(report: dict) -> None:
    doc, inputs = load_real()
    gid = graph_id_v2_of(doc)
    report["graph_id_v2"] = gid.hex()
    report["nodes"] = len(doc["nodes"])
    req, wk, ch, provs = setup_actors("rh")
    _ = ch
    req_balance_before = G.balance(req[1])
    wk_balance_before = G.balance(wk[1])
    report["requester_balance_before"] = req_balance_before
    report["worker_balance_before"] = wk_balance_before
    honest = execute_v2(doc, inputs)
    task_id = post_accept_commit(doc, gid, req, wk, honest, report, "real_honest")
    bundle_path = attest_bundle(doc, inputs, honest, provs, task_id, "real_honest", report)
    task = F.wait_graph_task(task_id, "result_submitted")
    wait_height(int(task["challenge_end"]) + 1, timeout=900)
    F.send_tx_big("finalize-graph-task", req[0], graph_task_id=task_id)
    final = F.wait_graph_task(task_id, "finalized")
    report["final_status"] = final.get("status")
    report["receipt_id"] = final.get("receipt_id") or ""
    report["exactly_one_vwr"] = bool(final.get("receipt_id"))
    # economics: the requester escrow is spent (burn 20% / monitors 2x5% /
    # worker 70%), the worker receives its share, nothing is refunded
    max_fee = 4_000_000
    burn = max_fee // 5
    per_monitor = max_fee // 20
    worker_share = max_fee - burn - 2 * per_monitor
    report["fee_split"] = {"burn": burn, "per_monitor": per_monitor, "worker": worker_share}
    report["requester_balance_after"] = G.balance(req[1])
    report["worker_balance_after"] = G.balance(wk[1])
    report["requester_paid_escrow"] = (req_balance_before - report["requester_balance_after"]) == max_fee
    report["worker_paid_share"] = (report["worker_balance_after"] - wk_balance_before) == worker_share
    run_watcher(bundle_path, report["task_meta"], report)
    converge_report(report)
    report["status"] = ("PASS" if report["final_status"] == "finalized"
                        and report["exactly_one_vwr"] and report["requester_paid_escrow"]
                        and report["worker_paid_share"]
                        and report["watcher_verdict"] == "pass"
                        and report["four_validators_converged"] else "FAIL")


def scenario_fraud(report: dict, kind: str) -> None:
    node_id, index, delta = (145, 0, 1) if kind == "gemm" else (85, 0, 100000)
    doc, inputs = load_real()
    gid = graph_id_v2_of(doc)
    report["graph_id_v2"] = gid.hex()
    report["nodes"] = len(doc["nodes"])
    report["fraud_node"] = node_id
    report["fraud_index"] = index
    report["fraud_delta"] = delta
    req, wk, ch, provs = setup_actors("rf" if kind == "gemm" else "rr")
    req_balance_before = G.balance(req[1])
    report["requester_balance_before"] = req_balance_before

    honest = execute_v2(doc, inputs)
    fraud = execute_v2(doc, inputs, tamper=(node_id, index, delta))
    honest_roots_in = input_roots(doc)
    honest_roots = node_roots(doc, honest)
    fraud_roots = node_roots(doc, fraud)
    report["divergent_nodes"] = [i for i, (a, b) in enumerate(zip(honest_roots, fraud_roots)) if a != b][:5]

    task_id = post_accept_commit(doc, gid, req, wk, fraud, report, f"real_{kind}")
    bundle_path = attest_bundle(doc, inputs, fraud, provs, task_id, f"real_{kind}", report)

    from f5c_v2_combined_adversarial import (validator_addresses, send_tx_observe,
                                             wait_for_predecessor_of, observe_inclusion)
    addresses = validator_addresses()
    ready = wait_for_predecessor_of("validator-a", addresses)
    txhash = send_tx_observe("open-graph-challenge", ch[0], graph_task_id=task_id,
                             challenger_output_roots=F.hexb(bytes.fromhex(honest_roots[-1])),
                             challenge_bond=F.MIN_BOND)
    observation = observe_inclusion(txhash, ready["height"], addresses)
    report["challenge_inclusion"] = observation
    report["censor_omission_window_observed"] = (
        len(observation["censored_by"]) >= 1
        and observation["included_by"] != "validator-a"
        and observation["inclusion_delay_blocks"] <= 8)

    fraud_trail = state_trail(honest_roots_in, fraud_roots)
    honest_trail = state_trail(honest_roots_in, honest_roots)

    def claim(party: str, t: list) -> None:
        p0 = R.trail_proof_v2(gid, t, 0)
        p1 = R.trail_proof_v2(gid, t, len(t) - 1)
        F.send_tx_big("graph-trail-claim", party, graph_task_id=task_id,
                      trail_root=F.hexb(R.trail_root_v2(gid, t)),
                      initial_root=F.hexb(t[0]),
                      initial_proof=[F.hexb(s) for s in p0],
                      final_root=F.hexb(t[-1]),
                      final_proof=[F.hexb(s) for s in p1])

    claim(wk[0], fraud_trail)
    claim(ch[0], honest_trail)
    low, high = graph_bisection(gid, task_id, wk[0], ch[0], fraud_trail, honest_trail, report)
    report["first_divergent_node"] = low
    report["bisection_interval"] = [low, high]

    if kind == "gemm":
        scenario_wide(doc, inputs, honest, report, task_id, wk, ch, low)
    else:
        scenario_rope(doc, inputs, honest, fraud, report, task_id, wk, ch, low)

    report["requester_refunded"] = G.balance(req[1]) == report["requester_balance_before"]
    run_watcher(bundle_path, report["task_meta"], report)
    converge_report(report)
    report["status"] = ("PASS" if report["final_status"] == "fraud" and report["zero_vwr"]
                        and report.get("first_divergent_node") == node_id
                        and report.get("challenger_wins_settlement")
                        and report.get("requester_refunded")
                        and report["watcher_verdict"] == "fraud_detected"
                        and report["four_validators_converged"] else "FAIL")


def scenario_wide(doc, inputs, honest, report, task_id, wk, ch, node_id):
    node, a_ref, w_ref = wide_geometry(doc, node_id)
    a_shape = doc_check_shape(doc, a_ref)
    w_shape = doc_check_shape(doc, w_ref)
    m, k = a_shape
    n = w_shape[1]
    a_data = node_tensor(doc, inputs, honest, a_ref)
    w_data = node_tensor(doc, inputs, honest, w_ref)
    report["wide_geometry"] = {"m": m, "n": n, "k": k}

    F.send_tx_big("open-wide-gemm-dispute", ch[0], graph_task_id=task_id,
                  node_id=node_id, tile_i=0, tile_j=0, challenge_bond=F.MIN_BOND)

    steps = k // 8
    a_tiles = [np.asarray(a_data[0:8, 8 * r:8 * r + 8], dtype=np.int64) for r in range(steps)]
    w_tiles = [np.asarray(w_data[8 * r:8 * r + 8, 0:8], dtype=np.int64) for r in range(steps)]
    honest_states = []
    acc = np.zeros((8, 8), dtype=np.int64)
    for r in range(steps):
        acc = acc + a_tiles[r] @ w_tiles[r]
        honest_states.append(acc.copy())
    worker_states = [np.zeros((8, 8), dtype=np.int64)]
    challenger_states = [np.zeros((8, 8), dtype=np.int64)]
    for r, s in enumerate(honest_states):
        lie = s.copy()
        lie[0, 0] += 1  # the locked fraud: output element (0,0) off by one
        worker_states.append(lie)
        challenger_states.append(s.copy())
    honest_bytes = [R.wide_state_bytes(s.reshape(-1).tolist()) for s in challenger_states]
    fraud_bytes = [R.wide_state_bytes(s.reshape(-1).tolist()) for s in worker_states]

    def claim_trace(party: str, states_bytes: list) -> None:
        root = R.wide_trace_root_v1(0, 0, states_bytes)
        p0 = R.wide_trace_proof_v1(0, 0, states_bytes, 0)
        p1 = R.wide_trace_proof_v1(0, 0, states_bytes, len(states_bytes) - 1)
        F.send_tx_big("wide-trace-claim", party, graph_task_id=task_id,
                      trace_root=F.hexb(root),
                      initial_state=F.hexb(states_bytes[0]),
                      initial_proof=[F.hexb(s) for s in p0],
                      final_state=F.hexb(states_bytes[-1]),
                      final_proof=[F.hexb(s) for s in p1])

    claim_trace(wk[0], fraud_bytes)
    claim_trace(ch[0], honest_bytes)
    wlow, whigh = wide_rounds(task_id, wk[0], ch[0], fraud_bytes, honest_bytes, report)
    report["wide_first_divergent_step"] = wlow

    # evidence: live state at the graph dispute's low boundary (inputs + nodes < node_id)
    live = {(0, i): r for i, r in enumerate(input_roots(doc))}
    for i in range(node_id):
        live[(1, i)] = bytes.fromhex(node_roots(doc, honest)[i])
    evidence = []
    a_root = live[(int(a_ref["kind"]), int(a_ref["index"]))]
    w_root = live[(int(w_ref["kind"]), int(w_ref["index"]))]
    for i in range(8):  # A rows 0..7 (M=16)
        row_start = 0
        chunk_index = (i * k + row_start) // 64
        evidence.append(evidence_entry(doc, live, a_ref, a_data, a_root, chunk_index))
    for d in range(8):  # W rows k=0..7 (non-transposed, tile_j = 0)
        chunk_index = (d * n + 0) // 64
        evidence.append(evidence_entry(doc, live, w_ref, w_data, w_root, chunk_index))

    ch_balance_before = G.balance(ch[1])
    F.send_tx_big("arbitrate-wide512", ch[0], graph_task_id=task_id,
                  worker_next_state=F.hexb(fraud_bytes[whigh]),
                  worker_next_proof=[F.hexb(s) for s in R.wide_trace_proof_v1(0, 0, fraud_bytes, whigh)],
                  challenger_next_state=F.hexb(honest_bytes[whigh]),
                  challenger_next_proof=[F.hexb(s) for s in R.wide_trace_proof_v1(0, 0, honest_bytes, whigh)],
                  evidence=[json.dumps(ev, separators=(",", ":")) for ev in evidence])
    final = F.wait_graph_task(task_id, "fraud")
    report["final_status"] = final.get("status")
    report["receipt_id"] = final.get("receipt_id") or ""
    report["zero_vwr"] = not final.get("receipt_id")
    report["challenger_balance_delta"] = G.balance(ch[1]) - ch_balance_before
    report["challenger_wins_settlement"] = report["challenger_balance_delta"] > 0
    return final


def doc_check_shape(doc, ref: dict):
    desc = doc["inputs"][int(ref["index"])]["desc"] if ref["kind"] == 0 \
        else doc["nodes"][int(ref["index"])]["output"]
    return tuple(int(d) for d in desc["shape"])


def scenario_rope(doc, inputs, honest, fraud, report, task_id, wk, ch, node_id):
    node = doc["nodes"][node_id]
    a_ref, t_ref = node["inputs"][0], node["inputs"][1]
    a_data = node_tensor(doc, inputs, honest, a_ref)
    table_vals = node_tensor(doc, inputs, honest, t_ref).reshape(-1).tolist()
    report["rope_table_values"] = len(table_vals)
    live = {(0, i): r for i, r in enumerate(input_roots(doc))}
    honest_roots = node_roots(doc, honest)
    for i in range(node_id):
        live[(1, i)] = bytes.fromhex(honest_roots[i])
    a_root = live[(int(a_ref["kind"]), int(a_ref["index"]))]

    w_honest_chunk, w_honest_proof, _ = chunk_of(node["output"], honest[node_id].reshape(-1), 0)
    w_fraud_chunk, w_fraud_proof, _ = chunk_of(node["output"], fraud[node_id].reshape(-1), 0)
    evidence = [evidence_entry(doc, live, a_ref, a_data, a_root, 0)]
    ch_balance_before = G.balance(ch[1])
    F.send_tx_big("arbitrate-graph-node", ch[0], graph_task_id=task_id,
                  worker_out_root=node_roots(doc, fraud)[node_id],
                  worker_chunk_index=0,
                  worker_chunk=F.hexb(w_fraud_chunk),
                  worker_chunk_proof=[F.hexb(bytes.fromhex(s)) for s in w_fraud_proof],
                  challenger_out_root=honest_roots[node_id],
                  challenger_chunk_index=0,
                  challenger_chunk=F.hexb(w_honest_chunk),
                  challenger_chunk_proof=[F.hexb(bytes.fromhex(s)) for s in w_honest_proof],
                  evidence=[json.dumps(ev, separators=(",", ":")) for ev in evidence],
                  rope_table=table_vals)
    final = F.wait_graph_task(task_id, "fraud")
    report["final_status"] = final.get("status")
    report["receipt_id"] = final.get("receipt_id") or ""
    report["zero_vwr"] = not final.get("receipt_id")
    report["challenger_balance_delta"] = G.balance(ch[1]) - ch_balance_before
    report["challenger_wins_settlement"] = report["challenger_balance_delta"] > 0
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["honest", "gemm", "rope"], required=True)
    args = ap.parse_args()

    report = {"scenario": f"real_{args.scenario}", "chain_id": "prisma-mv-1",
              "graph": str(REAL_GRAPH.name)}
    F.with_service("validator-a")
    if args.scenario == "honest":
        scenario_honest(report)
        out = REPO / "docs" / "phase-f5c-e2e-honest.json"
    else:
        scenario_fraud(report, args.scenario)
        name = "gemm-fraud" if args.scenario == "gemm" else "rope-fraud"
        out = REPO / "docs" / f"phase-f5c-e2e-{name}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in
                      ("status", "scenario", "task_id", "final_status", "zero_vwr",
                       "first_divergent_node", "wide_first_divergent_step", "receipt_id",
                       "watcher_verdict", "four_validators_converged")}, indent=1))
    print("written:", out)


if __name__ == "__main__":
    main()
