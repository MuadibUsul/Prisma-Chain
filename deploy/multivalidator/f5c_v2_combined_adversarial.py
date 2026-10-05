"""F.5C A6-08: combined adversarial devnet E2E (four-validator chain).

Three adversities at once against one CANONICAL_GRAPH_V2 task:

  1. a fraudulent worker result (self-consistent GEMM fraud, valid CommitV3),
  2. DA quorum loss: only one of the two providers attests the bundle,
  3. a censoring proposer: validator-a runs the development-only proposer
     omission harness (PRISMA_DEV_CENSOR_GEMM_CHALLENGES=all, now covering
     the CANONICAL_GRAPH_V2 dispute family).

The honest challenger broadcasts the challenge exactly when validator-a is
about to propose, so the omission window is observed objectively per block;
a later honest proposer must include it within a bounded number of blocks
and the whole wide-dispute settlement must still complete: challenger_wins
-> fraud -> requester refunded, zero VWR.  The watcher still detects the
fraud on the (only partially attested) bundle.

    python deploy/multivalidator/f5c_v2_combined_adversarial.py

Writes docs/phase-f5c-e2e-combined-adversarial.json.
"""

from __future__ import annotations

import base64
import hashlib
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
from f5c_watcher_v2 import WatcherV2  # noqa: E402
from f5c_v2_e2e import build_small_v2_graph, bundle_for, node_roots_hex  # noqa: E402
from f5c_v2_gemm_fraud import funded, bonded, state_root_v2  # noqa: E402

CENSOR = "validator-a"


# --- CometBFT block/proposal observation (shared style with the E1 harness) ---

def rpc_base(base: str, path: str):
    import urllib.request
    with urllib.request.urlopen(base + path, timeout=5) as resp:
        return json.load(resp)["result"]


def validator_addresses() -> dict:
    addresses = {}
    for name, base in F.RPCS.items():
        try:
            info = rpc_base(base, "/status")["validator_info"]
            if info.get("address"):
                addresses[info["address"]] = name
        except Exception:
            pass
    return addresses


def tx_hash_of(raw_b64: str) -> str:
    return hashlib.sha256(base64.b64decode(raw_b64)).hexdigest().upper()


def block_info(height: int) -> dict:
    raw = rpc_base(F.RPCS["validator-b"], f"/block?height={height}")
    header = raw["block"]["header"]
    txs = raw["block"]["data"].get("txs") or []
    return {"height": height, "proposer": header["proposer_address"].upper(),
            "tx_hashes": [tx_hash_of(tx) for tx in txs]}


def latest_height() -> int:
    return int(rpc_base(F.RPCS["validator-b"], "/status")["sync_info"]["latest_block_height"])


def learn_proposer_cycle(addresses: dict) -> dict:
    top = latest_height()
    order = []
    for height in range(top - 11, top + 1):
        try:
            proposer = block_info(height)["proposer"]
        except Exception:
            continue
        name = addresses.get(proposer)
        if name and (not order or order[-1] != name):
            order.append(name)
    return {order[i]: order[i + 1] for i in range(len(order) - 1)}


def wait_for_predecessor_of(target: str, addresses: dict, seconds: int = 300) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        nxt = learn_proposer_cycle(addresses)
        current = latest_height()
        name = addresses.get(block_info(current)["proposer"])
        if nxt.get(name) == target:
            time.sleep(0.15)
            if latest_height() == current:
                return {"height": current, "last_proposer": name, "next_expected": target}
        time.sleep(0.2)
    raise RuntimeError(f"never observed the predecessor of {target}")


def send_tx_observe(msg: str, frm: str, **flags) -> str:
    """Broadcast without waiting for inclusion; returns the CometBFT tx hash."""
    args = ["tx", "compute", msg, "--from", frm, "-b", "sync", "-y", "-o", "json",
            "--chain-id", G.CHAIN_ID, "--fees", "0uprsm", "--gas", F.GAS, *G.KEYRING_FLAGS]
    for key, value in flags.items():
        if isinstance(value, list):
            for item in value:
                args += [f"--{key.replace('_', '-')}", str(item)]
        else:
            args += [f"--{key.replace('_', '-')}", str(value)]
    out = F.cli_big(args)
    payload = json.loads(out[out.index("{"):])
    if int(payload.get("code", 1)) != 0:
        raise RuntimeError(f"{msg} rejected: code={payload.get('code')} log={payload.get('raw_log')}")
    return payload["txhash"]


def observe_inclusion(txhash: str, from_height: int, addresses: dict, seconds: int = 180) -> dict:
    observation = {"broadcast_height": from_height, "blocks": [], "censored_by": [],
                   "included_height": None, "included_by": None}
    deadline = time.time() + seconds
    height = from_height
    while time.time() < deadline:
        current = latest_height()
        while height <= current:
            info = block_info(height)
            has_tx = txhash.upper() in info["tx_hashes"]
            proposer = addresses.get(info["proposer"], info["proposer"][:12])
            observation["blocks"].append({"height": height, "proposer": proposer,
                                          "included": has_tx})
            if has_tx and observation["included_height"] is None:
                observation["included_height"] = height
                observation["included_by"] = proposer
            if not has_tx and proposer == CENSOR:
                observation["censored_by"].append(height)
            height += 1
        if observation["included_height"] is not None:
            observation["inclusion_delay_blocks"] = \
                observation["included_height"] - observation["broadcast_height"]
            return observation
        time.sleep(1)
    raise RuntimeError(f"tx not included within {seconds}s: {observation}")


def main() -> None:
    report = {"scenario": "combined_adversarial", "chain_id": "prisma-mv-1",
              "censoring_validator": CENSOR}
    F.with_service("validator-b")
    addresses = validator_addresses()
    report["validator_addresses"] = addresses

    req_name, req_addr, req_seed, _, _ = funded("creq", 40_000_000)
    wk_name, wk_addr, wk_seed, wk_priv, wk_pub = funded("cwrk", 40_000_000)
    ch_name, ch_addr, ch_seed, _, ch_pub = funded("cchal", 40_000_000)
    bonded(wk_name, wk_addr, wk_seed, wk_pub)
    bonded(ch_name, ch_addr, ch_seed, ch_pub)
    provs = []
    for tag in ("ca", "cb"):
        p_name, p_addr, p_seed, p_priv, p_pub = funded(f"c{tag}", 40_000_000)
        bonded(p_name, p_addr, p_seed, p_pub)
        F.send_tx_big("register-da-provider", p_name)
        provs.append({"name": p_name, "seed": p_seed})

    graph_doc, inputs, honest = build_small_v2_graph()
    gid = graph_id_v2_of(graph_doc)
    report["graph_id_v2"] = gid.hex()
    fraud0 = honest[0].copy()
    fraud0[0, 0] += 1
    fraud = {0: fraud0, 1: np.clip(fraud0, R.A13_MIN, R.A13_MAX)}
    honest_roots = node_roots_hex(graph_doc, honest)
    fraud_roots = node_roots_hex(graph_doc, fraud)

    req_balance_before = G.balance(req_addr)
    report["requester_balance_before"] = req_balance_before
    nonce = os.urandom(16)
    key_proof = F.requester_key_proof_graph(req_addr, req_seed,
                                            F.requester_pub_bytes(req_seed), nonce)
    F.send_tx_big("post-graph-task", req_name,
                  requester_protocol_pubkey=F.hexb(F.requester_pub_bytes(req_seed)),
                  requester_key_proof=F.hexb(key_proof), requester_nonce=F.hexb(nonce),
                  graph_json=F.hexb(json.dumps(graph_doc, separators=(",", ":")).encode()),
                  input_data_ref="dev://f5c-combined", challenge_window=F.CHALLENGE_WINDOW,
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

    # adversity 2: only ONE provider attests (the other stays offline)
    fraud_bundle = bundle_for(graph_doc, inputs, fraud)
    bundle_path = REPO / "testdata" / "f5c_tmp" / f"combined_bundle_{task_id}.bin"
    bundle_path.write_bytes(fraud_bundle)
    hdr = decode_bundle(fraud_bundle)["header"]
    task_meta = {"graph_id_v2": hdr["graph_id_v2"], "manifest_root_v2": hdr["manifest_root_v2"],
                 "final_output_root": hdr["final_output_root"], "policy_id": hdr["policy_id"],
                 "graph_task_id": task_id}
    meta_path = REPO / "testdata" / "f5c_tmp" / f"combined_task_{task_id}.json"
    meta_path.write_text(json.dumps(task_meta), encoding="utf-8")
    from f5c_da_provider import attest as da_attest
    h_now = F.height()
    att_path = REPO / "testdata" / "f5c_tmp" / f"combined_att_{task_id}_{provs[0]['name']}.json"
    da_attest(bundle_path, meta_path, provs[0]["name"], provs[0]["seed"], h_now + 500,
              att_path, attested_height=h_now)
    F.send_tx_big("submit-graph-da-attestation", provs[0]["name"], graph_task_id=task_id,
                  attestation_json=F.hexb(att_path.read_bytes()))
    report["da_attestations"] = 1
    report["da_quorum_required"] = 2

    # --- adversity 3: challenge broadcast right before the censor proposes ---
    ready = wait_for_predecessor_of(CENSOR, addresses)
    report["broadcast_window"] = ready
    txhash = send_tx_observe("open-graph-challenge", ch_name, graph_task_id=task_id,
                             challenger_output_roots=F.hexb(bytes.fromhex(honest_roots[-1])),
                             challenge_bond=F.MIN_BOND)
    observation = observe_inclusion(txhash, ready["height"], addresses)
    report["challenge_inclusion"] = observation
    report["censor_omission_window_observed"] = (
        len(observation["censored_by"]) >= 1
        and observation["included_by"] != CENSOR
        and observation["inclusion_delay_blocks"] <= 8)

    # --- the whole V2 dispute must still complete -----------------------------
    a_root = R.tensor_root_v2(graph_doc["inputs"][0]["desc"], inputs[0].reshape(-1).tolist())
    w_root = R.tensor_root_v2(graph_doc["inputs"][1]["desc"], inputs[1].reshape(-1).tolist())
    s0 = state_root_v2({(0, 0): a_root, (0, 1): w_root})

    def trail(n0, n1):
        s1 = state_root_v2({(0, 0): a_root, (0, 1): w_root, (1, 0): bytes.fromhex(n0)})
        s2 = state_root_v2({(0, 0): a_root, (0, 1): w_root,
                            (1, 0): bytes.fromhex(n0), (1, 1): bytes.fromhex(n1)})
        return [s0, s1, s2]

    fraud_trail = trail(fraud_roots[0], fraud_roots[1])
    honest_trail = trail(honest_roots[0], honest_roots[1])

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
        evidence.append({"desc_json": a_desc_json, "ref_kind": 0, "ref_index": 0,
                         "root": F.b64(a_root), "chunk_index": 0, "count": a_count,
                         "chunk": F.b64(a_chunk), "proof": a_chunk_proof,
                         "state_proof": [F.b64(leaf_w)]})
    for d in range(8):
        evidence.append({"desc_json": w_desc_json, "ref_kind": 0, "ref_index": 1,
                         "root": F.b64(w_root), "chunk_index": 0, "count": w_count,
                         "chunk": F.b64(w_chunk), "proof": w_chunk_proof,
                         "state_proof": [F.b64(leaf_a)]})

    # the arbitration is broadcast the same way and must also survive the
    # censoring proposer
    ready = wait_for_predecessor_of(CENSOR, addresses)
    arb_hash = send_tx_observe("arbitrate-wide512", ch_name, graph_task_id=task_id,
                               worker_next_state=F.hexb(fraud_s1_bytes),
                               worker_next_proof=[F.hexb(s) for s in worker_high_proof],
                               challenger_next_state=F.hexb(honest_s1_bytes),
                               challenger_next_proof=[F.hexb(s) for s in challenger_high_proof],
                               evidence=[json.dumps(ev, separators=(",", ":")) for ev in evidence])
    arb_observation = observe_inclusion(arb_hash, ready["height"], addresses)
    report["arbitration_inclusion"] = arb_observation

    final = F.wait_graph_task(task_id, "fraud")
    report["final_status"] = final.get("status")
    report["receipt_id"] = final.get("receipt_id") or ""
    report["zero_vwr"] = not final.get("receipt_id")
    req_balance_after = G.balance(req_addr)
    report["requester_balance_after"] = req_balance_after
    report["requester_refunded"] = req_balance_after == req_balance_before

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
    report["status"] = ("PASS" if report["censor_omission_window_observed"]
                        and report["final_status"] == "fraud" and report["zero_vwr"]
                        and report["requester_refunded"]
                        and report["watcher_verdict"] == "fraud_detected"
                        and report["four_validators_converged"] else "FAIL")
    out = REPO / "docs" / "phase-f5c-e2e-combined-adversarial.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in
                      ("status", "task_id", "censor_omission_window_observed",
                       "final_status", "zero_vwr", "requester_refunded",
                       "watcher_verdict", "four_validators_converged")}, indent=1))
    print("challenge inclusion:", json.dumps(report["challenge_inclusion"]["blocks"][:6]))
    print("written:", out)


if __name__ == "__main__":
    main()
