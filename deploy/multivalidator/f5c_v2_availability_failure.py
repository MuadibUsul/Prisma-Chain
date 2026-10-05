"""F.5C A6-11: DA availability-failure devnet E2E (four-validator chain).

A worker commits an honest V2 result but only one DA provider attests
(quorum is two).  After the challenge window plus the DA window closes, the
permissionless `fail-graph-availability` path must settle the task as
availability_failed: the requester escrow is refunded, the worker
reservation is released, and NO VerifiedGraphWorkReceiptV3 is issued.

    python deploy/multivalidator/f5c_v2_availability_failure.py

Writes docs/phase-f5c-e2e-availability-failure.json.
"""

from __future__ import annotations

import base64
import json
import os
import sys
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
from f5c_bundle_v2 import decode_bundle  # noqa: E402
from f5c_commit_v3 import build_signed_commit_v3  # noqa: E402
from f5c_v2_rope_fraud import build_rope_chain_v2_graph, funded, bonded  # noqa: E402


def main() -> None:
    report = {"scenario": "availability_failure", "chain_id": "prisma-mv-1"}
    F.with_service("validator-a")

    req_name, req_addr, req_seed, _, _ = funded("areq", 40_000_000)
    wk_name, wk_addr, wk_seed, wk_priv, wk_pub = funded("awrk", 40_000_000)
    bonded(wk_name, wk_addr, wk_seed, wk_pub)
    provs = []
    for tag in ("aa", "ab"):
        p_name, p_addr, p_seed, p_priv, p_pub = funded(f"a{tag}", 40_000_000)
        bonded(p_name, p_addr, p_seed, p_pub)
        F.send_tx_big("register-da-provider", p_name)
        provs.append({"name": p_name, "seed": p_seed})

    graph_doc, inputs, honest = build_rope_chain_v2_graph()
    honest_roots = node_roots_hex(graph_doc, honest)
    req_balance_before = G.balance(req_addr)
    report["requester_balance_before_post"] = req_balance_before

    nonce = os.urandom(16)
    key_proof = F.requester_key_proof_graph(req_addr, req_seed,
                                            F.requester_pub_bytes(req_seed), nonce)
    F.send_tx_big("post-graph-task", req_name,
                  requester_protocol_pubkey=F.hexb(F.requester_pub_bytes(req_seed)),
                  requester_key_proof=F.hexb(key_proof), requester_nonce=F.hexb(nonce),
                  graph_json=F.hexb(json.dumps(graph_doc, separators=(",", ":")).encode()),
                  input_data_ref="dev://f5c-availability", challenge_window=F.CHALLENGE_WINDOW,
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
    report["result_submitted_height"] = int(task["result_submitted_height"])
    _ = manifest

    # only ONE of two providers attests: quorum (2) is not reachable
    bundle = bundle_for(graph_doc, inputs, honest)
    bundle_path = REPO / "testdata" / "f5c_tmp" / f"availability_bundle_{task_id}.bin"
    bundle_path.write_bytes(bundle)
    hdr = decode_bundle(bundle)["header"]
    task_meta = {"graph_id_v2": hdr["graph_id_v2"], "manifest_root_v2": hdr["manifest_root_v2"],
                 "final_output_root": hdr["final_output_root"], "policy_id": hdr["policy_id"],
                 "graph_task_id": task_id}
    meta_path = REPO / "testdata" / "f5c_tmp" / f"availability_task_{task_id}.json"
    meta_path.write_text(json.dumps(task_meta), encoding="utf-8")
    from f5c_da_provider import attest as da_attest
    h_now = F.height()
    att_path = REPO / "testdata" / "f5c_tmp" / f"availability_att_{task_id}_{provs[0]['name']}.json"
    da_attest(bundle_path, meta_path, provs[0]["name"], provs[0]["seed"], h_now + 500,
              att_path, attested_height=h_now)
    F.send_tx_big("submit-graph-da-attestation", provs[0]["name"], graph_task_id=task_id,
                  attestation_json=F.hexb(att_path.read_bytes()))
    report["da_attestations"] = 1
    report["da_quorum_required"] = 2

    # settle after challenge window + DA window (+ margin for included height)
    required_until = int(task["result_submitted_height"]) + F.CHALLENGE_WINDOW + 30
    report["fail_not_before_height"] = required_until
    # devnet block time is 5s: the DA window is ~60 blocks
    wait_height(required_until + 2, timeout=900)
    F.send_tx_big("fail-graph-availability", req_name, graph_task_id=task_id)
    final = F.wait_graph_task(task_id, "availability_failed")
    report["final_status"] = final.get("status")
    report["receipt_id"] = final.get("receipt_id") or ""
    report["zero_vwr"] = not final.get("receipt_id")

    req_balance_after = G.balance(req_addr)
    report["requester_balance_after_failure"] = req_balance_after
    report["requester_refunded"] = req_balance_after == req_balance_before

    report["four_validators"] = {}
    for v in F.RPCS:
        st = F.rpc(F.RPCS[v], "/status")["sync_info"]
        report["four_validators"][v] = {"height": int(st["latest_block_height"]),
                                        "app_hash": st["latest_app_hash"]}
    hashes = {d["app_hash"] for d in report["four_validators"].values()}
    report["four_validators_converged"] = len(hashes) == 1
    report["status"] = ("PASS" if report["final_status"] == "availability_failed"
                        and report["zero_vwr"] and report["requester_refunded"]
                        and report["four_validators_converged"] else "FAIL")
    out = REPO / "docs" / "phase-f5c-e2e-availability-failure.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in
                      ("status", "task_id", "final_status", "zero_vwr", "requester_refunded",
                       "four_validators_converged")}, indent=1))
    print("written:", out)


if __name__ == "__main__":
    main()
