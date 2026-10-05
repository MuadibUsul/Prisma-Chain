"""F.5C A6-10: validator offline / recovery devnet E2E.

validator-d is stopped while the remaining three validators keep producing
blocks (>= 2/3 of the power).  State-changing transactions are submitted
during the outage; after validator-d is restarted it must catch up to the
others' height and report the SAME app hash (state sync, no fork).

    python deploy/multivalidator/f5c_v2_validator_recovery.py

Writes docs/phase-f5c-e2e-validator-recovery.json.
"""

from __future__ import annotations

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

import f_graph_test as F  # noqa: E402
import gemm_chain_smoke as G  # noqa: E402
from f5c_v2_rope_fraud import funded, bonded  # noqa: E402

DOWN = "validator-d"
DOWN_RPC = F.RPCS[DOWN]
PEERS = ["validator-a", "validator-b", "validator-c"]


def heights():
    out = {}
    for v in F.RPCS:
        try:
            st = F.rpc(F.RPCS[v], "/status")["sync_info"]
            out[v] = {"height": int(st["latest_block_height"]),
                      "app_hash": st["latest_app_hash"]}
        except Exception:
            out[v] = None
    return out


def main() -> None:
    report = {"scenario": "validator_offline_recovery", "chain_id": "prisma-mv-1"}
    F.with_service("validator-a")

    before = heights()
    report["heights_before_outage"] = before

    # --- take validator-d down ------------------------------------------------
    subprocess.run(["docker", "stop", f"prisma-multivalidator-{DOWN}-1"],
                   check=True, capture_output=True, timeout=120)
    stop_height = F.height()
    report["stop_height"] = stop_height

    # state-changing txs while one validator is offline
    name, addr, seed, _, pub = funded("down", 40_000_000)
    bonded(name, addr, seed, pub)
    other = funded("down2", 40_000_000)
    bonded(other[0], other[1], other[2], other[4])
    report["txs_sent_during_outage"] = 2

    deadline = time.time() + 120
    while F.height() < stop_height + 5 and time.time() < deadline:
        time.sleep(2)
    report["height_advanced_without_d"] = F.height() >= stop_height + 5

    during = heights()
    report["heights_during_outage"] = during
    peers_now = {during[v]["app_hash"] for v in PEERS if during[v]}
    report["peers_converged_during_outage"] = len(peers_now) == 1

    # --- bring validator-d back and let it catch up ---------------------------
    t0 = time.time()
    subprocess.run(["docker", "start", f"prisma-multivalidator-{DOWN}-1"],
                   check=True, capture_output=True, timeout=120)
    recovered = None
    deadline = time.time() + 300
    while time.time() < deadline:
        now = heights()
        if all(now.values()) and len({d["app_hash"] for d in now.values()}) == 1 and \
                len({d["height"] for d in now.values() if d["height"] > 0}) <= 2:
            recovered = now
            break
        time.sleep(3)
    report["recovery_seconds"] = round(time.time() - t0, 1)
    report["heights_after_recovery"] = recovered
    report["recovered_converged"] = recovered is not None
    if recovered:
        report["app_hash_after_recovery"] = recovered["validator-a"]["app_hash"]
        report["validator_d_height_lag"] = (recovered["validator-a"]["height"]
                                            - recovered[DOWN]["height"])

    out = REPO / "docs" / "phase-f5c-e2e-validator-recovery.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    report["status"] = ("PASS" if report["height_advanced_without_d"]
                        and report["peers_converged_during_outage"]
                        and report.get("recovered_converged", False) else "FAIL")
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in
                      ("status", "stop_height", "height_advanced_without_d",
                       "peers_converged_during_outage", "recovered_converged",
                       "recovery_seconds", "validator_d_height_lag")}, indent=1))
    print("written:", out)


if __name__ == "__main__":
    main()
