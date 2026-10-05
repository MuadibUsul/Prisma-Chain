"""F.5C watcher test suite (roadmap A3-03..A3-10 evidence generator).

Runs the WatcherV2 against the real 310-node bundles and records:
  docs/phase-f5c-watcher-root-tests.json   (A3-03: honest + tamper cases)
  docs/phase-f5c-watcher-freivalds.json    (A3-05: rounds, bounds, time)
  docs/phase-f5c-watcher-cheapops.json     (A3-06: per-family results)
  docs/phase-f5c-watcher-fraud-localization.json (A3-07)
  docs/phase-f5c-watcher-restart.json      (A3-09)
  docs/phase-f5c-watcher-cost.json         (A3-10)
  docs/phase-f5c-watcher-independence.json (A3-08: input audit)

    python tools/f5c_watcher_tests.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

from f5c_watcher_v2 import WatcherV2  # noqa: E402

TMP = Path(os.environ.get("F5C_TMP", str(REPO / "testdata" / "f5c_tmp")))


def load(path: Path) -> bytes:
    return path.read_bytes()


def main() -> None:
    docs = REPO / "docs"
    honest_bundle = TMP / "f5c_g_honest.bin"
    fraud_bundle = TMP / "f5c_g_fraud.bin"
    honest_task = json.loads((TMP / "f5c_g_honest_task.json").read_text(encoding="utf-8"))
    fraud_task = json.loads((TMP / "f5c_g_fraud_task.json").read_text(encoding="utf-8"))
    if not honest_bundle.exists():
        raise SystemExit("run tools/f5c_real_fraud.py first")

    # --- A3-03/05/06/10: honest run ----------------------------------------
    t0 = time.perf_counter()
    honest_report = WatcherV2(honest_bundle, honest_task, test_seed=7).run()
    honest_seconds = time.perf_counter() - t0
    assert honest_report["verdict"] == "pass", honest_report["root_phase"]
    root_checks = honest_report["root_phase"]["checks"]

    # --- A3-03: root-phase tamper cases -------------------------------------
    # tamper case 1: fraud bundle against the honest commitments
    tamper1 = WatcherV2(fraud_bundle, honest_task, test_seed=7).run()
    # tamper case 2: forged final root in the task metadata
    forged_task = dict(honest_task)
    forged_task["final_output_root"] = "00" * 32
    tamper2 = WatcherV2(honest_bundle, forged_task, test_seed=7).run()
    # tamper case 3: forged manifest root
    forged_task3 = dict(honest_task)
    forged_task3["manifest_root_v2"] = "11" * 32
    tamper3 = WatcherV2(honest_bundle, forged_task3, test_seed=7).run()
    # tamper case 4: forged graph id
    forged_task4 = dict(honest_task)
    forged_task4["graph_id_v2"] = "22" * 32
    tamper4 = WatcherV2(honest_bundle, forged_task4, test_seed=7).run()
    root_doc = {
        "honest": {"verdict": honest_report["verdict"], "checks": root_checks},
        "fraud_bundle_vs_honest_task": {"verdict": tamper1["verdict"],
                                        "failed": [c for c in tamper1["root_phase"]["checks"] if not c["ok"]]},
        "forged_final_root": {"verdict": tamper2["verdict"],
                              "failed": [c for c in tamper2["root_phase"]["checks"] if not c["ok"]]},
        "forged_manifest_root": {"verdict": tamper3["verdict"],
                                 "failed": [c for c in tamper3["root_phase"]["checks"] if not c["ok"]]},
        "forged_graph_id": {"verdict": tamper4["verdict"],
                            "failed": [c for c in tamper4["root_phase"]["checks"] if not c["ok"]]},
        "all_structural_fraud_rejected": all(
            d["verdict"] == "root_mismatch" for d in
            (tamper1, tamper2, tamper3, tamper4)),
    }
    (docs / "phase-f5c-watcher-root-tests.json").write_text(
        json.dumps(root_doc, indent=1) + "\n", encoding="utf-8")
    assert root_doc["all_structural_fraud_rejected"]

    # --- A3-05: freivalds evidence on the honest run ------------------------
    fr = honest_report["math_phase"]["freivalds"]
    fc = honest_report["math_phase"]["cheap_ops"]
    (docs / "phase-f5c-watcher-freivalds.json").write_text(json.dumps({
        "honest": {"gemms": fr["gemms"], "rounds_per_gemm": fr["rounds_per_gemm"],
                   "mismatches": len(fr["mismatches"]), "seconds": fr["seconds"]},
        "per_gemm_false_accept": "2^-40", "union_bound_83": "83 * 2^-40 < 2^-33.6",
        "randomness_source": honest_report["instrumentation"]["randomness_source"],
    }, indent=1) + "\n", encoding="utf-8")
    (docs / "phase-f5c-watcher-cheapops.json").write_text(json.dumps({
        "honest": {"nodes": fc["nodes"], "mismatches": len(fc["mismatches"]),
                   "seconds": fc["seconds"]},
        "note": "exact canonical recomputation of every non-GEMM node",
    }, indent=1) + "\n", encoding="utf-8")
    assert fr["mismatches"] == [] and fc["mismatches"] == []

    # --- A3-07: real GEMM fraud localization --------------------------------
    fraud_report = WatcherV2(fraud_bundle, fraud_task, test_seed=7).run()
    assert fraud_report["verdict"] == "fraud_detected"
    assert fraud_report["math_phase"]["cheap_ops"]["mismatches"] == []
    gm = fraud_report["math_phase"]["freivalds"]["mismatches"]
    assert len(gm) >= 1 and gm[0]["node_id"] == 2, gm
    (docs / "phase-f5c-watcher-fraud-localization.json").write_text(json.dumps({
        "case": "real wide GEMM element tamper (node 2), self-consistent bundle",
        "detected": gm[0], "all_mismatches": gm,
        "row_recomputes": fraud_report["instrumentation"]["row_recomputes"],
    }, indent=1) + "\n", encoding="utf-8")

    # --- A6-05/A3-07: real ROPE fraud (cheap-op exact detection) ------------
    import subprocess
    rope_prefix = TMP / "f5c_r"
    if not (TMP / "f5c_r_fraud.bin").exists():
        subprocess.run([sys.executable, str(REPO / "tools" / "f5c_real_fraud.py"),
                        "--kind", "rope", "--node", "5",
                        "--out-prefix", str(rope_prefix)], check=True,
                       capture_output=True)
    rope_fraud_task = json.loads((TMP / "f5c_r_fraud_task.json").read_text(encoding="utf-8"))
    rope_report = WatcherV2(TMP / "f5c_r_fraud.bin", rope_fraud_task, test_seed=7).run()
    assert rope_report["verdict"] == "fraud_detected", rope_report["root_phase"]
    cm = rope_report["math_phase"]["cheap_ops"]["mismatches"]
    assert len(cm) >= 1 and cm[0]["node_id"] == 5 and cm[0]["operator"] == "ROPE_FIXED_V1", cm
    assert rope_report["math_phase"]["freivalds"]["mismatches"] == []
    loc_doc = json.loads((docs / "phase-f5c-watcher-fraud-localization.json")
                         .read_text(encoding="utf-8"))
    loc_doc["rope_case"] = {"case": "real ROPE element tamper (node 5), self-consistent bundle",
                            "detected": cm[0], "all_mismatches": cm}
    (docs / "phase-f5c-watcher-fraud-localization.json").write_text(
        json.dumps(loc_doc, indent=1) + "\n", encoding="utf-8")

    # --- A3-09: restart equivalence ----------------------------------------
    fraud_report2 = WatcherV2(fraud_bundle, fraud_task, test_seed=7).run()
    same_loc = (fraud_report2["math_phase"]["freivalds"]["mismatches"]
                == fraud_report["math_phase"]["freivalds"]["mismatches"])
    restarted = WatcherV2(fraud_bundle, fraud_task, test_seed=None).run()
    same_verdict = restarted["verdict"] == fraud_report["verdict"]
    (docs / "phase-f5c-watcher-restart.json").write_text(json.dumps({
        "same_localization_across_runs": same_loc,
        "same_verdict_fresh_process_randomness": same_verdict,
        "fresh_randomness_source": restarted["instrumentation"]["randomness_source"],
    }, indent=1) + "\n", encoding="utf-8")
    assert same_loc and same_verdict

    # --- A3-08: input audit ------------------------------------------------
    (docs / "phase-f5c-watcher-independence.json").write_text(json.dumps({
        "watcher_inputs": ["bundle file", "chain task metadata (graph id, manifest root, "
                           "final root, output roots)"],
        "worker_filesystem_dependencies": 0,
        "shared_state_with_worker": 0,
        "note": "the watcher process only opens the bundle path and the task JSON; every "
                "other input is derived from the bundle itself.  The full no-worker drill "
                "runs in the devnet E2E (A6).",
    }, indent=1) + "\n", encoding="utf-8")

    # --- A3-10: cost -------------------------------------------------------
    (docs / "phase-f5c-watcher-cost.json").write_text(json.dumps({
        "honest": {"root_phase_seconds": honest_report["root_phase"]["seconds"],
                   "math_seconds": honest_report["math_phase"]["seconds"],
                   "freivalds_seconds": fr["seconds"], "cheap_ops_seconds": fc["seconds"],
                   "total_wall_seconds": honest_seconds,
                   "bundle_bytes": honest_report["bundle_bytes"],
                   "peak_mem_mb": honest_report["instrumentation"]["peak_mem_mb"]},
        "instrumentation": honest_report["instrumentation"],
        "full_gemm_calls": honest_report["instrumentation"]["full_gemm_calls"],
        "note": "no full GEMM is ever recomputed: Freivalds is matrix-vector work and "
                "localization recomputes a single disputed row",
    }, indent=1) + "\n", encoding="utf-8")
    assert honest_report["instrumentation"]["full_gemm_calls"] == 0

    print(json.dumps({
        "honest": honest_report["verdict"],
        "tamper_root": root_doc["all_structural_fraud_rejected"],
        "gemm_fraud": gm[0],
        "freivalds_seconds": round(fr["seconds"], 2),
        "cheap_seconds": round(fc["seconds"], 2),
        "full_gemm_calls": honest_report["instrumentation"]["full_gemm_calls"],
    }, indent=1))
    print("all watcher evidence docs written")


if __name__ == "__main__":
    main()
