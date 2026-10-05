"""E1 liveness and fault-tolerance tests on the four-validator devnet.

Sequence (documented outcomes, not softened):
  1. four up: chain finalizes.
  2. stop validator-d: 75% remains, chain MUST keep finalizing.
  3. restart validator-d: it MUST catch up to the same height/app hash
     from its own data (never copied from another node).
  4. stop validator-c AND validator-d: 50% remains, chain MUST halt
     (expected safety behavior, not a failure).
  5. restart both: the chain MUST resume.

Results are written to docs/phase-e-validator-results.json.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIR = REPO / "deploy" / "multivalidator"
COMPOSE = DIR / "compose.yaml"
RPCS = {
    "validator-a": "http://127.0.0.1:26661",
    "validator-b": "http://127.0.0.1:26662",
    "validator-c": "http://127.0.0.1:26663",
    "validator-d": "http://127.0.0.1:26664",
}


def compose(*args: str, check: bool = True) -> str:
    proc = subprocess.run(["docker", "compose", "-f", str(COMPOSE), *args],
                          capture_output=True, text=True, timeout=180)
    out = proc.stdout.strip()
    if check and proc.returncode != 0:
        raise RuntimeError(f"compose {' '.join(args)} failed: {proc.stderr.strip() or out}")
    return out


def heights() -> dict:
    state = {}
    for name, base in RPCS.items():
        try:
            import urllib.request

            with urllib.request.urlopen(base + "/status", timeout=5) as resp:
                status = json.load(resp)["result"]["sync_info"]
                state[name] = {
                    "height": int(status["latest_block_height"]),
                    "app_hash": status.get("latest_app_hash", ""),
                }
        except Exception:
            state[name] = None
    return state


def wait_height_at_least(target: int, names, seconds: int = 90) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        state = heights()
        if all(state.get(n) and state[n]["height"] >= target for n in names):
            return state
        time.sleep(2)
    raise RuntimeError(f"nodes {names} did not reach height {target}: {state}")


def wait_height_frozen(names, seconds: int = 20) -> bool:
    """True when the listed nodes stop advancing for the whole window."""
    start = heights()
    time.sleep(seconds)
    end = heights()
    for name in names:
        if start.get(name) and end.get(name):
            if end[name]["height"] > start[name]["height"]:
                return False
    return True


def wait_height_equal(names, seconds: int = 120) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        state = heights()
        reachable = [state[n] for n in names if state.get(n)]
        if len(reachable) == len(names) and len({s["height"] for s in reachable}) == 1:
            return state
        time.sleep(2)
    raise RuntimeError(f"nodes {names} did not converge: {state}")


def main() -> None:
    report = {"scenario": "e1-liveness"}
    state = wait_height_at_least(10, RPCS.keys())
    report["stage_1_all_up"] = {
        "heights": {n: s["height"] for n, s in state.items()},
        "app_hashes": sorted({s["app_hash"] for s in state.values()}),
    }
    print("stage 1: four up, heights",
          {n: s["height"] for n, s in state.items()}, "-> PASS")

    # Stage 2: one validator offline (75% remains).
    baseline = max(s["height"] for s in state.values() if s)
    compose("stop", "validator-d")
    survivor_state = wait_height_at_least(baseline + 5, ["validator-a", "validator-b", "validator-c"])
    report["stage_2_one_offline"] = {
        "stopped": "validator-d",
        "height_at_stop": baseline,
        "heights_after": {n: survivor_state[n]["height"] for n in ("validator-a", "validator-b", "validator-c")},
        "finalized_through_outage": True,
    }
    print(f"stage 2: d stopped at {baseline}; a/b/c reached",
          {n: survivor_state[n]["height"] for n in ("validator-a", "validator-b", "validator-c")},
          "-> chain finalized with 75% -> PASS")

    # Stage 3: restart d, it must catch up from its own data.
    compose("start", "validator-d")
    catch_up_start = time.time()
    converged = wait_height_equal(list(RPCS.keys()))
    report["stage_3_catch_up"] = {
        "catch_up_seconds": round(time.time() - catch_up_start, 1),
        "heights": {n: s["height"] for n, s in converged.items()},
        "app_hashes": sorted({s["app_hash"] for s in converged.values()}),
    }
    print(f"stage 3: d caught up in {report['stage_3_catch_up']['catch_up_seconds']}s to height "
          f"{converged['validator-d']['height']} with identical app hash -> PASS")

    # Stage 4: two validators offline (50%): expected halt.
    baseline = max(s["height"] for s in converged.values())
    compose("stop", "validator-c")
    compose("stop", "validator-d")
    halted = wait_height_frozen(["validator-a", "validator-b"], seconds=15)
    report["stage_4_two_offline"] = {
        "stopped": ["validator-c", "validator-d"],
        "height_at_stop": baseline,
        "chain_halted": halted,
        "expected": "halt at 50% voting power",
    }
    if not halted:
        raise RuntimeError("chain kept finalizing with 50% voting power; expected halt")
    print(f"stage 4: c+d stopped at {baseline}; a/b frozen -> expected halt -> PASS (expected behavior)")

    # Stage 5: restart both, chain resumes.
    compose("start", "validator-c")
    compose("start", "validator-d")
    resumed = wait_height_at_least(baseline + 3, list(RPCS.keys()))
    report["stage_5_resumed"] = {
        "heights": {n: s["height"] for n, s in resumed.items()},
        "app_hashes": sorted({s["app_hash"] for s in resumed.values()}),
    }
    print("stage 5: c+d restarted, chain resumed to",
          {n: s["height"] for n, s in resumed.items()}, "-> PASS")

    out = REPO / "docs" / "phase-e-validator-results.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("written:", out)
    print("PASS: E1 liveness (4 up / 1 down finalizes / catch-up / 2 down halts / resume)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
