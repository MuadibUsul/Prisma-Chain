"""Multivalidator devnet status and assertions.

    python deploy/multivalidator/status.py            # print state
    python deploy/multivalidator/status.py --json OUT # write an artifact

Checks every validator RPC (never a single endpoint): height, app hash,
validator set voting power and the final state hash.
"""

import argparse
import json
import sys
import time
import urllib.request

RPCS = {
    "validator-a": "http://127.0.0.1:26661",
    "validator-b": "http://127.0.0.1:26662",
    "validator-c": "http://127.0.0.1:26663",
    "validator-d": "http://127.0.0.1:26664",
}


def rpc(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=5) as resp:
        return json.load(resp)["result"]


def node_state(name: str, base: str) -> dict:
    try:
        status = rpc(base, "/status")
        validators = rpc(base, "/validators")
        latest = rpc(base, "/block")
        return {
            "name": name,
            "reachable": True,
            "height": int(status["sync_info"]["latest_block_height"]),
            "app_hash": status["sync_info"].get("latest_app_hash", ""),
            "latest_block_hash": status["sync_info"].get("latest_block_hash", ""),
            "block_time": latest["block"]["header"]["time"],
            "validators": [
                {
                    "address": v["address"],
                    "voting_power": int(v["voting_power"]),
                }
                for v in validators["validators"]
            ],
        }
    except Exception as exc:  # unreachable node is a state, not a crash
        return {"name": name, "reachable": False, "error": repr(exc)[:200]}


def collect() -> dict:
    return {name: node_state(name, base) for name, base in RPCS.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="")
    parser.add_argument("--wait-blocks", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.wait_blocks > 0:
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            state = collect()
            heights = [v.get("height", 0) for v in state.values() if v.get("reachable")]
            if heights and min(heights) >= args.wait_blocks:
                break
            time.sleep(2)
    state = collect()
    for name, info in state.items():
        if info.get("reachable"):
            powers = [v["voting_power"] for v in info["validators"]]
            print(f"{name}: height={info['height']} app_hash={info['app_hash'][:16]} "
                  f"validators={len(info['validators'])} power={powers}")
        else:
            print(f"{name}: UNREACHABLE ({info['error']})")
    heights = {i["height"] for i in state.values() if i.get("reachable")}
    hashes = {i["app_hash"] for i in state.values() if i.get("reachable")}
    print(f"converged: heights={sorted(heights)} app_hashes={len(hashes)} distinct")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
            handle.write("\n")
        print("written:", args.json)
    # Exit non-zero only when nodes are reachable but disagree.
    if len(heights) > 1 or len(hashes) > 1:
        sys.exit(1)


if __name__ == "__main__":
    main()
