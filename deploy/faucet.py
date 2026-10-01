"""Send a fixed 1 PRSM from the local dev validator's test keyring."""

import argparse
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CHAIN_ID = "prisma-local-1"
RPC = "http://127.0.0.1:26657"
COMPOSE = Path(__file__).resolve().parent / "compose.yaml"
AMOUNT = "1000000uprsm"


def rpc(path: str) -> dict:
    with urllib.request.urlopen(RPC + path, timeout=3) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", help="recipient prsm1... account address")
    args = parser.parse_args()
    if not re.fullmatch(r"prsm1[023456789acdefghjklmnpqrstuvwxyz]{38,64}", args.address):
        parser.error("expected a Prisma account address (prsm1...)")
    status = rpc("/status")
    if status["result"]["node_info"]["network"] != CHAIN_ID:
        raise SystemExit("Local RPC is not the prisma-local-1 devnet")
    command = ["docker", "compose", "-f", str(COMPOSE), "exec", "-T", "chain",
               "prismad", "tx", "bank", "send", "validator", args.address, AMOUNT,
               "--chain-id", CHAIN_ID, "--home", "/data", "--keyring-backend", "test",
               "--keyring-dir", "/data", "--fees", "0uprsm", "--yes", "--output", "json"]
    result = subprocess.run(command, cwd=COMPOSE.parent.parent, text=True, capture_output=True, timeout=20)
    if result.returncode:
        raise SystemExit(result.stderr.strip() or result.stdout.strip())
    tx = json.loads(result.stdout)
    if int(tx.get("code", 1)) != 0 or not tx.get("txhash"):
        raise SystemExit(f"Bank transfer rejected: {tx.get('raw_log', tx)}")
    txhash = tx["txhash"]
    for _ in range(20):
        try:
            included = rpc("/tx?" + urllib.parse.urlencode({"hash": "0x" + txhash}))
        except urllib.error.HTTPError as exc:
            if exc.code != 500:
                raise
        else:
            if "error" in included:
                time.sleep(1)
                continue
            if int(included["result"]["tx_result"]["code"]) != 0:
                raise SystemExit("Bank transfer failed in block: " + txhash)
            print(f"Sent {AMOUNT} to {args.address}; tx {txhash}")
            return
        time.sleep(1)
    raise SystemExit("Broadcast accepted; inclusion was not confirmed within 20 seconds: " + txhash)


if __name__ == "__main__":
    main()
