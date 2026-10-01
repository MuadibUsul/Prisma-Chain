"""Local-only acceptance: blocks, balances, signed bank transfer, balance delta."""

import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CHAIN_ID = "prisma-local-1"
RPC = "http://127.0.0.1:26657"
COMPOSE = Path(__file__).resolve().parent / "compose.yaml"
SEND = 1_000_000  # One test PRSM, in uprsm.
FLAGS = ["--home", "/data", "--keyring-backend", "test", "--keyring-dir", "/data"]


def rpc(path: str) -> dict:
    with urllib.request.urlopen(RPC + path, timeout=3) as response:
        return json.load(response)


def cli(*args: str) -> str:
    command = ["docker", "compose", "-f", str(COMPOSE), "exec", "-T", "chain", "prismad", *args]
    result = subprocess.run(command, cwd=COMPOSE.parent.parent, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def height() -> int:
    status = rpc("/status")["result"]
    if status["node_info"]["network"] != CHAIN_ID:
        raise RuntimeError("loopback RPC is not prisma-local-1")
    return int(status["sync_info"]["latest_block_height"])


def wait_for(check, description: str, seconds: int = 30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except (OSError, KeyError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError(f"timed out waiting for {description}")


def balance(address: str) -> int:
    data = json.loads(cli("query", "bank", "balances", address, "--home", "/data", "--output", "json"))
    return sum(int(coin["amount"]) for coin in data["balances"] if coin["denom"] == "uprsm")


def main() -> None:
    first = wait_for(lambda: (h if (h := height()) >= 1 else None), "first block")
    second = wait_for(lambda: (h if (h := height()) > first else None), "next block")
    sender = cli("keys", "show", "validator", "-a", *FLAGS)
    try:
        recipient = cli("keys", "show", "chain-smoke-recipient", "-a", *FLAGS)
    except RuntimeError:
        cli("keys", "add", "chain-smoke-recipient", *FLAGS, "--no-backup")
        recipient = cli("keys", "show", "chain-smoke-recipient", "-a", *FLAGS)
    before_sender, before_recipient = balance(sender), balance(recipient)
    if before_sender < SEND:
        raise RuntimeError("validator test balance is below one PRSM")
    broadcast = json.loads(cli("tx", "bank", "send", "validator", recipient, f"{SEND}uprsm",
                               "--chain-id", CHAIN_ID, *FLAGS, "--fees", "0uprsm", "--yes", "--output", "json"))
    if int(broadcast.get("code", 1)) != 0 or not broadcast.get("txhash"):
        raise RuntimeError(f"bank transfer rejected: {broadcast}")
    txhash = broadcast["txhash"]

    def included():
        result = rpc("/tx?" + urllib.parse.urlencode({"hash": "0x" + txhash}))
        if "error" in result:
            return None
        return result["result"]

    tx = wait_for(included, "bank transaction inclusion")
    if int(tx["tx_result"]["code"]) != 0:
        raise RuntimeError(f"bank transfer failed in block: {txhash}")
    after_sender, after_recipient = balance(sender), balance(recipient)
    if after_sender != before_sender - SEND or after_recipient != before_recipient + SEND:
        raise RuntimeError("bank balances did not change by exactly one PRSM")
    print(f"PASS: {CHAIN_ID} blocks {first}->{second}, signed bank tx {txhash}, "
          f"validator -{SEND}uprsm, recipient +{SEND}uprsm")


if __name__ == "__main__":
    main()
