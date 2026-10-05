#!/usr/bin/env python3
"""Live join smoke for the worker (roadmap B2-03).

Brings up a real single-validator chain from a generated netconfig bundle,
funds a fresh worker account from the validator, runs `prisma-worker join`
(bond + network-key registration + state file), checks the chain, then
exercises the negative paths (wrong chain-id, unfunded account) and one
heartbeat tick. Results are recorded as JSON.

    python deploy/worker_join_smoke.py [--out results.json] [--keep]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
CHAIN = REPO / "chain"
sys.path.insert(0, str(REPO / "worker"))

PASSPHRASE = "worker-join-smoke"
BOND = 1_000_000
FUNDING = 3_000_000


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed: {proc.stderr.strip()}")
    return proc.stdout


def json_rpc(url: str, path: str, timeout: float = 10.0):
    try:
        with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read().decode())


def wait_for(fn, timeout=90, what="condition"):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            ok, last = fn()
            if ok:
                return last
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for {what}: last={last}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "docs" / "productization" / "b2-03-join-smoke.json"))
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--prismad", default=None)
    parser.add_argument("--netconfig", default=None)
    args = parser.parse_args()

    work = pathlib.Path(tempfile.mkdtemp(prefix="worker-join-smoke-"))
    procs: list[subprocess.Popen] = []
    cases: list[dict] = []

    def record(name, expected, observed, ok, note=""):
        cases.append({"case": name, "expected": expected, "observed": str(observed),
                      "pass": bool(ok), "note": note})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: expected={expected} observed={observed} {note}".rstrip())

    try:
        env = dict(os.environ, GOTOOLCHAIN="local")

        def build(name, package, path):
            out = work / name
            run(["go", "build", "-o", str(out), package], cwd=path, env=env)
            return out

        prismad = pathlib.Path(args.prismad) if args.prismad else build(
            "prismad.exe" if os.name == "nt" else "prismad", "./cmd/prismad", CHAIN)
        netconfig = pathlib.Path(args.netconfig) if args.netconfig else build(
            "netconfig.exe" if os.name == "nt" else "netconfig", "./tools/netconfig/", CHAIN)

        rpc_port = free_port()
        rpc = f"http://127.0.0.1:{rpc_port}"
        spec = {
            "chain_id": "prisma-joinsmoke-1",
            "genesis_time": "2026-10-01T00:00:00Z",
            "denom": "uprsm",
            "validators": [{
                "moniker": "joinsmoke",
                "seed_b64": base64.b64encode(hashlib.sha256(b"join-smoke").digest()).decode(),
                "p2p_external_address": "203.0.113.60:26656",
                "account_funding_uprsm": 100000000000, "self_delegation_uprsm": 1000000,
            }],
            "faucet": {"enabled": True, "amount_uprsm": 10000000, "cooldown_seconds": 3600,
                       "per_address_cap_uprsm": 100000000},
            "rpc": {"laddr": f"tcp://127.0.0.1:{rpc_port}", "grpc_laddr": f"tcp://127.0.0.1:{free_port()}"},
            "p2p": {"laddr": f"tcp://127.0.0.1:{free_port()}"},
            "prismad_binary": str(prismad),
        }
        spec_path = work / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        bundle, home = work / "bundle", work / "node"
        run([str(netconfig), "generate", "--spec", str(spec_path), "--out", str(bundle)])
        run([str(netconfig), "provision", "--spec", str(spec_path), "--bundle", str(bundle),
             "--moniker", "joinsmoke", "--home", str(home)])
        procs.append(subprocess.Popen([str(prismad), "start", "--home", str(home),
                                       "--minimum-gas-prices", "0uprsm"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        wait_for(lambda: (json_rpc(rpc, "/status").get("result", {}).get("sync_info", {})
                          .get("latest_block_height", "0") != "0", "node"), what="node up")

        from prisma_worker.cli import main as worker_main
        os.environ["PRISMA_WORKER_PASSPHRASE"] = PASSPHRASE

        keystore = work / "worker-key.json"
        state = work / "worker-state.json"
        assert worker_main(["identity", "init", "--keystore", str(keystore), "--json"]) == 0
        keystore_doc = json.loads(keystore.read_text(encoding="utf-8"))
        address = keystore_doc["public"]["account_address"]
        node_id = keystore_doc["public"]["node_id"]

        # Fund the worker account from the validator (test tokens, local chain).
        fund = prismad_tx(prismad, home, "bank", "send", "validator", address, f"{FUNDING}uprsm",
                          signer="validator", chain_id="prisma-joinsmoke-1", rpc=rpc)
        wait_for(lambda: (isinstance(query_worker(prismad, rpc, address), dict),
                          "worker query answered"), what="worker query")
        record("worker account funded", f">= {FUNDING}", fund, True)

        join_cmd = ["join", "--keystore", str(keystore), "--chain-id", "prisma-joinsmoke-1",
                    "--node", rpc, "--prismad", str(prismad), "--state", str(state),
                    "--allow-unsupported", "smoke on an SM75 dev host (not a proven target)",
                    "--json"]
        code = worker_main(join_cmd)
        record("join", "exit 0", code, code == 0)

        worker = query_worker(prismad, rpc, address)
        bonded = int(worker.get("bonded_uprsm", 0))
        bound_key = base64.b64decode(worker.get("network_public_key", "") or "")
        record("bonded on chain", f">= {BOND}", bonded, bonded >= BOND)
        record("protocol key registered on chain", "32-byte ed25519 key",
               f"{len(bound_key)} bytes, node_id={node_id}", len(bound_key) == 32)

        state_doc = json.loads(state.read_text(encoding="utf-8"))
        record("state file written", "chain/address/node_id/capability",
               sorted(state_doc), state_doc["account_address"] == address
               and state_doc["node_id"] == node_id and "capability" in state_doc)

        # Wrong chain-id must be refused.
        wrong = worker_main(["join", "--keystore", str(keystore), "--chain-id", "prisma-other-1",
                             "--node", rpc, "--prismad", str(prismad), "--state", str(work / "s2.json"),
                             "--allow-unsupported", "smoke"])
        record("wrong chain-id refused", "exit 1", wrong, wrong == 1)

        # An unfunded account must get an actionable error.
        poor_keystore = work / "poor-key.json"
        assert worker_main(["identity", "init", "--keystore", str(poor_keystore), "--json"]) == 0
        poor_state = work / "poor-state.json"
        poor = worker_main(["join", "--keystore", str(poor_keystore), "--chain-id", "prisma-joinsmoke-1",
                            "--node", rpc, "--prismad", str(prismad), "--state", str(poor_state),
                            "--allow-unsupported", "smoke"])
        record("unfunded account: actionable failure", "exit 1", poor, poor == 1)

        beats = worker_main(["heartbeat", "--keystore", str(keystore), "--chain-id", "prisma-joinsmoke-1",
                             "--node", rpc, "--prismad", str(prismad), "--state", str(state),
                             "--iterations", "1", "--json"])
        record("heartbeat tick", "exit 0", beats, beats == 0)

        results = {"script": "deploy/worker_join_smoke.py", "chain_id": "prisma-joinsmoke-1",
                   "cases": cases, "passed": sum(1 for c in cases if c["pass"]),
                   "failed": sum(1 for c in cases if not c["pass"])}
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(f"\njoin smoke: {results['passed']} passed, {results['failed']} failed -> {out}")
        return 0 if results["failed"] == 0 else 1
    finally:
        for proc in procs:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if args.keep:
            print(f"work directory kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


def prismad_tx(prismad, home, *args, signer, chain_id, rpc):
    out = run([str(prismad), "tx", *args, "--from", signer, "--chain-id", chain_id, "--node", rpc,
               "--keyring-backend", "test", "--keyring-dir", str(home), "--home", str(home),
               "--fees", "0uprsm", "--gas", "400000", "--yes", "--output", "json"])
    payload = json.loads(out)
    if int(payload.get("code", 1)) != 0:
        raise RuntimeError(f"tx rejected: {payload}")
    txhash = payload["txhash"]
    wait_for(lambda: (json_rpc(rpc, "/tx?hash=0x" + txhash).get("result") is not None, "tx"),
             what="tx inclusion")
    return txhash


def query_worker(prismad, rpc, address) -> dict:
    out = run([str(prismad), "query", "compute", "worker", "--worker", address, "--node", rpc, "--output", "json"])
    return json.loads(out)


if __name__ == "__main__":
    sys.exit(main())
