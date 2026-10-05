#!/usr/bin/env python3
"""Live job-accept smoke for the worker (roadmap B2-04).

Real single-validator chain -> worker joins -> a frozen-profile task is
posted and accepted through the worker CLI (journaled), then the negative
paths are checked: an incompatible task is refused and a duplicate accept
submits nothing.

    python deploy/worker_accept_smoke.py [--out results.json] [--keep]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "worker"))

from worker_join_smoke import (  # noqa: E402  (shared chain bring-up helpers)
    CHAIN, REPO, free_port, json_rpc, query_worker, run, wait_for,
)

PASSPHRASE = "worker-accept-smoke"
FUNDING = 3_000_000
MODEL_ID = "qwen3-0.6b-layer0-v2"
CHAIN_ID = "prisma-acceptsmoke-1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "docs" / "productization"
                                            / "b2-04-accept-smoke.json"))
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--prismad", default=None)
    parser.add_argument("--netconfig", default=None)
    parser.add_argument("--debug-tasks", action="store_true")
    args = parser.parse_args()

    work = pathlib.Path(tempfile.mkdtemp(prefix="worker-accept-smoke-"))
    procs: list[subprocess.Popen] = []
    cases: list[dict] = []

    def record(name, expected, observed, ok, note=""):
        cases.append({"case": name, "expected": str(expected), "observed": str(observed),
                      "pass": bool(ok), "note": note})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: expected={expected} observed={observed} {note}".rstrip())

    try:
        env = dict(os.environ, GOTOOLCHAIN="local")
        prismad = pathlib.Path(args.prismad) if args.prismad else _build(work, "prismad",
                                                                        "./cmd/prismad", env)
        netconfig = pathlib.Path(args.netconfig) if args.netconfig else _build(
            work, "netconfig", "./tools/netconfig/", env)

        rpc_port = free_port()
        rpc = f"http://127.0.0.1:{rpc_port}"
        spec = {
            "chain_id": CHAIN_ID, "genesis_time": "2026-10-01T00:00:00Z", "denom": "uprsm",
            "validators": [{"moniker": "accept", "seed_b64": _seed("accept"),
                            "p2p_external_address": "203.0.113.80:26656",
                            "account_funding_uprsm": 100000000000, "self_delegation_uprsm": 1000000}],
            "faucet": {"enabled": True, "amount_uprsm": 1, "cooldown_seconds": 1,
                       "per_address_cap_uprsm": 1},
            "rpc": {"laddr": f"tcp://127.0.0.1:{rpc_port}", "grpc_laddr": f"tcp://127.0.0.1:{free_port()}"},
            "p2p": {"laddr": f"tcp://127.0.0.1:{free_port()}"},
            "prismad_binary": str(prismad),
        }
        spec_path = work / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        bundle, home = work / "bundle", work / "node"
        run([str(netconfig), "generate", "--spec", str(spec_path), "--out", str(bundle)])
        run([str(netconfig), "provision", "--spec", str(spec_path), "--bundle", str(bundle),
             "--moniker", "accept", "--home", str(home)])
        procs.append(subprocess.Popen([str(prismad), "start", "--home", str(home),
                                       "--minimum-gas-prices", "0uprsm"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        wait_for(lambda: (json_rpc(rpc, "/status").get("result", {}).get("sync_info", {})
                          .get("latest_block_height", "0") != "0", "node"), what="node up")

        from prisma_worker.cli import main as worker_main

        os.environ["PRISMA_WORKER_PASSPHRASE"] = PASSPHRASE
        keystore, state = work / "worker-key.json", work / "worker-state.json"
        journal_dir = work / "journal"
        assert worker_main(["identity", "init", "--keystore", str(keystore), "--json"]) == 0
        address = json.loads(keystore.read_text(encoding="utf-8"))["public"]["account_address"]

        # Fund + join the worker (reuses the B2-03 path).
        tx(prismad, home, "bank", "send", "validator", address, f"{FUNDING}uprsm",
           signer="validator", rpc=rpc)
        joined = worker_main(["join", "--keystore", str(keystore), "--chain-id", CHAIN_ID,
                              "--node", rpc, "--prismad", str(prismad), "--state", str(state),
                              "--allow-unsupported", "accept smoke on an SM75 host"])
        record("worker joined", "exit 0", joined, joined == 0)

        # Register the frozen profile as a model and post a compatible task.
        height = int(json_rpc(rpc, "/status")["result"]["sync_info"]["latest_block_height"])
        tx(prismad, home, "compute", "register-model", "--model-id", MODEL_ID,
           "--spec-version", "v1", "--mode", "lightweight",
           "--image-digest", "b" * 64, "--tokenizer-digest", "c" * 64, "--weights-digest", "a" * 64,
           signer="validator", rpc=rpc)
        tx(prismad, home, "compute", "post-task", "--mode", "lightweight", "--model-id", MODEL_ID,
           "--spec-version", "v1", "--input-commitment", hashlib.sha256(b"smoke-input").hexdigest(),
           "--data-ref", "encrypted:accept-smoke", "--max-fee", "1000000",
           "--deadline", str(height + 1000), "--privacy-tier", "tier0_relative",
           signer="validator", rpc=rpc)
        if args.debug_tasks:
            for probe_id in range(1, 4):
                try:
                    raw = run([str(prismad), "query", "compute", "task", "--task-id", str(probe_id),
                               "--node", rpc, "--output", "json"])
                    print(f"--- raw task {probe_id}: {raw[:400]}")
                    print(f"--- decoded: {json.dumps(decode_task(json.loads(raw)))[:400]}")
                except RuntimeError as exc:
                    print(f"--- task {probe_id}: {str(exc)[-160:]}")
            return 0
        task_id = find_task(prismad, rpc, MODEL_ID)
        record("compatible task posted", "task id", task_id, isinstance(task_id, int))

        # Discovery must mark it compatible.
        discovered = worker_main(["jobs", "discover", "--chain-id", CHAIN_ID, "--node", rpc,
                                  "--prismad", str(prismad), "--mode", "lightweight", "--json"])
        record("discover exit", "exit 0", discovered, discovered == 0)

        accepted = worker_main(["jobs", "accept", "--task-id", str(task_id), "--chain-id", CHAIN_ID,
                                "--node", rpc, "--prismad", str(prismad), "--keystore", str(keystore),
                                "--journal", str(journal_dir), "--mode", "lightweight", "--json"])
        record("accept exit", "exit 0", accepted, accepted == 0)
        task = decode_task(json.loads(run([str(prismad), "query", "compute", "task",
                                           "--task-id", str(task_id), "--node", rpc,
                                           "--output", "json"])))
        record("chain shows our worker as assignee", address, task.get("worker"), task.get("worker") == address)

        entry_path = journal_dir / f"task-{task_id:08d}.json"
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
        record("journaled accepted before side effects", "phase accepted + txhash",
               entry["phase"], entry["phase"] == "accepted" and entry.get("txhash"))
        history_len = len(entry["history"])

        duplicate = worker_main(["jobs", "accept", "--task-id", str(task_id), "--chain-id", CHAIN_ID,
                                 "--node", rpc, "--prismad", str(prismad), "--keystore", str(keystore),
                                 "--journal", str(journal_dir), "--mode", "lightweight", "--json"])
        entry_after = json.loads(entry_path.read_text(encoding="utf-8"))
        record("duplicate accept submits nothing", f"history stays {history_len}",
               len(entry_after["history"]),
               duplicate == 0 and len(entry_after["history"]) == history_len)

        # An incompatible task must be refused by accept.
        tx(prismad, home, "compute", "register-model", "--model-id", "someone-elses-model",
           "--spec-version", "v1", "--mode", "lightweight",
           "--image-digest", "b" * 64, "--tokenizer-digest", "c" * 64, "--weights-digest", "a" * 64,
           signer="validator", rpc=rpc)
        tx(prismad, home, "compute", "post-task", "--mode", "lightweight",
           "--model-id", "someone-elses-model", "--spec-version", "v1",
           "--input-commitment", hashlib.sha256(b"other-input").hexdigest(),
           "--data-ref", "encrypted:accept-smoke-2", "--max-fee", "1000000",
           "--deadline", str(height + 1000), "--privacy-tier", "tier0_relative",
           signer="validator", rpc=rpc)
        other_id = find_task(prismad, rpc, "someone-elses-model")
        refused = worker_main(["jobs", "accept", "--task-id", str(other_id), "--chain-id", CHAIN_ID,
                               "--node", rpc, "--prismad", str(prismad), "--keystore", str(keystore),
                               "--journal", str(journal_dir), "--mode", "lightweight"])
        other_task = decode_task(json.loads(run([str(prismad), "query", "compute", "task",
                                                 "--task-id", str(other_id), "--node", rpc,
                                                 "--output", "json"])))
        record("incompatible task refused", "exit 1 + still unassigned",
               f"exit={refused} worker={other_task.get('worker', '')!r}",
               refused == 1 and not other_task.get("worker"))

        results = {"script": "deploy/worker_accept_smoke.py", "chain_id": CHAIN_ID,
                   "cases": cases, "passed": sum(1 for c in cases if c["pass"]),
                   "failed": sum(1 for c in cases if not c["pass"])}
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(f"\naccept smoke: {results['passed']} passed, {results['failed']} failed -> {out}")
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


def _build(work: pathlib.Path, name: str, package: str, env) -> pathlib.Path:
    out = work / (name + (".exe" if os.name == "nt" else ""))
    run(["go", "build", "-o", str(out), package], cwd=CHAIN, env=env)
    return out


def _seed(tag: str) -> str:
    import base64

    return base64.b64encode(hashlib.sha256(tag.encode()).digest()).decode()


def tx(prismad, home, *args, signer, rpc):
    out = run([str(prismad), "tx", *args, "--from", signer, "--chain-id", CHAIN_ID, "--node", rpc,
               "--keyring-backend", "test", "--keyring-dir", str(home), "--home", str(home),
               "--fees", "0uprsm", "--gas", "2000000", "--yes", "--output", "json"])
    payload = json.loads(out)
    if int(payload.get("code", 1)) != 0:
        raise RuntimeError(f"tx rejected: {payload}")
    txhash = payload["txhash"]

    def included():
        result = json_rpc(rpc, "/tx?hash=0x" + txhash).get("result")
        return result is not None, result

    result = wait_for(included, what="tx inclusion")
    code = int(result.get("tx_result", {}).get("code", 1))
    if code != 0:
        raise RuntimeError(f"DeliverTx failed (code {code}): "
                           f"{result.get('tx_result', {}).get('log', '')[:400]}")
    return txhash


def decode_task(payload: dict) -> dict:
    """QueryTaskResponse carries bytes task_json (base64), not a flat object."""
    import base64

    blob = payload.get("task_json")
    return json.loads(base64.b64decode(blob)) if blob else payload


def find_task(prismad, rpc, model_id: str) -> int:
    for task_id in range(1, 64):
        try:
            raw = json.loads(run([str(prismad), "query", "compute", "task",
                                  "--task-id", str(task_id), "--node", rpc, "--output", "json"]))
        except RuntimeError as exc:
            if "not found" in str(exc):
                continue  # the frozen query errors for missing ids
            raise
        payload = decode_task(raw)
        if payload.get("model_id") == model_id and payload.get("status") == "posted":
            return task_id
    raise RuntimeError(f"no open task for model {model_id}")


if __name__ == "__main__":
    sys.exit(main())
