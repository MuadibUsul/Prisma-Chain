#!/usr/bin/env python3
"""Live DA smoke (B4-01 x B2-08): three real prisma-da processes, one worker client.

Starts three daemon processes with their own data directories and keys, then
drives them with the worker's DA client (`prisma_worker.da.ensure_quorum`):
upload the blob, collect attestations, verify the signatures with the frozen
encoder, and reach the 2-of-3 quorum — including the two offline drills.

    python deploy/da_smoke.py [--out results.json] [--keep]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
for extra in ("da", "worker", "network", "compute/canonical/python", "compute/gemmv1/python"):
    sys.path.insert(0, str(REPO / extra))

from prisma_da.frozen import frozen  # noqa: E402
from prisma_worker.da import DAProvider, QuorumNotReached, ensure_quorum  # noqa: E402

M, N = 8, 8
TASK_ID = 411
TASK_ID32 = "aa" * 32
ASSIGNMENT = "bb" * 32


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "docs" / "productization" / "b4-01-da-smoke.json"))
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    work = pathlib.Path(tempfile.mkdtemp(prefix="da-smoke-"))
    procs: list[subprocess.Popen] = []
    cases: list[dict] = []

    def record(name, expected, observed, ok, note=""):
        cases.append({"case": name, "expected": str(expected), "observed": str(observed),
                      "pass": bool(ok), "note": note})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: expected={expected} observed={observed} {note}".rstrip())

    try:
        values = list(range(1, M * N + 1))
        blob = b"".join(int(v).to_bytes(4, "big", signed=True) for v in values)
        tiles = frozen.tensors.output_tiles(values, M, N)
        cols_c = (N + 7) // 8
        leaves = [frozen.protocol.leaf_output_tile(bytes.fromhex(TASK_ID32), bytes.fromhex(ASSIGNMENT),
                                                   i // cols_c, i % cols_c,
                                                   frozen.tensors.int32s_to_canonical(tile))
                  for i, tile in enumerate(tiles)]
        output_root = frozen.protocol.merkle_root(leaves).hex()

        env = dict(os.environ, PRISMA_DA_PASSPHRASE="da-smoke", GOTOOLCHAIN="local")
        providers = []
        for index, name in enumerate(("a", "b", "c")):
            home = work / f"da-{name}"
            port = free_port()
            init = subprocess.run([sys.executable, "-m", "prisma_da", "init",
                                   "--config", str(home / "config.json"), "--chain-id", "prisma-dasmoke-1",
                                   "--node", "http://127.0.0.1:1", "--listen", f"127.0.0.1:{port}",
                                   "--data-dir", str(home / "data"), "--json"],
                                  cwd=REPO / "da", capture_output=True, text=True, env=env)
            if init.returncode != 0:
                raise RuntimeError(f"da init failed: {init.stderr}")
            keys = json.loads(init.stdout)
            proc = subprocess.Popen([sys.executable, "-m", "prisma_da", "run",
                                     "--config", str(home / "config.json")],
                                    cwd=REPO / "da", env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            procs.append(proc)
            providers.append(DAProvider(url=f"http://127.0.0.1:{port}", name=name))
            _wait_health(f"http://127.0.0.1:{port}/health")
        record("three daemons serve /health", "3 healthy",
               f"{len(providers)} providers", len(providers) == 3)

        try:
            result = ensure_quorum(providers, quorum=3, bundle_hex=blob.hex(),
                                   output_root_hex=output_root, m=M, n=N, k=1, task_id=TASK_ID,
                                   assignment_id_hex=ASSIGNMENT, task_id32_hex=TASK_ID32,
                                   available_until=9000, attested_height=100, log=lambda _m: None)
        except QuorumNotReached as exc:
            print("outcomes:", json.dumps(exc.outcomes, indent=1))
            raise
        record("upload + verified attestations", "3 verified",
               len(result["verified"]), len(result["verified"]) == 3)

        for provider in providers:
            with urllib.request.urlopen(provider.url + "/v1/da/%d/output" % TASK_ID, timeout=10) as response:
                served = json.loads(response.read().decode())
            if bytes.fromhex(served["c_hex"]) != blob:
                record(f"retrieval from {provider.name}", "identical bytes", "mismatch", False)
                break
        else:
            record("retrieval from all three", "identical bytes", "identical", True)

        # Drill: one provider offline -> quorum still reached.
        providers[2].url = "http://127.0.0.1:1"
        result = ensure_quorum(providers, quorum=2, bundle_hex=blob.hex(),
                               output_root_hex=output_root, m=M, n=N, k=1, task_id=TASK_ID,
                               assignment_id_hex=ASSIGNMENT, task_id32_hex=TASK_ID32,
                               available_until=9000, attested_height=100, retries=1,
                               backoff_seconds=0, log=lambda _m: None)
        record("1/3 offline drill", "quorum 2 reached", len(result["verified"]),
               len(result["verified"]) == 2)

        # Drill: two providers offline -> no quorum, explicit failure.
        providers[1].url = "http://127.0.0.1:2"
        try:
            ensure_quorum(providers, quorum=2, bundle_hex=blob.hex(), output_root_hex=output_root,
                          m=M, n=N, k=1, task_id=TASK_ID, assignment_id_hex=ASSIGNMENT,
                          task_id32_hex=TASK_ID32, available_until=9000, attested_height=100,
                          retries=0, backoff_seconds=0, log=lambda _m: None)
            record("2/3 offline drill", "QuorumNotReached", "no error", False)
        except QuorumNotReached as exc:
            record("2/3 offline drill", "QuorumNotReached", "raised", True, note=str(exc)[:60])

        results = {"script": "deploy/da_smoke.py", "cases": cases,
                   "passed": sum(1 for c in cases if c["pass"]),
                   "failed": sum(1 for c in cases if not c["pass"])}
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(f"\nda smoke: {results['passed']} passed, {results['failed']} failed -> {out}")
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


def _wait_health(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError("daemon at %s never became healthy" % url)


if __name__ == "__main__":
    sys.exit(main())
