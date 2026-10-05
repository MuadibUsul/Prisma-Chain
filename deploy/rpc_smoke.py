#!/usr/bin/env python3
"""Scripted load + abuse smoke for the public RPC front door (roadmap B1-02i).

Brings up a single-validator node from a generated netconfig bundle, fronts
its RPC with `prismad rpc-proxy`, then runs the normal and abuse cases and
records the observed results as JSON.

    python deploy/rpc_smoke.py [--out results.json] [--keep] \
        [--prismad BIN] [--netconfig BIN]

Without --prismad/--netconfig the script builds both binaries with `go
build` into a temporary directory (the Go toolchain is required). Findings
are recorded, never softened: any unexpected result fails the run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHAIN = REPO / "chain"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed: {proc.stderr.strip()}")
    return proc.stdout


def build_binaries(workdir: Path) -> tuple[Path, Path]:
    prismad = workdir / ("prismad.exe" if os.name == "nt" else "prismad")
    netconfig = workdir / ("netconfig.exe" if os.name == "nt" else "netconfig")
    env = dict(os.environ, GOTOOLCHAIN="local")
    run(["go", "build", "-o", str(prismad), "./cmd/prismad"], cwd=CHAIN, env=env)
    run(["go", "build", "-o", str(netconfig), "./tools/netconfig/"], cwd=CHAIN, env=env)
    return prismad, netconfig


def http(method: str, url: str, body: bytes | None = None, headers: dict | None = None,
         timeout: float = 10.0):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


class SlowUpstream(BaseHTTPRequestHandler):
    """A deliberately hanging upstream to prove the bounded timeout."""

    def do_GET(self):  # noqa: N802
        time.sleep(5)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):  # silence
        return


def wait_for(fn, timeout=90, interval=1.0, what="condition"):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            ok, last = fn()
            if ok:
                return
        except Exception as exc:  # noqa: BLE001 - report the last error
            last = str(exc)
        time.sleep(interval)
    raise RuntimeError(f"timed out waiting for {what}: last={last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs" / "productization" / "b1-02-rpc-smoke.json"))
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    ap.add_argument("--prismad", default=None)
    ap.add_argument("--netconfig", default=None)
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="rpc-smoke-"))
    procs: list[subprocess.Popen] = []
    cases: list[dict] = []
    slow_server: ThreadingHTTPServer | None = None

    def record(name: str, expected, observed, ok: bool, note: str = ""):
        cases.append({"case": name, "expected": expected, "observed": observed,
                      "pass": bool(ok), "note": note})
        mark = "PASS" if ok else "FAIL"
        print(f"  {mark}  {name}: expected={expected} observed={observed} {note}".rstrip())

    try:
        if args.prismad and args.netconfig:
            prismad, netconfig = Path(args.prismad), Path(args.netconfig)
        else:
            prismad, netconfig = build_binaries(work)

        rpc_port, proxy_port, slow_port = free_port(), free_port(), free_port()
        rpc = f"http://127.0.0.1:{rpc_port}"
        proxy = f"http://127.0.0.1:{proxy_port}"

        spec = {
            "chain_id": "prisma-rpcsmoke-1",
            "genesis_time": "2026-10-01T00:00:00Z",
            "denom": "uprsm",
            "validators": [{
                "moniker": "smoke",
                "seed_b64": base64.b64encode(hashlib.sha256(b"rpc-smoke").digest()).decode(),
                "p2p_external_address": "203.0.113.40:26656",
                "account_funding_uprsm": 100000000000,
                "self_delegation_uprsm": 1000000,
            }],
            "faucet": {"enabled": True, "amount_uprsm": 10000000, "cooldown_seconds": 3600,
                       "per_address_cap_uprsm": 100000000},
            "rpc": {"laddr": f"tcp://127.0.0.1:{rpc_port}", "grpc_laddr": f"tcp://127.0.0.1:{free_port()}"},
            "p2p": {"laddr": f"tcp://127.0.0.1:{free_port()}"},
            "prismad_binary": str(prismad),
        }
        spec_path = work / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        bundle, home = work / "bundle", work / "home"
        run([str(netconfig), "generate", "--spec", str(spec_path), "--out", str(bundle)])
        run([str(netconfig), "provision", "--spec", str(spec_path), "--bundle", str(bundle),
             "--moniker", "smoke", "--home", str(home)])

        procs.append(subprocess.Popen([str(prismad), "start", "--home", str(home),
                                       "--minimum-gas-prices", "0uprsm"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        wait_for(lambda: (http("GET", f"{rpc}/status")[0] == 200, "rpc up"), what="node rpc")

        procs.append(subprocess.Popen([str(prismad), "rpc-proxy", "--listen", f"127.0.0.1:{proxy_port}",
                                       "--upstream", rpc, "--rate", "5", "--burst", "10",
                                       "--max-body-bytes", "4096"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        wait_for(lambda: (http("GET", f"{proxy}/healthz")[0] == 200, "proxy up"), what="proxy health")

        print("abuse/load cases:")
        code, _ = http("GET", f"{proxy}/status")
        record("normal GET /status", 200, code, code == 200)

        code, _ = http("POST", f"{proxy}/", b'{"jsonrpc":"2.0","id":1,"method":"status"}',
                       {"Content-Type": "application/json"})
        record("allowed JSON-RPC method", 200, code, code == 200)

        code, body = http("POST", f"{proxy}/",
                          b'{"jsonrpc":"2.0","id":1,"method":"broadcast_tx_commit","params":{}}',
                          {"Content-Type": "application/json"})
        record("write method denied", 403, code, code == 403 and "broadcast_tx_commit" in body)

        direct, _ = http("GET", f"{rpc}/dump_consensus_state")
        code, _ = http("GET", f"{proxy}/dump_consensus_state")
        record("heavy path denied through proxy", 403, code, code == 403,
               note=f"(node itself serves it: {direct})")

        code, body = http("POST", f"{proxy}/",
                          json.dumps({"jsonrpc": "2.0", "id": 1, "method": "status",
                                      "params": {"pad": "x" * 8192}}).encode(),
                          {"Content-Type": "application/json"})
        record("oversize body rejected", 413, code, code == 413 and "cap" in body)

        code, _ = http("GET", f"{proxy}/websocket", headers={"Connection": "Upgrade",
                                                             "Upgrade": "websocket"})
        record("websocket refused", 501, code, code == 501)

        flood = [http("GET", f"{proxy}/status")[0] for _ in range(40)]
        limited = flood.count(429)
        record("flood rate-limited", ">=1 request answered 429",
               f"{limited}/40 with 429, {flood.count(200)} with 200", limited >= 1)

        code, body = http("GET", f"{proxy}/healthz")
        record("health endpoint", 200, code, code == 200 and '"status":"ok"' in body)

        code, body = http("GET", f"{proxy}/metrics")
        record("metrics exposed", "200 + counters", code,
               code == 200 and "prisma_rpc_proxy_rate_limited_total" in body)

        slow_server = ThreadingHTTPServer(("127.0.0.1", slow_port), SlowUpstream)
        threading.Thread(target=slow_server.serve_forever, daemon=True).start()
        timeout_proxy_port = free_port()
        procs.append(subprocess.Popen([str(prismad), "rpc-proxy", "--listen",
                                       f"127.0.0.1:{timeout_proxy_port}",
                                       "--upstream", f"http://127.0.0.1:{slow_port}",
                                       "--upstream-timeout", "1s"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        time.sleep(1.5)
        code, _ = http("GET", f"http://127.0.0.1:{timeout_proxy_port}/status", timeout=10)
        record("bounded upstream timeout", 504, code, code == 504,
               note="(upstream hangs 5s, proxy timeout 1s)")

        results = {
            "script": "deploy/rpc_smoke.py",
            "prismad": str(prismad),
            "chain_id": "prisma-rpcsmoke-1",
            "policy": {"rate_per_second": 5, "burst": 10, "max_body_bytes": 4096},
            "cases": cases,
            "passed": sum(1 for c in cases if c["pass"]),
            "failed": sum(1 for c in cases if not c["pass"]),
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(f"\nrpc smoke: {results['passed']} passed, {results['failed']} failed -> {out}")
        return 0 if results["failed"] == 0 else 1
    finally:
        for proc in procs:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if slow_server is not None:
            slow_server.shutdown()
        if args.keep:
            print(f"work directory kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
