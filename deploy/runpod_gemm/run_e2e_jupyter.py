"""Two-pod GEMM E2E driver over Jupyter terminals.

RunPod GPU pods expose only the Jupyter notebook proxy (port 8888). Kernels
may fail to start on some templates, so this driver runs everything through
the Jupyter terminal API: each pod runs a stateful JSON-RPC daemon
(deploy/runpod_gemm/daemon.py) and the coordinator relays protocol messages
with small python one-liners, verifying every Merkle proof itself.

Usage:
  python run_e2e_jupyter.py --worker-url URL --challenger-url URL \
      [--m 256 --n 256 --k 256 --seed 42] [--fraud-tile 1,1] [--out FILE]
"""

import argparse
import base64
import json
import re
import sys
import time
import uuid
from pathlib import Path

import requests
from websocket import create_connection

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "compute" / "gemmv1" / "python"))

from gemmv1.protocol import leaf_trace_state  # noqa: E402
from gemmv1.merkle_proofs import verify_inclusion  # noqa: E402
from gemmv1.tensors import canonical_to_int32s, int32s_to_canonical  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")


class Terminal:
    def __init__(self, url: str):
        base, _, query = url.partition("?")
        self.base = base.rstrip("/")
        self.token = dict(p.split("=", 1) for p in query.split("&"))["token"]
        self.s = requests.Session()
        self.s.params = {"token": self.token}
        r = self.s.post(self.base + "/api/terminals", json={}, timeout=60)
        r.raise_for_status()
        self.name = r.json()["name"]
        ws_base = self.base.replace("https://", "wss://").replace("http://", "ws://")
        self.ws = create_connection(
            f"{ws_base}/terminals/websocket/{self.name}?token={self.token}", timeout=30
        )
        time.sleep(2)
        self._drain(2)  # swallow the banner

    def _drain(self, seconds: float) -> str:
        buf = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                self.ws.settimeout(max(0.05, deadline - time.time()))
                frame = json.loads(self.ws.recv())
                if frame[0] == "stdout":
                    buf.append(frame[1])
            except Exception:
                break
        return "".join(buf)

    def run(self, cmd: str, timeout: int = 900) -> str:
        marker = f"__P{uuid.uuid4().hex[:8]}__RC"
        # Break only on the RESULT line (RC=<digit>); the shell echo of the
        # command itself contains the marker with an unexpanded $?, so a
        # plain substring match would return prematurely.
        done = re.compile(marker + r"=\d")
        self.ws.send(json.dumps(["stdin", f"{cmd}; echo {marker}=$?\n"]))
        buf = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.ws.settimeout(max(0.1, deadline - time.time()))
                frame = json.loads(self.ws.recv())
                if frame[0] == "stdout":
                    buf.append(frame[1])
                    if done.search("".join(buf)):
                        break
            except Exception:
                break
        return ANSI.sub("", "".join(buf))

    def rpc(self, port: int, kind: str, body: dict) -> dict:
        payload = base64.b64encode(
            (json.dumps({**body, "_kind": kind}) + "\n").encode()
        ).decode()
        mid = uuid.uuid4().hex[:8]
        # Large requests (whole tile sets) go through a chunked file; a
        # single giant terminal command gets truncated by the tty layer.
        req_file = f"/tmp/req_{mid}.b64"
        self.run(f"rm -f {req_file}")
        for i in range(0, len(payload), 60000):
            self.run(f"printf '%s' '{payload[i:i+60000]}' >> {req_file}")
        cmd = (
            "python3 -c \"import socket,base64;"
            f"s=socket.create_connection(('127.0.0.1',{port}),timeout=880);"
            f"s.sendall(base64.b64decode(open('{req_file}','rb').read()));"
            "d=s.makefile('rb').readline();"
            f"print('PRISMA_JSON_{mid}:'+d.decode())\""
        )
        out = self.run(cmd)
        sentinel = f"PRISMA_JSON_{mid}:"
        idx = out.rfind(sentinel)
        if idx == -1:
            raise RuntimeError(f"no RPC response for {kind}: ...{out[-600:]}")
        resp, _ = json.JSONDecoder().raw_decode(out[idx + len(sentinel):])
        if isinstance(resp, dict) and "fatal" in resp:
            raise RuntimeError(f"daemon error on {kind}: {resp.get('traceback', resp['fatal'])[-1500:]}")
        return resp

    def upload(self, path: str, data: bytes, is_text: bool = False):
        """Upload through the terminal in base64 chunks; the contents API
        writes relative to the Jupyter server root, not /root, so we do not
        rely on it."""
        b64 = base64.b64encode(data).decode()
        tmp = f"/tmp/up_{uuid.uuid4().hex[:8]}.b64"
        self.run(f"rm -f {tmp} /{path} 2>/dev/null; true")
        for i in range(0, len(b64), 60000):
            self.run(f"printf '%s' '{b64[i:i+60000]}' >> {tmp}")
        out = self.run(
            f"mkdir -p $(dirname /{path}) && base64 -d {tmp} > /{path}"
            f" && ls -la /{path} && md5sum /{path}"
        )
        expect = __import__("hashlib").md5(data).hexdigest()
        if expect not in out:
            raise SystemExit(f"upload failed for /{path}: ...{out[-400:]}")

def deploy(node: Terminal, mode: str, tgz: bytes, daemon_py: str, port: int) -> dict:
    node.upload("root/prisma_e2e/gemmv1.tgz", tgz)
    node.upload("root/prisma_e2e/daemon.py", daemon_py.encode(), is_text=True)
    tag = uuid.uuid4().hex[:8]
    out = node.run(
        "mkdir -p /root/prisma && tar xzf /root/prisma_e2e/gemmv1.tgz -C /root/prisma"
        f" && python3 -c \"import sys; sys.path.insert(0,'/root/prisma'); import gemmv1; print('DEPLOY_OK_{tag}')\""
    )
    if f"DEPLOY_OK_{tag}" not in out:
        raise SystemExit(f"deploy failed on {mode}: ...{out[-600:]}")
    node.run("pkill -f 'daemon.py' >/dev/null 2>&1; true")
    node.run(
        f"cd /root/prisma_e2e && nohup python3 daemon.py --mode {mode} --port {port}"
        f" > /root/prisma_e2e/daemon.log 2>&1 & sleep 1; true"
    )
    out = node.run(
        "for i in $(seq 1 90); do grep -q DAEMON_READY /root/prisma_e2e/daemon.log 2>/dev/null && break; sleep 2; done;"
        " grep -q DAEMON_READY /root/prisma_e2e/daemon.log && echo DAEMON_UP"
    )
    if "DAEMON_UP" not in out:
        raise SystemExit(f"daemon not ready on {mode}: ...{out[-800:]}")
    pong = node.rpc(port, "ping", {})
    if not pong.get("ok"):
        raise SystemExit(f"daemon ping failed on {mode}")
    return {"deployed": True}


def verify_trace_lock(task_hex, assignment_hex, tile_i, tile_j, r_steps, trace, claimed_tile_hex):
    root = bytes.fromhex(trace["trace_root"])
    initial = bytes.fromhex(trace["states"][0])
    final = bytes.fromhex(trace["states"][-1])
    if any(initial):
        raise SystemExit("FAIL: S0 is not the zero matrix")
    if final.hex() != claimed_tile_hex:
        raise SystemExit("FAIL: locked trace S_R does not match the party's claimed tile")
    task, assignment = bytes.fromhex(task_hex), bytes.fromhex(assignment_hex)
    leaf0 = leaf_trace_state(task, assignment, tile_i, tile_j, 0, initial)
    leaff = leaf_trace_state(task, assignment, tile_i, tile_j, r_steps, final)
    p0, pf = trace["proofs"][0], trace["proofs"][-1]
    if not verify_inclusion(root, leaf0, p0["index"], p0["count"], [bytes.fromhex(s) for s in p0["siblings"]]):
        raise SystemExit("FAIL: invalid S0 inclusion proof")
    if not verify_inclusion(root, leaff, pf["index"], pf["count"], [bytes.fromhex(s) for s in pf["siblings"]]):
        raise SystemExit("FAIL: invalid S_R inclusion proof")
    return root, final


def run_scenario(worker: Terminal, challenger: Terminal, m, n, k, seed, fraud, label):
    report = {"scenario": label, "m": m, "n": n, "k": k, "seed": seed}
    t0 = time.time()

    w_body = {"m": m, "n": n, "k": k, "seed": seed}
    if fraud:
        w_body["inject_fraud"] = list(fraud)
    w = worker.rpc(8311, "execute", w_body)
    report["worker_backend"] = w["backend"]["backend"]
    report["worker_output_root"] = w["result_commit"]["output_root"]

    c = challenger.rpc(8311, "prepare", {
        "m": m, "n": n, "k": k, "seed": seed,
        "task_id": w["result_commit"]["task_id"],
        "assignment_id": w["result_commit"]["assignment_id"],
        "output_root": w["result_commit"]["output_root"],
        "worker_tiles": w["output_tiles"]["tiles"],
    })
    report["challenger_backend"] = c["backend"]["backend"]
    report["challenger_recompute_s"] = c["recompute_s"]
    if not c["root_ok"]:
        raise SystemExit("FAIL: submitted tiles do not hash to the committed output_root")
    if not c["disputed_tile"]:
        report["outcome"] = "optimistic_unchallenged"
        report["total_s"] = round(time.time() - t0, 1)
        print(json.dumps(report, indent=2))
        print(f"PASS [{label}]: outputs identical across two machines; outcome=optimistic_unchallenged")
        return report

    tile_i, tile_j = c["disputed_tile"]
    r_steps = (k + 7) // 8
    print(f"[{label}] fraud detected at tile ({tile_i},{tile_j}); opening dispute")

    wt = worker.rpc(8311, "trace", {"tile_i": tile_i, "tile_j": tile_j})
    worker_tile_hex = int32s_to_canonical(
        w["output_tiles"]["tiles"][tile_i * c["cols_c"] + tile_j]
    ).hex()
    worker_root, worker_final = verify_trace_lock(
        w["result_commit"]["task_id"], w["result_commit"]["assignment_id"],
        tile_i, tile_j, r_steps, wt, worker_tile_hex)

    ct = challenger.rpc(8311, "trace", {"tile_i": tile_i, "tile_j": tile_j})
    challenger_root, challenger_final = verify_trace_lock(
        w["result_commit"]["task_id"], w["result_commit"]["assignment_id"],
        tile_i, tile_j, r_steps, ct, c["disputed_challenger_tile"])

    low, high = 0, r_steps
    low_state = canonical_to_int32s(bytes.fromhex(wt["states"][0]))
    worker_high = canonical_to_int32s(worker_final)
    challenger_high = canonical_to_int32s(challenger_final)
    rounds = 0
    while high - low > 1:
        mid = low + (high - low) // 2
        wm = worker.rpc(8311, "mid", {"step": mid, "tile_i": tile_i, "tile_j": tile_j})
        cm = challenger.rpc(8311, "mid", {"step": mid})
        task, assignment = bytes.fromhex(w["result_commit"]["task_id"]), bytes.fromhex(w["result_commit"]["assignment_id"])
        w_state = bytes.fromhex(wm["state"])
        wp = wm["proof"]
        if not verify_inclusion(worker_root,
                                leaf_trace_state(task, assignment, tile_i, tile_j, mid, w_state),
                                wp["index"], wp["count"], [bytes.fromhex(s) for s in wp["siblings"]]):
            raise SystemExit("FAIL: worker midpoint proof rejected")
        c_state = bytes.fromhex(cm["state"])
        cp = cm["proof"]
        if not verify_inclusion(challenger_root,
                                leaf_trace_state(task, assignment, tile_i, tile_j, mid, c_state),
                                cp["index"], cp["count"], [bytes.fromhex(s) for s in cp["siblings"]]):
            raise SystemExit("FAIL: challenger midpoint proof rejected")
        w_vals, c_vals = canonical_to_int32s(w_state), canonical_to_int32s(c_state)
        if w_vals == c_vals:
            low, low_state = mid, w_vals
        else:
            high = mid
            worker_high, challenger_high = w_vals, c_vals
        rounds += 1
        if rounds > 64:
            raise SystemExit("FAIL: bisection did not converge")

    arb = challenger.rpc(8311, "arbitrate", {
        "step": low, "high": high, "tile_i": tile_i, "tile_j": tile_j,
        "low_state": int32s_to_canonical(low_state).hex(),
        "worker_high": int32s_to_canonical(worker_high).hex(),
    })
    report.update({
        "outcome": arb["outcome"],
        "disputed_tile": [tile_i, tile_j],
        "bisection_rounds": rounds,
        "arbitration_step": low,
        "arbitration_macs": arb["arbitration_macs"],
        "on_demand_trace_bytes": (r_steps + 1) * 256 * 2,
        "worker_trace_root": worker_root.hex(),
        "challenger_trace_root": challenger_root.hex(),
        "total_s": round(time.time() - t0, 1),
    })
    print(json.dumps(report, indent=2))
    if arb["outcome"] == "challenger_wins":
        print(f"PASS [{label}]: fraud confirmed on two machines; ChallengerWins; worker gets NO receipt")
    elif arb["outcome"] == "worker_wins":
        print(f"PASS [{label}]: false challenge rejected; WorkerWins")
    else:
        print(f"FAIL [{label}]: both_invalid")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-url", required=True)
    ap.add_argument("--challenger-url", required=True)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fraud-tile", default="1,1")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tgz_path = Path(__file__).with_name("gemmv1.tgz")
    if not tgz_path.exists():
        import tarfile

        src = REPO / "compute" / "gemmv1" / "python" / "gemmv1"
        with tarfile.open(tgz_path, "w:gz") as t:
            t.add(src, arcname="gemmv1")
    tgz = tgz_path.read_bytes()
    daemon_py = (Path(__file__).with_name("daemon.py")).read_text(encoding="utf-8")

    print("== opening terminals ==")
    worker = Terminal(args.worker_url)
    challenger = Terminal(args.challenger_url)
    print("== deploying to both pods ==")
    print("[worker]", deploy(worker, "worker", tgz, daemon_py, 8311))
    print("[challenger]", deploy(challenger, "challenger", tgz, daemon_py, 8311))

    print("== GPU bit-exactness gate (torch._int_mm vs exact int64 oracle) ==")
    checks = {}
    for name, node in (("worker", worker), ("challenger", challenger)):
        checks[name] = node.rpc(8311, "gpu_check", {"size": 512, "seed": 7})
        print(f"[{name}]", json.dumps(checks[name]))
    gpu_ok = all(v.get("bit_exact") for v in checks.values())
    print("GPU_BIT_EXACT:", gpu_ok)

    print("== honest E2E ==")
    honest = run_scenario(worker, challenger, args.m, args.n, args.k, args.seed, None, "honest")
    print("== fraud E2E ==")
    fi, fj = args.fraud_tile.split(",")
    fraud_report = run_scenario(worker, challenger, args.m, args.n, args.k, args.seed,
                                (int(fi), int(fj)), "fraud")

    summary = {
        "gpu_bit_exact": gpu_ok,
        "gpu_checks": checks,
        "honest": honest,
        "fraud": fraud_report,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("written:", args.out)
    ok = gpu_ok and honest["outcome"] == "optimistic_unchallenged" and fraud_report["outcome"] == "challenger_wins"
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
