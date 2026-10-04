"""F.5B GPU runner driver over Jupyter terminals (two pods, in parallel).

Each pod: capture GPU/环境 → clone the F.5B branch (public repo) → run the
exact-wide-integer harness on CUDA → download the result JSON. The driver
never prints or stores pod tokens; URLs come from the command line.

Usage:
  python run_f5b_jupyter.py --pod-a URL --pod-b URL [--branch B] [--outdir DIR]
"""

import argparse
import json
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from websocket import create_connection

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")
PRINT_LOCK = threading.Lock()
BRANCH = "research/transformer-phase-f5b-gpu-feasibility"
REPO_URL = "https://github.com/MuadibUsul/Prisma-Chain.git"


def say(*parts):
    with PRINT_LOCK:
        print(*parts, flush=True)


class Terminal:
    def __init__(self, url: str, name: str):
        self.name = name
        base, _, query = url.partition("?")
        self.base = base.rstrip("/")
        token = dict(p.split("=", 1) for p in query.split("&"))["token"]
        self.s = requests.Session()
        self.s.params = {"token": token}
        r = self.s.post(self.base + "/api/terminals", json={}, timeout=60)
        r.raise_for_status()
        self.term = r.json()["name"]
        ws_base = self.base.replace("https://", "wss://").replace("http://", "ws://")
        self.ws = create_connection(
            f"{ws_base}/terminals/websocket/{self.term}?token={token}", timeout=30
        )
        time.sleep(2)
        self._drain(2)  # swallow banner

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
        text = ANSI.sub("", "".join(buf))
        rc = re.search(marker + r"=(\d)", text)
        self.last_rc = int(rc.group(1)) if rc else None
        return text

    def download_text(self, remote: str, timeout: int = 300) -> str:
        """Byte-exact pull: base64 in fixed-size chunks with markers, verified
        against the pod-side md5 and byte count (no silent drift accepted)."""
        import base64
        import hashlib

        tag = uuid.uuid4().hex[:8]
        out = self.run(
            f"rm -f /tmp/dl_{tag}_*; base64 -w0 {remote} | split -b 24000 -d -a 4 - /tmp/dl_{tag}_;"
            f" echo COUNT_$(ls /tmp/dl_{tag}_* | wc -l);"
            f" echo MD5_$(md5sum {remote} | cut -d' ' -f1);"
            f" echo SIZE_$(wc -c < {remote})",
            timeout=timeout,
        )
        count_m = re.findall(r"COUNT_(\d+)", out)
        md5_m = re.findall(r"MD5_([0-9a-f]{32})", out)
        size_m = re.findall(r"SIZE_(\d+)", out)
        if not count_m or not size_m:
            raise RuntimeError(f"[{self.name}] download header not found: {out[-300:]}")
        count = int(count_m[-1])
        size = int(size_m[-1])
        f_md5 = md5_m[-1] if md5_m else None
        chunks = []
        for i in range(count):
            part = self.run(
                f"printf 'C{i:04d}\n'; cat /tmp/dl_{tag}_{i:04d}; printf '\nD{i:04d}\n'"
            )
            s = part.rfind(f"\nC{i:04d}\n")
            e = part.rfind(f"\nD{i:04d}\n")
            if s == -1 or e == -1 or e <= s:
                raise RuntimeError(f"[{self.name}] chunk {i} markers not found")
            chunks.append(part[s + 7: e])
        blob = "".join(chunks).replace("\n", "")
        raw = base64.b64decode(blob)
        got_md5 = hashlib.md5(raw).hexdigest()
        if len(raw) != size or (f_md5 and got_md5 != f_md5):
            raise RuntimeError(
                f"[{self.name}] download mismatch: size {len(raw)} vs {size}, "
                f"md5 {got_md5} vs {f_md5}")
        self.dl_md5_ok = True
        return raw.decode("utf-8")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def pod_pipeline(url: str, name: str, branch: str, outdir: Path) -> dict:
    t = Terminal(url, name)
    say(f"[{name}] terminal open ({t.base.split('//')[1].split('.')[0]})")

    out = t.run("nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version "
                "--format=csv,noheader")
    gpu_line = next((l.strip() for l in out.splitlines()
                     if l.strip() and "name" not in l.lower().split(",")[0]
                     and not l.strip().startswith(("echo", "nvidia-smi"))), "")
    say(f"[{name}] GPU: {gpu_line}")
    gpu_name = gpu_line.split(",")[0].strip() if gpu_line else "unknown"

    out = t.run("python3 -c \"import torch,numpy,sys;print('PY',sys.version.split()[0],"
                "'TORCH',torch.__version__,'CUDA',torch.version.cuda,"
                "'AVAIL',torch.cuda.is_available(),'NUMPY',numpy.__version__)\"")
    env_line = next((l.strip() for l in out.splitlines() if l.strip().startswith("PY ")), "")
    say(f"[{name}] env: {env_line}")
    if "AVAIL True" not in env_line and "AVAIL \"True\"" not in env_line and "AVAIL=True" not in env_line:
        say(f"[{name}] WARNING: torch.cuda.is_available() not True — check output:")
        say(out[-800:])

    out = t.run(f"rm -rf $HOME/f5b && git clone -q --depth 1 -b {branch} {REPO_URL} $HOME/f5b"
                f" && cd $HOME/f5b && echo CLONE_OK_{uuid.uuid4().hex[:6]}", timeout=600)
    if "CLONE_OK" not in out:
        raise RuntimeError(f"[{name}] clone failed: ...{out[-600:]}")

    slug = slugify(gpu_name)
    res_rel = f"docs/phase-f5b-gpu-results.{slug}.json"
    say(f"[{name}] running harness on CUDA (this is the long step)...")
    out = t.run(f"cd $HOME/f5b && python3 gpu/f5b/f5b_gpu_runner.py --device cuda "
                f"--out {res_rel}", timeout=3600)
    tail = out[-1200:]
    say(f"[{name}] harness rc={t.last_rc}")
    for line in out.splitlines():
        s = line.strip()
        if any(k in s for k in ("vectors all exact", "nodes:", "schedule ratio",
                                "performance_pass", "written:", "Error", "Traceback")):
            say(f"[{name}] | {s}")
    if t.last_rc not in (0, None) or "written:" not in out:
        raise RuntimeError(f"[{name}] harness failed rc={t.last_rc}: ...{tail}")

    body = t.download_text(f"$HOME/f5b/{res_rel}")
    dest = outdir / Path(res_rel).name
    dest.write_text(body, encoding="utf-8")
    say(f"[{name}] downloaded {dest.name} ({len(body)} bytes, md5_ok={getattr(t, 'dl_md5_ok', None)})")
    data = json.loads(body)
    return {
        "name": name, "gpu": gpu_name, "env": env_line, "result_file": dest.name,
        "vectors_exact": data.get("vectors", {}).get("all_exact"),
        "nodes_exact": data.get("nodes", {}).get("all_exact"),
        "schedule_ratio": data.get("benchmark", {}).get("weighted_schedule", {}).get("ratio"),
        "performance_pass": data.get("benchmark", {}).get("performance_pass"),
        "float_ops": data.get("float_audit", {}).get("float_primitives_used_for_canonical"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-a", required=True)
    ap.add_argument("--pod-b", required=True)
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--outdir", default="docs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {
            ex.submit(pod_pipeline, args.pod_a, "podA", args.branch, outdir): "podA",
            ex.submit(pod_pipeline, args.pod_b, "podB", args.branch, outdir): "podB",
        }
        for f, name in futs.items():
            try:
                results[name] = f.result()
            except Exception as e:  # keep the other pod running
                results[name] = {"error": str(e)}
                say(f"[{name}] ERROR: {e}")

    say("=== summary ===")
    say(json.dumps(results, indent=1))
    if results.get("podA", {}).get("gpu") and results.get("podB", {}).get("gpu"):
        same = results["podA"]["gpu"] == results["podB"]["gpu"]
        say("DIFFERENT GPU MODELS:", not same)
    failed = [k for k, v in results.items() if "error" in v]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
