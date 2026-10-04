"""F.5B.2 single-pod driver over the Jupyter terminal relay.

Cost discipline (spec sections 60-63, 106-108): ONE GPU first; the second
pod only after the first passes.  Each pod: capture environment, clone the
F.5B.1 branch, build the exact-integer CUDA extension, run the fusion
ladder runner, download the result byte-exact.

Usage:
  python run_f5b1_jupyter.py --pod URL [--branch B] [--outdir DIR] [--label L]

The driver never prints or stores pod tokens; URLs come from the command line.
"""

import argparse
import base64
import hashlib
import json
import re
import sys
import time
import uuid
from pathlib import Path

import requests
from websocket import create_connection

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")
BRANCH = "research/transformer-phase-f5b2-full-block-gpu"
REPO_URL = "https://github.com/MuadibUsul/Prisma-Chain.git"


class Terminal:
    def __init__(self, url: str):
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
        self._drain(2)

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
        """Byte-exact pull: base64 chunks + markers, verified against the
        pod-side md5 and byte count."""
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
            raise RuntimeError(f"download header not found: {out[-300:]}")
        count, size = int(count_m[-1]), int(size_m[-1])
        f_md5 = md5_m[-1] if md5_m else None
        chunks = []
        for i in range(count):
            part = self.run(
                f"printf 'C{i:04d}\n'; cat /tmp/dl_{tag}_{i:04d}; printf '\nD{i:04d}\n'"
            )
            s = part.rfind(f"\nC{i:04d}\n")
            e = part.rfind(f"\nD{i:04d}\n")
            if s == -1 or e == -1 or e <= s:
                raise RuntimeError(f"chunk {i} markers not found")
            chunks.append(part[s + 7: e])
        blob = "".join(chunks).replace("\n", "")
        raw = base64.b64decode(blob)
        got_md5 = hashlib.md5(raw).hexdigest()
        if len(raw) != size or (f_md5 and got_md5 != f_md5):
            raise RuntimeError(f"download mismatch: size {len(raw)} vs {size}, "
                               f"md5 {got_md5} vs {f_md5}")
        return raw.decode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", required=True)
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--outdir", default="docs")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    t = Terminal(args.pod)
    print("[pod] terminal open", flush=True)

    out = t.run("nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version "
                "--format=csv,noheader")
    gpu_line = next((l.strip() for l in out.splitlines()
                     if l.strip() and "nvidia-smi" not in l and ";" not in l), "unknown")
    print("[pod] GPU:", gpu_line, flush=True)

    out = t.run("python3 -c \"import torch,numpy,sys;print('PY',sys.version.split()[0],"
                "'TORCH',torch.__version__,'CUDA',torch.version.cuda,"
                "'AVAIL',torch.cuda.is_available(),"
                "'NVCC',torch.utils.cpp_extension.CUDA_HOME)\"")
    env_line = next((l.strip() for l in out.splitlines()
                     if l.strip().startswith("PY") and "TORCH" in l), "")
    print("[pod] env:", env_line, flush=True)

    out = t.run("python3 -c 'import ninja' 2>/dev/null && echo NINJA_OK "
                "|| (python3 -m pip install -q ninja && python3 -c 'import ninja' "
                "&& echo NINJA_INSTALLED)", timeout=600)
    ok_ninja = "NINJA_OK" in out or "NINJA_INSTALLED" in out
    print("[pod] ninja:", "ok" if ok_ninja else f"MISSING ({out[-200:]})", flush=True)
    if not ok_ninja:
        raise RuntimeError("ninja unavailable on pod and pip install failed")

    out = t.run(f"rm -rf $HOME/f5b2 && git clone -q --depth 1 -b {args.branch} "
                f"{REPO_URL} $HOME/f5b2 && cd $HOME/f5b2 && echo CLONE_OK", timeout=600)
    if "CLONE_OK" not in out:
        raise RuntimeError(f"clone failed: ...{out[-600:]}")

    print("[pod] running fusion-ladder runner (build + selftest + correctness + "
          "profile + benchmark; long step)...", flush=True)
    out = t.run(
        "cd $HOME/f5b2 && TORCH_CUDA_ARCH_LIST='8.6;8.9' python3 gpu/f5b2/f5b2_runner.py "
        "--device cuda --out docs/phase-f5b2-gpu-results.current.json", timeout=5400)
    for line in out.splitlines():
        s = line.strip()
        if any(k in s for k in ("selftest", "correctness:", "weighted ", "ladder winner",
                                "written:", "Error", "Traceback", "SELFTEST FAILED")):
            print("[pod] |", s, flush=True)
    if t.last_rc != 0 or "written:" not in out:
        print("[pod] runner output tail:\n", out[-2500:], flush=True)
        raise RuntimeError(f"runner failed rc={t.last_rc}")

    slug = re.sub(r"[^a-z0-9]+", "_", gpu_line.split(",")[0].lower()).strip("_")
    body = t.download_text("$HOME/f5b2/docs/phase-f5b2-gpu-results.current.json")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"phase-f5b2-gpu-results.{slug}.json"
    dest.write_text(body, encoding="utf-8")
    data = json.loads(body)
    print("[pod] downloaded", dest.name, f"({len(body)} bytes, md5-verified)", flush=True)
    lad = data.get("benchmark", {}).get("ladder", {})
    blk = data.get("full_block", {})
    print("[pod] operator gate:", data.get("operator_gate_pass"),
          "| GEMM regression:", data.get("gemm_regression", {}).get("all_exact"))
    print("[pod] full block:", blk.get("nodes_exact"), "/", blk.get("nodes_total"),
          "nodes exact;", blk.get("roots_exact"), "roots exact; final_exact =",
          blk.get("final_exact"))
    print("[pod] audit:", json.dumps(blk.get("audit", {})))
    for r in (blk.get("first_bad_nodes") or [])[:5]:
        print("   BAD:", r)
    sys.exit(0)


if __name__ == "__main__":
    main()
