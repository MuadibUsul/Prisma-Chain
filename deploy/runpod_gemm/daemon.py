"""Stateful JSON-RPC daemon for one GEMM E2E pod.

Usage: python3 daemon.py --mode worker|challenger --port 8311

The daemon keeps the task state in memory (matrices, tiles, traces) so the
coordinator can drive multi-step protocol interactions over the Jupyter
terminal relay without re-importing torch on every call. Requests are
newline-delimited JSON on the socket; responses are one JSON line.
"""

import argparse
import json
import os
import socketserver
import sys

sys.path.insert(0, "/root/prisma")


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline()
            req = json.loads(line)
        except Exception as exc:
            self.wfile.write((json.dumps({"fatal": f"bad request: {exc}"}) + "\n").encode())
            return
        try:
            kind = req.pop("_kind")
            if kind == "ping":
                resp = {"ok": True, "pid": os.getpid()}
            elif SERVER["mode"] == "worker":
                import gemmv1.service as S

                if kind == "gpu_check":
                    from gemmv1 import e2e

                    resp = e2e.dispatch("gpu_check", req)
                else:
                    fn = {"execute": S.handle_execute, "trace": S.handle_trace, "mid": S.handle_mid}[kind]
                    resp = fn(req)
            else:
                from gemmv1 import e2e

                resp = e2e.dispatch(kind, req)
        except Exception as exc:
            import traceback

            resp = {"fatal": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
        self.wfile.write((json.dumps(resp) + "\n").encode())


SERVER = {"mode": "challenger"}


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["worker", "challenger"], required=True)
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    SERVER["mode"] = args.mode
    if args.mode == "worker":
        import gemmv1.service  # noqa: F401  (warm import)
    else:
        from gemmv1 import e2e  # noqa: F401
    srv = Server(("127.0.0.1", args.port), Handler)
    print("DAEMON_READY", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
