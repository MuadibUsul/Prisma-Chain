"""DA daemon HTTP surface (B4-01).

```text
GET  /health                    provider info, advertised key, stored tasks
GET  /metrics                   counters + index metrics
POST /store                     verify then persist (OUTPUT_ROOT_MISMATCH refuses)
POST /attest                    canonical signed DAAttestation (Go-shaped wire)
GET  /v1/da/{task}/output       canonical blob (hex), re-verified before serving
GET  /v1/da/{task}/tile/{i}/{j} typed chunk proof — lands with B4-02 (501 until then)
```

Design notes:

- **Routing is a pure function** (``Daemon.handle``), so the whole surface is
  testable without sockets; ``serve`` only wires it to a threaded HTTP server.
- **Graceful shutdown**: the server stops accepting, in-flight requests drain
  (bounded), then it exits — a write in progress is never cut in half (the
  storage layer stages and renames atomically anyway).
- **Restart recovery**: the index is rebuilt from disk at start-up
  (``ArtifactIndex.rescan``), so a restarted daemon serves exactly what still
  verifies.
- The attestation is signed with the *frozen* preimage
  (``PRISMA_GEMM_SIG_V1\\x00 || encode_canonical(...)`` from the frozen
  libraries) and in the Go-shaped wire form the chain unmarshals.
"""

from __future__ import annotations

import base64
import json
import pathlib
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from .config import DaemonConfig
from .frozen import frozen
from .storage import ArtifactIndex, IntegrityError, RootMismatch, StorageError

DA_PROTOCOL_VERSION = "DA_REPLICA_V1"
DA_SIGN_DOMAIN = b"PRISMA_GEMM_SIG_V1\x00"
BLOB_HEX_ENDPOINT = "/v1/da/"


class Daemon:
    def __init__(self, config: DaemonConfig, identity, *,
                 index: Optional[ArtifactIndex] = None,
                 log: Callable[[str], None] = print):
        self.config = config
        self.identity = identity
        self.index = index or ArtifactIndex(config.resolved_data_dir())
        self.log = log
        self.started_at = time.time()
        self.counters = {"requests": 0, "store_ok": 0, "store_rejected": 0, "attestations": 0,
                         "output_served": 0, "tiles_served": 0, "output_missing": 0, "errors": 0}
        self._active = 0
        self._lock = threading.Lock()
        self._shutting_down = False
        self.server: Optional[ThreadingHTTPServer] = None

    # --- attestation ------------------------------------------------------

    def attestation(self, *, task_id: int, assignment_id_hex: str, available_until: int,
                    attested_height: int) -> dict:
        """Build and sign the canonical DAAttestation in the Go-shaped wire form."""
        meta = self.index.load_meta(task_id)
        if meta is None:
            raise StorageError(f"task {task_id} is not stored; refusing to attest")
        if self.index.output(task_id) is None:  # re-verify before attesting
            raise StorageError(f"task {task_id} has no verifiable blob; refusing to attest")
        canonical = {
            "protocol_version": DA_PROTOCOL_VERSION,
            "task_id": task_id.to_bytes(8, "big"),
            "assignment_id": bytes.fromhex(assignment_id_hex),
            "output_root": bytes.fromhex(str(meta["output_root"])),
            "provider_account": self.config.account.encode(),
            "provider_pub_key": self.identity.protocol_public,
            "output_bytes": int(meta["m"]) * int(meta["n"]) * 4,
            "available_until_height": int(available_until),
            "attested_height": int(attested_height),
            "signature": b"",
        }
        preimage = DA_SIGN_DOMAIN + frozen.cbor.encode_canonical(canonical)
        signature = self.identity.sign_raw(preimage)
        wire = {
            "ProtocolVersion": canonical["protocol_version"],
            "TaskID": b64(canonical["task_id"]),
            "AssignmentID": b64(canonical["assignment_id"]),
            "OutputRoot": b64(canonical["output_root"]),
            "ProviderAccount": b64(canonical["provider_account"]),
            "ProviderPubKey": b64(canonical["provider_pub_key"]),
            "OutputBytes": canonical["output_bytes"],
            "AvailableUntilHeight": canonical["available_until_height"],
            "AttestedHeight": canonical["attested_height"],
            "Signature": b64(signature),
        }
        self.counters["attestations"] += 1
        return {"attestation": wire, "attestation_json": json.dumps(wire).encode().hex()}

    # --- routing (pure) ---------------------------------------------------

    def handle(self, method: str, path: str, body: Optional[dict]) -> tuple[int, dict]:
        self.counters["requests"] += 1
        body = body or {}
        try:
            if method == "GET" and path == "/health":
                return 200, self.health()
            if method == "GET" and path == "/metrics":
                return 200, self.metrics()
            if method == "POST" and path == "/store":
                required = ("task_id", "c_hex", "output_root", "m", "n", "task_id32",
                            "assignment_id")
                missing = [name for name in required if name not in body]
                if missing:
                    self.counters["store_rejected"] += 1
                    return 400, {"error": f"missing fields: {', '.join(missing)}"}
                try:
                    result = self.index.store(
                        task_id=int(body["task_id"]), c_hex=str(body["c_hex"]),
                        output_root_hex=str(body["output_root"]), m=int(body["m"]),
                        n=int(body["n"]), task_id32_hex=str(body["task_id32"]),
                        assignment_id_hex=str(body["assignment_id"]))
                except RootMismatch as exc:
                    self.counters["store_rejected"] += 1
                    return 400, {"error": str(exc)}
                except (StorageError, ValueError) as exc:
                    self.counters["store_rejected"] += 1
                    return 400, {"error": f"malformed store request: {exc}"}
                self.counters["store_ok"] += 1
                self.log(f"stored task {body['task_id']} ({result['bytes']} bytes)")
                return 200, {"ok": True, **result}
            if method == "POST" and path == "/attest":
                try:
                    result = self.attestation(
                        task_id=int(body["task_id"]), assignment_id_hex=str(body["assignment_id"]),
                        available_until=int(body["available_until"]),
                        attested_height=int(body["attested_height"]))
                except (StorageError, KeyError, ValueError) as exc:
                    return 404 if isinstance(exc, StorageError) else 400, {"error": str(exc)}
                return 200, result
            if method == "GET" and path.startswith(BLOB_HEX_ENDPOINT):
                return self._handle_blob(path)
            self.counters["errors"] += 1
            return 404, {"error": f"no such endpoint: {method} {path}"}
        except Exception as exc:  # noqa: BLE001 - never leak a raw traceback to a peer
            self.counters["errors"] += 1
            return 500, {"error": f"internal error: {exc}"}

    def _handle_blob(self, path: str) -> tuple[int, dict]:
        parts = [piece for piece in path.split("/") if piece]
        # /v1/da/{task}/output  or  /v1/da/{task}/tile/{i}/{j}
        if len(parts) == 4 and parts[3] == "output":
            task_id = int(parts[2])
            try:
                blob = self.index.output(task_id)
            except IntegrityError as exc:
                self.counters["errors"] += 1
                return 500, {"error": str(exc)}
            if blob is None:
                self.counters["output_missing"] += 1
                return 404, {"error": "DATA_UNAVAILABLE"}
            self.counters["output_served"] += 1
            return 200, {"task_id": task_id, "c_hex": blob.hex()}
        if len(parts) == 6 and parts[3] == "tile":
            try:
                task_id, tile_i, tile_j = int(parts[2]), int(parts[4]), int(parts[5])
            except ValueError:
                self.counters["errors"] += 1
                return 400, {"error": "tile coordinates must be integers"}
            try:
                tile = self.index.tile(task_id, tile_i, tile_j)
            except IntegrityError as exc:
                self.counters["errors"] += 1
                return 500, {"error": str(exc)}
            if tile is None:
                self.counters["output_missing"] += 1
                return 404, {"error": "DATA_UNAVAILABLE"}
            self.counters["tiles_served"] += 1
            return 200, tile
        return 404, {"error": "no such endpoint"}

    # --- data ------------------------------------------------------------

    def health(self) -> dict:
        return {
            "provider": self.config.account,
            "protocol": DA_PROTOCOL_VERSION,
            "pubkey": base64.b64encode(self.identity.protocol_public).decode(),
            "tasks": sorted(self.index.entries),
            "uptime_s": round(time.time() - self.started_at, 3),
            "shutting_down": self._shutting_down,
        }

    def metrics(self) -> dict:
        return {**self.counters, "index": self.index.metrics(),
                "uptime_s": round(time.time() - self.started_at, 3)}

    # --- serving ----------------------------------------------------------

    def serve(self, *, ready: Optional[Callable[[dict], None]] = None,
              run_forever: bool = True) -> None:
        """Start the HTTP surface; blocks until a shutdown signal."""
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _respond(self, code: int, payload: dict):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _dispatch(self, method: str):
                with outer._lock:
                    outer._active += 1
                try:
                    length = int(self.headers.get("Content-Length", "0") or 0)
                    raw = self.rfile.read(length) if length else b""
                    try:
                        body = json.loads(raw) if raw else {}
                    except json.JSONDecodeError:
                        self._respond(400, {"error": "request body is not JSON"})
                        return
                    code, payload = outer.handle(method, self.path.split("?")[0], body)
                    self._respond(code, payload)
                finally:
                    with outer._lock:
                        outer._active -= 1

            def do_GET(self):  # noqa: N802
                self._dispatch("GET")

            def do_POST(self):  # noqa: N802
                self._dispatch("POST")

            def log_message(self, *args):  # the daemon logs its own way
                return

        self.server = ThreadingHTTPServer((self.config.listen_host, self.config.listen_port),
                                          Handler)
        report = self.index.rescan()
        self.log(f"restart recovery: scanned {report['scanned']} artifact(s), "
                 f"serving {report['serving']}, quarantined {len(report['bad'])}")
        self.log(f"prisma-da listening on {self.config.listen_host}:{self.config.listen_port} "
                 f"as {self.config.account} (data {self.index.root})")

        stopping = threading.Event()

        def _signal(_signum, _frame):
            stopping.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _signal)
            except (ValueError, OSError):  # not the main thread / unsupported platform
                pass
        if ready:
            ready(self.health())
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        if run_forever:
            while not stopping.is_set():
                time.sleep(0.2)
            self.shutdown()

    def shutdown(self, *, drain_seconds: float = 10.0) -> None:
        """Stop accepting, drain in-flight requests (bounded), then exit."""
        self._shutting_down = True
        self.log("shutdown: draining in-flight requests")
        deadline = time.monotonic() + drain_seconds
        while time.monotonic() < deadline:
            with self._lock:
                if self._active == 0:
                    break
            time.sleep(0.05)
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        self.log("shutdown complete")


def b64(value: bytes) -> str:
    return base64.b64encode(value).decode()
