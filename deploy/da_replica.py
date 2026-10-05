"""DA_REPLICA_V1 replica daemon.

One process = one permissionless DA provider. It stores canonical full C
blobs on disk, recomputes output_root from the bytes it received BEFORE
attesting anything, serves bulk and per-tile endpoints, and signs
canonical DAAttestation objects with the provider's bonded network
Ed25519 key.

Endpoints (JSON over HTTP):
  GET  /health                          -> provider info and stored tasks
  POST /store    {task_id, c_hex, output_root, m, n, k, assignment_id,
                  task_id32(hex), assigned...} -> verifies output_root,
                  persists the blob under ./data/<task_id>/ and returns ok
  GET  /v1/da/{task_id}/output          -> canonical C bytes (hex)
  GET  /v1/da/{task_id}/tile/{i}/{j}    -> 256-byte tile + Merkle proof
  POST /attest   {task_id, ...}         -> canonical signed DAAttestation
  POST /forget   {task_id}              -> DEVELOPMENT: drop a stored blob
                  to simulate a provider that lost data after attesting

The daemon never trusts a claimed root: a blob whose recomputed root does
not match is rejected and never stored (DoD E2 scenario D).
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "compute" / "gemmv1" / "python"))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from gemmv1.merkle_proofs import build_levels, prove  # noqa: E402
from gemmv1.protocol import (  # noqa: E402
    DAProtocolVersion,
    leaf_output_tile,
    merkle_root,
)
from gemmv1.tensors import int32s_to_canonical, output_tiles  # noqa: E402
from gemmv1.canonical_cbor import encode_canonical  # noqa: E402


class Replica:
    def __init__(self, account: str, key_hex: str, store_dir: Path):
        self.account = account
        self.private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(key_hex))
        self.pubkey = self.private.public_key().public_bytes_raw()
        self.store = store_dir
        self.store.mkdir(parents=True, exist_ok=True)

    def task_dir(self, task_id: int) -> Path:
        return self.store / str(task_id)

    def store_blob(self, task_id: int, c_hex: str, output_root: bytes,
                   m: int, n: int, task_id32: bytes, assignment_id: bytes) -> dict:
        """Verify then persist. Never store an unverified blob."""
        c = bytes.fromhex(c_hex)
        if len(c) != m * n * 4:
            return {"error": "blob length does not match M*N*4"}
        values = [int.from_bytes(c[i:i + 4], "big", signed=True) for i in range(0, len(c), 4)]
        tile_grid = output_tiles(values, m, n)
        cols_c = (n + 7) // 8
        leaves = [
            leaf_output_tile(task_id32, assignment_id, i // cols_c, i % cols_c,
                             int32s_to_canonical(t))
            for i, t in enumerate(tile_grid)
        ]
        root = merkle_root(leaves)
        if root != output_root:
            return {"error": "OUTPUT_ROOT_MISMATCH: refusing to store or attest"}
        directory = self.task_dir(task_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "output.bin").write_bytes(c)
        (directory / "meta.json").write_text(json.dumps({
            "m": m, "n": n, "output_root": output_root.hex(),
            "task_id32": task_id32.hex(), "assignment_id": assignment_id.hex(),
        }))
        return {"stored": True, "bytes": len(c), "output_root": root.hex()}

    def load_meta(self, task_id: int) -> dict:
        return json.loads((self.task_dir(task_id) / "meta.json").read_text())

    def serve_output(self, task_id: int):
        path = self.task_dir(task_id) / "output.bin"
        if not path.exists():
            return None
        return path.read_bytes()

    def serve_tile(self, task_id: int, i: int, j: int) -> dict:
        meta = self.load_meta(task_id)
        c = self.serve_output(task_id)
        if c is None:
            return {"error": "DATA_UNAVAILABLE"}
        m, n = meta["m"], meta["n"]
        values = [int.from_bytes(c[k:k + 4], "big", signed=True) for k in range(0, len(c), 4)]
        tiles = output_tiles(values, m, n)
        cols_c = (n + 7) // 8
        leaves = [
            leaf_output_tile(bytes.fromhex(meta["task_id32"]), bytes.fromhex(meta["assignment_id"]),
                             t // cols_c, t % cols_c, int32s_to_canonical(tile))
            for t, tile in enumerate(tiles)
        ]
        levels = build_levels(leaves)
        index = i * cols_c + j
        idx, count, siblings = prove(levels, index)
        return {
            "tile": int32s_to_canonical(tiles[index]).hex(),
            "proof": {"index": idx, "count": count, "siblings": [s.hex() for s in siblings]},
        }

    def attest(self, task_id: int, assignment_id: bytes, available_until: int,
               attested_height: int) -> dict:
        """Build and sign the canonical DAAttestation. The provider signs
        only after the blob was verified at store time."""
        meta = self.load_meta(task_id)
        attestation = {
            "protocol_version": DAProtocolVersion,
            "task_id": bytes.fromhex(meta["task_id32"]),
            "assignment_id": assignment_id,
            "output_root": bytes.fromhex(meta["output_root"]),
            "provider_account": self.account.encode(),
            "provider_pub_key": self.pubkey,
            "output_bytes": meta["m"] * meta["n"] * 4,
            "available_until_height": available_until,
            "attested_height": attested_height,
            "signature": b"",
        }
        preimage = b"PRISMA_GEMM_SIG_V1\x00" + encode_canonical(attestation)
        attestation["signature"] = self.private.sign(preimage)
        # Wire form: the Go JSON shape (Go field names, base64 byte fields)
        # so the chain's json.Unmarshal reconstructs exactly the signed
        # object before re-encoding the canonical CBOR preimage.
        import base64 as b64

        def b(value):
            return b64.b64encode(value).decode()

        wire = {
            "ProtocolVersion": DAProtocolVersion,
            "TaskID": b(attestation["task_id"]),
            "AssignmentID": b(attestation["assignment_id"]),
            "OutputRoot": b(attestation["output_root"]),
            "ProviderAccount": b(attestation["provider_account"]),
            "ProviderPubKey": b(attestation["provider_pub_key"]),
            "OutputBytes": attestation["output_bytes"],
            "AvailableUntilHeight": attestation["available_until_height"],
            "AttestedHeight": attestation["attested_height"],
            "Signature": b(attestation["signature"]),
        }
        return {"attestation_json": json.dumps(wire).encode().hex(), "attestation": wire}


class Handler(BaseHTTPRequestHandler):
    replica: Replica = None

    def log_message(self, *args):
        pass

    def _json(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parts = [p for p in self.path.split("/") if p]
        try:
            if self.path == "/health":
                stored = sorted(p.name for p in self.replica.store.iterdir() if p.is_dir())
                self._json(200, {"account": self.replica.account,
                                 "pubkey": self.replica.pubkey.hex(), "tasks": stored})
                return
            if len(parts) >= 4 and parts[0] == "v1" and parts[1] == "da" and parts[3] == "output":
                blob = self.replica.serve_output(int(parts[2]))
                if blob is None:
                    self._json(404, {"error": "DATA_UNAVAILABLE"})
                else:
                    self._json(200, {"c_hex": blob.hex()})
                return
            if len(parts) >= 6 and parts[0] == "v1" and parts[1] == "da" and parts[3] == "tile":
                self._json(200, self.replica.serve_tile(int(parts[2]), int(parts[4]), int(parts[5])))
                return
        except Exception as exc:
            self._json(500, {"error": repr(exc)})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            if self.path == "/store":
                result = self.replica.store_blob(
                    int(body["task_id"]), body["c_hex"], bytes.fromhex(body["output_root"]),
                    int(body["m"]), int(body["n"]), bytes.fromhex(body["task_id32"]),
                    bytes.fromhex(body["assignment_id"]))
                self._json(200 if result.get("stored") else 400, result)
                return
            if self.path == "/attest":
                result = self.replica.attest(
                    int(body["task_id"]), bytes.fromhex(body["assignment_id"]),
                    int(body["available_until"]), int(body["attested_height"]))
                self._json(200, result)
                return
            if self.path == "/forget":
                directory = self.replica.task_dir(int(body["task_id"]))
                if (directory / "output.bin").exists():
                    os.remove(directory / "output.bin")
                self._json(200, {"forgotten": True})
                return
        except FileNotFoundError:
            self._json(404, {"error": "DATA_UNAVAILABLE"})
            return
        except Exception as exc:
            self._json(500, {"error": repr(exc)})
            return
        self._json(404, {"error": "not found"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True)
    parser.add_argument("--key-hex", required=True, help="bonded network Ed25519 seed hex")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--store", required=True)
    args = parser.parse_args()
    Handler.replica = Replica(args.account, args.key_hex, Path(args.store))
    print(f"DA replica for {args.account} on :{args.port} store={args.store}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
