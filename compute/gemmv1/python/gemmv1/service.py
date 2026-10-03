"""Worker HTTP service for the two-node GEMM E2E.

Endpoints (all POST JSON):
  /health                  -> backend status
  /execute                 -> run the task, return ResultCommit + output tiles
  /fraud                   -> DEVELOPMENT ONLY: corrupt one output tile in the
                              cached result so the challenger must dispute it
  /trace                   -> on-demand tile trace for a disputed tile
  /mid                     -> bisection midpoint state with proof

The worker keeps the last task state in memory; this is a development
service for the E2E experiment, not a production server.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .protocol import (
    ArithmeticSpec,
    Operator,
    ProtocolVersion,
    assignment_id,
    build_matrix_roots,
    leaf_output_tile,
    leaf_trace_state,
    merkle_root,
    task_id,
)
from .tensors import int32s_to_canonical, micro_step, output_tiles, reference_gemm
from .testgen import gen_test_matrix
from .trace import build_tile_trace, submit_mid_state
from .merkle_proofs import prove, build_levels
from . import gpu

STATE_LOCK = threading.Lock()
STATE = {}


def _bytes_of(values) -> bytes:
    return bytes(v & 0xFF for v in values)


def handle_execute(body: dict) -> dict:
    m, n, k, seed = int(body["m"]), int(body["n"]), int(body["k"]), int(body["seed"])
    matrix_a = gen_test_matrix(ord("A"), seed, m * k)
    matrix_b = gen_test_matrix(ord("B"), seed, k * n)
    root_a, root_b, _, _, _, _ = build_matrix_roots(matrix_a, matrix_b, m, n, k)
    descriptor = {
        "protocol_version": ProtocolVersion,
        "operator": Operator,
        "requester_pubkey": _bytes_of(gen_test_matrix(ord("Q"), seed, 32)),
        "requester_nonce": _bytes_of(gen_test_matrix(ord("N"), seed, 16)),
        "issued_epoch": 1000,
        "m": m, "n": n, "k": k,
        "matrix_a_root": root_a,
        "matrix_b_root": root_b,
        "arithmetic_spec": ArithmeticSpec,
        "tile_size": 8,
        "challenge_window": 100,
        "max_price_per_cwu": 1000,
        "settlement_asset": "uprsm",
    }
    tid = task_id(descriptor)
    assignment = {
        "task_id": tid,
        "worker_pubkey": _bytes_of(gen_test_matrix(ord("W"), seed, 32)),
        "assignment_nonce": _bytes_of(gen_test_matrix(ord("M"), seed, 16)),
        "accepted_epoch": 1001,
    }
    aid = assignment_id(assignment)

    backend = gpu.backend_status()
    if backend["backend"] == "torch_int_mm_cuda":
        c = gpu.gpu_gemm(matrix_a, matrix_b, m, n, k)
    else:
        c = reference_gemm(matrix_a, matrix_b, m, n, k)
    cols_c = (n + 7) // 8
    tiles = output_tiles(c, m, n)
    inject = body.get("inject_fraud")
    fabricated = None
    if inject is not None:
        # DEVELOPMENT ONLY: the lying worker corrupts one output tile before
        # committing, so the challenger must detect and dispute it.
        fi, fj = int(inject[0]), int(inject[1])
        tiles[fi * cols_c + fj][0] += 1
        fabricated = (fi, fj)
        print(f"[worker] DEVELOPMENT fraud injected into tile ({fi},{fj})", flush=True)
    leaves = [
        leaf_output_tile(tid, aid, idx // cols_c, idx % cols_c, int32s_to_canonical(tile))
        for idx, tile in enumerate(tiles)
    ]
    output_root = merkle_root(leaves)

    with STATE_LOCK:
        STATE.clear()
        STATE.update({
            "descriptor": descriptor,
            "task_id": tid,
            "assignment": assignment,
            "assignment_id": aid,
            "matrix_a": matrix_a,
            "matrix_b": matrix_b,
            "tiles": tiles,
            "cols_c": cols_c,
            "backend": backend,
            "fabricated_tile": fabricated,
        })
    commit = {
        "protocol_version": ProtocolVersion,
        "task_id": tid.hex(),
        "assignment_id": aid.hex(),
        "worker_pubkey": assignment["worker_pubkey"].hex(),
        "output_root": output_root.hex(),
        "canonical_mac_count": m * n * k,
        "completed_epoch": 1002,
    }
    return {
        "result_commit": commit,
        "output_tiles": {"cols_c": cols_c, "tiles": tiles},
        "backend": backend,
        "descriptor": {kk: (vv.hex() if isinstance(vv, bytes) else vv) for kk, vv in descriptor.items()},
        "assignment_id": aid.hex(),
    }


def handle_fraud(body: dict) -> dict:
    """DEVELOPMENT ONLY: corrupt one committed output tile."""
    tile_i, tile_j = int(body["tile_i"]), int(body["tile_j"])
    with STATE_LOCK:
        tiles = STATE["tiles"]
        cols_c = STATE["cols_c"]
        idx = tile_i * cols_c + tile_j
        tiles[idx][0] += 1
        descriptor = STATE["descriptor"]
        leaves = [
            leaf_output_tile(STATE["task_id"], STATE["assignment_id"], t // cols_c, t % cols_c, int32s_to_canonical(tile))
            for t, tile in enumerate(tiles)
        ]
        STATE["output_root"] = merkle_root(leaves)
        m, n, k = descriptor["m"], descriptor["n"], descriptor["k"]
    return {
        "corrupted_tile": [tile_i, tile_j],
        "new_output_root": STATE["output_root"].hex(),
    }


def _trace_for(tile_i: int, tile_j: int):
    """States, levels and root for one tile; applies the development trace
    fabrication when this tile was corrupted so that the locked trace, the
    /trace response and every /mid response stay consistent."""
    with STATE_LOCK:
        descriptor = STATE["descriptor"]
        matrix_a, matrix_b = STATE["matrix_a"], STATE["matrix_b"]
        task, aid = STATE["task_id"], STATE["assignment_id"]
        committed_tile = list(STATE["tiles"][tile_i * STATE["cols_c"] + tile_j])
        fabricated = STATE.get("fabricated_tile")
    m, n, k = descriptor["m"], descriptor["n"], descriptor["k"]
    states, levels, root = build_tile_trace(matrix_a, matrix_b, m, n, k, task, aid, tile_i, tile_j, micro_step)
    if fabricated == (tile_i, tile_j):
        # DEVELOPMENT SIMULATION of a lying worker: fabricate the trace so
        # S_R ends exactly at the committed (corrupted) tile.
        r_steps = len(states) - 1
        delta = committed_tile[0] - states[r_steps][0]
        for r in range(r_steps, len(states)):
            states[r][0] += delta
        leaves = [
            leaf_trace_state(task, aid, tile_i, tile_j, step, int32s_to_canonical(s))
            for step, s in enumerate(states)
        ]
        levels = build_levels(leaves)
        root = levels[-1][0]
    return states, levels, root


def handle_trace(body: dict) -> dict:
    tile_i, tile_j = int(body["tile_i"]), int(body["tile_j"])
    states, levels, root = _trace_for(tile_i, tile_j)
    proofs = []
    for step in range(len(states)):
        idx, count, siblings = prove(levels, step)
        proofs.append({"index": idx, "count": count, "siblings": [s.hex() for s in siblings]})
    return {
        "trace_root": root.hex(),
        "states": [int32s_to_canonical(s).hex() for s in states],
        "proofs": proofs,
    }


def handle_mid(body: dict) -> dict:
    step = int(body["step"])
    tile_i, tile_j = int(body["tile_i"]), int(body["tile_j"])
    states, levels, _ = _trace_for(tile_i, tile_j)
    state_bytes, proof = submit_mid_state(states, levels, step)
    return {"state": state_bytes.hex(), "proof": proof}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console readable
        pass

    def _json(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            if self.path == "/health":
                self._json(200, gpu.backend_status())
            elif self.path == "/execute":
                self._json(200, handle_execute(body))
            elif self.path == "/fraud":
                self._json(200, handle_fraud(body))
            elif self.path == "/trace":
                self._json(200, handle_trace(body))
            elif self.path == "/mid":
                self._json(200, handle_mid(body))
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:  # surface errors to the challenger
            self._json(500, {"error": str(exc)})


def serve(host: str, port: int) -> None:
    ThreadingHTTPServer((host, port), Handler).serve_forever()
