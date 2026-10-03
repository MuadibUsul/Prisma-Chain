"""JSON-request handlers for the two-pod GEMM E2E over any transport.

The worker side reuses gemmv1.service (its handlers are transport-agnostic
dict-in/dict-out functions). This module adds the challenger-side compute
handlers: independent GPU recomputation, tile comparison, its own tile
trace, bisection midpoints and the 8x8x8 arbitration micro-step. Both sides
run on separate machines with separate GPUs and never see each other's
intermediates; the coordinator only relays protocol messages.
"""

import time

from .protocol import (
    ArithmeticSpec,
    Operator,
    ProtocolVersion,
    assignment_id,
    build_matrix_roots,
    leaf_output_tile,
    merkle_root,
    task_id,
)
from .merkle_proofs import prove, verify_inclusion
from .tensors import (
    canonical_to_int32s,
    extract_a_tile,
    extract_b_tile,
    int32s_to_canonical,
    micro_step,
    output_tiles,
    reference_gemm,
)
from .testgen import gen_test_matrix
from .trace import build_tile_trace
from . import gpu

_CH = {}


def _bytes_of(values) -> bytes:
    return bytes(v & 0xFF for v in values)


def gpu_check(req) -> dict:
    """Bit-exactness gate: GPU INT8->INT32 vs the exact int64 oracle."""
    m = n = k = int(req.get("size", 512))
    seed = int(req.get("seed", 7))
    a = gen_test_matrix(ord("A"), seed, m * k)
    b = gen_test_matrix(ord("B"), seed, k * n)
    status = gpu.backend_status()
    t0 = time.time()
    if status["backend"] == "torch_int_mm_cuda":
        c_gpu = gpu.gpu_gemm(a, b, m, n, k)
    else:
        return {"backend": status, "bit_exact": False, "unsupported": True}
    gpu_s = time.time() - t0
    import numpy as np

    t0 = time.time()
    a_np = np.array(a, dtype=np.int64).reshape(m, k)
    b_np = np.array(b, dtype=np.int64).reshape(k, n)
    c_oracle = (a_np @ b_np).ravel().tolist()
    oracle_s = time.time() - t0
    bit_exact = c_gpu == c_oracle
    return {
        "backend": status,
        "bit_exact": bit_exact,
        "size": [m, n, k],
        "gpu_s": round(gpu_s, 4),
        "oracle_s": round(oracle_s, 2),
        "sample": c_gpu[:4],
    }


def challenger_prepare(req) -> dict:
    m, n, k, seed = int(req["m"]), int(req["n"]), int(req["k"]), int(req["seed"])
    matrix_a = gen_test_matrix(ord("A"), seed, m * k)
    matrix_b = gen_test_matrix(ord("B"), seed, k * n)
    root_a, root_b, _, _, _, _ = build_matrix_roots(matrix_a, matrix_b, m, n, k)
    descriptor = {
        "protocol_version": ProtocolVersion,
        "operator": Operator,
        "requester_pubkey": _bytes_of(gen_test_matrix(ord("Q"), seed, 32)),
        "requester_nonce": _bytes_of(gen_test_matrix(ord("N"), seed, 16)),
        "issued_epoch": 1000,
        "m": m,
        "n": n,
        "k": k,
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

    # Cross-machine task identity: the challenger must derive the same
    # task_id and assignment_id from the descriptor before anything else.
    if tid.hex() != req["task_id"] or aid.hex() != req["assignment_id"]:
        return {"rejected": "task/assignment identity mismatch"}

    status = gpu.backend_status()
    t0 = time.time()
    if status["backend"] == "torch_int_mm_cuda":
        c = gpu.gpu_gemm(matrix_a, matrix_b, m, n, k)
    else:
        c = reference_gemm(matrix_a, matrix_b, m, n, k)
    challenger_tiles = output_tiles(c, m, n)
    recompute_s = time.time() - t0

    cols_c = (n + 7) // 8

    def _tile_to_ints(t):
        if isinstance(t, str):
            return canonical_to_int32s(bytes.fromhex(t))
        return list(t)

    worker_tiles = [_tile_to_ints(t) for t in req["worker_tiles"]]
    leaves = [
        leaf_output_tile(tid, aid, idx // cols_c, idx % cols_c, int32s_to_canonical(tile))
        for idx, tile in enumerate(worker_tiles)
    ]
    root_ok = merkle_root(leaves).hex() == req["output_root"]

    disputed = None
    for idx, (w, ch) in enumerate(zip(worker_tiles, challenger_tiles)):
        if w != ch:
            disputed = (idx // cols_c, idx % cols_c)
            break

    _CH.update({
        "matrix_a": matrix_a,
        "matrix_b": matrix_b,
        "m": m, "n": n, "k": k,
        "task_id": tid,
        "assignment_id": aid,
        "root_a": root_a,
        "root_b": root_b,
        "cols_c": cols_c,
        "tiles": challenger_tiles,
    })
    return {
        "backend": status,
        "root_ok": root_ok,
        "disputed_tile": list(disputed) if disputed else None,
        "disputed_challenger_tile": (
            int32s_to_canonical(challenger_tiles[disputed[0] * cols_c + disputed[1]]).hex()
            if disputed else None
        ),
        "recompute_s": round(recompute_s, 3),
        "cols_c": cols_c,
    }


def challenger_trace(req) -> dict:
    tile_i, tile_j = int(req["tile_i"]), int(req["tile_j"])
    states, levels, root = build_tile_trace(
        _CH["matrix_a"], _CH["matrix_b"], _CH["m"], _CH["n"], _CH["k"],
        _CH["task_id"], _CH["assignment_id"], tile_i, tile_j, micro_step,
    )
    _CH["last_trace"] = (states, levels)
    proofs = []
    for step in range(len(states)):
        idx, count, siblings = prove(levels, step)
        proofs.append({"index": idx, "count": count, "siblings": [s.hex() for s in siblings]})
    return {
        "trace_root": root.hex(),
        "states": [int32s_to_canonical(s).hex() for s in states],
        "proofs": proofs,
    }


def challenger_mid(req) -> dict:
    step = int(req["step"])
    states, levels = _CH["last_trace"]
    idx, count, siblings = prove(levels, step)
    return {
        "state": int32s_to_canonical(states[step]).hex(),
        "proof": {"index": idx, "count": count, "siblings": [s.hex() for s in siblings]},
    }


def challenger_arbitrate(req) -> dict:
    """The arbiter micro-step: verify both input tiles against the matrix
    roots, recompute exactly 512 canonical MACs, compare both parties."""
    step = int(req["step"])
    high = int(req["high"])
    tile_i, tile_j = int(req["tile_i"]), int(req["tile_j"])
    low_state = canonical_to_int32s(bytes.fromhex(req["low_state"]))
    worker_high = canonical_to_int32s(bytes.fromhex(req["worker_high"]))
    a_tile = extract_a_tile(_CH["matrix_a"], _CH["m"], _CH["k"], tile_i, step)
    b_tile = extract_b_tile(_CH["matrix_b"], _CH["k"], _CH["n"], step, tile_j)
    expected = micro_step(low_state, a_tile, b_tile)
    states, _levels = _CH["last_trace"]
    challenger_high = states[high]
    worker_ok = expected == worker_high
    challenger_ok = expected == challenger_high
    if worker_ok:
        outcome = "worker_wins"
    elif challenger_ok:
        outcome = "challenger_wins"
    else:
        outcome = "both_invalid"
    return {
        "outcome": outcome,
        "step": step,
        "arbitration_macs": 512,
        "expected": int32s_to_canonical(expected).hex(),
        "worker_ok": worker_ok,
        "challenger_ok": challenger_ok,
    }


def dispatch(kind: str, req: dict) -> dict:
    handlers = {
        "gpu_check": gpu_check,
        "prepare": challenger_prepare,
        "trace": challenger_trace,
        "mid": challenger_mid,
        "arbitrate": challenger_arbitrate,
    }
    return handlers[kind](req)
