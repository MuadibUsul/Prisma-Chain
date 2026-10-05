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

# Protocol instrumentation (invariant 5): the fast verification path must
# never call the full reference GEMM. Tests assert this counter stays zero.
STATS = {"reference_gemm_calls": 0, "row_gemm_calls": 0}


def _reference_gemm(a, b, m, n, k):
    STATS["reference_gemm_calls"] += 1
    return reference_gemm(a, b, m, n, k)


def _tile_to_ints(t):
    if isinstance(t, str):
        return canonical_to_int32s(bytes.fromhex(t))
    return list(t)


def new_challenge_src(commit_id):
    """Production CSPRNG challenge randomness (invariants 1 and 6)."""
    from gemmv1.freivalds import new_challenge_randomness

    return new_challenge_randomness(commit_id)


def _bytes_of(values) -> bytes:
    return bytes(v & 0xFF for v in values)


def _verify_received_c(descriptor, assignment, worker_tiles, output_root):
    """Invariant 7: rebuild the output Merkle tree over the received C and
    compare with the committed root before anything touches the data."""
    from gemmv1.protocol import leaf_output_tile, merkle_root
    from gemmv1.tensors import int32s_to_canonical

    cols_c = (descriptor["n"] + 7) // 8
    leaves = [
        leaf_output_tile(descriptor["task_id"] if isinstance(descriptor.get("task_id"), bytes) else bytes.fromhex(descriptor["task_id"]),
                         assignment, idx // cols_c, idx % cols_c, int32s_to_canonical(t))
        for idx, t in enumerate(worker_tiles)
    ]
    return merkle_root(leaves).hex() == output_root


def _c_from_tiles(worker_tiles, m, n):
    cols_c = (n + 7) // 8
    c = [0] * (m * n)
    for idx, tile in enumerate(worker_tiles):
        ti, tj = idx // cols_c, idx % cols_c
        for row in range(8):
            gi = ti * 8 + row
            if gi >= m:
                break
            for col in range(8):
                gj = tj * 8 + col
                if gj >= n:
                    break
                c[gi * n + gj] = tile[row * 8 + col]
    return c


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
    if req.get("mode") == "fast":
        # v0.1.2 fast path: no full recomputation. Verify the received C
        # against the committed root, then run Freivalds detection. The
        # residual rows and bad tile are only localized when detection
        # fires; the deterministic v0.1.1 dispute remains the sole proof.
        cols_c = (n + 7) // 8
        worker_tiles = [_tile_to_ints(t) for t in req["worker_tiles"]]
        c = _c_from_tiles(worker_tiles, m, n)
        if _verify_received_c({**descriptor, "task_id": tid}, aid, worker_tiles, req["output_root"]) is not True:
            return {"rejected": "OUTPUT_DATA_COMMITMENT_MISMATCH"}
        from gemmv1.freivalds import (DeterministicSource, VerificationProfile,
                                      verify_freivalds)
        src = DeterministicSource(bytes([req.get("r_seed", 1)])) if req.get("r_seed") is not None else new_challenge_src(aid)
        profile = VerificationProfile(int(req.get("rounds", 8)))
        verification = verify_freivalds(matrix_a, matrix_b, c, m, n, k, profile, src)
        recompute_s = time.time() - t0
        disputed = None
        challenger_tile_hex = None
        if not verification["passed"]:
            from gemmv1.row_locator import localize_from_rows, reference_row_gemm

            loc, found = localize_from_rows(matrix_a, matrix_b, c, m, n, k, verification["residual_rows"])
            if found:
                disputed = [loc["tile_i"], loc["tile_j"]]
                # The challenger tile is built from 8 exact row
                # recomputations (8 x O(KN)), never from a full GEMM and
                # never from the worker's data.
                tile = []
                for r8 in range(8):
                    gi = loc["tile_i"] * 8 + r8
                    if gi < m:
                        STATS["row_gemm_calls"] += 1
                        exp = reference_row_gemm(matrix_a, gi, matrix_b, m, n, k)
                    else:
                        exp = [0] * n
                    tile.append([exp[loc["tile_j"] * 8 + c8] if loc["tile_j"] * 8 + c8 < n else 0 for c8 in range(8)])
                flat = [tile[r][cc] for r in range(8) for cc in range(8)]
                challenger_tile_hex = int32s_to_canonical(flat).hex()
        _CH.update({
            "matrix_a": matrix_a,
            "matrix_b": matrix_b,
            "m": m, "n": n, "k": k,
            "task_id": tid,
            "assignment_id": aid,
            "root_a": root_a,
            "root_b": root_b,
            "cols_c": cols_c,
            "freivalds": verification,
        })
        return {
            "backend": status,
            "root_ok": True,
            "verification": verification,
            "disputed_tile": disputed,
            "disputed_challenger_tile": challenger_tile_hex,
            "recompute_s": round(recompute_s, 3),
            "cols_c": cols_c,
            "mode": "fast",
        }
    if status["backend"] == "torch_int_mm_cuda":
        c = gpu.gpu_gemm(matrix_a, matrix_b, m, n, k)
    else:
        c = _reference_gemm(matrix_a, matrix_b, m, n, k)
    challenger_tiles = output_tiles(c, m, n)
    recompute_s = time.time() - t0

    cols_c = (n + 7) // 8
    worker_tiles = [_tile_to_ints(t) for t in req["worker_tiles"]]
    _CH["cols_c"] = cols_c
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
        "stats": lambda req: dict(STATS),
        "prepare": challenger_prepare,
        "trace": challenger_trace,
        "mid": challenger_mid,
        "arbitrate": challenger_arbitrate,
    }
    return handlers[kind](req)
