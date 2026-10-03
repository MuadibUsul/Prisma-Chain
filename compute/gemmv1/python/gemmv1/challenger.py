"""Challenger client for the two-node GEMM E2E.

Independently recomputes the task, compares output tiles against the
worker's ResultCommit, and on a mismatch drives the full dispute: lock both
on-demand tile traces, bisect through /mid queries, arbitrate the final
8x8x8 micro-step and print the outcome.

Run:
  python -m gemmv1.challenger --worker http://127.0.0.1:8301 \
      --m 128 --n 128 --k 256 --seed 42 [--inject-fraud 0,1]
"""

import argparse
import json
import sys
import time
import urllib.request

from .protocol import (
    build_matrix_roots,
    leaf_output_tile,
    leaf_trace_state,
    merkle_root,
    task_id,
)
from . import gpu
from .merkle_proofs import verify_inclusion
from .tensors import (
    INT_TILE_BYTES,
    canonical_to_int32s,
    extract_a_tile,
    extract_b_tile,
    int32s_to_canonical,
    micro_step,
    output_tiles,
    reference_gemm,
)
from .testgen import gen_test_matrix
from .trace import build_tile_trace, coordinator_verify_trace


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def _bytes_of(values) -> bytes:
    return bytes(v & 0xFF for v in values)


def run(worker: str, m: int, n: int, k: int, seed: int, inject_fraud) -> dict:
    matrix_a = gen_test_matrix(ord("A"), seed, m * k)
    matrix_b = gen_test_matrix(ord("B"), seed, k * n)
    root_a, root_b, cols_a, _, _, _ = build_matrix_roots(matrix_a, matrix_b, m, n, k)

    print(f"[challenger] executing on worker {worker}")
    body = {"m": m, "n": n, "k": k, "seed": seed}
    if inject_fraud is not None:
        body["inject_fraud"] = list(inject_fraud)
        print(f"[challenger] DEVELOPMENT: asking worker to inject fraud into tile {tuple(inject_fraud)}")
    result = post(worker + "/execute", body)
    print(f"[challenger] worker backend: {result['backend']}")
    commit = result["result_commit"]

    # Independent recomputation (full verification allowed in v0.1.1).
    started = time.time()
    backend = result["backend"]
    if backend["backend"] == "torch_int_mm_cuda":
        c = gpu.gpu_gemm(matrix_a, matrix_b, m, n, k)
    else:
        c = reference_gemm(matrix_a, matrix_b, m, n, k)
    cols_c = (n + 7) // 8
    challenger_tiles = output_tiles(c, m, n)

    task_hex = commit["task_id"]
    assignment_hex = commit["assignment_id"]
    task = bytes.fromhex(task_hex)
    assignment = bytes.fromhex(assignment_hex)

    # The submitted tiles must hash to the committed root.
    worker_tiles_flat = result["output_tiles"]["tiles"]
    leaves = [
        leaf_output_tile(task, assignment, idx // cols_c, idx % cols_c, int32s_to_canonical(tile))
        for idx, tile in enumerate(worker_tiles_flat)
    ]
    if merkle_root(leaves).hex() != commit["output_root"]:
        raise SystemExit("FAIL: submitted tiles do not hash to the committed output_root")

    # Find a differing tile.
    disputed = None
    for idx, (w, ch) in enumerate(zip(worker_tiles_flat, challenger_tiles)):
        if w != ch:
            disputed = (idx // cols_c, idx % cols_c)
            break
    detection = time.time() - started

    if disputed is None:
        report = {
            "outcome": "optimistic_unchallenged",
            "detection_s": round(detection, 3),
            "output_root": commit["output_root"],
            "canonical_mac_count": m * n * k,
        }
        print(json.dumps(report, indent=2))
        print(f"[challenger] PASS: outputs identical ({detection:.1f}s); outcome=optimistic_unchallenged")
        return report

    tile_i, tile_j = disputed
    print(f"[challenger] fraud detected at tile ({tile_i},{tile_j}) after {detection:.1f}s; opening dispute")
    idx = tile_i * cols_c + tile_j
    worker_tile = worker_tiles_flat[idx]
    challenger_tile = challenger_tiles[idx]

    # Worker tile membership in the committed root was implicitly proven by
    # the root check above; lock both traces.
    worker_trace = post(worker + "/trace", {"tile_i": tile_i, "tile_j": tile_j})
    r_steps = (k + 7) // 8
    worker_root = coordinator_verify_trace(task, assignment, tile_i, tile_j, r_steps, {
        "disputed_tile_i": tile_i, "disputed_tile_j": tile_j,
        "trace_root": worker_trace["trace_root"],
        "initial_state": worker_trace["states"][0],
        "initial_proof": worker_trace["proofs"][0],
        "final_state": worker_trace["states"][-1],
        "final_proof": worker_trace["proofs"][-1],
    })
    # Worker S_R must equal the committed tile.
    if bytes.fromhex(worker_trace["states"][-1]) != int32s_to_canonical(worker_tile):
        raise SystemExit("FAIL: worker trace S_R does not match its committed tile")

    my_states, my_levels, my_root = build_tile_trace(
        matrix_a, matrix_b, m, n, k, task, assignment, tile_i, tile_j, micro_step
    )
    if int32s_to_canonical(my_states[-1]) != int32s_to_canonical(challenger_tile):
        raise SystemExit("FAIL: challenger trace S_R does not match its own tile")

    # Bisection over [0, R].
    low, high = 0, r_steps
    low_state = [0] * 64
    worker_high = canonical_to_int32s(bytes.fromhex(worker_trace["states"][-1]))
    challenger_high = challenger_tile
    rounds = 0
    while high - low > 1:
        mid = low + (high - low) // 2
        w = post(worker + "/mid", {"step": mid, "tile_i": tile_i, "tile_j": tile_j})
        w_state = bytes.fromhex(w["state"])
        p = w["proof"]
        w_leaf = leaf_trace_state(task, assignment, tile_i, tile_j, mid, w_state)
        if not verify_inclusion(worker_root, w_leaf, p["index"], p["count"], [bytes.fromhex(s) for s in p["siblings"]]):
            raise SystemExit("FAIL: worker midpoint proof rejected")
        w_vals = canonical_to_int32s(w_state)
        c_vals = my_states[mid]
        if w_vals == c_vals:
            low, low_state = mid, c_vals
        else:
            high = mid
            worker_high, challenger_high = w_vals, c_vals
        rounds += 1
        if rounds > 64:
            raise SystemExit("FAIL: bisection did not converge")

    # Arbitration: one 8x8x8 micro-step, 512 canonical MACs.
    step = low
    a_tile = extract_a_tile(matrix_a, m, k, tile_i, step)
    b_tile = extract_b_tile(matrix_b, k, n, step, tile_j)
    expected = micro_step(low_state, a_tile, b_tile)
    worker_ok = expected == worker_high
    challenger_ok = expected == challenger_high
    if worker_ok:
        outcome = "worker_wins"
    elif challenger_ok:
        outcome = "challenger_wins"
    else:
        outcome = "both_invalid"

    report = {
        "outcome": outcome,
        "disputed_tile": [tile_i, tile_j],
        "bisection_rounds": rounds,
        "arbitration_step": step,
        "arbitration_macs": 512,
        "worker_trace_root": worker_root.hex(),
        "challenger_trace_root": my_root.hex(),
        "on_demand_trace_bytes": (r_steps + 1) * INT_TILE_BYTES,
        "detection_s": round(detection, 3),
    }
    print(json.dumps(report, indent=2))
    if outcome == "challenger_wins":
        print("PASS: fraud confirmed; ChallengerWins; the worker receives NO Verified Work Receipt")
    elif outcome == "worker_wins":
        print("PASS: false challenge; WorkerWins")
    else:
        print("FAIL: both invalid; refund path")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="http://127.0.0.1:8301")
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inject-fraud", default="")
    args = ap.parse_args()
    fraud = None
    if args.inject_fraud:
        i, j = args.inject_fraud.split(",")
        fraud = (int(i), int(j))
    report = run(args.worker, args.m, args.n, args.k, args.seed, fraud)
    if report["outcome"] == "both_invalid":
        sys.exit(1)


if __name__ == "__main__":
    main()
