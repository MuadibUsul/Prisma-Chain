"""Permissionless graph watcher (Phase F.1).

Verifies a committed real-model block WITHOUT fully recomputing any GEMM:

  * stage 1 — commitments: GraphID, node-output manifest (every node root
    proven into the manifest root), final output root, input roots;
  * stage 2 — per node: GEMM nodes by Freivalds over Z_p with randomness
    derived from the COMMITTED manifest root (post-commitment), all other
    operators by exact canonical recompute;
  * instrumentation: any call into a full reference GEMM fails the run.

The watcher never reads the worker's filesystem: everything comes from
the verification bundle (which the DA layer stores). Detection is not a
slashing proof; a mismatch escalates through the graph dispute.

    python tools/graph_watcher.py --bundle <dir> [--out docs/...json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import canonical_ref as R  # noqa: E402
import f1_canonical_numpy as N  # noqa: E402

FREIVALDS_PRIME = (1 << 31) - 1          # 2^31-1 is prime
FREIVALDS_ROUNDS = 2
FULL_GEMM_CALLS = 0                       # instrumentation counter


class WatcherFailure(Exception):
    """A commitment or semantic mismatch (not a crash)."""

    def __init__(self, message: str, *, node: int | None = None, operator: str = "",
                 chunk_index: int | None = None):
        super().__init__(message)
        self.node = node
        self.operator = operator
        self.chunk_index = chunk_index


def csprng_vector(manifest_root: bytes, node_id: int, round_index: int, size: int) -> np.ndarray:
    """Deterministic randomness derived ONLY from committed values.

    Every watcher derives the same vector, and it exists only after the
    manifest is locked, so a worker can never adapt its output to it.
    """
    seed = hashlib.sha256(b"PRISMA_F1_FREIVALDS_V1\x00" + manifest_root +
                          node_id.to_bytes(8, "big") + round_index.to_bytes(4, "big")).digest()
    out = np.empty(size, dtype=np.int64)
    counter = 0
    while counter * 32 < size * 8:
        block = hashlib.sha512(seed + counter.to_bytes(4, "big")).digest()
        for i in range(0, 32, 4):
            if counter * 32 + i >= size * 8:
                break
            value = int.from_bytes(block[i:i + 4], "big")
            idx = (counter * 32 + i) // 8
            if idx < size:
                out[idx] = 1 + (value % (FREIVALDS_PRIME - 1))
        counter += 1
    return out


def freivalds_check(a: np.ndarray, b: np.ndarray, c: np.ndarray, transpose_b: bool,
                    manifest_root: bytes, node_id: int) -> int:
    """Return the number of rounds run; raise on mismatch. Never runs a
    full GEMM: only two matrix-vector products per round."""
    m, k = a.shape
    n = b.shape[0] if transpose_b else b.shape[1]
    assert c.shape == (m, n), "claimed output shape mismatch"
    rounds = 0
    for round_index in range(FREIVALDS_ROUNDS):
        r = csprng_vector(manifest_root, node_id, round_index, n)
        # u = B^T r  (transpose_b: B is [N,K]; else B is [K,N])
        u = ((b.astype(np.int64) * r[:, None]) % FREIVALDS_PRIME).sum(axis=0) % FREIVALDS_PRIME \
            if transpose_b else (b.astype(np.int64).T @ r) % FREIVALDS_PRIME
        v = (a.astype(np.int64) @ u) % FREIVALDS_PRIME
        w = (c.astype(np.int64) @ r) % FREIVALDS_PRIME
        if not np.array_equal(v, w):
            raise WatcherFailure("freivalds mismatch", node=node_id)
        rounds += 1
    return rounds


def verify_bundle(bundle_dir: Path, expected: dict) -> dict:
    """Run the full verification; expected carries the chain-committed
    values (graph_id, manifest root, final output root)."""
    global FULL_GEMM_CALLS
    started = time.time()
    graph = json.loads((bundle_dir / "graph.json").read_text(encoding="utf-8"))
    index = json.loads((bundle_dir / "index.json").read_text(encoding="utf-8"))

    # --- stage 1: commitments -------------------------------------------
    snake = _to_snake(graph)
    for ginput, meta in zip(snake["inputs"], index["inputs"]):
        ginput["root"] = bytes.fromhex(meta["root"])
    graph_id = R.graph_id(snake)
    if graph_id.hex() != expected["graph_id"]:
        raise WatcherError("graph id mismatch")
    manifest_root = bytes.fromhex(expected["manifest_root"])
    node_roots = [bytes.fromhex(h) for h in index["node_roots"]]
    computed_manifest, leaves = R.build_node_output_manifest(graph_id, snake["nodes"], node_roots)
    if computed_manifest != manifest_root:
        raise WatcherError("manifest root mismatch")
    # every node output leaf proven into the manifest
    levels = R.build_levels(leaves)
    for i in range(len(snake["nodes"])):
        proof = R.prove(levels, i)
        if not R.verify_inclusion(manifest_root, leaves[i], i, len(leaves), proof):
            raise WatcherError(f"manifest proof failed for node {i}")
    outputs = [node_roots[ref["index"]] for ref in snake["outputs"]]
    if R.final_output_root(outputs).hex() != expected["final_output_root"]:
        raise WatcherError("final output root mismatch")
    commit_time = time.time() - started

    # --- tensor map -------------------------------------------------------
    inputs = {}
    for i, meta in enumerate(index["inputs"]):
        inputs[i] = np.load(bundle_dir / meta["file"]).astype(np.int32)
    tensors = {(0, i): t for i, t in inputs.items()}
    for i, meta in enumerate(index["node_outputs"]):
        tensors[(1, i)] = np.load(bundle_dir / meta["file"]).astype(np.int32)

    # --- stage 2: per-node verification ----------------------------------
    stats = {"nodes": len(snake["nodes"]), "gemm_nodes": 0, "freivalds_rounds": 0,
             "exact_add": 0, "exact_mul": 0, "exact_requantize": 0, "exact_rmsnorm": 0,
             "exact_rope": 0, "exact_silu": 0, "exact_softmax": 0}
    verify_started = time.time()
    for node in snake["nodes"]:
        op = node["operator_id"]
        params = {k: v for k, v in node["params"]}
        ins = [tensors[(r["kind"], r["index"])] for r in node["inputs"]]
        claimed = tensors[(1, node["node_id"])]
        if op == R.OP_GEMM:
            stats["gemm_nodes"] += 1
            stats["freivalds_rounds"] += freivalds_check(
                ins[0], ins[1], claimed, params.get("transpose_b", 0) == 1,
                manifest_root, node["node_id"])
            continue
        if op == R.OP_ADD:
            got = N.op_add(ins[0], ins[1]); stats["exact_add"] += 1
        elif op == R.OP_MUL:
            got = N.op_mul(ins[0], ins[1]); stats["exact_mul"] += 1
        elif op == R.OP_REQUANTIZE:
            got = N.op_requantize(ins[0], params["mult"], params["shift"],
                                  params.get("clamp_lo", N.MIN_FX), params.get("clamp_hi", N.MAX_FX))
            stats["exact_requantize"] += 1
        elif op == R.OP_RMSNORM:
            shape = tuple(int(d) for d in node["output"]["shape"])
            got = N.op_rmsnorm(ins[0].reshape(shape), ins[1], params["eps_fx"]); stats["exact_rmsnorm"] += 1
        elif op == R.OP_ROPE:
            shape = tuple(int(d) for d in node["output"]["shape"])
            table = tensors[(0, node["inputs"][-1]["index"])]
            got = N.op_rope_adjacent(ins[0].reshape(-1, shape[-1]), table, shape[-1] // 2)
            stats["exact_rope"] += 1
        elif op == R.OP_SILU:
            got = N.op_silu(ins[0]); stats["exact_silu"] += 1
        elif op == R.OP_SOFTMAX:
            shape = tuple(int(d) for d in node["output"]["shape"])
            got = N.op_softmax_rows(ins[0].reshape(shape)); stats["exact_softmax"] += 1
        else:
            raise WatcherError(f"unknown operator {op}", node=node["node_id"], operator=op)
        got = np.asarray(got, dtype=np.int32).reshape(claimed.shape)
        diff = np.flatnonzero(got.reshape(-1) != claimed.reshape(-1))
        if diff.size:
            raise WatcherFailure(
                f"exact mismatch on node {node['node_id']} ({op})",
                node=node["node_id"], operator=op, chunk_index=int(diff[0]) // 64)
    verify_time = time.time() - verify_started
    if FULL_GEMM_CALLS != 0:
        raise WatcherError("instrumentation: a full GEMM was executed")
    return {
        "status": "PASS",
        "commitment_seconds": commit_time,
        "verification_seconds": verify_time,
        "full_gemm_calls": FULL_GEMM_CALLS,
        "stats": stats,
        "graph_id": graph_id.hex(),
        "manifest_root": manifest_root.hex(),
    }


class WatcherError(Exception):
    pass


def _to_snake(graph_go: dict) -> dict:
    def desc(d):
        return {"dtype": d["Dtype"], "layout": 1, "shape": list(d["Shape"])}
    return {
        "protocol_version": graph_go["ProtocolVersion"],
        "spec": graph_go["Spec"],
        "inputs": [{"name": i["Name"], "desc": desc(i["Desc"]), "root": i["Root"]}
                   for i in graph_go["Inputs"]],
        "nodes": [{
            "node_id": n["NodeID"], "operator_id": n["OperatorID"],
            "operator_version": n["Version"],
            "inputs": [{"kind": r["Kind"], "index": r["Index"]} for r in n["Inputs"]],
            "output": desc(n["Output"]),
            "params": [[p["Key"], p["Value"]] for p in n["Params"]],
        } for n in graph_go["Nodes"]],
        "outputs": [{"kind": r["Kind"], "index": r["Index"]} for r in graph_go["Outputs"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--expected", required=True, help="json with graph_id/manifest_root/final_output_root")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    bundle = Path(args.bundle)
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    started = time.time()
    try:
        report = verify_bundle(bundle, expected)
    except WatcherFailure as exc:
        report = {"status": "FRAUD_DETECTED", "first_suspected_node": exc.node,
                  "operator_id": exc.operator, "bad_chunk_index": exc.chunk_index,
                  "reason": str(exc)}
    except (WatcherError, ValueError) as exc:
        report = {"status": "COMMITMENT_MISMATCH", "reason": str(exc)}
    report["wall_seconds"] = time.time() - started
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
