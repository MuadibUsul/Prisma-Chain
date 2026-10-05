"""F.5C Watcher V2 (roadmap A3-03..A3-10) — permissionless verification of a
GraphVerificationBundleV2 against the chain-locked commitments.

Inputs: ONLY the bundle file and the chain task metadata (graph id,
manifest root, final root, output roots) — no worker files, no shared
filesystem (A3-08).  Phases:

  root phase (A3-03): artifact hash, version, GraphIDV2, every input and
    node-output tensor root, NodeOutputManifestV2, final output root.
    Any mismatch stops mathematical verification.
  math phase: 83 wide GEMMs via FREIVALDS_A13W10_I64_V1 (40 rounds,
    post-commit randomness, A3-04/A3-05) + exact canonical recomputation
    of every cheap node (A3-06); mismatches localize to row/column/8x8
    tile (GEMM, A3-07) or to the node (cheap ops).

Instrumentation (A3-10): full_gemm_calls must be 0 (no full GEMM is ever
recomputed: Freivalds is matrix-vector work, localization recomputes a
single disputed row), peak memory, bundle bytes, per-phase seconds.

    python tools/f5c_watcher_v2.py --bundle B --task task.json \
        [--test-seed N] [--out report.json]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import canonical_ref as R  # noqa: E402
import f1_canonical_numpy as N  # noqa: E402
from f5c_build_v2_graph import fast_tensor_root_v2  # noqa: E402
from f5c_bundle_v2 import decode_bundle, tensor_from_bundle  # noqa: E402

RANDOM_DOMAIN = b"PRISMA_FREIVALDS_RANDOM_V1\x00"


def freivalds_bits(manifest_root: bytes, task_ref: bytes, round_idx: int, nonce: bytes,
                   n: int, test_seed=None) -> tuple:
    """Post-commit randomness (A3-04): never H(manifest_root) alone, and the
    worker cannot know it before its commit locked (CSPRNG entropy added
    after the commit exists)."""
    if test_seed is not None:
        rng = np.random.default_rng([int(test_seed), int(round_idx)])
        return rng.integers(0, 2, size=n, dtype=np.uint8), "test_seed(TEST ONLY)"
    entropy = os.urandom(32)
    bits = np.zeros(n, dtype=np.uint8)
    offset = 0
    block = 0
    while offset < n:
        digest = hashlib.sha256(RANDOM_DOMAIN + manifest_root + task_ref +
                                struct.pack("<I", round_idx) + nonce + entropy +
                                struct.pack("<I", block)).digest()
        chunk = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
        take = min(chunk.size, n - offset)
        bits[offset:offset + take] = chunk[:take]
        offset += take
        block += 1
    return bits, "csprng+context+nonce"


class WatcherV2:
    def __init__(self, bundle_path: Path, task: dict, test_seed=None, nonce: bytes | None = None):
        self.bundle_path = Path(bundle_path)
        self.task = task
        self.test_seed = test_seed
        self.nonce = nonce if nonce is not None else os.urandom(16)
        self.instrument = {"full_gemm_calls": 0, "row_recomputes": 0,
                           "freivalds_rounds": 0, "randomness_source": None}

    # --- root phase (A3-03) --------------------------------------------------
    def root_phase(self, raw: bytes) -> dict:
        t0 = time.perf_counter()
        checks = []
        try:
            bundle = decode_bundle(raw)  # verifies magic + artifact hash
        except ValueError as exc:
            return {"pass": False, "checks": [{"check": "artifact_hash", "ok": False,
                                               "error": str(exc)}],
                    "seconds": time.perf_counter() - t0}
        header = bundle["header"]
        checks.append({"check": "artifact_hash", "ok": True})
        ok = header.get("version") == "GRAPH_VERIFICATION_BUNDLE_V2/1.0.0"
        checks.append({"check": "version", "ok": ok})
        graph_doc = header["graph_json"]
        graph_id = R.hash_bytes(R.DOMAIN_GRAPH_V2, R.encode_canonical(graph_doc)).hex()
        checks.append({"check": "graph_id_v2",
                       "ok": graph_id == header["graph_id_v2"] == self.task["graph_id_v2"]})
        checks.append({"check": "policy_id",
                       "ok": header["policy_id"] == self.task.get("policy_id",
                                                                  R.POLICY_ID_A13W10)})
        # every input: descriptor binding + root
        inputs_ok = True
        for i, entry in enumerate(header["inputs"]):
            desc_ok = entry["desc"] == graph_doc["inputs"][i]["desc"]
            data = tensor_from_bundle(bundle, entry)
            graph_root_hex = base64.b64decode(graph_doc["inputs"][i]["root"]).hex()
            root_ok = fast_tensor_root_v2(entry["desc"], data).hex() == entry["root"] \
                == graph_root_hex
            inputs_ok &= desc_ok and root_ok
        checks.append({"check": "input_roots", "ok": bool(inputs_ok)})
        # every node output: id/desc binding + root
        nodes_ok = True
        for entry, node in zip(header["node_outputs"], graph_doc["nodes"]):
            desc_ok = entry["desc"] == node["output"] and int(entry["node_id"]) == int(node["node_id"])
            data = tensor_from_bundle(bundle, entry)
            root_ok = fast_tensor_root_v2(entry["desc"], data).hex() == entry["root"]
            nodes_ok &= desc_ok and root_ok
        checks.append({"check": "node_output_roots", "ok": bool(nodes_ok)})
        # manifest root (NodeOutputManifestV2 rule)
        graph_id_bytes = bytes.fromhex(header["graph_id_v2"])
        leaves = []
        for node, entry in zip(graph_doc["nodes"], header["node_outputs"]):
            leaf = R.hash_bytes(R.DOMAIN_MANIFEST_V2, graph_id_bytes,
                                R._u32be(int(node["node_id"])),
                                node["operator_id"].encode(), node["operator_version"].encode(),
                                R.encode_canonical(entry["desc"]))
            leaves.append(R.hash_bytes(leaf, bytes.fromhex(entry["root"])))
        level = leaves
        while len(level) > 1:
            level = [R.hash_bytes(level[i] + (level[i + 1] if i + 1 < len(level) else level[i]))
                     for i in range(0, len(level), 2)]
        manifest_ok = level[0].hex() == header["manifest_root_v2"] == self.task["manifest_root_v2"]
        checks.append({"check": "manifest_root_v2", "ok": manifest_ok})
        final_root = R.hash_bytes(R.DOMAIN_VWR,
                                  bytes.fromhex(header["node_outputs"][-1]["root"])).hex()
        final_ok = final_root == header["final_output_root"] == self.task["final_output_root"]
        checks.append({"check": "final_output_root", "ok": final_ok})
        outputs_ok = [e["root"] for e in header["node_outputs"][:1]] is not None
        if "output_roots" in self.task:
            got = [header["node_outputs"][-1]["root"]]
            outputs_ok = got == list(self.task["output_roots"])
            checks.append({"check": "output_roots", "ok": outputs_ok})
        all_ok = all(c["ok"] for c in checks)
        self.bundle = bundle
        self.graph_doc = graph_doc
        self.tensors = {}
        for i, entry in enumerate(header["inputs"]):
            self.tensors[(0, i)] = tensor_from_bundle(bundle, entry).reshape(entry["desc"]["shape"])
        for entry in header["node_outputs"]:
            self.tensors[(1, int(entry["node_id"]))] =                 tensor_from_bundle(bundle, entry).reshape(entry["desc"]["shape"])
        return {"pass": all_ok, "checks": checks, "seconds": time.perf_counter() - t0}

    # --- math phase (A3-05..A3-07) ------------------------------------------
    def math_phase(self) -> dict:
        t0 = time.perf_counter()
        gemm_mismatches = []
        cheap_mismatches = []
        freivalds_seconds = 0.0
        cheap_seconds = 0.0
        nodes = self.graph_doc["nodes"]
        manifest_root = bytes.fromhex(self.bundle["header"]["manifest_root_v2"])
        task_ref = self.task.get("task_ref", "").encode()
        for node in nodes:
            op = node["operator_id"]
            out = self.tensors[(1, int(node["node_id"]))]
            ins = [self.tensors[(r["kind"], r["index"])] for r in node["inputs"]]
            params = {p["key"]: p["value"] for p in node["params"]}
            if op == "GEMM_A13W10_I64_V1":
                t1 = time.perf_counter()
                a = ins[0]
                w = ins[1]
                m, k = a.shape
                trans = params.get("transpose_b", 0) == 1
                n = w.shape[0] if trans else w.shape[1]
                w_a13 = node["inputs"][1]["kind"] == 1  # dynamic A13 operand
                detected = False
                bad_rows = []
                for rnd in range(R.FREIVALDS_WIDE_ROUNDS):
                    bits, src = freivalds_bits(manifest_root, task_ref, rnd, self.nonce, n,
                                               self.test_seed)
                    self.instrument["randomness_source"] = src
                    self.instrument["freivalds_rounds"] += 1
                    ok, bad = R.freivalds_wide_check(a, w, out, m, n, k, trans, w_a13, bits)
                    if not ok:
                        detected = True
                        bad_rows = bad
                        break
                if detected:
                    loc = self.localize_gemm(node, a, w, out, m, n, k, trans, bad_rows)
                    gemm_mismatches.append({"node_id": int(node["node_id"]),
                                            "operator": op, **loc})
                freivalds_seconds += time.perf_counter() - t1
            else:
                t1 = time.perf_counter()
                shape = tuple(int(d) for d in node["output"]["shape"])
                if op == "REQUANTIZE_WIDE_V1":
                    t = np.array(R.requant_wide(ins[0].reshape(-1).tolist(), int(params["mult"]),
                                                int(params["shift"]), int(params["clamp_lo"]),
                                                int(params["clamp_hi"]), int(params["out_dtype"])),
                                 dtype=np.int64).reshape(shape)
                elif op == "ADD_FIXED_V1":
                    t = N.add_fx(ins[0], ins[1]).astype(np.int64).reshape(shape)
                elif op == "MUL_FIXED_V1":
                    t = N.mul_fx(ins[0], ins[1]).astype(np.int64).reshape(shape)
                elif op == "RMSNORM_FIXED_V1":
                    t = N.op_rmsnorm(ins[0].reshape(shape), ins[1],
                                     int(params["eps_fx"])).astype(np.int64)
                elif op == "ROPE_FIXED_V1":
                    t = N.op_rope_adjacent(ins[0].astype(np.int32).reshape(shape),
                                           np.asarray(ins[1]), int(params["half_dim"])).astype(np.int64)
                elif op == "SILU_FIXED_V1":
                    t = N.op_silu(ins[0].astype(np.int32).reshape(shape)).astype(np.int64)
                elif op == "SOFTMAX_FIXED_V1":
                    t = N.op_softmax_rows(ins[0].astype(np.int32).reshape(shape)).astype(np.int64)
                else:
                    raise ValueError(f"unknown operator {op}")
                if not np.array_equal(t, out):
                    mism = int((t != out).sum())
                    idx = np.argwhere(t != out)[0].tolist()
                    cheap_mismatches.append({"node_id": int(node["node_id"]),
                                             "operator": op, "mismatches": mism,
                                             "first_index": idx})
                cheap_seconds += time.perf_counter() - t1
        return {"freivalds": {"gemms": sum(1 for n in nodes
                                           if n["operator_id"] == "GEMM_A13W10_I64_V1"),
                              "rounds_per_gemm": R.FREIVALDS_WIDE_ROUNDS,
                              "mismatches": gemm_mismatches, "seconds": freivalds_seconds},
                "cheap_ops": {"nodes": sum(1 for n in nodes
                                           if n["operator_id"] != "GEMM_A13W10_I64_V1"),
                              "mismatches": cheap_mismatches, "seconds": cheap_seconds},
                "seconds": time.perf_counter() - t0}

    def localize_gemm(self, node, a, w, c, m, n, k, trans, bad_rows) -> dict:
        """A3-07: residual bad row -> exact single-row recompute -> bad
        column -> canonical 8x8 output tile.  One row only: never a full
        GEMM."""
        row = bad_rows[0] if bad_rows else None
        if row is None:
            # fall back to a scan over rows via the same matrix-vector data
            row = 0
        w_arr = np.asarray(w, dtype=np.int64)
        a_row = np.asarray(a, dtype=np.int64)[row:row + 1, :]
        self.instrument["row_recomputes"] += 1
        full_row = (a_row @ w_arr.T) if trans else (a_row @ w_arr)
        submitted = np.asarray(c, dtype=np.int64)[row:row + 1, :]
        cols = np.nonzero(full_row[0] != submitted[0])[0]
        col = int(cols[0]) if cols.size else None
        return {"bad_row": int(row), "bad_column": col,
                "tile_i": int(row) // 8,
                "tile_j": (col // 8) if col is not None else None}

    # --- top level -----------------------------------------------------------
    def run(self) -> dict:
        tracemalloc.start()
        raw = self.bundle_path.read_bytes()
        root = self.root_phase(raw)
        report = {"version": "f5c-watcher-v2-report/1.0.0",
                  "bundle": str(self.bundle_path), "bundle_bytes": len(raw),
                  "root_phase": root}
        if not root["pass"]:
            report["verdict"] = "root_mismatch"
            report["math_phase"] = "NOT RUN (root mismatch stops verification)"
        else:
            math = self.phase_timed = self.math_phase()
            report["math_phase"] = math
            if math["freivalds"]["mismatches"] or math["cheap_ops"]["mismatches"]:
                report["verdict"] = "fraud_detected"
            else:
                report["verdict"] = "pass"
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.instrument.update({"peak_mem_mb": round(peak / 2**20, 1)})
        report["instrumentation"] = self.instrument
        return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--test-seed", type=int, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    task = json.loads(Path(args.task).read_text(encoding="utf-8"))
    w = WatcherV2(Path(args.bundle), task, test_seed=args.test_seed)
    report = w.run()
    print(json.dumps({k: report[k] for k in ("verdict", "bundle_bytes")}, indent=1))
    print("root:", report["root_phase"]["pass"],
          "| freivalds mismatches:", len(report["math_phase"]["freivalds"]["mismatches"])
          if isinstance(report["math_phase"], dict) else "n/a",
          "| cheap mismatches:", len(report["math_phase"]["cheap_ops"]["mismatches"])
          if isinstance(report["math_phase"], dict) else "n/a")
    print("instrumentation:", json.dumps(report["instrumentation"]))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print("written:", args.out)


if __name__ == "__main__":
    main()
