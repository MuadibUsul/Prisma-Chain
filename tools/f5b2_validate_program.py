"""EXPERIMENTAL / NON-PROTOCOL — F.5B.2 program validation + adversarial vectors.

1. Interprets testdata/f5b2_block_program.json over testdata/
   f5b2_block_bundle.npz with the canonical numpy kernels (the SAME
   semantics the GPU kernels port) and asserts the result reproduces
   testdata/f5b2_cpu_manifest.npz node-for-node (310/310) and every
   f5b2.wide.v1 root.  This validates that the committed program is
   self-contained and executable BEFORE any pod time is spent; the
   F.5B.2 verdict itself remains GPU-vs-frozen-CPU-manifest.

2. Generates testdata/f5b2_adversarial_vectors.json for the operator
   gate (spec section 34): min/max int32, near-saturation, tie
   rounding, all-equal rows, one dominant logit, exp underflow
   boundary, division ties — expected outputs from the canonical numpy
   reference.

    python tools/f5b2_validate_program.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import f1_canonical_numpy as N1  # noqa: E402
from f5a_joint_precision import gemm64  # noqa: E402
from f5b2_dump_block_program import wide_node_root  # noqa: E402

ONE = 1 << 20
MIN_FX, MAX_FX = -(1 << 31), (1 << 31) - 1


def requant(x, mult, shift, lo, hi):
    return np.clip(N1.rshift_round_even(np.asarray(x, np.int64) * np.int64(mult), shift), lo, hi)


def run_program(program: dict, bundle) -> list[np.ndarray]:
    consts = {c["id"]: bundle[f"const_{c['id']}"] for c in program["consts"]}
    tensors: list[np.ndarray] = [None] * len(program["nodes"])
    for n in program["nodes"]:
        ins = [consts[r["index"]] if r["kind"] == "const" else tensors[r["index"]]
               for r in n["inputs"]]
        op, params, shape = n["op"], n["params"], tuple(n["shape"])
        if op == "GEMM":
            out = gemm64(np.asarray(ins[0], np.int64), np.asarray(ins[1], np.int64),
                         bool(params["transpose_b"]))
        elif op == "REQUANTIZE":
            out = requant(ins[0], params["mult"], params["shift"], params["lo"], params["hi"])
        elif op == "ADD":
            out = N1.add_fx(ins[0], ins[1]).astype(np.int64)
        elif op == "MUL":
            out = N1.mul_fx(ins[0], ins[1]).astype(np.int64)
        elif op == "RMSNORM":
            out = N1.op_rmsnorm(np.asarray(ins[0]).reshape(shape), ins[1],
                                params["eps_fx"]).astype(np.int64)
        elif op == "ROPE":
            out = N1.op_rope_adjacent(np.asarray(ins[0]).astype(np.int32).reshape(shape),
                                      np.asarray(ins[1]), params["half_dim"]).astype(np.int64)
        elif op == "SILU":
            out = N1.op_silu(np.asarray(ins[0]).astype(np.int32).reshape(shape)).astype(np.int64)
        elif op == "SOFTMAX":
            out = N1.op_softmax_rows(np.asarray(ins[0]).astype(np.int32).reshape(shape)
                                     ).astype(np.int64)
        else:
            raise ValueError(op)
        tensors[n["seq"]] = np.asarray(out, dtype=np.int64).reshape(shape)
    return tensors


def validate() -> bool:
    program = json.loads((REPO / "testdata" / "f5b2_block_program.json")
                         .read_text(encoding="utf-8"))
    bundle = np.load(REPO / "testdata" / "f5b2_block_bundle.npz")
    manifest = np.load(REPO / "testdata" / "f5b2_cpu_manifest.npz")
    roots = json.loads((REPO / "testdata" / "f5b2_cpu_roots.json").read_text(encoding="utf-8"))
    tensors = run_program(program, bundle)
    ok = True
    bad_nodes = []
    for n in program["nodes"]:
        got = tensors[n["seq"]]
        want = manifest[f"out_{n['seq']}"].reshape(got.shape)
        if not np.array_equal(got, want):
            ok = False
            bad_nodes.append({"seq": n["seq"], "op": n["op"],
                              "mismatches": int((got != want).sum()),
                              "first": np.argwhere(got != want)[0].tolist()})
        root = wide_node_root(got)
        if root != roots["nodes"][n["seq"]]["root"]:
            ok = False
            bad_nodes.append({"seq": n["seq"], "op": n["op"], "root_mismatch": True})
    print(f"program validation: {'PASS' if ok else 'FAIL'} "
          f"({len(program['nodes']) - len(bad_nodes)}/{len(program['nodes'])} exact)")
    for b in bad_nodes[:10]:
        print("  ", b)
    return ok


# --- adversarial vectors -----------------------------------------------------


def gen_adversarial() -> dict:
    rng = np.random.default_rng(20261005)
    vectors = []

    def add_vec(op, name, params, ins, out):
        vectors.append({"op": op, "name": name, "params": params,
                        "inputs": [{"shape": list(a.shape), "data": a.reshape(-1).tolist()}
                                   for a in ins],
                        "expected": np.asarray(out).reshape(-1).tolist()})

    # RMSNORM: extremes, all-equal row, near-saturation magnitudes
    w = np.rint(rng.uniform(0.5, 1.5, 64) * ONE).astype(np.int64)
    cases = {
        "all_equal": np.full((4, 64), 12345, dtype=np.int64),
        "min_int32": np.full((2, 64), MIN_FX, dtype=np.int64),
        "max_int32": np.full((2, 64), MAX_FX, dtype=np.int64),
        "mixed_extremes": np.where(np.arange(2 * 64).reshape(2, 64) % 2 == 0,
                                   MIN_FX, MAX_FX).astype(np.int64),
        "zeros_row": np.zeros((2, 64), dtype=np.int64),
    }
    for name, x in cases.items():
        add_vec("RMSNORM", f"rmsnorm_{name}", {"eps_fx": 1}, [x, w],
                N1.op_rmsnorm(x, w, 1))
    # REQUANTIZE: tie rounding (exact halves), shift boundaries, saturation
    acc = np.array([0, 1, -1, (1 << 19), -(1 << 19), (1 << 19) + 1, MAX_FX, MIN_FX],
                   dtype=np.int64)
    add_vec("REQUANTIZE", "requant_ties_shift20", {"mult": 1 << 20, "shift": 20,
                                                   "lo": MIN_FX, "hi": MAX_FX},
            [acc], requant(acc, 1 << 20, 20, MIN_FX, MAX_FX))
    acc2 = np.array([0, 1, -1, (1 << 29), -(1 << 29), (1 << 29) + 1, (1 << 29) - 1],
                    dtype=np.int64)
    add_vec("REQUANTIZE", "requant_ties_shift30", {"mult": 3, "shift": 30,
                                                   "lo": -4095, "hi": 4095},
            [acc2], requant(acc2, 3, 30, -4095, 4095))
    acc3 = np.array([MAX_FX, MIN_FX, MAX_FX // 2, MIN_FX // 2, 1 << 32, -(1 << 32)],
                    dtype=np.int64)
    add_vec("REQUANTIZE", "requant_clamp_saturation", {"mult": 1 << 30, "shift": 20,
                                                       "lo": -100, "hi": 100},
            [acc3], requant(acc3, 1 << 30, 20, -100, 100))
    # SOFTMAX: one dominant logit, all-equal, very negative (exp underflow),
    # min/max rows
    rows = {
        "one_dominant": np.where(np.arange(16).reshape(4, 4) % 4 == 0,
                                 20 * ONE, -30 * ONE).astype(np.int32),
        "all_equal": np.full((4, 4), 7 * ONE, dtype=np.int32),
        "very_negative": np.full((2, 4), -25 * ONE, dtype=np.int32),
        "min_int32": np.full((2, 4), MIN_FX, dtype=np.int32),
        "mixed": np.array([[0, -24 * ONE, 21 * ONE, 1], [MIN_FX, MAX_FX, 0, -1]],
                          dtype=np.int32),
    }
    for name, x in rows.items():
        add_vec("SOFTMAX", f"softmax_{name}", {}, [x], N1.op_softmax_rows(x))
    # SILU: exp branch boundaries (-24*ONE underflow, +21*ONE saturation, 0)
    sx = np.array([[0, ONE, -ONE, 21 * ONE], [-21 * ONE, -24 * ONE, -24 * ONE - 1, MAX_FX]],
                  dtype=np.int32)
    add_vec("SILU", "silu_branches", {}, [sx], N1.op_silu(sx))
    # MUL: tie products and extremes
    mx = np.array([[1 << 20, -(1 << 20), (1 << 20) + 1, 3], [MIN_FX // (1 << 20), MAX_FX // (1 << 20), 0, -3]],
                  dtype=np.int64)
    add_vec("MUL", "mul_ties", {}, [mx, mx], N1.mul_fx(mx, mx))
    # ADD: saturating extremes
    ax = np.array([MAX_FX, MIN_FX, MAX_FX - 1, MIN_FX + 1, 0], dtype=np.int64)
    add_vec("ADD", "add_saturation", {}, [ax, ax], N1.add_fx(ax, ax))
    add_vec("ADD", "add_neg_saturation", {}, [ax, -ax if False else -ax], N1.add_fx(ax, -ax))
    return {"version": "f5b2-adversarial-v1", "vectors": vectors}


def main() -> None:
    ok = validate()
    adv = gen_adversarial()
    (REPO / "testdata" / "f5b2_adversarial_vectors.json").write_text(
        json.dumps(adv, indent=1) + "\n", encoding="utf-8")
    print("adversarial vectors:", len(adv["vectors"]))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
