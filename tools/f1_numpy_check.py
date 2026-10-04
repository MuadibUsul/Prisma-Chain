"""Bit-exactness check: the vectorized numpy executor must reproduce the
scalar canonical mirror (and therefore Go) byte for byte.

    python tools/f1_numpy_check.py

Covers the frozen math primitives on dense samples and every operator
vector in compute/canonical/testdata/canonical_vectors.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))
sys.path.insert(0, str(REPO / "tools"))

import canonical_ref as R  # noqa: E402
import f1_canonical_numpy as N  # noqa: E402


def check_primitives() -> None:
    rng = np.random.default_rng(7)
    samples = np.concatenate([
        rng.integers(-(2**31), 2**31, size=20000, dtype=np.int64),
        rng.integers(-(2**20), 2**20, size=20000, dtype=np.int64),
        np.array([0, 1, -1, 2**20, -(2**20), 2**31 - 1, -(2**31)], dtype=np.int64),
    ])
    for i, v in enumerate(samples):
        want = R.saturate(int(v))
        got = int(N.saturate(np.array([v]))[0])
        assert got == want, f"saturate {v}: {got} != {want}"
    shifts = rng.integers(1, 62, size=5000)
    for i, s in enumerate(shifts):
        v = int(rng.integers(-(2**62), 2**62))
        want = R.rshift_round_even(v, int(s))
        got = int(N.rshift_round_even(np.array([v], dtype=np.int64), int(s))[0])
        assert got == want, f"rshift {v} >> {s}: {got} != {want}"
    xs = np.concatenate([
        rng.integers(-(26 << 20), 26 << 20, size=20000),
        np.array([0, 1, -1, R.LN2_FX, -R.LN2_FX, 21 << 20, 20 << 20, -(24 << 20)], dtype=np.int64),
    ])
    for x in xs:
        want = R.exp_fx(int(x))
        got = int(N.exp_fx(np.array([x], dtype=np.int32))[0])
        assert got == want, f"exp {x}: {got} != {want}"
        want_s = R.sigmoid_fx(int(x))
        got_s = int(N.sigmoid_fx(np.array([x], dtype=np.int32))[0])
        assert got_s == want_s, f"sigmoid {x}: {got_s} != {want_s}"
    inv_x = np.concatenate([
        rng.integers(1, 2**62, size=20000),
        np.array([1, 2, 3, (1 << 62) - 1, (1 << 62), 2**40], dtype=np.int64),
    ])
    for x in inv_x:
        want = R.inv_sqrt_fx(int(x))
        got = int(N.inv_sqrt_fx(np.array([x], dtype=np.int64))[0])
        assert got == want, f"invsqrt {x}: {got} != {want}"
    print(f"primitives OK ({len(samples)}+{len(shifts)}+{len(xs)}+{len(inv_x)} samples)")


def check_operator_vectors() -> None:
    raw = json.loads((REPO / "compute" / "canonical" / "testdata" / "canonical_vectors.json")
                     .read_text(encoding="utf-8"))
    checked = 0
    for vec in raw["operators"]:
        name = vec["name"]
        op = vec["operator_id"]
        params = {k: v for k, v in vec["params"]}

        def arr(item):
            desc = item["desc"]
            data = np.array(item["data"], dtype=np.int32)
            return desc, data

        descs = [arr(i)[0] for i in vec["inputs"]]
        datas = [arr(i)[1] for i in vec["inputs"]]
        if op == R.OP_ADD:
            out = N.op_add(datas[0], datas[1])
        elif op == R.OP_MUL:
            out = N.op_mul(datas[0], datas[1])
        elif op == R.OP_REQUANTIZE:
            out = N.op_requantize(datas[0], params["mult"], params["shift"],
                                  params["clamp_lo"], params["clamp_hi"])
        elif op == R.OP_RMSNORM:
            out = N.op_rmsnorm(datas[0].reshape(descs[0]["shape"]), datas[1], params["eps_fx"])
            out = out.reshape(-1)
        elif op == R.OP_SILU:
            out = N.op_silu(datas[0])
        elif op == R.OP_SOFTMAX:
            out = N.op_softmax_rows(datas[0].reshape(descs[0]["shape"])).reshape(-1)
        elif op == R.OP_ROPE:
            hidden = descs[0]["shape"][-1]
            out = N.op_rope_adjacent(datas[0].reshape(-1, hidden), datas[1], hidden // 2).reshape(-1)
        elif op == R.OP_GEMM:
            trans = params.get("transpose_b", 0) == 1
            out = N.op_gemm(datas[0].reshape(descs[0]["shape"]),
                            datas[1].reshape(descs[1]["shape"]), trans).reshape(-1)
        else:
            raise AssertionError(f"unhandled operator {op}")
        want = np.array(vec["output_data"], dtype=np.int32)
        assert out.shape == want.shape, f"{name}: shape {out.shape} != {want.shape}"
        diff = np.flatnonzero(out != want)
        assert diff.size == 0, f"{name}: {diff.size} mismatches, first at {diff[:4]}: {out[diff[:4]]} != {want[diff[:4]]}"
        root = R.tensor_root({k: v for k, v in vec["output_desc"].items()},
                             [int(v) for v in out.tolist()])
        assert root.hex() == vec["output_root"], f"{name}: tensor root mismatch"
        fast_root = N.tensor_root_np({k: v for k, v in vec["output_desc"].items()}, out)
        assert fast_root.hex() == vec["output_root"], f"{name}: numpy tensor root mismatch"
        fast_merkle = N.tensor_merkle_root_np({k: v for k, v in vec["output_desc"].items()}, out)
        assert fast_merkle.hex() == R.tensor_merkle_root(
            {k: v for k, v in vec["output_desc"].items()}, [int(v) for v in out.tolist()]).hex(),             f"{name}: numpy merkle mismatch"
        checked += 1
    print(f"operator vectors OK ({checked} vectors)")


if __name__ == "__main__":
    check_primitives()
    check_operator_vectors()
    print("ALL BIT-EXACT")


def check_graph_vector() -> None:
    raw = json.loads((REPO / "compute" / "canonical" / "testdata" / "canonical_vectors.json")
                     .read_text(encoding="utf-8"))
    gvec = raw["graph"]
    desc = json.loads(json.dumps(gvec["descriptor"]))
    for ginput in desc["inputs"]:
        ginput["root"] = bytes.fromhex(ginput["root"])
    inputs = {}
    for item in gvec["inputs"]:
        inputs[item["index"]] = np.array(item["data"], dtype=np.int32)
    exec_out = N.execute_graph_np(desc, inputs)
    assert len(exec_out["trail"]) == len(gvec["trail"]), "trail length"
    for i, want in enumerate(gvec["trail"]):
        assert exec_out["trail"][i].hex() == want, f"trail[{i}] mismatch"
    for i, want in enumerate(gvec["outputs"]):
        assert exec_out["outputs"][i].hex() == want, f"output[{i}] mismatch"
    want_work = {k: v for k, v in gvec["work"]}
    got_work = dict(exec_out["work"])
    assert got_work == want_work, f"work mismatch: {got_work} != {want_work}"
    print("graph engine OK (trail, outputs, work)")


if __name__ == "__main__":
    check_primitives()
    check_operator_vectors()
    check_graph_vector()
    print("ALL BIT-EXACT")
