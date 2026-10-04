"""EXPERIMENTAL / NON-PROTOCOL — F.5C step 4.

Builds the formal CANONICAL_GRAPH_V2 descriptor of the real pinned Qwen3
layer-0 block from the frozen F.5B.2 artifacts (testdata/
f5b2_block_program.json + f5b2_block_bundle.npz), then executes it with
the Python V2 reference and asserts the raw node values are identical to
the F.5B.2 CPU manifest (310/310).  The formal GraphIDV2 differs from the
research GraphID by design (versioned domains and typed descriptors);
the VALUES must not differ.

Outputs:
  testdata/f5c_qwen3_block_v2.json          formal GraphDescriptorV2 (JSON wire)
  testdata/f5c_v2_inputs/inputs.bin         all input tensors, packed int64 LE
  testdata/f5c_qwen3_block_v2_roots.json    GraphIDV2, input roots, node roots (V2), work vector

    python tools/f5c_build_v2_graph.py
"""

from __future__ import annotations

import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import canonical_ref as R  # noqa: E402
sys.path.insert(0, str(REPO / "tools"))
import f1_canonical_numpy as N  # noqa: E402

DT = {4: R.DTYPE_V2_Q12_20, 2: R.DTYPE_V2_A13, 1: R.DTYPE_V2_W10, 3: R.DTYPE_V2_INT64_ACCUM}

import hashlib

_WIDTH = {R.DTYPE_V2_Q12_20: 4, R.DTYPE_V2_A13: 2, R.DTYPE_V2_W10: 2, R.DTYPE_V2_INT64_ACCUM: 8}


def fast_validate(desc, arr: np.ndarray) -> None:
    lo, hi = R._dtype_v2_range(int(desc["dtype"]))
    if lo is None:
        return
    if arr.size and (int(arr.min()) < lo or int(arr.max()) > hi):
        raise ValueError(f"canonical/v2: values outside logical range [{lo},{hi}]")


def fast_tensor_root_v2(desc, arr: np.ndarray) -> bytes:
    """Vectorized equivalent of canonical_ref.tensor_root_v2 (self-checked
    below against the scalar mirror on small cases)."""
    fast_validate(desc, arr)
    width = _WIDTH[int(desc["dtype"])]
    flat = np.asarray(arr, dtype=np.int64).reshape(-1)
    count = max(1, (flat.size + 63) // 64)
    be_dtype = {2: ">i2", 4: ">i4", 8: ">i8"}[width]
    padded = np.zeros(count * 64, dtype=np.int64)
    padded[:flat.size] = flat
    blob = padded.astype(be_dtype).tobytes()
    desc_bytes = R.encode_canonical(desc)
    base = hashlib.sha256(R.DOMAIN_TENSOR_V2 + desc_bytes)
    chunks = count * 64 * width
    leaves = []
    for i in range(count):
        h = base.copy()
        h.update(i.to_bytes(4, "big"))
        h.update(blob[i * 256:(i + 1) * 256] if width == 4 else blob[i * 64 * width:(i + 1) * 64 * width])
        leaves.append(h.digest())
    level = leaves
    while len(level) > 1:
        level = [hashlib.sha256(level[i] + (level[i + 1] if i + 1 < len(level) else level[i])).digest()
                 for i in range(0, len(level), 2)]
    merkle = level[0]
    return hashlib.sha256(R.DOMAIN_TENSOR_ROOT_V2 + desc_bytes + merkle).digest()


def fast_requant_wide(arr: np.ndarray, mult: int, shift: int, lo: int, hi: int) -> np.ndarray:
    v = np.asarray(arr, dtype=np.int64)
    prod = v * np.int64(mult)
    q = prod >> shift
    r = prod & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    bump = (r > half) | ((r == half) & ((q & 1) == 1))
    return np.clip(q + bump.astype(np.int64), lo, hi).astype(np.int64)


def selfcheck_fast_paths() -> None:
    rng = np.random.default_rng(7)
    for dtype, lo, hi in ((R.DTYPE_V2_A13, -4096, 4095), (R.DTYPE_V2_W10, -512, 511),
                          (R.DTYPE_V2_Q12_20, -(1 << 31), (1 << 31) - 1),
                          (R.DTYPE_V2_INT64_ACCUM, -(1 << 63), (1 << 63) - 1)):
        for shape in ((2, 3), (70, 2)):
            data = rng.integers(lo, hi + 1, size=shape, dtype=np.int64)
            if dtype == R.DTYPE_V2_INT64_ACCUM:
                data = data * (1 << 20)
            desc = R.tensor_desc_v2(dtype, shape)
            fast = fast_tensor_root_v2(desc, data)
            slow = R.tensor_root_v2(desc, data.reshape(-1).tolist())
            assert fast == slow, f"fast root mismatch dtype={dtype} shape={shape}"
    arr = rng.integers(-(1 << 40), 1 << 40, size=500, dtype=np.int64)
    fast = fast_requant_wide(arr, 1234567, 21, -(1 << 31), (1 << 31) - 1)
    slow = np.array(R.requant_wide(arr.tolist(), 1234567, 21, -(1 << 31), (1 << 31) - 1, R.DTYPE_V2_Q12_20))
    assert np.array_equal(fast, slow), "fast requant mismatch"
    print("fast-path self-check OK (roots + requant identical to the scalar mirror)")




def main() -> None:
    selfcheck_fast_paths()
    prog = json.loads((REPO / "testdata" / "f5b2_block_program.json").read_text(encoding="utf-8"))
    bundle = np.load(REPO / "testdata" / "f5b2_block_bundle.npz")
    manifest = np.load(REPO / "testdata" / "f5b2_cpu_manifest.npz")

    const_data = [bundle[f"const_{c['id']}"].reshape(c["shape"]).astype(np.int64)
                  for c in prog["consts"]]
    # input roles: a const used as a GEMM A operand is A13, as a GEMM W
    # operand is W10, everything else is Q12.20 (mask, rope table, norms, x).
    role: dict[int, int] = {}
    for node in prog["nodes"]:
        defs = node["inputs"]
        for pos, ref in enumerate(defs):
            if ref["kind"] != "const":
                continue
            cid = ref["index"]
            if node["op"] == "GEMM":
                # left operand is A13; a const right operand is a W10 model
                # weight (dynamic right operands are A13 attention chains)
                new = R.DTYPE_V2_A13 if pos == 0 else R.DTYPE_V2_W10
            else:
                new = R.DTYPE_V2_Q12_20
            if cid in role and role[cid] != new:
                raise SystemExit(f"const {cid} has conflicting roles {role[cid]} vs {new}")
            role[cid] = new

    inputs_wire = []
    inputs_canon = []
    for c in prog["consts"]:
        cid = c["id"]
        dtype = role.get(cid, R.DTYPE_V2_Q12_20)
        data = const_data[cid]
        desc = R.tensor_desc_v2(dtype, c["shape"])
        root = fast_tensor_root_v2(desc, data)
        # wire JSON uses base64 for the []byte root (Go's standard form);
        # the canonical-encoding view keeps the raw bytes
        inputs_wire.append({"name": f"const_{cid}", "desc": desc,
                            "root": base64.b64encode(root).decode()})
        inputs_canon.append({"name": f"const_{cid}", "desc": desc, "root": root})

    # node output roles: A13 when consumed as a GEMM A operand, Q12.20 when
    # consumed by a Q12.20 operator (or when it is a graph output/final),
    # INT64_ACCUM for GEMM outputs.
    out_role: dict[int, int] = {}
    for node in prog["nodes"]:
        seq = node["seq"]
        if node["op"] == "GEMM":
            out_role[seq] = R.DTYPE_V2_INT64_ACCUM
            continue
        out_role.setdefault(seq, R.DTYPE_V2_Q12_20)
    for node in prog["nodes"]:
        for pos, ref in enumerate(node["inputs"]):
            if ref["kind"] != "node":
                continue
            src = ref["index"]
            if node["op"] == "GEMM" and pos == 0:
                if out_role.get(src) == R.DTYPE_V2_INT64_ACCUM:
                    raise SystemExit(f"GEMM A operand {src} is an accumulator")
                out_role[src] = R.DTYPE_V2_A13
            elif node["op"] == "GEMM" and pos == 1:
                # dynamic right operand: attention k/v chains (A13 containers)
                if out_role.get(src) == R.DTYPE_V2_INT64_ACCUM:
                    raise SystemExit(f"GEMM right operand {src} is an accumulator")
                out_role[src] = R.DTYPE_V2_A13
            elif node["op"] == "REQUANTIZE":
                pass
            else:
                if out_role.get(src) == R.DTYPE_V2_A13:
                    raise SystemExit(f"node {src} consumed as A13 and as Q12.20")
                if out_role.get(src) != R.DTYPE_V2_INT64_ACCUM:
                    out_role[src] = R.DTYPE_V2_Q12_20

    nodes = []
    v1_version = "1.0.0"
    v1_ops = {"RMSNORM", "ROPE", "SILU", "SOFTMAX", "MUL", "ADD"}
    op_ids = {"RMSNORM": "RMSNORM_FIXED_V1", "ROPE": "ROPE_FIXED_V1",
              "SILU": "SILU_FIXED_V1", "SOFTMAX": "SOFTMAX_FIXED_V1",
              "MUL": "MUL_FIXED_V1", "ADD": "ADD_FIXED_V1"}
    for node in prog["nodes"]:
        seq = node["seq"]
        params = dict(node["params"])
        if node["op"] == "GEMM":
            op_id, version = "GEMM_A13W10_I64_V1", "GEMM_A13W10_I64_V1/1.0.0"
            out_dtype = R.DTYPE_V2_INT64_ACCUM
            params = {"transpose_b": int(params.get("transpose_b", False))}
        elif node["op"] == "REQUANTIZE":
            op_id, version = "REQUANTIZE_WIDE_V1", "REQUANTIZE_WIDE_V1/1.0.0"
            out_dtype = out_role[seq]
            params = {"mult": int(params["mult"]), "shift": int(params["shift"]),
                      "clamp_lo": int(params["lo"]), "clamp_hi": int(params["hi"]),
                      "out_dtype": int(out_dtype)}
        else:
            op_id, version = op_ids[node["op"]], v1_version
            out_dtype = R.DTYPE_V2_Q12_20
            # keep only real operator parameters (the capture harness added
            # side-band markers like residual/attn_chain/mlp, which are not
            # part of any operator's semantics)
            keep = {"RMSNORM": ("eps_fx",), "ROPE": ("half_dim", "max_pos")}.get(node["op"], ())
            params = {k: int(params[k]) for k in keep if k in params}
        nodes.append({
            "node_id": int(node["node_id"]),
            "operator_id": op_id,
            "operator_version": version,
            "inputs": [{"kind": 0 if r["kind"] == "const" else 1, "index": int(r["index"])} for r in node["inputs"]],
            "output": R.tensor_desc_v2(out_dtype, node["shape"]),
            "params": [{"key": k, "value": v} for k, v in sorted(params.items())],
        })

    common = {
        "protocol_version": R.PROTOCOL_VERSION_GRAPH_V2,
        "spec": "QWEN3_BLOCK_PROFILE_V2",
        "arithmetic": R.arithmetic_profile_a13w10(),
        "nodes": nodes,
        "outputs": [{"kind": 1, "index": len(nodes) - 1}],
    }
    descriptor_wire = dict(common, inputs=inputs_wire)
    descriptor_canon = dict(common, inputs=inputs_canon)
    (REPO / "testdata" / "f5c_qwen3_block_v2.json").write_text(
        json.dumps(descriptor_wire, indent=1) + "\n", encoding="utf-8")
    graph_id = R.hash_bytes(R.DOMAIN_GRAPH_V2, R.encode_canonical(descriptor_canon))
    print("GraphIDV2:", graph_id.hex())

    # --- pack inputs for the Go replay ---------------------------------------
    bin_path = REPO / "testdata" / "f5c_v2_inputs" / "inputs.bin"
    bin_path.parent.mkdir(exist_ok=True)
    with bin_path.open("wb") as f:
        f.write(b"F5CV2IN1")
        f.write(struct.pack("<I", len(const_data)))
        for arr in const_data:
            flat = arr.reshape(-1).astype("<i8")
            f.write(struct.pack("<Q", flat.size))
            f.write(flat.tobytes())
    print("inputs.bin bytes:", bin_path.stat().st_size)

    # --- Python V2 executor: raw-value equivalence vs the F.5B.2 manifest -----
    tensors: dict[tuple[str, int], np.ndarray] = {}
    for i, arr in enumerate(const_data):
        tensors[("const", i)] = arr
    work: dict[str, int] = {}
    node_roots = []
    bad = []
    for node in prog["nodes"]:
        ins = [tensors[(r["kind"], r["index"])] for r in node["inputs"]]
        op = node["op"]
        params = {k: int(v) for k, v in node["params"].items()
                  if isinstance(v, (int, np.integer))}
        if op == "GEMM":
            # vectorized int64 matmul: exact integer arithmetic (MaxSafeK64
            # admits every real shape), identical semantics to the scalar
            # reference; the scalar Go/Python reference is exercised on the
            # frozen vectors and cross-language test files.
            a = ins[0].astype(np.int64)
            w = ins[1].astype(np.int64)
            m, k = a.shape
            n = w.shape[1] if not params["transpose_b"] else w.shape[0]
            out = (a @ w.T) if params["transpose_b"] else (a @ w)
            work["GEMM_A13W10_MAC"] = work.get("GEMM_A13W10_MAC", 0) + m * n * k
        elif op == "REQUANTIZE":
            tgt = out_role[node["seq"]]
            out = fast_requant_wide(ins[0], params["mult"], params["shift"],
                                    params["lo"], params["hi"])
            work["REQUANTIZE_WIDE_ELEMENT"] = work.get("REQUANTIZE_WIDE_ELEMENT", 0) + ins[0].size
        elif op == "ADD":
            out = np.clip(ins[0].astype(np.int64) + ins[1].astype(np.int64),
                          R.MIN_FX, R.MAX_FX).astype(np.int64)
            work["ADD_ELEMENT"] = work.get("ADD_ELEMENT", 0) + ins[0].size
        elif op == "MUL":
            out = N.mul_fx(ins[0], ins[1]).astype(np.int64)
            work["MUL_ELEMENT"] = work.get("MUL_ELEMENT", 0) + ins[0].size
        elif op == "RMSNORM":
            out = N.op_rmsnorm(ins[0].reshape(node["shape"]), ins[1], params["eps_fx"]).astype(np.int64)
            work["RMSNORM_ELEMENT"] = work.get("RMSNORM_ELEMENT", 0) + ins[0].size
            work["RMSNORM_REDUCTION"] = work.get("RMSNORM_REDUCTION", 0) + ins[0].size
        elif op == "ROPE":
            out = N.op_rope_adjacent(ins[0].astype(np.int32).reshape(node["shape"]),
                                     np.asarray(ins[1]), params["half_dim"]).astype(np.int64)
            work["ROPE_PAIR"] = work.get("ROPE_PAIR", 0) + ins[0].size // 2
        elif op == "SILU":
            out = N.op_silu(ins[0].astype(np.int32).reshape(node["shape"])).astype(np.int64)
            work["SILU_ELEMENT"] = work.get("SILU_ELEMENT", 0) + ins[0].size
        elif op == "SOFTMAX":
            out = N.op_softmax_rows(ins[0].astype(np.int32).reshape(node["shape"])).astype(np.int64)
            work["SOFTMAX_ELEMENT"] = work.get("SOFTMAX_ELEMENT", 0) + ins[0].size
            work["SOFTMAX_EXP"] = work.get("SOFTMAX_EXP", 0) + ins[0].size
        else:
            raise SystemExit(f"unknown op {op}")
        out = out.reshape(node["shape"])
        want = manifest[f"out_{node['seq']}"].reshape(node["shape"])
        if not np.array_equal(out, want):
            bad.append({"seq": node["seq"], "op": op,
                        "mismatches": int((out != want).sum())})
        tensors[("node", node["seq"])] = out
        desc = R.tensor_desc_v2(out_role.get(node["seq"], R.DTYPE_V2_Q12_20), node["shape"])
        node_roots.append({"seq": node["seq"], "node_id": node["node_id"],
                           "op": op, "dtype": desc["dtype"],
                           "root": fast_tensor_root_v2(desc, out).hex()})

    print(f"Python V2 executor vs F.5B.2 manifest: {len(prog['nodes']) - len(bad)}/"
          f"{len(prog['nodes'])} nodes value-identical")
    for b in bad[:5]:
        print("  BAD:", b)
    roots_doc = {
        "graph_id_v2": graph_id.hex(),
        "policy_id": R.POLICY_ID_A13W10,
        "arithmetic_profile": R.ARITHMETIC_PROFILE_A13W10,
        "input_roots": [{"name": i["name"], "root": i["root"].hex()} for i in inputs_canon],
        "node_roots_v2": node_roots,
        "work_vector": sorted(work.items()),
        "equivalence": {"vs_f5b2_cpu_manifest": f"{len(prog['nodes']) - len(bad)}/{len(prog['nodes'])}",
                        "value_identical": not bad},
    }
    (REPO / "testdata" / "f5c_qwen3_block_v2_roots.json").write_text(
        json.dumps(roots_doc, indent=1) + "\n", encoding="utf-8")
    print("work vector:", sorted(work.items()))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
