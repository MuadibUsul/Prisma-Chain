"""Generate compute/canonical/testdata/canonical_vectors.json.

Every value in the file is produced by the independent Python mirror
(canonical_ref.py); the Go implementation must reproduce all of it
bit-for-bit (compute/canonical/crosslang_test.go). Regenerate with:

    python compute/canonical/python/gen_canonical_vectors.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import canonical_ref as ref


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.s >> 33

    def in_range(self, lo, hi):
        return lo + self.next() % (hi - lo + 1)

    def data(self, n, lo, hi):
        return [self.in_range(lo, hi) for _ in range(n)]


def hexify(value):
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {k: hexify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [hexify(v) for v in value]
    return value


def math_vectors():
    out = {}

    cases = []
    pairs = [
        (1 << 19, 1 << 19),          # 0.5 * 0.5
        (3 << 19, 5 << 18),          # tie behaviour
        (-(3 << 19), 5 << 18),
        (ref.ONE, ref.ONE),          # 1.0 * 1.0
        (-ref.ONE, ref.ONE),
        (ref.MAX_FX, ref.MAX_FX),    # saturation
        (ref.MIN_FX, ref.ONE),
        (123456789, -987654),
        (0, ref.MAX_FX),
        (1, 1),
        (-1, 1),
        (7 << 20, 3),
    ]
    for a, b in pairs:
        cases.append([a, b, ref.mul_fx(a, b)])
    out["mul_fx"] = cases

    cases = []
    for a, b in [(1 << 20, 1 << 20), (ref.MAX_FX, 1), (ref.MIN_FX, -1), (ref.MAX_FX, ref.MAX_FX), (-5, 3), (5, -3), (0, 0)]:
        cases.append([a, b, ref.add_fx(a, b)])
    out["add_fx"] = cases

    cases = []
    for a, b in [(1 << 20, 1 << 20), (ref.MIN_FX, 1), (ref.MAX_FX, -1), (0, 5), (-5, -5)]:
        cases.append([a, b, ref.sub_fx(a, b)])
    out["sub_fx"] = cases

    cases = []
    for a, b in [(ref.ONE, 2 << 20), (3 << 19, ref.ONE), (ref.ONE, 3 << 20), (-ref.ONE, 3 << 20), (ref.ONE, -3 << 20), (ref.MAX_FX, ref.ONE), (ref.ONE, ref.MAX_FX), (5, 2), (-5, 2), (5, -2), (-5, -2)]:
        cases.append([a, b, ref.div_fx(a, b)])
    out["div_fx"] = cases

    cases = []
    for x in [0, ref.ONE, 2 * ref.ONE, -ref.ONE, -2 * ref.ONE, -(ref.ONE * 24), -(ref.ONE * 25), ref.ONE * 21, ref.ONE * 100, -(6 << 20), 1, -1, 1048576, -1048576, 3 << 19, -(3 << 19)]:
        cases.append([x, ref.exp_fx(x)])
    out["exp_fx"] = cases

    cases = []
    for x in [0, ref.ONE, -ref.ONE, 5 << 20, -(5 << 20), 1, -1, ref.ONE * 20, -(ref.ONE * 20)]:
        cases.append([x, ref.sigmoid_fx(x)])
    out["sigmoid_fx"] = cases

    cases = []
    for x in [1, 2, 3, 4, 16, 1 << 20, 3 << 19, 1 << 40, 7, 1023, (1 << 62), 1 << 10]:
        cases.append([x, ref.inv_sqrt_fx(x)])
    out["inv_sqrt_fx"] = cases

    cases = []
    for v, s in [(7, 1), (5, 1), (9, 2), (6, 2), (10, 2), (-1, 1), (-3, 1), (-5, 2), (123456789, 20), (-123456789, 20), (0, 4), (1 << 40, 30)]:
        cases.append([v, s, ref.rshift_round_even(v, s)])
    out["rshift_round_even"] = cases

    return out


def tensor_root_vectors():
    r = LCG(0xF00D)
    vectors = [
        ("q12_20_multi_chunk", ref.new_desc(ref.DTYPE_Q12_20, 5, 20), r.data(100, -(3 << 20), 3 << 20)),
        ("int8_odd", ref.new_desc(ref.DTYPE_INT8, 13), r.data(13, -128, 127)),
        ("accum_single", ref.new_desc(ref.DTYPE_INT32_ACCUM, 64), r.data(64, -(1 << 30), 1 << 30)),
        ("int8_exact_chunk", ref.new_desc(ref.DTYPE_INT8, 8, 8), r.data(64, -128, 127)),
        ("q12_20_three_chunks", ref.new_desc(ref.DTYPE_Q12_20, 3, 64), r.data(192, -(1 << 24), 1 << 24)),
    ]
    out = []
    for name, desc, data in vectors:
        out.append({
            "name": name,
            "desc": desc,
            "data": data,
            "merkle": ref.tensor_merkle_root(desc, data),
            "root": ref.tensor_root(desc, data),
        })
    return out


def operator_vectors():
    r = LCG(0xBEEF)
    out = []

    def emit(name, op_id, descs_data, params, rope_table=None):
        inputs = [{"desc": d, "data": data} for d, data in descs_data]
        if op_id == ref.OP_ROPE:
            out_desc, out_data = ref.op_rope(inputs, params, rope_table=rope_table)
        else:
            out_desc, out_data = ref.OPERATORS[op_id](inputs, params)
        want = ref.op_out_spec(op_id, [i["desc"] for i in inputs], params)
        assert want == out_desc, name
        out.append({
            "name": name,
            "operator_id": op_id,
            "inputs": inputs,
            "params": params,
            "output_desc": out_desc,
            "output_data": out_data,
            "output_root": ref.tensor_root(out_desc, out_data),
            "work": sorted(ref.op_work(op_id, inputs, params)),
        })

    q = ref.new_desc(ref.DTYPE_Q12_20, 4, 8)
    emit("add_fixed_basic", ref.OP_ADD,
         [(q, r.data(32, -(2 << 20), 2 << 20)), (q, r.data(32, -(2 << 20), 2 << 20))], [])
    emit("add_fixed_saturating", ref.OP_ADD,
         [(q, [ref.MAX_FX] * 32), (q, [ref.MAX_FX] * 32)], [])
    emit("mul_fixed_basic", ref.OP_MUL,
         [(q, r.data(32, -(4 << 20), 4 << 20)), (q, r.data(32, -(4 << 20), 4 << 20))], [])
    accum = ref.new_desc(ref.DTYPE_INT32_ACCUM, 4, 8)
    emit("requantize_int8", ref.OP_REQUANTIZE,
         [(accum, r.data(32, -(1 << 20), 1 << 20))],
         [["clamp_hi", 127], ["clamp_lo", -127], ["mult", 1], ["out_dtype", 1], ["shift", 5]])
    emit("requantize_fx", ref.OP_REQUANTIZE,
         [(accum, r.data(32, -(1 << 26), 1 << 26))],
         [["clamp_hi", ref.MAX_FX], ["clamp_lo", ref.MIN_FX], ["mult", 1024], ["shift", 0]])
    xdesc = ref.new_desc(ref.DTYPE_Q12_20, 4, 16)
    wdesc = ref.new_desc(ref.DTYPE_Q12_20, 16)
    emit("rmsnorm_rows", ref.OP_RMSNORM,
         [(xdesc, r.data(64, -(2 << 20), 2 << 20)), (wdesc, r.data(16, 0, 2 << 20))],
         [["chunk", 16], ["eps_fx", 10]])
    emit("silu_range", ref.OP_SILU,
         [(ref.new_desc(ref.DTYPE_Q12_20, 16), [-(12 << 20), -(3 << 20), -(1 << 20), -1, 0, 1, 1 << 20, 3 << 20, 12 << 20, 20 << 20, -(20 << 20), 5, -5, 7 << 19, -(7 << 19), 100])], [])
    sdesc = ref.new_desc(ref.DTYPE_Q12_20, 3, 8)
    rows = [
        [1 << 20, 2 << 20, 3 << 20, 0, -(1 << 20), -(2 << 20), 5 << 19, 0],
        [-(25 << 20), 0, 0, 0, 0, 0, 0, 0],          # exp underflow row
        [21 << 20, 20 << 20, 19 << 20, 0, 0, 0, 0, -(30 << 20)],  # saturation row
    ]
    flat = [v for row in rows for v in row]
    emit("softmax_rows", ref.OP_SOFTMAX, [(sdesc, flat)], [])
    rdesc = ref.new_desc(ref.DTYPE_Q12_20, 4, 8)
    table = r.data(32, -(1 << 20), 1 << 20)
    emit("rope_pairs", ref.OP_ROPE,
         [(rdesc, r.data(32, -(3 << 20), 3 << 20)), (ref.new_desc(ref.DTYPE_Q12_20, 32), table)],
         [["half_dim", 4], ["max_pos", 4]], rope_table=table)
    a8 = ref.new_desc(ref.DTYPE_INT8, 4, 8)
    b84 = ref.new_desc(ref.DTYPE_INT8, 8, 4)
    emit("gemm_plain", ref.OP_GEMM,
         [(a8, r.data(32, -128, 127)), (b84, r.data(32, -128, 127))], [])
    b48 = ref.new_desc(ref.DTYPE_INT8, 4, 8)
    emit("gemm_transpose_b", ref.OP_GEMM,
         [(a8, r.data(32, -128, 127)), (b48, r.data(32, -128, 127))],
         [["transpose_b", 1]])

    # lowercase snake_case spelling of the tensor desc: reuse the dicts
    return out


def graph_vector():
    r = LCG(0x5EED)
    x = ref.new_desc(ref.DTYPE_INT8, 4, 8)
    wq = ref.new_desc(ref.DTYPE_INT8, 8, 4)
    wk = ref.new_desc(ref.DTYPE_INT8, 8, 4)
    xt = ref.new_desc(ref.DTYPE_Q12_20, 4, 4)
    data = {
        0: {"desc": x, "data": r.data(32, -128, 127)},
        1: {"desc": wq, "data": r.data(32, -128, 127)},
        2: {"desc": wk, "data": r.data(32, -128, 127)},
        3: {"desc": xt, "data": r.data(16, -(2 << 20), 2 << 20)},
    }

    def node(node_id, op_id, inputs, params, output):
        return {
            "node_id": node_id,
            "operator_id": op_id,
            "operator_version": ref.VERSION_GEMM if op_id == ref.OP_GEMM else ref.VERSION_FXFUSION,
            "inputs": inputs,
            "output": output,
            "params": sorted(params),
        }

    nodes = [
        node(0, ref.OP_GEMM, [{"kind": 0, "index": 0}, {"kind": 0, "index": 1}], [],
             ref.new_desc(ref.DTYPE_INT32_ACCUM, 4, 4)),
        node(1, ref.OP_REQUANTIZE, [{"kind": 1, "index": 0}],
             [["clamp_hi", 127], ["clamp_lo", -127], ["mult", 1], ["out_dtype", 1], ["shift", 15]],
             ref.new_desc(ref.DTYPE_INT8, 4, 4)),
        node(2, ref.OP_GEMM, [{"kind": 0, "index": 0}, {"kind": 0, "index": 2}], [],
             ref.new_desc(ref.DTYPE_INT32_ACCUM, 4, 4)),
        node(3, ref.OP_REQUANTIZE, [{"kind": 1, "index": 2}],
             [["clamp_hi", 127], ["clamp_lo", -127], ["mult", 1], ["out_dtype", 1], ["shift", 5]],
             ref.new_desc(ref.DTYPE_INT8, 4, 4)),
        node(4, ref.OP_GEMM, [{"kind": 1, "index": 1}, {"kind": 1, "index": 3}],
             [["transpose_b", 1]], ref.new_desc(ref.DTYPE_INT32_ACCUM, 4, 4)),
        node(5, ref.OP_REQUANTIZE, [{"kind": 1, "index": 4}],
             [["clamp_hi", ref.MAX_FX], ["clamp_lo", ref.MIN_FX], ["mult", 64], ["shift", 0]],
             ref.new_desc(ref.DTYPE_Q12_20, 4, 4)),
        node(6, ref.OP_ADD, [{"kind": 1, "index": 5}, {"kind": 0, "index": 3}], [],
             ref.new_desc(ref.DTYPE_Q12_20, 4, 4)),
        node(7, ref.OP_SOFTMAX, [{"kind": 1, "index": 6}], [],
             ref.new_desc(ref.DTYPE_Q12_20, 4, 4)),
        node(8, ref.OP_SILU, [{"kind": 1, "index": 7}], [],
             ref.new_desc(ref.DTYPE_Q12_20, 4, 4)),
        node(9, ref.OP_MUL, [{"kind": 1, "index": 8}, {"kind": 1, "index": 6}], [],
             ref.new_desc(ref.DTYPE_Q12_20, 4, 4)),
    ]

    inputs = []
    input_meta = []
    for idx in sorted(data):
        root = ref.tensor_root(data[idx]["desc"], data[idx]["data"])
        input_meta.append({"name": f"in{idx}", "desc": data[idx]["desc"], "root": root})
        inputs.append({"index": idx, "desc": data[idx]["desc"], "data": data[idx]["data"]})

    descriptor = {
        "protocol_version": ref.GRAPH_PROTOCOL_VERSION,
        "spec": "CANONICAL_VECTOR_GRAPH",
        "inputs": input_meta,
        "nodes": nodes,
        "outputs": [{"kind": 1, "index": 9}],
    }
    execution = ref.execute_graph(descriptor, data)
    node_roots = [execution["roots"][(1, i)] for i in range(len(nodes))]
    manifest_root, _ = ref.build_node_output_manifest(ref.graph_id(descriptor), nodes, node_roots)
    return {
        "descriptor": descriptor,
        "inputs": inputs,
        "graph_id": ref.graph_id(descriptor),
        "trail": execution["trail"],
        "outputs": execution["outputs"],
        "work": execution["work"],
        "node_roots": node_roots,
        "node_output_manifest_root": manifest_root,
    }


def main():
    vectors = {
        "version": "CANONICAL_VECTORS_V1",
        "math_version": ref.MATH_VERSION,
        "math": math_vectors(),
        "tensor_roots": tensor_root_vectors(),
        "operators": operator_vectors(),
        "graph": graph_vector(),
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "testdata", "canonical_vectors.json")
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(hexify(vectors), handle, indent=1, sort_keys=False)
        handle.write("\n")
    print("written:", out_path)


if __name__ == "__main__":
    main()
