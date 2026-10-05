"""EXPERIMENTAL / NON-PROTOCOL — F.5B.2 CPU manifest + block program dump.

Runs the FROZEN F.5A A13W10 executor (tools/f5a_joint_precision.py,
run_block_bits) on heldout case 0 and captures every canonical operator
invocation as a graph node (op, params, inputs, output), producing:

  testdata/f5b2_block_program.json   node program (310 nodes: op, params,
                                     input refs, output desc; GraphID)
  testdata/f5b2_block_bundle.npz     consts (input x, mask, rope table,
                                     norm weights, GEMM weight tensors)
  testdata/f5b2_cpu_manifest.npz     per-node CPU outputs (int64)
  testdata/f5b2_cpu_roots.json       per-node F.5B.2 roots + final output

Method: the frozen source is transformed by PURE-CAPTURE insertions only
(no arithmetic statement is modified); the hooked module's final output is
asserted bit-identical to the pristine module's output.  Nodes are mapped
onto the converted real-block graph's node ids positionally with a full
310-node operator-sequence assertion (the F.5B GEMM-dump precedent
generalized to all operators).

Roots: wide_node_root() — the canonical 64-element chunk merkle over the
full int64 values (each element as two big-endian int32 words), because
GEMM accumulators exceed int32.  This is a research validation root, not
a protocol object.

    PRISMA_MODEL_DIR=... python tools/f5b2_dump_block_program.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

SRC = (REPO / "tools" / "f5a_joint_precision.py").read_text(encoding="utf-8")

PREAMBLE = '''

# --- F.5B.2 capture harness (pure observation; no arithmetic change) -------
_NODE_DUMP: list = []

def _cap(op, params, ins, out):
    _NODE_DUMP.append({
        "op": op,
        "params": {k: (int(v) if isinstance(v, (int, np.integer)) else v)
                   for k, v in params.items()},
        "ins": [np.array(a, copy=True) for a in ins],
        "out": np.array(out, copy=True),
    })
# --- end capture harness ----------------------------------------------------
'''

GEMM_ANCHOR = ("    def gemm(a, b, transpose_b=False):\n"
               "        acc = gemm64(a, b, transpose_b)\n"
               '        if "gemm_dump" in stats:\n'
               '            stats["gemm_dump"].append({"a": np.asarray(a, np.int16),\n'
               '                                       "w": np.asarray(b, np.int16),\n'
               '                                       "transpose_b": bool(transpose_b),\n'
               '                                       "c": acc.astype(np.int64)})\n'
               "        return acc")
GEMM_REPL = ("    def gemm(a, b, transpose_b=False):\n"
             "        acc = gemm64(a, b, transpose_b)\n"
             '        _cap("GEMM", {"transpose_b": bool(transpose_b)}, [a, b], acc)\n'
             "        return acc")

# (anchor, replacement) pairs — every replacement only APPENDS observation
# statements or splits an expression into a temp + the identical expression.
REPLACEMENTS: list[tuple[str, str]] = [
    ("input_ln", (
        '    h = N.op_rmsnorm(x, np.rint(conf.weights["input_layernorm.weight"] * ONE).astype(np.int64), 1)\n'
        '    h8 = quant_bits(h.astype(np.int64), S["h"]["step_fx"], bits)',
        '    _w_in = np.rint(conf.weights["input_layernorm.weight"] * ONE).astype(np.int64)\n'
        '    h = N.op_rmsnorm(x, _w_in, 1)\n'
        '    _cap("RMSNORM", {"eps_fx": 1}, [x, _w_in], h)\n'
        '    h8 = quant_bits(h.astype(np.int64), S["h"]["step_fx"], bits)\n'
        '    _cap("REQUANTIZE", {"mult": max(1, int(round((1 << 30) / S["h"]["step_fx"]))),'
        ' "shift": 30, "lo": -qmax(bits), "hi": qmax(bits)}, [h.astype(np.int64)], h8)')),
    ("q_head", (
        '        qv = requant_acc(acc, S["h"]["step_fx"] * W["wq"][qh])\n'
        '        qv = N.op_rmsnorm(qv, qn, 1).astype(np.int64)\n'
        '        qv = rope(qv)\n'
        '        q8s.append(quant_bits(qv, S["q_heads"][qh], bits))',
        '        qv = requant_acc(acc, S["h"]["step_fx"] * W["wq"][qh])\n'
        '        _cap("REQUANTIZE", {"mult": int(S["h"]["step_fx"] * W["wq"][qh]), "shift": 20,'
        ' "lo": MIN_FX, "hi": MAX_FX}, [acc], qv)\n'
        '        _qv = qv\n'
        '        qv = N.op_rmsnorm(qv, qn, 1).astype(np.int64)\n'
        '        _cap("RMSNORM", {"eps_fx": 1}, [_qv, qn], qv)\n'
        '        _qv = qv\n'
        '        qv = rope(qv)\n'
        '        _cap("ROPE", {"half_dim": 64, "max_pos": 16}, [_qv, table], qv)\n'
        '        _q8 = quant_bits(qv, S["q_heads"][qh], bits)\n'
        '        _cap("REQUANTIZE", {"mult": max(1, int(round((1 << 30) / S["q_heads"][qh]))),'
        ' "shift": 30, "lo": -qmax(bits), "hi": qmax(bits)}, [qv], _q8)\n'
        '        q8s.append(_q8)')),
    ("k_head", (
        '        kvx = requant_acc(acc, S["h"]["step_fx"] * W["wk"][kh])\n'
        '        kvx = N.op_rmsnorm(kvx, kn, 1).astype(np.int64)\n'
        '        kvx = rope(kvx)\n'
        '        k8s.append(quant_bits(kvx, S["k_heads"][kh], bits))',
        '        kvx = requant_acc(acc, S["h"]["step_fx"] * W["wk"][kh])\n'
        '        _cap("REQUANTIZE", {"mult": int(S["h"]["step_fx"] * W["wk"][kh]), "shift": 20,'
        ' "lo": MIN_FX, "hi": MAX_FX}, [acc], kvx)\n'
        '        _kv = kvx\n'
        '        kvx = N.op_rmsnorm(kvx, kn, 1).astype(np.int64)\n'
        '        _cap("RMSNORM", {"eps_fx": 1}, [_kv, kn], kvx)\n'
        '        _kv = kvx\n'
        '        kvx = rope(kvx)\n'
        '        _cap("ROPE", {"half_dim": 64, "max_pos": 16}, [_kv, table], kvx)\n'
        '        _k8 = quant_bits(kvx, S["k_heads"][kh], bits)\n'
        '        _cap("REQUANTIZE", {"mult": max(1, int(round((1 << 30) / S["k_heads"][kh]))),'
        ' "shift": 30, "lo": -qmax(bits), "hi": qmax(bits)}, [kvx], _k8)\n'
        '        k8s.append(_k8)')),
    ("v_requant", (
        '        v8s.append(np.clip(mul_shift_guard(acc, mult_v, sh_v), -qmax(bits), qmax(bits)))',
        '        _v8 = np.clip(mul_shift_guard(acc, mult_v, sh_v), -qmax(bits), qmax(bits))\n'
        '        _cap("REQUANTIZE", {"mult": int(mult_v), "shift": int(sh_v),'
        ' "lo": -qmax(bits), "hi": qmax(bits)}, [acc], _v8)\n'
        '        v8s.append(_v8)')),
    ("scores_requant", (
        '        sfx = np.clip(mul_shift_guard(acc, mult, SCORES_S), MIN_FX, MAX_FX)',
        '        sfx = np.clip(mul_shift_guard(acc, mult, SCORES_S), MIN_FX, MAX_FX)\n'
        '        _cap("REQUANTIZE", {"mult": int(mult), "shift": int(SCORES_S),'
        ' "lo": MIN_FX, "hi": MAX_FX}, [acc], sfx)')),
    ("mask_add_softmax", (
        '        probs = N.op_softmax_rows(np.clip(sfx + causal, MIN_FX, MAX_FX).astype(np.int32)).astype(np.int64)',
        '        _madd = np.clip(sfx + causal, MIN_FX, MAX_FX)\n'
        '        _cap("ADD", {}, [sfx, causal], _madd)\n'
        '        probs = N.op_softmax_rows(_madd.astype(np.int32)).astype(np.int64)\n'
        '        _cap("SOFTMAX", {}, [_madd.astype(np.int32)], probs)')),
    ("p_requant", (
        '        p8 = quant_bits(probs, S["p"]["step_fx"], bits, lo=0)',
        '        p8 = quant_bits(probs, S["p"]["step_fx"], bits, lo=0)\n'
        '        _cap("REQUANTIZE", {"mult": max(1, int(round((1 << 30) / S["p"]["step_fx"]))),'
        ' "shift": 30, "lo": 0, "hi": qmax(bits)}, [probs], p8)')),
    ("ctx_requant", (
        '        c8 = np.clip(mul_shift_guard(acc, mult_c, sh_c), -qmax(bits), qmax(bits))',
        '        c8 = np.clip(mul_shift_guard(acc, mult_c, sh_c), -qmax(bits), qmax(bits))\n'
        '        _cap("REQUANTIZE", {"mult": int(mult_c), "shift": int(sh_c),'
        ' "lo": -qmax(bits), "hi": qmax(bits)}, [acc], c8)')),
    ("oproj_attn", (
        '        ofx = requant_acc(acc, S["ctx"]["step_fx"] * W["wo"][qh])\n'
        '        raw = attn + ofx\n'
        '        stats["fx_saturations"] += int(np.count_nonzero((raw < MIN_FX) | (raw > MAX_FX)))\n'
        '        attn = np.clip(raw, MIN_FX, MAX_FX)',
        '        ofx = requant_acc(acc, S["ctx"]["step_fx"] * W["wo"][qh])\n'
        '        _cap("REQUANTIZE", {"mult": int(S["ctx"]["step_fx"] * W["wo"][qh]), "shift": 20,'
        ' "lo": MIN_FX, "hi": MAX_FX}, [acc], ofx)\n'
        '        raw = attn + ofx\n'
        '        stats["fx_saturations"] += int(np.count_nonzero((raw < MIN_FX) | (raw > MAX_FX)))\n'
        '        _prev_attn = attn\n'
        '        attn = np.clip(raw, MIN_FX, MAX_FX)\n'
        '        if qh > 0:\n'
        '            _cap("ADD", {"attn_chain": qh}, [_prev_attn, ofx], attn)\n'
        '        # head 0 adds to the zero-initialized accumulator; the converted\n'
        '        # graph folds that add away (ofx is already clamped).  The graph\n'
        '        # also defers the 15 head-sum ADDs until after the head loop\n'
        '        # (same left fold in head order); the capture side-bands them\n'
        '        # and splices them into graph order before the x1 residual.')),
    ("residual1", (
        '    x1 = np.clip(x + attn, MIN_FX, MAX_FX)',
        '    x1 = np.clip(x + attn, MIN_FX, MAX_FX)\n'
        '    _cap("ADD", {"residual": "x1"}, [x, attn], x1)')),
    ("post_ln", (
        '    h2 = N.op_rmsnorm(x1.astype(np.int32),\n'
        '                      np.rint(conf.weights["post_attention_layernorm.weight"] * ONE).astype(np.int64), 1)',
        '    _w_post = np.rint(conf.weights["post_attention_layernorm.weight"] * ONE).astype(np.int64)\n'
        '    h2 = N.op_rmsnorm(x1.astype(np.int32), _w_post, 1)\n'
        '    _cap("RMSNORM", {"eps_fx": 1}, [x1.astype(np.int32), _w_post], h2)')),
    ("h2_requant", (
        '    h2q = quant_bits(h2.astype(np.int64), S["h2"]["step_fx"], bits)',
        '    h2q = quant_bits(h2.astype(np.int64), S["h2"]["step_fx"], bits)\n'
        '    _cap("REQUANTIZE", {"mult": max(1, int(round((1 << 30) / S["h2"]["step_fx"]))),'
        ' "shift": 30, "lo": -qmax(bits), "hi": qmax(bits)}, [h2.astype(np.int64)], h2q)')),
    ("gate_requant", (
        '    gate = requant_acc(acc, S["h2"]["step_fx"] * W["wg"])',
        '    gate = requant_acc(acc, S["h2"]["step_fx"] * W["wg"])\n'
        '    _cap("REQUANTIZE", {"mult": int(S["h2"]["step_fx"] * W["wg"]), "shift": 20,'
        ' "lo": MIN_FX, "hi": MAX_FX, "mlp": "gate_req"}, [acc], gate)')),
    ("up_requant", (
        '    upv = requant_acc(acc, S["h2"]["step_fx"] * W["wu"])',
        '    upv = requant_acc(acc, S["h2"]["step_fx"] * W["wu"])\n'
        '    _cap("REQUANTIZE", {"mult": int(S["h2"]["step_fx"] * W["wu"]), "shift": 20,'
        ' "lo": MIN_FX, "hi": MAX_FX}, [acc], upv)')),
    ("silu_mul", (
        '    hm = N.op_mul(N.op_silu(gate.astype(np.int32)), upv.astype(np.int32)).astype(np.int64)',
        '    _gs = N.op_silu(gate.astype(np.int32))\n'
        '    _cap("SILU", {"mlp": "silu"}, [gate.astype(np.int32)], _gs)\n'
        '    hm = N.op_mul(_gs, upv.astype(np.int32)).astype(np.int64)\n'
        '    _cap("MUL", {}, [_gs, upv.astype(np.int32)], hm)')),
    ("hm_requant", (
        '    hmq = quant_bits(hm, S["hm"]["step_fx"], bits)',
        '    hmq = quant_bits(hm, S["hm"]["step_fx"], bits)\n'
        '    _cap("REQUANTIZE", {"mult": max(1, int(round((1 << 30) / S["hm"]["step_fx"]))),'
        ' "shift": 30, "lo": -qmax(bits), "hi": qmax(bits)}, [hm], hmq)')),
    ("down_requant", (
        '    down = requant_acc(acc, S["hm"]["step_fx"] * W["wd"])',
        '    down = requant_acc(acc, S["hm"]["step_fx"] * W["wd"])\n'
        '    _cap("REQUANTIZE", {"mult": int(S["hm"]["step_fx"] * W["wd"]), "shift": 20,'
        ' "lo": MIN_FX, "hi": MAX_FX}, [acc], down)')),
    ("residual2", (
        '    return np.clip(x1 + down, MIN_FX, MAX_FX)',
        '    _final = np.clip(x1 + down, MIN_FX, MAX_FX)\n'
        '    _cap("ADD", {}, [x1, down], _final)\n'
        '    return _final')),
]

OP_TO_GRAPH = {"ADD": "ADD_FIXED_V1", "MUL": "MUL_FIXED_V1",
               "REQUANTIZE": "REQUANTIZE_V1", "RMSNORM": "RMSNORM_FIXED_V1",
               "ROPE": "ROPE_FIXED_V1", "SILU": "SILU_FIXED_V1",
               "SOFTMAX": "SOFTMAX_FIXED_V1", "GEMM": "GEMM_INT8_V1"}

POLICY_ID = "eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0"


def wide_node_root(data: np.ndarray) -> str:
    """F.5B.2 validation root: canonical 64-element chunk merkle over the
    FULL int64 values (each element as big-endian (hi, lo) int32 words)."""
    flat = np.ascontiguousarray(data, dtype=np.int64).reshape(-1)
    hi = (flat >> 32).astype(np.int32)
    lo = (flat & 0xFFFFFFFF).astype(np.int32)
    inter = np.empty(flat.size * 2, dtype=np.int64)
    inter[0::2] = hi
    inter[1::2] = lo
    words = inter.astype(">i4")
    count = max(1, (flat.size + 63) // 64)
    level = []
    for i in range(count):
        chunk = np.zeros(128, dtype=">i4")
        take = min(128, max(0, words.size - i * 128))
        if take > 0:
            chunk[:take] = words[i * 128:i * 128 + take]
        level.append(hashlib.sha256(b"f5b2.wide.v1" + i.to_bytes(4, "big") + chunk.tobytes()).digest())
    while len(level) > 1:
        level = [hashlib.sha256(level[i] + (level[i + 1] if i + 1 < len(level) else level[i])).digest()
                 for i in range(0, len(level), 2)]
    return level[0].hex()


def _arr_key(a: np.ndarray) -> tuple:
    a64 = np.ascontiguousarray(a, dtype=np.int64)
    return (a64.shape, hashlib.sha256(a64.tobytes()).hexdigest())


def build_hooked(path: Path):
    hooked_src = SRC
    for name, (anchor, repl) in REPLACEMENTS:
        assert anchor in hooked_src, f"anchor missing ({name}): {anchor[:70]!r}"
        assert hooked_src.count(anchor) == 1, f"anchor not unique ({name})"
        hooked_src = hooked_src.replace(anchor, repl)
    assert GEMM_ANCHOR in hooked_src, "gemm anchor missing"
    hooked_src = hooked_src.replace(GEMM_ANCHOR, GEMM_REPL)
    marker = "import f1_canonical_numpy as N  # noqa: E402"
    hooked_src = hooked_src.replace(marker, marker + PREAMBLE, 1)
    path.write_text(hooked_src, encoding="utf-8")


def main() -> None:
    import f5a_joint_precision as F5A
    from convert_qwen3_block import Conformer, build_converted_graph

    hooked_path = REPO / "tools" / "_f5b2_hooked_exec.py"
    build_hooked(hooked_path)
    sys.path.insert(0, str(REPO / "tools"))
    import _f5b2_hooked_exec as H  # noqa: E402

    conf = Conformer(0)
    cases, samples, head_samples = F5A.collect_calibration(conf)
    plan = F5A.build_plan(conf, samples, head_samples, 13, 10)
    held = json.loads((REPO / "testdata" / "qwen3_f5a_heldout.json")
                      .read_text(encoding="utf-8"))["cases"]
    hidden = conf.embedding(held[0]["token_ids"])

    stats0 = {"acc_abs_max": 0, "fx_saturations": 0}
    ref_out = F5A.run_block_bits(conf, hidden, plan, stats0)

    stats = {"acc_abs_max": 0, "fx_saturations": 0}
    out = H.run_block_bits(conf, hidden, plan, stats)
    assert np.array_equal(np.asarray(out, dtype=np.int64), np.asarray(ref_out, dtype=np.int64)), \
        "ANCHOR FAILED: hooked executor diverged from the frozen executor"
    print("anchor OK: hooked final output bit-identical to the frozen executor")

    raw_dump = H._NODE_DUMP
    assert stats["fx_saturations"] == 0, (
        "fx saturation occurred: the pipeline's interleaved head fold and the "
        "graph's deferred head_sum chain would not be provably identical")
    attn_adds = [d for d in raw_dump if d["params"].get("attn_chain")]
    silu_caps = [d for d in raw_dump if d["params"].get("mlp") == "silu"]
    stream = [d for d in raw_dump
              if not d["params"].get("attn_chain") and d["params"].get("mlp") != "silu"]
    x1_pos = next(i for i, d in enumerate(stream) if d["params"].get("residual") == "x1")
    stream = stream[:x1_pos] + attn_adds + stream[x1_pos:]
    # the graph computes SILU directly after the gate requant, before the up GEMM
    gate_req_pos = next(i for i, d in enumerate(stream) if d["params"].get("mlp") == "gate_req")
    dump = stream[:gate_req_pos + 1] + silu_caps + stream[gate_req_pos + 1:]
    assert len(dump) == 310, f"expected 310 captured nodes, got {len(dump)}"
    x = np.rint(hidden * (1 << 20)).astype(np.int64)
    print("captured 310 nodes (attn chain spliced into graph order); anchor bit-identical")

    # --- map onto the converted real-block graph ------------------------------
    chosen = {k: {"step_fx": v["step_fx"], "percentile": v["percentile"], "mse": v["mse"],
                  "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
              for k, v in plan["sites"].items() if k not in ("k_heads", "q_heads", "p")}
    chosen["p"] = {"step_fx": plan["sites"]["p"]["step_fx"], "percentile": 100.0, "mse": 0.0,
                   "clipped_fraction": 0.0, "calibration_max_abs": 1.0}
    for i, st in enumerate(plan["sites"]["k_heads"]):
        chosen[f"k_h{i}"] = {"step_fx": st, "percentile": 100.0, "mse": 0.0,
                             "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
    for i, st in enumerate(plan["sites"]["q_heads"]):
        chosen[f"q_h{i}"] = {"step_fx": st, "percentile": 100.0, "mse": 0.0,
                             "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
    snake, _go, _plan, _tdata, gid = build_converted_graph(conf, 16, {"chosen": chosen, "p": 1.0})
    graph_seq = [n["operator_id"] for n in snake["nodes"]]
    cap_seq = [OP_TO_GRAPH[d["op"]] for d in dump]
    bad = next((i for i, (a, b) in enumerate(zip(graph_seq, cap_seq)) if a != b), None)
    assert len(graph_seq) == len(cap_seq) and bad is None, (
        f"operator sequence mismatch at index {bad}: graph={graph_seq[bad] if bad is not None else '?'} "
        f"cap={cap_seq[bad] if bad is not None else '?'}")
    print("node mapping OK: 310-node operator sequence identical to the converted graph")
    gid_hex = gid.hex() if isinstance(gid, (bytes, bytearray)) else str(gid)
    print("GraphID:", gid_hex)

    # --- content-addressed consts + program -----------------------------------
    node_out_keys = {_arr_key(d["out"]) for d in dump}
    consts: dict[tuple, dict] = {}
    const_arrays: list[np.ndarray | None] = []

    def const_ref(a: np.ndarray) -> dict:
        k = _arr_key(a)
        if k not in consts:
            consts[k] = {"id": len(const_arrays), "shape": list(np.shape(a))}
            const_arrays.append(np.array(a, copy=True))
        return {"kind": "const", "index": consts[k]["id"]}

    program_nodes = []
    for i, d in enumerate(dump):
        ins = []
        for a in d["ins"]:
            k = _arr_key(a)
            if k in node_out_keys:
                producer = next(j for j, e in enumerate(dump) if _arr_key(e["out"]) == k)
                ins.append({"kind": "node", "index": producer})
            else:
                ins.append(const_ref(a))
        program_nodes.append({"seq": i, "op": d["op"], "params": d["params"],
                              "inputs": ins, "shape": list(np.asarray(d["out"]).shape)})

    bundle = {}
    for c, arr in zip(consts.values(), const_arrays):
        bundle[f"const_{c['id']}"] = np.ascontiguousarray(arr, dtype=np.int64)
    manifest = {f"out_{i}": np.asarray(d["out"], dtype=np.int64) for i, d in enumerate(dump)}
    np.savez_compressed(REPO / "testdata" / "f5b2_block_bundle.npz", **bundle)
    np.savez_compressed(REPO / "testdata" / "f5b2_cpu_manifest.npz", **manifest)

    nodes_meta = []
    roots_doc = {"graph_id": gid_hex, "case_index": 0, "policy_id": POLICY_ID,
                 "root_kind": "f5b2.wide.v1 (int64 values, canonical 64-element chunk merkle)",
                 "nodes": [], "final_root": None}
    for i, d in enumerate(dump):
        gnode = snake["nodes"][i]
        nodes_meta.append({"seq": i, "node_id": int(gnode["node_id"]), "op": d["op"],
                           "params": d["params"], "inputs": program_nodes[i]["inputs"],
                           "shape": program_nodes[i]["shape"],
                           "output_desc": {"dtype": int(gnode["output"]["dtype"]),
                                           "layout": int(gnode["output"]["layout"]),
                                           "shape": [int(x) for x in gnode["output"]["shape"]]}})
        roots_doc["nodes"].append({"seq": i, "node_id": int(gnode["node_id"]),
                                   "op": d["op"], "root": wide_node_root(d["out"])})
    roots_doc["final_root"] = roots_doc["nodes"][-1]["root"]
    roots_doc["final_output"] = np.asarray(dump[-1]["out"], dtype=np.int64).tolist()
    program = {"graph_id": gid_hex, "case_index": 0, "policy_id": POLICY_ID,
               "input_x_shape": [16, 1024],
               "note": "research node program derived from the frozen F.5A executor by pure "
                       "capture insertions; node ids are the converted real-block graph's "
                       "(positional operator-sequence identity, F.5B GEMM-dump precedent)",
               "consts": [{"id": c["id"], "shape": c["shape"]} for c in consts.values()],
               "nodes": nodes_meta}
    (REPO / "testdata" / "f5b2_block_program.json").write_text(
        json.dumps(program, indent=1) + "\n", encoding="utf-8")
    (REPO / "testdata" / "f5b2_cpu_roots.json").write_text(
        json.dumps(roots_doc, indent=1) + "\n", encoding="utf-8")
    hooked_path.unlink()
    from collections import Counter
    print("op counts:", dict(Counter(d["op"] for d in dump)))
    print("consts:", len(consts), "| bundle arrays:", len(bundle))
    print("written: f5b2_block_program.json, f5b2_block_bundle.npz, "
          "f5b2_cpu_manifest.npz, f5b2_cpu_roots.json")


if __name__ == "__main__":
    main()
