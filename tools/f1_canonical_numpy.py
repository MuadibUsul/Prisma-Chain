"""Vectorized numpy executor of CANONICAL_MATH_V1 operators.

Bit-exact with the scalar Python mirror (compute/canonical/python/
canonical_ref.py) and therefore with the Go reference: every rounding is
explicit ties-to-even on int64 arrays, every saturation is explicit, and
the exp/invsqrt algorithms are the same frozen integer algorithms.
`tools/f1_numpy_check.py` proves bit-exactness on the canonical vector
file and on converted real-block graphs.

This module exists only for throughput in the off-chain pipeline
(converter calibration, accuracy gate, watcher); it is never part of
consensus semantics.
"""

from __future__ import annotations

import numpy as np

FRAC_BITS = 20
ONE = 1 << FRAC_BITS
MAX_FX = 2**31 - 1
MIN_FX = -(2**31)

LN2_FX = 726817
C1_FX, C2_FX, C3_FX, C4_FX = 1048576, 524288, 174763, 43691

INV_SQRT_TABLE = np.array([
    1048576, 962239, 894228, 838860, 792649, 753319, 719317, 689539,
    663177, 639625, 618416, 599186, 581645, 565560, 550739, 537025,
], dtype=np.int64)


def rshift_round_even(v: np.ndarray, s: int) -> np.ndarray:
    """Arithmetic right shift with round-to-nearest, ties-to-even (int64)."""
    v = np.asarray(v, dtype=np.int64)
    if s == 0:
        return v
    q = v >> s
    r = v & ((1 << s) - 1)
    half = 1 << (s - 1)
    bump = (r > half) | ((r == half) & ((q & 1) == 1))
    return q + bump.astype(np.int64)


def saturate(v: np.ndarray) -> np.ndarray:
    return np.clip(v, MIN_FX, MAX_FX).astype(np.int32)


def mul_fx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return saturate(rshift_round_even(a.astype(np.int64) * b.astype(np.int64), FRAC_BITS))


def add_fx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return saturate(a.astype(np.int64) + b.astype(np.int64))


def sub_fx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return saturate(a.astype(np.int64) - b.astype(np.int64))


def _trunc_div(a: np.ndarray, b: int) -> np.ndarray:
    q = np.abs(a) // b
    return np.where(a < 0, -q, q)


def _mul_fx_wide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return rshift_round_even(a * b, FRAC_BITS)


def exp_fx(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.int64)
    out = np.zeros_like(x)
    under = x <= -(ONE * 24)
    over = x >= ONE * 21
    mid = ~(under | over)
    xi = x[mid]
    k = _trunc_div(xi, LN2_FX)
    f = xi - k * LN2_FX
    poly = ONE + _mul_fx_wide(
        f,
        C1_FX + _mul_fx_wide(
            f,
            C2_FX + _mul_fx_wide(f, C3_FX + _mul_fx_wide(f, np.full_like(f, C4_FX))),
        ),
    )
    neg = k < 0
    result = np.zeros_like(xi)
    # k < 0 branch: 2^k * poly with k in 1..31 (>= 32 underflows to 0)
    if np.any(neg):
        kk = -k[neg]
        vals = np.zeros_like(kk)
        for shift in np.unique(np.minimum(kk, 63)):
            mask = np.minimum(kk, 63) == shift
            if shift == 0:
                vals[mask] = poly[neg][mask]
            else:
                vals[mask] = rshift_round_even(poly[neg][mask], int(shift))
        vals = np.where(kk >= 32, 0, vals)
        result[neg] = vals
    pos = ~neg
    if np.any(pos):
        kp = k[pos]
        vals = np.where(kp > 31, MAX_FX, np.left_shift(poly[pos], np.minimum(kp, 62)))
        result[pos] = vals
    out[mid] = np.clip(result, MIN_FX, MAX_FX)
    out[over] = MAX_FX
    return out.astype(np.int32)


def sigmoid_fx(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.int32)
    out = np.empty_like(x, dtype=np.int64)
    nonneg = x >= 0
    if np.any(nonneg):
        den = ONE + exp_fx(-x[nonneg]).astype(np.int64)
        # DivFx(ONE, den): num is (ONE << FRAC_BITS), exactly the Go form.
        num = np.full_like(den, ONE << FRAC_BITS)
        out[nonneg] = _div_ties_even(num, den)
    if np.any(~nonneg):
        e = exp_fx(x[~nonneg]).astype(np.int64)
        den = ONE + e
        out[~nonneg] = _div_ties_even(e << FRAC_BITS, den)
    return np.clip(out, MIN_FX, MAX_FX).astype(np.int32)


def _div_ties_even(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Exact division with ties-to-even; den may be negative."""
    num = num.astype(np.int64)
    den = den.astype(np.int64)
    q = _trunc_div_arr(num, den)
    r = num - q * den
    twice = 2 * np.abs(r)
    ab = np.abs(den)
    bump = (twice > ab) | ((twice == ab) & ((q & 1) == 1))
    adj = np.where((den > 0) == (r > 0), 1, -1)
    q = q + np.where(bump & (r != 0), adj, 0)
    return q


def _trunc_div_arr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    q = np.abs(a) // np.abs(b)
    return np.where((a < 0) != (b < 0), -q, q)


def inv_sqrt_fx(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    if np.any(x <= 0):
        raise ValueError("canonical: invsqrt domain")
    b = _bit_length(x)
    target = FRAC_BITS + 2
    e = np.where(b >= target, (b - target + 1) // 2, -((target - b + 1) // 2))
    m = np.where(b >= target, x >> (2 * e), x << (-2 * e))
    adjust = m >= (ONE << 2)
    m = np.where(adjust, m >> 2, m)
    e = np.where(adjust, e + 1, e)
    idx = np.clip(((m - ONE) * 16) // (ONE * 3), 0, 15)
    y = INV_SQRT_TABLE[idx]
    for _ in range(4):
        y2 = (y * y) >> FRAC_BITS
        my2 = (m * y2) >> FRAC_BITS
        corr = (3 << (FRAC_BITS - 1)) - (my2 >> 1)
        y = (y * corr) >> FRAC_BITS
    scale_down = e > 0
    scale_up = e < 0
    y = np.where(scale_down & (e < 63), y >> np.minimum(np.maximum(e, 0), 62), y)
    y = np.where(scale_up & ((-e) < 63), y << np.minimum(np.maximum(-e, 0), 62), y)
    y = np.where(y < 1, 1, y)
    return np.clip(y, MIN_FX, MAX_FX).astype(np.int32)


def _bit_length(x: np.ndarray) -> np.ndarray:
    """64 - leading_zeros for positive int64 arrays."""
    x = x.astype(np.uint64)
    out = np.zeros_like(x, dtype=np.int64)
    mask = x > 0
    for shift in range(0, 64):
        out += (mask & ((x >> np.uint64(shift)) > 0)).astype(np.int64)
        mask &= (x >> np.uint64(shift)) > 0
    return out


# --- operator kernels (tensor level) ---------------------------------------


def op_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return add_fx(a, b)


def op_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return mul_fx(a, b)


def op_requantize(x: np.ndarray, mult: int, shift: int, lo: int, hi: int) -> np.ndarray:
    v = rshift_round_even(x.astype(np.int64) * np.int64(mult), shift)
    return np.clip(v, lo, hi).astype(np.int32)


def op_rmsnorm(x: np.ndarray, w: np.ndarray, eps_fx: int) -> np.ndarray:
    """x: [rows, hidden] q12.20; w: [hidden]; canonical chunked reduction."""
    x64 = x.astype(np.int64)
    sum_sq = ((x64 * x64) >> FRAC_BITS).sum(axis=-1, keepdims=True)
    mean = sum_sq // x.shape[-1]
    inv = inv_sqrt_fx(mean + eps_fx).astype(np.int64)
    return mul_fx(mul_fx(x64, inv), w.astype(np.int64))


def op_silu(x: np.ndarray) -> np.ndarray:
    return mul_fx(x.astype(np.int64), sigmoid_fx(x))


def op_softmax_rows(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.int64)
    mx = x64.max(axis=-1, keepdims=True)
    exps = exp_fx(sub_fx(x64, mx)).astype(np.int64)
    total = exps.sum(axis=-1, keepdims=True)
    total = np.where(total == 0, 1, total)
    return saturate(_div_ties_even(exps << FRAC_BITS, np.broadcast_to(total, exps.shape)))


def op_rope_adjacent(x: np.ndarray, table: np.ndarray, pairs_per_row: int) -> np.ndarray:
    """x: [positions, hidden] with hidden = 2*pairs; table flat [pos*pairs*2]."""
    positions, hidden = x.shape
    table = table.reshape(positions, pairs_per_row, 2)
    cos = table[:, :, 0].astype(np.int64)
    sin = table[:, :, 1].astype(np.int64)
    xp = x.reshape(positions, pairs_per_row, 2).astype(np.int64)
    even = xp[:, :, 0]
    odd = xp[:, :, 1]
    out = np.empty_like(xp)
    out[:, :, 0] = sub_fx(mul_fx(even, cos), mul_fx(odd, sin)).astype(np.int64)
    out[:, :, 1] = add_fx(mul_fx(even, sin), mul_fx(odd, cos)).astype(np.int64)
    return out.reshape(positions, hidden).astype(np.int32)


def op_gemm(a: np.ndarray, b: np.ndarray, transpose_b: bool = False) -> np.ndarray:
    """int8 x int8 -> int32 accumulator, int64 accumulation (MaxSafeK-safe)."""
    a64 = a.astype(np.int64)
    b64 = b.astype(np.int64)
    if transpose_b:
        acc = a64 @ b64.T
    else:
        acc = a64 @ b64
    assert np.all(np.abs(acc) <= 2**31 - 1), "GEMM accumulator overflow"
    return acc.astype(np.int32)


# --- commitments (bit-identical to the scalar mirror) ----------------------


def chunk_bytes_np(desc: dict, data: np.ndarray, index: int) -> bytes:
    """Canonical zero-padded chunk: 64 elements, 4-byte big-endian int32."""
    flat = np.asarray(data, dtype=np.int32).reshape(-1)
    base = index * 64
    chunk = np.zeros(64, dtype=">i4")
    take = min(64, max(0, flat.size - base))
    if take > 0:
        chunk[:take] = flat[base:base + take]
    return chunk.tobytes()


def tensor_merkle_root_np(desc: dict, data: np.ndarray) -> bytes:
    import canonical_ref as R
    desc_bytes = R.encode_canonical(desc)
    elems = int(np.prod(desc["shape"]))
    count = max(1, (elems + 63) // 64)
    leaves = [R.tensor_leaf(desc_bytes, i, chunk_bytes_np(desc, data, i)) for i in range(count)]
    return R.build_levels(leaves)[-1][0]


def tensor_root_np(desc: dict, data: np.ndarray) -> bytes:
    import canonical_ref as R
    desc_bytes = R.encode_canonical(desc)
    merkle = tensor_merkle_root_np(desc, data)
    return R.hash_bytes(R.DOMAIN_TENSOR_ROOT, desc_bytes, merkle)


# --- graph engine (mirrors canonical_ref.execute_graph bit-exactly) --------


def execute_graph_np(descriptor: dict, inputs: dict, rope_tables=None):
    """Walk a canonical graph with the vectorized kernels.

    descriptor is the snake_case form used by canonical_ref; inputs maps
    input index -> np.int32 array, rope_tables maps input index -> flat
    cos/sin table. Returns tensors, tensor roots, state trail, outputs and
    work counters exactly like the scalar executor.
    """
    import canonical_ref as R
    rope_tables = rope_tables or {}
    tensors, roots = {}, {}
    for idx, ginput in enumerate(descriptor["inputs"]):
        data = np.asarray(inputs[idx], dtype=np.int32).reshape(
            [int(d) for d in ginput["desc"]["shape"]])
        root = tensor_root_np(ginput["desc"], data)
        if root != bytes(ginput["root"]):
            raise ValueError(f"input {idx} does not match its committed root")
        tensors[(0, idx)] = data
        roots[(0, idx)] = root
    trail = [R.state_root_for(roots)]
    work = {}
    for node in descriptor["nodes"]:
        node_id = node["node_id"]
        op = node["operator_id"]
        params = {k: v for k, v in node["params"]}
        ins = [tensors[(r["kind"], r["index"])] for r in node["inputs"]]
        if op == R.OP_ADD:
            out = op_add(ins[0], ins[1])
        elif op == R.OP_MUL:
            out = op_mul(ins[0], ins[1])
        elif op == R.OP_REQUANTIZE:
            out = op_requantize(ins[0], params["mult"], params["shift"],
                                params.get("clamp_lo", MIN_FX), params.get("clamp_hi", MAX_FX))
        elif op == R.OP_RMSNORM:
            shape = tuple(int(d) for d in node["output"]["shape"])
            out = op_rmsnorm(ins[0].reshape(shape), ins[1], params["eps_fx"])
        elif op == R.OP_SILU:
            out = op_silu(ins[0])
        elif op == R.OP_SOFTMAX:
            shape = tuple(int(d) for d in node["output"]["shape"])
            out = op_softmax_rows(ins[0].reshape(shape))
        elif op == R.OP_ROPE:
            shape = tuple(int(d) for d in node["output"]["shape"])
            table = rope_tables[node["inputs"][-1]["index"]]
            out = op_rope_adjacent(ins[0].reshape(-1, shape[-1]), np.asarray(table), int(shape[-1]) // 2)
        elif op == R.OP_GEMM:
            trans = params.get("transpose_b", 0) == 1
            out = op_gemm(ins[0], ins[1], trans)
        else:
            raise ValueError(f"unknown operator {op}")
        out = np.asarray(out, dtype=np.int32).reshape([int(d) for d in node["output"]["shape"]])
        tensors[(1, node_id)] = out
        roots[(1, node_id)] = tensor_root_np(node["output"], out)
        for key, value in _node_work(op, node, params, [t.shape for t in ins]):
            work[key] = work.get(key, 0) + value
        trail.append(R.state_root_for(roots))
    outputs = [roots[(r["kind"], r["index"])] for r in descriptor["outputs"]]
    return {"tensors": tensors, "roots": roots, "trail": trail, "outputs": outputs,
            "work": sorted(work.items())}


def _node_work(op: str, node: dict, params: dict, in_shapes):
    """Work counters from the static descriptor (same keys as Go/Python)."""
    elems = 1
    for dim in node["output"]["shape"]:
        elems *= int(dim)
    if op == "GEMM_INT8_V1":
        a = in_shapes[0]
        b = in_shapes[1]
        m, k = int(a[0]), int(a[1])
        if params.get("transpose_b", 0) == 1:
            n = int(b[0])
        else:
            n = int(b[1])
        return [("GEMM_MAC", m * n * k)]
    return {
        "ADD_FIXED_V1": [("ADD_ELEMENT", elems)],
        "MUL_FIXED_V1": [("MUL_ELEMENT", elems)],
        "REQUANTIZE_V1": [("REQUANTIZE_ELEMENT", elems)],
        "RMSNORM_FIXED_V1": [("RMSNORM_ELEMENT", elems), ("RMSNORM_REDUCTION", elems)],
        "ROPE_FIXED_V1": [("ROPE_PAIR", elems // 2)],
        "SILU_FIXED_V1": [("SILU_ELEMENT", elems)],
        "SOFTMAX_FIXED_V1": [("SOFTMAX_ELEMENT", elems), ("SOFTMAX_EXP", elems)],
    }[op]
