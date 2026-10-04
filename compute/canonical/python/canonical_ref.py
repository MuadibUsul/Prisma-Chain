"""Bit-exact Python mirror of CANONICAL_MATH_V1 and CANONICAL_GRAPH_V1.

Independent implementation of compute/canonical (Go). Every routine below
follows the frozen algorithms in docs/canonical-math-v1.md exactly:

* Q12.20 in a signed int32, ties-to-even rounding, saturating adds;
* no libm at runtime: exp is the frozen k*ln2 split (truncating division,
  f in (-ln2, ln2)) with the four-term e^f Taylor polynomial; invsqrt uses
  the pinned 16-point table and four Newton steps on the normalized
  domain [1,4); RoPE sin/cos come from the committed table only;
* tensor roots: 64-element chunks of big-endian int32 values, zero
  padding, odd-node self-pairing Merkle tree, descriptor-bound domains;
* GEMM_INT8_V1 arithmetic is the frozen v0.1.1 int8 x int8 -> int32
  accumulator semantics with the K <= MaxSafeK admission.

The mirror exists so an independent party (watcher, converter, auditor)
can reproduce every commitment byte for byte.
"""

import hashlib
import struct
from math import isqrt  # exact integer square root

# --- canonical CBOR ---------------------------------------------------------


class CanonicalCBORSError(ValueError):
    pass


def _head(major: int, val: int) -> bytes:
    m = major << 5
    if val < 24:
        return bytes([m | val])
    if val < 0x100:
        return bytes([m | 24, val])
    if val < 0x10000:
        return bytes([m | 25]) + val.to_bytes(2, "big")
    if val < 0x100000000:
        return bytes([m | 26]) + val.to_bytes(4, "big")
    return bytes([m | 27]) + val.to_bytes(8, "big")


def _encode(value) -> bytes:
    if isinstance(value, bool) or value is None:
        raise CanonicalCBORSError(f"unsupported canonical CBOR type: {type(value)!r}")
    if isinstance(value, int):
        if value >= 0:
            if value > 2**64 - 1:
                raise CanonicalCBORSError("uint64 overflow")
            return _head(0, value)
        if value < -(2**63):
            raise CanonicalCBORSError("int64 underflow")
        return _head(1, -value - 1)
    if isinstance(value, (bytes, bytearray)):
        return _head(2, len(value)) + bytes(value)
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return _head(3, len(raw)) + raw
    if isinstance(value, (list, tuple)):
        out = [_head(4, len(value))]
        for item in value:
            out.append(_encode(item))
        return b"".join(out)
    if isinstance(value, dict):
        pairs = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalCBORSError("map keys must be text strings")
            pairs.append((_encode(key), _encode(item)))
        pairs.sort(key=lambda kv: kv[0])
        out = [_head(5, len(pairs))]
        for enc_key, enc_val in pairs:
            out.append(enc_key)
            out.append(enc_val)
        return b"".join(out)
    raise CanonicalCBORSError(f"unsupported canonical CBOR type: {type(value)!r}")


def encode_canonical(value) -> bytes:
    return _encode(value)


# --- domains and hashes -----------------------------------------------------

DOMAIN_TENSOR = b"PRISMA_CANONICAL_TENSOR_V1\x00"
DOMAIN_TENSOR_ROOT = b"PRISMA_CANONICAL_TENSOR_ROOT_V1\x00"
DOMAIN_GRAPH = b"PRISMA_CANONICAL_GRAPH_V1\x00"
DOMAIN_GRAPH_STATE = b"PRISMA_CANONICAL_GRAPH_STATE_V1\x00"
DOMAIN_GRAPH_TRACE = b"PRISMA_CANONICAL_GRAPH_TRACE_V1\x00"
DOMAIN_VWR = b"PRISMA_GRAPH_VWR_V1\x00"
DOMAIN_RECEIPT = b"PRISMA_GRAPH_RECEIPT_V1\x00"
DOMAIN_SIG = b"PRISMA_CANONICAL_SIG_V1\x00"
DOMAIN_WORK_VECTOR = b"PRISMA_CANONICAL_WORK_VECTOR_V1\x00"

GRAPH_PROTOCOL_VERSION = "1.0.0"
MATH_VERSION = "CANONICAL_MATH_V1"
FRAC_BITS = 20
ONE = 1 << FRAC_BITS
MAX_FX = 2**31 - 1
MIN_FX = -(2**31)
CHUNK_ELEMS = 64

DTYPE_INT8 = 1
DTYPE_INT32_ACCUM = 2
DTYPE_Q12_20 = 3
LAYOUT_ROW_MAJOR = 1

OP_GEMM = "GEMM_INT8_V1"
OP_ADD = "ADD_FIXED_V1"
OP_MUL = "MUL_FIXED_V1"
OP_REQUANTIZE = "REQUANTIZE_V1"
OP_RMSNORM = "RMSNORM_FIXED_V1"
OP_ROPE = "ROPE_FIXED_V1"
OP_SILU = "SILU_FIXED_V1"
OP_SOFTMAX = "SOFTMAX_FIXED_V1"

VERSION_GEMM = "0.1.1"
VERSION_FXFUSION = "1.0.0"

NORM_CHUNK = 16
MAX_SAFE_K = 131071  # 2^31-1 // (128*128), frozen v0.1.1


def hash_bytes(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def _u32be(v: int) -> bytes:
    return struct.pack(">I", v)


def canonical_object_hash(domain: bytes, value) -> bytes:
    return hash_bytes(domain, encode_canonical(value))


# --- CanonicalMathV1 --------------------------------------------------------


def saturate(v: int) -> int:
    if v > MAX_FX:
        return MAX_FX
    if v < MIN_FX:
        return MIN_FX
    return v


def rshift_round_even(v: int, s: int) -> int:
    if s == 0:
        return v
    q = v >> s
    r = v & ((1 << s) - 1)
    half = 1 << (s - 1)
    if r > half or (r == half and q & 1 == 1):
        q += 1
    return q


def mul_fx(a: int, b: int) -> int:
    return saturate(rshift_round_even(a * b, FRAC_BITS))


def add_fx(a: int, b: int) -> int:
    return saturate(a + b)


def sub_fx(a: int, b: int) -> int:
    return saturate(a - b)


def div_fx(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("canonical: division by zero is a spec violation")
    num = a << FRAC_BITS
    q = _trunc_div(num, b)
    r = num - q * b  # truncated remainder, same sign as num
    if r != 0:
        twice = 2 * abs(r)
        ab = abs(b)
        if twice > ab or (twice == ab and q & 1 == 1):
            if (b > 0) == (r > 0):
                q += 1
            else:
                q -= 1
    return saturate(q)


LN2_FX = 726817
C1_FX = 1048576
C2_FX = 524288
C3_FX = 174763
C4_FX = 43691


def _trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def exp_fx(x: int) -> int:
    xi = x
    if xi <= -(ONE * 24):
        return 0
    if xi >= ONE * 21:
        return MAX_FX
    k = _trunc_div(xi, LN2_FX)
    f = xi - k * LN2_FX
    poly = ONE + rshift_round_even(
        f * (C1_FX + rshift_round_even(
            f * (C2_FX + rshift_round_even(
                f * (C3_FX + rshift_round_even(f * C4_FX, FRAC_BITS)),
                FRAC_BITS)),
            FRAC_BITS)),
        FRAC_BITS,
    )
    if k < 0:
        if -k >= 32:
            return 0
        return saturate(rshift_round_even(poly, -k))
    if k > 31:
        return MAX_FX
    return saturate(poly << k)


def sigmoid_fx(x: int) -> int:
    if x >= 0:
        den = ONE + exp_fx(-x)
        return div_fx(ONE, den)
    e = exp_fx(x)
    den = ONE + e
    num = e << FRAC_BITS
    q = num // den
    r = num % den
    if r != 0:
        if 2 * abs(r) > abs(den) or (2 * abs(r) == abs(den) and q & 1 == 1):
            q += 1
    return saturate(q)


INV_SQRT_TABLE_POINTS = 16
INV_SQRT_TABLE = [
    1048576, 962239, 894228, 838860, 792649, 753319, 719317, 689539,
    663177, 639625, 618416, 599186, 581645, 565560, 550739, 537025,
]


def build_inv_sqrt_table():
    """Regenerate the pinned table with exact integer square roots."""
    out = []
    for i in range(INV_SQRT_TABLE_POINTS):
        m_num = (1 << (2 * FRAC_BITS)) + i * 3 * (1 << (2 * FRAC_BITS - 4))
        root = isqrt(m_num)
        out.append((1 << (2 * FRAC_BITS)) // root)
    return out


def inv_sqrt_fx(x: int) -> int:
    if x <= 0:
        raise ValueError("canonical: invsqrt domain (input must be positive)")
    b = x.bit_length()  # Go: 64 - leadingZeros64(x) for x > 0
    target = FRAC_BITS + 2
    if b >= target:
        shift = b - target
        e = (shift + 1) // 2
        m = x >> (2 * e)
    else:
        e = -((target - b + 1) // 2)
        m = x << (-2 * e)
    if m >= ONE << 2:
        m >>= 2
        e += 1
    idx = ((m - ONE) * INV_SQRT_TABLE_POINTS) // (ONE * 3)
    idx = max(0, min(INV_SQRT_TABLE_POINTS - 1, idx))
    y = INV_SQRT_TABLE[idx]
    for _ in range(4):
        y2 = (y * y) >> FRAC_BITS
        my2 = (m * y2) >> FRAC_BITS
        corr = (3 << (FRAC_BITS - 1)) - (my2 >> 1)
        y = (y * corr) >> FRAC_BITS
    if e > 0:
        if e >= 63:
            return 1
        y >>= e
    elif e < 0:
        if -e < 63:
            y <<= -e
    if y < 1:
        y = 1
    return saturate(y)


def sin_cos_fx(table, position: int, freq_index: int, pairs_per_row: int):
    idx = (position * pairs_per_row + freq_index) * 2
    if idx + 1 >= len(table):
        raise ValueError("canonical: RoPE position out of table range")
    return table[idx], table[idx + 1]


def attention_scale_fx(head_dim: int) -> int:
    """floor(2^20 / sqrt(head_dim)) with integer-only arithmetic."""
    s = isqrt(head_dim << 40)
    return (1 << 40) // s


# --- tensors ----------------------------------------------------------------


def new_desc(dtype: int, *shape: int) -> dict:
    return {"dtype": dtype, "layout": LAYOUT_ROW_MAJOR, "shape": list(shape)}


def desc_elems(desc: dict) -> int:
    total = 1
    for dim in desc["shape"]:
        total *= dim
    return total


def chunk_count(desc: dict) -> int:
    elems = desc_elems(desc)
    return max(1, (elems + CHUNK_ELEMS - 1) // CHUNK_ELEMS)


def chunk_bytes(desc: dict, data, index: int) -> bytes:
    buf = bytearray(CHUNK_ELEMS * 4)
    for i in range(CHUNK_ELEMS):
        pos = index * CHUNK_ELEMS + i
        v = 0
        if pos < len(data):
            v = data[pos] & 0xFFFFFFFF
        struct.pack_into(">I", buf, i * 4, v)
    return bytes(buf)


def tensor_leaf(desc_bytes: bytes, index: int, chunk: bytes) -> bytes:
    return hash_bytes(DOMAIN_TENSOR, desc_bytes, _u32be(index), chunk)


def build_levels(leaves):
    if not leaves:
        raise ValueError("canonical: merkle tree needs at least one leaf")
    level = list(leaves)
    levels = [level]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(hash_bytes(left, right))
        levels.append(nxt)
        level = nxt
    return levels


def tensor_merkle_root(desc: dict, data) -> bytes:
    desc_bytes = encode_canonical(desc)
    leaves = [
        tensor_leaf(desc_bytes, i, chunk_bytes(desc, data, i))
        for i in range(chunk_count(desc))
    ]
    return build_levels(leaves)[-1][0]


def tensor_root(desc: dict, data) -> bytes:
    desc_bytes = encode_canonical(desc)
    merkle = tensor_merkle_root(desc, data)
    return hash_bytes(DOMAIN_TENSOR_ROOT, desc_bytes, merkle)


def tensor_root_of(desc: dict, data) -> bytes:
    return tensor_root(desc, data)


# --- operators --------------------------------------------------------------

VALID_IN = (DTYPE_INT32_ACCUM, DTYPE_Q12_20)


def _require_q12(desc: dict):
    if desc["dtype"] != DTYPE_Q12_20:
        raise ValueError(f"canonical: operator requires q12.20 tensors, got {desc['dtype']}")


def _same_desc(a: dict, b: dict) -> bool:
    return a == b


def op_add_fixed(inputs, params):
    a, b = inputs
    if not _same_desc(a["desc"], b["desc"]):
        raise ValueError("canonical: ADD inputs must share dtype, layout and shape")
    return a["desc"], [add_fx(x, y) for x, y in zip(a["data"], b["data"])]


def op_mul_fixed(inputs, params):
    a, b = inputs
    if not _same_desc(a["desc"], b["desc"]):
        raise ValueError("canonical: MUL inputs must share dtype, layout and shape")
    return a["desc"], [mul_fx(x, y) for x, y in zip(a["data"], b["data"])]


def _param(params, key, default):
    for k, v in params:
        if k == key:
            return v
    return default


def op_requantize(inputs, params):
    (inp,) = inputs
    if inp["desc"]["dtype"] not in VALID_IN:
        raise ValueError("canonical: REQUANTIZE input must be accumulator or q12.20")
    mult = _param(params, "mult", 0)
    shift = _param(params, "shift", -1)
    lo = _param(params, "clamp_lo", 0)
    hi = _param(params, "clamp_hi", 1)
    out_dtype = _param(params, "out_dtype", 0)
    if mult == 0 or shift < 0 or shift > 62 or lo >= hi:
        raise ValueError("canonical: bad REQUANTIZE params")
    if out_dtype == 1 and (lo < -128 or hi > 127):
        raise ValueError("canonical: int8 clamps out of range")
    if out_dtype not in (0, 1):
        raise ValueError("canonical: bad out_dtype")
    data = []
    for x in inp["data"]:
        v = rshift_round_even(x * mult, shift)
        v = max(lo, min(hi, v))
        data.append(v)
    dtype = DTYPE_INT8 if out_dtype == 1 else DTYPE_Q12_20
    return new_desc(dtype, *inp["desc"]["shape"]), data


def _rms_for(row, eps_fx):
    total = 0
    for v in row:
        total += (v * v) >> FRAC_BITS
    mean = total // len(row)
    return inv_sqrt_fx(mean + eps_fx)


def op_rmsnorm(inputs, params):
    x, w = inputs
    eps_fx = _param(params, "eps_fx", 0)
    hidden = x["desc"]["shape"][-1]
    rows = len(x["data"]) // hidden
    out = []
    for r in range(rows):
        row = x["data"][r * hidden : (r + 1) * hidden]
        inv = _rms_for(row, eps_fx)
        for i in range(hidden):
            out.append(mul_fx(mul_fx(row[i], inv), w["data"][i % len(w["data"])]))
    return x["desc"], out


def op_silu(inputs, params):
    (x,) = inputs
    return x["desc"], [mul_fx(v, sigmoid_fx(v)) for v in x["data"]]


def _softmax_row(row):
    mx = max(row)
    exps = [exp_fx(sub_fx(v, mx)) for v in row]
    total = sum(exps)
    if total == 0:
        total = 1
    out = []
    for e in exps:
        num = e << FRAC_BITS
        q = num // total
        r = num % total
        if r != 0:
            two = 2 * abs(r)
            if two > total or (two == total and q & 1 == 1):
                q += 1
        out.append(saturate(q))
    return out


def op_softmax(inputs, params):
    (x,) = inputs
    hidden = x["desc"]["shape"][-1]
    rows = len(x["data"]) // hidden
    out = []
    for r in range(rows):
        out.extend(_softmax_row(x["data"][r * hidden : (r + 1) * hidden]))
    return x["desc"], out


def op_rope(inputs, params, rope_table=None):
    x = inputs[0]
    if rope_table is None:
        raise ValueError("canonical: ROPE requires the pinned table")
    half_dim = _param(params, "half_dim", 0)
    hidden = x["desc"]["shape"][-1]
    if hidden != 2 * half_dim:
        raise ValueError("canonical: ROPE hidden must equal 2*half_dim")
    pairs_per_row = hidden // 2
    positions = len(x["data"]) // hidden
    out = [0] * len(x["data"])
    for pos in range(positions):
        for pair in range(pairs_per_row):
            cos_fx, sin_fx = sin_cos_fx(rope_table, pos, pair, pairs_per_row)
            even = x["data"][pos * hidden + 2 * pair]
            odd = x["data"][pos * hidden + 2 * pair + 1]
            out[pos * hidden + 2 * pair] = sub_fx(mul_fx(even, cos_fx), mul_fx(odd, sin_fx))
            out[pos * hidden + 2 * pair + 1] = add_fx(mul_fx(even, sin_fx), mul_fx(odd, cos_fx))
    return x["desc"], out


def _int8_range_check(t):
    for v in t["data"]:
        if v < -128 or v > 127:
            raise ValueError("canonical: int8 tensor holds out-of-range value")


def gemm_dims(a_desc, b_desc, params):
    if len(a_desc["shape"]) != 2 or len(b_desc["shape"]) != 2:
        raise ValueError("canonical: GEMM operands must be 2-D")
    trans_b = _param(params, "transpose_b", 0) == 1
    m = a_desc["shape"][0]
    k = a_desc["shape"][1]
    if trans_b:
        n = b_desc["shape"][0]
        if b_desc["shape"][1] != k:
            raise ValueError("canonical: GEMM transpose_b needs B [N,K]")
    else:
        n = b_desc["shape"][1]
        if b_desc["shape"][0] != k:
            raise ValueError("canonical: GEMM needs A [M,K] and B [K,N]")
    if k > MAX_SAFE_K:
        raise ValueError("canonical: GEMM K exceeds MaxSafeK")
    return m, n, k, trans_b


def op_gemm(inputs, params):
    a, b = inputs
    if a["desc"]["dtype"] != DTYPE_INT8 or b["desc"]["dtype"] != DTYPE_INT8:
        raise ValueError("canonical: GEMM operands must be int8")
    _int8_range_check(a)
    _int8_range_check(b)
    m, n, k, trans_b = gemm_dims(a["desc"], b["desc"], params)
    out = []
    for i in range(m):
        for j in range(n):
            acc = 0
            for d in range(k):
                bv = b["data"][j * k + d] if trans_b else b["data"][d * n + j]
                acc += a["data"][i * k + d] * bv
            if acc > 2**31 - 1 or acc < -(2**31):
                raise ValueError("canonical: GEMM accumulator overflow")
            out.append(acc)
    return new_desc(DTYPE_INT32_ACCUM, m, n), out


OPERATORS = {
    OP_ADD: op_add_fixed,
    OP_MUL: op_mul_fixed,
    OP_REQUANTIZE: op_requantize,
    OP_RMSNORM: op_rmsnorm,
    OP_ROPE: op_rope,
    OP_SILU: op_silu,
    OP_SOFTMAX: op_softmax,
    OP_GEMM: op_gemm,
}


def op_work(op_id: str, inputs, params):
    """Mirror of the Go WorkVector counters."""
    if op_id == OP_ADD:
        return [("ADD_ELEMENT", len(inputs[0]["data"]))]
    if op_id == OP_MUL:
        return [("MUL_ELEMENT", len(inputs[0]["data"]))]
    if op_id == OP_REQUANTIZE:
        return [("REQUANTIZE_ELEMENT", len(inputs[0]["data"]))]
    if op_id == OP_RMSNORM:
        n = len(inputs[0]["data"])
        return [("RMSNORM_ELEMENT", n), ("RMSNORM_REDUCTION", n)]
    if op_id == OP_SILU:
        return [("SILU_ELEMENT", len(inputs[0]["data"]))]
    if op_id == OP_SOFTMAX:
        n = len(inputs[0]["data"])
        return [("SOFTMAX_ELEMENT", n), ("SOFTMAX_EXP", n)]
    if op_id == OP_ROPE:
        return [("ROPE_PAIR", len(inputs[0]["data"]) // 2)]
    if op_id == OP_GEMM:
        m, n, k, _ = gemm_dims(inputs[0]["desc"], inputs[1]["desc"], params)
        return [("GEMM_MAC", m * n * k)]
    raise ValueError(f"canonical: no work counters for {op_id}")


def op_out_spec(op_id: str, in_descs, params):
    if op_id == OP_ADD or op_id == OP_MUL:
        return dict(in_descs[0])
    if op_id == OP_REQUANTIZE:
        out_dtype = _param(params, "out_dtype", 0)
        dtype = DTYPE_INT8 if out_dtype == 1 else DTYPE_Q12_20
        return new_desc(dtype, *in_descs[0]["shape"])
    if op_id in (OP_RMSNORM, OP_SILU, OP_SOFTMAX, OP_ROPE):
        desc = dict(in_descs[0])
        if op_id == OP_ROPE:
            if desc["shape"][-1] != 2 * _param(params, "half_dim", 0):
                raise ValueError("canonical: ROPE hidden mismatch")
        return desc
    if op_id == OP_GEMM:
        m, n, _, _ = gemm_dims(in_descs[0], in_descs[1], params)
        return new_desc(DTYPE_INT32_ACCUM, m, n)
    raise ValueError(f"canonical: unknown operator {op_id}")


# --- graph ------------------------------------------------------------------


def normalize_descriptor(descriptor: dict) -> dict:
    """Render the descriptor exactly as the Go structure encodes: node
    parameters are []Pair, i.e. an array of {key, value} maps."""
    nodes = []
    for node in descriptor["nodes"]:
        normalized = dict(node)
        normalized["params"] = [{"key": k, "value": v} for k, v in node["params"]]
        nodes.append(normalized)
    out = dict(descriptor)
    out["nodes"] = nodes
    return out


def graph_id(descriptor: dict) -> bytes:
    return canonical_object_hash(DOMAIN_GRAPH, normalize_descriptor(descriptor))


def state_root_for(tensors: dict) -> bytes:
    """tensors: {(kind, index): root_bytes} for every live tensor."""
    ids = sorted((kind << 32) | index for (kind, index) in tensors)
    leaves = []
    for tid in ids:
        kind = tid >> 32
        index = tid & 0xFFFFFFFF
        leaves.append(hash_bytes(DOMAIN_GRAPH_STATE, _u32be(kind), _u32be(index), tensors[(kind, index)]))
    return build_levels(leaves)[-1][0]


def final_output_root(outputs) -> bytes:
    parts = [DOMAIN_VWR] + list(outputs)
    return hash_bytes(*parts)


def _desc_from_json(desc_json: dict) -> dict:
    return {
        "dtype": desc_json["dtype"],
        "layout": desc_json["layout"],
        "shape": list(desc_json["shape"]),
    }


def _params_from_json(params_json) -> list:
    return [(k, v) for k, v in params_json]


def execute_graph(descriptor: dict, inputs: dict, rope_tables=None) -> dict:
    """Run one graph with the reference operators.

    inputs: {input_index: {"desc":..., "data":[...]}}
    rope_tables: {input_index: [cos,sin,...] table} for ROPE nodes
    Every input must reproduce the committed root in the descriptor
    (exactly as the Go execution does).
    """
    if descriptor["protocol_version"] != GRAPH_PROTOCOL_VERSION:
        raise ValueError("canonical: graph protocol_version mismatch")
    rope_tables = rope_tables or {}
    tensors = {}
    tensor_roots = {}
    for idx, ginput in enumerate(descriptor["inputs"]):
        if idx not in inputs:
            raise ValueError(f"canonical: missing graph input {idx}")
        t = inputs[idx]
        if _desc_from_json(ginput["desc"]) != t["desc"]:
            raise ValueError(f"canonical: input {idx} descriptor mismatch")
        root = tensor_root(t["desc"], t["data"])
        if bytes(ginput["root"]) != root:
            raise ValueError(f"canonical: input {idx} does not match its committed root")
        tensors[(0, idx)] = {"desc": t["desc"], "data": t["data"]}
        tensor_roots[(0, idx)] = root

    trail = [state_root_for(tensor_roots)]
    work = {}
    for node in descriptor["nodes"]:
        node_id = node["node_id"]
        op_id = node["operator_id"]
        params = _params_from_json(node["params"])
        node_inputs = []
        for ref in node["inputs"]:
            key = (ref["kind"], ref["index"])
            if key not in tensors:
                raise ValueError("canonical: forward or missing input reference")
            node_inputs.append(tensors[key])
        if op_id == OP_ROPE:
            table = rope_tables.get(node["inputs"][-1]["index"])
            out_desc, out_data = op_rope(node_inputs, params, rope_table=table)
        else:
            out_desc, out_data = OPERATORS[op_id](node_inputs, params)
        if _desc_from_json(node["output"]) != out_desc:
            raise ValueError(f"canonical: node {node_id} declared output mismatch")
        tensors[(1, node_id)] = {"desc": out_desc, "data": out_data}
        tensor_roots[(1, node_id)] = tensor_root(out_desc, out_data)
        for key, value in op_work(op_id, node_inputs, params):
            work[key] = work.get(key, 0) + value
        trail.append(state_root_for(tensor_roots))

    outputs = []
    for ref in descriptor["outputs"]:
        key = (ref["kind"], ref["index"])
        outputs.append(tensor_roots[key])
    work_list = sorted(work.items())
    return {
        "tensors": tensors,
        "roots": tensor_roots,
        "trail": trail,
        "outputs": outputs,
        "work": work_list,
    }


# --- trails, proofs and commitments (watcher / E2E tooling) -----------------


def prove(levels, index: int):
    """Sibling path of one leaf, canonical odd-node self-pairing rule."""
    count = len(levels[0])
    if index >= count:
        raise ValueError("canonical: leaf index out of range")
    siblings = []
    idx = index
    for nodes in levels[:-1]:
        if idx % 2 == 0:
            sib = idx + 1
            if sib >= len(nodes):
                sib = idx
        else:
            sib = idx - 1
        siblings.append(nodes[sib])
        idx //= 2
    return siblings


def verify_inclusion(root: bytes, leaf: bytes, index: int, count: int, siblings) -> bool:
    if count == 0 or index >= count or len(siblings) != _depth_for(count):
        return False
    h = leaf
    idx, cnt = index, count
    for sib in siblings:
        if idx % 2 == 0:
            right = sib
            if idx + 1 >= cnt:
                right = h
            h = hash_bytes(h, right)
        else:
            h = hash_bytes(sib, h)
        idx //= 2
        cnt = (cnt + 1) // 2
    return h == root


def _depth_for(count: int) -> int:
    depth = 0
    while count > 1:
        count = (count + 1) // 2
        depth += 1
    return depth


def trail_leaf(graph_id: bytes, step: int, state_root: bytes) -> bytes:
    return hash_bytes(DOMAIN_GRAPH_TRACE, graph_id, _u32be(step), state_root)


def trail_levels(graph_id: bytes, trail):
    return build_levels([trail_leaf(graph_id, i, s) for i, s in enumerate(trail)])


def trail_root(graph_id: bytes, trail) -> bytes:
    return trail_levels(graph_id, trail)[-1][0]


def trail_proof(graph_id: bytes, trail, index: int):
    levels = trail_levels(graph_id, trail)
    return index, len(trail), prove(levels, index)


def chunk_proof(desc: dict, data, index: int):
    """Sibling path of one tensor chunk leaf."""
    desc_bytes = encode_canonical(desc)
    count = chunk_count(desc)
    leaves = [tensor_leaf(desc_bytes, i, chunk_bytes(desc, data, i)) for i in range(count)]
    return count, prove(build_levels(leaves), index)


def state_leaf_index(live, ref):
    """Index of one tensor in the sorted live-set state tree."""
    ids = sorted((kind << 32) | index for (kind, index) in live)
    return ids.index((ref[0] << 32) | ref[1]), len(ids)


def state_proof(live, ref):
    """live: {(kind, index): root_bytes}; ref: (kind, index)."""
    ids = sorted((kind << 32) | index for (kind, index) in live)
    leaves = []
    for tid in ids:
        kind, index = tid >> 32, tid & 0xFFFFFFFF
        leaves.append(hash_bytes(DOMAIN_GRAPH_STATE, _u32be(kind), _u32be(index), live[(kind, index)]))
    levels = build_levels(leaves)
    index = ids.index((ref[0] << 32) | ref[1])
    return index, len(leaves), prove(levels, index)


def graph_result_commit(graph_id: bytes, task_ref: bytes, assignment_ref: bytes,
                        worker_pubkey: bytes, outputs, completed_epoch: int):
    """The canonical signed object (signature filled by the caller)."""
    roots = [bytes(o) for o in outputs]
    return {
        "protocol_version": "1.0.0",
        "graph_id": graph_id,
        "task_ref": task_ref,
        "assignment_ref": assignment_ref,
        "worker_pubkey": worker_pubkey,
        "final_output_root": final_output_root(roots),
        "output_roots": roots,
        "completed_epoch": completed_epoch,
        "signature": b"",
    }


def graph_result_commit_preimage(commit: dict) -> bytes:
    unsigned = dict(commit)
    unsigned["signature"] = b""
    return DOMAIN_SIG + encode_canonical(unsigned)


# --- node-output manifest (Phase F.1) ---------------------------------------


DOMAIN_NODE_OUTPUT = b"PRISMA_NODE_OUTPUT_V1\x00"


def node_output_leaf(graph_id: bytes, node: dict) -> bytes:
    """One node's output commitment (descriptor-bound)."""
    desc_bytes = encode_canonical(_desc_from_json(node["output"]))
    return hash_bytes(DOMAIN_NODE_OUTPUT, graph_id, _u32be(node["node_id"]),
                      node["operator_id"].encode(), desc_bytes)


def build_node_output_manifest(graph_id: bytes, nodes, node_roots):
    """Merkle root over every node's (leaf, output TensorRoot) combined."""
    leaves = [hash_bytes(node_output_leaf(graph_id, node), root)
              for node, root in zip(nodes, node_roots)]
    return build_levels(leaves)[-1][0], leaves


def node_output_manifest_proof(graph_id: bytes, nodes, node_roots, index: int):
    _, leaves = build_node_output_manifest(graph_id, nodes, node_roots)
    return index, len(leaves), prove(build_levels(leaves), index)


def graph_result_commit_v2(graph_id: bytes, task_ref: bytes, assignment_ref: bytes,
                           worker_pubkey: bytes, manifest_root: bytes, outputs,
                           completed_epoch: int) -> dict:
    roots = [bytes(o) for o in outputs]
    return {
        "protocol_version": "2.0.0",
        "graph_id": graph_id,
        "task_ref": task_ref,
        "assignment_ref": assignment_ref,
        "worker_pubkey": worker_pubkey,
        "node_output_manifest_root": manifest_root,
        "final_output_root": final_output_root(roots),
        "output_roots": roots,
        "completed_epoch": completed_epoch,
        "signature": b"",
    }


def graph_result_commit_v2_preimage(commit: dict) -> bytes:
    unsigned = dict(commit)
    unsigned["signature"] = b""
    return DOMAIN_SIG + encode_canonical(unsigned)


# =============================================================================
# Phase F.5C — CANONICAL_TENSOR_V2 / GEMM_A13W10_I64_V1 / REQUANTIZE_WIDE_V1.
# Additive version space: every V2 hash is domain-separated from V1; the V1
# objects above are untouched.  Logical width != storage width (A13 = 13
# logical bits in a 16-bit BE container; W10 = 10 in 16; INT64_ACCUM in 64).
# =============================================================================

DOMAIN_TENSOR_V2 = b"PRISMA_CANONICAL_TENSOR_V2\x00"
DOMAIN_TENSOR_ROOT_V2 = b"PRISMA_CANONICAL_TENSOR_ROOT_V2\x00"
DOMAIN_GRAPH_V2 = b"PRISMA_CANONICAL_GRAPH_V2\x00"
DOMAIN_GRAPH_STATE_V2 = b"PRISMA_CANONICAL_GRAPH_STATE_V2\x00"
DOMAIN_GRAPH_TRACE_V2 = b"PRISMA_CANONICAL_GRAPH_TRACE_V2\x00"
DOMAIN_MANIFEST_V2 = b"PRISMA_GRAPH_NODE_MANIFEST_V2\x00"
DOMAIN_COMMIT_V3 = b"PRISMA_GRAPH_RESULT_COMMIT_V3\x00"
DOMAIN_RECEIPT_V3 = b"PRISMA_GRAPH_RECEIPT_V3\x00"
DOMAIN_WIDE_GEMM_DISPUTE = b"PRISMA_WIDE_GEMM_DISPUTE_V1\x00"
DOMAIN_WIDE_TRACE = b"PRISMA_WIDE_GEMM_TRACE_V1\x00"

PROTOCOL_VERSION_TENSOR_V2 = "CANONICAL_TENSOR_V2/1.0.0"
PROTOCOL_VERSION_GRAPH_V2 = "CANONICAL_GRAPH_V2/1.0.0"
OP_GEMM_VERSION_WIDE = "GEMM_A13W10_I64_V1/1.0.0"
OP_REQUANT_VERSION_WIDE = "REQUANTIZE_WIDE_V1/1.0.0"
ARITHMETIC_PROFILE_A13W10 = "A13W10_I64_PROFILE_V1"
FREIVALDS_WIDE_VERSION = "FREIVALDS_A13W10_I64_V1/1.0.0"
POLICY_ID_A13W10 = ("eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0")

DTYPE_V2_Q12_20 = 1
DTYPE_V2_A13 = 129
DTYPE_V2_W10 = 130
DTYPE_V2_INT64_ACCUM = 131

A13_MIN, A13_MAX = -4096, 4095
W10_MIN, W10_MAX = -512, 511

_CHUNK_ELEMS = 64


def arithmetic_profile_a13w10() -> dict:
    return {"id": ARITHMETIC_PROFILE_A13W10, "a_bits": 13, "w_bits": 10,
            "accum": "int64", "rounding": "ties_to_even", "policy_id": POLICY_ID_A13W10}


def _dtype_v2_range(dtype: int):
    if dtype == DTYPE_V2_Q12_20:
        return MIN_FX, MAX_FX
    if dtype == DTYPE_V2_A13:
        return A13_MIN, A13_MAX
    if dtype == DTYPE_V2_W10:
        return W10_MIN, W10_MAX
    if dtype == DTYPE_V2_INT64_ACCUM:
        return None, None
    raise ValueError(f"canonical/v2: unknown dtype {dtype}")


def _dtype_v2_width(dtype: int) -> int:
    if dtype == DTYPE_V2_Q12_20:
        return 4
    if dtype in (DTYPE_V2_A13, DTYPE_V2_W10):
        return 2
    if dtype == DTYPE_V2_INT64_ACCUM:
        return 8
    raise ValueError(f"canonical/v2: unknown dtype {dtype}")


def tensor_desc_v2(dtype: int, shape) -> dict:
    return {"dtype": int(dtype), "layout": 1, "shape": [int(d) for d in shape]}


def validate_values_v2(desc: dict, data) -> None:
    lo, hi = _dtype_v2_range(int(desc["dtype"]))
    if lo is None:
        return
    for i, v in enumerate(data):
        if v < lo or v > hi:
            raise ValueError(
                f"canonical/v2: element {i} value {v} outside logical range [{lo},{hi}]")


def _enc_be(v: int, width: int) -> bytes:
    return (v & ((1 << (8 * width)) - 1)).to_bytes(width, "big")


def tensor_v2_chunk_bytes(desc: dict, data, index: int) -> bytes:
    """64 logical elements at the dtype width, zero-padded (all-zero bytes
    are the canonical zero of every V2 dtype)."""
    width = _dtype_v2_width(int(desc["dtype"]))
    flat = list(data)
    base = index * _CHUNK_ELEMS
    if base >= len(flat) or index < 0:
        raise ValueError("canonical/v2: chunk index out of range")
    buf = bytearray(_CHUNK_ELEMS * width)
    for i in range(_CHUNK_ELEMS):
        if base + i >= len(flat):
            break
        buf[i * width:(i + 1) * width] = _enc_be(int(flat[base + i]), width)
    return bytes(buf)


def tensor_v2_leaf(desc_bytes: bytes, index: int, chunk: bytes) -> bytes:
    return hash_bytes(DOMAIN_TENSOR_V2, desc_bytes, _u32be(index), chunk)


def tensor_merkle_root_v2(desc: dict, data) -> bytes:
    desc_bytes = encode_canonical(desc)
    count = max(1, (len(data) + _CHUNK_ELEMS - 1) // _CHUNK_ELEMS)
    leaves = []
    for i in range(count):
        leaves.append(tensor_v2_leaf(desc_bytes, i,
                                     tensor_v2_chunk_bytes(desc, data, i)))
    level = leaves
    while len(level) > 1:
        level = [hash_bytes(level[i] + (level[i + 1] if i + 1 < len(level) else level[i]))
                 for i in range(0, len(level), 2)]
    return level[0]


def tensor_root_v2(desc: dict, data) -> bytes:
    validate_values_v2(desc, data)
    desc_bytes = encode_canonical(desc)
    return hash_bytes(DOMAIN_TENSOR_ROOT_V2, desc_bytes,
                      tensor_merkle_root_v2(desc, data))


def rshift_round_even64(v: int, s: int) -> int:
    if s == 0:
        return v
    q = v >> s
    r = v & ((1 << s) - 1)
    half = 1 << (s - 1)
    if r > half or (r == half and (q & 1) == 1):
        return q + 1
    return q


def reference_wide_gemm(a, w, m: int, n: int, k: int, transpose_b: bool = False,
                        w_a13: bool = False):
    """GEMM_A13W10_I64_V1 reference: C = sum_k A13 * W10, exact int64.
    The right operand is W10 for model weights or A13 for attention
    inner products (w_a13=True)."""
    w_bits = 13 if w_a13 else 10
    max_safe_k64 = (2**63 - 1) // (2**12 * 2**(w_bits - 1))
    if k > max_safe_k64:
        raise ValueError("canonical/v2: K exceeds MaxSafeK64")
    validate_values_v2(tensor_desc_v2(DTYPE_V2_A13, (m, k)), a)
    w_dtype = DTYPE_V2_A13 if w_a13 else DTYPE_V2_W10
    w_desc = tensor_desc_v2(w_dtype, (n, k) if transpose_b else (k, n))
    validate_values_v2(w_desc, w)
    out = [0] * (m * n)
    for i in range(m):
        for j in range(n):
            acc = 0
            for d in range(k):
                wv = w[j * k + d] if transpose_b else w[d * n + j]
                acc += int(a[i * k + d]) * int(wv)
            out[i * n + j] = acc
    return out


def requant_wide(inp, mult: int, shift: int, lo: int, hi: int, target_dtype: int):
    """REQUANTIZE_WIDE_V1: v = clamp(round_ties_even(x * mult >> shift), lo, hi)."""
    if mult == 0 or mult > (1 << 62):
        raise ValueError("canonical/v2: requant multiplier outside admission bounds")
    out = []
    for v in inp:
        prod = int(v) * mult
        if prod > 2**63 - 1 or prod < -(2**63):
            raise ValueError("canonical/v2: requant intermediate overflow")
        out.append(max(lo, min(hi, rshift_round_even64(prod, shift))))
    validate_values_v2(tensor_desc_v2(target_dtype, (len(out),)), out)
    return out


def max_safe_k64(a_bits: int, w_bits: int) -> int:
    return (2**63 - 1) // (2**(a_bits - 1) * 2**(w_bits - 1))


# --- A1-02/A1-03: GraphStateRootV2 + TrailRootV2 (Python mirror) -------------

def graph_state_root_v2(tensors: dict) -> bytes:
    """tensors: {(kind, index): TensorRootV2 bytes}; ids sorted."""
    ids = sorted((int(kind) << 32) | int(index) for (kind, index) in tensors)
    leaves = [hash_bytes(DOMAIN_GRAPH_STATE_V2, _u32be(tid >> 32),
                         _u32be(tid & 0xFFFFFFFF), tensors[(tid >> 32, tid & 0xFFFFFFFF)])
              for tid in ids]
    level = leaves
    while len(level) > 1:
        level = [hash_bytes(level[i] + (level[i + 1] if i + 1 < len(level) else level[i]))
                 for i in range(0, len(level), 2)]
    return level[0]


def trail_leaf_v2(graph_id: bytes, step: int, state_root: bytes) -> bytes:
    return hash_bytes(DOMAIN_GRAPH_TRACE_V2, graph_id, _u32be(step), state_root)


def _trail_levels_v2(graph_id: bytes, trail) -> list:
    level = [trail_leaf_v2(graph_id, i, sr) for i, sr in enumerate(trail)]
    levels = [level]
    while len(level) > 1:
        level = [hash_bytes(level[i] + (level[i + 1] if i + 1 < len(level) else level[i]))
                 for i in range(0, len(level), 2)]
        levels.append(level)
    return levels


def trail_root_v2(graph_id: bytes, trail) -> bytes:
    return _trail_levels_v2(graph_id, trail)[-1][0]


def trail_proof_v2(graph_id: bytes, trail, index: int) -> list:
    levels = _trail_levels_v2(graph_id, trail)
    siblings = []
    idx = index
    for level in levels[:-1]:
        sib = idx ^ 1
        siblings.append(level[sib] if sib < len(level) else level[idx])
        idx //= 2
    return siblings


def verify_trail_proof_v2(root: bytes, graph_id: bytes, step: int, count: int,
                          state_root: bytes, siblings) -> bool:
    if count == 0 or step >= count:
        return False
    h = trail_leaf_v2(graph_id, step, state_root)
    idx, cnt = step, count
    for sib in siblings:
        if idx % 2 == 0:
            right = sib if idx + 1 < cnt else h
            h = hash_bytes(h + right)
        else:
            h = hash_bytes(sib + h)
        idx //= 2
        cnt = (cnt + 1) // 2
    return h == root
