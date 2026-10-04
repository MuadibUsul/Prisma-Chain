"""One-shot patcher: derive tools/f5a_joint_precision.py from the F.4A executor.

EXPERIMENTAL / NON-PROTOCOL build tooling.
"""

import io

src = io.open("tools/f4a_joint_precision.py", encoding="utf-8").read()

# --- 1. int64 accumulation: replace op_gemm with an exact int64 GEMM --------
src = src.replace(
    "INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1",
    "INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1\n"
    "INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1\n\n\n"
    "def gemm64(a, b, transpose_b=False):\n"
    "    \"\"\"Exact signed GEMM with an int64 accumulator (research semantics).\n\n"
    "    The F.5A study widens ONLY the accumulator; operands stay at their\n"
    "    logical bit depths (stored in int16 research containers). The int64\n"
    "    range is ASSERTED, never wrapped.\n"
    "    \"\"\"\n"
    "    a64 = np.asarray(a).astype(np.int64)\n"
    "    b64 = np.asarray(b).astype(np.int64)\n"
    "    acc = a64 @ b64.T if transpose_b else a64 @ b64\n"
    "    lo, hi = int(acc.min()), int(acc.max())\n"
    "    assert INT64_MIN <= lo and hi <= INT64_MAX, \"int64 accumulator overflow\"\n"
    "    return acc")

src = src.replace("N.op_gemm(", "gemm64(")

# --- 2. requant accounting + exact wide-multiply fallback -------------------
src = src.replace(
    '''    def requant_acc(acc, mult, shift=20):
        gated(acc, mult, shift)
        return np.clip(N.rshift_round_even(acc.astype(np.int64) * np.int64(mult), shift), MIN_FX, MAX_FX)''',
    '''    def requant_acc(acc, mult, shift=20):
        gated(acc, mult, shift)
        m = int(np.max(np.abs(acc))) if acc.size else 0
        bits = m.bit_length() + int(mult).bit_length()
        if "requant_links" in stats:
            stats["requant_links"].append({"max_abs_acc": m, "mult": int(mult),
                                           "shift": int(shift), "product_bits": bits})
        if bits <= 62:
            return np.clip(N.rshift_round_even(acc.astype(np.int64) * np.int64(mult), shift),
                           MIN_FX, MAX_FX)
        # exact ties-to-even bigint path (research executor only; the F.5A
        # report records exactly when this is required)
        s = int(shift)
        out = []
        for v in acc.reshape(-1):
            p = int(v) * int(mult)
            q, r = p >> s, p & ((1 << s) - 1)
            half = 1 << (s - 1)
            q += int(r > half or (r == half and (q & 1) == 1))
            out.append(q)
        big = np.array(out, dtype=object).reshape(acc.shape)
        return np.clip(np.vectorize(int, otypes=[np.int64])(big), MIN_FX, MAX_FX)''')

src = src.replace(
    '''    def gated(acc, mult, shift=20, lo=None, hi=None):
        stats["acc_abs_max"] = max(stats["acc_abs_max"], int(np.max(np.abs(acc))))
        assert INT32_MIN <= int(acc.min()) and int(acc.max()) <= INT32_MAX, "int32 accumulator overflow"
        return int(acc.min()), int(acc.max())''',
    '''    def gated(acc, mult, shift=20, lo=None, hi=None):
        lo_v, hi_v = int(acc.min()), int(acc.max())
        stats["acc_abs_max"] = max(stats.get("acc_abs_max", 0), max(abs(lo_v), abs(hi_v)))
        assert INT64_MIN <= lo_v and hi_v <= INT64_MAX, "int64 accumulator overflow"
        return lo_v, hi_v''')

# --- 3. F.5A candidate family: the 15 previously int32-unsafe pairs ---------
src = src.replace(
    '''def pair_family():
    """Pre-registered 21 candidate pairs with a + w <= 21 (MaxSafeK >= 3072)."""
    pairs = []
    for a in range(8, 14):
        for w in range(8, 14):
            if a + w <= 21:
                pairs.append((a, w))
    return pairs''',
    '''def max_safe_k64(a_bits: int, w_bits: int) -> int:
    """Worst-case signed int64 bound: 2^(a+w-2) * K <= 2^63-1."""
    return ((1 << 63) - 1) // (1 << (a_bits + w_bits - 2))


def pair_family():
    """The 15 pairs excluded by int32 safety in F.4A (a+w from 22 to 26)."""
    pairs = []
    for a in range(8, 14):
        for w in range(8, 14):
            if a + w >= 22:
                pairs.append((a, w))
    return pairs''')

# --- 4. split paths ---------------------------------------------------------
src = src.replace('f"qwen3_f4a_{name}.json"', 'f"qwen3_f5a_{name}.json"')

io.open("tools/f5a_joint_precision.py", "w", encoding="utf-8", newline="\n").write(src)
print("f5a executor derived")
print("gemm64 calls:", src.count("gemm64("))
print("pairs:", len([1 for a in range(8, 14) for w in range(8, 14) if a + w >= 22]))
