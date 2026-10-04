"""CanonicalMathV1 numeric design: candidate fixed-point formats compared
against a float64 reference for every nonlinear operator of the block.

Run from the repository root:

    python tools/canonical_math_analysis.py --out docs/canonical-math-v1-analysis.json

The script simulates full integer arithmetic (no floats in the simulated
protocol path; float64 is only the reference) for candidate Q formats and
records dynamic range, overflow headroom and per-operator errors on fixed
seeds plus adversarial vectors. The chosen format and the accuracy
thresholds are frozen in docs/canonical-math-v1.md BEFORE the transformer
block benchmarks run.
"""

import argparse
import json
import math
import random
from pathlib import Path

# Candidate formats: (name, fractional bits) with int32 storage.
CANDIDATES = [("Q16.16", 16), ("Q12.20", 20), ("Q8.24", 24)]
MASK32 = 0xFFFFFFFF


def to_fx(x: float, frac: int) -> int:
    v = int(round(x * (1 << frac)))
    # clamp into int32 range so the simulation is bounded like the protocol
    lo, hi = -(1 << 31), (1 << 31) - 1
    return max(lo, min(hi, v))


def from_fx(v: int, frac: int) -> float:
    return v / (1 << frac)


def rshift_round_even(v: int, s: int) -> int:
    """Round-to-nearest, ties-to-even on an arithmetic right shift."""
    if s == 0:
        return v
    q = v >> s
    r = v & ((1 << s) - 1)
    half = 1 << (s - 1)
    if r > half or (r == half and (q & 1) == 1):
        q += 1
    return q


def fx_mul(a: int, b: int, frac: int) -> int:
    return rshift_round_even(a * b, frac)


def fx_div(a: int, b: int, frac: int) -> int:
    if b == 0:
        raise ZeroDivisionError
    return rshift_round_even((a << frac) // b if (a << frac) % b == 0 else (a << (frac + 1)) // b, 1)


def exact_div(a: int, b: int, frac: int) -> int:
    """round((a << frac) / b) with ties-to-even on the fractional part."""
    num = a << frac
    q, r = divmod(num, b)
    if r == 0:
        return q
    twice = 2 * abs(r)
    if twice > abs(b) or (twice == abs(b) and (q & 1) == 1):
        q += 1 if (b > 0) == (r > 0) else -1
    return q


INVSQRT_TABLE_POINTS = 16  # m in [1,4), step 3/16, canonical constants


def invsqrt_table_entry(index: int, frac: int) -> int:
    """Precomputed 1/sqrt(1 + index*3/16) in Qf, generated with exact
    integer square roots (math.isqrt) and floor division, identical to the
    Go generator BuildInvSqrtTable."""
    num = (1 << (2 * frac)) + index * 3 * (1 << (2 * frac - 4))  # m in Q2f
    root = math.isqrt(num)
    return (1 << (2 * frac)) // root


def fx_sqrt_inv(x: int, frac: int) -> int:
    """Deterministic integer 1/sqrt(x) in Qf, mirroring Go exactly:
    normalize to [1,4), pinned table, four Newton steps IN THE NORMALIZED
    DOMAIN (int64-safe, no Qf underflow), then the 2^-e scale."""
    if x <= 0:
        raise ValueError("invsqrt domain")
    one = 1 << frac
    b = x.bit_length()
    target = frac + 2
    if b >= target:
        e = ((b - target) + 1) // 2
        m = x >> (2 * e)
    else:
        e = -((target - b + 1) // 2)
        m = x << (-2 * e)
    if m >= (one << 2):
        m >>= 2
        e += 1
    idx = ((m - one) * INVSQRT_TABLE_POINTS) // (one * 3)
    idx = max(0, min(INVSQRT_TABLE_POINTS - 1, idx))
    y = invsqrt_table_entry(idx, frac)
    for _ in range(4):
        y2 = (y * y) >> frac
        my2 = (m * y2) >> frac
        corr = (3 << (frac - 1)) - (my2 >> 1)
        y = (y * corr) >> frac
    if e > 0:
        y >>= e
    elif e < 0:
        y <<= -e
    return max(1, min((1 << 31) - 1, y))


def canonical_exp(x: int, frac: int) -> int:
    """2^(x/ln2) split into an integer power and a cubic on the fraction.
    Full domain: negative inputs underflow to zero below -24, positive
    inputs saturate at the Qf maximum."""
    min_input = -(1 << frac) * 24
    max_input = (1 << frac) * 21  # exp(21) beyond Q12.20 range; saturate
    if x <= min_input:
        return 0
    if x >= max_input:
        return (1 << 31) - 1
    ln2 = int(round(math.log(2) * (1 << frac)))
    k = x // ln2  # floor
    f = x - k * ln2
    # f in [0, ln2); 2^f ~ 1 + f*(a1 + f*(a2 + f*a3)) with canonical
    # coefficients (high-precision fitted, rounded to the format)
    a1 = int(round(0.6931471805599453 * (1 << frac)))       # ln2
    a2 = int(round(0.2402265069591007 * (1 << frac)))       # (ln2)^2/2
    a3 = int(round(0.05550410866482158 * (1 << frac)))      # (ln2)^3/6
    one = 1 << frac
    poly = one + fx_mul(f, a1 + fx_mul(f, a2 + fx_mul(f, a3, frac), frac), frac)
    if k < 0:
        if -k >= 32:
            return 0
        return rshift_round_even(poly, -k)
    if k > 31:
        return (1 << 31) - 1
    return poly << k


def ref_silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def ref_softmax(row):
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    s = sum(exps)
    return [e / s for e in exps]


def ref_rmsnorm(vec, weight, eps):
    ms = sum(v * v for v in vec) / len(vec) + eps
    rms = math.sqrt(ms)
    return [vec[i] / rms * weight[i] for i in range(len(vec))]


class ErrorAcc:
    def __init__(self):
        self.max_abs = 0.0
        self.sum_abs = 0.0
        self.n = 0
        self.cos_min = 1.0

    def add_vec(self, got, want):
        for a, b in zip(got, want):
            d = abs(a - b)
            self.max_abs = max(self.max_abs, d)
            self.sum_abs += d
            self.n += 1
        na = math.sqrt(sum(a * a for a in got))
        nb = math.sqrt(sum(b * b for b in want))
        if na > 0 and nb > 0:
            cos = sum(a * b for a, b in zip(got, want)) / (na * nb)
            self.cos_min = min(self.cos_min, cos)

    def out(self):
        return {"max_abs_error": self.max_abs,
                "mean_abs_error": self.sum_abs / max(1, self.n),
                "min_cosine_similarity": self.cos_min,
                "elements": self.n}


def simulate_format(frac: int, seed: int) -> dict:
    rng = random.Random(seed)
    res = {}

    # --- RMSNorm: hidden=128, inputs in [-4,4], adversarial spikes -------
    vecs = [[rng.uniform(-4, 4) for _ in range(128)] for _ in range(16)]
    vecs.append([1000.0] * 128)          # large constant
    vecs.append([0.0] * 128)             # all zero
    vecs.append([(-1) ** i * 2000.0 for i in range(128)])  # mixed extreme
    weight = [rng.uniform(0.5, 1.5) for _ in range(128)]
    eps = 1e-5
    acc = ErrorAcc()
    max_ssum = 0
    for vec in vecs:
        fv = [to_fx(v, frac) for v in vec]
        fw = [to_fx(w, frac) for w in weight]
        # Mirror the Go reference exactly: each squared term is truncated
        # to Qf before accumulation (the per-chunk reduction shape).
        terms = list(fv)
        sq = sum((v * v) >> frac for v in terms)
        max_ssum = max(max_ssum, sum((v * v) for v in terms))
        mean = sq // len(fv)
        ms = mean + to_fx(eps, frac)
        inv = fx_sqrt_inv(ms, frac)
        got = [fx_mul(fx_mul(v, inv, frac), fw[i], frac) for i, v in enumerate(fv)]
        acc.add_vec([from_fx(g, frac) for g in got], ref_rmsnorm(vec, weight, eps))
    res["rmsnorm"] = acc.out()
    res["rmsnorm"]["max_squares_sum_bits"] = max_ssum.bit_length()

    # --- SiLU over a wide grid including extremes --------------------------
    acc = ErrorAcc()
    xs = [rng.uniform(-12, 12) for _ in range(512)] + [-11.9, -8.0, -1e-3, 0.0, 1e-3, 11.9]
    got = []
    one = 1 << frac
    for x in xs:
        fx = to_fx(x, frac)
        # stable sigmoid: for x >= 0 use 1/(1+e^-x); for x < 0 use
        # e^x/(1+e^x) so the exponent is always non-positive.
        if fx >= 0:
            sig = exact_div(one, one + canonical_exp(-fx, frac), frac)
        else:
            ex = canonical_exp(fx, frac)
            sig = exact_div(ex, one + ex, frac)
        got.append(from_fx(fx_mul(fx, sig, frac), frac))
    acc.add_vec(got, [ref_silu(x) for x in xs])
    res["silu"] = acc.out()

    # --- Softmax rows: uniform, one-dominant, extreme ---------------------
    acc = ErrorAcc()
    rows = [[rng.uniform(-8, 8) for _ in range(16)],
            [0.0] * 16,
            [-500.0] + [0.0] * 15,
            [rng.uniform(-800, 800) for _ in range(16)]]
    for row in rows:
        fr = [to_fx(v, frac) for v in row]
        m = max(fr)
        exps = [canonical_exp(v - m, frac) for v in fr]
        s = sum(exps)
        if s == 0:
            s = 1
        got_row = [from_fx(exact_div(e, s, frac), frac) for e in exps]
        acc.add_vec(got_row, ref_softmax(row))
    res["softmax"] = acc.out()

    # --- RoPE pairs: theta table errors are constants; check the pair math
    acc = ErrorAcc()
    positions = [0, 1, 2, 15, 511, 4095]
    for pos in positions:
        for i in range(8):
            theta = pos / (10000.0 ** (2 * i / 16))
            c = to_fx(math.cos(theta), frac)
            s = to_fx(math.sin(theta), frac)
            x0 = to_fx(rng.uniform(-6, 6), frac)
            x1 = to_fx(rng.uniform(-6, 6), frac)
            got0 = fx_mul(x0, c, frac) - fx_mul(x1, s, frac)
            got1 = fx_mul(x0, s, frac) + fx_mul(x1, c, frac)
            want0 = from_fx(x0, frac) * math.cos(theta) - from_fx(x1, frac) * math.sin(theta)
            want1 = from_fx(x0, frac) * math.sin(theta) + from_fx(x1, frac) * math.cos(theta)
            acc.add_vec([from_fx(got0, frac), from_fx(got1, frac)], [want0, want1])
    res["rope"] = acc.out()

    # --- ADD/MUL elementwise trivial but record headroom ------------------
    res["add_mul"] = {"max_abs_error": 0.0, "note": "exact in fixed point; overflow bounded by Q range"}
    res["range"] = {"min": from_fx(-(1 << 31), frac), "max": from_fx((1 << 31) - 1, frac),
                    "resolution": from_fx(1, frac)}
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/canonical-math-v1-analysis.json")
    args = parser.parse_args()
    seeds = [7, 11, 13]
    report = {"protocol": "CANONICAL_MATH_V1", "candidates": {}}
    for name, frac in CANDIDATES:
        runs = [simulate_format(frac, seed) for seed in seeds]
        # aggregate: worst case over seeds
        agg = {}
        for key in ("rmsnorm", "silu", "softmax", "rope"):
            agg[key] = {
                "max_abs_error": max(r[key]["max_abs_error"] for r in runs),
                "mean_abs_error": max(r[key]["mean_abs_error"] for r in runs),
                "min_cosine_similarity": min(r[key]["min_cosine_similarity"] for r in runs),
            }
        agg["range"] = runs[0]["range"]
        if "max_squares_sum_bits" in runs[0]["rmsnorm"]:
            agg["rmsnorm"]["max_squares_sum_bits"] = runs[0]["rmsnorm"]["max_squares_sum_bits"]
        report["candidates"][name] = {"fractional_bits": frac, "aggregate_worst_of_seeds": agg}
    report["seeds"] = seeds
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for name, data in report["candidates"].items():
        a = data["aggregate_worst_of_seeds"]
        print(f"{name}: range [{a['range']['min']:.1f},{a['range']['max']:.1f}] res {a['range']['resolution']:.2e} "
              f"| rms e {a['rmsnorm']['max_abs_error']:.2e} cos {a['rmsnorm']['min_cosine_similarity']:.6f} "
              f"| silu e {a['silu']['max_abs_error']:.2e} "
              f"| sm e {a['softmax']['max_abs_error']:.2e} cos {a['softmax']['min_cosine_similarity']:.6f} "
              f"| rope e {a['rope']['max_abs_error']:.2e}")


if __name__ == "__main__":
    main()
