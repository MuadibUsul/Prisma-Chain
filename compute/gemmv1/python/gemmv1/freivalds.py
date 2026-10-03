"""Freivalds probabilistic detection for GEMM v0.1.2, mirroring
compute/gemmv1/verify (Go).

Freivalds is a DETECTION strategy only, never a slashing proof: a mismatch
routes a challenger into the deterministic v0.1.1 tile dispute. All
arithmetic is exact signed int64 — Python integers do not wrap, and every
accumulated element is asserted to stay inside the int64 range, which the
profile admission bound guarantees.
"""

import hashlib
import secrets

from .tensors import canonical_to_int32s, int32s_to_canonical

ALGORITHM_FREIVALDS_BINARY_V1 = "FREIVALDS_BINARY_V1"
PROFILE_VERSION = "0.1.2"

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class FreivaldsError(ValueError):
    pass


class VerificationProfile:
    """{algorithm, rounds} with the 2^-rounds theoretical detection bound."""

    def __init__(self, rounds: int, algorithm: str = ALGORITHM_FREIVALDS_BINARY_V1):
        if algorithm != ALGORITHM_FREIVALDS_BINARY_V1:
            raise FreivaldsError("unknown verification algorithm")
        if rounds <= 0:
            raise FreivaldsError("rounds must be at least 1")
        self.algorithm = algorithm
        self.rounds = rounds

    def false_accept_upper_bound(self) -> float:
        return 2.0 ** (-self.rounds)

    def admit_for(self, m: int, n: int, k: int) -> None:
        """Prove the int64 overflow bound for one shape before running."""
        if m <= 0 or n <= 0 or k <= 0:
            raise FreivaldsError("matrix shape mismatch or empty dimension")
        y_bound = 16384 * n * k
        z_bound = n * (2**31 - 1)
        if y_bound + z_bound >= 2**62:
            raise FreivaldsError("shape exceeds the proven int64 Freivalds bound")


class CSPRNGSource:
    """Production randomness: os.urandom-backed."""

    def fill(self, p: bytearray) -> None:
        p[:] = secrets.token_bytes(len(p))


class DeterministicSource:
    """DEVELOPMENT ONLY: SHA-256 counter stream for tests and
    cross-language vectors; never back production challenges."""

    def __init__(self, seed: bytes):
        self.state = bytes(seed)
        self.count = 0

    def fill(self, p: bytearray) -> None:
        out = bytearray()
        while len(out) < len(p):
            h = hashlib.sha256()
            h.update(self.state)
            h.update(self.count.to_bytes(8, "big"))
            self.count += 1
            out.extend(h.digest())
        p[:] = out[: len(p)]


def new_challenge_randomness(commit_id: bytes):
    """Production randomness requires an immutable ResultCommit binding."""
    if not commit_id:
        raise FreivaldsError(
            "challenge randomness requires an immutable ResultCommit; "
            "refusing to generate before commit lock"
        )
    return CSPRNGSource()


def random_binary_vector(src, n: int) -> list:
    raw = bytearray((n + 7) // 8)
    src.fill(raw)
    return [(raw[j // 8] >> (j % 8)) & 1 for j in range(n)]


def _check_int64(v: int, what: str) -> int:
    if not (_INT64_MIN <= v <= _INT64_MAX):
        raise FreivaldsError(f"{what} exceeds int64 range; admission bound was violated")
    return v


def freivalds_round(a, b, c, m: int, n: int, k: int, r):
    """x = B x r, y = A x x, z = C x r with exact int64 semantics."""
    x = []
    for t in range(k):
        acc = 0
        for j in range(n):
            if r[j]:
                acc += b[t * n + j]
        x.append(_check_int64(acc, "x"))
    y = []
    for i in range(m):
        acc = 0
        for t in range(k):
            acc += a[i * k + t] * x[t]
        y.append(_check_int64(acc, "y"))
    z = []
    for i in range(m):
        acc = 0
        for j in range(n):
            if r[j]:
                acc += c[i * n + j]
        z.append(_check_int64(acc, "z"))
    return x, y, z


def residual_bad_rows(y, z):
    return [i for i in range(len(y)) if y[i] != z[i]]


def verify_freivalds(a, b, c, m: int, n: int, k: int, profile: VerificationProfile, src):
    """Scalar reference implementation; returns a result dict."""
    profile.admit_for(m, n, k)
    if len(a) != m * k or len(b) != k * n or len(c) != m * n:
        raise FreivaldsError("matrix shape mismatch")
    import time

    started = time.time()
    macs_per_round = n * k + m * k + m * n
    result = {
        "passed": True,
        "rounds_executed": 0,
        "residual_rows": [],
        "mismatch_round": -1,
        "verification_macs": 0,
    }
    for rnd in range(profile.rounds):
        r = random_binary_vector(src, n)
        _x, y, z = freivalds_round(a, b, c, m, n, k, r)
        result["rounds_executed"] = rnd + 1
        result["verification_macs"] += macs_per_round
        if y != z:
            result["passed"] = False
            result["mismatch_round"] = rnd
            result["residual_rows"] = residual_bad_rows(y, z)
            break
    result["duration_ms"] = round((time.time() - started) * 1000, 3)
    return result


def verify_freivalds_batch(a, b, c, m: int, n: int, k: int, profile: VerificationProfile, src):
    """Batched form: R in {0,1}^{N x q}. Must agree with the scalar path."""
    profile.admit_for(m, n, k)
    if len(a) != m * k or len(b) != k * n or len(c) != m * n:
        raise FreivaldsError("matrix shape mismatch")
    macs_per_round = n * k + m * k + m * n
    result = {
        "passed": True,
        "rounds_executed": profile.rounds,
        "residual_rows": [],
        "mismatch_round": -1,
        "verification_macs": macs_per_round * profile.rounds,
    }
    # One INDEPENDENT binary vector per round; reusing a single vector
    # would collapse q rounds to one round of detection power.
    R = [[0] * profile.rounds for _ in range(n)]
    for s in range(profile.rounds):
        vec = random_binary_vector(src, n)
        for j in range(n):
            R[j][s] = vec[j]
    X = [[0] * profile.rounds for _ in range(k)]
    for t in range(k):
        for s in range(profile.rounds):
            X[t][s] = sum(b[t * n + j] * R[j][s] for j in range(n))
    Y = [[0] * profile.rounds for _ in range(m)]
    for i in range(m):
        for s in range(profile.rounds):
            Y[i][s] = sum(a[i * k + t] * X[t][s] for t in range(k))
    Z = [[0] * profile.rounds for _ in range(m)]
    for i in range(m):
        for s in range(profile.rounds):
            Z[i][s] = sum(c[i * n + j] * R[j][s] for j in range(n))
    for s in range(profile.rounds):
        y_col = [Y[i][s] for i in range(m)]
        z_col = [Z[i][s] for i in range(m)]
        if y_col != z_col:
            result["passed"] = False
            result["mismatch_round"] = s
            result["residual_rows"] = residual_bad_rows(y_col, z_col)
            break
    return result


def c_from_canonical(hex_str: str):
    return canonical_to_int32s(bytes.fromhex(hex_str))


def c_to_canonical(c) -> str:
    return int32s_to_canonical(c).hex()
