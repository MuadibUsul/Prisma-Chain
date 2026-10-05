"""Generate compute/gemmv1/testdata/freivalds_vectors.json.

Run from the repository root:

    python compute/gemmv1/python/gemmv1/gen_freivalds_vectors.py

The file pins, for fixed A/B/C (honest and fraud) and a deterministic
development-only R stream, the exact X/Y/Z of every round, the residual
rows, the pass/fail outcome and the bad-row/bad-tile localization. The Go
and Python test suites must both reproduce them bit-for-bit (protocol
test H of v0.1.2). Production randomness stays CSPRNG; fixed R exists only
for these vectors.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemmv1.freivalds import (  # noqa: E402
    DeterministicSource,
    VerificationProfile,
    freivalds_round,
    random_binary_vector,
    residual_bad_rows,
)
from gemmv1.row_locator import localize_from_rows  # noqa: E402
from gemmv1.testgen import gen_test_matrix  # noqa: E402

M, N, K, SEED = 16, 16, 24, 5
ROUNDS = 4
# The development-only R stream seed is searched so the vectors contain a
# stream that actually detects the injected fraud; production randomness is
# CSPRNG and needs no seed.
R_SEED = None


def hex_i8(vals) -> str:
    return bytes(v & 0xFF for v in vals).hex()


def hex_i64(vals) -> str:
    out = []
    for v in vals:
        out.append((v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "big").hex())
    return "".join(out)


def detect_seed(a, b, fraud):
    for seed in range(1, 9):
        src = DeterministicSource(bytes([seed]))
        for _ in range(ROUNDS):
            r = random_binary_vector(src, N)
            _x, y, z = freivalds_round(a, b, fraud, M, N, K, r)
            if y != z:
                return seed
    raise SystemExit("no seed detected the fraud")


def build_case() -> dict:
    a = gen_test_matrix(ord("A"), SEED, M * K)
    b = gen_test_matrix(ord("B"), SEED, K * N)
    # Honest C via the independent python reference GEMM.
    c = [0] * (M * N)
    for i in range(M):
        for t in range(K):
            av = a[i * K + t]
            if av == 0:
                continue
            for j in range(N):
                c[i * N + j] += av * b[t * N + j]
    fraud = list(c)
    fraud[5 * N + 9] += 1

    global R_SEED
    R_SEED = detect_seed(a, b, fraud)
    src = DeterministicSource(bytes([R_SEED]))
    rounds = []
    detected_at = None
    residual_rows = []
    loc = None
    for rnd in range(ROUNDS):
        r = random_binary_vector(src, N)
        x, y, z = freivalds_round(a, b, fraud, M, N, K, r)
        rows = residual_bad_rows(y, z)
        rounds.append({
            "round": rnd,
            "r": bytes(r).hex(),
            "x": hex_i64(x),
            "y": hex_i64(y),
            "z": hex_i64(z),
            "residual_rows": rows,
        })
        if rows and detected_at is None:
            detected_at = rnd
            residual_rows = rows
    if residual_rows:
        localization, found = localize_from_rows(a, b, fraud, M, N, K, residual_rows)
        if found:
            loc = {
                "row": localization["row"],
                "bad_cols": localization["columns"][:4],
                "tile_i": localization["tile_i"],
                "tile_j": localization["tile_j"],
            }
    return {
        "name": "fraud_single_element",
        "m": M, "n": N, "k": K, "seed": SEED, "r_seed": R_SEED, "rounds": ROUNDS,
        "matrix_a": hex_i8(a),
        "matrix_b": hex_i8(b),
        "c_honest": bytes(__import__("struct").pack(">" + "i" * len(c), *c)).hex(),
        "c_fraud": bytes(__import__("struct").pack(">" + "i" * len(fraud), *fraud)).hex(),
        "rounds_detail": rounds,
        "fraud_detected_at_round": detected_at,
        "residual_rows": residual_rows,
        "localization": loc,
    }


def main() -> None:
    vectors = {"protocol_version": "0.1.2", "algorithm": "FREIVALDS_BINARY_V1",
               "cases": [build_case()]}
    out = Path(__file__).resolve().parents[2] / "testdata" / "freivalds_vectors.json"
    out.write_text(json.dumps(vectors, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
