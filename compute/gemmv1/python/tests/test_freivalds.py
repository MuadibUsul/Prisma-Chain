"""v0.1.2 Freivalds tests for the Python mirror: cross-language vectors
(test H), honest pass, fraud detection + localization, batch/scalar
agreement, false challenge rejection, and availability routes."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemmv1.availability import (  # noqa: E402
    DATA_UNAVAILABLE,
    OUTPUT_DATA_COMMITMENT_MISMATCH,
    AvailabilityError,
    can_finalize_optimistic,
    new_output_availability_descriptor,
)
from gemmv1.freivalds import (  # noqa: E402
    DeterministicSource,
    FreivaldsError,
    VerificationProfile,
    new_challenge_randomness,
    random_binary_vector,
    residual_bad_rows,
    verify_freivalds,
    verify_freivalds_batch,
)
from gemmv1.protocol import build_matrix_roots, task_id  # noqa: E402
from gemmv1.row_locator import localize_from_rows, reference_row_gemm  # noqa: E402
from gemmv1.testgen import gen_test_matrix  # noqa: E402

VECTORS = Path(__file__).resolve().parents[2] / "testdata" / "freivalds_vectors.json"


def reference_gemm(a, b, m, n, k):
    from gemmv1.tensors import reference_gemm as rg

    return rg(a, b, m, n, k)


class TestCrossLanguageVectors(unittest.TestCase):
    def test_freivalds_vectors_bit_exact(self):
        data = json.loads(VECTORS.read_text(encoding="utf-8"))
        for case in data["cases"]:
            m, n, k = case["m"], case["n"], case["k"]
            a = gen_test_matrix(ord("A"), case["seed"], m * k)
            b = gen_test_matrix(ord("B"), case["seed"], k * n)
            self.assertEqual(bytes(v & 0xFF for v in a).hex(), case["matrix_a"])
            c_honest = reference_gemm(a, b, m, n, k)
            from gemmv1.tensors import int32s_to_canonical

            self.assertEqual(int32s_to_canonical(c_honest).hex(), case["c_honest"])
            fraud = list(c_honest)
            fraud[5 * n + 9] += 1
            self.assertEqual(int32s_to_canonical(fraud).hex(), case["c_fraud"])

            src = DeterministicSource(bytes([case["r_seed"]]))
            detected = None
            for rd in case["rounds_detail"]:
                r = random_binary_vector(src, n)
                self.assertEqual(bytes(r).hex(), rd["r"])
                _x, y, z = __import__("gemmv1.freivalds", fromlist=["freivalds_round"]).freivalds_round(a, b, fraud, m, n, k, r)
                from gemmv1.freivalds import freivalds_round as fr

                _x, y, z = fr(a, b, fraud, m, n, k, r)
                rows = residual_bad_rows(y, z)
                self.assertEqual(rows, rd["residual_rows"])
                if rows and detected is None:
                    detected = rd["round"]
            self.assertEqual(detected, case["fraud_detected_at_round"])
            loc, found = localize_from_rows(a, b, fraud, m, n, k, case["residual_rows"])
            self.assertTrue(found)
            self.assertEqual(loc["row"], case["localization"]["row"])
            self.assertEqual((loc["tile_i"], loc["tile_j"]),
                             (case["localization"]["tile_i"], case["localization"]["tile_j"]))


class TestFreivaldsBehavior(unittest.TestCase):
    def setUp(self):
        self.m = self.n = self.k = 16
        self.a = gen_test_matrix(ord("A"), 8, self.m * self.k)
        self.b = gen_test_matrix(ord("B"), 9, self.k * self.n)
        self.c = reference_gemm(self.a, self.b, self.m, self.n, self.k)

    def test_a_honest_passes(self):
        res = verify_freivalds(self.a, self.b, self.c, self.m, self.n, self.k,
                               VerificationProfile(16), new_challenge_randomness(b"commit"))
        self.assertTrue(res["passed"])
        res = verify_freivalds_batch(self.a, self.b, self.c, self.m, self.n, self.k,
                                     VerificationProfile(16), new_challenge_randomness(b"commit"))
        self.assertTrue(res["passed"])

    def test_b_fraud_detected_and_localized(self):
        fraud = list(self.c)
        fraud[5 * self.n + 9] += 1
        res = verify_freivalds(self.a, self.b, fraud, self.m, self.n, self.k,
                               VerificationProfile(16), DeterministicSource(b"\x01"))
        self.assertFalse(res["passed"])
        loc, found = localize_from_rows(self.a, self.b, fraud, self.m, self.n, self.k,
                                        res["residual_rows"])
        self.assertTrue(found)
        self.assertEqual((loc["tile_i"], loc["tile_j"]), (0, 1))

    def test_d_false_challenge_rejected(self):
        loc, found = localize_from_rows(self.a, self.b, self.c, self.m, self.n, self.k, [2])
        self.assertFalse(found)

    def test_g_randomness_requires_commit(self):
        with self.assertRaises(FreivaldsError):
            new_challenge_randomness(b"")
        new_challenge_randomness(b"\x01" * 32)

    def test_batch_matches_scalar(self):
        fraud = list(self.c)
        fraud[9 * self.n + 2] += 3
        rs = verify_freivalds(self.a, self.b, fraud, self.m, self.n, self.k,
                              VerificationProfile(8), DeterministicSource(b"\x2a"))
        rb = verify_freivalds_batch(self.a, self.b, fraud, self.m, self.n, self.k,
                                    VerificationProfile(8), DeterministicSource(b"\x2a"))
        self.assertEqual(rs["passed"], rb["passed"])
        self.assertEqual(rs["residual_rows"], rb["residual_rows"])

    def test_profile_rejects_unsafe_shape(self):
        with self.assertRaises(FreivaldsError):
            VerificationProfile(4).admit_for(0, 4, 4)
        with self.assertRaises(FreivaldsError):
            VerificationProfile(4).admit_for(1, 2**40, 2**40)


class TestAvailability(unittest.TestCase):
    def setUp(self):
        self.m = self.n = self.k = 16
        self.a = gen_test_matrix(ord("A"), 3, self.m * self.k)
        self.b = gen_test_matrix(ord("B"), 4, self.k * self.n)
        self.c = reference_gemm(self.a, self.b, self.m, self.n, self.k)
        root_a, root_b, _, _, _, _ = build_matrix_roots(self.a, self.b, self.m, self.n, self.k)
        descriptor = {
            "protocol_version": "0.1.1", "operator": "GEMM_INT8_V1",
            "requester_pubkey": b"\x01" * 32, "requester_nonce": b"n",
            "issued_epoch": 1000, "m": self.m, "n": self.n, "k": self.k,
            "matrix_a_root": root_a, "matrix_b_root": root_b,
            "arithmetic_spec": "INT8_INT32_V1", "tile_size": 8,
            "challenge_window": 100, "max_price_per_cwu": 1000,
            "settlement_asset": "uprsm",
        }
        self.descriptor = descriptor
        self.task_id = task_id(descriptor)
        self.assignment = b"\x02" * 32
        from gemmv1.protocol import leaf_output_tile, merkle_root
        from gemmv1.tensors import int32s_to_canonical, output_tiles

        cols_c = (self.n + 7) // 8
        tiles = output_tiles(self.c, self.m, self.n)
        self.output_root = merkle_root([
            leaf_output_tile(self.task_id, self.assignment, i // cols_c, i % cols_c, int32s_to_canonical(t))
            for i, t in enumerate(tiles)
        ])
        self.desc = new_output_availability_descriptor(
            self.descriptor, self.assignment, self.output_root, "dev://c")

    def test_e_data_unavailable(self):
        with self.assertRaises(AvailabilityError) as ctx:
            can_finalize_optimistic(self.descriptor, self.assignment, self.desc,
                                    lambda d: (_ for _ in ()).throw(RuntimeError("404")))
        self.assertEqual(str(ctx.exception), DATA_UNAVAILABLE)

    def test_f_commitment_mismatch(self):
        corrupted = list(self.c)
        corrupted[0] += 1
        with self.assertRaises(AvailabilityError) as ctx:
            can_finalize_optimistic(self.descriptor, self.assignment, self.desc, lambda d: corrupted)
        self.assertEqual(str(ctx.exception), OUTPUT_DATA_COMMITMENT_MISMATCH)

    def test_honest_c_finalizes(self):
        self.assertTrue(can_finalize_optimistic(
            self.descriptor, self.assignment, self.desc, lambda d: self.c))


if __name__ == "__main__":
    unittest.main()
