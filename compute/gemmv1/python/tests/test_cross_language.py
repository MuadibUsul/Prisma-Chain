"""Cross-language vector tests: the Python implementation must reproduce
compute/gemmv1/testdata/cross_language_vectors.json bit-for-bit. The Go
side runs the same check in crosslang_test.go."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemmv1 import (  # noqa: E402
    ArithmeticSpec,
    Operator,
    ProtocolVersion,
    assignment_id,
    build_matrix_roots,
    encode_canonical,
    gen_test_matrix,
    leaf_output_tile,
    merkle_root,
    output_tiles,
    receipt_id,
    reference_gemm,
    task_id,
)
from gemmv1.tensors import int32s_to_canonical  # noqa: E402

VECTORS = (
    Path(__file__).resolve().parents[2] / "testdata" / "cross_language_vectors.json"
)


def as_bytes(values) -> bytes:
    return bytes(v & 0xFF for v in values)


class TestCrossLanguageVectors(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(VECTORS.read_text(encoding="utf-8"))

    def rebuild(self, case):
        m, n, k, seed = case["m"], case["n"], case["k"], case["seed"]
        matrix_a = gen_test_matrix(ord("A"), seed, m * k)
        matrix_b = gen_test_matrix(ord("B"), seed, k * n)
        root_a, root_b, _, _, _, _ = build_matrix_roots(matrix_a, matrix_b, m, n, k)
        descriptor = {
            "protocol_version": ProtocolVersion,
            "operator": Operator,
            "requester_pubkey": as_bytes(gen_test_matrix(ord("Q"), seed, 32)),
            "requester_nonce": as_bytes(gen_test_matrix(ord("N"), seed, 16)),
            "issued_epoch": 1000,
            "m": m,
            "n": n,
            "k": k,
            "matrix_a_root": root_a,
            "matrix_b_root": root_b,
            "arithmetic_spec": ArithmeticSpec,
            "tile_size": 8,
            "challenge_window": 100,
            "max_price_per_cwu": 1000,
            "settlement_asset": "uprsm",
        }
        tid = task_id(descriptor)
        worker_pub = as_bytes(gen_test_matrix(ord("W"), seed, 32))
        assignment = {
            "task_id": tid,
            "worker_pubkey": worker_pub,
            "assignment_nonce": as_bytes(gen_test_matrix(ord("M"), seed, 16)),
            "accepted_epoch": 1001,
        }
        aid = assignment_id(assignment)
        c = reference_gemm(matrix_a, matrix_b, m, n, k)
        cols_c = (n + 7) // 8
        tiles = output_tiles(c, m, n)
        leaves = [
            leaf_output_tile(tid, aid, idx // cols_c, idx % cols_c, int32s_to_canonical(tile))
            for idx, tile in enumerate(tiles)
        ]
        output_root = merkle_root(leaves)
        receipt = {
            "protocol_version": ProtocolVersion,
            "task_id": tid,
            "assignment_id": aid,
            "worker_pubkey": worker_pub,
            "operator": Operator,
            "canonical_mac_count": m * n * k,
            "output_root": output_root,
            "verification_mode": "optimistic_unchallenged",
            "finalized_epoch": 2000,
            "settlement_reference": b"",
            "dispute_transcript_digest": b"",
        }
        return {
            "matrix_a_root": root_a.hex(),
            "matrix_b_root": root_b.hex(),
            "task_id": tid.hex(),
            "assignment_id": aid.hex(),
            "output_root": output_root.hex(),
            "receipt_id": receipt_id(receipt).hex(),
            "canonical_task_cbor": encode_canonical(descriptor).hex(),
            "canonical_receipt_cbor": encode_canonical(receipt).hex(),
        }

    def test_all_cases_bit_exact(self):
        for case in self.data["cases"]:
            with self.subTest(case=case["name"]):
                got = self.rebuild(case)
                for field, want in got.items():
                    self.assertEqual(
                        want,
                        case[field],
                        f"{case['name']}.{field}: got {got[field]}, want {case[field]}",
                    )


if __name__ == "__main__":
    unittest.main()
