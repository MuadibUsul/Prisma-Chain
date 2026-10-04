"""Self-checks of the Python canonical mirror.

Run:  python compute/canonical/python/test_canonical_ref.py

Covers the pinned constants (regenerated with exact integer square
roots), accuracy sanity of the frozen algorithms, and a regeneration
check that the committed vectors still come out of this code unchanged.
"""

import json
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import canonical_ref as ref
import gen_canonical_vectors as gen


class TestPinnedConstants(unittest.TestCase):
    def test_inv_sqrt_table_is_the_frozen_generator_output(self):
        self.assertEqual(ref.build_inv_sqrt_table(), ref.INV_SQRT_TABLE)

    def test_inv_sqrt_accuracy(self):
        # x is a Q12.20 value: want = 2^20 / sqrt(x / 2^20).
        # The canonical result is never below 1 (minimum positive), so the
        # tolerance is one unit plus 2e-3 relative.
        for x in [1, 2, 3, 4, 7, 16, 1023, 1 << 20, 1 << 40, (1 << 62) - 1]:
            got = ref.inv_sqrt_fx(x)
            want = (1 << 20) / math.sqrt(x / (1 << 20))
            self.assertLessEqual(abs(got - want), 1 + 2e-3 * want,
                                 f"invsqrt({x}) = {got}, want {want}")

    def test_exp_fx_known_values(self):
        self.assertEqual(ref.exp_fx(0), ref.ONE)
        got = ref.exp_fx(ref.LN2_FX)
        self.assertLess(abs(got - 2 * ref.ONE), ref.ONE >> 9)
        self.assertEqual(ref.exp_fx(-(ref.ONE * 24)), 0)
        self.assertEqual(ref.exp_fx(ref.ONE * 21), ref.MAX_FX)
        prev = -1
        for x in range(-(ref.ONE * 4), ref.ONE * 4, ref.ONE // 3):
            v = ref.exp_fx(x)
            self.assertGreaterEqual(v, prev)
            prev = v

    def test_sigmoid_bounds(self):
        self.assertEqual(ref.sigmoid_fx(0), ref.ONE // 2)
        # Saturating tails may underflow/overflow to the exact bounds;
        # symmetry holds within a few ulp of Q12.20.
        for x in [1, ref.ONE, 5 << 20, -(5 << 20), 20 << 20, -(20 << 20)]:
            v = ref.sigmoid_fx(x)
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, ref.ONE)
            if abs(x) <= 8 << 20:
                w = ref.sigmoid_fx(-x)
                self.assertLessEqual(abs(v + w - ref.ONE), 4,
                                     f"sigmoid symmetry at {x}: {v} + {w}")


class TestVectorsRegenerate(unittest.TestCase):
    def test_committed_vectors_match_this_code(self):
        path = os.path.normpath(os.path.join(HERE, "..", "testdata", "canonical_vectors.json"))
        with open(path, encoding="utf-8") as handle:
            committed = json.load(handle)
        self.assertEqual(committed["math"], gen.hexify(gen.math_vectors()))
        self.assertEqual(committed["tensor_roots"], gen.hexify(gen.tensor_root_vectors()))
        self.assertEqual(committed["operators"], gen.hexify(gen.operator_vectors()))
        self.assertEqual(committed["graph"], gen.hexify(gen.graph_vector()))

    def test_graph_id_binds_params(self):
        graph = gen.graph_vector()
        descriptor = graph["descriptor"]
        gid = ref.graph_id(descriptor)
        nodes = [dict(n) for n in descriptor["nodes"]]
        nodes[6] = dict(nodes[6])
        nodes[6]["params"] = [["clamp_hi", ref.MAX_FX]]
        tampered = dict(descriptor)
        tampered["nodes"] = nodes
        self.assertNotEqual(gid, ref.graph_id(tampered))


if __name__ == "__main__":
    unittest.main(verbosity=2)
