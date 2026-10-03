"""v0.1.2 fast-path E2E: fast-honest and fast-fraud.

The challenger runs Freivalds detection plus exact one-row localization
and then falls into the existing v0.1.1 dispute. Instrumentation asserts
invariant 5: the fast path never calls the full reference GEMM.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemmv1 import e2e  # noqa: E402
from gemmv1.freivalds import DeterministicSource  # noqa: E402
from gemmv1.merkle_proofs import verify_inclusion  # noqa: E402
from gemmv1.protocol import leaf_trace_state  # noqa: E402
from gemmv1.service import handle_execute, handle_mid, handle_trace  # noqa: E402
from gemmv1.tensors import canonical_to_int32s, int32s_to_canonical  # noqa: E402

M, N, K, SEED = 96, 88, 192, 42


class TestFastE2E(unittest.TestCase):
    def _execute(self, fraud=None):
        body = {"m": M, "n": N, "k": K, "seed": SEED}
        if fraud:
            body["inject_fraud"] = list(fraud)
        return handle_execute(body)

    def _prepare(self, w, mode, rounds=8, r_seed=None):
        body = {
            "m": M, "n": N, "k": K, "seed": SEED,
            "task_id": w["result_commit"]["task_id"],
            "assignment_id": w["result_commit"]["assignment_id"],
            "output_root": w["result_commit"]["output_root"],
            "worker_tiles": w["output_tiles"]["tiles"],
            "mode": mode,
            "rounds": rounds,
        }
        if r_seed is not None:
            body["r_seed"] = r_seed
        return e2e.dispatch("prepare", body)

    def _dispute_to_challenger_win(self, w, c, tile_i, tile_j):
        """Drive the existing v0.1.1 dispute from the localized tile."""
        task = bytes.fromhex(w["result_commit"]["task_id"])
        assignment = bytes.fromhex(w["result_commit"]["assignment_id"])
        r_steps = (K + 7) // 8
        wt = handle_trace({"tile_i": tile_i, "tile_j": tile_j})
        worker_root = bytes.fromhex(wt["trace_root"])
        worker_final = bytes.fromhex(wt["states"][-1])
        self.assertEqual(worker_final.hex(), self._worker_tile_hex(w, tile_i, tile_j))
        ct = e2e.dispatch("trace", {"tile_i": tile_i, "tile_j": tile_j})
        challenger_root = bytes.fromhex(ct["trace_root"])
        challenger_final = bytes.fromhex(ct["states"][-1])
        self.assertEqual(challenger_final, bytes.fromhex(c["disputed_challenger_tile"]))

        low, high = 0, r_steps
        low_state = [0] * 64
        worker_high = canonical_to_int32s(worker_final)
        challenger_high = canonical_to_int32s(challenger_final)
        rounds = 0
        while high - low > 1:
            mid = low + (high - low) // 2
            wm = handle_mid({"step": mid, "tile_i": tile_i, "tile_j": tile_j})
            w_state = bytes.fromhex(wm["state"])
            p = wm["proof"]
            self.assertTrue(verify_inclusion(
                worker_root,
                leaf_trace_state(task, assignment, tile_i, tile_j, mid, w_state),
                p["index"], p["count"], [bytes.fromhex(s) for s in p["siblings"]]))
            cm = e2e.dispatch("mid", {"step": mid})
            c_state = bytes.fromhex(cm["state"])
            p = cm["proof"]
            self.assertTrue(verify_inclusion(
                challenger_root,
                leaf_trace_state(task, assignment, tile_i, tile_j, mid, c_state),
                p["index"], p["count"], [bytes.fromhex(s) for s in p["siblings"]]))
            w_vals, c_vals = canonical_to_int32s(w_state), canonical_to_int32s(c_state)
            if w_vals == c_vals:
                low, low_state = mid, w_vals
            else:
                high = mid
                worker_high, challenger_high = w_vals, c_vals
            rounds += 1
            self.assertLess(rounds, 64)
        arb = e2e.dispatch("arbitrate", {
            "step": low, "high": high, "tile_i": tile_i, "tile_j": tile_j,
            "low_state": int32s_to_canonical(low_state).hex(),
            "worker_high": int32s_to_canonical(worker_high).hex(),
        })
        return arb, rounds

    @staticmethod
    def _worker_tile_hex(w, tile_i, tile_j):
        cols_c = w["output_tiles"]["cols_c"]
        return int32s_to_canonical(w["output_tiles"]["tiles"][tile_i * cols_c + tile_j]).hex()

    def test_fast_honest_no_challenge(self):
        w = self._execute()
        c = self._prepare(w, mode="fast", rounds=8, r_seed=7)
        self.assertTrue(c["root_ok"])
        self.assertIsNone(c["disputed_tile"])
        self.assertTrue(c["verification"]["passed"])
        self.assertEqual(e2e.dispatch("stats", {})["reference_gemm_calls"], 0,
                         "invariant 5 violated: fast path called the full GEMM")

    def test_fast_fraud_localizes_and_wins_dispute(self):
        w = self._execute(fraud=(1, 1))
        c = self._prepare(w, mode="fast", rounds=8, r_seed=1)
        self.assertTrue(c["root_ok"])
        self.assertFalse(c["verification"]["passed"])
        self.assertEqual(c["disputed_tile"], [1, 1])
        self.assertEqual(e2e.dispatch("stats", {})["reference_gemm_calls"], 0,
                         "invariant 5 violated: fast path called the full GEMM")
        arb, rounds = self._dispute_to_challenger_win(w, c, 1, 1)
        self.assertEqual(arb["outcome"], "challenger_wins")
        self.assertEqual(arb["arbitration_macs"], 512)
        self.assertLessEqual(rounds, 6)

    def test_full_mode_still_uses_reference(self):
        """The v0.1.1 full-verification path must keep working unchanged."""
        w = self._execute()
        c = self._prepare(w, mode="full")
        self.assertIsNone(c["disputed_tile"])
        self.assertGreaterEqual(e2e.dispatch("stats", {})["reference_gemm_calls"], 1)


if __name__ == "__main__":
    unittest.main()
