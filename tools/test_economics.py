import unittest

from economics import Month, Parameters, annual_emission, fraud_break_even_probability, simulate


class EconomicsTest(unittest.TestCase):
    def test_supply_and_reserve_are_separate(self):
        params = Parameters()
        month = Month(external_fees=10_000, tasks=1, audit_fraction_bps=10_000,
                      replay_cost=500, reserve_release=2_000)
        row = simulate(params, [month])[0]
        self.assertEqual(row.burned, 2_000)
        self.assertEqual(row.worker_paid, 7_000)
        self.assertEqual(row.watcher_paid, 1_000)
        self.assertEqual(row.supply, params.genesis_supply + row.newly_minted - 2_000)
        self.assertEqual(row.reserve_left, params.locked_reward_reserve - 2_000)
        self.assertEqual(row.watcher_shortfall, 0)

    def test_audit_shortfall_is_visible(self):
        row = simulate(Parameters(), [Month(1_000, 10, 10_000, 100)])[0]
        self.assertEqual(row.watcher_shortfall, 900)

    def test_underfunded_bond_needs_more_detection(self):
        self.assertAlmostEqual(fraud_break_even_probability(100, 100), 0.5)
        self.assertAlmostEqual(fraud_break_even_probability(100, 900), 0.1)

    def test_emission_halves_but_does_not_fall_below_tail(self):
        params = Parameters()
        self.assertEqual(annual_emission(params, 24), annual_emission(params, 0) // 2)
        self.assertEqual(annual_emission(params, 240),
                         params.genesis_supply * params.tail_annual_emission_bps // 10_000)


if __name__ == "__main__":
    unittest.main()
