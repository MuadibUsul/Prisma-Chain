# Testnet PRSM accounting experiment

`python tools/economics.py tools/scenarios.json` prints month-by-month supply,
burn, worker payment, watcher payment and the shortfall against specified
replay costs. `python -m unittest discover -s tools` checks the supply and
reserve invariants.

The default scenario uses the whitepaper's illustrative 1 billion genesis
PRSM, 40% locked reward reserve, 5% initial annual emission (halving every
two years), a 0.5% tail, and 20% burn on successful external fees. Remaining
external fees are modeled as 70% worker compensation and 10% watcher budget.
These are **experiment inputs**, not validated sustainable parameters.

The locked genesis reserve is transferred when released, so it does not change
total supply. New issuance does. A test or chain implementation that counts a
reserve release as newly minted has counted the same tokens twice.

The first sample month deliberately has no external orders. Its watcher
budget is zero even though auditing public jobs has a cost. A real launch
requires an explicit monitoring subsidy or a fee schedule that covers this
cost. `fraud_break_even_probability` gives a narrow, non-collusion model for
the audit probability needed to deter skipped computation; it is not a
security proof.

No price, liquidity, exchange demand, legal classification or real-world
profitability is inferred from this simulator.
