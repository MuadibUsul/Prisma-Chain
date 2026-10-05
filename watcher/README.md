# prisma-watcher

Prisma Chain permissionless watcher daemon (Phase B productization, B3-01).
It watches the chain for completed graph results, retrieves the verification
bundle from DA providers, verifies it with the **frozen WatcherV2** (root
phase + Freivalds detection + cheap exact recomputations) and journals every
task so a restart resumes instead of re-verifying.

```text
prisma-watcher init    # challenger key + config (worker identity rules)
prisma-watcher run     # discover -> retrieve -> verify -> journal
prisma-watcher status  # journal summary
prisma-watcher verify  # one-off verification of a downloaded bundle
```

Detection is **detection-only**: a fraud verdict is recorded (and, in B3-02,
drives the challenge transaction); the chain adjudicates and slashing is
never triggered probabilistically. The frozen libraries
(`compute/canonical/python`, `compute/gemmv1/python`, `tools/f5c_watcher_v2.py`)
are **loaded**, not re-implemented; point `PRISMA_FROZEN_PYTHON` at the
repository root when running outside a checkout.
