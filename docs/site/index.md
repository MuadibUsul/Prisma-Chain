# Prisma Chain — Testnet Documentation

> **Testnet only.** Test tokens have no monetary value. There is no mainnet
> and no monetary claim anywhere in this documentation.

Prisma Chain is a verification-first compute network: a developer submits a
job, a GPU worker executes it, the result is committed and
availability-checked, and independent watchers verify it. Payment settles on
chain only when a Verified Work Receipt (VWR) exists — fraud is settled
through deterministic disputes, not trust.

## Quickstarts

| role | start here |
|---|---|
| Developer (submit a job) | [Developer Quickstart](developer-quickstart.md) |
| GPU operator (earn test tokens) | [Worker Quickstart](worker-quickstart.md) |
| Verifier (run a watcher) | [Watcher Quickstart](watcher-quickstart.md) |
| Storage operator (DA provider) | [DA Quickstart](da-quickstart.md) |
| Node operator (validator) | [Validator guide](run-validator.md) |

## Reference

- [API reference](api.md) — Job API v1 (B6)
- [CLI reference](cli.md) — the `prisma` command (B5)
- [Job lifecycle](job-lifecycle.md) — every state, and who may transition it
- [Verified Work Receipt](verified-work-receipt.md) — what a receipt proves
- [Troubleshooting](troubleshooting.md)
- [Known limitations](testnet-known-limitations.md) — read before relying on anything
- [Security model](security-model.md)

## The frozen protocol (plain language)

1. **Canonical encoding** — every input, weight and output is encoded
   deterministically (canonical CBOR, big-endian integers). Same bytes
   everywhere: worker, watcher, chain. Spec: `CANONICAL_TENSOR_V2`,
   `CANONICAL_GRAPH_V2` (GraphIDV2 `8fb86087…`).
2. **Exact integer execution** — the model runs in exact int64 arithmetic
   (A13W10 profile, PolicyID `eb9a9fef…`). No floats anywhere in canonical
   execution; the GPU backend (SM86/SM89) is bit-exact against the CPU
   reference or it does not ship. Spec: `GEMM_A13W10_I64_V1`,
   `REQUANTIZE_WIDE_V1`.
3. **Cheap verification first** — a watcher runs a 40-round Freivalds check
   (detection only) plus exact recomputation of cheap nodes before opening a
   challenge. Spec: `FREIVALDS_A13W10_I64_V1`.
4. **Deterministic disputes** — a challenge bisects the computation to a
   single node, then a single 8x8 tile, then 512 logical MACs that the chain
   re-computes exactly. No probabilistic slashing. Spec: `GraphDisputeV2`,
   `WIDE_GEMM_DISPUTE_V1`.
5. **Availability before payment** — the result bundle must be stored by a
   2-of-3 quorum of bonded DA providers, with typed tile proofs on challenge.
   Spec: `GRAPH_VERIFICATION_BUNDLE_V2`, `GRAPH_DA_ATTESTATION_V2`.

The protocol is **frozen** at tag `transformer-phase-f-wide-integer`
(commit `89547fa`); product software consumes it and never changes it.
