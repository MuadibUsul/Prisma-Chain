# Phase F.5C report — Formal Wide-Integer Protocol Integration

**Status: PHASE_F = PASS (F.5C gate closed).**  Every protocol component
of CANONICAL_GRAPH_V2 is implemented, cross-language verified and
committed, and the complete roadmap A6 4-validator devnet E2E suite has
been executed — both on the deterministic fixture graphs and on the REAL
310-node Qwen3 block — with per-scenario evidence JSONs in `docs/`.
Nothing in this report is NOT TESTED.

Branch `protocol/transformer-phase-f5c-wide-integer`.  Zero V1
reinterpretation; the full legacy golden suite pins V1 byte-for-byte.

## Protocol (all additive + versioned)

| component | evidence |
|---|---|
| CANONICAL_TENSOR_V2 (A13/W10/INT64_ACCUM, BE16/BE64, range rejection, typed chunk proofs) | Go/Python identical; encoding edges (-4096/-1/0/1/4095, -512/511, MinInt64/MaxInt64, no negative zero); 4096/512/2^31 rejected; V1 material never verifies as V2 |
| GEMM_A13W10_I64_V1 | frozen F.5B wide vectors replay element-exact through the Go reference; `MaxSafeK64(13,10)=4398046511103`; right operand W10 or A13 |
| REQUANTIZE_WIDE_V1 | ties/saturation/overflow tests; 55-bit theoretical int64 bound; GPU path exact (F.5B.2) |
| CANONICAL_GRAPH_V2 | `GraphIDV2 = 8fb86087697e277602bb4b2088ba48eb4172c21d9c4f83914582f2cbf00d4def`; PolicyID `eb9a9fef…`; Python V2 executor **310/310 node values == F.5B.2 manifest**; Go replay **310/310 TensorRootV2 == Python**; work vector identical incl. `GEMM_A13W10_MAC = 252,706,816` |
| GraphStateRootV2 / TrailRootV2 | Python-generated vectors reproduced bit-exact by Go; wrong id/step/V1 material rejected |
| NodeOutputManifestV2 / GraphResultCommitV3 / VWR V3 | small-graph execution tests; tampering invalidates; wrong-domain signatures rejected; V1/V3 receipts live in separate domains |
| GraphDisputeV2 (trail claim, midpoint, snapshot incl. in-flight medians, restart) | injected fraud at node 0/1 bisects to the exact first-divergent node; mid-round snapshot -> restore -> same verdict; **devnet: 8–9 midpoint rounds on the real 310-node block converge to the injected node (145 GEMM / 85 ROPE)**; single-node graphs promote to arb_ready at lock time (fix + chain test `TestGraphV2SingleNodeDisputeArbitration`) |
| Typed node arbitration | requant element-wise (INT64_ACCUM input chunks), Q12.20 ops share the frozen V1 bounded arbiter; wide GEMM nodes refused; **devnet: the real ROPE node 85 fraud adjudicated with the pinned 2048-value table and one committed input chunk** |
| WIDE_GEMM_DISPUTE_V1 chain path | 4 additive messages; open only on the first divergent GEMM; trace claims; K-bisection; **512-MAC arbiter extracts tiles from type-proven chunks**; **devnet: the real block's GEMM 145 (A = node 144 output — the kind-1 operand path, fixed + regression-tested) settles ChallengerWins -> fraud -> zero VWR** |
| DA (GRAPH_VERIFICATION_BUNDLE_V2) | typed GraphDAAttestationV2 (own domain; Python provider signature **cross-verified under the Go verifier**); **2-replica quorum gate on finalize**; typed chunk challenge (node root via manifest proof + typed chunk proof); objective timeout penalty; availability-failed refund with zero VWR; **devnet: 2-of-3 providers attest the 39,648,960-byte real bundle (one provider offline) and the honest real block finalizes** |
| Queries / gas | aggregate GraphV2Status query; per-tx gas measured on the frozen schedule; **512-MAC witness: exactly 512 MAC, 5,550 bytes, 10 evidence chunks, 366,347 gas units** (`docs/phase-f5c-gas-results.json`) |

## Watcher V2 (permissionless, A3)

`tools/f5c_watcher_v2.py` on the real 310-node bundle (39.6 MB,
`docs/phase-f5c-bundle-size.json`):

- root phase: artifact hash, version, GraphIDV2, all 58 input roots, all
  310 node roots, manifest V2, final root — honest PASS; four structural
  frauds (fraud-bundle-vs-honest-commitments, forged final, forged
  manifest, forged graph id) all `root_mismatch`
  (`phase-f5c-watcher-root-tests.json`).
- math phase: 40-round Freivalds over all 83 GEMMs (1.1 s) + exact
  recomputation of all 227 cheap nodes (3.2 s); **`full_gemm_calls = 0`**.
- the self-consistent GEMM fraud (element altered, downstream recomputed,
  valid signature) is **detected and localized to node 2 tile (0,0)**;
  the ROPE fraud is detected at node 5 by the exact cheap-op check
  (`phase-f5c-watcher-fraud-localization.json`).
- restart equivalence across process runs; cost/peak memory recorded;
  inputs = bundle + chain task metadata only (no worker filesystem).
- **devnet re-runs**: the watcher ran end-to-end inside every A6 scenario
  against the exact bundles whose commitments the chain settled
  (watcher verdict `pass` for the honest real block, `fraud_detected`
  for the real GEMM-145 and ROPE-85 frauds).

## A6 — 4-validator devnet E2E (executed, all PASS)

Four equal-power validators (compose stack `prisma-mv-1`, image rebuilt
from this branch); every scenario records the four validators' height and
app hash and requires convergence.

| roadmap task | scenario | evidence | status |
|---|---|---|---|
| A6-02 | honest REAL 310-node block: post -> execute -> CommitV3 -> 2-of-3 DA -> window -> finalize | `docs/phase-f5c-e2e-honest.json` — exactly one VWR V3 `fk7JvIrf…`, requester escrow spent (burn 20% / monitors 2×5% / worker 70%) verified mechanically, watcher `pass` | **PASS** |
| A6-04 | real GEMM-145 fraud (self-consistent, valid CommitV3) -> challenge -> 8 V2 bisection rounds -> graph->wide bridge -> K-trace bisection -> 512-MAC arbitration -> ChallengerWins | `docs/phase-f5c-e2e-gemm-fraud.json` — first divergent node **145**, wide first divergent step **0**, fraud status, zero VWR, challenger settlement, requester refund, watcher `fraud_detected` | **PASS** |
| A6-06 | real ROPE-85 fraud -> 9 bisection rounds localize the ROPE node -> bounded table-pinned arbitration | `docs/phase-f5c-e2e-rope-fraud.json` — first divergent node **85**, fraud, zero VWR, watcher `fraud_detected` | **PASS** |
| A6-07 | false challenge (fabricated counter-claim on an honest result) | `docs/phase-f5c-e2e-false-challenge.json` — WorkerWins, task survives, challenger bond burned (balance-verified), exactly one VWR after the window, second finalize refused | **PASS** |
| A6-08 | combined: real GEMM fraud + one DA provider offline (2-of-3) + censoring proposer (validator-a, dev harness incl. the V2 family) | `docs/phase-f5c-e2e-combined.json` — proposer omission window observed per block (censored by validator-a at height 1335, included by validator-b the next block), dispute still completes with zero VWR | **PASS** |
| A6-09 | wide dispute: both K-traces locked, then ALL four validators restarted, then arbitration resumed | `docs/phase-f5c-e2e-wide-restart.json` — persisted dispute byte-identical after restart (same roots/high/low/first step), same verdict, challenger settlement, watcher still localizes | **PASS** |
| A6-10 | validator-d stopped for >5 blocks (chain keeps finalizing on 3/4), then restarted | `docs/phase-f5c-e2e-validator-restart.json` (also `…-validator-recovery.json`) — height advanced without d, peers converged during the outage, d caught up to the same height and app hash | **PASS** |
| A6-11 | DA quorum unreachable (1 attestation), window closed | `docs/phase-f5c-e2e-availability-failure.json` — `availability_failed`, requester escrow refunded (balance-verified), worker reservation released, zero VWR | **PASS** |

The fixture-graph variants (K=8 two-node / ROPE three-node chains) are
kept as fast deterministic regressions under the `*-small.json` names;
the roadmap-named files above hold the real-block runs where the roadmap
specifies them (A6-02/04/06/08).

### Defects found BY the devnet E2E (fixed on this branch, each with a
regression test)

1. **Single-node V2 disputes were unreachable**: `NewGraphDisputeV2` marks
   an already-converged interval arb-ready at creation, but nothing
   promoted the stored record (midpoints are refused once arb-ready), so
   a one-node graph could never reach `ArbitrateGraphNode`.
   `TestGraphV2SingleNodeDisputeArbitration` fails on the pre-fix code
   ("status bisection, want arb_ready").
2. **Wide arbitration panicked on node-output operands**: `ArbitrateWide512`
   read `graph.Inputs[ref.Index]` before checking `ref.Kind`, so a real
   profile GEMM whose activations come from an upstream node output
   (kind 1) panicked with `index out of range [144] with length 58`
   deterministically on every validator.  Fixed to resolve descriptors by
   kind; `TestWideGEMMDisputeNodeOperandArbitration` reproduces the exact
   panic on the pre-fix code and passes after.

Both were invisible to the fixture-only suites because the fixtures used
input operands and multi-node graphs — the real block exposed them.

## Compatibility

`go test ./...` (root) and `cd chain && go test ./...` green; the legacy
golden test pins V1 tensor roots, GraphID `f1ba2ebb…`, commit preimage
and receipt id byte-for-byte; pb.go regenerations were verified additive
(zero removed old protobuf tags).  No protocol file was reinterpreted;
the chain binary remains CPU-only (no CUDA, no float consensus path).

## Deviations from the roadmap (recorded per §0)

- The A6-08 combined artifact is produced by the same run as A6-04 (the
  censoring proposer stayed active for the whole scenario); the evidence
  name `docs/phase-f5c-e2e-combined.json` is the roadmap's.
- The dev censorship harness (`chain/app/devcensor.go`, devnet-only, off
  by default) was extended to the CANONICAL_GRAPH_V2 dispute messages so
  the omission scenario could target the V2 family.
- The devnet block time is 5 s (compose `timeout_commit`), so the
  30-block challenge/DA windows take ~5 minutes of wall clock; the
  harness waits them out instead of shortening protocol windows.
- A6-09 restarts all four validators (stronger than "relevant process").
- A6-10's deliverable name is `phase-f5c-e2e-validator-restart.json`
  (roadmap) with `phase-f5c-e2e-validator-recovery.json` kept as the
  harness's original name.
- CLI verb polish (friendly subcommands) remains deferred to Phase B
  B5-01; the E2E harness drives the Msg-service commands directly.
