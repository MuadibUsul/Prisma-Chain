# Phase F.5C report — Formal Wide-Integer Protocol Integration

**Status: PROTOCOL CORE DELIVERED, DEVNET E2E NOT TESTED.**  Every
protocol component of CANONICAL_GRAPH_V2 is implemented, cross-language
verified and committed; the four-validator devnet scenarios (A6) were not
executed in this work session, so per the roadmap's own rule any key item
NOT TESTED keeps Phase F at PARTIAL.  This report cites only artifacts
that exist in the repository.

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
| GraphDisputeV2 (trail claim, midpoint, snapshot incl. in-flight medians, restart) | injected fraud at node 0/1 bisects to the exact first-divergent node; mid-round snapshot -> restore -> same verdict |
| Typed node arbitration | requant element-wise (INT64_ACCUM input chunks), Q12.20 ops share the frozen V1 bounded arbiter; wide GEMM nodes refused |
| WIDE_GEMM_DISPUTE_V1 chain path | 4 additive messages; open only on the first divergent GEMM; trace claims; K-bisection; **512-MAC arbiter extracts tiles from type-proven chunks**; full on-chain e2e: self-consistent fraudulent result -> ChallengerWins -> fraud -> **zero VWR** |
| DA (GRAPH_VERIFICATION_BUNDLE_V2) | typed GraphDAAttestationV2 (own domain; Python provider signature **cross-verified under the Go verifier**); **2-replica quorum gate on finalize**; typed chunk challenge (node root via manifest proof + typed chunk proof); objective timeout penalty; availability-failed refund with zero VWR |
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

## Not tested (the honest gap)

- **A6 (4-validator devnet E2E)**: honest / GEMM-fraud / ROPE-fraud /
  false-challenge / combined-censorship / restart / availability-failure
  scenarios on the compose devnet, including four-validator app-hash
  convergence.  The scenario state machines are covered by chain unit
  tests (per scenario, listed above), but the cross-machine runs did not
  happen.  Required for any PHASE_F = PASS: per the freeze rule no pass
  tag is created.
- Watcher no-worker drill on a real stopped worker (A3-08 note covers the
  process-input audit; the drill belongs to A6).
- CLI verb polish (messages are full Msg-service entries; the devnet
  harness pattern in `deploy/multivalidator/f_graph_test.py` works with
  them; dedicated verbs are productization scope B5-01).

## Compatibility

`go test ./...` (root) and `cd chain && go test ./...` green; the legacy
golden test pins V1 tensor roots, GraphID `f1ba2ebb…`, commit preimage
and receipt id byte-for-byte; pb.go regenerations were verified additive
(zero removed old protobuf tags).  No protocol file was reinterpreted;
the chain binary remains CPU-only (no CUDA, no float consensus path).
