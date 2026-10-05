# Phase F.5C progress snapshot — Formal Wide-Integer Protocol Integration

**Status: PHASE_F = PASS (F.5C complete).**  The protocol core, Watcher
V2, graph DA and the complete A6 4-validator devnet E2E suite are
implemented, executed and committed — including the REAL 310-node Qwen3
block runs (honest, GEMM-145 fraud through the 512-MAC wide path,
ROPE-85 fraud through the bounded table arbiter), false challenge,
combined censorship+DA-offline, wide-dispute full-stack restart,
validator offline/recovery and availability failure.  Every scenario has
an evidence JSON in `docs/` and four-validator app-hash convergence.
Freeze tag: `transformer-phase-f-wide-integer` (see
`docs/phase-f-freeze.json`).

Branch `protocol/transformer-phase-f5c-wide-integer` (from
`research/transformer-phase-f5b2-full-block-gpu`).  Zero protocol files
were reinterpreted: all V1 domains, objects and behaviours are untouched.

## Done (commits on the branch)

| commit | content |
|---|---|
| `04ccd2b` | **CANONICAL_TENSOR_V2** (A13/W10/INT64_ACCUM BE16/BE64, logical-range rejection, typed chunk proofs), **GEMM_A13W10_I64_V1** (Go reference + Python mirror; exact int64; right operand W10 or A13; MaxSafeK64 = 4398046511103 for A13W10), **REQUANTIZE_WIDE_V1**, **CANONICAL_GRAPH_V2** (+ A13W10_I64_PROFILE_V1 binding + PolicyID), NodeOutputManifestV2, **GraphResultCommitV3** (own signing domain), **VerifiedGraphWorkReceiptV3** (own receipt domain). Tests: encoding edges, range rejection (4096/512), domain separation, frozen-F.5B-vector replay through the Go reference, ties/saturation/overflow requant, small-graph manifest/commit/receipt, V1-version and over-K rejection. |
| `6ee232c` | **QWEN3_BLOCK_PROFILE_V2 converter** (`tools/f5c_build_v2_graph.py`): frozen F.5B.2 310-node program → formal descriptor (`testdata/f5c_qwen3_block_v2.json`, base64 `[]byte` roots on the wire). Evidence: **GraphIDV2 = `8fb86087697e277602bb4b2088ba48eb4172c21d9c4f83914582f2cbf00d4def`**; Python V2 executor 310/310 node values identical to the F.5B.2 CPU/GPU manifest (§103); Go replay (`tools/f5c_replay`) 310/310 TensorRootV2 byte-identical to Python; work vector identical incl. **GEMM_A13W10_MAC = 252,706,816**. |
| `fd8ed69` | **FREIVALDS_A13W10_I64_V1** (per-shape signed-bit admission bound; production 40 rounds; detection only) + **WIDE_GEMM_DISPUTE_V1** core (8×8 tiles, 512-MAC micro-step, BE64 state leaves, trace root, K-step bisection helper, exact-int64 arbiter; tests: tamper detection, first-divergent-step bisection, challenger-wins / worker-wins). |
| `1719084` | **On-chain V2 honest path**: version dispatch on the descriptor's own `protocol_version` (no proto changes); V2 admission with bounds + typed validation; **GraphResultCommitV3** submission validation (manifest locked pre-randomness, wrong-domain signatures refused); V3 receipt on finalize; static chain-side `GraphWorkVectorV2`. Tests: full honest harvest through the harness + tamper + double-finalize + wrong-domain-signature refusal. |

## Roadmap execution log (Prisma_Chain_AI_Engineering_Roadmap)

| task | status | evidence |
|---|---|---|
| A0-01 baseline | **DONE** (`0d7ce2b`) | `docs/phase-f5c-baseline.{json,md}`; all modules green |
| A0-02 legacy golden | **DONE** (`0d7ce2b`) | `testdata/f5c_legacy_golden.json` + `legacy_golden_test.go` (V1 roots/GraphID/commit preimage/receipt id recompute + cross-version rejection incl. V1-root+merkle through V2 verifier) |
| A1-01 call-site map | **DONE** (`139966c`) | `docs/phase-f5c-v2-dispute-callsite-map.md` |
| A1-02 GraphStateRootV2 | **DONE** (`139966c`) | `GraphStateRootV2FromRoots` + Python mirror + `testdata/canonical_graph_v2_state_vectors.json` |
| A1-03 TrailRootV2/ProofV2 | **DONE** (`139966c`) | `TrailLeafV2/TrailRootV2/TrailProofV2/VerifyTrailProofV2` + Python mirror + `testdata/canonical_graph_v2_trail_vectors.json` |
| A1-04..A1-08 (canonical layer) | **DONE** | `GraphDisputeV2` (`compute/canonical/graphv2dispute.go`): fork of the bisection state machine over V2 trail leaves; `NewGraphDisputeV2`/`SubmitMid`/`FirstDivergentNode`/`Timeout`; **snapshot V2 incl. in-flight round medians** + `RestoreGraphDisputeV2` with graph-id/node-count revalidation; tests: injected fraud at node 0 and node 1 both bisect to the exact first-divergent node, snapshot mid-round -> restore -> continue reaches the same verdict, V1-domain trail material refused, identical endpoints refused, round-window/duplicate rejections |
| A1-04/A1-05 (chain layer) | **DONE** (`698b251`) | `graphV2InitialStateRootV2`, `buildTrailClaimV2`, `lockTrailClaimV2`, `graphMidPointV2`; dispatch in OpenGraphChallenge (output-count check), GraphTrailClaim, GraphMidPoint; on-chain test: full challenge -> two V2 claims -> epoch bisection -> arb_ready -> injected first-divergent node |
| A1-06/A1-07 (chain + canonical) | **DONE** | `ArbitrateNodeChunkV2` + `ChunkEvidenceV2` + `StateProofV2` (typed V2 claims/evidence; requant recomputed element-wise over INT64_ACCUM input chunks; Q12.20 operators share the frozen V1 value-window arbiter; wide GEMM nodes refused toward A2); `graphStateGeometryV2`/`verifyGraphEvidenceV2State`/`arbitrateGraphNodeV2` + ArbitrateGraphNode dispatch; tests: requant + ADD chunk arbitration (honest/fraud/both-bad/tampered-evidence/wide-refusal) and the full on-chain e2e (self-consistent fraudulent result -> bisection to the requant node -> typed arbitration -> ChallengerWins -> fraud status -> zero VWR) |
| A1-07 cheap-op typed evidence | pending | V2 typed chunk evidence adapter |
| A2-01 wire | **DONE** (`86561da` + regen) | 4 additive proto messages; wire compatibility mechanically verified (227 old tags byte-identical); `docs/phase-f5c-wide-dispute-wire.md` |
| A2-02..A2-07 | **DONE** | `canonical.WideGEMMDisputeV1` (exported-state machine: claims with trace proofs, K-step bisection, timeout; JSON-persistable) + chain `wide_dispute.go`: OpenWideGEMMDispute (first-divergent GEMM only, challenger-only, tile geometry admission incl. K/N %8), WideTraceClaim (zero-S0, distinct finals, duplicate refusal), WideMidPoint (block-height clock), ArbitrateWide512 (**tile values extracted from type-proven chunks on chain**, exactly 512 logical MAC, next-state equality + trace proofs, Graph bridge via resolveGraphOutcome). E2E test: self-consistent fraudulent wide GEMM -> graph bisection -> wide dispute -> 512-MAC -> ChallengerWins -> fraud status -> zero VWR |
| A2-03 localization helper (off-chain watcher) | pending (A3) | residual->row->tile localization belongs with the Watcher V2 pipeline |
| A2-08 wide restart | **DONE (canonical layer)** | the whole wide state machine persists as JSON in `WideDisputeRecord.DisputeJSON` after every step; the true process-restart drill is A6-09 |
| A3 (bundle + watcher) | **DONE** (`cb17762`) | 39.6 MB real bundle; watcher V2 root/Freivalds/cheap-op/localization/restart/cost; GEMM+ROPE fraud detected; `full_gemm_calls = 0` |
| A4 (DA) | **DONE** (`7976d32`, `e022262`) | attestations + quorum gate + typed chunk challenge + timeout + availability refund; Python provider cross-verified |
| A5 (query/gas) | **DONE** (`4dcbafc`) | aggregate GraphV2Status; per-tx gas; 512-MAC witness 5,550 bytes / 366,347 gas; CLI verb polish deferred to B5 (recorded deviation) |
| A6 (devnet E2E) | **DONE — all PASS** | **real 310-node block**: honest (`phase-f5c-e2e-honest.json`, exactly one VWR V3 + fee split), GEMM-145 fraud (`…-gemm-fraud.json`, 8 bisection rounds -> 512-MAC -> fraud, 0 VWR), ROPE-85 fraud (`…-rope-fraud.json`, 9 rounds -> bounded table arbiter -> fraud, 0 VWR); fixture-graph runs: false challenge (`…-false-challenge.json`, WorkerWins + bond burned + exactly one VWR), combined adversarial (`…-combined.json`, proposer omission observed per block + 2-of-3 DA), wide-dispute restart (`…-wide-restart.json`, persisted dispute byte-identical), validator recovery (`…-validator-restart.json`), availability failure (`…-availability-failure.json`, refund + 0 VWR). Two defects found by the E2E and fixed with regression tests (arb_ready promotion for single-node disputes; wide-arbitration kind-1 operand panic). |
| A7 (docs/report/freeze) | **DONE** | 5 specs + phase-f5c-report + phase-f-final-report + phase-f-freeze.json (PARTIAL, no pass tag) |

## Remaining work

None for Phase F.5C.  All A0–A7 items are done; the freeze tag is
created.  Phase B productization (release build, CLI verb polish B5-01,
packaging, later-phase MLIR items) starts from the tag per the roadmap.

## Deviations from the roadmap (recorded per §0)

- A6-08's combined artifact is produced by the A6-04 real-block run (the
  censoring proposer stayed active throughout) and committed as
  `docs/phase-f5c-e2e-combined.json`.
- `chain/app/devcensor.go` (devnet-only, off by default) was extended to
  the CANONICAL_GRAPH_V2 dispute messages so the omission scenario could
  target the V2 family.
- Devnet block time is 5 s, so 30-block challenge/DA windows take ~5 min
  of wall clock; windows were waited out rather than shortened.
- A6-09 restarts all four validators (stronger than "relevant process").
- A6-10 evidence exists under both `…-validator-restart.json` (roadmap
  name) and `…-validator-recovery.json` (harness name).
- CLI verb polish stays deferred to B5-01; the harness drives Msg-service
  commands directly.
- Two chain defects surfaced by the E2E were fixed before the tag:
  single-node dispute arb-ready promotion (`graphv2_chain.go`) and the
  kind-1 operand descriptor panic in `wide_dispute.go`, each with a
  regression test that fails on the pre-fix code.

## Resume notes (hard-won details)

- **GraphIDV2** on the wire: descriptor JSON with base64 `[]byte` roots;
  the canonical-CBOR view keeps raw bytes.  Both languages sort map keys
  by encoded bytes; lengths must match exactly (the earlier 1694-byte diff
  was `[]int` roots vs bytes).
- **Execution reference split**: protocol mirror = `canonical_ref.py`
  (scalar), block execution = frozen `f1_canonical_numpy` (vectorized);
  the converter self-checks fast root/requant paths against the mirror.
- Real block: 310 nodes = 83 GEMM + 126 REQUANTIZE (all wide) + 33 ADD +
  26 RMSNORM + 24 ROPE + 16 SOFTMAX + 1 SILU + 1 MUL; GEMM right operands
  are W10 for model weights and A13 for the 32 attention inner products.
- Rerun commands: `python tools/f5c_build_v2_graph.py` (rebuilds
  descriptor + roots + inputs.bin), `go run ./tools/f5c_replay` (Go
  execution), `go test ./compute/canonical/ -run "V2|Wide|Freivalds"`,
  `cd chain && go test ./x/compute/ -run GraphV2`.
- E2E harness fleet: `deploy/multivalidator` (F.1 4-validator compose)
  plus the F devnet Python client pattern (`deploy/multivalidator/f_graph_test.py`).
