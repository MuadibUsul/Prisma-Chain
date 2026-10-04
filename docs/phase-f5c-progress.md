# Phase F.5C progress snapshot — Formal Wide-Integer Protocol Integration

**Status: PARTIAL.**  The protocol core of CANONICAL_GRAPH_V2 is
implemented, tested and committed; the devnet E2E / dispute / DA / watcher
layers are NOT finished.  Per the phase rules (any key item NOT TESTED ⇒
PHASE_F = PARTIAL), Phase F is not declared PASS yet.  This file is the
resume map for the next session.

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
| A1-04..A1-05 (chain layer) | pending | `buildTrailClaimV2` + GraphTrailClaim / GraphMidPoint dispatch (`graphV2InitialStateRoot` from descriptor inputs) |
| A1-07 cheap-op typed evidence | pending | V2 typed chunk evidence adapter |
| A2-01..A2-08 | pending | wide dispute wire/open/localization/trace/midpoint/512/bridge/restart |
| A3..A7 | pending | watcher/bundle, DA, query/CLI/gas, E2E, docs/freeze |

## Not done (in dependency order)

1. **V2 graph dispute chain** (§68–§72): trail claim / midpoint /
   node arbitration need V2 dispatch (GraphStateRootV2 trails, typed
   chunk evidence, `graphDescriptorV2` at the remaining call sites —
   `graph_keeper.go` lines ~264/339/480/542/736 still call the V1
   `graphDescriptor`).  Bisection state machine and bounded cheap-op
   evidence follow the V1 shape with V2 roots.
2. **Wide GEMM on-chain dispute messages** (§73–§83): OpenWideGEMMDispute /
   WideTraceClaim / WideMidPoint / ArbitrateWide512 (new proto messages
   or envelope dispatch), wired to `canonical.WIDE_GEMM_DISPUTE_V1`; the
   Graph→WideGEMM bridge (first divergent node is a GEMM ⇒ wide path);
   row/column/tile localization helper (§82).
3. **Watcher V2 + GraphVerificationBundleV2** (§54–§55, §105–§110):
   bundle codec (descriptor + inputs + node outputs + TensorRootV2 proofs
   + manifest proofs + final), Python watcher (root phase → Freivalds over
   83 GEMMs with post-commit randomness → exact cheap-op recompute),
   `full_gemm_calls = 0` instrumentation, restart-equivalence; bundle size
   measurement (§136).
4. **DA integration** (§57–§65, §130–§131): `GRAPH_VERIFICATION_BUNDLE_V2`
   artifact kind over the Phase E `DA_REPLICA_V1` registry (no second
   network); provider attestation-time verification; on-chain chunk
   challenge; quorum-gated finalize; availability-failure path; wrong-blob
   refusal; after-attest loss.
5. **Queries + CLI** (§132–§134): graph task/dispute/receipt V3/DA status
   query surface and CLI verbs for the E2E harness.
6. **Gas** (§96–§99, §140–§141): measured schedule incl. the 512-MAC
   arbiter witness (bytes, gas, proof depth) → `docs/phase-f5c-gas-results.json`.
7. **Devnet E2E** (§111–§129, §166–§172): 4-validator honest / real-GEMM
   fraud with downstream recompute / real-ROPE fraud / false challenge /
   combined adversarial (DA-offline + one-proposer censorship) / restarts;
   balances per scenario; four-validator app-hash convergence.
8. **Docs + reports** (§142–§143, §173–§174, §191): canonical-tensor-v2.md,
   canonical-graph-v2.md, wide-gemm-a13w10-v1.md, wide-requant-v1.md,
   freivalds-a13w10-v1.md, phase-f5c-report.md, phase-f-final-report.md
   (full lineage incl. the failed W8A8 / groupwise / A-only / int32
   frontier rounds), Phase F freeze tag.

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
