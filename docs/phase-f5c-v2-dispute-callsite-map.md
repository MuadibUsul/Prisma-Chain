# F.5C V2 dispute call-site map (roadmap A1-01)

Static classification of every graph-dispute call site in
`chain/x/compute/` at `0d7ce2b`.  Rule: V1 tasks keep the exact V1 route;
V2 tasks dispatch on `task.ProtocolVersion`; no message is ever silently
reinterpreted.

| # | call site | material used today | classification | V2 action |
|---|---|---|---|---|
| 1 | `graph_keeper.go` `SubmitGraphResult` (V1 message) | `graphDescriptor` (V1), V1 `GraphResultCommit` | **V1-only** | none — a V2 task's descriptor cannot parse as V1 (`protocol_version` mismatch makes `graphDescriptor` fail), so this route is physically unreachable for V2 tasks. V2 submissions use the V2-envelope dispatch at #8. |
| 2 | `OpenGraphChallenge` | `graphDescriptor` + challenger output-root count | **needs dispatch** | use `graphDescriptorV2` when `graphTaskProtocolV2`; dispute record gains an explicit protocol-version tag; challenger roots must match V2 output count; bond rules unchanged. |
| 3 | `GraphTrailClaim` → `buildTrailClaim` | V1 `GraphStateRoot` trail, V1 `StateLeaf`/`TrailLeaf` | **needs dispatch** | V2 claim builds with `GraphStateRootV2` + `TrailLeafV2` (A1-02/A1-03); endpoint proofs verified with the V2 tray verifier; claim layout stored in the same record struct with a version tag. |
| 4 | `GraphMidPoint` | V1 trail proof verification | **needs dispatch** | V2 midpoint verifies `TrailProofV2`; midpoint index stays derived from the locked interval (never party-supplied). |
| 5 | `ArbitrateGraphNode` | `RestoreGraphDispute(snapshot, V1 graph)` + V1 typed evidence | **needs dispatch** | V2 restores `GraphDisputeV2` from the V2 snapshot; node arbitration uses V2 typed chunk evidence (A1-07) and routes GEMM nodes to the wide dispute (A2). |
| 6 | dispute finalize branch (`FinalizeGraphTask`) | `RestoreGraphDispute` again for the resolved-outcome read | **needs dispatch** | same restore path as #5 with the version tag from the record. |
| 7 | `finalizeGraphOptimistic` | — | **already dispatched** (V2 build on `graphTaskProtocolV2`, V3 receipt) | done (commit `1719084`). |
| 8 | `SubmitGraphResultV2` | — | **already dispatched** (V3 commitment for V2 tasks) | done (commit `1719084`). |

Shared, version-safe helpers (no change required): `equalBytes32`,
`graphReceiptKey`, `cloneBytes2D`, `consumeGraphGas`, escrow/bond/settlement
mechanics, task storage layout (additive `ProtocolVersion` field).

Canonical-layer work this map implies (dependency order):
1. `GraphStateRootV2` public API + Python mirror (A1-02).
2. `TrailLeafV2` / `TrailLevelsV2` / `TrailProofV2` + verifier (A1-03).
3. `GraphDisputeV2` state machine + snapshot (A1-05/A1-08), forked from
   the V1 implementation with V2 domains — the V1 types stay frozen.
4. V2 typed cheap-op evidence adapter (A1-07).
5. Wide GEMM dispute envelope + bridge (A2).

No unclassified V1-only call site remains.
