# Prisma Chain — AI Engineering Master Roadmap v1.1

**Frozen Protocol → Productization → Private Testnet → Public Testnet Alpha**

> **Phase F is complete and frozen. The project is no longer in protocol
> research mode. The active engineering goal is to turn the frozen protocol
> into software that an external developer and an external GPU operator can
> actually use.**
>
> **Phase F 已完成并冻结。项目不再处于协议研究模式。当前工程目标是把冻结
> 的协议变成外部开发者与外部 GPU 运营者可以真正安装、使用、提交任务、
> 贡献 GPU、独立验证的软件与网络。**

This file is the **single authoritative engineering roadmap** for all
future AI/Codex sessions.  If any other document conflicts with it, this
file wins.  The previous roadmap is kept only as historical v1.0 (see
§"Roadmap file strategy").

---

## 0. Metadata (front matter)

| field | value |
|---|---|
| roadmap_version | `1.1` |
| status | `ACTIVE` |
| protocol_phase | `FROZEN` |
| phase_f | `PASS` |
| active_phase | `B` |
| active_task | `B0-05` |
| freeze_tag | `transformer-phase-f-wide-integer` |
| freeze_tag_object | `330d27f5fc92d009aca9599cc1e0bb13a496c766` (annotated tag object) |
| freeze_commit | `89547fa97b62d72a4cd336a5dfd3b000b03eac60` (frozen protocol commit) |
| frozen_branch | `protocol/transformer-phase-f5c-wide-integer` |
| frozen_branch_head | `40b4fe353df7d2203d1609cceb10d24380abed7a` (docs-only commit after the tag) |
| product_branch | `product/testnet-alpha` (created 2026-10-05 by B0-02, based at the freeze tag commit) |
| product_baseline_sha | `89547fa97b62d72a4cd336a5dfd3b000b03eac60` (see `docs/productization/productization-baseline.json`) |
| graph_id_v2 | `8fb86087697e277602bb4b2088ba48eb4172c21d9c4f83914582f2cbf00d4def` |
| policy_id | `eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0` |
| predecessor | `Prisma_Chain_AI_Engineering_Roadmap_F5C_to_Public_Testnet_Alpha.md` (v1.0, historical) |

Verified mechanically on 2026-10-05 from the repository (commands and
results in §2.1).  Do not trust these values blindly in later sessions —
re-run the verification commands before relying on a SHA.

### Tag vs branch head (they are different — record both, never merge them)

| ref | value | meaning |
|---|---|---|
| freeze tag `transformer-phase-f-wide-integer` | tag object `330d27f5…`, resolves to commit `89547fa9…` | **the frozen protocol commit** — contains the complete F.5C protocol, both E2E bug fixes, the PASS reports and the freeze decision |
| branch head `protocol/transformer-phase-f5c-wide-integer` | `40b4fe35…` | `89547fa` + exactly one docs-only commit (`docs/phase-f-freeze.json` mapping record); no code difference |
| `origin/main` | `a01c5edc…` (merge of PR #1) | contains the F lineage only up to `37fd022` (`merge-base(origin/main, f5c-branch) = 37fd022`); **the frozen commits and the tag are NOT yet merged into main** |

Consequence for Phase B: the productization baseline **must** branch from
the freeze tag (`89547fa`), not from `origin/main` and not from the
long-lived protocol branch head.  Reconciling `main` with the frozen
lineage is an explicit B0-02 decision (see B0-02).

---

## 1. AI Critical Rule (read before every session)

```text
PHASE_F IS FROZEN.

AI MUST NOT execute A0-A7 again unless explicitly instructed to perform
a regression fix or a protocol-versioned follow-up.

The active execution entrypoint is B0-05.
```

- Phase A tasks (A0-* … A7-*) are `DONE` + `FROZEN`.  They exist in this
  document for lineage, audit trail and AI context only.
- Do **not** re-design the protocol.  Research is closed (§"Research
  Reopen Gate").
- Do **not** modify frozen protocol code without an explicit versioned
  protocol change (§"Protocol Freeze Boundary").
- Execute exactly **one** active/unlocked task per work session, then
  update this roadmap's status fields and emit the standard task report
  (§"AI Task Completion Report").

---

## 2. Ground Truth

### 2.1 Phase F verification (mechanical, 2026-10-05)

Commands executed and their results:

```text
git rev-parse transformer-phase-f-wide-integer
  -> 330d27f5fc92d009aca9599cc1e0bb13a496c766      (annotated tag object)
git rev-parse transformer-phase-f-wide-integer^{commit}
  -> 89547fa97b62d72a4cd336a5dfd3b000b03eac60      (frozen protocol commit)
git rev-parse origin/protocol/transformer-phase-f5c-wide-integer
  -> 40b4fe353df7d2203d1609cceb10d24380abed7a      (frozen branch head)
git merge-base --is-ancestor 89547fa origin/main
  -> no  (freeze commit not yet in main)
git merge-base origin/main origin/protocol/transformer-phase-f5c-wide-integer
  -> 37fd022
git diff --stat 89547fa 40b4fe3
  -> docs/phase-f-freeze.json | 6 ++++--   (docs only)
git merge-base --is-ancestor c4fa56a 89547fa  -> yes   (bug-fix 1 inside the tag)
git merge-base --is-ancestor dcb4ebc 89547fa  -> yes   (bug-fix 2 inside the tag)
```

Frozen identities recomputed from authoritative artifacts:

```text
GraphIDV2 (tools/f5c_bundle_v2.graph_id_v2_of over testdata/f5c_qwen3_block_v2.json)
  = 8fb86087697e277602bb4b2088ba48eb4172c21d9c4f83914582f2cbf00d4def     MATCH
PolicyID (descriptor arithmetic profile, same file)
  = eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0     MATCH
docs/phase-f-freeze.json records both values identically.
```

Authoritative Phase F status:

```text
PHASE_F = PASS
```

Not `PARTIAL`, not `MOSTLY PASS`, not `ESSENTIALLY COMPLETE`.

### 2.2 Numeric

```text
arith family      A13W10 (activation 13 bits, weight 10 bits)
accumulator       signed int64 (exact, no saturation in-range)
accuracy gate     PASS
  F.5A evidence: 0.999912 / 0.03893, zero elements > 0.05
  (docs/phase-f5a-verdict-final.json; MaxSafeK64(13,10) = 4398046511103)
```

### 2.3 GPU

```text
backend           TRUE_FUSED_MMA_A13W10
SM86              PASS   (NVIDIA A40)
SM89              PASS   (RTX 2000 Ada)
83/83 GEMM        bit-exact vs frozen CPU reference
310/310 real block bit-exact (all node values + roots + final output)
CPU fallback      0
canonical float ops 0
evidence          F.5B.1 (fused GEMM, weighted 1.17× both GPUs) and
                  F.5B.2 (full 310-node block exact on both GPUs)
```

### 2.4 Protocol (frozen, versioned, additive over V1)

```text
CANONICAL_TENSOR_V2          (A13=129, W10=130, INT64_ACCUM=131, Q12.20=1;
                              BE16/BE64 encodings; typed chunk proofs)
CANONICAL_GRAPH_V2           (GraphIDV2 above; A13W10_I64_PROFILE_V1 bound in;
                              PolicyID above)
GEMM_A13W10_I64_V1           (exact int64; right operand W10 or A13)
REQUANTIZE_WIDE_V1           (round-ties-even, saturating clamp, 55-bit bound)
NodeOutputManifestV2
GraphResultCommitV3          (own signing domain)
VerifiedGraphWorkReceiptV3   (own receipt domain)
```

V1 artifacts are byte-for-byte preserved (legacy golden pins GraphID V1
`f1ba2ebb…`, commit preimage, receipt id; zero reinterpretation).

### 2.5 Verification

```text
FREIVALDS_A13W10_I64_V1
  40 rounds per GEMM, detection only (never probabilistic slashing)
  post-commit randomness (manifest root | task_ref | round | nonce | CSPRNG)
  83 GEMMs in ~1.1 s on the real block
full_gemm_calls = 0
```

### 2.6 Fraud proof

```text
GraphDisputeV2            trail claim -> midpoint bisection -> first divergent node
WIDE_GEMM_DISPUTE_V1      8×8×8 tiles, K-step trace bisection,
                          512 logical MAC final arbitration on chain
                          (tile values extracted from type-proven chunks)
settlement                ChallengerWins / WorkerWins / BothInvalid,
                          deterministic, evidence-bound
```

### 2.7 DA

```text
GraphVerificationBundleV2     typed, canonical, artifact-hashed
2-of-3 quorum                 finalization gate (2 replicas required)
typed chunk challenge         objective, node-root-verified
objective timeout             deadline-bound, no discretion
availability_failed           quorum loss -> refund, zero VWR
```

### 2.8 Gas evidence (authoritative, `docs/phase-f5c-gas-results.json`)

```text
512-MAC witness:
  witness_bytes            5,550
  mac                      512          (exactly 512 logical MAC)
  evidence_chunks          10           (typed GraphChunkEvidence)
  chunk_payload_bytes      128
  proof_siblings           worker 1 / challenger 1
arbitrate_wide_512 gas     366,347      (schedule GraphGasV1, arb_unit 4, evidence 800)
outcome of the metered run: challenger_wins
```

Full per-step table (all from the same JSON): post 101,510; accept
124,972; submit V3 111,726; DA attestation #1 104,758 / #2 129,506;
finalize with quorum 186,696; challenge 134,856; trail claims 167,620;
midpoint round 217,592; open wide dispute 104,160; wide trace claims
390,760; arbitrate 366,347.

---

## 3. A6 — 4-validator devnet E2E: ALL PASS

Eight scenarios, each with a committed evidence JSON and four-validator
app-hash convergence.  The chain is `prisma-mv-1`, four equal-power
validators (1/1/1/1); the image is built from the frozen branch.

| roadmap task | scenario | evidence JSON | key verified facts |
|---|---|---|---|
| A6-02 | **honest REAL 310-node Qwen3 block**, end-to-end | `docs/phase-f5c-e2e-honest.json` | 310-node Graph V2; CommitV3; 2-of-3 DA (one provider intentionally offline); bundle 39,648,960 bytes; Watcher `pass`; finalization gives **exactly one VWR V3** `fk7JvIrf10d4NlR00CUeM9pIpxRBX9lv0GpfMcQviig=`; fee split 20% burn / 2×5% monitor / 70% worker balance-verified; four validators same app hash |
| A6-04 | **real GEMM fraud** (node 145) through the full 512-MAC path | `docs/phase-f5c-e2e-gemm-fraud.json` | self-consistent fraudulent computation; 8-round GraphDisputeV2 bisection; **first divergent node = 145**; graph→wide bridge; K-trace bisection; **first divergent step = 0**; exactly 512 MAC arbitration; ChallengerWins; `fraud`; **0 VWR**; challenger settlement + requester refund verified |
| A6-06 | **real RoPE fraud** (node 85) through the bounded arbiter | `docs/phase-f5c-e2e-rope-fraud.json` | self-consistent ROPE corruption (delta 100000); 9-round bisection localizes **real ROPE node 85**; bounded table-pinned arbitration (2048-value committed table + one committed input chunk); ChallengerWins; `fraud`; **0 VWR** |
| A6-07 | false challenge | `docs/phase-f5c-e2e-false-challenge.json` | fabricated counter-claim on an honest result; WorkerWins; task survives; challenger bond burned (balance-verified); exactly one VWR after the window; second finalize refused |
| A6-08 | combined adversarial (real GEMM fraud + DA provider offline + censoring proposer) | `docs/phase-f5c-e2e-combined.json` | 2-of-3 DA with one provider offline; proposer omission window observed per block (`censored_by=[1335]`, included by `validator-b` the next block — delay only, never invalid); dispute still completes, 0 VWR |
| A6-09 | wide-dispute restart | `docs/phase-f5c-e2e-wide-restart.json` | both K-traces locked, **all four validators restarted**, persisted dispute byte-identical (same roots / high / low / first step), same verdict, watcher still localizes |
| A6-10 | validator offline / recovery | `docs/phase-f5c-e2e-validator-restart.json` | one validator stopped: remaining 3/4 keep finalizing and processing V2 txs; restart catches up to same height + app hash |
| A6-11 | availability failure | `docs/phase-f5c-e2e-availability-failure.json` | DA quorum unreachable (1 attestation); window closes; `availability_failed`; requester refunded (balance-verified); worker reservation released; **0 VWR** |

The three critical scenarios — **honest, real GEMM fraud, real ROPE
fraud** — all ran on the **real 310-node Qwen3 block**, not on mini
fixture graphs.  Fixture-graph variants (K=8 two-node / ROPE three-node
chains) exist only as fast deterministic regressions under the
`docs/phase-f5c-e2e-*-small.json` names.

### 3.1 Two real bugs found by the E2E (lessons — preserved)

**Bug 1 — single-node graph dispute never reached arbitration**

```text
symptom     bisection already converged (high == 1) at trail-lock time,
            but the stored dispute record was never promoted to
            `arb_ready`; midpoints are refused on an arb-ready dispute,
            so `ArbitrateGraphNode` was unreachable.
fix         chain/x/compute/graphv2_chain.go — promote the record when the
            freshly created dispute reports ArbReady()
            (commit c4fa56a)
regression  TestGraphV2SingleNodeDisputeArbitration fails on pre-fix code
            ("status bisection, want arb_ready")
```

**Bug 2 — wide arbitration panicked on node-output operands**

```text
symptom     ArbitrateWide512 read graph.Inputs[ref.Index] before checking
            ref.Kind; a real-profile GEMM whose A operand is an upstream
            node output (kind = 1) provoked
            `index out of range [144] with length 58` deterministically on
            every validator (tx failed, consensus unaffected).
fix         chain/x/compute/wide_dispute.go — resolve operand descriptors
            by kind, never index the wrong table
            (commit dcb4ebc)
regression  TestWideGEMMDisputeNodeOperandArbitration reproduces the exact
            panic on pre-fix code and passes after
```

**Why this matters:** a mini fixture structurally could not expose bug 2
(the small graphs used input (kind 0) A operands), and the single-node
case (bug 1) was invisible to multi-node fixtures.  This is exactly why
the **real 310-node block E2E is a freeze condition**, not an optional
demo.  Never weaken a real-block gate to a mini-graph gate again.

---

## 4. Milestones

| Milestone | 目标 | Exit criteria | Status |
|---|---|---|---|
| M1 — F.5C Dispute PASS | V2 Graph dispute + Graph→Wide bridge + 512-MAC | real wide GEMM fraud deterministically adjudicated on chain | **DONE / FROZEN** |
| M2 — Watcher/DA PASS | Bundle V2 + Watcher V2 + 2-of-3 DA | worker disappearance still verifiable / challengeable | **DONE / FROZEN** |
| M3 — PHASE_F PASS | full 4-validator real-Qwen E2E suite | honest/fraud/false/DA/censor/restart all PASS + freeze tag | **DONE / FROZEN** |
| M4 — Productization Ready | 5 binaries + Job API + Scheduler + docs | a clean machine can install and run the network | **ACTIVE** (unlocked; entrypoint B0-05) |
| M5 — Private Testnet Alpha | multi-host, public-internet, invite-only | external simulated nodes close the loop across the internet | `PLANNED` / `BLOCKED_BY_M4` |
| M6 — Public Testnet Alpha | strangers can join, submit, execute, verify | ≥1 external Worker + external Watcher produce real settlement | `PLANNED` / `BLOCKED_BY_M5` |

### 4.1 M4 hard gate (no partial credit)

`M4 PRODUCTIZATION_READY` may be declared only when **all** of:

```text
prismad            builds reproducibly, starts, healthchecks, version metadata correct
prisma-worker      installs on a clean machine, joins, executes, commits
prisma-da          stores, challenges, responds, survives restart
prisma-watcher     detects fraud, auto-challenges, survives restart
prisma-cli         drives the above without editing source
Job API            submit/query a job end-to-end
Scheduler          capability filtering + FIFO + failure/reassign
clean-machine smoke   PASS for each component independently
```

Anything less keeps M4 `ACTIVE`.

### 4.2 Post-M4 / post-M6 rules

- Private Testnet Alpha (M5) must be **real multi-host, public
  IP/network** — not one compose host.
- Public Testnet Alpha (M6) is a real release: Protocol + Network + Node
  Software + API + Docs — not "deploy one server".
- Phase E (Beta / hardening) is `BLOCKED` until M6 PASS.  AI must not
  start it early.

---

## 5. Product context and architecture

Prisma = a verifiable decentralized AI compute network: a developer
submits a job, a GPU worker executes the frozen canonical computation,
the result is committed with cryptographic roots, data availability
providers store the verification bundle, permissionless watchers verify,
and the chain settles with a Verified Graph Work Receipt (VWR).

### Data flow

```text
Developer
   ↓
Job API
   ↓
Scheduler
   ↓
Worker GPU
   ↓
CommitV3
   ↓
DA
   ↓
Watcher
   ↓
Chain Settlement
   ↓
VWR
```

### Product architecture

```text
                     Developer
                         │
                      Job API
                         │
                      Scheduler
                         │
                  ┌──────┴──────┐
                  │             │
              Worker GPU    Worker GPU
                  │
              CommitV3
                  │
          ┌───────┼────────┐
          │       │        │
         DA1     DA2      DA3
          │       │        │
          └───────┼────────┘
                  │
               Watcher
                  │
             Prisma Chain
                  │
                  VWR
```

---

## 6. Protocol Freeze Boundary

```text
DO NOT MODIFY WITHOUT AN EXPLICIT VERSIONED PROTOCOL CHANGE
```

Frozen objects — hashes, encodings, wire formats, arithmetic and
semantics are all locked at the freeze tag:

```text
CANONICAL_TENSOR_V2          float-free canonical tensor encoding
CANONICAL_GRAPH_V2           GraphIDV2 8fb86087…  (descriptor semantics)
GEMM_A13W10_I64_V1           exact int64 wide GEMM
REQUANTIZE_WIDE_V1           wide requantization
FREIVALDS_A13W10_I64_V1      detection-only verifier
GraphDisputeV2               trail bisection state machine
WIDE_GEMM_DISPUTE_V1         512-MAC arbitration
CommitV3                     GraphResultCommitV3 signing domain
VWR V3                       VerifiedGraphWorkReceiptV3 domain
GraphIDV2                    8fb86087697e277602bb4b2088ba48eb4172c21d9c4f83914582f2cbf00d4def
PolicyID                     eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0
```

Also frozen: all V1 protocols, hashes, GraphID V1, Commit V1/V2,
receipts, and the CANONICAL_MATH_V1 layer.

Product code may **consume** these; it may not change them.  If a product
need appears to require a protocol change, stop and open a versioned
protocol proposal — do not "fix it in the worker".

---

## 7. Code layering (protocol vs product)

```text
protocol / chain / compute        frozen semantics; versioned only
worker / watcher / da / api /     product software; free to evolve
scheduler / cli / ops             as long as frozen semantics are consumed as-is
```

The physical directory layout is not mandated, but the **logical
boundary is**: product convenience must never bend protocol semantics.
A product bug is fixed in product code; a protocol defect requires a
versioned change.

---

## 8. Research Reopen Gate

```text
DEFAULT = DO NOT RESEARCH
```

Research may be reopened only for one of:

```text
- the bounded fraud proof turns out to be un-implementable in engineering
- the Watcher verification architecture proves hard-infeasible
- the DA artifact remains unusable after streaming/chunking engineering
- supporting a new model / new version (new profile, e.g. a second
  Qwen3 block or a different architecture)
- PoUW / consensus changes as a separate future phase
```

Explicitly **not** research (these are Engineering, handled inside Phase
B/C/D/E tasks):

```text
daemon bugs / network reconnect / Docker / config / CLI / API /
scheduler / performance tuning / IO / RPC overload / deployment / monitoring
```

Failed research history is preserved and must not be deleted (see
§"Phase F lineage").
---

## 9. Operating conventions

### 9.1 Status taxonomy (only these words)

```text
DONE        finished, evidence committed
ACTIVE      the one task currently being executed
PLANNED     unlocked-in-principle, waiting for its turn
BLOCKED     dependencies unmet (state the blocker)
FROZEN      finished and locked; must not be redone
DEPRECATED  superseded; explain why, never delete silently
```

Never write "mostly done", "almost finished", "essentially complete".

### 9.2 Standard task structure (every unfinished task)

```text
Task ID
Status
Goal
Dependencies
Allowed changes
Forbidden changes
Implementation
Tests
Artifacts
Definition of Done
Failure / Blocked behavior
Stop Rule
Next unlocked tasks
```

### 9.3 Priority

```text
P0   B0, B1, B2, B4, B3, B5 core, B6 core
P1   B7, Private Testnet operational polish
P2   Beta improvements
```

Dependencies may reorder within a priority; priority never overrides a
dependency.

### 9.4 No time estimates

Do not write "2 days" / "1 week" unless there is real historical data.
Use dependency / complexity / priority instead.

### 9.5 AI Resume Protocol (start of every new session)

```text
1. Read this roadmap (v1.1).
2. Read current git branch / head / status.
3. Find the ACTIVE task (§Execution Queue).
4. Check its dependencies.
5. Execute exactly ONE active/unlocked task.
6. Update this roadmap's status fields.
7. Emit the standard task report (§10).
8. Do not re-ask for project background; it is all here.
```

### 9.6 Stop Rules (global)

- If `PHASE_F != PASS` ever became false (it is PASS; only a protocol
  regression could change that) — stop and report.
- If a task requires changing frozen protocol semantics — stop, report,
  propose a versioned change.
- Never create a second conflicting roadmap.
- Never delete historical evidence to make a current task look cleaner.

---

## 10. AI Task Completion Report (mandatory format)

```text
Task ID

Status

branch / head

files changed

implementation

tests run

real results

artifacts

compatibility

blockers

roadmap status changes

newly unlocked task IDs

verdict
```

Rules: numbers must come from real runs or committed artifacts; if a
key item was not tested, say `NOT TESTED` — that word is not softened
anywhere in this project.

---

## 11. Execution Queue

```text
ACTIVE:
  B0-05   Productization toolchain baseline

NEXT (dependency-ordered, unlocked as predecessors complete):
  B1-01   prismad reproducible release build

BLOCKED (do not start):
  B1-02, B1-03            -> after B1-01
  B2-*                    -> after B1-01 (+ B0 done)
  B4-*, B3-*, B5-*, B6-*, B7-*  -> see each task's dependencies; the
                                   recommended product order is
                                   B0 -> B1 -> B2 -> B4 -> B3 -> B5 -> B6 -> B7
  Phase C (M5)            -> BLOCKED_BY_M4
  Phase D (M6)            -> BLOCKED_BY_M5
  Phase E                 -> BLOCKED_BY_M6
```

Only `B0-05` is `ACTIVE`.  Do not mark all B tasks ACTIVE.
(B0-01, B0-02, B0-03, B0-04 are DONE — see their task blocks for evidence.)

### 11.1 Phase B execution order (revised in v1.1)

```text
B0   Productization transition (NEW in v1.1)
 ↓
B1   Core distribution (prismad / genesis / public RPC)
 ↓
B2   Worker
 ↓
B4   DA
 ↓
B3   Watcher
 ↓
B5   CLI
 ↓
B6   Job API / Scheduler
 ↓
B7   Faucet / Docs / Status
 ↓
M4  PRODUCTIZATION_READY
```

**Why B4 (DA) before B3 (Watcher):** the real data flow is
`Worker → CommitV3 → GraphVerificationBundleV2 → DA → Watcher →
Settlement`.  The Watcher product must be built on a stable DA retrieval
surface, so DA productization unlocks Watcher productization, not the
other way round.

### 11.2 B1 internal order (revised)

```text
B1-01  prismad reproducible release build
  ↓
B1-03  genesis / config generator
  ↓
B1-02  public RPC / sentry mode
```

Rationale: the genesis/config generator produces the artifacts (chain-id,
genesis, seeds, DA defaults) that the public RPC node mode must be
configured against; building the RPC mode first would hardcode devnet
assumptions.

---

## 12. Roadmap file strategy

- v1.0 (`Prisma_Chain_AI_Engineering_Roadmap_F5C_to_Public_Testnet_Alpha.md`)
  is **historical**; it is preserved un-deleted (committed copy:
  `docs/roadmap/Prisma_Chain_AI_Engineering_Roadmap_v1.0_historical.md`).
  Its content is not current state; only this v1.1 file is.
- v1.1 lives at `docs/roadmap/Prisma_Chain_AI_Engineering_Roadmap_v1.1.md`
  (this file).
- No second roadmap may be created.  If a rewrite is needed, bump the
  version in place (v1.2) and mark this file historical.

### 12.1 Migration bookkeeping

- `docs/roadmap-v1.1-migration.json` — machine-generated counts and
  identity mapping for this rewrite (task counts, added/removed/renamed
  IDs, freeze refs).
- Task-count reconciliation v1.0 → v1.1: see §Appendix C.

---

## 13. Phase A — Completed & Frozen Tasks (historical, do not execute)

All tasks below are `Status: DONE`, `Phase: FROZEN`.  They are kept for
lineage, audit and AI context.  Each entry preserves its original task
ID, goal, dependencies and Definition of Done, plus the delivery
evidence.  Full original wording remains in the v1.0 historical file.

Evidence key (abbreviations used in the table):
`3f5c` = worktree `E:\Codes\Prisma-Chain-phase-f5c`, branch
`protocol/transformer-phase-f5c-wide-integer` (frozen; tag in §0).


### A0 — Baseline / Safety

- **A0-01 · 冻结当前基线**
  - 目标：记录当前 branch、head、Go/Python toolchain 版本；运行 `go test ./...` 与 Python canonical / F.5C tests；保存 baseline test summary；确认 working tree 干净或明确记录非本任务改动
  - 依赖：无｜DoD：baseline 可复现；后续所有失败能与 baseline 对比
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `0d7ce2b` · docs/phase-f5c-baseline.json, docs/phase-f5c-baseline.md
- **A0-02 · 增加旧协议不可变回归锚点**
  - 目标：把历史 V1 TensorRoot / GraphID / Commit / Receipt / GEMM vectors 固定为 golden fixtures；为 V1→V2、V2→V1 domain confusion 增加明确 rejection tests；为旧 transaction behavior 增加 regression tests
  - 依赖：A0-01｜DoD：旧协议 byte-for-byte / behavior-for-behavior 不变
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `0d7ce2b` · testdata/f5c_legacy_golden.json + legacy golden test: V1 GraphID f1ba2ebb…, commit preimage and receipt id pinned byte-for-byte; cross-version rejection

### A1 — V2 Graph Dispute Chain

- **A1-01 · 枚举所有 V1-only graph dispute call sites**
  - 目标：静态搜索所有 `graphDescriptor` / V1 GraphStateRoot / V1 trail root / V1 TensorRoot proof 调用点；为每个 call site 标记：V1-only / needs version dispatch / shared safe helper；生成迁移表
  - 依赖：A0-01｜DoD：不存在未分类的 V1-only call site
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `139966c` · docs/phase-f5c-v2-dispute-callsite-map.md
- **A1-02 · 实现 GraphStateRootV2**
  - 目标：定义独立 V2 state leaf domain；state leaf 绑定 `kind/index/TensorRootV2`；实现 deterministic ordering；实现 Go + Python mirror
  - 依赖：A1-01｜DoD：GraphStateRootV2 跨语言 bit-exact；domain-separated
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `139966c` · GraphStateRootV2FromRoots + testdata/canonical_graph_v2_state_vectors.json (Go == Python)
- **A1-03 · 实现 TrailRootV2 / TrailProofV2**
  - 目标：定义 V2 trail leaf domain；leaf 绑定 GraphIDV2、step index、GraphStateRootV2；实现 Merkle build / inclusion proof / verify；禁止复用 V1 leaf domain
  - 依赖：A1-02｜DoD：TrailRootV2 可跨语言重放
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `139966c` · TrailLeafV2/TrailRootV2/TrailProofV2/VerifyTrailProofV2 + testdata/canonical_graph_v2_trail_vectors.json; V1 trail material rejected
- **A1-04 · 为 GraphTrailClaim 增加 V2 dispatch**
  - 目标：按 task descriptor `protocol_version` 明确 dispatch V1/V2；V2 initial root 必须等于 descriptor-derived GraphStateRootV2；V2 endpoint proof 使用 TrailProofV2；V1 路径完全保持原行为
  - 依赖：A1-02,A1-03｜DoD：同一 message 不能跨版本解释；V2 claim 可进入 dispute
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `698b251` · compute/canonical/graphv2dispute.go (NewGraphDisputeV2/SubmitMid/Timeout, restart-safe snapshot incl. in-flight medians) + chain lockTrailClaimV2/buildTrailClaimV2
- **A1-05 · 为 GraphMidPoint 增加 V2 dispatch**
  - 目标：V2 midpoint proof 验证 TrailRootV2；继续使用 block-height derived round clock；锁定 midpoint index 计算，禁止 party 自报任意 index；V1 路径不变
  - 依赖：A1-04｜DoD：能稳定二分到单节点
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `698b251` · graphMidPointV2 + chain test: challenge -> two V2 claims -> epoch bisection -> arb_ready -> injected first-divergent node
- **A1-06 · 实现 V2 first-divergent-node finalize**
  - 目标：当 high-low=1 时解析 first divergent node；校验 node id 与 descriptor V2 中 operator/version 一致；输出 versioned node arbitration context
  - 依赖：A1-05｜DoD：定位 node 与 injected node 100% 一致
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `698b251` · chain/x/compute/graphv2arbiter.go (ChunkEvidenceV2/StateProofV2/ArbitrateNodeChunkV2) + chain/x/compute/graphv2_chain.go arbitrateGraphNodeV2 + graphStateGeometryV2
- **A1-07 · V2 cheap-op typed evidence adapter**
  - 目标：ADD/MUL/REQUANTIZE/RMSNORM/ROPE/SILU/SOFTMAX evidence 全部切换到 TensorRootV2 typed chunk proofs；每个 operator 重新从 graph geometry derive evidence index；不得信任 caller supplied dtype/index
  - 依赖：A1-06｜DoD：所有 cheap op V2 arbitration bounded 且 typed
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `698b251` · typed chunk arbitration tests: requant element-wise over INT64_ACCUM chunks; Q12.20 ops reuse the frozen V1 window arbiter; wide GEMM refused toward A2; honest/fraud/both-bad/tampered-evidence cases
- **A1-08 · V2 dispute snapshot / restart**
  - 目标：定义 V2 snapshot domain/version；snapshot 持久化 interval、claims、round clock、GraphIDV2；restore 时验证 GraphIDV2 / protocol version；旧 V1 snapshot 不变
  - 依赖：A1-05,A1-06｜DoD：restart-equivalence PASS
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `698b251` · full on-chain e2e: self-consistent fraudulent result -> bisection to the requant node -> typed arbitration -> ChallengerWins -> fraud -> zero VWR

### A2 — Graph → Wide GEMM Bridge / 512-MAC

- **A2-01 · 定义 Wide GEMM dispute chain envelope**
  - 目标：确定是否新增 proto messages 或使用明确 versioned envelope；选择最小 additive 方案；字段必须绑定 task id、GraphIDV2、node id、tile、round/epoch、party identity；定义 replay protection
  - 依赖：A1-06｜DoD：wire version 明确且 additive
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `86561da` · docs/phase-f5c-wide-dispute-wire.md; 4 additive messages; 227 old protobuf tags byte-identical after regen
- **A2-02 · 实现 OpenWideGEMMDispute**
  - 目标：只允许 first-divergent node 为 `GEMM_A13W10_I64_V1` 时进入；校验 task active dispute / challenger identity / bond；绑定 M/N/K/transpose_b/tile geometry；初始化 wide dispute record
  - 依赖：A2-01｜DoD：Graph node 与 wide dispute context 一一绑定
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `86561da` · chain/x/compute/wide_dispute.go OpenWideGEMMDispute: challenger-only, first-divergent-GEMM-only, tile geometry admission incl. K/N % 8
- **A2-03 · 实现 bad row / bad column / 8×8 tile localization helper**
  - 目标：Freivalds mismatch 后从 residual 定位 bad row；exact row recompute 找 bad column；映射到 canonical 8×8 output tile；处理 edge tile zero padding
  - 依赖：A2-02｜DoD：所有 injected wide GEMM fraud 可确定性定位到 tile
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `cb17762` · tools/f5c_watcher_v2.py residual -> row/column -> 8x8 tile localization + docs/phase-f5c-watcher-fraud-localization.json
- **A2-04 · 实现 WideTraceClaim**
  - 目标：partial state 固定 64×signed int64，BE64；trace length = ceil(K/8)+1；生成 / 验证 trace root + endpoint proofs；trace 只在 challenge 后生成
  - 依赖：A2-02｜DoD：chain 可锁定双方 trace claims
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `86561da` · WideTraceClaim (canonical.WideTraceClaimV1 + chain): zero-S0, distinct finals, duplicate refusal, trace-root proofs
- **A2-05 · 实现 WideMidPoint**
  - 目标：按 K-step trace midpoint 二分；round clock 绑定 block height；proof index 由 interval derive；禁止跳步
  - 依赖：A2-04｜DoD：区间收敛到一个 8-wide K step
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `86561da` · WideMidPoint on the block-height clock + canonical K-step bisection (SubmitMid/Timeout)
- **A2-06 · 实现 ArbitrateWide512**
  - 目标：输入：8×8 A tile、8×8 B tile、prev state、party next states、typed proofs；按 logical arithmetic exact int64 重算 8×8×8 = 512 MAC；处理 transpose_b；GPU decomposition / Karatsuba 不得进入 chain implementation；输出 WorkerWins / ChallengerWins / BothInvalid
  - 依赖：A2-05｜DoD：512-MAC arbiter 与 Go/Python wide reference bit-exact
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `86561da` · ArbitrateWide512: tile values extracted from type-proven chunks on chain, exactly 512 logical MAC, next-state equality + trace proofs
- **A2-07 · Graph→Wide settlement bridge**
  - 目标：Graph first-divergent GEMM node 转入 wide dispute；wide verdict 回写 Graph dispute transcript；调用现有 economics settlement：refund/slash/reward；fraud worker 永不生成 VWR
  - 依赖：A1-06,A2-06｜DoD：Graph dispute 与 wide verdict 完整闭环
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `86561da` · Graph->Wide bridge via wideOutcomeToGraphOutcome/resolveGraphOutcome + chain e2e test (wide_dispute_test.go)
- **A2-08 · Wide dispute restart persistence**
  - 目标：snapshot wide interval / claims / tile / trace roots / deadline；恢复时校验 task/node/GraphIDV2；继续到相同 final step
  - 依赖：A2-05,A2-06｜DoD：restart-equivalence PASS
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `52519b3` · WideDisputeRecord.DisputeJSON persisted after every step; real process-restart drill executed as A6-09 (docs/phase-f5c-e2e-wide-restart.json: byte-identical persisted dispute, same verdict)

### A3 — Watcher V2 / Verification Bundle

- **A3-01 · 定义 GraphVerificationBundleV2 schema**
  - 目标：包含 GraphDescriptorV2、inputs、weights/constants、all node outputs、TensorRootV2 proofs、ManifestV2 proofs、final outputs；明确 artifact version 与 hash domain；禁止包含正常路径 execution trace / dispute trail
  - 依赖：A0-02｜DoD：bundle schema 可独立解析并验证结构
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `cb17762` · tools/f5c_bundle_v2.py canonical codec (header canonical-CBOR; artifact hash SHA256('PRISMA_GRAPH_BUNDLE_V2\0' || header || SHA256(blobs)))
- **A3-02 · 实现 BundleV2 builder**
  - 目标：从 formal V2 execution 构造 bundle；节点顺序严格匹配 ManifestV2；生成 artifact root / size breakdown；支持 real Qwen 310-node bundle
  - 依赖：A3-01｜DoD：真实 bundle 可机械生成
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `cb17762` · tools/f5c_real_fraud.py: self-consistent fraud bundles (one node altered, all downstream recomputed, manifest/final/signature rebuilt)
- **A3-03 · 实现 Watcher root phase**
  - 目标：验证 GraphIDV2；验证 artifact root；验证 NodeOutputManifestRootV2；验证每个 TensorRootV2 / proof；验证 final output root；任一 root mismatch 立即停止数学验证
  - 依赖：A3-01,A3-02｜DoD：root phase 对结构性 fraud 100% 拒绝
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `cb17762` · docs/phase-f5c-watcher-root-tests.json: artifact hash, version, GraphIDV2, 58 input roots, 310 node roots, ManifestV2, final root; four structural frauds rejected
- **A3-04 · 实现 production post-commit randomness**
  - 目标：Worker CommitV3 锁定后才生成 Freivalds randomness；使用 CSPRNG + commit context + watcher nonce + round index；测试模式允许固定 seed，但显式 TEST ONLY；禁止 `r=H(manifest_root)` 作为唯一随机源
  - 依赖：A3-03｜DoD：随机性时序满足 post-commit requirement
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `cb17762` · post-commit randomness source (manifest_root || task_ref || round || nonce || CSPRNG) in tools/f5c_watcher_v2.py; never H(manifest) alone
- **A3-05 · Watcher wide GEMM Freivalds verifier**
  - 目标：83 real GEMM 使用 `FREIVALDS_A13W10_I64_V1`；production 40 rounds；支持 A13×W10 与 A13×A13、transpose_b；记录 per-node rounds/time/bounds
  - 依赖：A3-04｜DoD：`wide_freivalds_gemms=83`; 无 full GEMM
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `cb17762` · docs/phase-f5c-watcher-freivalds.json: 40 rounds over all 83 GEMMs (~1.1 s), detection only
- **A3-06 · Watcher cheap-op exact verifier**
  - 目标：RMSNORM / REQUANTIZE / ROPE / ADD / SOFTMAX / SILU / MUL exact canonical recompute；节点数由 graph 派生；逐 node compare committed output
  - 依赖：A3-03｜DoD：所有 non-GEMM nodes exact verified
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `cb17762` · docs/phase-f5c-watcher-cheapops.json: all 227 cheap nodes recomputed exactly (~3.2 s); full_gemm_calls = 0
- **A3-07 · Watcher fraud localization pipeline**
  - 目标：root phase PASS 后进入 math phase；GEMM mismatch → row/column/tile localization；cheap op mismatch → first suspected node/evidence builder；输出 challenge plan
  - 依赖：A3-05,A3-06,A2-03｜DoD：可自动生成链上 challenge 所需上下文
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `cb17762` · docs/phase-f5c-watcher-fraud-localization.json: self-consistent GEMM fraud -> node 2 tile (0,0); ROPE fraud -> node 5
- **A3-08 · Watcher no-worker-filesystem audit**
  - 目标：将 Worker local artifacts 从 test environment 移除；Watcher 只给 chain metadata + BundleV2；加入 forbidden-path audit
  - 依赖：A3-07｜DoD：Watcher 对 Worker 本地状态零依赖
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `cb17762` · docs/phase-f5c-watcher-independence.json: inputs = bundle + chain task metadata only (no worker filesystem); the live re-run happened inside the A6 scenarios (watcher verdicts recorded in each E2E JSON)
- **A3-09 · Watcher restart-equivalence**
  - 目标：在 root phase / Freivalds / challenge build 之间 kill process；新进程从 chain + DA 重新获取 state；不从旧内存恢复随机未提交状态
  - 依赖：A3-08｜DoD：fresh watcher instance produces same verifiable outcome
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `cb17762` · docs/phase-f5c-watcher-restart.json: restart equivalence across process runs
- **A3-10 · Watcher cost instrumentation**
  - 目标：记录 root validation、Freivalds、cheap-op、bundle decode、total seconds；记录 `full_gemm_calls`；记录 peak memory / downloaded bytes
  - 依赖：A3-05,A3-06｜DoD：`full_gemm_calls=0`；所有 timing 非负且来自真实运行
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `cb17762` · docs/phase-f5c-watcher-cost.json + docs/phase-f5c-bundle-size.json (real bundle 39,648,960 bytes)

### A4 — DA Integration

- **A4-01 · 新增 GRAPH_VERIFICATION_BUNDLE_V2 artifact kind**
  - 目标：新增 versioned artifact kind；复用现有 DA_REPLICA_V1 provider identity / bond / quorum；不创建新 provider registry
  - 依赖：A3-01｜DoD：同一 DA network 支持 V1 GEMM 与 Graph Bundle V2
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `7976d32` · chain/x/compute/graph_da.go: GraphDAAttestationV2 in its own domain (GRAPH_DA_ATTESTATION_V2/1.0.0)
- **A4-02 · Provider pre-attestation verification**
  - 目标：decode bundle；验证 GraphIDV2 / artifact hash / all TensorRootV2 / manifest / final root；仅证明 stored bytes 与 Worker commitment 一致；不得运行 Transformer 数学
  - 依赖：A4-01,A3-02｜DoD：Provider 不会为不匹配 commitment 的 bytes attest
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `e022262` · tools/f5c_da_provider.py pre-attestation verification; canonical signing preimage byte-matches Go; cross-verified through tools/f5c_verify_att
- **A4-03 · Graph V2 DA quorum gate**
  - 目标：ResultCommitV3 后等待 2-of-3 valid attestations；只有 quorum 后进入 challenge_window_ready；finalize 再次确认 availability state
  - 依赖：A4-02｜DoD：Graph finalize 被 DA quorum 正确约束
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `e022262` · 2-replica quorum gate on finalize (graph_keeper.go); devnet proved it with 2-of-3 real-bundle attestations gating the honest finalize
- **A4-04 · Typed on-chain DA chunk challenge**
  - 目标：challenge 绑定 task/artifact/node/tensor/chunk/provider；provider 响应 typed chunk bytes + TensorChunkProofV2 + artifact binding；chain 验证 proof 与 deadline
  - 依赖：A4-01,A4-02｜DoD：DA availability 可由客观 on-chain proof 结算
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `e022262` · typed on-chain chunk challenge: node root verified via manifest inclusion proof + typed chunk proof; responder path in tools/f5c_da_provider.py
- **A4-05 · DA after-attest loss timeout**
  - 目标：provider attest 后删除 artifact；发起 on-chain challenge；等待 block-height deadline；按已有 Phase E semantics 处罚 provider
  - 依赖：A4-04｜DoD：HTTP failure 不直接 slash；只有 missed on-chain deadline 才处罚
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `e022262` · objective timeout penalty (DAWindowBlocks = 30, DAPenaltyUprsm); devnet availability-failure run (A6-11)
- **A4-06 · DA quorum loss availability_failed**
  - 目标：使有效 replicas 降到 <2；推进到 availability expiry；task 标记 availability_failed；refund requester，0 VWR
  - 依赖：A4-03,A4-05｜DoD：quorum loss 永不生成 VWR
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `45ad9f8` · FailGraphAvailability -> availability_failed + requester refund + zero VWR; docs/phase-f5c-e2e-availability-failure.json
- **A4-07 · One-provider-offline recovery**
  - 目标：让 1/3 provider offline；Watcher 从剩余 2 provider 拉 bundle；完成完整 verification / challenge
  - 依赖：A4-03,A3-08｜DoD：单 provider offline 不阻断 verification
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `d5b2a5a` · one-provider-offline recovery: the real-block honest run finalized on 2-of-3 attestations with one provider offline; re-run under censorship in the combined scenario

### A5 — Query / CLI / Gas

- **A5-01 · GraphTaskV2 query**
  - 目标：返回 task version/status/GraphIDV2/worker/challenge window/DA status/final root；不要暴露内部 raw store layout
  - 依赖：A1-04,A4-03｜DoD：外部工具不需要读 ABCI raw store
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `4dcbafc` · chain/x/compute/graph_query_v2.go: aggregate GraphV2Status query
- **A5-02 · GraphDisputeV2 query**
  - 目标：返回 dispute phase、interval、deadline、node id、wide dispute state、parties；敏感/冗余 witness 不全量返回
  - 依赖：A1-06,A2-07｜DoD：CLI/Watcher 可从正式 query 恢复状态
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `4dcbafc` · dispute status exposed through the aggregate query (same file)
- **A5-03 · ReceiptV3 query**
  - 目标：按 task / receipt id 查询 VWR V3；返回 GraphIDV2、PolicyID、work vector、verification mode、settlement ref、DA ref
  - 依赖：A2-07｜DoD：receipt query 可外部使用
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `4dcbafc` · receipt id / finalization info exposed through the aggregate query (same file)
- **A5-04 · DA status query**
  - 目标：返回 provider attestations、valid replica count、required quorum、challenge state、expiry
  - 依赖：A4-03｜DoD：Watcher/CLI 可判断 availability
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `4dcbafc` · DA status exposed through the aggregate query (same file)
- **A5-05 · CLI V2 verbs**
  - 目标：post graph-v2 task；accept；submit commit-v3；query task/dispute/receipt/DA；challenge / trail / midpoint / wide dispute / arbitrate actions 覆盖 E2E harness
  - 依赖：A5-01,A5-02,A5-03,A5-04｜DoD：devnet E2E 不再依赖 ad-hoc raw-store helpers
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `4dcbafc` · Msg-service CLI commands drive the whole E2E suite; friendly-verb polish deliberately deferred to B5-01 (recorded deviation)
- **A5-06 · Gas instrumentation points**
  - 目标：为 PostGraphV2 / Accept / CommitV3 / DA attestation / challenge / trail / midpoint / wide open / wide trace / wide midpoint / wide512 / RoPE arbiter / finalize 记录 consumed gas；Gas 按 bounded units，不按 wall time
  - 依赖：A2-06,A4-04｜DoD：所有关键 tx 有可测 gas path
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `4dcbafc` · gas instrumentation points, schedule GraphGasV1 (docs/phase-f5c-gas-results.json)
- **A5-07 · 实测 512-MAC witness bytes / gas / proof depth**
  - 目标：在真实 Qwen GEMM node 上构造 final wide arbiter tx；记录 witness bytes、Merkle siblings/depth、gas；确认 transaction 未突破当前 block gas envelope
  - 依赖：A2-06,A5-06｜DoD：报告包含 authoritative 512-MAC measurements
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `4dcbafc` · measured 512-MAC witness: 5,550 bytes, 10 evidence chunks, 128-byte chunk payloads, proof depth 1+1, exactly 512 MAC, 366,347 gas

### A6 — 4-Validator Real-Qwen E2E

- **A6-01 · 升级 multivalidator harness 到 Graph V2**
  - 目标：4 validators 25% each；支持 Graph V2 tx / queries / DA provider endpoints / watcher runner；记录每个 validator height/app hash
  - 依赖：A5-05｜DoD：harness 能机械执行场景并收集证据
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `37fd022`+`35b1dd5` · deploy/multivalidator/f5c_v2_e2e.py + f_graph_test.py client pattern (per-validator height/app-hash collection)
- **A6-02 · Honest real-Qwen E2E**
  - 目标：Post 310-node Graph V2；Worker accept；执行 frozen protocol executor；CommitV3；DA 2-of-3；Watcher PASS；challenge window end；Finalize
  - 依赖：A3-10,A4-07,A6-01｜DoD：honest real block end-to-end PASS
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `d5b2a5a` · docs/phase-f5c-e2e-honest.json: real 310-node block finalizes with exactly one VWR V3; fee split balance-verified
- **A6-03 · Real wide GEMM fraud artifact generator**
  - 目标：选择真实 `GEMM_A13W10_I64_V1` node；修改一个 output element/tile；从错误 tensor 继续重算所有 downstream nodes；重新生成 self-consistent ManifestV2 / final output / valid worker signature
  - 依赖：A3-02｜DoD：不是简单 manifest mismatch，而是真正 self-consistent wrong computation
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `35b1dd5` · tools/f5c_real_fraud.py real-GEMM artifact; node 145 validated as propagating to the final output
- **A6-04 · Real GEMM fraud E2E**
  - 目标：Worker 提交 bad-but-self-consistent bundle；DA 正常 attest；Watcher root phase PASS；Freivalds detects；定位 bad tile；Open Graph challenge；Graph V2 bisection；Graph→Wide bridge；K-trace bisection；ArbitrateWide512；settlement
  - 依赖：A6-03,A2-07,A3-07,A4-07,A6-01｜DoD：完整 512-MAC fraud proof 在真实图上闭环
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `d81fab0` · docs/phase-f5c-e2e-gemm-fraud.json: first divergent node 145, wide step 0, 512-MAC, ChallengerWins, 0 VWR, four validators converged
- **A6-05 · Real ROPE fraud artifact generator**
  - 目标：选择真实 ROPE_FIXED_V1 node；修改 output；从错误 tensor 继续重算 downstream；构造 self-consistent bad bundle/commit
  - 依赖：A3-02｜DoD：只留下数学 fraud
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `35b1dd5` · tools/f5c_real_fraud.py real-ROPE artifact; node 85 with delta 100000
- **A6-06 · Real ROPE fraud E2E**
  - 目标：Watcher exact cheap-op detects；Graph V2 bisection 定位 ROPE node；bounded RoPE arbiter；settlement
  - 依赖：A6-05,A1-07,A6-01｜DoD：Phase F 原 ROPE devnet 缺口关闭
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `5bd1f7a` · docs/phase-f5c-e2e-rope-fraud.json: node 85 localized, bounded table-pinned arbitration, 0 VWR
- **A6-07 · False challenge E2E**
  - 目标：honest worker result；恶意 challenger 提交不同 final/trail；走完整 challenge path；WorkerWins；worker 继续 finalization
  - 依赖：A6-02,A1-07｜DoD：spam challenge 不破坏 honest settlement
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `df8232d` · docs/phase-f5c-e2e-false-challenge.json: WorkerWins, bond burned, exactly one VWR, second finalize refused
- **A6-08 · Combined adversarial E2E**
  - 目标：real GEMM fraud；DA provider A offline；validator-a proposer 对 challenge omission 1 block；next honest proposer include；继续到 512-MAC / ChallengerWins
  - 依赖：A6-04,A4-07｜DoD：permissionless path 在单 proposer censorship + DA offline 下闭环
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `290d0ac`+`5483e02`+`d81fab0` · docs/phase-f5c-e2e-combined.json: dev censorship harness extended to the V2 family; per-block omission window observed; dispute completes
- **A6-09 · Wide dispute restart E2E**
  - 目标：在 wide midpoint 后停止相关进程/节点；从 persisted state restart；继续 arbitration
  - 依赖：A2-08,A6-04｜DoD：dispute state crash-safe
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `52519b3` · docs/phase-f5c-e2e-wide-restart.json: persisted dispute byte-identical after a full-stack restart; same verdict
- **A6-10 · Validator offline/recovery E2E**
  - 目标：停止 1 validator；其余 3 继续出块并处理 V2 tx；恢复 validator 并 catch up
  - 依赖：A6-01｜DoD：Phase E liveness regression 保持
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `15fa0ac` · docs/phase-f5c-e2e-validator-restart.json (+ …-validator-recovery.json): 3/4 liveness, same height + app hash after recovery
- **A6-11 · Availability failure E2E**
  - 目标：让 DA quorum <2；推进 expiry；task availability_failed；refund
  - 依赖：A4-06,A6-01｜DoD：availability failure 不被误判 computation fraud
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `45ad9f8` · docs/phase-f5c-e2e-availability-failure.json: quorum loss -> availability_failed, refund, 0 VWR

### A7 — Final Docs / Freeze

- **A7-01 · 完成 Canonical Tensor V2 spec**
  - 目标：写 dtype ranges / encoding / chunking / root domains / proof rules / rejection rules；包含 A13/W10/INT64_ACCUM edge cases
  - 依赖：A2-06,A4-04｜DoD：第三方不看代码也能实现 typed tensor commitment
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `3816b09` · docs/canonical-tensor-v2.md
- **A7-02 · 完成 Canonical Graph V2 spec**
  - 目标：Graph descriptor / GraphIDV2 / state/trail roots / ManifestV2 / CommitV3 / ReceiptV3 / dispute path；明确 V1 compatibility
  - 依赖：A1-08,A2-07｜DoD：第三方可独立实现 Graph V2
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `3816b09` · docs/canonical-graph-v2.md
- **A7-03 · 完成 Wide GEMM spec**
  - 目标：A13/W10/A13 right operand semantics；transpose_b；MaxSafeK64；work accounting；tile/trace/512-MAC dispute；明确 GPU decomposition 非 protocol
  - 依赖：A2-06｜DoD：数学定义完全 backend-independent
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `3816b09` · docs/wide-gemm-a13w10-v1.md
- **A7-04 · 完成 Wide Requant spec**
  - 目标：int64 input、mult、shift、ties-to-even、clamp、overflow admission；列出真实 profile usage
  - 依赖：A0-02｜DoD：Go/Python 实现可由文档独立复现
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `3816b09` · docs/wide-requant-v1.md
- **A7-05 · 完成 Freivalds spec**
  - 目标：40 rounds；post-commit randomness；A13×W10 / A13×A13 / transpose_b；detection-only security boundary；union-bound reporting
  - 依赖：A3-05｜DoD：不会被误读成概率 slashing
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `3816b09` · docs/freivalds-a13w10-v1.md (detection-only wording enforced)
- **A7-06 · 生成 phase-f5c-report**
  - 目标：汇总协议、Watcher、DA、Gas、E2E、兼容性；所有关键数字从 JSON 读取；明确 CPU protocol E2E 与 F.5B.2 GPU evidence 是两条证据链
  - 依赖：A6-11,A5-07｜DoD：Q1..Qn 只给可证实结论
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `3816b09`+`89547fa` · docs/phase-f5c-report.md (final: PASS with every A6 evidence artifact)
- **A7-07 · 生成 phase-f-final-report**
  - 目标：完整 lineage：F / F.1 / F.2A / F.3A / F.4A / F.5A / F.5B / F.5B.1 / F.5B.2 / F.5C；保留 W8A8/groupwise/A-only/int32-frontier 的失败研究；列出最终 GraphIDV2 / PolicyID / protocol versions / E2E evidence
  - 依赖：A7-01..A7-06｜DoD：只有所有预声明 gates PASS 才写 `PHASE_F = PASS`
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `89547fa` · docs/phase-f-final-report.md (Q1-Q15 all YES; PHASE_F = PASS)
- **A7-08 · Phase F freeze tag**
  - 目标：确认全测试绿；确认 reports / JSON committed；创建 freeze tag（遵循 repo naming convention）；记录 tag→commit→GraphIDV2→PolicyID mapping
  - 依赖：A7-07｜DoD：可从 tag 完整重建协议与测试证据
  - Status: `DONE`｜Phase: `FROZEN`｜Frozen: `YES`｜Evidence: `89547fa` · tag transformer-phase-f-wide-integer (object 330d27f5… -> commit 89547fa9…); docs/phase-f-freeze.json; fresh-checkout regression green

### Phase A post-E2E protocol fixes (inside the freeze tag; lineage notes)

- `c4fa56a` **Bug 1** — V2 keeper: promote an already-converged trail dispute
  to `arb_ready` (single-node graphs) + chain regression test.
  Belongs to the A1 lineage (dispute state machine wiring).
- `dcb4ebc` **Bug 2** — wide arbitration: resolve operand descriptors by
  `ref.Kind` instead of indexing `graph.Inputs` with a node index
  (panic `index out of range [144] with length 58` on the real block) +
  chain regression test reproducing the exact panic on pre-fix code.
  Belongs to the A2 lineage (WIDE_GEMM_DISPUTE_V1 chain path).
- Both fixes are ancestors of the freeze commit and are covered by the
  fresh-checkout regression run recorded in `docs/phase-f-freeze.json`.

### Phase A task count (mechanical)

```text
A0 2  A1 8  A2 8  A3 10  A4 7  A5 7  A6 11  A7 8   = 61 tasks, all DONE/FROZEN
```

---

# Phase B — Testnet Productization (ACTIVE)

Phase B starts only because `PHASE_F = PASS`.  These tasks are engineering
productization of the frozen protocol — **no protocol redesign**.

## B0 — Productization Transition (NEW in v1.1)

**Goal:** move from the protocol-research repository state to a clean,
traceable, freeze-tag-based Testnet Productization baseline.

Rationale: `main` does not yet contain the frozen lineage (see §0), the
main working tree carries uncommitted public-client WIP, and product work
must start from a known-good, tagged protocol ancestry — otherwise
protocol and product history will diverge silently.

---

### B0-01 — Repository Reconciliation

```text
Task ID            B0-01
Status             DONE (2026-10-05)
Priority           P0
Evidence           docs/productization/b0-repository-reconciliation.md (product line; original
                   a241af9); pre-existing WIP parked on feature/public-node-admission @ 520abb1;
                   working tree clean; nothing deleted; roadmap bookkeeping recorded here by B0-03
Goal               inventory, classify and reconcile every pre-existing
                   working-tree change so the productization baseline
                   starts from a clean, fully-understood tree
Dependencies       PHASE_F = PASS (satisfied)
Allowed changes    docs/productization/** ; git branch/commit mechanics ;
                   moving pre-existing WIP into a dedicated transition
                   commit or branch
Forbidden changes  product code, protocol code, chain logic; deleting
                   anything without a recorded reason; `git add .`
```

**Implementation**

- **B0-01a Inventory** — record mechanically, from the repository itself
  (never from this roadmap's prose):
  ```text
  git status
  git diff
  git ls-files --others --exclude-standard
  current branch + base commit
  ```
  Known starting point (2026-10-05, main worktree, branch
  `protocol/gemm-v0.1.2-freivalds`, HEAD `e550cbb`) — re-verify, do not
  trust:
  ```text
  M  network/README.md
  M  network/prisma_network/core.py
  M  network/prisma_network/server.py
  ?? LICENSE
  ?? Prisma_Chain_AI_Engineering_Roadmap_F5C_to_Public_Testnet_Alpha.md   (v1.0 historical roadmap)
  ?? network/prisma_network/node_http.py
  ?? network/tests/test_public_nodes.py
  ```
- **B0-01b Classify each file**: `KEEP` / `MIGRATE` / `OBSOLETE` /
  `UNRELATED` / `NEEDS_REVIEW`.
- **B0-01c Conflict check**: for each item decide whether it conflicts
  with the Phase B roadmap, the frozen protocol, or the current public
  network architecture.  Record the verdict per file.
- **B0-01d** Never `git add .` — stage file-by-file, path-by-path.
- **B0-01e** Valuable code moves into a dedicated transition commit (or
  branch) with a message explaining what it is and where it belongs.
- **B0-01f** Obsolete code is never silently deleted; record why it is
  obsolete in the reconciliation report.
- **B0-01g** End state: working tree clean.

**Tests / validation**

```text
every pre-existing modification classified
no accidental overwrite
no unrelated WIP lost
working tree clean
```

**Artifacts**

```text
docs/productization/b0-repository-reconciliation.md   (classification table + reasons)
```

**Definition of Done** — all four validation lines above true; the
reconciliation report committed; the planned transition commit(s)
created; nothing deleted without a written reason.

**Failure / Blocked behavior** — if a file's classification is genuinely
ambiguous, mark `NEEDS_REVIEW`, keep the tree clean around it (stash or
transition commit with the item explicitly listed), and surface the
question in the task report instead of guessing.

**Stop Rule** — do not start product code while any tree entry is
unclassified.

**Next unlocked tasks** — B0-02.

---

### B0-02 — Productization Baseline

```text
Task ID            B0-02
Status             DONE (2026-10-05)
Evidence           product/testnet-alpha based at 89547fa (freeze tag commit); baseline artifact
                   docs/productization/productization-baseline.json (commit 10a930f on the
                   product line); 6 post-tag commits carried with recorded reasons; regression
                   green at the baseline commit (root+chain go test, builds, GraphIDV2/PolicyID
                   recompute) and at the tip (network suite 31/31); main NOT merged (recorded)
Priority           P0
Goal               create the formal productization branch, based on the
                   FROZEN PHASE F TAG as its protocol ancestry
Dependencies       B0-01
Allowed changes    branch creation; docs/productization/**; cherry-pick /
                   merge of post-tag docs/tests/non-protocol fixes with
                   recorded reasons
Forbidden changes  protocol semantics; rewriting protocol history
```

**Implementation**

- Branch name: check the repository convention first.  Existing
  two-level prefixes are `protocol/*` and `research/*`; the proposed new
  line is `product/*`.  Proposed name: `product/testnet-alpha` — adopt it
  unless a conflicting convention is found, and record the decision.
- **Base = the freeze tag** `transformer-phase-f-wide-integer`
  (`89547fa`), not `origin/main` and not the long-lived protocol branch
  head.  Then explicitly decide what post-tag material enters:
  ```text
  post-tag commits on protocol/transformer-phase-f5c-wide-integer = 1
    (40b4fe3: docs/phase-f-freeze.json mapping record — docs only)
  post-tag docs/tests/non-protocol bugfixes: enumerate, cherry-pick or
    merge each with a recorded reason
  ```
- The two E2E-found protocol fixes are already inside the tag
  (`c4fa56a`, `dcb4ebc` are ancestors of `89547fa`) — no special handling
  needed; state this explicitly so nobody "re-fixes" them.
- Note for follow-up (do not perform here): reconciling `main` with the
  frozen lineage (fast-forward/merge of the f5c branch to `40b4fe3` or
  the tag) is a repository-level decision to record in the
  reconciliation report.

**Artifacts**

```text
docs/productization/productization-baseline.json   (fields below)
```

```json
{
  "freeze_tag": "transformer-phase-f-wide-integer",
  "freeze_commit": "89547fa97b62d72a4cd336a5dfd3b000b03eac60",
  "source_branch": "protocol/transformer-phase-f5c-wide-integer",
  "product_branch": "product/testnet-alpha",
  "product_baseline_sha": "<the branch's first commit or its base commit>",
  "graph_id_v2": "8fb86087697e277602bb4b2088ba48eb4172c21d9c4f83914582f2cbf00d4def",
  "policy_id": "eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0",
  "protocol_versions": { "…": "copied from docs/phase-f-freeze.json, not retyped" },
  "legacy_compatibility_status": "V1 golden suite green (verify at baseline commit)"
}
```

**Tests** — at the product baseline commit: root `go test ./...`,
`cd chain && go test ./...`, the GraphIDV2 recomputation, and the legacy
golden suite — all green.  (This repeats the fresh-checkout regression on
the very commit the product line starts from.)

**Definition of Done** — the product branch exists with the tag as
ancestry; the baseline JSON is committed on it; the regression above is
green at that commit.

**Failure / Blocked behavior** — if the regression is not green at the
proposed baseline, stop: the baseline is wrong, not the tests.

**Stop Rule** — do not begin B1 work on an untagged or unverified base.

**Next unlocked tasks** — B0-03, B0-04, B0-05.

---

### B0-03 — Roadmap State Migration

```text
Task ID            B0-03
Status             DONE (2026-10-05, this bookkeeping commit)
Priority           P0
Evidence           roadmap v1.1 updated in place on the product line: §0 metadata (active_task,
                   product_branch, product_baseline_sha), §11 execution queue, B0-01/B0-02
                   status+evidence recorded; migration JSON present with real counts;
                   §B0-03 checks below verified
Goal               make roadmap v1.1 the single authoritative plan and
                   close the v1.0 state (PARTIAL-era) bookkeeping
Dependencies       — (this rewrite is the main artifact; the bookkeeping
                   completes after B0-01/B0-02 so it can record their result)
Allowed changes    docs/roadmap/**, docs/roadmap-v1.1-migration.json
Forbidden changes  everything else
```

**Implementation**

- This document (v1.1) is the primary artifact: `PHASE_F = PASS /
  FROZEN`, `PHASE_B = ACTIVE`, active entrypoint `B0-01` at rewrite
  time — updated to `B0-04` by the B0-03 bookkeeping as tasks complete.
- Update the v1.0-era progress notes that still describe a PARTIAL
  state *as current* (historical references may remain). — DONE: the
  only remaining `PARTIAL` strings in this file are (a) the §"not a
  PARTIAL verdict" rule, (b) B0-03's own historical description, and
  (c) the §Appendix C checklist item that asserts the property; the
  v1.0 copy carries the superseded banner.
- Record the migration counts in `docs/roadmap-v1.1-migration.json`
  (machine-generated, see §Appendix C).

**Definition of Done** — v1.1 exists at its path; the migration JSON
exists with real counts; no document claims Phase F is unfinished
outside of explicitly historical text.

**Next unlocked tasks** — B0-04, B0-05 (B1-01 unlocks after B0-02).

---

### B0-04 — Development Environment Cleanup

```text
Task ID            B0-04
Status             DONE (2026-10-05)
Priority           P0 (cheap, do before B1 so the baseline is honest)
Evidence           docs/productization/b0-04-environment-cleanup.md; devnet stopped via
                   deploy/multivalidator/stop.sh after recording chain_id prisma-mv-1 height
                   2120; containers+network removed, ports 26661-26664 released, data
                   preserved; nothing referenced by reports deleted
Goal               bring the development environment to a known, minimal,
                   documented state before productization
Dependencies       B0-01 (so cleanup cannot destroy unclassified WIP)
Allowed changes    local environment only: stopping containers, removing
                   temporary volumes/ports/processes, deleting temporary
                   generated binaries; record every action
Forbidden changes  deleting protocol evidence (evidence JSON, logs
                   referenced by reports, freeze artifacts, testdata)
```

**Implementation**

- Inventory the environment:
  ```text
  local 4-validator compose stack (prisma-multivalidator)
  temporary devnet volumes / ports 26655-26664, 9090+
  old/stale processes
  RunPod GPU pods (must be stopped by the user; no API key exists)
  temporary generated binaries (e.g. build outputs, scratch clones)
  ```
- Re-query live state at execution time (do not hardcode, e.g. the
  2026-10-05 devnet was at height ≈1550, 4 validators converged,
  censorship disabled — that number is not a fact to reuse).
- Stop policy: if the local stack is not needed, run
  `deploy/multivalidator/stop.sh` (or the real equivalent) and record it
  in the cleanup report.
- Keep: evidence JSONs, result logs referenced by reports, freeze
  artifacts, `testdata/`.

**Artifacts** — cleanup section appended to
`docs/productization/b0-repository-reconciliation.md` (or its own
`b0-04.md`).

**Definition of Done** — environment state documented; nothing
referenced by committed reports/evidence deleted; the retained state
(what runs, where) written down for the next session.

**Next unlocked tasks** — B0-05.

---

### B0-05 — Productization Toolchain Baseline

```text
Task ID            B0-05
Status             PLANNED
Priority           P0
Goal               pin the toolchain and target matrix that Phase B
                   components are built and tested against
Dependencies       B0-02
Allowed changes    docs/productization/**
Forbidden changes  silently adding platform support that does not exist
```

**Implementation** — record from the real repository/CI, not from memory:

```text
Go version            (the chain is a separate module; record the exact toolchain used, e.g. Go 1.24)
Python version        (worker/watcher/DA/tooling)
Docker version        (compose stack + image builds)
CUDA expectations     for Worker (driver + runtime + compute capability floor; SM86/SM89 are the proven targets)
supported OS          honestly: what is actually tested today
architecture          first release target: Linux amd64
arm64                 only if the repo already supports it — do not invent support
```

**Definition of Done** — the matrix is documented; every entry cites
where it is enforced or tested; unsupported combinations are explicitly
listed rather than implied.

**Next unlocked tasks** — B1-01 (after B0-02).

---

## B1 — Core Distribution

### B1-01 — prismad reproducible release build

```text
Task ID            B1-01
Status             PLANNED (first B1 task; unlock after B0-02/B0-03)
Priority           P0
Goal               reproducible Linux binaries + version metadata +
                   container + healthcheck for the chain node
Dependencies       PHASE_F=PASS; B0-02 (baseline branch)
Allowed changes    build scripts, version package, Dockerfile, docs/productization/**
Forbidden changes  chain logic; protocol semantics; silent fallbacks
```

**Implementation**

- **B1-01a Decide metadata policy**
  ```text
  binary name                 (prismad)
  module version              (source of truth: repo state, e.g. git describe)
  git SHA embedding           (real, from the build)
  protocol version embedding  (from the frozen protocol; not hand-typed)
  build date policy           (must NOT break reproducibility: no wall-clock
                               timestamp and no random build path may enter
                               the binary hash — use SOURCE_DATE_EPOCH-style
                               deterministic stamping or omit the date)
  ```
- **B1-01b Implement `prismad version`** printing at least:
  ```text
  software version
  git commit
  freeze protocol version
  Graph protocol support (V2 / GraphIDV2 support flag)
  build target (OS/arch)
  ```
- **B1-01c** clean build #1 into an empty output dir, fixed flags
  (`-trimpath`, no buildid randomness, CGO as decided).
- **B1-01d** clean build #2 into a different empty dir.
- **B1-01e** SHA256 compare of the two binaries — must be identical.
- **B1-01f** Docker build (multi-stage, pinned base images, no secrets in
  any layer).
- **B1-01g** container healthcheck (real endpoint, bounded timeout,
  meaningful exit codes).
- **B1-01h** clean-machine smoke: on a machine without the dev toolchain,
  initialize a node from the release artifacts and start it.

**Tests**

```text
unit tests for the version package
two-build reproducibility check (B1-01e) as a scripted check
container healthcheck test
fresh-machine smoke (documented commands + observed output)
```

**Artifacts**

```text
docs/productization/b1-01.md   (policy, commands, hashes, smoke log)
```

**Definition of Done**

```text
two clean builds identical (identical SHA256)
binary starts; node initializes
healthcheck works
version metadata correct (git SHA / protocol version / target)
container starts without secrets
fresh machine smoke PASS
```

**Failure / Blocked behavior** — a non-reproducible build is a failure
to fix, not to document around.  Errors must be explicit; no silent
fallback path may be added to make a smoke test pass.

**Stop Rule** — do not proceed to publish anything (Phase D / M6 rules)
from a non-reproducible build.

**Next unlocked tasks** — B1-03.

---

### B1-03 — Genesis / config generator

```text
Task ID            B1-03
Status             PLANNED
Priority           P0
Goal               generate chain-id, genesis, validator seeds, RPC/DA
                   defaults, window parameters and faucet params with a
                   checksum manifest and validation
Dependencies       B1-01
Allowed changes    genesis/config tooling; docs/productization/**
Forbidden changes  localhost devnet assumptions leaking into public
                   testnet config; hand-edited genesis
```

**Implementation** — atomic steps:

```text
B1-03a chain-id            deterministic rule, recorded
B1-03b genesis             generated from the frozen module set; module
                           params derived, not copied from the devnet
B1-03c validator seeds     generation + recording (public keys only in artifacts)
B1-03d RPC defaults        sane public defaults (no 0.0.0.0 wildcard by accident)
B1-03e DA defaults         DAWindowBlocks, quorum, penalty parameters
B1-03f challenge windows   challenge/challenge-round/claim windows consistent
                           with the frozen constants
B1-03g faucet test params  amounts, cooldowns (test tokens only)
B1-03h checksum manifest   every generated artifact hashed into one manifest
B1-03i config validation   a validator that can load and verify the whole bundle
```

**Tests** — generator unit tests; regenerate-twice byte-identity;
validation rejects a tampered artifact (checksum mismatch must be a hard
error).

**Artifacts** — `docs/productization/b1-03.md` + the generated
config bundle in a documented location.

**Definition of Done** — one command produces a complete, checksummed,
validated network config; no devnet-only value survives into it.

**Next unlocked tasks** — B1-02.

---

### B1-02 — Public RPC / sentry mode

```text
Task ID            B1-02
Status             PLANNED
Priority           P0
Goal               a non-validator public RPC profile with limits,
                   sentry topology and observability
Dependencies       B1-03
Allowed changes    node config profiles, proxy middleware, docs
Forbidden changes  validator key material on public nodes; consensus
                   behavior changes
```

**Implementation**

```text
B1-02a RPC-only node mode        no validator key, no signing, read/write proxy policy
B1-02b validator peer segregation private validator <-> sentry topology only
B1-02c request size limits       hard caps with explicit errors
B1-02d rate limiting             per-IP and per-method budgets
B1-02e query timeouts            bounded, non-hanging
B1-02f metrics                   Prometheus-style counters/latency
B1-02g health endpoint           separate from full RPC surface
B1-02h sentry topology docs      diagram + concrete config example
B1-02i abuse/load smoke          scripted load + abuse case (oversize, flood)
```

**Definition of Done** — a public RPC/sentry pair can be stood up from
the generated config; limits enforced with explicit errors; validators
never expose key material; the load/abuse smoke has recorded results.

**Next unlocked tasks** — B2-01 (worker line) and B4 (DA line).
---

## B2 — Worker (prisma-worker)

### B2-01 — Worker identity / keystore

```text
Task ID            B2-01
Status             PLANNED
Priority           P0
Goal               local identity: chain account + protocol key, safely stored
Dependencies       B1-01
Allowed changes    worker/** ; docs/productization/**
Forbidden changes  baking private keys into images; keys in logs
```

**Implementation**

```text
B2-01a generate protocol key      (ed25519, local)
B2-01b import protocol key        (hex file / existing operator key)
B2-01c chain account              (address derivation + record)
B2-01d encrypted local storage    (passphrase-protected keystore; 0600 perms)
B2-01e permissions                (refuse to run with world-readable key files)
B2-01f no secrets in logs         (audit + test: no key material in any log path)
B2-01g key rotation               (documented procedure, new keys announced to
                                    the chain before old ones retire)
B2-01h recovery behavior          (lost passphrase -> explicit unrecoverable error;
                                    lost local file -> re-import procedure)
```

**Tests** — unit tests for each sub-step; a log-scrubbing test that fails
if any secret pattern appears in emitted logs.

**Definition of Done** — a fresh machine can generate/import keys,
register on testnet, and nothing secret ever appears in logs or images.

**Next unlocked tasks** — B2-02.

---

### B2-02 — GPU capability probe

```text
Task ID            B2-02
Status             PLANNED
Priority           P0
Goal               detect, report and gate on real GPU capability
Dependencies       B2-01
Allowed changes    worker/**
Forbidden changes  silent CPU fallback when the GPU is unsupported
```

**Implementation** — probe and report:

```text
GPU name
compute capability
VRAM
driver version
CUDA runtime version
backend compatibility   (does TRUE_FUSED_MMA_A13W10 support this device?)
profile compatibility   (does the device meet the frozen profile's requirements?)
```

Behavior rules:

```text
unsupported GPU -> explicit CAPABILITY_UNSUPPORTED state, refuse work;
                   NEVER a silent CPU fallback
```

**Tests** — probe unit tests with canned device descriptions (including
an unsupported device); refusal path exercised.

**Definition of Done** — the worker reports its capability truthfully and
refuses incompatible work with a typed error.

**Next unlocked tasks** — B2-03.

---

### B2-03 — Worker join / bond / register

```text
Task ID            B2-03
Status             PLANNED
Priority           P0
Goal               join the network: connect, validate, bond, register, heartbeat
Dependencies       B2-01, B2-02
Allowed changes    worker/**
Forbidden changes  bypassing bond/registration; hardcoded chain-ids
```

**Implementation**

```text
RPC connection               configurable endpoints; fail over with explicit logs
chain-id validation          refuse to talk to the wrong chain
protocol version check       refuse a chain that does not support the frozen protocol
balance                      query + display; floor for bond operations
faucet/test balance          testnet faucet integration (B7-01)
bond                         bond transaction + confirmation handling
register                     worker registration incl. network key binding
heartbeat                    periodic liveness + capability refresh
capability update            re-register when GPU/profile changes
```

**Tests** — integration test against a local chain; wrong chain-id
refused; insufficient balance gives an explicit actionable error.

**Definition of Done** — a fresh worker goes from config to bonded,
registered and heart-beating with no manual chain surgery.

**Next unlocked tasks** — B2-04.

---

### B2-04 — Worker job fetch / accept

```text
Task ID            B2-04
Status             PLANNED
Priority           P0
Goal               discover compatible jobs, accept atomically, survive restarts
Dependencies       B2-03
Allowed changes    worker/**
Forbidden changes  accepting incompatible jobs (profile/GPU mismatch)
```

**Implementation**

```text
job discovery               chain/event or API source (per B6 design)
compatibility filter        profile + capability + policy match
accept                      transaction + error handling
assignment ref              capture and persist the assignment reference
journal write               local write-ahead journal entry BEFORE accept returns
timeout                     accept deadline handling
duplicate assignment        idempotent re-accept semantics
```

**Definition of Done** — the worker accepts only compatible jobs; every
accept is journaled before side effects; duplicate/late acceptance cannot
double-execute.

**Next unlocked tasks** — B2-05.

---

### B2-05 — Worker model / profile manager

```text
Task ID            B2-05
Status             PLANNED
Priority           P0
Goal               manage the frozen model/profile artifacts the worker
                   executes against
Dependencies       B2-04
Allowed changes    worker/**
Forbidden changes  executing an unverified or mismatched profile
```

**Implementation**

```text
model manifest              frozen manifest file (public, versioned)
checkpoint hash             verify every byte against the manifest
profile                     frozen profile binding
PolicyID                    verify against the descriptor (eb9a9fef…)
GraphID/profile support     verify the graph descriptor matches the frozen GraphIDV2
download                    resumable download
resume                      interrupted transfer continues
cache                       content-addressed local cache; multiple versions side-by-side
corruption detection        hash mismatch -> quarantine + explicit error, never execute
```

**Definition of Done** — the worker can only execute artifacts that hash
to the frozen manifest; corruption is detected before any execution.

**Next unlocked tasks** — B2-06.

---

### B2-06 — Worker execution backend

```text
Task ID            B2-06
Status             PLANNED
Priority           P0
Goal               production execution backend: TRUE_FUSED_MMA_A13W10
Dependencies       B2-05
Allowed changes    worker/**
Forbidden changes  any execution path that is not bit-exact vs the frozen
                   reference; hidden float paths
```

**Note (must be stated in product docs):** the GPU backend is **not a
consensus identity**.  Consensus is over canonical tensor roots and the
frozen arithmetic; the GPU is an execution accelerator whose outputs must
match the frozen reference bit-for-bit.  A worker on a different (proven)
device must produce identical roots.

**Implementation**

```text
load                 frozen program/bundle + weights into the backend
prepack              prepack weights per the frozen packing rules
execute              full block execution (all operators, fixed reduction orders)
metrics              per-node timing, wall time, memory high-water
OOM                  explicit out-of-memory handling (fail the task, keep the worker alive)
CUDA error           explicit device-error handling with device reset policy
unsupported shape    typed refusal (never approximate)
deterministic output bit-exact vs the frozen reference (self-check hooks)
```

**Tests** — run the frozen operator/vector gates and the 310-node block
self-check on every supported compute capability; bit-exactness is the
gate, not tolerance.

**Definition of Done** — the worker executes the frozen block
bit-exactly, with explicit failure modes for OOM/CUDA/unsupported cases.

**Next unlocked tasks** — B2-07.

---

### B2-07 — Worker CommitV3 builder

```text
Task ID            B2-07
Status             PLANNED
Priority           P0
Goal               build and sign GraphResultCommitV3 exactly as the
                   frozen protocol expects
Dependencies       B2-06
Allowed changes    worker/**
Forbidden changes  any preimage/serialization deviation from the frozen
                   construction (signature domain, field clearing, ordering)
```

**Implementation**

```text
TensorRootV2                compute roots exactly (canonical encoding, BE16/BE64 chunks)
ManifestV2                  NodeOutputManifestV2 over all delivered outputs
GraphResultCommitV3         build the canonical object (signature cleared to b"")
signature                   ed25519 over the frozen signing preimage
task binding                bind graph_task_id / task_ref
assignment binding          bind the assignment reference
duplicate-submit protection refuse to submit twice for one assignment
```

**Tests** — cross-language verification: the worker's produced commit is
verified by an independent verifier (the repo already contains the Go/Py
verifiers used in F.5C); a tampered field must fail verification.

**Definition of Done** — commits produced by the worker verify under the
independent cross-language verifier; duplicate submission impossible.

**Next unlocked tasks** — B2-08.

---

### B2-08 — Worker DA upload

```text
Task ID            B2-08
Status             PLANNED
Priority           P0
Goal               get the verification bundle into ≥2 DA replicas and
                   prove it before finalization
Dependencies       B2-07, B4-01 (DA surface exists)
Allowed changes    worker/**
Forbidden changes  skipping quorum; treating upload success as storage
                   success without verified attestations
```

**Implementation**

```text
bundle build            GraphVerificationBundleV2 for the committed work
provider discovery      find registered, active providers
upload provider A/B/C   upload to each (protocol per B4 design)
attestation verification verify each provider's typed attestation
2-of-3 quorum           require the configured quorum before declaring the
                        result available
retry                   bounded retries with backoff
provider timeout        per-provider deadlines
submit DA refs          where the chain requires references, submit them
```

**Tests** — provider-down drills (1 of 3 offline, 2 of 3 offline);
timeout handling; quorum never bypassed.

**Definition of Done** — a result is only declared available when the
quorum of verified attestations exists; the 2-of-3 path is proven under
provider failure.

**Next unlocked tasks** — B2-09.

---

### B2-09 — Worker crash recovery

```text
Task ID            B2-09
Status             PLANNED
Priority           P0
Goal               the worker can crash at any point and recover deterministically
Dependencies       B2-08
Allowed changes    worker/**
Forbidden changes  resuming into an inconsistent chain view
```

**Implementation** — crash points that must be covered:

```text
before accept
after accept
mid execution
after execution
before CommitV3
after CommitV3
before DA quorum
after DA quorum
before finalization
```

For each state, classify and document:

```text
retryable                   can redo locally
non-retryable               must abandon the task
requires chain reconciliation reconcile local state against the chain first
```

**Tests** — fault-injection: kill the worker at each point, restart,
verify the classification and that no double-commit / double-charge
happens.

**Definition of Done** — every crash point has a recorded, tested
behavior; no state transition can double-submit or lose funds.

**Next unlocked tasks** — B4-01 (DA line) follows for the integration
drill; B3 after B4.

---

## B4 — DA (prisma-da)

### B4-01 — prisma-da daemon packaging

```text
Task ID            B4-01
Status             PLANNED
Priority           P0 (before B3 — see §11.1 rationale)
Goal               a runnable DA provider daemon
Dependencies       B1-01 (release tooling); the frozen DA protocol from F.5C
Allowed changes    da/**
Forbidden changes  changing the frozen attestation/challenge format
```

**Implementation**

```text
identity             provider key (same rules as B2-01)
bond/register        registration flow against the chain
HTTP/API             artifact upload + retrieval surface
health               health endpoint
metrics              counters/latency/size
graceful shutdown    finish in-flight writes, then exit
restart recovery     re-scan local index; resume serving
```

**Definition of Done** — a DA daemon can be installed, registered, serve
uploads/retrievals and survive restarts; the surface the worker needs
(B2-08) and the watcher needs (B3) is stable and documented.

**Next unlocked tasks** — B4-02.

---

### B4-02 — DA storage engine

```text
Task ID            B4-02
Status             PLANNED
Priority           P0
Goal               durable, verifiable artifact storage
Dependencies       B4-01
Allowed changes    da/**
Forbidden changes  serving bytes that fail their content hash
```

**Implementation**

```text
artifact index        (artifact hash -> location, size, timestamps)
content hash          verify on write and on read
atomic write          write-temp + fsync + rename; never half artifacts
NVMe backend          the default local backend
optional object store interface  (S3-compatible), same semantics
quota                 per-provider size cap with explicit rejection
TTL                   retention policy per artifact class
garbage collection    safe deletion only after TTL AND no live challenge
integrity scan        periodic full-hash scan; corruption -> quarantine + report
```

**Definition of Done** — every served byte verifies against its content
hash; crash during write never leaves a served corrupt artifact; GC can
never delete a challenge-relevant artifact.

**Next unlocked tasks** — B4-03.

---

### B4-03 — DA challenge responder

```text
Task ID            B4-03
Status             PLANNED
Priority           P0
Goal               answer on-chain typed chunk challenges within deadlines
Dependencies       B4-02
Allowed changes    da/**
Forbidden changes  fabricated proofs; missing deadlines silently
```

**Implementation**

```text
chain challenge subscription  watch for challenges naming this provider
proof lookup                  locate the artifact + chunk under the manifest
typed chunk proof             build exactly the frozen proof format
deadline scheduler            per-challenge timers with safety margin
response tx                   submit the response transaction
after-loss detection          artifact already gone -> report objectively (do not fabricate)
metrics                       response latency, failures, losses
```

**Tests** — drill: challenge a stored artifact (answer in time), challenge
a deliberately removed artifact (objective after-loss behavior recorded).

**Definition of Done** — challenges are answered with verifiable proofs
before deadlines; after-loss is reported honestly (the chain's objective
penalty path applies).

**Next unlocked tasks** — B3-01 (Watcher).

---

## B3 — Watcher (prisma-watcher)

### B3-01 — prisma-watcher daemon packaging

```text
Task ID            B3-01
Status             PLANNED
Priority           P0 (after B4 — retrieval surface must exist first)
Goal               a runnable permissionless watcher daemon
Dependencies       B4-03
Allowed changes    watcher/**
Forbidden changes  probabilistic detection triggering slashing directly
                   (Freivalds is detection-only; the chain adjudicates)
```

**Implementation** — daemon lifecycle + pipeline:

```text
event subscription       new graph results / tasks from the chain
DA discovery             locate the bundle via DA providers
Bundle download          resumable download + artifact hash check
root phase               artifact/version/GraphIDV2/roots/manifest/final checks
Freivalds                the frozen 40-round verifier (detection only)
cheap ops                exact recomputation of cheap nodes
fraud localization       residual -> row/column -> 8x8 tile / node
challenge tx             open the graph challenge
trail                    claim trails at midpoints
midpoint                 participate in bisection
wide dispute             participate in the wide path when applicable
deadline scheduler       all dispute clocks
receipt observation      observe settlement / VWR outcome
persistence              journal everything needed to resume
```

**Definition of Done** — the daemon runs unattended on the real bundle,
detects the known fraud artifacts, and drives a complete deterministic
dispute to settlement.

**Next unlocked tasks** — B3-02.

---

### B3-02 — Watcher automatic challenge

```text
Task ID            B3-02
Status             PLANNED
Priority           P0
Goal               automatic, correctly-bonded, correctly-timed challenges
Dependencies       B3-01
Allowed changes    watcher/**
Forbidden changes  challenging without evidence; challenging after the window
```

**Implementation**

```text
decision policy      (what triggers a challenge; threshold wording tied to
                      detection results, never to probability of guilt)
bond management      (fund the challenge; never spend into unfunded state)
window tracking      (open challenge inside the challenge window)
dispute driving      (trail claim, midpoints, wide dispute steps, deadlines)
outcome handling     (ChallengerWins / WorkerWins / BothInvalid / timeout)
on-chain evidence    (record the transcript digest for audit)
```

**Definition of Done** — the watcher challenges the injected frauds
automatically and never opens a challenge it cannot evidence.

**Next unlocked tasks** — B3-03.

---

### B3-03 — Watcher restart recovery

```text
Task ID            B3-03
Status             PLANNED
Priority           P0
Goal               watcher restarts never lose a dispute it is running
Dependencies       B3-02
Allowed changes    watcher/**
Forbidden changes  exiting mid-dispute without persisting state
```

**Implementation** — persist: active disputes, locked trail roots, midpoint
history, challenge bonds, deadlines; on restart: re-derive chain state and
resume exactly.

**Tests** — kill the watcher at each dispute phase (claim locked, mid-round,
wide trace, arbitration pending), restart, verify it resumes with the same
verdict.

**Definition of Done** — restart equivalence proven across all dispute
phases (mirrors the F.5C A6-09 requirement, now in product code).

**Next unlocked tasks** — B5-01.

---

## B5 — CLI (prisma-cli)

### B5-01 — prisma-cli packaging

```text
Task ID            B5-01
Status             PLANNED
Priority           P0 (core)
Goal               one CLI entrypoint that drives the whole testnet flow
Dependencies       B2 / B3 / B4 exist (command groups unlock as their
                   components land)
Allowed changes    cli/**
Forbidden changes  duplicating protocol logic; drifting from the frozen
                   wire formats
```

**Command surface** (mark ownership honestly: some are chain-tx helpers,
some are API clients — do not pretend everything is a chain tx):

```text
wallet               local key/account management (chain)
balance              account balance (chain/query)
faucet               testnet faucet request (API)
bond                 worker/DA bond operations (chain)
worker register      worker registration (chain)
post job             submit a job (API, B6) — not a raw chain tx
task query           task/job status (API + chain)
receipt query        VWR lookup (chain)
DA query             DA status/artifact info (chain + DA API)
dispute query        dispute status (chain)
run worker           run the worker daemon
run watcher          run the watcher daemon
run da               run the DA daemon
version              version metadata (from B1-01)
config               config management + validation (B1-03)
```

**Definition of Done** — a developer can complete submit→execute→verify→
receipt entirely through the CLI against a private testnet, without
editing source.

**Next unlocked tasks** — B6-01.

---

## B6 — Job API / Scheduler

### B6-01 — Job API schema

```text
Task ID            B6-01
Status             PLANNED
Priority           P0 (core)
Goal               the developer-facing job API
Dependencies       B2-04/B2-07 (worker can consume jobs), B5-01
Allowed changes    api/**
Forbidden changes  claiming support for arbitrary HuggingFace models
```

**Implementation** — v1 endpoints:

```text
POST /v1/jobs           submit (declares a FROZEN SUPPORTED PROFILE only)
GET  /v1/jobs/{id}      status + result references
GET  /v1/jobs           list (pagination)
cancel semantics        explicit: pre-accept cancel vs post-accept (per B6-05)
supported profiles      enumerate the frozen supported profile(s); nothing else
error schema            stable machine-readable errors
idempotency             submit idempotency keys
pagination              cursor-based
```

First version supports **only the frozen supported profile(s)**.  The API
must never claim arbitrary model support.

**Definition of Done** — schema documented; profile gate enforced with
explicit errors; idempotent submission proven by test.

**Next unlocked tasks** — B6-02.

---

### B6-02 — API authentication

```text
Task ID            B6-02
Status             PLANNED
Priority           P0 (core)
Goal               testnet-grade auth: API keys, limits, quotas
Dependencies       B6-01
Allowed changes    api/**
Forbidden changes  building billing/KYC/Stripe in the testnet phase
```

**Implementation**

```text
API key              issue/revoke; hashed at rest
rate limits          per-key request budgets
request quotas       per-key job quotas
explicit 401/403/429 with stable error schema
```

**Definition of Done** — keys can be issued/revoked; limits enforced;
test tokens remain the only economy (no billing).

**Next unlocked tasks** — B6-03.

---

### B6-03 — Scheduler worker registry

```text
Task ID            B6-03
Status             PLANNED
Priority           P0 (core)
Goal               a truthful registry of workers
Dependencies       B6-02, B2-03 (heartbeats)
Allowed changes    scheduler/**
Forbidden changes  inventing reliability scores the data does not support
```

**Implementation** — record:

```text
worker id
chain account
protocol key
GPU capability
supported profiles
heartbeat
availability
current jobs
reliability          v1: RECORD ONLY (raw counts), no composite scoring yet
```

**Definition of Done** — registry state derives only from observed facts;
v1 reliability is raw data, clearly marked as non-authoritative.

**Next unlocked tasks** — B6-04.

---

### B6-04 — Scheduler matching

```text
Task ID            B6-04
Status             PLANNED
Priority           P0 (core)
Goal               simple, predictable matching
Dependencies       B6-03
Allowed changes    scheduler/**
Forbidden changes  market auction / dynamic token pricing / complex bidding
                   in the first version
```

**Implementation**

```text
capability filtering   hard filter on profile + capability
availability           assign only to available workers
FIFO                   default ordering
simple priority        a small, documented tiebreaker set (e.g. priority field)
```

**Definition of Done** — deterministic, explainable assignment; no
speculative economics.

**Next unlocked tasks** — B6-05.

---

### B6-05 — Scheduler failure / reassign

```text
Task ID            B6-05
Status             PLANNED
Priority           P0 (core)
Goal               define who recovers what, at every failure point
Dependencies       B6-04
Allowed changes    scheduler/** + docs
Forbidden changes  silently swapping workers after a commit
```

**Implementation** — failure matrix (each row: recovery owner + action):

```text
pre-accept                   scheduler reassigns
post-accept pre-execution    worker timeout -> scheduler reassigns
execution failure            worker reports failure -> reassign per policy
pre-commit disconnect        journaled resume (B2-09) or reassign
post-commit disconnect       NO silent worker swap; the commit is on chain —
                             the DA/watcher/settlement path takes over
DA failure                   worker retries per B2-08; quorum loss leads to
                             availability_failed settlement (chain-defined)
```

**Definition of Done** — every failure point has a documented owner and
an implemented action; "post-commit" never silently reassigns.

**Next unlocked tasks** — B6-06.

---

### B6-06 — Job / VWR presentation

```text
Task ID            B6-06
Status             PLANNED
Priority           P0 (core)
Goal               expose job status and proof references
Dependencies       B6-05
Allowed changes    api/**
Forbidden changes  fabricating verification status the chain did not confirm
```

**Implementation** — job presentation includes at least:

```text
job_id
task_id
status                (see the job state machine below)
worker
GraphID
final root
verification status   (as reported by chain/verification, never inferred ahead)
settlement tx
VWR id
```

**Job state machine** (fixed by this roadmap; map to real chain states and
document the mapping):

```text
queued
assigned
accepted
executing
committed
availability_ready
verifying
challenged
finalized

failed
availability_failed
```

**Definition of Done** — the mapping table (API state ↔ chain state) is
committed and testable; presentation never claims more than the chain
proves.

**Next unlocked tasks** — B7-01.

---

## B7 — Faucet / Docs / Status

### B7-01 — Faucet service

```text
Task ID            B7-01
Status             PLANNED
Priority           P1
Goal               test-token faucet for onboarding
Dependencies       B1-02, B3-03 (network + observability surface)
Allowed changes    faucet/**
Forbidden changes  presenting test tokens as valuable
```

**Implementation**

```text
address validation     bech32 + chain prefix
per-account rate limit
per-IP rate limit
cooldown               documented window
balance                faucet account balance monitoring + alerting
anti-loop              detection of farming patterns (bounded, documented)
metrics                request counters/outcomes
```

Product text must state plainly: **test tokens have no monetary value**.

**Definition of Done** — a new user can fund a test account safely; the
limits are enforced and documented.

---

### B7-02 — Docs site

```text
Task ID            B7-02
Status             PLANNED
Priority           P1
Goal               documentation a stranger can follow
Dependencies       B5-01, B6-06, B7-01
Allowed changes    docs/**
Forbidden changes  documenting features that are not implemented
```

**Required documents**

```text
Developer Quickstart
Worker Quickstart
Watcher Quickstart
DA Quickstart
Validator guide
Protocol overview   (frozen protocol, plain-language; links the five specs)
API reference
CLI reference
Troubleshooting
Known limitations   (honest list: supported profile(s) only, testnet economics, etc.)
```

**Definition of Done** — each document is executable by a stranger
against the private testnet; every claim is backed by a shipped component.

---

### B7-03 — Status page

```text
Task ID            B7-03
Status             PLANNED
Priority           P1
Goal               read-only public picture of the network
Dependencies       B1-02
Allowed changes    ops/**
Forbidden changes  write access; leaking validator keys or private infra
```

**Display at least**

```text
chain height
RPC status
validators
DA quorum
workers
jobs
challenges
faucet
```

**Definition of Done** — the page is read-only, reflects real chain/Daemon
state, and is useful for the public alpha.

---

## M4 — Productization Ready (hard gate)

`M4 PRODUCTIZATION_READY` may be declared only when the §4.1 checklist is
fully green, with evidence committed under `docs/productization/`:
five binaries, Job API, Scheduler, clean-machine smoke for each component.

Until then, M4 remains `ACTIVE`.
---

# Phase C / D / E — Preserved plans (concepts unchanged)

These phases keep their original task IDs, goals, tests and
deliverables (v1.0 wording preserved; full text also in the v1.0 historical
file).  Only Status fields are added.  None of them may start early:

```text
Phase C (M5)  BLOCKED_BY_M4   Private Testnet Alpha must be REAL multi-host,
                              public-IP/network — never one compose host
Phase D (M6)  BLOCKED_BY_M5   Public Testnet Alpha = a real release:
                              Protocol + Network + Node Software + API + Docs,
                              never "deploy one server"
Phase E       BLOCKED_BY_M6   Beta/hardening; may not start before M6 PASS
```


## Phase C — Private Testnet Alpha


### C1 — Deployment (multi-host)

- **C1-01 · 多主机 validator 拓扑**
  - 目标：4 validators 放到独立主机；最好不同 provider/region；public RPC 与 validator 分离
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c1-01.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`
- **C1-02 · 部署 2 个 public RPC/sentry**
  - 目标：TLS/限流/metrics；validator 不直接暴露公共 RPC
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c1-02.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`
- **C1-03 · 部署 3 个 DA providers**
  - 目标：独立进程/主机；至少一个不同 provider；磁盘监控
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c1-03.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`
- **C1-04 · 部署 2 个 Watchers**
  - 目标：至少一个与 Worker/DA 不共享主机/磁盘
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c1-04.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`
- **C1-05 · 部署 2 个官方 bootstrap GPU Workers**
  - 目标：只用于保证基础算力，不作为去中心化证据
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c1-05.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`
- **C1-06 · 部署 Job API/Scheduler/Faucet**
  - 目标：控制面与 validator 分离；Postgres/queue 备份
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c1-06.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`

### C2 — Drills (cross-internet)

- **C2-01 · 跨公网 Worker join test**
  - 目标：全新机器按文档安装、bond、接任务、完成 VWR
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c2-01.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`
- **C2-02 · 跨公网 Watcher test**
  - 目标：不同网络环境验证 bundle 并正确不挑战 honest task
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c2-02.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`
- **C2-03 · Worker disappearance drill**
  - 目标：Commit+DA quorum 后关机；Watcher/chain 仍完成流程
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c2-03.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`
- **C2-04 · Validator offline drill**
  - 目标：关 1 validator；3 个继续；恢复 catch-up
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c2-04.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`
- **C2-05 · DA offline drill**
  - 目标：关 1 DA 仍工作；关 2 DA 阻止 finalize
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c2-05.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`
- **C2-06 · Watcher crash drill**
  - 目标：challenge 处理中重启 Watcher；从 chain/DA 恢复
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c2-06.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`
- **C2-07 · Fraud injection drill**
  - 目标：真实 wide GEMM fraud 在公网环境完成 512-MAC settlement
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c2-07.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`
- **C2-08 · 连续运行稳定性**
  - 目标：至少持续运行足够时间捕捉内存泄漏/状态漂移/重复 receipt；具体时长由当期运维计划定义并预先记录
  - 依赖：Productization Ready｜测试：每项有 runbook + result JSON/incident log｜交付物：`docs/private-testnet/c2-08.md`｜DoD：测试结果可复现；失败有 root cause / issue
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M4`

## Phase D — Public Testnet Alpha


### D1 — Release & public onboarding

- **D1-01 · 发布版本化二进制**
  - 目标：GitHub Release：prismad / prisma-worker / prisma-watcher / prisma-da / prisma-cli；SHA256
  - 依赖：Private Testnet Alpha PASS｜测试：在干净外部环境完成 smoke / onboarding test｜交付物：`docs/public-testnet/d1-01.md`｜DoD：外部用户不需要改源码即可完成目标
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M5`
- **D1-02 · 发布容器镜像**
  - 目标：固定 image digest；node/worker/watcher/da；不包含 secret
  - 依赖：Private Testnet Alpha PASS｜测试：在干净外部环境完成 smoke / onboarding test｜交付物：`docs/public-testnet/d1-02.md`｜DoD：外部用户不需要改源码即可完成目标
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M5`
- **D1-03 · 发布 genesis / chain-id / seeds**
  - 目标：公开 checksum；文档化 reset policy
  - 依赖：Private Testnet Alpha PASS｜测试：在干净外部环境完成 smoke / onboarding test｜交付物：`docs/public-testnet/d1-03.md`｜DoD：外部用户不需要改源码即可完成目标
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M5`
- **D1-04 · 公开 RPC / Faucet / Job API**
  - 目标：域名/TLS/status；限流
  - 依赖：Private Testnet Alpha PASS｜测试：在干净外部环境完成 smoke / onboarding test｜交付物：`docs/public-testnet/d1-04.md`｜DoD：外部用户不需要改源码即可完成目标
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M5`
- **D1-05 · 公开 Worker onboarding**
  - 目标：一条文档路径完成安装→faucet→bond→join→job
  - 依赖：Private Testnet Alpha PASS｜测试：在干净外部环境完成 smoke / onboarding test｜交付物：`docs/public-testnet/d1-05.md`｜DoD：外部用户不需要改源码即可完成目标
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M5`
- **D1-06 · 公开 Watcher onboarding**
  - 目标：安装→sync→verify；默认安全配置
  - 依赖：Private Testnet Alpha PASS｜测试：在干净外部环境完成 smoke / onboarding test｜交付物：`docs/public-testnet/d1-06.md`｜DoD：外部用户不需要改源码即可完成目标
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M5`
- **D1-07 · 公开 DA onboarding**
  - 目标：register/bond/store/respond challenge
  - 依赖：Private Testnet Alpha PASS｜测试：在干净外部环境完成 smoke / onboarding test｜交付物：`docs/public-testnet/d1-07.md`｜DoD：外部用户不需要改源码即可完成目标
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M5`
- **D1-08 · 公开 Developer quickstart**
  - 目标：API key→submit job→observe verify/finalize→query VWR
  - 依赖：Private Testnet Alpha PASS｜测试：在干净外部环境完成 smoke / onboarding test｜交付物：`docs/public-testnet/d1-08.md`｜DoD：外部用户不需要改源码即可完成目标
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M5`

### D2 — First external actors

- **D2-01 · 首个外部 GPU Worker 成功**
  - 目标：非官方控制机器完成真实 job，收到 test reward/VWR
  - 依赖：Private Testnet Alpha PASS｜测试：在干净外部环境完成 smoke / onboarding test｜交付物：`docs/public-testnet/d2-01.md`｜DoD：外部用户不需要改源码即可完成目标
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M5`
- **D2-02 · 首个外部 Watcher 成功**
  - 目标：非官方 watcher 独立验证真实 job
  - 依赖：Private Testnet Alpha PASS｜测试：在干净外部环境完成 smoke / onboarding test｜交付物：`docs/public-testnet/d2-02.md`｜DoD：外部用户不需要改源码即可完成目标
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M5`
- **D2-03 · 首个外部 DA provider 成功**
  - 目标：加入 registry 并参与 2-of-3 availability
  - 依赖：Private Testnet Alpha PASS｜测试：在干净外部环境完成 smoke / onboarding test｜交付物：`docs/public-testnet/d2-03.md`｜DoD：外部用户不需要改源码即可完成目标
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M5`
- **D2-04 · 公开 fraud drill**
  - 目标：计划性 testnet fraud injection；社区可观察 challenge→512-MAC→settlement
  - 依赖：Private Testnet Alpha PASS｜测试：在干净外部环境完成 smoke / onboarding test｜交付物：`docs/public-testnet/d2-04.md`｜DoD：外部用户不需要改源码即可完成目标
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M5`
- **D2-05 · Known limitations 文档**
  - 目标：明确只支持冻结 Profile、test token 无真实价值、单 proposer censorship ≠ cartel resistance
  - 依赖：Private Testnet Alpha PASS｜测试：在干净外部环境完成 smoke / onboarding test｜交付物：`docs/public-testnet/d2-05.md`｜DoD：外部用户不需要改源码即可完成目标
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M5`

## Phase E — Public Testnet Beta / Hardening


### E1 — Hardening

- **E1-01 · 引入第三方 validators**
  - 目标：逐步降低官方 voting power 集中度；记录 validator operator runbook
  - 依赖：Public Testnet Alpha stable｜测试：预声明场景 + 结果记录｜交付物：`docs/testnet-beta/e1-01.md`｜DoD：hardening 结果可量化并进入 Mainnet Candidate 决策
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M6`
- **E1-02 · 扩大外部 DA / Watcher**
  - 目标：多云多地域；观察 availability/challenge 稳定性
  - 依赖：Public Testnet Alpha stable｜测试：预声明场景 + 结果记录｜交付物：`docs/testnet-beta/e1-02.md`｜DoD：hardening 结果可量化并进入 Mainnet Candidate 决策
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M6`
- **E1-03 · Worker reliability scoring**
  - 目标：基于 uptime/job success/latency 形成调度权重；不改变 consensus
  - 依赖：Public Testnet Alpha stable｜测试：预声明场景 + 结果记录｜交付物：`docs/testnet-beta/e1-03.md`｜DoD：hardening 结果可量化并进入 Mainnet Candidate 决策
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M6`
- **E1-04 · Scheduler fairness / backpressure**
  - 目标：队列、公平性、限流、失败重派；防单用户独占
  - 依赖：Public Testnet Alpha stable｜测试：预声明场景 + 结果记录｜交付物：`docs/testnet-beta/e1-04.md`｜DoD：hardening 结果可量化并进入 Mainnet Candidate 决策
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M6`
- **E1-05 · Upgrade rehearsal**
  - 目标：协议兼容升级 / binary upgrade / state migration / rollback runbook
  - 依赖：Public Testnet Alpha stable｜测试：预声明场景 + 结果记录｜交付物：`docs/testnet-beta/e1-05.md`｜DoD：hardening 结果可量化并进入 Mainnet Candidate 决策
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M6`
- **E1-06 · Chaos testing**
  - 目标：RPC overload、DA disk loss、worker churn、watcher crash、validator restart
  - 依赖：Public Testnet Alpha stable｜测试：预声明场景 + 结果记录｜交付物：`docs/testnet-beta/e1-06.md`｜DoD：hardening 结果可量化并进入 Mainnet Candidate 决策
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M6`
- **E1-07 · Security review**
  - 目标：domain separation、signature replay、proof confusion、resource DoS、key management
  - 依赖：Public Testnet Alpha stable｜测试：预声明场景 + 结果记录｜交付物：`docs/testnet-beta/e1-07.md`｜DoD：hardening 结果可量化并进入 Mainnet Candidate 决策
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M6`
- **E1-08 · 经济参数回放**
  - 目标：bond/challenge bond/reward/DA penalty/worker price；test token only
  - 依赖：Public Testnet Alpha stable｜测试：预声明场景 + 结果记录｜交付物：`docs/testnet-beta/e1-08.md`｜DoD：hardening 结果可量化并进入 Mainnet Candidate 决策
  - Status: `PLANNED`｜Blocked: `BLOCKED_BY_M6`

### Phase C/D/E exit criteria (unchanged, restated)

```text
M5 exit  external simulated nodes close the loop across the internet
         (multi-host validators, public RPC/sentry, 3 DA providers,
          2 watchers, 2 bootstrap GPU workers, Job API/Scheduler/Faucet,
          all C2 drills passed, sustained stability run)
M6 exit  >= 1 external Worker and >= 1 external Watcher succeed on real jobs:
         developer API submission -> real job -> real verification ->
         real settlement; public fraud drill recorded; known-limitations doc shipped
```

---

# Phase F lineage (one table — research history preserved, failures included)

| phase | branch | outcome |
|---|---|---|
| F | `protocol/transformer-phase-f` | canonical operator framework, CANONICAL_GRAPH_V1, chain settlement, 4-validator E2E — PASS at tested scope |
| F.1 | `protocol/transformer-phase-f1-real-model` | real pinned Qwen3-0.6B layer block converted (310 nodes, GraphID `6ad46fde…`); accuracy gate **FAIL 0.957/1.91** (later corrected to 0.9933/0.423 after the F.2A scale-bug fix); node manifest + watcher demo |
| F.2A | `research/transformer-phase-f2a-quant` | groupwise INT8 **NOT FEASIBLE** (0.994210/0.313896); found and fixed the F.1 scores-scale bug |
| F.3A | `research/transformer-phase-f3a-activation-bits` | through A13 **NOT FEASIBLE** (0.99941/0.0906); floor located in W8 |
| F.4A | `research/transformer-phase-f4a-joint-precision` | int32 AW frontier **NOT FEASIBLE** (best A12W9 0.99970/0.0887) |
| F.5A | `research/transformer-phase-f5a-int64-frontier` | **FEASIBLE_INT64_A13W10** (0.999912/0.03893, 0 errors > 0.05); exactly +2 operand bits beyond the int32-safe budget |
| F.5B | `research/transformer-phase-f5b-gpu-feasibility` | exact on two GPUs; **GPU_CORRECT_BUT_NOT_PRACTICAL** (eager 10.5×/12.8×) → fusion required |
| F.5B.1 | `research/transformer-phase-f5b1-fused-gemm` | **TRUE_FUSED_MMA_A13W10** weighted 1.17× on both architectures; 83/83 exact; SM fact: `_int_mm` needs M > 16, CUDA has no generic integer matmul |
| F.5B.2 | `research/transformer-phase-f5b2-full-block-gpu` | **FUSED_GPU_FEASIBLE_A13W10** — full 310-node block 310/310 nodes + roots + final exact on A40 + RTX 2000 Ada; fallback 0; float 0 |
| F.5C | `protocol/transformer-phase-f5c-wide-integer` | protocol core + Watcher V2 + graph DA + measured gas + **all 4-validator real-block E2E scenarios PASS (A6)** + docs + **PHASE_F = PASS, freeze tag `transformer-phase-f-wide-integer`** |

**Failed research is preserved unsoftened**: plain W8A8 and groupwise
INT8, activation-only up to A13, and the whole int32 joint frontier do
NOT pass the frozen accuracy gate for this model and profile.  Do not
delete these conclusions; they are why A13W10 (int64) is the frozen
arithmetic.

---

# Appendix A — Validation checklist (v1.1 rewrite)

Mechanical checks required by the migration; results recorded in
`docs/roadmap-v1.1-migration.json` and re-runnable at any time:

```text
[X] PHASE_F = PARTIAL does not appear as current state anywhere in v1.1
[X] no A0-*..A7-* task is ACTIVE or NEXT (all DONE/FROZEN; NEXT lists B0/B1 only)
[X] Phase A task count preserved: 61/61 IDs present
[X] Phase B task count preserved: 28/28 old IDs present (+5 new B0-*)
[X] Phase C task count preserved: 14/14
[X] Phase D task count preserved: 13/13
[X] Phase E task count preserved: 8/8
[X] removed task IDs: none
[X] renamed task IDs: none
[X] single-authoritative-roadmap rule stated; v1.0 kept as historical copy
[X] freeze tag / commit / GraphIDV2 / PolicyID re-verified from the repo (see §2.1)
```

# Appendix B — Task reconciliation v1.0 → v1.1

```text
old Phase A task IDs   61   (A0-01..A0-02, A1-01..A1-08, A2-01..A2-08,
                             A3-01..A3-10, A4-01..A4-07, A5-01..A5-07,
                             A6-01..A6-11, A7-01..A7-08)  -> all DONE/FROZEN
old Phase B task IDs   28   (B1-01..B1-03, B2-01..B2-09, B3-01..B3-03,
                             B4-01..B4-03, B5-01, B6-01..B6-06, B7-01..B7-03)
new B0 task IDs         5   B0-01, B0-02, B0-03, B0-04, B0-05  (all new in v1.1)
new Phase B task IDs   33   = 28 preserved + 5 new
old Phase C task IDs   14   preserved (C1-01..C1-06, C2-01..C2-08)
old Phase D task IDs   13   preserved (D1-01..D1-08, D2-01..D2-05)
old Phase E task IDs    8   preserved (E1-01..E1-08)
removed task IDs         0
renamed task IDs         0
v1.1 Phase B refinements  B1-01 -> subtasks a..h; B1-03 -> a..i; B1-02 -> a..i;
                          B2-01 -> a..h; B2-02..B2-09 detailed;
                          B3/B4/B5/B6/B7 detailed per v1.1 sections
execution-order change    B4 before B3 (DA retrieval surface first);
                          B1 order B1-01 -> B1-03 -> B1-02
```

# Appendix C — Migration artifact

Machine-readable bookkeeping lives in `docs/roadmap-v1.1-migration.json`
(counts above are generated there from the repository, not hand-typed).

---

# Final principle

```text
Freeze the science.
Preserve the evidence.
Clean the repository.
Package the nodes.
Connect real users.
Ship the testnet.
```

This is a **state transition**, not a new invention: the old roadmap
answered *"can Prisma's core protocol work?"* — **YES**.  v1.1 answers
*"how do we turn the frozen protocol into a real network that strangers
can install, join, submit jobs to, contribute GPUs to, verify, and use?"*
