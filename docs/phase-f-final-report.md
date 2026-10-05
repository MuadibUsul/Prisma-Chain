# Phase F — final report and lineage

## PHASE_F = PASS

The frozen gates that would allow a PASS are: real-model accuracy PASS;
GPU backend PASS; canonical wide protocol PASS; graph V2 PASS; watcher
PASS; graph DA PASS; wide-GEMM 512 dispute PASS; RoPE E2E PASS; honest
real-block E2E PASS; false-challenge PASS; 4-validator E2E PASS; all
regressions PASS.  Every gate is now evidenced: the protocol/GPU/
watcher/DA layers as before, and the complete A6 4-validator devnet
suite — including the REAL 310-node block's honest, GEMM-fraud (node 145)
and ROPE-fraud (node 85) runs, the false challenge, the combined
censorship+DA-offline scenario, the wide-dispute full-stack restart, the
validator offline/recovery drill and the availability-failure refund —
each with a committed evidence JSON and four-validator app-hash
convergence.  The freeze tag `transformer-phase-f-wide-integer` carries
pass meaning and its mapping is recorded in `docs/phase-f-freeze.json`.

## Lineage (all branches on origin)

| phase | branch | outcome |
|---|---|---|
| F | protocol/transformer-phase-f | canonical operator framework, CANONICAL_GRAPH_V1, chain settlement, 4-validator E2E — PASS at tested scope |
| F.1 | protocol/transformer-phase-f1-real-model | real pinned Qwen3-0.6B layer block converted (310 nodes, GraphID 6ad46fde…); accuracy gate **FAIL 0.957/1.91** (later corrected to 0.9933/0.423 after the F.2A scale bug fix); node manifest + watcher demo |
| F.2A | research/transformer-phase-f2a-quant | groupwise INT8 **NOT FEASIBLE** (0.994210/0.313896); found and fixed the F.1 scores-scale bug |
| F.3A | research/transformer-phase-f3a-activation-bits | through A13 **NOT FEASIBLE** (0.99941/0.0906); floor located in W8 |
| F.4A | research/transformer-phase-f4a-joint-precision | int32 AW frontier **NOT FEASIBLE** (best A12W9 0.99970/0.0887) |
| F.5A | research/transformer-phase-f5a-int64-frontier | **FEASIBLE_INT64_A13W10** (0.999912/0.03893, 0 errors >0.05); exactly +2 operand bits beyond the int32-safe budget |
| F.5B | research/transformer-phase-f5b-gpu-feasibility | exact on two GPUs; **GPU_CORRECT_BUT_NOT_PRACTICAL** (eager 10.5x/12.8x) → fusion required |
| F.5B.1 | research/transformer-phase-f5b1-fused-gemm | **TRUE_FUSED_MMA_A13W10 weighted 1.17x on both architectures**; 83/83 exact; SM-FACT: `_int_mm` M>16, no generic CUDA integer matmul |
| F.5B.2 | research/transformer-phase-f5b2-full-block-gpu | **FUSED_GPU_FEASIBLE_A13W10** — full 310-node block 310/310 nodes + roots + final exact on A40 + RTX 2000 Ada; fallback 0; float 0 |
| F.5C | protocol/transformer-phase-f5c-wide-integer | **complete**: protocol core + Watcher V2 + graph DA + real-block 4-validator E2E (honest/GEMM-145 fraud/ROPE-85 fraud/false challenge/combined adversarial/restart/recovery/availability) — **PASS**, freeze tag `transformer-phase-f-wide-integer` |

Failed research is preserved unsoftened in F.2A–F.4A reports: plain W8A8
and groupwise INT8, activation-only up to A13, and the whole int32 joint
frontier do NOT pass the frozen gate for this model and profile.

## What F.5C changed

Formal, versioned CANONICAL_TENSOR_V2 + GEMM_A13W10_I64_V1 +
REQUANTIZE_WIDE_V1 + CANONICAL_GRAPH_V2 (+ A13W10_I64_PROFILE_V1 bound
into GraphIDV2 = 8fb86087…, PolicyID eb9a…), ManifestV2 + CommitV3 +
ReceiptV3 with independent domains; GraphDisputeV2 + typed cheap-op
arbitration + WIDE_GEMM_DISPUTE_V1 (512-MAC, proved operand chunks);
FREIVALDS_A13W10_I64_V1 detection-only at 40 rounds with post-commit
randomness; GraphVerificationBundleV2 (39.6 MB real) with a permissionless
Watcher V2 (root phase + 83-GEMM Freivalds + 227 exact cheap ops,
`full_gemm_calls = 0`, fraud localization incl. the self-consistent GEMM
case and the RoPE case); typed graph DA (attestations, 2-replica quorum
gate, chunk challenges, objective timeout, availability-failed refund);
aggregate query; measured gas incl. the authoritative 512-MAC witness
(5,550 bytes, 366,347 gas).  All V1 objects and hashes unchanged.

## Report questions

**Q1.** Fully specified, versioned canonical protocol for the proven
A13W10 arithmetic? **YES** (this branch; five specs in docs/).

**Q2.** Does the formal Go/Python protocol reproduce the frozen real
Qwen3 block computation? **YES** — 310/310 node values vs the F.5B.2
manifest (Python executor) and 310/310 TensorRootV2 equality Go==Python.

**Q3.** Are all old V1 protocols and hashes unchanged? **YES** — legacy
golden suite pins V1 roots/GraphID/preimages/receipt ids byte-for-byte;
pb.go regenerations verified additive.

**Q4.** Can a permissionless Watcher verify the real 310-node block
without full GEMM recomputation? **YES** — 83 GEMMs via 40-round
Freivalds (1.1 s) + 227 exact cheap nodes (3.2 s); `full_gemm_calls = 0`.

**Q5.** Can verification data remain available after Worker
disappearance? **YES** — bundle + DA quorum + typed challenges; the
devnet drill ran with only 2-of-3 providers attesting the 39,648,960-byte
real bundle (one provider offline) and the honest real block still
finalized on quorum (`phase-f5c-e2e-honest.json`); with quorum
unreachable the task settles `availability_failed` and the requester is
refunded (`phase-f5c-e2e-availability-failure.json`).

**Q6.** Can a fraudulent real A13W10 GEMM be reduced to a deterministic
512-MAC chain proof? **YES** — chain e2e test: self-consistent fraud ->
bisection -> wide dispute -> 512-MAC -> ChallengerWins -> zero VWR.

**Q7.** Can a real RoPE fraud be adjudicated end-to-end? **YES** — on
the real 310-node block a self-consistent ROPE fraud (node 85, δ=100000)
bisected in 9 rounds to the exact node, and the bounded table-pinned
arbiter (2048-value committed table + one input chunk) settled
ChallengerWins -> fraud -> zero VWR; the watcher detects it and the four
validators converge (`phase-f5c-e2e-rope-fraud.json`).

**Q8.** Does an honest real block finalize exactly one VWR? **YES** —
the real 310-node block finalized on the devnet with exactly one VWR V3
(`fk7JvIrf…`), the fee split verified mechanically against balances
(escrow spent 20% burn / 2×5% monitors / 70% worker) and the watcher
passing (`phase-f5c-e2e-honest.json`).

**Q9.** Does a fraudulent real block finalize zero VWR? **YES** — both
the real GEMM-145 fraud (through the 512-MAC wide path) and the real
ROPE-85 fraud settle to `fraud` with an empty receipt, challenger
settlement and requester refund on the devnet.

**Q10.** Do all four validators converge under honest/fraud/DA-offline/
censorship scenarios? **YES** — every A6 scenario records the four
validators' height and app hash with a single distinct app hash; under a
censoring proposer the pending challenge was omitted for one block by
validator-a and included by validator-b the next (`censored_by: [1335]`,
`included_by: validator-b`); the full-stack restart kept the persisted
dispute byte-identical; a stopped validator caught up to the same height
and app hash.

**Q11.** 310 graph nodes / 83 GEMM / 252,706,816 logical MAC preserved?
**YES** — identical counts and chain-derived work vector.

**Q12.** Measured 512-MAC arbitration: witness bytes / gas / proof depth?
**5,550 bytes; 366,347 gas units; boundary trace proofs (1 sibling each
on the devnet-scale case), 10 typed evidence chunks, 128-byte chunk
payloads, exactly 512 MAC.**

**Q13.** GraphVerificationBundleV2 size? **39,648,960 bytes** for the
real 310-node bundle (inputs 31.5 MB + node outputs 8.0 MB + header).

**Q14.** Watcher verification cost? **~5 s total** on the real bundle
(root phase ~1.3 s incl. root recomputation, Freivalds 1.1 s, cheap ops
3.2 s); peak memory recorded in `phase-f5c-watcher-cost.json`.

**Q15.** Should Phase F be declared PASS? **YES — PASS.** Every
predeclared gate is evidenced, including the complete 4-validator devnet
E2E on the real 310-node block; no key item is NOT TESTED.  The freeze
tag `transformer-phase-f-wide-integer` is created and mapped in
`docs/phase-f-freeze.json`.  Two defects were found by the E2E and fixed
with regression tests before the tag (single-node dispute arb-ready
promotion; wide-arbitration kind-1 operand panic).

## Post-freeze work (Phase B, per the roadmap)

Phase B productization starts only now that PHASE_F = PASS: release
builds, CLI verb polish (B5-01), packaging, MLIR items of the later
phases.  No protocol semantics change after the tag.
