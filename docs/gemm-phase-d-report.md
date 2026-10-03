# Phase D Report — On-chain GEMM Settlement

Baseline branch `protocol/gemm-v0.1.2-freivalds`, baseline SHA `64742af`.
Phase D branch `protocol/gemm-phase-d-chain` in the worktree
`../Prisma-Chain-phase-d` (the original worktree and its uncommitted
network/ WIP and LICENSE were untouched by construction).

## Commits (oldest first)

- `c425e69` GEMM dispute snapshot, transcript chain, signature helpers
- `b583587` GEMM task storage, admission, ResultCommit verification, challenges
- `5458c1f` settlement tests: honest, fraud, false challenge
- `0ab74c7` adversarial tests; challenge bond moved after admission
- `f33ed77` devnet E2E (honest, fraud, false-challenge) plus interop fixes

## Files changed

- `compute/gemmv1/`: `dispute_snapshot.go`, `transcript.go`, signature
  helpers, `DepthFor`, value-receiver state encoding
- `chain/x/compute/`: `gemm.go`, `gemm_keeper.go`, `gemm_dispute.go`,
  `gemm_dispute_helpers.go`, `gemm_gas.go`, `gemm_receipt.go`,
  `gemm_query.go`, `gemm_test.go`, `gemm_adversarial_test.go`,
  `gemm_gas_measure_test.go`
- `chain/proto/prisma/compute/v1/{tx,query}.proto` + regenerated pb.go
  (via protoc 29.3 + protoc-gen-gogofaster 1.7.2; the regenerated files
  for the pre-existing messages are byte-identical to the checked-in
  ones apart from line endings)
- `chain/local-entrypoint.sh`, `deploy/gemm_chain_smoke.py`
- `docs/`: this report, `gemm-phase-d-chain.md`,
  `gemm-chain-gas-results.json`

## Test results

| Suite | Result |
| --- | --- |
| Bounded-VM regression (`TestVerifiableTask...` etc.) | PASS unchanged |
| Lightweight regression (`light_test.go`) | PASS unchanged |
| GEMM settlement (11 tests) | PASS |
| Root-module suites (`vm`, `compute/gemmv1`, `verify`) | PASS |
| Devnet honest E2E | PASS |
| Devnet fraud E2E | PASS |
| Devnet false-challenge E2E | PASS |

Devnet evidence (real chain, prismad transactions):

- honest: task settled once, VWR stored, worker balance 5,000,000 ->
  6,400,000 (bond 5M locked then returned; worker share 1.4M of the 2M
  fee), requester escrow consumed by the fee split.
- fraud: corrupted tile, on-chain dispute with 1 bisection round and
  arbitration at K-step 1, status `fraud`, requester refunded, challenger
  bond returned plus the 10% worker-bond slash, worker balance 5,000,000
  (no payout).
- false-challenge: honest worker survived (`survived_challenge=true`,
  32-byte transcript digest), then settled with
  `challenged_worker_won`; the losing challenger's bond was burned
  (challenger 5,000,000 -> 4,980,000 net of the burn).

## Adversarial coverage

- Replay: cross-task ResultCommit signature reuse rejected; cross-task
  challenge tile proof rejected; stale/future midpoint step rejected;
  state-machine jumps (post->arbitrate, result without accept, arbitrate
  without challenge, arbitrate before trace lock) all rejected.
- Timeouts: worker silent, challenger silent, both silent — the three
  trace-phase outcomes (`worker_wins`, `challenger_wins`, `both_invalid`
  via the refund path).
- Queue griefing: a worker-controlled false challenger cannot block an
  honest queued challenger (promotion with a fresh deadline and intact
  bond); duplicates rejected without a second bond lock; saturation
  rejected deterministically.
- Restart: keeper recreated mid-bisection resumes from the GMD1 snapshot
  byte-identically and finishes with the same outcome.
- DoS: oversized proof depth, short siblings and absurd proof counts are
  rejected before any hashing; every proof bound is derived from the task
  shape.
- Gas: `GEMMGasV1` schedule measured per handler
  (`docs/gemm-chain-gas-results.json`): Post 32,417; Accept 68,706;
  SubmitResult 55,545; OpenChallenge 114,890; CommitTrace 119,486 /
  248,494 (worker / challenger at K=64); one bisection round about 1.9M
  including store writes; **ArbitrateGEMM 145,797 gas units for the
  512-MAC micro-step with proof depth 4 and 320 witness bytes**;
  Finalize 87,035. Gas is consensus accounting; wall time is separate.

## Interop defects found by running the real chain (all fixed)

1. Autocli commands read the keyring backend from `client.toml` and do
   not re-read `--keyring-backend`; the devnet entrypoint now pins
   `keyring-backend = "test"` and symlinks the default node home to the
   data volume.
2. The receipt rebuild used the submission height instead of the worker's
   declared `completed_epoch`, so the canonical signature stopped
   verifying at finalize; the declared epoch is now persisted.
3. `ChallengeOpen` embeds `opened_epoch` in the signed preimage, which a
   client cannot predict from the execution height; the challenger now
   supplies it and the chain range-checks it.

## Known limitations

- Data availability: `output_data_ref` is metadata; the chain cannot
  verify that the worker served C. Strong DA remains an open gate and no
  availability failure is a consensus slash.
- Receipts are stored as queryable JSON metadata whose authoritative ID is
  the gemmv1 canonical CBOR hash; a light-client receipt proof format is
  future work.
- The Verified-Work ledger counts canonical MACs only; CWU display and
  cross-operator weighting stay out of Phase D.
- Validator censorship of challenge transactions remains possible; the
  chain does not have an anti-censorship inclusion proof yet.

## Definition of Done (§68)

1-7 task/economy path: PASS (keeper tests + devnet honest).
8-12 challenge, single-tile traces, persisted bisection, restart, 512-MAC
arbitration: PASS (keeper tests + devnet fraud).
13-15 worker loss, false challenger, timeouts: PASS.
16-19 replay, queue, duplicate settlement/VWR: PASS.
20-22 supply invariants, DoS bounds, explicit gas measured: PASS.
23-25 bounded-VM, lightweight and full-repo suites: PASS.
26-29 devnet honest/fraud/false-challenge and this report: PASS.

## Final answers

### Q1 — Can a GEMM_INT8_V1 task move from on-chain escrow to a
GPU-compatible ResultCommit to challenge/finalization to settlement
without altering CometBFT consensus?

**YES.** The devnet chain (unmodified CometBFT, single validator, PoS/BFT)
executes the full path: escrow at post, bonded acceptance, canonical
ResultCommit verification under the bonded GPU-worker network key (the
same key type the RunPod E2E used), challenge window, settlement with a
canonical VWR. No consensus parameter, module or ABCI surface changed.

### Q2 — Can an incorrect GEMM ResultCommit be challenged and
deterministically adjudicated on the chain with only one 512-MAC
micro-step at final arbitration?

**YES.** On the live chain: corrupted tile -> challenge with the committed
output-root membership proof -> both parties lock traces for that tile
only -> chain-driven bisection -> `ArbitrateGEMM` executes exactly one
8x8x8 micro-step (512 canonical MACs, measured 145,797 gas) ->
`ChallengerWins`, refund, no receipt.

### Q3 — Can a valid GEMM task issue exactly one Verified Work Receipt and
an invalid worker none?

**YES.** The honest devnet run mints one receipt (duplicate finalize and
duplicate receipt ID rejected); the fraud run ends `fraud` with no receipt
and no work credit; the false-challenge run settles with
`challenged_worker_won` and the transcript digest attached.

### Q4 — Do the existing bounded-VM and lightweight paths remain
regression-free after GEMM integration?

**YES.** `go test ./...` in the chain module is green including the
pre-existing VM and lightweight suites; storage namespaces are proven
disjoint; no existing message, state key or settlement rule changed.

## Next blocker

A second independent validator and multi-machine devnet for the economic
drills of the operational gates, then the first real transformer block
RMSNorm/Softmax/RoPE/SiLU/Attention as the next canonical operators — not
before the Permissionless data-availability design is settled.
