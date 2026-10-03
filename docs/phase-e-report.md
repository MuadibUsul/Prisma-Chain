# Phase E Report — Permissionless Verification

Baseline branch `protocol/gemm-phase-d-chain`, frozen tag
`gemm-phase-d-devnet` (commit `477dc99`); Phase E branch
`protocol/gemm-phase-e-permissionless` in the worktree
`../Prisma-Chain-phase-e`. The original worktree and its uncommitted
`network/` WIP and `LICENSE` were untouched.

## Commits (oldest first)

`477dc99` docs consistency + tag · `4f786e4` four-validator devnet,
liveness, censorship harness and E1 proof · `2e38fad` DA_REPLICA_V1 chain
layer · `24193e0` combined fraud + censorship + DA E2E · plus the gas and
docs commit.

## Validator topology

Four independent validators, 25% voting power each, distinct node keys,
consensus keys, homes, P2P identities and RPCs, one shared genesis
(`docs/phase-e-validator-results.json`, `docs/phase-e-validator-topology.json`
fields are in the validator results).

| Stage | Result |
| --- | --- |
| Four up | finalizing, identical app hash, power [1,1,1,1] |
| validator-d stopped | a/b/c kept finalizing at 75% (34 -> 39) |
| validator-d restarted | caught up from its own data in 6.2 s to the same height and app hash |
| validator-c + d stopped | 50%: chain froze (expected CometBFT behavior, not a failure) |
| both restarted | chain resumed to identical state |

Real safety behavior observed and documented: a validator that stays down
long enough is **downtime-jailed and slashed 1%** by the staking/slashing
modules. This is expected chain safety, not a harness failure; the
incident is recorded here so it is not mistaken for a bug.

## E1 — challenge inclusion under a censoring proposer

Two independent runs on the live four-validator chain, with
validator-a running `PRISMA_DEV_CENSOR_GEMM_CHALLENGES=all` (proposer
omission only; the transaction stays valid for every validator):

- Standalone run (`docs/phase-e-censorship-results.json`): block 26 was
  proposed by the censoring validator and omitted the challenge it held;
  block 27 (validator-b) included it; the dispute completed to
  ChallengerWins with no receipt.
- Combined run (`docs/phase-e-da-results.json`): challenge broadcast at
  the window before validator-a's slot (height 749 window, broadcast
  750); **block 750 omitted it (censored_by=[750])**; block 751 by
  validator-b included it (delay 1 block); full dispute completed.

## E2 — DA_REPLICA_V1

Topology: three independent provider accounts with distinct bonded
network keys (`prism16wq83…`, `prism1nmm9…`, `prism1arc6…`), persistent
disk stores, plus a permissionless watcher with its own funded account
and network key. Quorum: 2-of-3 required, 3/3 attained in the run.

| Scenario | Result |
| --- | --- |
| Replication with root verification | 3/3 providers recomputed output_root and stored; wrong-blob store refused (`OUTPUT_ROOT_MISMATCH`) |
| Attestations | 3 distinct signed `DAAttestation`s; status `challenge_window_ready` (valid 3/required 2) |
| Bulk retrieval | watcher downloaded full C through the DA layer in 2.0 s and rebuilt `output_root` (mandatory, instrumented) |
| Freivalds | detected the corrupted tile; localized to tile (0,0) |
| One provider offline | da-a terminated before retrieval; the watcher still obtained C (no HTTP failure ever treated as evidence) |
| Objective timeout path | keeper tests: unanswered challenge past deadline -> fixed 100,000 uprsm penalty slashed from the provider bond, bond returned plus penalty to the challenger; duplicate timeout rejected |
| Quorum loss / replacement | keeper tests: 1/2 quorum blocks finalize; replacement attestation restores it; deadline expiry routes to `availability_failed` with a requester refund and no receipt |
| DoS bounds | one open challenge per provider, mismatched bond rejected, oversized proof rejected before hashing |

## Combined path (DoD 64) — the core demonstration

4 validators + fraudulent worker + permissionless watcher + 3 DA
providers: GEMM task -> fraudulent ResultCommit -> C replicated to DA ->
2-of-3 quorum -> watcher downloads C through DA only -> rebuilds
output_root -> Freivalds detects fraud -> bad tile (0,0) -> challenge
broadcast -> **censoring proposer omits it (block 750)** -> next honest
proposer includes it (751) -> single-tile traces -> 1 bisection round ->
512-MAC arbitration at step 1 -> **ChallengerWins** -> requester
refunded, worker balance unchanged and **no Verified Work Receipt** ->
**all four validators converged to the same app hash**
(`70F7C5CE98…` at height 644 in that run).

## DA gas (docs/phase-e-da-gas-results.json)

| Phase | gas units |
| --- | --- |
| RegisterDAProvider | 41,318 |
| SubmitDAAttestation | 94,510 / 115,684 |
| OpenGEMMDAChallenge | 117,940 |
| RespondGEMMDAChallenge (256-byte tile + bounded proof) | 123,849 |
| TimeoutGEMMDAChallenge | 162,122 |
| FailGEMMAvailability (refusal path) | 34,167 |

Gas is consensus accounting under the versioned schedule, not CPU time.

## Tests

`go test ./...` in the chain module is green including all Phase A-D
suites; the new DA adversarial suite covers wrong blob, wrong identity,
duplicate/retrograde attestation, quorum gate, sampling derivation,
passing/failing/timeout responses, replacement and DoS bounds. Devnet
suites: liveness (`e1_liveness.py`), censorship (`censorship_test.py`),
combined (`e2_combined_test.py`) all PASS.

## Known limitations

- Single-proposer censorship resistance only; a validator cartel can still
  censor, and propagation plus a sufficiently long challenge window are
  assumptions.
- DA_REPLICA_V1 is replicated full-output availability with bonded
  attestations, not erasure coding, DAS, KZG or a dedicated DA network.
  Retention is bounded by `available_until_height`.
- DA penalties are fixed testnet parameters, not an economic design.
- The watcher uses a deterministic Freivalds stream in the devnet driver;
  production watchers use CSPRNG (v0.1.2 layer unchanged).
- Receipt light-proofs and multi-operator drills remain open.

## Answers

**Q1 — Can one censoring proposer permanently prevent a valid GEMM fraud
challenge while the network retains >2/3 honest voting power?**
**NO.** Evidence: the censoring validator's own blocks omitted the
pending challenge (block 26 in the standalone run; block 750 in the
combined run) while the next honest proposer included it (27 / 751,
delay one block) and the dispute completed to ChallengerWins both times.

**Q2 — Can a permissionless Watcher obtain committed GEMM output from
independent availability providers and verify it matches output_root
before running Freivalds?**
**YES.** The watcher downloaded the full C through the DA layer (bulk 2.0
s), rebuilt the Merkle root from the bytes and matched the on-chain
`output_root` before running Freivalds; the check is mandatory and
instrumented.

**Q3 — Can a DA provider that attested availability be objectively
challenged on-chain for a specific committed output tile without relying
on an HTTP-failure claim?**
**YES.** The tile is derived on chain from committed facts; a valid
response is a 256-byte tile with an OutputTileProof against the
committed root; a missed deadline is an objective height check that
slashes the fixed versioned penalty. HTTP outcomes are never evidence.

**Q4 — Can loss of DA quorum prevent payment/VWR until availability is
restored or the task is refunded?**
**YES.** One attestation (1/2) blocks finalize; a replacement restores
the quorum; expiry without a quorum routes to `availability_failed` and a
requester refund with no receipt and no worker payment.

**Q5 — Can fraud still be detected and deterministically adjudicated when
one validator censors and one DA provider disappears simultaneously?**
**YES.** In the combined run validator-a censored the challenge and da-a
was terminated before retrieval; the watcher still obtained C from the
remaining providers, and the dispute completed to ChallengerWins with
the refund and no receipt.

**Q6 — Do all four Validators converge to identical final application
state after the complete fraud dispute and settlement?**
**YES.** All four validators reported the same height and identical app
hash at the end of the combined run (`validators_converged: true`).

## Next blocker

Canonical operator expansion (ADD, RMSNorm, RoPE, SiLU, Softmax,
Attention, Transformer block) can now start on a chain whose verification
right is permissionless; the open engineering items are receipt
light-proofs and multi-operator drills.
