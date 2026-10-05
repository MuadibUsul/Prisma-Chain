# Phase E — Permissionless Verification

This document records the implemented permissionless-verification layer:
E1 challenge inclusion under a censoring proposer, and E2 DA_REPLICA_V1
replicated output availability with bonded providers and objective
on-chain sampling challenges. It is not a claim that censorship is
impossible or that data availability is solved; see the security model at
the end for the exact limits.

## E1 — Challenge inclusion

A single proposer that omits a valid GEMM challenge transaction from its
proposal must not be able to suppress it while the network retains more
than two thirds of honest voting power and transaction propagation
works. Other validators still consider the transaction valid; only the
censoring proposer's block lacks it.

Topology: four equal-power validators (25% each). One offline validator
leaves 75% > 2/3 so finalization continues; two offline leave 50% and the
chain halts, which is the expected CometBFT safety behavior, not a
failure.

The devnet ships a development-only proposer harness
(`chain/app/devcensor.go`) that drops selected GEMM messages from the
proposal when `PRISMA_DEV_CENSOR_GEMM_CHALLENGES` is set on that node. It
never rejects or invalidates a transaction (CheckTx, ProcessProposal and
DeliverTx still accept it), it is off by default, and it exists solely to
prove inclusion by an honest proposer later.

Observed on the live four-validator devnet (both the standalone E1 run
and the combined run):

- The challenge is broadcast into the censoring validator's own mempool
  at the rotation window before its slot.
- The censoring block omits the pending transaction (objective omission).
- The next honest proposer includes it one block later.
- The dispute then completes: single-tile traces, bisection, 512-MAC
  arbitration, ChallengerWins, requester refunded, no worker receipt.

Limits: a validator cartel can still censor; transaction propagation must
keep working; the challenge window must be long enough to cover proposer
rotation (measured worst-case inclusion delay here was one block).

## E2 — DA_REPLICA_V1

The chain already commits every 8x8 int32 output tile through
`output_root`; DA_REPLICA_V1 reuses that commitment. There is no second
Merkle tree and no second identity system: a DA provider is any account
with a bonded network Ed25519 key that registers as a provider.

- **Replication**: the worker replicates the canonical full C to providers;
  each provider recomputes `output_root` from the received bytes and
  REFUSES to store or attest anything whose root does not match the
  chain's committed root.
- **Attestation**: `DAAttestation` (canonical CBOR, `DA_REPLICA_V1`,
  signed with the bonded network key) claims "I hold the blob and can
  serve it until `available_until_height`". It is a bonded claim, not a
  proof of future availability.
- **Quorum**: 2-of-3 required; a task cannot finalize without the quorum
  covering the mandatory retention horizon (challenge window plus the DA
  window), and cannot finalize while any sampling challenge is open.
- **Sampling challenge**: `MsgOpenGEMMDAChallenge` derives the tile from
  chain facts (`task_id`, provider, challenger, nonce, height) via a
  domain-separated hash. This is a sampling challenge, not a randomness
  beacon, and is documented as such. The provider answers with the
  256-byte canonical tile and a `gemmv1.OutputTileProof` against the
  committed `output_root`.
- **Objective timeout**: an unanswered challenge past its deadline is a
  chain-verifiable fact. The provider's bond loses a fixed, versioned
  testnet penalty (100,000 uprsm) paid to the challenger together with
  the returned bond; a passing answer burns the challenger's bond
  (anti-griefing). No HTTP failure is ever evidence.
- **Loss and repair**: quorum loss blocks finalization; the worker can add
  an independent replacement attestation to restore it, otherwise after
  the DA window anyone can route the task to `availability_failed` and the
  requester refund, with no receipt and no worker payment.

Endpoints of a replica: `GET /v1/da/{task}/output` (bulk canonical C),
`GET /v1/da/{task}/tile/{i}/{j}` (tile plus proof), `POST /store`,
`POST /attest`, `GET /health`. Storage is on disk and survives restarts.

## Security model (exact limits)

DA_REPLICA_V1 proves **permissionless replicated availability with bonded
provider attestations and objective on-chain tile challenge-response**.
It depends on enough independent providers remaining available and honest
and on challenges being includable by the chain. It is not perfect global
data availability, not erasure coding, not data availability sampling,
not a dedicated DA network and not a KZG/polynomial commitment. Stronger
DA designs are future work. HTTP failures never slash anyone; only the
chain's own deadline is objective.

## Consensus boundary

DA handlers verify signatures, bounded bytes, Merkle proofs, block
heights, bonds and state only: no network or filesystem I/O, no
Freivalds, no output reconstruction, no floating point. Freivalds remains
an off-chain watcher strategy; useful work never influences block
production or voting power.
