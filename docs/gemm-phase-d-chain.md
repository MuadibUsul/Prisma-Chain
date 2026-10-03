# Phase D — On-chain GEMM Settlement

This document records the implemented on-chain settlement path for
GEMM_INT8_V1: `chain/x/compute/gemm*.go` plus the devnet driver
`deploy/gemm_chain_smoke.py`. The execution/dispute wire protocol stays
`GEMM v0.1.1`; the off-chain v0.1.2 Freivalds watcher is not part of the
chain and never appears in consensus logic.

## Layering

```
OFF-CHAIN (never on chain)        ON-CHAIN (this document)
output download                   PostGEMMTask -> escrow
output_root rebuild               AcceptGEMMTask -> bonded worker
Freivalds detection               SubmitGEMMResult (canonical signature)
bad-row / bad-tile localization   OpenGEMMChallenge (committed tile proof)
                                  CommitGEMMTrace x2 (disputed tile only)
                                  SubmitGEMMMidState (chain-derived mid)
                                  ArbitrateGEMM (512-MAC micro-step)
                                  FinalizeGEMM -> VWR / refund
```

Freivalds is a detection strategy only; the only fraud proof the chain
accepts is the deterministic 8x8x8 micro-step executed inside
`ArbitrateGEMM`. No handler on this path calls `ReferenceGEMM` or any
full-matrix recomputation.

## Storage

Independent namespace, proven disjoint from the bounded-VM task keys:

| Key | Domain |
| --- | --- |
| `g` + u64 | GEMM task |
| `d` + u64 | active dispute record |
| `v` + 32B | Verified Work Receipt |
| `w` + address | worker verified-work ledger (canonical MAC total) |
| `gemm:next` | task id counter (separate from the VM `next`) |

Dispute internals live in the canonical `GMD1` snapshot from
`compute/gemmv1` (fixed size, identity-bound, midpoint-presence
canonical), so a node restart resumes bisection byte-identically. The
chain never stores a full trace.

## Transactions

- `MsgPostGEMMTask` — requester protocol key binding by Ed25519 possession
  proof over `prisma:gemm-requester-key-binding:v1` (chain id, requester
  address, protocol pubkey, nonce); the canonical `TaskDescriptor` and its
  task id come from `gemmv1`, never from the client; `max_fee` is escrowed.
- `MsgAcceptGEMMTask` — the worker's bonded network Ed25519 key is read
  from chain state (never self-reported); the canonical `Assignment` and
  assignment id are derived; the bond is reserved with the existing
  `ReserveBond` mechanism.
- `MsgSubmitGEMMResult` — the chain rebuilds the canonical `ResultCommit`
  (chain-derived `canonical_mac_count = M*N*K`, stored signature) and
  verifies it under the bonded worker key. No trace is accepted on the
  normal path.
- `MsgOpenGEMMChallenge` — chain-derived bond, worker-tile membership proof
  verified with the protocol Merkle code (proof depth bounded by the task
  shape), challenger signature over the fully deterministic canonical
  object (the challenger supplies `opened_epoch`, range-checked), and the
  queued-challenge mechanism (bounded queue, duplicate rejection, bond
  locked only after admission).
- `MsgCommitGEMMTrace` — per-party, disputed tile only; S0 zero-check,
  endpoint proofs, party signature, S_R bound to the party's claimed tile;
  both locks start the persisted dispute.
- `MsgSubmitGEMMMidState` — the step must equal the chain-derived midpoint;
  the state carries an inclusion proof against the party's locked root.
- `MsgArbitrateGEMM` — permissionless; witness tiles verified against the
  committed matrix roots, then exactly one `MicroStep` (512 canonical
  MACs).
- `MsgTimeoutGEMM` / `MsgFinalizeGEMM` / `MsgAbortGEMMTask` — permissionless
  progress; finalize is idempotent and mints exactly one receipt;
  abort is requester-only for expired, never-accepted tasks.

## Economics

Settlement reuses the existing verifiable-task semantics: burn 20%, each
of two bonded monitors 5%, worker the remainder; challenger bond return on
ChallengerWins and burn on WorkerWins/BothInvalid; worker bond slashed
10% capped by the reserved amount on a loss. All parameters are testnet
parameters. `output_data_ref` is availability metadata only: the chain
cannot fetch URLs, strong data availability remains an open gate, and no
HTTP failure maps to a consensus penalty.

## Gas

`GEMMGasV1` is an explicit, deterministic, versioned schedule charged
before verification work (see `docs/gemm-chain-gas-results.json`): base
per tx, per Merkle sibling, per hash, per state byte, per store write and
a fixed arbitration charge. Gas units are consensus resource accounting,
never CPU nanoseconds; wall time is reported separately.
