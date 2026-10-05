# Security model

What the network guarantees, what it explicitly does not, and where trust
sits.

## Trust placement

- **The chain is the only settlement authority.** Payment exists when the
  chain finalizes a task with a Verified Work Receipt; every dispute is
  adjudicated deterministically on chain (bisection → tile → 512 exact MACs).
  No component's word — API, scheduler, gateway, worker — settles anything.
- **Watchers verify independently.** Any party can run one; it needs chain +
  DA access only, never the Job API. Detection (Freivalds) is probabilistic
  and used only to decide *whether to challenge*; slashing is deterministic
  and on chain.
- **DA providers are bonded and accountable.** They verify a blob before
  attesting, post bonds, and lose them on provable unavailability (objective
  deadlines, typed tile proofs). 2-of-3 verified attestations gate
  finalization.

## Keys

- Worker/DA/watcher identities: ed25519 protocol key (network) + secp256k1
  account (chain), both derived/stored under the same rules: scrypt+AES-GCM
  keystore, 0600, group/world-readable refused, no recovery from a lost
  passphrase (stated, not hidden).
- A bonded network key **cannot be replaced** on chain — key rotation means a
  fresh account with its own bond.
- No key material is ever logged (two-layer redaction in the worker; daemons
  register their secrets).

## Surfaces

- The public RPC is a **read-only allow-list proxy**: per-IP rate limits,
  hard body caps, bounded upstream timeouts, websockets refused, no trust of
  `X-Forwarded-For`, health/metrics separate from the RPC surface. Writes go
  through authenticated services (faucet, Job API), not the public RPC.
- The Job API authenticates with scoped bearer keys (`jobs:read/write`),
  hashed at rest, revocable. It and the scheduler are centralized for the
  alpha **by design** — and hold no settlement power.
- Validators must not expose RPC publicly; sentries front them, and the
  private validator peers only with its sentries.

## What is NOT claimed

- No mainnet security claim; testnet only.
- No protection against a validator cartel censoring (single-proposer
  censorship is handled; a cartel is not).
- No sybil resistance beyond faucet guardrails and bonds.
- No confidentiality of job inputs beyond the documented privacy tiers;
  the ingress/egress of the gateway sees inputs by design (tier 0).
- DA is replicated full-output storage, not erasure coding / DAS / KZG.

## Bug reporting

Testnet only: report to the repository issues. Do not test against any
system you do not control.
