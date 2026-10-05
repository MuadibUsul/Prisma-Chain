# Testnet known limitations

Honest list. If something is not here, it is not a claim.

## Protocol scope

- **One frozen profile.** The only supported profile is
  `qwen3-0.6b-layer0-v2` (Qwen3-0.6B layer 0, CANONICAL_GRAPH_V2
  wide-integer). Arbitrary models / arbitrary HuggingFace checkpoints are not
  supported and the API refuses to claim them.
- **GPU hardware floor.** Execution requires compute capability **SM86 or
  SM89** (A40 / RTX 2000 Ada class — the proven, bit-exact targets). Other
  GPUs are refused (`CAPABILITY_UNSUPPORTED`); there is **no CPU fallback**.
- The GPU backend is an accelerator, **not** a consensus identity: outputs
  must match the frozen CPU reference bit-for-bit or the worker does not run.

## Economics

- **Test tokens have no monetary value.** Zero mint inflation; issuance is
  disabled until a separate, audited base-task schedule exists. Bonds, fees
  and slashes are test parameters.
- Faucet limits (cooldown, per-IP, anti-loop) are onboarding guardrails, not
  sybil resistance.

## Network / topology

- The Job API and the scheduler are **centralized by design** for the alpha;
  protocol verification and settlement never depend on trusting them. The
  watcher and the DA providers work against chain + DA only.
- Single-proposer censorship is observable and bounded by the dispute
  machinery, but **a validator cartel can still censor** (recorded since
  Phase E).
- The public RPC proxy serves a **read-only** allow-list, no websockets, no
  TLS termination of its own (put it behind your ingress), and rate limits
  keyed by connection address (it does not trust `X-Forwarded-For`).
- DA is **replicated full-output storage with bonded attestations** (2-of-3),
  not erasure coding / DAS / KZG.
- A validator down long enough is jailed and slashed by the staking module
  (expected chain behaviour, not a bug).

## Productization state

- prismad release binaries are reproducible from source; **published
  container images / GitHub releases** require registry credentials (B7/M6
  scope, pending).
- The live dispute drill, the GPU-backed product gate, and multi-host private
  testnet runs are the remaining acceptance items (Phase C) — the components
  are implemented and unit/integration tested; the end-to-end claims are
  withheld until they run.
- Worker transactions use a temporary 0700 keyring per transaction (bounded
  exposure); a native signer is future work.

## Out of scope entirely

Mainnet, real token sales, real fiat billing, governance, PoUW consensus.
