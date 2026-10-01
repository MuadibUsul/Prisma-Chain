# Prisma task protocol (testnet draft)

The consensus chain and the compute network have different jobs. CometBFT orders
deterministic state transitions. GPU inference, routing, monitoring and model
training happen outside consensus. A computation result is not a block vote.

## Roles

- **User** escrows PRSM and specifies a model, verification mode, fee and deadline.
- **Worker** bonds PRSM, runs a task and signs its result commitment.
- **Watcher** independently checks verifiable work and may challenge it.
- **Coordinator** selects an eligible compute group and relays signed messages;
  it has no unilateral authority to release escrow or slash a worker.
- **Validator** runs the chain state machine and checks bounded dispute steps.

Consensus stake and worker bond are separate accounts and penalty rules. A
worker's incorrect result cannot slash delegated consensus stake.

## Task envelope

Every task binds these fields before a worker accepts it:

| Field | Meaning |
| --- | --- |
| `task_id` | Chain-assigned stable identifier |
| `mode` | `verifiable` or `lightweight` |
| `model_id` | Registry entry with pinned weights, tokenizer and image digests |
| `spec_version` | Exact execution and accounting rules |
| `input_commitment` | Digest of canonical input bytes or an encrypted payload |
| `data_ref` | Retrieval location; it is not by itself an availability guarantee |
| `max_fee` | Escrowed upper bound in testnet PRSM |
| `deadline` | Last acceptable submission height |
| `privacy_tier` | `public` or `tier0_relative` |

The chain records a `task_id`; the off-chain gateway records an `attempt_id`
for each delivery attempt. A result is provisional until its mode-specific
acceptance condition is met. Finalization and refund are mutually exclusive.
A retry can change the attempt, never the task, and only one accepted task
receives payment.

## Verifiable mode

Only immutable, public, bounded integer programs are admitted. Program bytes,
weights and canonical input must remain retrievable throughout the challenge
window. Workers commit to output and execution trace. A watcher actually
replays the entire program before disputing a result. The interactive game
narrows a mismatch to one instruction; the chain executes that instruction
with a size-bounded witness. A missing input, missing model, timeout or
unanswered dispute cannot be finalized as an unchallenged correct result.

The logarithmic number of dispute rounds reduces on-chain work. It does not
remove the watcher's full replay cost. Any account may challenge; randomly
assigned watchers are a monitoring service, not the sole challenge gate.
The current chain admits one active dispute per task. A losing challenger's
bond is burned and a full challenge window restarts before the task can be
paid. This removes the zero-cost worker-controlled false-challenge loop, but
it does not solve concurrent challenger admission or transaction censorship;
those remain permissionless-release gates.

## Lightweight mode

The worker and gateway sign a delivery receipt, including the pinned model
version, attempt ID, output commitment, delivered token count and timestamp.
Independent re-execution, canary tasks and user feedback update reputation.
Subjective answer quality is not an objective slashing condition. The client
must see the mode before payment and must not receive a fraud-proof claim for
an LLM result that has not been proved.

## Tier 0 privacy

The chain stores commitments and settlement data, not private prompts. Inputs
are encrypted in transit and at rest. The ingress may see the entire prompt;
the egress may see the entire answer; intermediate activations can leak
information. Model-layer or activation sharding is not cryptographic secret
sharing. `tier0_relative` therefore never means that no worker can see the
input, and it must not be offered as a strong confidentiality guarantee.

## Settlement

External task escrow is refunded on accepted failure or paid once on success.
Successful external orders burn 20% of the charged amount. The remaining 80%
pays the work and verification budget under versioned chain parameters.
Base-engine issuance and genesis locked-reserve release use distinct ledgers.
Tests must assert the supply equation after every transition:

`end_supply = start_supply + newly_minted - burned`.
