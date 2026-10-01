# Testnet delivery criteria

This repository is a greenfield implementation of the Prisma Chain testnet.
A source file or mock demonstration is not evidence that a live testnet or an
audited security guarantee exists. The following are release gates.

## Software gates

1. A reproducible Cosmos SDK / CometBFT binary starts from documented genesis
   with PRSM transfer, staking, governance, upgrade and compute modules.
2. The deterministic VM and interactive dispute tests cover honest output,
   dishonest output, unavailable data, invalid proof, false challenge and
   response timeout. Independent watchers can replay public tasks. Adversarial
   tests also cover worker-controlled false challengers, challenge admission
   races and repeated attempts to exclude an honest challenger.
3. A client can post, claim, submit, challenge and settle a task end to end.
   Retries never pay more than once; failed jobs retain a refund route.
4. Signed node announcements expire, route based on measured performance and
   reject stale coordinator epochs. A second operator can join without
   changing source code.
5. Real pinned Qwen3-14B runs one request across two physical GPU machines;
   Qwen3-32B runs across four stages. Hardware, latency, throughput and cost
   measurements are published against the same GPUs in one location.
6. Tier 0 input and intermediate-data handling matches the disclosed threat
   model. Strong privacy claims require a separate security review.
7. PRSM burn, escrow, reserve release, new issuance and slashing satisfy
   accounting invariants under adversarial transaction sequences.

## Unified API product gate

The final developer-facing deliverable is one versioned Prisma API and SDK,
implemented by interchangeable gateways. A client can submit inference,
verifiable-compute and public fine-tuning tasks without operating an internal
router or invoking chain commands manually. The gateway does not acquire
unilateral authority over on-chain escrow, disputes or settlement.

- Before execution, the API gives a price quote and fee cap. A documented
  inference call streams real pinned Qwen output from the required multi-host
  GPU group. It returns the task ID, model and spec version, usage,
  provisional-delivery status, and independently verifiable worker and gateway
  signatures. Production usage metering is checked against the pinned tokenizer.
- An asynchronous task interface accepts bounded public compute and public
  fine-tuning jobs. Clients can query delivery, challenge, settlement and refund
  states, with clear separation between a delivered result and final payment.
- Client retry uses an idempotency key and stable task ID. After a disconnected
  stream, the client can query status and signed receipts; replaying output
  requires encrypted, time-limited retention or a client-held checkpoint with
  documented deletion rules. Coordinator or worker failure cannot create a
  second charge or silently discard an escrowed task.
- Documented managed-billing and self-custody paths disclose who holds funds,
  who signs chain transactions and how a client verifies the final chain state.
  Both paths enforce per-tenant authentication, quotas and the disclosed privacy
  boundary.
- An external client passes an end-to-end test against independently operated
  gateways and workers, including a real Qwen request, a successful verifiable
  task, a valid challenge, timeout/refund and a worker-failure retry. The test
  records latency, completion, cost and settlement time. The API specification
  and SDK are public before this gate is marked complete.

The current local gateway exposes test-only inference and receipt endpoints
using a CPU mock model, plus a read-only chain task status endpoint. The caller
supplies a task envelope. Unfunded development mode uses one development API
key. A separate funded API smoke test connects the gateway and worker to the
local chain: it verifies an accepted escrowed task, independently counts
delivered text with a hash-pinned tokenizer, checks a signed receipt, retries a
worker failure under the same task ID and observes pending and settled billing.
The test uses a synthetic model. After one injected receipt submission failure,
the gateway retries the saved receipt through the worker's injected devnet CLI
signer; the script still submits monitor and settlement transactions manually.
The packaged worker can now use an isolated development `prismad` test keyring;
a separate live-container smoke check proves its account-bound submission,
duplicate handling, monitor attestations, settlement and gateway status. That
check assembles a synthetic signed receipt instead of delivering an answer via
the Compose gateway. Production key management and real Qwen streaming remain
open. The unified API product gate remains open.

## Open participation and source release gate

The mission is to aggregate **eligible, independently owned idle compute** and
give developers a portable alternative to a small set of proprietary AI
platforms. Progress is measured by completed work, usable capacity, cost,
reliability, privacy and operator concentration, rather than registered nodes.

- Publish explicit reuse licenses for the chain, VM, gateway, worker and SDK;
  publish protocol specifications and reproducible build instructions. Identify
  the separate licenses governing model weights, training data and dependencies.
  The model catalog includes at least one pinned model whose weights can be
  obtained and served independently under its published license.
- An independent operator can install the released worker software, bind its
  signing key to a bonded chain account, pass measured hardware and network
  admission, and serve a compatible workload under published rules. No
  project-specific source change or hidden operator allowlist is required.
- An independent operator can deploy a compatible gateway with the same public
  API and independently verify receipts and chain finality. The product remains
  usable when the founding gateway or its discovery/control service is
  unavailable.
- Report completed jobs and effective capacity by hardware class, non-genesis
  worker share, operator concentration, completion and retry rates, P50/P95
  latency, cost per delivered unit against matched centralized hardware, and
  privacy incidents or audit findings. Explain the measurement method and its
  trust limits.

The repository is publicly readable but currently has no `LICENSE` file.
Worker keys, allowed hosts and chain-account bindings are pre-provisioned; VRAM
is advertised but not used for admission. Open participation and formal source
release remain future gates.

## Operational gates

- At least seven validators across independently operated nodes run the
  network. Voting-power distribution and genesis-operator allocation are
  published.
- Kill workers and coordinators during acceptance, streaming and settlement;
  prove that no task is lost silently and no task is paid twice.
- Exercise Byzantine validators, partitions, withheld data, colluding workers
  and watchers, model-version mismatch and governance upgrades.
- Publish independent code and protocol audits before permissionless paid
  work. A testnet faucet token is not evidence of real economic security.

The repository README tracks which gates are actually met. Unmet gates remain
open work; they are never described as shipped behavior.
