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
key. A small Python client now wraps this existing API and status polling; it
still requires the caller to obtain an accepted chain task envelope. A separate
funded API smoke test connects the gateway and worker to the
local chain: it verifies an accepted escrowed task, independently counts
delivered text with a hash-pinned tokenizer, checks a signed receipt, retries a
worker failure under the same task ID and observes pending and settled billing.
The test uses a synthetic model. After one injected receipt submission failure,
the gateway retries the saved receipt in the background through the worker's
injected devnet CLI signer, without a second client request; the script still
submits monitor and settlement transactions manually.
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
An optional chain query admits signed announcements from bonded worker keys
without a pre-provisioned key or account binding; the first pairing remains
fixed across gateway restarts. An additional opt-in public-node transport accepts
bonded workers without a URL host allowlist and pins each outbound HTTPS request
to a freshly checked public DNS answer. The deployed mock stack still uses
pre-provisioned keys, VRAM is advertised but not independently measured, and
the open operator test has not been completed. Open participation and formal
source release remain future gates.

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

## Verifiable GEMM gate

GEMM_INT8_V1 (docs/gemm-protocol-v0.1.1.md) is a second verifiable operator
next to the bounded-integer VM. The gate is complete only when the
following hold against the real implementation, not a plan:

1. The Go reference and an independent implementation produce the identical
   task ID, matrix roots, output root and receipt ID for fixed vectors
   (cross-language determinism).
2. A normal worker reaches FINALIZED and holds a Verified Work Receipt with
   `verification_mode = optimistic_unchallenged`.
3. A single corrupted output tile is detected by a challenger that never
   received the worker's intermediates.
4. The resulting dispute creates an execution trace for the disputed tile
   only; no full-task trace is ever committed or generated on the normal
   path, including at 4096x4096x4096.
5. Interactive bisection localizes the first differing K-step, and the
   arbiter resolves the dispute by recomputing exactly one 8x8x8 micro-step
   (512 canonical MACs) after checking both input tiles against the
   committed matrix roots.
6. A worker that submitted a wrong tile receives no receipt; a challenger
   that submitted a wrong tile cannot defeat a correct worker; replays of
   ResultCommits, assignments and tile proofs across tasks or workers are
   rejected; response timeouts resolve by the existing refund rules.
7. Two independent GPU nodes complete the honest and fraud E2E scenarios
   over the network with a bit-exact INT8 x INT8 -> INT32 backend, and the
   measured benchmark ladder (128 through 4096) is published with the
   normal-path overhead and the challenge overhead against full
   recomputation.

Phases A and B (library, CLI, local two-process E2E) and the two-GPU
RunPod E2E of item 7 have passed: two pods with different GPU models
(RTX 4000 Ada, RTX 2000 Ada) both pass the `torch._int_mm` bit-exactness
gate, and the honest and fraud scenarios complete across them
(`docs/gemm-e2e-results.json`). The GPU benchmark ladder and the Phase D
chain integration (GEMM task spec, dispute state and VWR settlement in the
compute module) remain open; the gate is not fully met until Phase D lands
and the measured GPU ladder is published.

## On-chain GEMM settlement gate

Phase D (docs/gemm-phase-d-report.md) connects the verified GEMM path to
chain state and settlement. The gate is met only when the following hold
against the real implementation:

1. A GEMM task posts with escrow and a bonded worker accepts it.
2. The canonical ResultCommit signature verifies on chain under the
   bonded network key, with a chain-derived canonical MAC count.
3. No challenge settles exactly once and mints exactly one canonical VWR.
4. A challenged task locks traces for the disputed tile only, persists
   bisection across restarts and adjudicates with one 512-MAC micro-step.
5. A false challenger cannot defeat a correct worker; the worker still
   settles (challenged_worker_won).
6. Replay, queue griefing, oversized proofs and duplicate settlement are
   rejected; escrow/bond/burn invariants hold.
7. The bounded-VM and lightweight suites remain green.

Status: items 1-7 pass in the keeper integration suite and in the local
devnet E2E (`deploy/gemm_chain_smoke.py`, honest / fraud /
false-challenge; evidence in docs/gemm-phase-d-report.md). Multi-validator
economic drills, permissionless data availability and receipt light-proofs
remain open; the gate is met only for the single-validator devnet scope it
was tested at and must not be described beyond it.

## Cheap verification gate

v0.1.2 (docs/gemm-verification-v0.1.2.md) adds Freivalds detection so the
challenger no longer needs a full recomputation to find fraud. The gate is
met only when all of the following hold against the real implementation:

1. An honest C passes every detection round.
2. A fraudulent C is detected without full recomputation.
3. The fast path never calls the full reference GEMM (asserted by
   instrumentation in the E2E).
4. The bad output row is localized and confirmed by an exact O(KN) row
   recomputation.
5. The bad 8x8 tile is derived from exact row recomputations, never from
   worker data.
6. The existing v0.1.1 dispute proves the fraud end to end and ends
   ChallengerWins with no receipt for the worker.
7. The final arbiter remains exactly 512 canonical MACs.
8. The 4096^3 benchmark is published with DetectionRatio and
   TotalFraudPathRatio measured on one machine, separating probabilistic
   detection cost from deterministic arbitration cost.
9. A false-accept simulation is published next to the 2^-rounds
   theoretical bound, with PASS / FAIL / NOT TESTED kept distinct.

Status: items 1-7 and 9 pass on the CPU path (`docs/gemm-v0.1.2-report.md`);
the 4096^3 CPU ladder is published (`docs/gemm-v0.1.2-benchmark-results.json`,
detection ratio 0.0080 at 8 rounds, total fraud-path ratio 0.0692 at 40
rounds including the exact 8-row tile build). The GPU fast-verification benchmark remains NOT TESTED until two
pods are available again. Phase D chain integration is still open.
