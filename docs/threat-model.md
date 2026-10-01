# Testnet threat model

## Assets and trust boundaries

The assets are task escrow, worker bonds, model identities, public execution
data, private task payloads and the chain's consensus state. Validators agree
on **task state**, not on an AI answer by running a GPU inside consensus.

The chain may trust CometBFT safety only within its configured voting-power
assumption. Worker bonds do not provide consensus voting power, and consensus
stake is not exposed to compute-result disputes.

| Adversary | Desired failure | Required defense / test |
| --- | --- | --- |
| Worker | Submit a false result or a different model | Pin model and program hashes; watcher replay and bounded dispute for verifiable tasks |
| Worker | Withhold execution data | Data-availability gate, timeout, refund and worker penalty |
| Watcher | File a false challenge or withhold responses | Challenge bond and symmetric on-chain response deadlines |
| Workers and watchers together | Let a false result pass | Permissionless challenge entry and independent audit operators; measure concentration |
| Coordinator | Censor tasks, double route, or replay an old lease | User direct chain submission, signed job/attempt IDs, monotonic fencing epoch |
| Storage provider | Delete model or task data during challenge | Immutable public artifacts with availability checks; do not finalize missing-data jobs |
| Validator subset | Censor dispute transactions or halt chain | Independent operators, stake distribution, partition and fault tests |
| User | Submit malformed inputs or dispute subjective quality | Canonical bounded public inputs; no automatic quality slashing in lightweight mode |

`BondWorker` now requires an Ed25519 possession proof whenever it registers a
network public key. The signed v1 payload binds the chain ID, worker account
and raw public key; the chain records a proof marker and hides unproven legacy
keys from worker queries and lightweight receipt verification. Bond top-ups
without a network key remain possible. This does not establish that distinct
accounts belong to distinct operators. Key rotation and a testnet upgrade plan
for previously registered accounts are still required before permissionless
funded operation.

Two different bonded monitor addresses are required for a verifiable payout.
The chain cannot prove that they belong to independent operators. Random
monitor assignment, concentration disclosure and collusion tests remain
release gates.

A losing challenger's bond is burned and the full challenge window restarts,
so a worker cannot use a controlled challenger to recover the bond or shorten
the next review period. One dispute runs at a time, with up to eight bonded
challengers queued in arrival order. An honest challenger admitted during a
false active challenge proceeds automatically when the false challenge loses.
The finite queue can still be saturated by well-funded attackers, and validator
censorship can exclude a challenge transaction. Test both before permissionless
paid work; a rejected challenger must still have the restarted review window
after losing challenges.

## Verification boundary

A Merkle root proves commitment to bytes, not that an AI computation is
correct. A dispute is secure only if a challenger can obtain the program,
weights, input and enough trace material, reproduce the result and reach the
chain before the window closes. The final one-step adjudication requires an
audited deterministic VM. Its gas and witness limits are consensus limits.

The lightweight mode accepts that floating-point LLM execution and answer
quality are not covered by this fraud-proof path. Redundancy, canaries and
reputation are service-quality tools, not cryptographic correctness proofs.

## Tier 0 privacy boundary

TLS or mTLS protects data in transit; encrypted object storage protects it at
rest. Neither hides plaintext from an authorized ingress worker once it
decrypts a prompt. The final stage may see the output. Intermediate
activations and a private model shard can leak information or be combined by
colluding workers. Public-chain metadata remains visible. Tier 0 reduces
concentration of data but does not meet a zero-knowledge, FHE or MPC privacy
claim. Sensitive workloads must be rejected if the user asks for a stronger
guarantee than this mode provides.

## Economics boundary

Testnet PRSM has no external value. Burning testnet units demonstrates correct
accounting, not a sustainable token economy. A worker bond can cap loss of
escrow and specified dispute costs; it cannot cover arbitrary harm to a user
who relies on a wrong model answer. Colluding auditors invalidate any model
that assumes independent sampling. Simulate low demand and monitoring costs
before choosing live-network reward rates.

## Release checks

- Inject a wrong result at the first, middle and final VM instruction and
  prove that each challenge resolves correctly.
- Withhold inputs, model bytes, intermediate witnesses and response messages;
  no such task may finalize as correct.
- Partition a coordinator and replay its old epoch after failover; workers
  must reject it, and escrow must pay at most once.
- Compromise a head worker in Tier 0; documentation and API warnings must
  match the data it can actually observe.
- Run zero-demand and high-demand supply scenarios; report watcher budget
  shortfalls rather than assuming a 20% burn funds security.
