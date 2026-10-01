# Testnet delivery criteria

This repository is a greenfield implementation of the Prisma Chain testnet.
A source file or mock demonstration is not evidence that a live testnet or an
audited security guarantee exists. The following are release gates.

## Software gates

1. A reproducible Cosmos SDK / CometBFT binary starts from documented genesis
   with PRSM transfer, staking, governance, upgrade and compute modules.
2. The deterministic VM and interactive dispute tests cover honest output,
   dishonest output, unavailable data, invalid proof, false challenge and
   response timeout. Independent watchers can replay public tasks.
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
