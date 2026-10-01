# Prisma Chain testnet

Prisma's long-term goal is to turn independently owned idle compute into an
open AI service that developers can access through one API, reducing dependence
on a small number of large proprietary platforms. Worker admission, model
eligibility, service quality and privacy must be measured rather than assumed.

This repository implements the Prisma Chain whitepaper as an **application
chain plus off-chain compute network**. PoS/BFT orders transactions; useful AI
work is rewarded by the task protocol and does not choose blocks. The source
concept paper is [Prisma Chain v0.1](https://www.annulus.us/notes/prisma-chain-whitepaper).
The [Chinese whitepaper v0.2](docs/whitepaper-v0.2.zh-CN.md) states the revised
protocol, current evidence and testnet delivery gates.
The final developer-facing deliverable is a unified Prisma API and SDK for
inference and asynchronous tasks; its measurable release gate is in
[docs/delivery.md](docs/delivery.md).

This is development software. A passing local test is not an audited chain,
an economically secured network, or evidence of multi-machine GPU inference.
The [delivery gates](docs/delivery.md) define those distinct outcomes.

Prisma is an independent Cosmos SDK / CometBFT application chain with its own
`prisma-local-1` development chain ID and `uprsm` unit. It is not Cosmos Hub and
does not use ATOM for gas or task escrow. The current Compose network has one
local validator; permissionless public testnet operation is a later delivery
gate.

## Components

| Path | Role |
| --- | --- |
| `vm/` | Bounded deterministic integer VM, execution commitments and dispute game |
| `cmd/prisma-vm/` | Local public-task runner and Merkle-proof export |
| `chain/` | Cosmos SDK / CometBFT application and PRSM compute protocol |
| `network/` | Signed node discovery, measured routing, coordinator fencing and GPU sidecars |
| `tools/economics.py` | Testnet supply and watcher-budget stress test |
| `docs/` | Protocol, threat model, delivery criteria and economic assumptions |

## Run the parts that need no GPU

```sh
go test ./...
go run ./cmd/prisma-vm -job examples/public-classifier.json -proof 3
go run ./cmd/prisma-vm -job examples/public-classifier.json -claim > worker-result.json
go run ./cmd/prisma-vm -job examples/public-classifier.json -review worker-result.json
python -m unittest discover -s tools -p 'test_*.py'
python tools/economics.py tools/scenarios.json
```

The review command independently replays every VM step before it reports
`attest`, `challenge`, or `withhold`. It exports a challenger endpoint claim
when an incorrect final result can enter the bisection game. This is an
offline monitor tool; it does not sign or submit a chain transaction.

For the network package:

```sh
python -m pip install -e './network[test]'
python -m pytest -q network/tests
```

The chain has its own Go module and toolchain requirement; see `chain/` for
its build and local-node instructions. Its code and integration tests must be
verified before a chain or payment feature is claimed operational.

## Local development network

The reproducible, CPU-only stack starts one chain validator, a gateway, a
worker and a fixed-output mock model:

```sh
python -m pip install -e ./network
python deploy/init_devnet.py
docker compose -f deploy/compose.yaml up --build -d
python deploy/smoke.py
python deploy/chain_smoke.py
```

The first smoke test verifies blocks and a signed but unfunded mock delivery.
The second verifies a real signed bank transfer and exact balance changes. See
[the local runbook](deploy/README.md) for setup, test-token funding and volume
handling. The mock model is not Qwen; no GPU hosts are currently available.

## Delivery status

| Area | Current evidence | Open gate |
| --- | --- | --- |
| Bounded VM | Local execution, trace proofs, independent replay and dispute tests pass | Broader adversarial review and audit |
| Chain | One local validator, signed bank transfer and a verifiable task settled with 20% burn and duplicate-payment rejection | Independent validators, public genesis and governance rehearsal |
| Compute network | Signed discovery, measured routing, lease fencing and provisional mock receipts pass locally | Multi-host Qwen serving, shared leases, checkpoint recovery and funded-task integration |
| Lightweight payment | Escrow can be refunded; automatic finalization is disabled | Verified gateway and worker receipts, pinned token metering and tariff rules |
| Unified API | Local mock inference and receipt endpoints; caller supplies a task envelope, but development mode does not verify chain acceptance or funding | Quoting, streaming multi-host Qwen, asynchronous jobs, finality status, billing and external-client acceptance |
| Open participation and source | Publicly readable repository; workers require pre-provisioned trust bindings; no `LICENSE` file yet | Explicit software license, independent gateway deployment, published worker admission and measured useful capacity |
| Economics and operations | Scenario simulator and deployment configuration exist | Issuance ledger, training pool, concentration controls, benchmarks and independent audit |

These gates correspond to the [delivery criteria](docs/delivery.md). The
local signed task test exercises a small public integer program, not a Qwen
request.

## Security modes

- `verifiable`: bounded public integer programs, full watcher replay and
  interactive single-step dispute. A Merkle root alone is not a proof.
- `lightweight`: signed LLM delivery evidence and sampled service checks.
  Subjective model quality is not fraud-proofed and cannot be auto-slashed.
- `tier0_relative`: target for encrypted transport/storage and workload
  partitioning. The local devnet has none of those protections. Ingress may
  see the prompt and egress may see the answer; this is not cryptographic
  secret sharing.

The [task protocol](docs/protocol.md) and [threat model](docs/threat-model.md)
describe the exact promises and failure handling. A testnet PRSM faucet token
has no real monetary value.
