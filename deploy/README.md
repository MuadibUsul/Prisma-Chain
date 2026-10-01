# Local Prisma devnet

This Compose stack runs one local Cosmos validator, a signed compute gateway,
one worker sidecar, and a deterministic mock model. It exercises discovery,
probing, routing, lease fencing, signed delivery, and idempotent replay. It does
**not** run Qwen, span multiple GPU machines, authorize tasks against chain
escrow, or pay workers. Mock requests are explicitly unfunded. The machine
needs Docker Compose and Python 3.10+; no GPU is needed.

From the repository root:

```sh
python -m pip install -e ./network
python deploy/init_devnet.py
docker compose -f deploy/compose.yaml up --build -d
python deploy/smoke.py
```

For a compute-only check while the chain image is being built:

```sh
docker compose -f deploy/compose.yaml up -d --build gateway mock-model worker
python deploy/smoke.py --no-chain
```

`init_devnet.py` creates random Ed25519 identities, a gateway API key, a
trusted-key file, and a pinned **mock** model bundle in `deploy/.env` and
`deploy/data/`. These files are local secrets and are ignored by Git. The
initializer refuses to overwrite them. Keep them on this machine; generated
keys are not linked to chain worker bonds. All public ports bind loopback.

Check the live services:

```sh
docker compose -f deploy/compose.yaml ps
docker compose -f deploy/compose.yaml logs --tail=100 chain gateway worker mock-model
```

The local endpoints are chain RPC `http://127.0.0.1:26657`, chain gRPC
`127.0.0.1:9090`, and gateway `http://127.0.0.1:8080`. The gateway's
`GET /v1/privacy` discloses the Tier 0 limits. The smoke script posts a
temporary task to `POST /v1/inference`, checks a signed provisional receipt,
then repeats the same request to confirm one delivery record. Authenticated
`GET /v1/tasks/{task_id}` reads chain status for numeric task IDs and labels
mock task IDs `unfunded_dev`. The chain and gateway run side by side; the mock
inference request is not submitted to chain state.

Run the separate chain acceptance check after startup:

```sh
python deploy/chain_smoke.py
```

It waits for successive blocks, creates a local test recipient if needed,
queries both balances, signs a fixed 1 PRSM bank transfer using the local
validator key, waits for inclusion, and verifies the exact balance changes.
Each run sends another 1 test PRSM.

Fund any other Prisma account with exactly 1 test PRSM:

```sh
docker compose -f deploy/compose.yaml exec chain prismad keys add recipient --keyring-backend test --keyring-dir /data --home /data --no-backup
python deploy/faucet.py <recipient-prsm1-address>
```

The faucet checks the loopback RPC chain ID and uses the local validator's
test keyring. Neither command can select another chain or an arbitrary amount;
these tokens have no real monetary value.

Run the chain compute protocol acceptance check:

```sh
python deploy/compute_smoke.py
python deploy/compute_smoke.py --scenario honest
python deploy/compute_smoke.py --scenario fraud
python deploy/compute_smoke.py --scenario fraud-race --check-gateway
python deploy/compute_smoke.py --scenario honest --check-gateway
python deploy/compute_smoke.py --scenario fraud --check-gateway
```

The default scenario registers a bounded public VM model, bonds a separate
worker account, posts a verifiable task, accepts it, and checks the model,
worker, task, reserved bond, and module escrow through chain queries. Its
accepted task and escrow remain on the devnet. The `honest` scenario runs the
VM, submits a result, has two separately bonded accounts replay and attest,
waits out the challenge window, and verifies settlement, fee split, 20% burn,
bond release, and duplicate-payment rejection. The `fraud` scenario submits
an incorrect two-step trace, independently recomputes the canonical trace,
plays the midpoint dispute on-chain, and verifies requester refund, worker
slash, challenger bond return, and duplicate-payment rejection. The
`fraud-race` scenario places a false challenge first and a valid challenge
in the bounded queue. It verifies that the false challenger loses its bond,
the valid challenger is promoted automatically, and the fraudulent result is
refunded rather than paid. These are
scripted local checks using the same VM implementation, not a continuously
running independent monitor or a proof against common-mode VM bugs. Each run
creates a new task and uses test PRSM only.

With `--check-gateway`, the script also checks the authenticated API's
`accepted`, `pending`, `challenged` (fraud path), and final `settled` or
`refunded` chain views. Start the
full Compose stack before using that flag.

To stop processes while preserving local chain state and receipts:

```sh
docker compose -f deploy/compose.yaml down
```

The chain data and gateway SQLite database live in named Docker volumes.
`docker compose -f deploy/compose.yaml down -v` erases those volumes. Keep the
identity files in `deploy/.env` and `deploy/data/` with their corresponding
gateway volume; replacing keys while retaining an old receipt database may
leave historical evidence signed by no-longer-trusted keys.

## Real model path

The mock model is a contract test only. Real Qwen3-14B and Qwen3-32B serving
needs separately provisioned GPU hosts with pinned model artifacts, a shared
Ray/vLLM deployment, stage sidecars, TLS, funded-task authorization, and a
shared durable lease store. The network package's
[`examples/launch_vllm_ray.sh`](../network/examples/launch_vllm_ray.sh)
describes the serving experiment. Do not point a public client at the local
HTTP/dev-unfunded gateway or interpret this stack as a multi-host benchmark.
