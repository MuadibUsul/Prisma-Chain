# Local Prisma devnet

This Compose stack runs one local Cosmos validator, a signed compute gateway,
one worker sidecar, and a deterministic mock model. It exercises discovery,
probing, routing, lease fencing, signed delivery, worker failure retry, and
idempotent replay. It does
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
then repeats the same request to confirm one delivery record. It also injects
one backend failure for a second mock task, retries the same task ID, and
checks that only the recovered attempt produces a receipt. Authenticated
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

Run the separate lightweight settlement check against the live local chain:

```sh
python deploy/light_smoke.py
```

It creates distinct bonded worker, gateway, and monitor accounts, posts a
funded synthetic task, rejects a hash-only result, submits a canonical
worker/gateway signed receipt, then verifies metered charge, unused escrow
refund, burn, payout, final API status, and duplicate-settlement rejection.
This tests chain accounting and signatures; it does not run a GPU model or
prove that the two signers belong to independent operators.

Run the funded API integration check:

```sh
python -m pip install -e './network[chain,model]'
python deploy/funded_api_smoke.py
```

It creates a fresh local chain task and bonded identities, uses the actual
chain gRPC authorization query, sends a signed request through the gateway and
worker FastAPI applications, injects one worker failure, retries under the
same task ID, then submits the gateway receipt through the local CLI. It
checks the API's accepted, pending and settled views, including proposed
charge and final refund. Its model output and tokenizer are synthetic and
in-process; this is not a Qwen or multi-host benchmark. Each run uses new
test PRSM accounts and a new escrow task.

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

The 2026-10-01 RunPod preflight reached two live GPUs and completed CUDA
matrix operations on each: one RTX A4500 with 20 GiB in Europe and one RTX
A5000 with 24 GiB in Canada. A TCP connection from the first Pod to the
second Pod's exposed SSH port succeeded; ICMP ping received no replies.
Each Pod had a 30 GiB ephemeral container disk, no cached model weights,
and no Ray/vLLM executable. This establishes hardware and basic TCP access
only. The Qwen3-14B two-stage inference acceptance remains pending. Before
that run, provision adequate persistent model/cache storage, a pinned
Ray/vLLM environment, authenticated worker access, and the actual Ray data
ports or private networking. Measure the cross-region latency against a
same-region pair before using it as a performance baseline.
