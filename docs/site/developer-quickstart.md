# Developer Quickstart

Everything here runs against a testnet. Test tokens have no monetary value.

## 1. Install

```bash
pip install -e ./network -e ./worker -e ./cli
```

## 2. Configure the CLI

```bash
prisma config set --chain-id prisma-testnet-1 \
                   --node http://rpc.testnet.example:26657 \
                   --api https://api.testnet.example \
                   --faucet https://faucet.testnet.example
prisma version          # shows the frozen protocol identity the release implements
```

## 3. Get test tokens (faucet)

```bash
prisma wallet init
prisma faucet --address $(prisma wallet show --json | python -c "import sys,json;print(json.load(sys.stdin)['account_address'])")
```

## 4. Submit a job

Only the frozen supported profile is served (this is enforced server-side):

```bash
prisma job post --profile qwen3-0.6b-layer0-v2 \
                --input '{"tokens": [1,2,3]}' \
                --idempotency-key my-first-job \
                --json
```

An `Idempotency-Key` (or `--idempotency-key`) makes client retries safe:
the same key returns the same job.

## 5. Follow the job

```bash
prisma job get --job-id <id>          # API view (B6 presentation)
prisma task --task-id <chain task>    # chain view
prisma da --task-id <chain task>      # availability status
prisma dispute --task-id <chain task> # dispute status, if challenged
prisma receipt --receipt-id <vwr>     # the Verified Work Receipt
```

The job reaches `finalized` only when the chain has settled it with a VWR.
See [Job lifecycle](job-lifecycle.md).
