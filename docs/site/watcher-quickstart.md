# Watcher Quickstart

A watcher verifies completed work independently. It needs no GPU: the frozen
verifier (root checks, 40-round Freivalds **detection**, cheap exact
recomputation) runs on CPU.

```bash
pip install -e ./network -e ./worker -e ./watcher
prisma-watcher init --chain-id prisma-testnet-1 \
                    --node http://rpc.testnet.example:26657 \
                    --provider http://da1.testnet.example:8401 \
                    --provider http://da2.testnet.example:8402
prisma-watcher run               # discover -> retrieve -> verify -> journal
```

- Fraud detection is **detection only**: the watcher opens an evidence-gated
  challenge (`prisma-watcher challenge --task-id N`), and the chain
  adjudicates deterministically. The watcher never slashes anything by itself.
- Every task is journalled; a restart resumes instead of re-verifying, and a
  running dispute is never lost (`disputes` ledger).
- Status: `prisma-watcher status`.
