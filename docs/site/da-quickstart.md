# DA Quickstart (storage provider)

A DA provider stores canonical result blobs, verifies them **before**
attesting, serves bulk retrieval and typed tile proofs, and answers the
chain's sampling challenges inside their deadlines.

```bash
pip install -e ./network -e ./worker -e ./da
prisma-da init --chain-id prisma-testnet-1 --node http://rpc.testnet.example:26657
prisma-da register            # registration transaction (account must be funded)
prisma-da run                 # serve /health /metrics /store /attest /v1/da/...
```

Storage guarantees (enforced in code):

- a blob whose recomputed `output_root` does not match is **refused**;
- writes are atomic (stage + fsync + rename) — a crash never leaves a half
  artifact;
- every served byte is re-hashed; a mismatch quarantines the artifact and
  refuses to serve it;
- GC deletes only after the TTL **and** only with no live challenge;
- `prisma-da scan` re-verifies everything on demand.

Metrics live at `/metrics`; health at `/health`. The daemon survives restarts:
it re-scans its data directory at start-up and serves exactly what still
verifies.

