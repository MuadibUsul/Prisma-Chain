# Status page data (B7-03)

The status page is **generated from this file + the network's health
endpoints** — no hand-maintained numbers. Deployment (Phase C) wires a static
generator that reads this manifest and the live endpoints below; the page
renders what the endpoints answer, or "unknown", and never invents a value.

```json
{
  "generated_from": "docs/site/status.md (B7-03 manifest) + live endpoints",
  "network": {
    "chain_id": "from /status of any RPC endpoint",
    "height": "from /status",
    "validators": "from /validators"
  },
  "endpoints": {
    "rpc": "see docs/productization network section (Phase C publishes them)",
    "job_api": "GET /v1/profiles (B6)",
    "faucet": "POST /fund (B7-01)",
    "da": "GET /health of the registered providers"
  },
  "services": {
    "node": {"component": "prismad", "health": "prismad health (bounded RPC /status check)"},
    "rpc_proxy": {"component": "prismad rpc-proxy", "health": "GET /healthz"},
    "worker": {"component": "prisma-worker", "health": "state file + heartbeat"},
    "da": {"component": "prisma-da", "health": "GET /health + /metrics"},
    "watcher": {"component": "prisma-watcher", "health": "prisma-watcher status"},
    "api": {"component": "prisma-api", "health": "GET /v1/profiles"}
  },
  "rules": [
    "every value comes from a live endpoint or 'unknown' — never hand-typed",
    "test tokens have no monetary value (footer on every render)",
    "verification/settlement numbers come from the chain only"
  ]
}
```

The rendered page shows: chain id, height, validator count, the five
services' up/down (from their health endpoints), the supported profile list
(from `/v1/profiles`), and the standing notice that test tokens have no
monetary value.
