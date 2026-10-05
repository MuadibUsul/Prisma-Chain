# API reference — Job API v1

Base URL: the deployed Job API (Phase C). All errors share one schema:

```json
{"error": {"code": "unsupported_profile", "message": "…", "details": {"supported": ["qwen3-0.6b-layer0-v2"]}}}
```

Stable codes: `invalid_request`, `unsupported_profile`, `unauthorized`,
`forbidden`, `job_not_found`, `job_not_cancellable`, `duplicate_job`,
`rate_limited`, `internal`.

Authentication: `Authorization: Bearer prsk_…` (issued out of process; shown
once). Scopes: `jobs:read`, `jobs:write`.

## POST /v1/jobs

Submit a job. Only the frozen supported profile is served — the API never
claims arbitrary model support.

```json
{"profile": "qwen3-0.6b-layer0-v2", "input": {"tokens": [1, 2, 3]}}
```

- `202` → `{"job_id", "status": "pending", "task_id": …}`
- send `Idempotency-Key: <client key>`: a retry with the same key returns
  `200` + the original job (`"idempotent_replay": true`).
- `400 unsupported_profile` for anything not in `/v1/profiles`.

## GET /v1/jobs/{id}

The full job view (see [Job lifecycle](job-lifecycle.md) for the states):

```json
{
  "job_id": "job-…", "task_id": 42, "status": "finalized",
  "worker": "prsm1…", "graph_id": "8fb86087…", "final_root": "…",
  "verification": {"confirmed": true, "receipt_id": "vwr-…", "verdict": "pass"},
  "settlement_tx": "…", "vwr_id": "vwr-…"
}
```

`verification` is exactly what the chain confirmed — it is never inferred.
Before settlement it is `{"confirmed": false, "note": "…"}`.

## GET /v1/jobs

Cursor pagination: `GET /v1/jobs?cursor=<next_cursor>&limit=50` →
`{"jobs": […], "next_cursor": "…"}`. `limit` is 1–200.

## DELETE /v1/jobs/{id}

Cancel — only while the job is `pending` and nothing was accepted on chain
(`200 {"cancelled": true}`). Anything later: `409 job_not_cancellable`.

## GET /v1/profiles

```json
{"profiles": [{"name": "qwen3-0.6b-layer0-v2", "spec_version": "v1",
               "mode": "lightweight", "graph_id_v2": "8fb86087…", "policy_id": "eb9a9fef…",
               "description": "…"}]}
```

## Known limitations

See [testnet-known-limitations](testnet-known-limitations.md). Highlights:
one frozen profile, testnet economics (zero inflation, test tokens worthless),
the Job API and scheduler are centralized by design (verification/settlement
is not), and the HTTP wrapper deployment is Phase C infrastructure.
