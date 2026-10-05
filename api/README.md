# prisma-api

The developer-facing Job API + scheduler (B6).

```text
POST   /v1/jobs        submit (frozen supported profiles only; Idempotency-Key)
GET    /v1/jobs/{id}   status + result references (B6-06 presentation)
GET    /v1/jobs        cursor-paginated list
DELETE /v1/jobs/{id}   cancel (pre-accept only; 409 job_not_cancellable later)
GET    /v1/profiles    enumerate the frozen supported profiles
```

- **B6-01** jobs.py: schema, profile gate, idempotency, pagination, cancel
  semantics, stable machine-readable errors (every failure is
  `{"error": {"code", "message", "details"?}}`).
- **B6-02** auth.py: bearer API keys (`prsk_…`), sha256-at-rest, shown once at
  issuance, `jobs:read` / `jobs:write` scopes, immediate revocation,
  constant-time checks, 401/403 in the same error schema.
- **B6-03/04/05** scheduler.py: worker registry (capability + health),
  FIFO matching over compatible healthy workers, and stage-aware
  reassignment — pending/assigned/accepted/running are reassignable on
  failure (the failed worker leaves the rotation until re-registered);
  committed-and-later are NEVER reassigned (the chain's dispute path owns
  them). No auction, no bidding — frozen scope.
- **B6-06** presentation.py: one job view across the chain states with the
  documented status mapping (unknown chain states stay `unknown`), and
  verification status as a **passthrough of what the chain confirmed** —
  never inferred ahead.

The core is framework-free (`handle_request`); a thin FastAPI/uvicorn wrapper
lands with the deployment tasks. First release supports **only** the frozen
profile `qwen3-0.6b-layer0-v2`; the API never claims arbitrary model support.
