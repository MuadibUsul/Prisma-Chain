# Troubleshooting

## Worker

| symptom | meaning | fix |
|---|---|---|
| `CAPABILITY_UNSUPPORTED` (exit 2) | the GPU is not SM86/SM89 | use proven hardware; there is no CPU fallback by design |
| `refusing to join chain '…'` | chain-id mismatch | `--chain-id` must match the node exactly |
| `the chain does not expose the frozen compute surface` | talking to a chain without the compute module | point `--node` at a Prisma node |
| `account … has N uprsm but M uprsm is required to bond` | unfunded | fund via the faucet, or lower `--bond` |
| `wrong passphrase … no recovery path` | keystore decrypt failed | restore the keystore backup or re-import the key |
| `refusing task N: model '…' not in […]` | compatibility filter | the worker only takes its configured frozen profile |
| task stuck `accepting` after a crash | normal: reconciled on restart | `prisma-worker recover` (no transactions; the chain decides) |

## DA provider

| symptom | meaning | fix |
|---|---|---|
| `OUTPUT_ROOT_MISMATCH` on store | the blob does not hash to the claimed root | the sender's bytes are wrong; nothing was stored |
| `DATA_UNAVAILABLE` on retrieval | artifact missing or quarantined | check `prisma-da scan`; a quarantined artifact stays quarantined |
| `413 quota exceeded` | the provider's size cap | the operator raises `--quota-bytes` or runs `prisma-da gc` |
| challenge answered `too_late` | the deadline was inside the safety margin | the response was not started in time; the chain's timeout path applies |

## Watcher

| symptom | meaning | fix |
|---|---|---|
| `frozen libraries` import error | `PRISMA_FROZEN_PYTHON` unset outside a repo checkout | point it at the repository root (the release ships the libraries) |
| `no provider returned the bundle` | all DA endpoints failed | check the provider URLs in the watcher config |
| `refusing to challenge: fraud verdict without recorded detection evidence` | policy gate | a challenge needs concrete detection facts, never a bare verdict |
| `challenger … has N uprsm but the bond needs M` | unfunded challenger | fund the watcher's account |

## Faucet / API

| symptom | meaning | fix |
|---|---|---|
| `was funded Ns ago; the cooldown is …` | per-address cooldown | wait the documented window |
| `per-IP rate limit exceeded` | token bucket empty | wait for refill |
| `too many distinct addresses` | anti-loop | one IP funds a bounded number of addresses per window |
| `401 unauthorized` / `403 forbidden` | API key missing/revoked/scope | re-issue with the right scopes |
| `409 job_not_cancellable` | the job was already accepted on chain | only pre-accept jobs cancel |

## Node / RPC

| symptom | meaning | fix |
|---|---|---|
| `403 … not in the public RPC allow-list` | the proxy's read-only preset | use the documented surface; writes go through the API/faucet |
| `429 rate limit exceeded` | per-IP bucket | retry after `Retry-After` |
| `504 upstream did not answer` | the node is slow/down | check the node; the proxy timeout is bounded by design |
| `501 websockets are not served` | by design | no websocket surface on the public proxy |
| genesis "sleeping until" | `genesis_time` is in the future | regenerate with a past genesis_time (netconfig) |
