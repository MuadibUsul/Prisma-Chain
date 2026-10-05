# CLI reference — `prisma`

Global overrides (before the command): `--chain-id --node --api --prismad`.
Config: `~/.prisma/cli.json` (`PRISMA_CLI_CONFIG`). Passphrase:
`--passphrase-stdin` / `PRISMA_CLI_PASSPHRASE` / prompt.

| command | ownership | notes |
|---|---|---|
| `prisma wallet init [--import-protocol-hex HEX]` | local | worker identity rules; encrypted keystore 0600 |
| `prisma wallet show` | local | public record only, no passphrase needed |
| `prisma balance [--address ADDR]` | chain query | defaults to the operator identity |
| `prisma bond --amount N [--register]` | **chain tx** | + network-key proof when `--register` |
| `prisma worker-register --amount N` | **chain tx** | the bond+register spelling |
| `prisma faucet --address ADDR` | **API (B7)** | test tokens, no monetary value |
| `prisma job post --profile P --input JSON [--idempotency-key K]` | **API (B6)** | never a raw chain tx |
| `prisma job get --job-id ID` | **API (B6)** | B6-06 presentation |
| `prisma task --task-id N` | chain query | base64 task envelope decoded |
| `prisma receipt --receipt-id ID` | chain query | VWR lookup |
| `prisma da --task-id N` | chain query | DA availability status |
| `prisma dispute --task-id N` | chain query | dispute status |
| `prisma run-worker` / `run-da` / `run-watcher` | exec | delegates to each daemon's CLI |
| `prisma version` | chain query | release + frozen protocol identity (B1-01) |
| `prisma config set` / `show` | local | validated before saving |

Daemon CLIs (full flags in their own READMEs): `prisma-worker`
(identity/join/heartbeat/jobs/gpu), `prisma-da` (init/register/run/status/gc/
scan/responder), `prisma-watcher` (init/run/status/verify/challenge).
