# prisma (CLI)

One entrypoint over the Prisma product packages (B5-01). Ownership per group
is stated honestly — chain ops delegate to `prismad`, API groups are HTTP
clients, daemon groups delegate to the components' own CLIs:

```text
prisma wallet init|show     operator key/account (worker identity rules, B2-01)
prisma balance              account balance (chain)
prisma bond [--register]    bond (+ protocol-key registration) (chain)
prisma worker-register      bond + register (chain)
prisma faucet               testnet faucet request (API, B7)
prisma job post|get         developer job surface (API, B6)
prisma task --task-id N     task status (chain)
prisma receipt --id …       VWR lookup (chain)
prisma da --task-id N       DA availability status (chain)
prisma dispute --task-id N  dispute status (chain)
prisma run-worker|run-da|run-watcher   launch the daemons
prisma version              release identity (B1-01)
prisma config set|show      one operator config file, validated
```

Global overrides: `--chain-id --node --api --prismad`. Config file:
`~/.prisma/cli.json` (`PRISMA_CLI_CONFIG` overrides). Passphrase:
`--passphrase-stdin` / `PRISMA_CLI_PASSPHRASE` / prompt.
