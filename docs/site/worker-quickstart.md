# Worker Quickstart

Requirements: an NVIDIA GPU with compute capability **8.6 or 8.9** (A40,
RTX 2000 Ada class — the proven targets). Anything else is refused with
`CAPABILITY_UNSUPPORTED`: there is no CPU fallback, by protocol design.

```bash
pip install -e ./network -e ./worker
prisma-worker gpu probe          # must say CAPABILITY_SUPPORTED
prisma-worker identity init      # encrypted keystore (0600); keep the passphrase safe
```

## Join

```bash
prisma-worker join --chain-id prisma-testnet-1 \
                   --node http://rpc.testnet.example:26657 \
                   --bond 1000000
```

`join` validates the chain id, the frozen compute surface, your balance, then
submits one bond+register transaction and verifies the chain shows your
protocol key. The state file (`~/.prisma-worker/state.json`) records the
capability report and the pending gateway announcement.

## Run

```bash
prisma-worker heartbeat ...   # liveness + capability refresh
prisma jobs discover ...      # list open compatible tasks
prisma jobs accept --task-id N ...   # journaled, exactly-once accept
```

Execution (the frozen `TRUE_FUSED_MMA_A13W10` kernels) runs bit-exactly
against the reference; the commit (CommitV3), the DA upload and the receipt
follow automatically. If the process dies at any point, restart recovery
classifies every journalled task and never double-submits.

**No secrets in logs** is enforced (redaction is built in); never disable it.
