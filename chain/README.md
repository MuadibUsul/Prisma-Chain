# Prisma Chain node (testnet foundation)

`prismad` is a Cosmos SDK v0.53.7 / CometBFT application with auth, bank,
staking, distribution, slashing, governance, mint, upgrade and `x/compute`.
The local genesis uses `uprsm` for gas and staking. The local entrypoint sets
SDK mint inflation to zero because the proposed base-task issuance schedule
has not been implemented.

From the repository root:

```sh
docker build -f chain/Dockerfile -t prismachain:local .
docker run --rm -p 26657:26657 -p 9090:9090 -v prisma-node:/data prismachain:local
```

The entrypoint creates a single test-keyring validator only when `/data` is
empty. Use `GET http://localhost:26657/status` as the local node readiness
check. This is a development genesis, not a public testnet configuration.
Genesis writes a compute schema marker so the compute IAVL store has a loadable
version before any task is posted. A regression test covers historical queries
and restart with no compute transactions.

## Compute wire API

The source of truth is `proto/prisma/compute/v1/`. Cosmos transactions use the
`prisma.compute.v1.Msg` service: `RegisterModel`, `BondWorker`, `PostTask`,
`AcceptTask`, `SubmitResult`, `AttestResult`, `StartChallenge`,
`ChallengeMidpoint`, `TimeoutChallenge`, `FinalizeTask`, and `RefundTask`.
Queries use `/prisma.compute.v1.Query/Task`, `/Model`, and `/Worker` on gRPC
port 9090. A task ID is a chain-assigned `uint64`; all fee and bond amounts are
integer `uprsm`.

`PostTask` accepts the protocol envelope: `mode`, `model_id`, `spec_version`,
`input_commitment`, `data_ref`, `max_fee`, `deadline`, and `privacy_tier`.
Verifiable tasks additionally put bounded `public_input` values on-chain;
registered VM programs also remain on-chain throughout challenges. The only
accepted mode values are `verifiable` and `lightweight`; privacy values are
`public` and `tier0_relative` respectively. The model version is immutable.

Verifiable result submission includes a committed trace and endpoint proofs.
Two separately bonded monitors must attest, the challenge window must close,
and a valid challenged step can overturn the result. Settlement burns 20% of
the fee, pays 5% to each monitor, and pays the residual to the worker. A
worker's bond is reserved while its task is active; late or unconfirmed tasks
can refund. Challenge timeouts follow the deterministic VM dispute outcome.

Lightweight result submission includes a canonical signed receipt. The module
checks the worker and gateway Ed25519 signatures against distinct bonded
network keys, binds the receipt to the task, pinned model, output and token
count, and stores its bytes for audit. After two other bonded monitors attest
and the confirmation window closes, the testnet charges 1,000 `uprsm` per
output token up to `max_fee`, refunds unused escrow, burns 20% of the charge,
and splits the rest 70% to the worker and 5% to each monitor. A receipt digest
alone cannot authorize payment. Missing attestations or a legacy hash-only
pending result keep a refund path open after timeout. This fixed testnet tariff
does not yet cover input tokens, GPU time, quality or a governed price table.

## Signed local acceptance (2026-10-01)

The running `prisma-local-1` node accepted a real verifiable task through
AutoCLI. The model program was one integer instruction,
`[{"Op":1,"Dst":0,"A":0,"B":0,"C":0,"Imm":42}]` (hex
`5b7b224f70223a312c22447374223a302c2241223a302c2242223a302c2243223a302c22496d6d223a34327d5d`).
It had no public inputs, so `vm.InputDigest(nil)` was
`c40db7b6e1d6624ba2d732d90d9bb2d5443c6b39283cb090a638b69117881f9c`.
The VM-generated endpoint claim is in
[`testdata/verifiable-smoke-claim.json`](testdata/verifiable-smoke-claim.json).
Its trace root is `ab37ea1a03929ef89d66b4951d4caf93ab596a19ee93e71f4dfef4093c38f3eb`
and output digest is `452ac84d513b34ba00a68b79f90564bc9767db071758ed71ab66ca2df871c908`.

Every signed transaction used `--chain-id prisma-local-1 --home /data
--keyring-dir /data --keyring-backend test --fees 0uprsm --gas 2000000
--yes --output json` inside the chain container. The key AutoCLI calls were:

```sh
prismad tx compute register-model --model-id smoke-vm --spec-version v1 --mode verifiable \
  --image-digest <32-byte-hex> --tokenizer-digest <32-byte-hex> \
  --weights-digest <32-byte-hex> --program <program-hex> --from validator
prismad tx compute bond-worker --amount 1000000 --from validator
prismad tx compute post-task --mode verifiable --model-id smoke-vm --spec-version v1 \
  --input-commitment <input-digest-hex> --max-fee 1000000 --deadline 1000 \
  --privacy-tier public --from validator
prismad tx compute accept-task --task-id 1 --from validator
prismad tx compute submit-result --task-id 1 --output-digest <output-digest-hex> \
  --trace-root <trace-root-hex> --trace-claim-json /tmp/verifiable-smoke-claim.json \
  --from validator
prismad tx compute attest-result --task-id 1 --from <bonded-monitor-a>
prismad tx compute attest-result --task-id 1 --from <bonded-monitor-b>
prismad tx compute finalize-task --task-id 1 --from validator
```

The two monitor accounts each bonded 1,000,000 `uprsm` and differed from
the requester/worker. Each transaction was checked for `DeliverTx code=0`,
not only mempool acceptance. Register, worker bond, post, accept, submit,
the two attestations, and finalization landed at heights 44, 46, 49, 51,
57, 68, 70, and 80. The task's challenge deadline was height 77.
`prismad query compute task --task-id 1 --home /data --output json` returned
`task_json` (base64 JSON) with status `settled` and released bond reservation.

For the 1,000,000 `uprsm` fee, the compute module balance fell from
4,000,000 to 3,000,000; the worker received 700,000; each monitor received
50,000; and total supply fell from 100,000,000,000 to 99,999,800,000
(200,000 burned). A second `FinalizeTask` landed at height 84 with
`DeliverTx code=1` (`task not finalizable`); module balance and supply did
not change. All figures are from live bank and compute queries.

This code does not yet implement public model training rewards, issuance
halving, treasury accounting, governance controls for compute parameters,
genesis-operator caps, a public rate-limited faucet, multi-validator deployment,
or an audit. A local developer faucet exists in `deploy/faucet.py`.
State export is also not implemented; the CLI explicitly rejects it.
