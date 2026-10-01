# Compute network foundation

This package is the off-chain control plane for Prisma's **lightweight** mode.
It handles signed node discovery, measured group admission, short coordinator
leases with fencing epochs, worker and gateway delivery signatures, and one
durable delivery receipt per chain task ID. The receipt is **provisional
delivery evidence**, not payment or proof that an LLM answer is correct.

## Run locally

Python 3.10+ is required. From this directory:

```sh
python -m pip install -e '.[test]'
python -m pytest -q tests
python -m uvicorn prisma_network.server:gateway_app_from_env --factory --host 0.0.0.0 --port 8080
```

The gateway requires `PRISMA_NODE_SEED_B64`, `PRISMA_TRUSTED_KEYS_FILE`,
`PRISMA_ALLOWED_NODE_HOSTS`, and `PRISMA_CLIENT_API_KEY`. Its SQLite path is
`PRISMA_DB` (default `prisma-network.sqlite3`). The trusted keys file is a JSON
map of `sha256(raw Ed25519 public key)` to base64 raw public key; include the
gateway and every worker. Node URLs must be HTTPS and have a host in
`PRISMA_ALLOWED_NODE_HOSTS`. For a loopback-only local devnet, explicitly set
`PRISMA_DEV_HTTP=1` and `PRISMA_DEV_UNFUNDED=1`. Production mode rejects
inference until a chain task authorizer and pinned-tokenizer counter are wired
in. Install `python -m pip install -e '.[chain,model]'`, set
`PRISMA_CHAIN_GRPC_ADDR` and `PRISMA_CHAIN_RPC_URL`, and provide
`PRISMA_NODE_ACCOUNT_BINDINGS_FILE` as a JSON map of node ID to its bonded
chain worker address. The adapter checks accepted task state, the full task
envelope, pinned chain model hashes, latest chain height, each stage's bond,
and the on-chain Ed25519 key behind each announced node ID.
It uses the chain's gRPC Query API and does not sign or broadcast transactions.
In funded mode the gateway also needs `PRISMA_MODEL_ID`, `PRISMA_MODEL_DIGEST`,
`PRISMA_WEIGHTS_DIGEST`, `PRISMA_TOKENIZER_DIGEST`, `PRISMA_RUNTIME_DIGEST`,
`PRISMA_SPEC_VERSION`, `PRISMA_MODEL_DIR`, and `PRISMA_MODEL_FILES_FILE`.
It hashes the manifest's `tokenizer.json` locally and uses that exact tokenizer
to count tokens in delivered text with no added special tokens. The worker
uses the same rule; vLLM's generated-token usage remains diagnostic only.
This tariff excludes hidden reasoning and tokens omitted from delivered text.
With those chain endpoints configured, authenticated `GET /v1/tasks/{task_id}`
returns the node-observed chain status, observation height, challenge deadline,
whether settlement/refund is final, and whether this gateway holds a provisional
delivery receipt. Its `billing` object reports escrow, proposed charge while
pending, and final charged/refunded `uprsm` amounts as decimal strings; it is
`null` for unfunded development tasks. A read failure returns 503; a receipt that conflicts with the
chain task returns 409. The endpoint reports the configured node's query result,
not an independently verified light-client proof; the height is sampled in a
separate RPC call. In unfunded local development,
nonnumeric mock task IDs are explicitly labeled `unfunded_dev` and never final.
Configure optional
`PRISMA_GOSSIP_PEERS` as comma-separated gateway base URLs.

The stage-0 worker sidecar starts with:

```sh
python -m uvicorn prisma_network.server:worker_app_from_env --factory --host 0.0.0.0 --port 8081
```

Other pipeline stages use `prisma_network.server:stage_app_from_env --factory`
on their own ports. Every sidecar needs `PRISMA_NODE_SEED_B64`,
`PRISMA_TRUSTED_KEYS_FILE`, `PRISMA_CONTROL_URL`, `PRISMA_PUBLIC_URL`,
`PRISMA_GROUP_ID`, `PRISMA_STAGE_INDEX`, `PRISMA_STAGE_COUNT`,
`PRISMA_MODEL_ID`, `PRISMA_MODEL_DIGEST`, `PRISMA_WEIGHTS_DIGEST`,
`PRISMA_TOKENIZER_DIGEST`, `PRISMA_RUNTIME_DIGEST`, `PRISMA_SPEC_VERSION`,
`PRISMA_GPU_COUNT`, and `PRISMA_VRAM_MB`. Stage 0 additionally needs
`PRISMA_VLLM_URL` and `PRISMA_VLLM_API_KEY`. Non-head stages need
`PRISMA_STAGE_HEALTH_URL` so their probes fail when the actual stage is down.
Funded nodes set `PRISMA_CHAIN_WORKER` to the chain account bound to their
Ed25519 node ID in the gateway bindings file.
The sidecars sign and renew 45-second announcements every 20 seconds. At
startup each stage also requires `PRISMA_MODEL_DIR` and
`PRISMA_MODEL_FILES_FILE` (JSON with nonempty `weights` and `tokenizer`
path-to-SHA256 maps). It hashes every listed local file and rejects a mismatch.
`PRISMA_DEV_SKIP_FILE_HASH=1` bypasses this only for the local mock devnet.
The image's `PRISMA_RUNTIME_DIGEST` must come from the deployment registry;
the process cannot independently attest its own container image.

Private prompts are accepted only in the gateway and worker request body over
an operator-provided TLS/mTLS transport. The gateway stores commitments,
attempts and receipts but no prompt or answer text. Place TLS termination and
egress restrictions in front of these HTTP services; the optional HTTP dev
flag provides no transport privacy. `GET /v1/privacy` and every inference
response disclose the Tier 0 boundary.

## Wire flow

1. Stage sidecars `POST /v1/announcements` with Ed25519-signed, expiring
   capabilities. Gateways may exchange these via `GET /v1/announcements`.
   `model_digest` is SHA-256 of sorted compact UTF-8 JSON for
   `{model_id, weights_digest, tokenizer_digest, runtime_digest, spec_version}`.
   Weight/tokenizer digests are SHA-256 of their sorted file-hash maps. The
   gateway binds routes and requests to this digest and exact spec version.
   Trust keys must be provisioned from bonded chain workers by the operator;
   a signature alone does not prove a worker has a bond or GPU. A worker binding
   a new Ed25519 network key supplies the v1 chain/account/key possession proof
   as `--network-key-proof` in its `bond-worker` transaction; see
   [the protocol](../docs/protocol.md).
2. Gateway polls `/v1/probe` on every stage. A route is eligible only with all
   stages present, the same pinned model/runtime hashes and recent successful
   probes. Scores use observed latency/bandwidth and sidecar queue depth.
3. Client `POST /v1/inference` provides a chain task envelope and messages.
   `input_commitment` is SHA-256 of sorted compact UTF-8 JSON containing
   `messages`, `max_tokens` and `temperature`. The gateway checks the chain
   task through its injected authorizer before dispatching to the selected group. Funded
   tasks use a locally loaded, hash-verified tokenizer so a
   worker-reported billable token count is checked against delivered text. The
   environment factory provides the read-only chain adapter and tokenizer
   counter when configured. The
   `PRISMA_DEV_UNFUNDED=1` switch is solely for local devnet tests.
4. Gateway acquires a durable task lease, creates an attempt ID, and sends a
   signed execution request to the stage-0 worker. The worker queries the
   current lease before and after generation. It signs output commitment and
   token count; the gateway verifies that signature and signs a delivery
   receipt. Retries use a new attempt under the same task ID. A newer lease
   epoch fences old coordinators and invalidates unfinished attempts.
5. With a worker-owned chain submission callback configured, the gateway sends
   the double-signed receipt to `/v1/submit-receipt` after saving it durably.
   Set `PRISMA_RECEIPT_AUTOSUBMIT=1` on the gateway to enable this handoff.
   The worker verifies both signatures before its own account broadcasts a
   `SubmitResult` transaction. A failed broadcast leaves the signed receipt
   available for retry under the same task ID without rerunning inference.
   The local integration test injects a devnet CLI signer; the packaged worker
   factory does not yet configure a production signer. The chain module
   verifies escrow, bond, model version, signatures, timeout and acceptance
   rules before settlement. A repeated completed request returns the same
   receipt without answer text, because private outputs are not kept.
   Inference responses expose `chain_submission` as `unconfigured`,
   `retry_required`, `submitted`, `confirmed`, `conflict`, or `closed`.
   `submitted` only means the worker submission callback returned successfully;
   clients must still read `/v1/tasks/{task_id}` for chain confirmation and
   final settlement.

Signatures use Ed25519 over `domain + "\n" + sorted compact UTF-8 JSON`.
Domains are `prisma:capability:v1`, `prisma:execution:v1`,
`prisma:worker-receipt:v1`, and `prisma:gateway-receipt:v1`. Exact signed
wire examples are in [examples/contract.json](examples/contract.json).

## vLLM deployment boundary

[vLLM's distributed serving guide](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
supports one Ray-backed model replica across multiple nodes with pipeline
parallelism. Start a Ray cluster with identical, pinned vLLM images and model
artifacts on all GPU nodes, then run
`examples/launch_vllm_ray.sh` once on its head with `MODEL_SIZE=14` for two
stages or `MODEL_SIZE=32` for four. Set a full `HF_REVISION` commit, API key,
GPU count and enough VRAM before use. This repository has **not** run either
model on GPUs; the launch script is a deployment template, not a benchmark.
The stage-0 sidecar talks to the vLLM OpenAI-compatible API. Other stages
are managed by Ray/vLLM; this code does not implement tensor transfer or KV
checkpoint restoration. If generation fails, the attempt fails and is
retryable after fencing. The control plane's SQLite database supports
process-level standby only when both processes share that transactional
database; a multi-host testnet requires a shared transactional or chain-backed
lease store. Neither cross-host failover nor encrypted-at-rest payload storage
is claimed here.

For a CPU-only protocol smoke test, run
`python -m uvicorn prisma_network.mock_model:app --host 0.0.0.0 --port 8000`
and use model ID `demo-model`. Its fixed `MOCK` output exercises HTTP and
receipt plumbing only. It does not exercise GPU, Ray, Qwen or distributed
inference, and it must run only with `PRISMA_DEV_UNFUNDED=1`.
