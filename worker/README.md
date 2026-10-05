# prisma-worker

Prisma Chain worker daemon (Phase B productization). B2-01 delivers the
identity layer: the local keys, their encrypted storage, and the safety
rules around them. Later B2 tasks add the GPU probe, join/bond, the job
lifecycle, execution, commit pipeline, DA and crash recovery.

```text
pip install -e ./network -e ./worker      # prisma-worker depends on prisma-network
prisma-worker identity init               # generate + store an encrypted identity
prisma-worker identity show               # public record (no passphrase needed)
```

## Identity model

A worker holds **two keys with different jobs** — never confuse them:

| key | algorithm | what it does | public form |
|---|---|---|---|
| protocol key | ed25519 | signs capability announcements, receipts, and the network-key binding proof | `node_id` = sha256(pubkey), plus base64/hex public key |
| chain account | secp256k1 | the funded, bonded account that submits transactions | bech32 address `prsm1…` |

Address derivation is checked against the chain's own tooling in the test
suite (the vector comes from `prismad keys`), so this package cannot drift
from the node's address format without failing.

## Commands

```text
prisma-worker identity init   [--keystore PATH] [--json]
prisma-worker identity import --protocol-key-hex HEX [--account-key-hex HEX | --account-key-file FILE]
prisma-worker identity show   [--keystore PATH] [--json]
prisma-worker identity rotate [--keystore PATH] [--json]
```

Passphrase sources, in order: `--passphrase-stdin` (one line on stdin),
`PRISMA_WORKER_PASSPHRASE`, interactive prompt (with confirmation on
`init`/`import`). An empty passphrase is refused. The keystore path
defaults to `~/.prisma-worker/worker-key.json` (override with
`--keystore` or `PRISMA_WORKER_KEYSTORE`).

## Storage and permissions

- scrypt (n=2^15, r=8, p=1) wraps the secrets with AES-GCM; the keystore
  version is authenticated as associated data, so a re-labelled or
  downgraded file fails closed (a weakened KDF in the file is refused).
- The file is written atomically, 0600, and **refuses group/world-readable
  modes on POSIX** (Windows relies on user-profile ACLs; that is recorded,
  not silently assumed).
- The public record is stored in the clear inside the keystore (it is
  public by definition) so `identity show` needs no passphrase; `load`
  verifies that the decrypted secrets reproduce it.

## Recovery behaviour (B2-01h)

- **Lost passphrase → unrecoverable.** The error says so explicitly; there
  is no backdoor and no partial decryption. Restore from a backup
  (`cp` the keystore to offline media; it is useless without the
  passphrase) or re-import the operator key.
- **Lost local file → re-import.** `identity import` accepts an existing
  ed25519 protocol key (hex, `0x` optional) and a secp256k1 account key
  (hex or a file containing it), so an operator outlives a lost disk. After
  re-import, re-announce the network-key binding so the chain learns the
  key again.
- **Compromised key → rotate.** `identity rotate` replaces the identity in
  place and prints the retired and new public records; announce the new
  protocol key to the chain (network-key binding) **before** removing the
  old one anywhere, and keep the old key usable until the announcement is
  accepted (B2-01g).

## No secrets in logs (B2-01f)

Two layers, both tested:

1. every secret representation the process knows about is registered and
   replaced by `«redacted»` in messages, arguments and exception
   tracebacks;
2. any 64+ character hex run is replaced by `«redacted-hex»` even if it was
   never registered, so a new code path cannot leak a raw key by accident.

Call `prisma_worker.redact.install()` in daemons before logging anything,
and `register_secret(...)` for every secret derived at runtime.

## Keystore format (v1)

```json
{
  "version": 1,
  "kdf": {"name": "scrypt", "n": 32768, "r": 8, "p": 1, "salt_b64": "…"},
  "cipher": "AESGCM",
  "nonce_b64": "…",
  "ciphertext_b64": "…",
  "public": {"node_id": "…", "account_address": "prsm1…", "protocol_public_b64": "…", "…": "…"}
}
```

The `public` block is the only part an operator should ever paste into a
ticket, a commit or a chat.
