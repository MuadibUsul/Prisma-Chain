"""Passphrase-protected local storage for the worker identity (B2-01d..f).

Design:

- scrypt (n=2^15, r=8, p=1) derives the wrapping key from the passphrase;
  AES-GCM authenticates and encrypts the secret material, with the
  keystore version as associated data so a re-labelled file fails closed.
- The file also carries the *public* record in the clear: it is public by
  definition, it makes ``prisma-worker identity show`` passphrase-free for
  operators, and ``load`` verifies the decrypted secrets reproduce it.
- Permissions: the file is written 0600 and refuses group/world-readable
  modes on POSIX; on platforms without meaningful modes the check is
  documented as unavailable (Windows relies on user-profile ACLs).
- There is no recovery path for a lost passphrase: the error says so
  explicitly instead of pretending.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .identity import WorkerIdentity

KEYSTORE_VERSION = 1
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**15, 8, 1
_MIN_SCRYPT_N = 2**14
_AAD = b"prisma-worker-keystore-v1"


class KeystoreError(Exception):
    """Base class for keystore failures."""


class UnrecoverableKeystoreError(KeystoreError):
    """Wrong passphrase or corrupt file: there is no recovery path."""


class KeystorePermissionsError(KeystoreError):
    """The key file is readable beyond its owner."""


def default_path() -> pathlib.Path:
    override = os.environ.get("PRISMA_WORKER_KEYSTORE")
    if override:
        return pathlib.Path(override).expanduser()
    return pathlib.Path.home() / ".prisma-worker" / "worker-key.json"


def _derive_key(passphrase: str, salt: bytes, n: int = _SCRYPT_N) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=n, r=_SCRYPT_R, p=_SCRYPT_P)
    return kdf.derive(passphrase.encode("utf-8"))


def _enforce_owner_only(path: pathlib.Path) -> None:
    if os.name == "posix":
        os.chmod(path, 0o600)


def _assert_owner_only(path: pathlib.Path) -> None:
    if os.name != "posix":
        # Windows file modes are not meaningful; access is controlled by
        # the user-profile ACL. Recorded, not silently assumed.
        return
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise KeystorePermissionsError(
            f"{path} is group/world accessible (mode {oct(mode)}); "
            "refusing to read key material — run: chmod 600 " + str(path))


def save(path: pathlib.Path | str, identity: WorkerIdentity, passphrase: str) -> pathlib.Path:
    if not passphrase:
        raise KeystoreError("a passphrase is required")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    salt, nonce = os.urandom(16), os.urandom(12)
    secrets_doc = json.dumps({
        "protocol_seed_b64": base64.b64encode(identity.protocol_seed).decode(),
        "account_scalar_hex": identity.account_scalar.hex(),
    }, sort_keys=True).encode()
    ciphertext = AESGCM(_derive_key(passphrase, salt)).encrypt(nonce, secrets_doc, _AAD)
    document = {
        "version": KEYSTORE_VERSION,
        "kdf": {"name": "scrypt", "n": _SCRYPT_N, "r": _SCRYPT_R, "p": _SCRYPT_P,
                "salt_b64": base64.b64encode(salt).decode()},
        "cipher": "AESGCM",
        "nonce_b64": base64.b64encode(nonce).decode(),
        "ciphertext_b64": base64.b64encode(ciphertext).decode(),
        "public": identity.public_record(),
    }
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(json.dumps(document, indent=1).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _enforce_owner_only(path)
    return path


def load(path: pathlib.Path | str, passphrase: str) -> WorkerIdentity:
    path = pathlib.Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KeystoreError(f"no keystore at {path}; run `prisma-worker identity init`") from exc
    except json.JSONDecodeError as exc:
        raise KeystoreError(f"{path} is not a valid keystore document") from exc
    if document.get("version") != KEYSTORE_VERSION:
        raise KeystoreError(f"unsupported keystore version {document.get('version')!r}")
    kdf = document.get("kdf", {})
    if kdf.get("name") != "scrypt" or int(kdf.get("n", 0)) < _MIN_SCRYPT_N:
        raise KeystoreError("keystore uses an unsupported or weakened KDF; refusing to load")
    _assert_owner_only(path)
    try:
        salt = base64.b64decode(kdf["salt_b64"], validate=True)
        nonce = base64.b64decode(document["nonce_b64"], validate=True)
        ciphertext = base64.b64decode(document["ciphertext_b64"], validate=True)
    except (KeyError, ValueError) as exc:
        raise KeystoreError(f"{path} is missing keystore fields") from exc
    key = _derive_key(passphrase, salt, int(kdf["n"]))
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, _AAD)
    except InvalidTag as exc:
        raise UnrecoverableKeystoreError(
            "the passphrase is wrong or the keystore is corrupt; there is no recovery path — "
            "restore from a backup or re-import the operator key, then re-announce it"
        ) from exc
    secrets_doc = json.loads(plaintext)
    identity = WorkerIdentity(
        protocol_seed=base64.b64decode(secrets_doc["protocol_seed_b64"], validate=True),
        account_scalar=bytes.fromhex(secrets_doc["account_scalar_hex"]),
    )
    recorded = document.get("public", {})
    if recorded.get("account_address") and recorded["account_address"] != identity.account_address:
        raise KeystoreError("keystore public record does not match the decrypted secrets")
    return identity


def public_info(path: pathlib.Path | str) -> dict:
    """The public record without needing the passphrase (B2-01c)."""
    path = pathlib.Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("version") != KEYSTORE_VERSION:
        raise KeystoreError(f"unsupported keystore version {document.get('version')!r}")
    return document.get("public", {})


def rotate(path: pathlib.Path | str, passphrase: str,
           new_passphrase: Optional[str] = None) -> tuple[dict, WorkerIdentity]:
    """Replace the identity in place (B2-01g).

    Returns the *old* public record and the new identity so the operator can
    announce the new keys to the chain before retiring the old ones. The old
    public record is the only record kept in logs/artifacts — never the old
    secrets.
    """
    old_public = public_info(path)
    new_identity = WorkerIdentity.generate()
    save(path, new_identity, new_passphrase or passphrase)
    return old_public, new_identity
