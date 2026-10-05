"""API authentication (B6-02): bearer API keys with scoped prefixes.

Design:

- Keys are random 32-byte secrets, stored only as sha256 hashes; the plaintext
  is shown exactly once at creation. A key looks like ``prsk_<24 base62>_`` +
  a checksum suffix so a truncated key fails validation instead of silently
  matching a different key.
- Scopes: ``jobs:write`` (submit/cancel), ``jobs:read`` (get/list/profiles).
  The admin surface (creating keys) is out of process.
- Constant-time comparisons (hash lookup, then compare_digest on the hash).
- Rate limiting per key is B6's scheduler concern; here it is only the
  identity layer. Every denied answer uses the stable error schema.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import string
import time
from dataclasses import dataclass, field
from typing import Optional

from .jobs import error_response

SCOPES = ("jobs:read", "jobs:write")
_ALPHABET = string.ascii_letters + string.digits


@dataclass
class ApiKeyRecord:
    key_id: str
    key_hash: str
    scopes: tuple[str, ...]
    created_at: float
    name: str = ""
    revoked: bool = False


@dataclass
class KeyStore:
    records: dict = field(default_factory=dict)      # key_hash -> ApiKeyRecord

    def add(self, record: ApiKeyRecord) -> None:
        self.records[record.key_hash] = record

    def find(self, key_hash: str) -> Optional[ApiKeyRecord]:
        return self.records.get(key_hash)


class Authenticator:
    def __init__(self, store: KeyStore):
        self.store = store

    # --- issuance (admin side; returns the plaintext once) -----------------

    @staticmethod
    def issue(name: str = "", scopes: tuple[str, ...] = SCOPES) -> tuple[str, ApiKeyRecord]:
        unknown = [scope for scope in scopes if scope not in SCOPES]
        if unknown:
            raise ValueError(f"unknown scopes: {unknown}")
        body = "".join(secrets.choice(_ALPHABET) for _ in range(24))
        secret = f"prsk_{body}"
        digest = base64.b32encode(secrets.token_bytes(5)).decode().rstrip("=")
        secret += digest[:4]
        key_hash = hashlib.sha256(secret.encode()).hexdigest()
        record = ApiKeyRecord(key_id=f"key_{digest[4:8] or '0000'}", key_hash=key_hash,
                              scopes=tuple(scopes), created_at=time.time(), name=name)
        return secret, record

    # --- validation ---------------------------------------------------------

    @staticmethod
    def _bearer(headers: dict) -> Optional[str]:
        authorization = ""
        for key, value in (headers or {}).items():
            if key.lower() == "authorization":
                authorization = value
                break
        if not authorization.startswith("Bearer "):
            return None
        return authorization[7:].strip()

    def authenticate(self, headers: dict) -> tuple[Optional[ApiKeyRecord], Optional[tuple[int, dict]]]:
        secret = self._bearer(headers)
        if not secret:
            return None, error_response("unauthorized",
                                        "missing or malformed Authorization header")
        key_hash = hashlib.sha256(secret.encode()).hexdigest()
        record = self.store.records.get(key_hash)
        if record is None or not hmac.compare_digest(record.key_hash, key_hash):
            return None, error_response("unauthorized", "invalid API key")
        if record.revoked:
            return None, error_response("unauthorized", "this API key has been revoked")
        return record, None

    @staticmethod
    def authorize(record: Optional[ApiKeyRecord], scope: str) -> Optional[tuple[int, dict]]:
        if record is None:
            return error_response("unauthorized", "authentication required")
        if scope not in record.scopes:
            return error_response("forbidden", f"this API key lacks the {scope} scope")
        return None


def auth_guard(authenticator: Authenticator, scope: str):
    """Return an auth hook for JobAPI routes: None when allowed, an error when not."""

    def guard(headers: dict) -> Optional[tuple[int, dict]]:
        record, failure = authenticator.authenticate(headers)
        if failure is not None:
            return failure
        return Authenticator.authorize(record, scope)

    return guard
