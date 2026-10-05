"""DA_REPLICA_V1 client: bundle upload, attestation verification, quorum (B2-08).

The frozen flow (Phase E's reference provider, ``deploy/da_replica.py``):

```text
GET  /health   -> provider info (including the advertised Ed25519 key)
POST /store    -> the provider recomputes output_root from the bytes it
                  received BEFORE storing; a mismatch is refused
POST /attest   -> a canonical signed DAAttestation, only after the blob was
                  verified at store time
```

Rules this module enforces:

- **An upload is not storage.** A provider only counts towards the quorum once
  its attestation *verifies* here: the typed bindings (protocol version, task,
  assignment, output root, output bytes) and the Ed25519 signature over
  ``PRISMA_GEMM_SIG_V1\\x00 || encode_canonical(attestation with signature
  cleared)`` — the same preimage the chain verifies.
- **The quorum is never bypassed.** ``ensure_quorum`` raises unless the
  configured number of *verified* attestations exists; the failure lists each
  provider's outcome.
- **Bounded by construction.** Per-provider timeouts, bounded retries with
  backoff, and no unbounded retry loop.

The chain side is out of scope here: providers sign and submit their own
attestations, and the chain's finalize gate requires exactly two attesters
(``keeper.go``). The worker's job is to get the bundle into the providers and
observe the quorum.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

from cryptography.hazmat.primitives.asymmetric import ed25519

from .graphid import encode_canonical

DA_PROTOCOL_VERSION = "DA_REPLICA_V1"
DA_SIGN_DOMAIN = b"PRISMA_GEMM_SIG_V1\x00"

_WIRE_BYTES_FIELDS = ("TaskID", "AssignmentID", "OutputRoot", "ProviderAccount",
                      "ProviderPubKey", "Signature")
_WIRE_TO_CANONICAL = {
    "ProtocolVersion": "protocol_version",
    "TaskID": "task_id",
    "AssignmentID": "assignment_id",
    "OutputRoot": "output_root",
    "ProviderAccount": "provider_account",
    "ProviderPubKey": "provider_pub_key",
    "OutputBytes": "output_bytes",
    "AvailableUntilHeight": "available_until_height",
    "AttestedHeight": "attested_height",
    "Signature": "signature",
}


class DAError(Exception):
    """Base class for DA interaction failures."""


class ProviderUnavailable(DAError):
    """The provider did not answer within its deadline."""


class AttestationInvalid(DAError):
    """The attestation does not verify: it must not count towards a quorum."""


class QuorumNotReached(DAError):
    """Fewer than the required number of verified attestations exist."""

    def __init__(self, message: str, outcomes: dict):
        super().__init__(message)
        self.outcomes = outcomes


@dataclass
class DAProvider:
    url: str
    name: str = ""
    public_key: Optional[bytes] = None  # advertised Ed25519 key (from /health or the chain)
    outcomes: list = field(default_factory=list)

    def __post_init__(self):
        self.name = self.name or self.url


def _decode_body(body: str) -> dict:
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"error": body.strip()[:200] or "empty response"}


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        # A provider refusal carries the reason in the body (e.g.
        # OUTPUT_ROOT_MISMATCH) — surface it instead of losing it to the
        # generic "HTTP Error 400" text.
        result = _decode_body(exc.read().decode("utf-8", "replace"))
        result.setdefault("error", f"HTTP {exc.code}")
        return result
    except urllib.error.URLError as exc:
        raise ProviderUnavailable(f"{url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DAError(f"{url}: response is not JSON: {exc}") from exc


def _get_json(url: str, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.URLError as exc:
        raise ProviderUnavailable(f"{url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DAError(f"{url}: response is not JSON: {exc}") from exc


def provider_health(provider: DAProvider, *, timeout: float = 10.0) -> dict:
    info = _get_json(provider.url.rstrip("/") + "/health", timeout)
    advertised = info.get("pubkey") or info.get("public_key") or ""
    if advertised:
        try:
            provider.public_key = base64.b64decode(advertised, validate=True)
        except ValueError:
            provider.public_key = bytes.fromhex(advertised)
    return info


def store_bundle(provider: DAProvider, *, task_id: int, c_hex: str, output_root_hex: str,
                 m: int, n: int, k: int, assignment_id_hex: str, task_id32_hex: str,
                 timeout: float = 30.0, extra: Optional[dict] = None) -> dict:
    payload = {"task_id": task_id, "c_hex": c_hex, "output_root": output_root_hex,
               "m": m, "n": n, "k": k, "assignment_id": assignment_id_hex,
               # the provider recomputes the root over tiles keyed by the 32-byte task id
               "task_id32": task_id32_hex}
    payload.update(extra or {})
    result = _post_json(provider.url.rstrip("/") + "/store", payload, timeout)
    if result.get("error"):
        raise DAError(f"{provider.name}: store refused: {result['error']}")
    return result


def request_attestation(provider: DAProvider, *, task_id: int, assignment_id_hex: str,
                        available_until: int, attested_height: int,
                        timeout: float = 15.0) -> dict:
    result = _post_json(provider.url.rstrip("/") + "/attest", {
        "task_id": task_id, "assignment_id": assignment_id_hex,
        "available_until": available_until, "attested_height": attested_height}, timeout)
    wire = result.get("attestation") or {}
    if not wire:
        raise AttestationInvalid(
            f"{provider.name}: no attestation in the response"
            + (f" ({result['error']})" if result.get("error") else ""))
    return wire


def canonical_from_wire(wire: dict) -> dict:
    """Rebuild the canonical (signing) view from the Go-shaped wire object."""
    out = {}
    for wire_key, canonical_key in _WIRE_TO_CANONICAL.items():
        if wire_key not in wire:
            raise AttestationInvalid(f"attestation is missing {wire_key}")
        value = wire[wire_key]
        if wire_key in _WIRE_BYTES_FIELDS:
            try:
                out[canonical_key] = base64.b64decode(value, validate=True)
            except (ValueError, TypeError) as exc:
                raise AttestationInvalid(f"attestation field {wire_key} is not base64") from exc
        elif wire_key == "ProtocolVersion":
            out[canonical_key] = str(value)
        else:
            try:
                out[canonical_key] = int(value)
            except (TypeError, ValueError) as exc:
                raise AttestationInvalid(f"attestation field {wire_key} is not an integer") from exc
    return out


def verify_attestation(wire: dict, *, expected: dict) -> dict:
    """Verify a provider attestation against the expectation; returns the canonical form.

    ``expected`` keys: provider_public (bytes), task_id (int), assignment_id_hex,
    output_root_hex, output_bytes (int), min_available_until (int, optional).
    """
    canonical = canonical_from_wire(wire)
    if canonical["protocol_version"] != DA_PROTOCOL_VERSION:
        raise AttestationInvalid(f"protocol_version {canonical['protocol_version']!r}")
    provider_public = expected["provider_public"]
    if canonical["provider_pub_key"] != provider_public:
        raise AttestationInvalid("attestation is signed by a different provider key")
    if canonical["task_id"] != int(expected["task_id"]).to_bytes(8, "big"):
        raise AttestationInvalid("attestation binds a different task")
    if canonical["assignment_id"].hex() != expected["assignment_id_hex"]:
        raise AttestationInvalid("attestation binds a different assignment")
    if canonical["output_root"].hex() != expected["output_root_hex"]:
        raise AttestationInvalid("attestation covers a different output root")
    if canonical["output_bytes"] != int(expected["output_bytes"]):
        raise AttestationInvalid("attestation covers different output bytes")
    if "min_available_until" in expected and canonical["available_until_height"] < int(
            expected["min_available_until"]):
        raise AttestationInvalid("attestation retention window is too short")
    signature = canonical["signature"]
    if len(signature) != 64:
        raise AttestationInvalid("attestation signature is not 64 bytes")
    preimage = DA_SIGN_DOMAIN + encode_canonical({**canonical, "signature": b""})
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(provider_public).verify(signature, preimage)
    except Exception as exc:  # noqa: BLE001 - cryptography raises InvalidSignature
        raise AttestationInvalid(f"attestation signature does not verify: {exc}") from exc
    return canonical


def ensure_quorum(providers: list[DAProvider], *, quorum: int, bundle_hex: str,
                  output_root_hex: str, m: int, n: int, k: int, task_id: int,
                  assignment_id_hex: str, task_id32_hex: str, available_until: int,
                  attested_height: int,
                  retries: int = 2, backoff_seconds: float = 0.5,
                  per_provider_timeout: float = 20.0,
                  sleep: Callable[[float], None] = time.sleep,
                  log: Callable[[str], None] = lambda _m: None) -> dict:
    """Upload to every provider and require ``quorum`` *verified* attestations.

    Returns ``{"verified": [...], "outcomes": {...}}`` on success; raises
    ``QuorumNotReached`` (carrying the per-provider outcomes) otherwise. An
    upload without a verified attestation never counts.
    """
    if quorum < 1:
        raise DAError("quorum must be at least 1")
    if quorum > len(providers):
        raise DAError(f"quorum {quorum} exceeds the provider count {len(providers)}")
    output_bytes = m * n * 4
    verified: list[dict] = []
    outcomes: dict[str, str] = {}
    for provider in providers:
        if len(verified) >= quorum:
            outcomes[provider.name] = "not needed (quorum already reached)"
            log(f"{provider.name}: skipped, quorum already reached")
            continue
        attempt = 0
        while True:
            attempt += 1
            try:
                provider_health(provider, timeout=per_provider_timeout)
                if not provider.public_key:
                    raise DAError(f"{provider.name}: provider advertises no public key")
                store_bundle(provider, task_id=task_id, c_hex=bundle_hex,
                             output_root_hex=output_root_hex, m=m, n=n, k=k,
                             assignment_id_hex=assignment_id_hex, task_id32_hex=task_id32_hex,
                             timeout=per_provider_timeout * 2)
                wire = request_attestation(provider, task_id=task_id,
                                           assignment_id_hex=assignment_id_hex,
                                           available_until=available_until,
                                           attested_height=attested_height,
                                           timeout=per_provider_timeout)
                canonical = verify_attestation(wire, expected={
                    "provider_public": provider.public_key, "task_id": task_id,
                    "assignment_id_hex": assignment_id_hex, "output_root_hex": output_root_hex,
                    "output_bytes": output_bytes, "min_available_until": available_until})
                verified.append({"provider": provider.name, "attestation": wire,
                                 "canonical": {k2: (v.hex() if isinstance(v, bytes) else v)
                                               for k2, v in canonical.items()}})
                outcomes[provider.name] = f"verified (attempt {attempt})"
                log(f"{provider.name}: attestation verified (attempt {attempt})")
                break
            except (DAError, ProviderUnavailable, AttestationInvalid) as exc:
                if attempt > retries:
                    outcomes[provider.name] = f"failed after {attempt} attempts: {exc}"
                    log(f"{provider.name}: {outcomes[provider.name]}")
                    break
                log(f"{provider.name}: attempt {attempt} failed ({exc}); retrying")
                sleep(backoff_seconds * attempt)
    if len(verified) < quorum:
        raise QuorumNotReached(
            f"only {len(verified)} of {len(providers)} providers produced verified attestations; "
            f"quorum {quorum} not reached", outcomes)
    return {"verified": verified, "outcomes": outcomes}
