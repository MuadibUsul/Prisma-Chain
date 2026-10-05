"""Worker identity: the protocol key (ed25519) and the chain account (secp256k1).

The two keys have different jobs and must never be confused:

- **protocol key** (ed25519): signs capability announcements, receipts and
  network-key binding proofs. Its public half is the worker's ``node_id``.
- **chain account** (secp256k1): the funded, bonded account that submits
  transactions. Its bech32 address (``prsm1...``) is what the chain sees.

Address derivation is validated against the chain's own tooling (see
``tests/test_identity.py``: the vector produced by ``prismad keys``), so
this module can never silently drift from the node's address format.
"""

from __future__ import annotations

import base64
import hashlib
import secrets as _secrets
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from prisma_network.core import canonical

from ._ripemd160 import ripemd160_pure

ACCOUNT_HRP = "prsm"
_PROTOCOL_BINDING_DOMAIN = "prisma:network-key-binding:v1"
SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _bech32_polymod(values: list[int]) -> int:
    generator = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for value in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ value
        for i in range(5):
            chk ^= generator[i] if ((top >> i) & 1) else 0
    return chk


def _bech32_hrp_expand(hrp: str) -> list[int]:
    return [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]


def _convertbits(data: bytes, frombits: int, tobits: int) -> list[int]:
    acc = 0
    bits = 0
    out: list[int] = []
    maxv = (1 << tobits) - 1
    for value in data:
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            out.append((acc >> bits) & maxv)
    if bits:
        out.append((acc << (tobits - bits)) & maxv)
    return out


def bech32_encode(hrp: str, data: list[int]) -> str:
    values = _bech32_hrp_expand(hrp) + data
    polymod = _bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ 1
    checksum = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(_BECH32_CHARSET[d] for d in data + checksum)


def bech32_decode(value: str) -> tuple[str, bytes]:
    """Decode a bech32 string; used by the tests, hence kept public."""
    pos = value.rfind("1")
    if pos < 1 or pos + 7 > len(value):
        raise ValueError("invalid bech32 string")
    hrp, data_part = value[:pos], value[pos + 1:]
    data = [_BECH32_CHARSET.find(c) for c in data_part]
    if any(d == -1 for d in data):
        raise ValueError("invalid bech32 character")
    if _bech32_polymod(_bech32_hrp_expand(hrp) + data) != 1:
        raise ValueError("invalid bech32 checksum")
    payload = data[:-6]
    # 5 -> 8 bits, rejecting invalid padding
    acc = 0
    bits = 0
    out = bytearray()
    for v in payload:
        acc = (acc << 5) | v
        bits += 5
        if bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 0xFF)
    if bits >= 5 or ((acc << (8 - bits)) & 0xFF):
        raise ValueError("invalid bech32 padding")
    return hrp, bytes(out)


def _ripemd160(data: bytes) -> bytes:
    """RIPEMD160 via hashlib when OpenSSL still provides it, else pure Python."""
    try:
        return hashlib.new("ripemd160", data).digest()
    except (ValueError, TypeError):
        return ripemd160_pure(data)


def account_address(public_key_compressed: bytes, hrp: str = ACCOUNT_HRP) -> str:
    """Cosmos account address: RIPEMD160(SHA256(compressed secp256k1 pubkey))."""
    return bech32_encode(hrp, _convertbits(_ripemd160(hashlib.sha256(public_key_compressed).digest()), 8, 5))


def node_id(public_key: bytes) -> str:
    """Protocol identity: sha256 of the ed25519 public key (network convention)."""
    return hashlib.sha256(public_key).hexdigest()


@dataclass
class WorkerIdentity:
    protocol_seed: bytes
    account_scalar: bytes
    _protocol_key: ed25519.Ed25519PrivateKey = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.protocol_seed) != 32:
            raise ValueError("protocol seed must be 32 bytes")
        if len(self.account_scalar) != 32:
            raise ValueError("account scalar must be 32 bytes")
        scalar = int.from_bytes(self.account_scalar, "big")
        if not 0 < scalar < SECP256K1_ORDER:
            raise ValueError("account scalar is out of range for secp256k1")
        self._protocol_key = ed25519.Ed25519PrivateKey.from_private_bytes(self.protocol_seed)

    # --- construction -----------------------------------------------------

    @classmethod
    def generate(cls) -> "WorkerIdentity":
        return cls(protocol_seed=_secrets.token_bytes(32), account_scalar=_secrets.token_bytes(32))

    @classmethod
    def from_protocol_hex(cls, protocol_hex: str, account_scalar: bytes) -> "WorkerIdentity":
        """Import an existing operator protocol key (hex, with or without 0x)."""
        raw = bytes.fromhex(protocol_hex[2:] if protocol_hex.startswith("0x") else protocol_hex)
        return cls(protocol_seed=raw, account_scalar=account_scalar)

    # --- public material --------------------------------------------------

    @property
    def protocol_public(self) -> bytes:
        return self._protocol_key.public_key().public_bytes_raw()

    @property
    def protocol_public_b64(self) -> str:
        return base64.b64encode(self.protocol_public).decode()

    @property
    def protocol_public_hex(self) -> str:
        return self.protocol_public.hex()

    @property
    def node_id(self) -> str:
        return node_id(self.protocol_public)

    @property
    def account_public_compressed(self) -> bytes:
        key = ec.derive_private_key(int.from_bytes(self.account_scalar, "big"), ec.SECP256K1())
        return key.public_key().public_bytes(Encoding.X962, PublicFormat.CompressedPoint)

    @property
    def account_address(self) -> str:
        return account_address(self.account_public_compressed)

    @property
    def account_public_b64(self) -> str:
        """Cosmos pubkey encoding (base64 of the compressed point)."""
        return base64.b64encode(self.account_public_compressed).decode()

    # --- signing ----------------------------------------------------------

    def sign_protocol(self, domain: str, payload: object) -> str:
        """Same wire format as prisma_network.core.Identity.sign."""
        return base64.b64encode(
            self._protocol_key.sign(domain.encode() + b"\n" + canonical(payload))).decode()

    def network_key_proof_hex(self, chain_id: str, worker_account: str) -> str:
        """Prove the protocol key belongs to one chain and bonded account."""
        if not chain_id or not worker_account:
            raise ValueError("chain ID and worker account are required")
        payload = {"chain_id": chain_id, "worker": worker_account,
                   "network_public_key": self.protocol_public.hex()}
        return base64.b64decode(self.sign_protocol(_PROTOCOL_BINDING_DOMAIN, payload)).hex()

    # --- records ----------------------------------------------------------

    def public_record(self) -> dict:
        """Everything that is safe to print, log or commit. No secrets."""
        return {
            "node_id": self.node_id,
            "protocol_public_b64": self.protocol_public_b64,
            "protocol_public_hex": self.protocol_public_hex,
            "account_address": self.account_address,
            "account_public_b64": self.account_public_b64,
        }

    def secret_material(self) -> list[str]:
        """Secret representations, for the log-redaction filter to scrub."""
        return [
            self.protocol_seed.hex(),
            base64.b64encode(self.protocol_seed).decode(),
            self.account_scalar.hex(),
        ]
