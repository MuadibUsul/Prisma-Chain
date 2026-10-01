"""Signed discovery, measured routing, fenced leases and one delivery receipt per task."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from pydantic import BaseModel, ConfigDict, Field


def canonical(value: object) -> bytes:
    """The v1 wire signing format: sorted, compact UTF-8 JSON without NaN."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _unb64(value: str) -> bytes:
    return base64.b64decode(value, validate=True)


def node_id(public_key: bytes) -> str:
    return hashlib.sha256(public_key).hexdigest()


class Identity:
    def __init__(self, private_key: Ed25519PrivateKey):
        self._key = private_key
        self.public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.public_key_b64 = base64.b64encode(self.public_key).decode()
        self.node_id = node_id(self.public_key)

    @classmethod
    def from_seed_b64(cls, seed: str) -> "Identity":
        return cls(Ed25519PrivateKey.from_private_bytes(_unb64(seed)))

    @classmethod
    def generate(cls) -> "Identity":
        return cls(Ed25519PrivateKey.generate())

    def sign(self, domain: str, payload: object) -> str:
        return base64.b64encode(self._key.sign(domain.encode() + b"\n" + canonical(payload))).decode()

    def network_key_proof_hex(self, chain_id: str, worker_account: str) -> str:
        """Prove this network key belongs to one chain and bonded account."""
        if not chain_id or not worker_account:
            raise ValueError("chain ID and worker account are required")
        payload = {"chain_id": chain_id, "worker": worker_account,
                   "network_public_key": self.public_key.hex()}
        return base64.b64decode(self.sign("prisma:network-key-binding:v1", payload)).hex()


def verify(public_key_b64: str, domain: str, payload: object, signature_b64: str) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(_unb64(public_key_b64)).verify(
            _unb64(signature_b64), domain.encode() + b"\n" + canonical(payload)
        )
        return True
    except (ValueError, InvalidSignature):
        return False


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelPin(StrictModel):
    """Registry bundle binding weights, tokenizer, runtime image and rules."""
    model_id: str = Field(min_length=1, max_length=128)
    weights_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec_version: str = Field(min_length=1, max_length=64)

    @property
    def model_digest(self) -> str:
        return digest(self.model_dump())


class Capability(StrictModel):
    node_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_key: str
    sequence: int = Field(ge=0)
    issued_at_ms: int = Field(ge=0)
    expires_at_ms: int = Field(ge=0)
    group_id: str = Field(min_length=1, max_length=128)
    chain_worker: str = Field(min_length=1, max_length=128)
    stage_index: int = Field(ge=0, le=127)
    stage_count: int = Field(ge=1, le=128)
    model_id: str = Field(min_length=1, max_length=128)
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec_version: str = Field(min_length=1, max_length=64)
    probe_url: str
    api_url: str | None = None  # Stage 0 worker sidecar, not a raw vLLM endpoint.
    gpu_count: int = Field(ge=0, le=256)
    vram_mb: int = Field(ge=0)


class SignedCapability(StrictModel):
    capability: Capability
    signature: str


class TaskEnvelope(StrictModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    mode: Literal["lightweight", "verifiable"]
    model_id: str
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec_version: str = Field(min_length=1, max_length=64)
    input_commitment: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_ref: str = Field(max_length=2048)
    max_fee: str = Field(pattern=r"^(0|[1-9][0-9]*)$")  # Chain base denomination.
    deadline: int = Field(ge=0)  # Chain height; checked by the chain, not wall time here.
    privacy_tier: Literal["public", "tier0_relative"]


@dataclass(frozen=True)
class Route:
    group_id: str
    api_url: str
    worker_node_id: str
    stage_node_ids: tuple[str, ...]
    stage_chain_workers: tuple[str, ...]
    measured_score: float


class Conflict(ValueError):
    pass


class Unavailable(ValueError):
    pass


class ControlPlane:
    """Single SQLite authority for one testnet control domain.

    A standby process may use the same durable database. A multi-host deployment
    needs a transactional shared store or a chain-backed lease authority.
    """

    def __init__(
        self,
        db_path: str,
        trusted_keys: dict[str, str],
        allowed_hosts: set[str],
        *,
        bound_worker_accounts: dict[str, str] | None = None,
        allow_http: bool = False,
        clock_ms: Callable[[], int] | None = None,
    ):
        self.trusted_keys = trusted_keys
        self.bound_worker_accounts = bound_worker_accounts or {}
        self.allowed_hosts = allowed_hosts
        self.allow_http = allow_http
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self.lock = threading.RLock()
        self.db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS capabilities (
              node_id TEXT PRIMARY KEY, sequence INTEGER NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS probes (
              node_id TEXT PRIMARY KEY, ok INTEGER NOT NULL, latency_ms REAL NOT NULL,
              bandwidth_mbps REAL NOT NULL, queue_depth INTEGER NOT NULL, checked_at_ms INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS leases (
              task_id TEXT PRIMARY KEY, group_id TEXT NOT NULL, owner_id TEXT NOT NULL,
              epoch INTEGER NOT NULL, expires_at_ms INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
              attempt_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, epoch INTEGER NOT NULL,
              status TEXT NOT NULL, created_at_ms INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS receipts (
              task_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL, data TEXT NOT NULL);
        """)

    def _check_url(self, value: str) -> None:
        url = urlsplit(value)
        if url.scheme not in ({"http", "https"} if self.allow_http else {"https"}):
            raise ValueError("node URL must use HTTPS")
        if not url.hostname or url.hostname not in self.allowed_hosts or url.username or url.password or url.fragment:
            raise ValueError("node URL host is not allowed")

    def announce(self, signed: SignedCapability) -> None:
        c = signed.capability
        try:
            pub = _unb64(c.public_key)
        except ValueError as exc:
            raise ValueError("invalid public key") from exc
        if len(pub) != 32 or node_id(pub) != c.node_id or self.trusted_keys.get(c.node_id) != c.public_key:
            raise ValueError("unregistered node key")
        if self.bound_worker_accounts and self.bound_worker_accounts.get(c.node_id) != c.chain_worker:
            raise ValueError("node key is not bound to the announced chain worker")
        if not verify(c.public_key, "prisma:capability:v1", c.model_dump(), signed.signature):
            raise ValueError("invalid capability signature")
        now = self.clock_ms()
        if c.issued_at_ms > now + 5_000 or c.expires_at_ms <= now or not 0 < c.expires_at_ms - c.issued_at_ms <= 60_000:
            raise ValueError("invalid capability lease")
        if c.stage_index >= c.stage_count or (c.stage_index == 0) != (c.api_url is not None):
            raise ValueError("invalid pipeline stage")
        pin = ModelPin(model_id=c.model_id, weights_digest=c.weights_digest,
                       tokenizer_digest=c.tokenizer_digest, runtime_digest=c.runtime_digest,
                       spec_version=c.spec_version)
        if c.model_digest != pin.model_digest:
            raise ValueError("capability model digest does not bind the artifact bundle")
        self._check_url(c.probe_url)
        if c.api_url:
            self._check_url(c.api_url)
        data = signed.model_dump_json()
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                old = self.db.execute("SELECT sequence, data FROM capabilities WHERE node_id=?", (c.node_id,)).fetchone()
                if old and c.sequence <= old["sequence"]:
                    if c.sequence == old["sequence"] and json.loads(old["data"]) == json.loads(data):
                        self.db.execute("COMMIT")
                        return
                    raise Conflict("stale or equivocated capability sequence")
                if old:
                    previous = SignedCapability.model_validate_json(old["data"]).capability.model_dump()
                    current = c.model_dump()
                    for ephemeral in ("sequence", "issued_at_ms", "expires_at_ms"):
                        previous.pop(ephemeral)
                        current.pop(ephemeral)
                    if previous != current:
                        self.db.execute("DELETE FROM probes WHERE node_id=?", (c.node_id,))
                self.db.execute("INSERT OR REPLACE INTO capabilities VALUES (?,?,?)", (c.node_id, c.sequence, data))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise

    def announcements(self) -> list[SignedCapability]:
        now = self.clock_ms()
        with self.lock:
            rows = self.db.execute("SELECT data FROM capabilities").fetchall()
        return [SignedCapability.model_validate_json(row["data"]) for row in rows
                if json.loads(row["data"])["capability"]["expires_at_ms"] > now]

    def probe(self, node: str, *, ok: bool, latency_ms: float, bandwidth_mbps: float,
              queue_depth: int = 0, expected_sequence: int | None = None) -> bool:
        if node not in self.trusted_keys or not all(math.isfinite(x) and x >= 0 for x in (latency_ms, bandwidth_mbps)) or queue_depth < 0:
            raise ValueError("invalid probe")
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                current = self.db.execute("SELECT sequence FROM capabilities WHERE node_id=?", (node,)).fetchone()
                if not current or (expected_sequence is not None and expected_sequence != current["sequence"]):
                    self.db.execute("COMMIT")
                    return False
                self.db.execute("INSERT OR REPLACE INTO probes VALUES (?,?,?,?,?,?)",
                                (node, int(ok), latency_ms, bandwidth_mbps, queue_depth, self.clock_ms()))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise
        return True

    def route(self, model_id: str, model_digest: str, spec_version: str) -> Route:
        now = self.clock_ms()
        with self.lock:
            rows = self.db.execute("""SELECT c.data, p.ok, p.latency_ms, p.bandwidth_mbps,
                p.queue_depth, p.checked_at_ms FROM capabilities c LEFT JOIN probes p USING(node_id)""").fetchall()
        groups: dict[str, list[tuple[Capability, sqlite3.Row]]] = {}
        for row in rows:
            c = SignedCapability.model_validate_json(row["data"]).capability
            if (c.model_id == model_id and c.model_digest == model_digest and c.spec_version == spec_version
                    and c.expires_at_ms > now
                    and row["ok"] == 1 and row["checked_at_ms"] is not None
                    and now - row["checked_at_ms"] <= 15_000 and row["bandwidth_mbps"] > 0):
                groups.setdefault(c.group_id, []).append((c, row))
        choices: list[Route] = []
        for group_id, members in groups.items():
            count = members[0][0].stage_count
            if (len(members) != count or {c.stage_index for c, _ in members} != set(range(count))
                    or len({(c.stage_count, c.runtime_digest, c.tokenizer_digest,
                             c.weights_digest, c.spec_version, c.model_digest) for c, _ in members}) != 1):
                continue
            members.sort(key=lambda item: item[0].stage_index)
            head = members[0][0]
            if not head.api_url:
                continue
            score = max(row["latency_ms"] for _, row in members) + 50 * sum(row["queue_depth"] for _, row in members)
            score += 1000 / min(row["bandwidth_mbps"] for _, row in members)
            choices.append(Route(group_id, head.api_url, head.node_id,
                                 tuple(c.node_id for c, _ in members),
                                 tuple(c.chain_worker for c, _ in members), score))
        if not choices:
            raise Unavailable("no complete, healthy, measured compute group")
        return min(choices, key=lambda route: (route.measured_score, route.group_id))

    def acquire(self, task_id: str, group_id: str, owner_id: str, ttl_ms: int = 30_000) -> dict:
        if owner_id not in self.trusted_keys or not 1_000 <= ttl_ms <= 60_000:
            raise ValueError("invalid lease request")
        now = self.clock_ms()
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                old = self.db.execute("SELECT * FROM leases WHERE task_id=?", (task_id,)).fetchone()
                if old and old["expires_at_ms"] > now:
                    if old["owner_id"] != owner_id or old["group_id"] != group_id:
                        raise Conflict("active lease belongs to another coordinator or group")
                    epoch = old["epoch"]
                else:
                    epoch = old["epoch"] + 1 if old else 1
                    if old:
                        self.db.execute("UPDATE attempts SET status='failed' WHERE task_id=? AND status='running'",
                                        (task_id,))
                expiry = now + ttl_ms
                self.db.execute("INSERT OR REPLACE INTO leases VALUES (?,?,?,?,?)",
                                (task_id, group_id, owner_id, epoch, expiry))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise
        return {"task_id": task_id, "group_id": group_id, "owner_id": owner_id,
                "epoch": epoch, "expires_at_ms": expiry}

    def lease(self, task_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM leases WHERE task_id=?", (task_id,)).fetchone()
        return dict(row) if row else None

    def check_fence(self, task_id: str, group_id: str, owner_id: str, epoch: int) -> bool:
        lease = self.lease(task_id)
        return bool(lease and lease["group_id"] == group_id and lease["owner_id"] == owner_id
                    and lease["epoch"] == epoch and lease["expires_at_ms"] > self.clock_ms())

    def start_attempt(self, task_id: str, group_id: str, owner_id: str, epoch: int, attempt_id: str) -> None:
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                lease = self.db.execute("SELECT * FROM leases WHERE task_id=?", (task_id,)).fetchone()
                if (not lease or lease["group_id"] != group_id or lease["owner_id"] != owner_id
                        or lease["epoch"] != epoch or lease["expires_at_ms"] <= self.clock_ms()):
                    raise Conflict("stale coordinator fence")
                if self.db.execute("SELECT 1 FROM receipts WHERE task_id=?", (task_id,)).fetchone():
                    raise Conflict("task already has a delivery receipt")
                active = self.db.execute("SELECT 1 FROM attempts WHERE task_id=? AND status='running'", (task_id,)).fetchone()
                if active:
                    raise Conflict("task already has a running attempt")
                self.db.execute("INSERT INTO attempts VALUES (?,?,?,?,?)",
                                (attempt_id, task_id, epoch, "running", self.clock_ms()))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise

    def finish_attempt(self, attempt_id: str, success: bool) -> None:
        with self.lock:
            self.db.execute("UPDATE attempts SET status=? WHERE attempt_id=? AND status='running'",
                            ("delivered" if success else "failed", attempt_id))

    def commit_receipt(self, task_id: str, attempt_id: str, group_id: str, owner_id: str,
                       epoch: int, receipt: dict) -> dict:
        encoded = canonical(receipt).decode()
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                old = self.db.execute("SELECT data FROM receipts WHERE task_id=?", (task_id,)).fetchone()
                if old:
                    if old["data"] != encoded:
                        raise Conflict("task already has a different delivery receipt")
                    self.db.execute("COMMIT")
                    return json.loads(old["data"])
                lease = self.db.execute("SELECT * FROM leases WHERE task_id=?", (task_id,)).fetchone()
                attempt = self.db.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
                if (not lease or lease["group_id"] != group_id or lease["owner_id"] != owner_id
                        or lease["epoch"] != epoch or lease["expires_at_ms"] <= self.clock_ms()
                        or not attempt or attempt["task_id"] != task_id or attempt["epoch"] != epoch
                        or attempt["status"] != "running"):
                    raise Conflict("stale fence or attempt")
                self.db.execute("INSERT INTO receipts VALUES (?,?,?)", (task_id, attempt_id, encoded))
                self.db.execute("UPDATE attempts SET status='delivered' WHERE attempt_id=?", (attempt_id,))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise
        return receipt

    def receipt(self, task_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT data FROM receipts WHERE task_id=?", (task_id,)).fetchone()
        return json.loads(row["data"]) if row else None
