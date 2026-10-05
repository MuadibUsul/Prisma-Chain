"""Chain client for the worker (B2-03): status, queries and transactions.

Reads go through two layers:

- **HTTP RPC** for liveness and chain-id validation (``/status``), with
  explicit fail-over across configured endpoints — a failure is logged with
  the endpoint that failed, never swallowed;
- **``prismad`` CLI** for module queries and transactions, which is the
  same path every Phase D/E driver uses and the only one that understands
  the frozen module messages.

Transactions need the chain account key. The canonical secret lives in the
encrypted keystore (B2-01); for the duration of a transaction the key is
imported into a **temporary 0700 keyring** that is shredded immediately
afterwards. That bounded exposure is deliberate and documented: headless
worker hosts have no OS keyring, and the alternative (a persistent copy)
would be worse. See docs/productization/b2-03.md.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional


class ChainError(Exception):
    """Base class for chain interaction failures."""


class ChainMismatchError(ChainError):
    """The node belongs to a different chain than the worker is configured for."""


class ProtocolUnsupportedError(ChainError):
    """The chain does not expose the frozen compute surface."""


class InsufficientBalanceError(ChainError):
    """The account cannot pay for the requested operation (actionable)."""


class TransactionError(ChainError):
    """A transaction was rejected or never included."""


@dataclass
class ChainConfig:
    rpc_urls: list[str]
    chain_id: str
    prismad: str = "prismad"
    fees: str = "0uprsm"
    gas: str = "2000000"
    tx_timeout_seconds: float = 90.0


@dataclass
class FrozenSurface:
    """What the chain answered on the frozen compute query surface."""

    worker_query_fields: list[str]
    bonded_uprsm: int
    network_public_key_b64: str = ""
    notes: list[str] = field(default_factory=list)


def _default_http_get(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def _default_runner(cmd: list[str], timeout: float = 60.0) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise ChainError(f"{' '.join(cmd[:3])}… failed: {proc.stderr.strip() or proc.returncode}")
    return proc.stdout


class ChainClient:
    def __init__(self, config: ChainConfig, *, runner: Optional[Callable[[list[str]], str]] = None,
                 http_get: Optional[Callable[[str], str]] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.config = config
        self._runner = runner or _default_runner
        self._http_get = http_get or _default_http_get
        self._log = log or (lambda message: None)
        self._active_rpc: Optional[str] = None

    # --- RPC -----------------------------------------------------------------

    def status(self) -> dict:
        errors: list[str] = []
        for url in self.config.rpc_urls:
            try:
                payload = json.loads(self._http_get(url.rstrip("/") + "/status"))
            except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
                self._log(f"rpc endpoint {url} failed: {exc}")
                errors.append(f"{url}: {exc}")
                continue
            self._active_rpc = url
            return payload.get("result", {})
        raise ChainError("no reachable RPC endpoint: " + "; ".join(errors))

    @property
    def active_rpc(self) -> str:
        if self._active_rpc:
            return self._active_rpc
        self.status()
        return self._active_rpc or self.config.rpc_urls[0]

    def validate_chain_id(self) -> str:
        node = self.status().get("node_info", {})
        network = node.get("network", "")
        if network != self.config.chain_id:
            raise ChainMismatchError(
                f"refusing to join chain {network!r}: worker is configured for {self.config.chain_id!r}")
        return network

    # --- CLI reads -----------------------------------------------------------

    def cli(self, *args: str, timeout: float = 60.0) -> str:
        cmd = [self.config.prismad, *args, "--node", self.active_rpc, "--output", "json"]
        return self._runner(cmd, timeout=timeout)

    def balance_uprsm(self, address: str) -> int:
        out = self.cli("query", "bank", "balance", address, "uprsm")
        payload = json.loads(out)
        return int(payload.get("balance", {}).get("amount", "0"))

    def task(self, task_id: int) -> dict:
        """Task query; a genuinely missing task returns {} (keeper: "task N not found").

        The frozen answer is a base64 ``task_json`` blob (QueryTaskResponse
        carries bytes), not a flat object — decoded here so callers see the
        task document.
        """
        try:
            payload = json.loads(self.cli("query", "compute", "task", "--task-id", str(task_id)))
        except ChainError as exc:
            if "not found" in str(exc):
                return {}
            raise
        return decode_blob(payload, "task_json")

    def model(self, model_id: str, spec_version: str) -> dict:
        """Model query, same base64 ``model_json`` envelope as tasks."""
        try:
            payload = json.loads(self.cli("query", "compute", "model", "--model-id", model_id,
                                          "--spec-version", spec_version))
        except ChainError as exc:
            if "not found" in str(exc):
                return {}
            raise
        return decode_blob(payload, "model_json")

    def query_worker(self, address: str) -> dict:
        """The autocli query takes the address as a flag, not positionally."""
        payload = json.loads(self.cli("query", "compute", "worker", "--worker", address))
        return payload

    def check_frozen_surface(self, address: str) -> FrozenSurface:
        """Protocol-support proxy: the frozen compute query surface must answer.

        The frozen protocol has no version query, so this checks the surface
        the frozen release depends on (``query compute worker`` with its
        documented fields) and records what it saw. A chain without the
        compute module fails the command and is refused.
        """
        try:
            raw = self.cli("query", "compute", "worker", "--worker", address)
            payload = json.loads(raw)
        except (ChainError, json.JSONDecodeError) as exc:
            raise ProtocolUnsupportedError(
                f"the chain does not expose the frozen compute surface: {exc}") from exc
        if not isinstance(payload, dict):
            raise ProtocolUnsupportedError(
                f"compute worker query answered {type(payload).__name__}, not a JSON object")
        # proto3 JSON omits zero values, so a fresh worker legitimately answers
        # {} — the check is that the module serves the frozen query, not that
        # the fields are present yet.
        notes = ["frozen compute surface present (no on-chain version query exists in the frozen "
                 "protocol; chain-id + the served query surface are the enforceable checks); "
                 "proto3 omits zero values, so a fresh worker answers {}"]
        return FrozenSurface(worker_query_fields=sorted(payload),
                             bonded_uprsm=int(payload.get("bonded_uprsm", 0) or 0),
                             network_public_key_b64=payload.get("network_public_key", "") or "",
                             notes=notes)

    # --- transactions --------------------------------------------------------

    @contextmanager
    def _temporary_keyring(self, account_scalar_hex: str):
        """A 0700 keyring that exists only for the duration of one transaction."""
        root = pathlib.Path(tempfile.mkdtemp(prefix="prisma-worker-tx-"))
        try:
            os.chmod(root, 0o700)
            config_dir = root / "config"
            config_dir.mkdir(mode=0o700)
            (config_dir / "client.toml").write_text(
                f'chain-id = "{self.config.chain_id}"\n'
                f'keyring-backend = "test"\n'
                f'keyring-dir = "{root.as_posix()}"\n'
                f'node = "{self.active_rpc}"\n'
                'output = "json"\n'
                'broadcast-mode = "sync"\n', encoding="utf-8")
            self._runner([self.config.prismad, "keys", "import-hex", "worker",
                          account_scalar_hex, "--keyring-backend", "test",
                          "--keyring-dir", str(root), "--home", str(root)], timeout=60)
            yield root
        finally:
            _shred_tree(root)

    def _tx(self, homes: pathlib.Path, *args: str) -> str:
        cmd = [self.config.prismad, "tx", *args,
               "--from", "worker",
               "--chain-id", self.config.chain_id,
               "--node", self.active_rpc,
               "--keyring-backend", "test", "--keyring-dir", str(homes), "--home", str(homes),
               "--fees", self.config.fees, "--gas", self.config.gas,
               "--yes", "--output", "json"]
        out = self._runner(cmd, timeout=120)
        payload = json.loads(out)
        if int(payload.get("code", 1)) != 0 or not payload.get("txhash"):
            raise TransactionError(f"transaction rejected: {payload}")
        return payload["txhash"]

    def wait_tx(self, txhash: str) -> dict:
        deadline = time.monotonic() + self.config.tx_timeout_seconds
        last: object = "not found in the mempool yet"
        while time.monotonic() < deadline:
            try:
                payload = json.loads(self._http_get(
                    self.active_rpc.rstrip("/") + "/tx?hash=0x" + txhash))
                result = payload.get("result")
                if result is None:
                    last = payload.get("error", "no result")
                else:
                    code = int(result.get("tx_result", {}).get("code", 1))
                    if code != 0:
                        raise TransactionError(
                            f"tx {txhash} failed with code {code}: {result.get('tx_result', {}).get('log', '')}")
                    return result
            except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
                last = exc
            time.sleep(0.5)
        raise TransactionError(f"tx {txhash} was not included within "
                               f"{self.config.tx_timeout_seconds:.0f}s (last: {last})")

    def accept_task(self, task_id: int, account_scalar_hex: str) -> str:
        with self._temporary_keyring(account_scalar_hex) as root:
            txhash = self._tx(root, "compute", "accept-task", "--task-id", str(task_id))
        self.wait_tx(txhash)
        return txhash

    def bond_worker(self, *, amount_uprsm: int, network_public_key_hex: str,
                    network_key_proof_hex: str, account_scalar_hex: str) -> str:
        """Bond (or top up) and bind the protocol key — one frozen transaction.

        Amount 0 with no key is refused by the chain; a top-up omits the key.
        """
        with self._temporary_keyring(account_scalar_hex) as root:
            args = ["compute", "bond-worker", "--amount", str(amount_uprsm)]
            if network_public_key_hex:
                args += ["--network-public-key", network_public_key_hex,
                         "--network-key-proof", network_key_proof_hex]
            txhash = self._tx(root, *args)
        self.wait_tx(txhash)
        return txhash


def _shred_tree(root: pathlib.Path) -> None:
    """Overwrite and delete the temporary keyring (best effort, then remove)."""
    try:
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    size = path.stat().st_size
                    with open(path, "r+b") as handle:
                        handle.write(b"\x00" * size)
                        handle.flush()
                        os.fsync(handle.fileno())
                except OSError:
                    pass
    finally:
        shutil.rmtree(root, ignore_errors=True)


def decode_blob(payload: dict, key: str) -> dict:
    """Decode a ``{key: <base64 json>}`` module query answer into its document.

    Accepts an already-flat document too, so a future chain/proto change that
    drops the envelope does not silently break discovery.
    """
    if not isinstance(payload, dict):
        raise ChainError(f"query answered {type(payload).__name__}, not a JSON object")
    if key not in payload:
        return payload
    raw = payload[key]
    if not raw:
        return {}
    try:
        return json.loads(base64.b64decode(raw, validate=True))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ChainError(f"query field {key} is not base64 JSON: {exc}") from exc


def decode_network_key(b64_value: str) -> bytes:
    """The chain returns the bound network key base64-encoded."""
    return base64.b64decode(b64_value, validate=True) if b64_value else b""
