"""Minimal chain transaction helper for the DA daemon (B4-01 register).

Same mechanism the worker uses for its own transactions: the provider key is
imported into a temporary 0700 keyring for the duration of one transaction and
that keyring is shredded immediately afterwards. Consolidating this into one
shared client is B5's job (prisma-cli); until then the duplication is explicit.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Callable, Optional


class ChainTxError(Exception):
    """A chain transaction failed or was never included."""


def _http_get(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def wait_for_inclusion(rpc_url: str, txhash: str, *, timeout: float = 90.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    last: object = "not found yet"
    while time.monotonic() < deadline:
        try:
            payload = _http_get(rpc_url.rstrip("/") + "/tx?hash=0x" + txhash)
            result = payload.get("result")
            if result is not None:
                code = int(result.get("tx_result", {}).get("code", 1))
                if code != 0:
                    raise ChainTxError(
                        f"tx {txhash} failed with code {code}: "
                        f"{result.get('tx_result', {}).get('log', '')}")
                return result
            last = payload.get("error", last)
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            last = exc
        time.sleep(0.5)
    raise ChainTxError(f"tx {txhash} was not included within {timeout:.0f}s (last: {last})")


@contextmanager
def provider_keyring(account_scalar_hex: str, *, prismad: str, chain_id: str, rpc_url: str):
    root = pathlib.Path(tempfile.mkdtemp(prefix="prisma-da-tx-"))
    try:
        (root / "config").mkdir(mode=0o700)
        (root / "config" / "client.toml").write_text(
            f'chain-id = "{chain_id}"\nkeyring-backend = "test"\n'
            f'keyring-dir = "{root.as_posix()}"\nnode = "{rpc_url}"\n'
            'output = "json"\nbroadcast-mode = "sync"\n', encoding="utf-8")
        subprocess.run([prismad, "keys", "import-hex", "provider", account_scalar_hex,
                        "--keyring-backend", "test", "--keyring-dir", str(root),
                        "--home", str(root)], check=True, capture_output=True, text=True)
        yield root
    finally:
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    size = path.stat().st_size
                    with open(path, "r+b") as handle:
                        handle.write(b"\x00" * size)
                except OSError:
                    pass
        shutil.rmtree(root, ignore_errors=True)


def run_tx(keyring: pathlib.Path, args: list[str], *, prismad: str, chain_id: str,
           rpc_url: str, fees: str = "0uprsm", gas: str = "2000000",
           runner: Optional[Callable[..., subprocess.CompletedProcess]] = None) -> str:
    """Submit a transaction from the temporary provider keyring; returns the tx hash."""
    command = [prismad, "tx", *args, "--from", "provider", "--chain-id", chain_id,
               "--node", rpc_url, "--keyring-backend", "test", "--keyring-dir", str(keyring),
               "--home", str(keyring), "--fees", fees, "--gas", gas, "--yes", "--output", "json"]
    run = runner or (lambda cmd, **kw: subprocess.run(cmd, capture_output=True, text=True, **kw))
    proc = run(command, timeout=120)
    if proc.returncode != 0:
        raise ChainTxError(f"{' '.join(args)} failed: {proc.stderr.strip()}")
    payload = json.loads(proc.stdout)
    if int(payload.get("code", 1)) != 0 or not payload.get("txhash"):
        raise ChainTxError(f"transaction rejected: {payload}")
    txhash = payload["txhash"]
    wait_for_inclusion(rpc_url, txhash)
    return txhash
