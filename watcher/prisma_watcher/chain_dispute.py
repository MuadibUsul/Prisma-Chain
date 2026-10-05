"""The prismad adapter for the dispute flow (B3-02): balance, dispute-state
reads and the challenge transaction. The only component that knows the exact
command shapes; everything else is testable without a chain."""

from __future__ import annotations

import base64
import json
import subprocess
import urllib.request
from typing import Callable, Optional

from .chain_tx_shim import provider_keyring, run_tx


class DisputeStateError(Exception):
    """The dispute state could not be read."""


class PrismadDisputeOps:
    def __init__(self, *, account: str, account_scalar_hex: str, prismad: str, chain_id: str,
                 rpc_urls: list[str],
                 runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
                 http_get: Optional[Callable[[str], dict]] = None):
        self.account = account
        self.account_scalar_hex = account_scalar_hex
        self.prismad = prismad
        self.chain_id = chain_id
        self.rpc_urls = rpc_urls
        self._runner = runner
        self._http_get = http_get or self._default_get

    @staticmethod
    def _default_get(url: str) -> dict:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8", "replace"))

    @property
    def rpc(self) -> str:
        return self.rpc_urls[0]

    def height(self) -> int:
        payload = self._http_get(self.rpc.rstrip("/") + "/status")
        return int(payload.get("result", {}).get("sync_info", {}).get("latest_block_height", "0"))

    def _cli(self, *args: str) -> str:
        command = [self.prismad, *args, "--node", self.rpc, "--output", "json"]
        run = self._runner or (lambda cmd, **kw: subprocess.run(cmd, capture_output=True,
                                                                text=True, **kw))
        proc = run(command, timeout=60)
        if proc.returncode != 0:
            raise DisputeStateError(proc.stderr.strip() or f"{' '.join(args)} failed")
        return proc.stdout

    def balance(self, account: str) -> int:
        payload = json.loads(self._cli("query", "bank", "balance", account, "uprsm"))
        return int(payload.get("balance", {}).get("amount", "0"))

    def dispute_state(self, task_id: int) -> dict:
        try:
            raw = json.loads(self._cli("query", "compute", "graph-v2-status",
                                       "--task-id", str(task_id)))
        except DisputeStateError:
            return {}
        blob = raw.get("status_json")
        if blob:
            return json.loads(base64.b64decode(blob))
        return raw

    def open_challenge(self, task_id: int, bond: int, evidence_digest: str) -> str:
        args = ["compute", "start-challenge", "--task-id", str(task_id), "--bond", str(bond)]
        with provider_keyring(self.account_scalar_hex, prismad=self.prismad,
                              chain_id=self.chain_id, rpc_url=self.rpc,
                              runner=self._runner) as keyring:
            return run_tx(keyring, args, prismad=self.prismad, chain_id=self.chain_id,
                          rpc_url=self.rpc, runner=self._runner,
                          waiter=lambda rpc, tx: self._http_get(
                              rpc.rstrip("/") + "/tx?hash=0x" + tx))
