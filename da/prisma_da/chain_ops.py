"""The ``prismad`` CLI adapter for the responder's ChainOps interface.

Everything that knows a command shape lives here; the responder logic stays
testable without a chain. Responses use the real frozen transaction:

    prismad tx compute respond-graph-da-challenge \
        --challenge-id ID --chunk <tile hex> --chunk-index i --chunk-count n \
        --chunk-proof <sibling hex> (repeated) --from provider ...

Challenge **discovery** scans tasks (the frozen chain has per-task queries and
no per-provider challenge list), so the scan bound is explicit. The task
document's DA section is read defensively: a challenge counts when it names
this provider and carries a deadline; the exact field spelling is pinned by
the live drill, and this adapter is the single place that would change.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import urllib.request
from typing import Callable, Optional

from .chain_tx import ChainTxError, provider_keyring, run_tx, wait_for_inclusion
from .storage import ArtifactIndex

_CHALLENGE_KEYS = ("da_challenge", "graph_da_challenge", "challenge", "da")
_DEADLINE_KEYS = ("deadline_height", "deadline", "expires_height")
_PROVIDER_KEYS = ("provider", "da_provider")


class PrismadChainOps:
    def __init__(self, *, account: str, account_scalar_hex: str, prismad: str, chain_id: str,
                 rpc_urls: list[str], storage: ArtifactIndex, max_scan: int = 512,
                 runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
                 http_get: Optional[Callable[[str], dict]] = None):
        self.account = account
        self.account_scalar_hex = account_scalar_hex
        self.prismad = prismad
        self.chain_id = chain_id
        self.rpc_urls = rpc_urls
        self.storage = storage
        self.max_scan = max_scan
        self._runner = runner
        self._http_get = http_get or self._default_get

    # --- chain reads ------------------------------------------------------

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
        proc = (self._runner or (lambda c, **kw: subprocess.run(c, capture_output=True,
                                                                text=True, **kw)))(command, timeout=60)
        if proc.returncode != 0:
            raise ChainTxError(proc.stderr.strip() or f"{' '.join(args)} failed")
        return proc.stdout

    def task_document(self, task_id: int) -> dict:
        try:
            raw = json.loads(self._cli("query", "compute", "task", "--task-id", str(task_id)))
        except ChainTxError:
            return {}
        blob = raw.get("task_json")
        if blob:
            import base64
            return json.loads(base64.b64decode(blob))
        return raw

    def open_challenges(self, provider: str) -> list[dict]:
        """Scan tasks for challenges naming this provider (bounded scan)."""
        challenges: list[dict] = []
        missing_run = 0
        for task_id in range(1, self.max_scan + 1):
            document = self.task_document(task_id)
            if not document:
                missing_run += 1
                if missing_run >= 64:
                    break
                continue
            missing_run = 0
            challenge = self._extract_challenge(document)
            if challenge and challenge.get("provider") in ("", provider):
                challenges.append(challenge)
        return challenges

    @staticmethod
    def _extract_challenge(document: dict) -> Optional[dict]:
        for key in _CHALLENGE_KEYS:
            block = document.get(key)
            if not isinstance(block, dict):
                continue
            status = str(block.get("status", "open"))
            if status not in ("open", "pending"):
                continue
            deadline = 0
            for deadline_key in _DEADLINE_KEYS:
                if block.get(deadline_key) is not None:
                    deadline = int(block[deadline_key])
                    break
            provider = ""
            for provider_key in _PROVIDER_KEYS:
                if block.get(provider_key):
                    provider = str(block[provider_key])
                    break
            if not deadline:
                continue
            return {"task_id": int(document.get("id", 0)), "challenge_id":
                    int(block.get("id", block.get("challenge_id", 0)) or 0),
                    "provider": provider, "deadline_height": deadline,
                    "chunk_index": int(block.get("chunk_index", block.get("index", 0)) or 0)}
        return None

    # --- transaction ------------------------------------------------------

    def respond(self, challenge: dict, tile: dict) -> str:
        proof = tile.get("proof") or {}
        args = ["compute", "respond-graph-da-challenge",
                "--challenge-id", str(int(challenge.get("challenge_id", 0))),
                "--chunk", str(tile["tile"]),
                "--chunk-index", str(int(proof.get("index", 0))),
                "--chunk-count", str(int(proof.get("count", 1)))]
        for sibling in proof.get("siblings", []):
            args += ["--chunk-proof", str(sibling)]
        def waiter(rpc_url: str, txhash: str) -> dict:
            payload = self._http_get(rpc_url.rstrip("/") + "/tx?hash=0x" + txhash)
            result = payload.get("result")
            if result is None:
                raise ChainTxError(f"tx {txhash} not found on {rpc_url}")
            code = int(result.get("tx_result", {}).get("code", 1))
            if code != 0:
                raise ChainTxError(f"tx {txhash} failed with code {code}")
            return result

        with provider_keyring(self.account_scalar_hex, prismad=self.prismad,
                              chain_id=self.chain_id, rpc_url=self.rpc,
                              runner=self._runner) as keyring:
            return run_tx(keyring, args, prismad=self.prismad, chain_id=self.chain_id,
                          rpc_url=self.rpc, runner=self._runner, waiter=waiter)
