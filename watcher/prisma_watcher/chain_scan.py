"""Chain task discovery for the watcher (B3-01): bounded per-task scan.

The frozen chain exposes per-task queries and no "list completed results"
query, so discovery scans task ids upwards (like the worker's job discovery)
and returns the documents whose status says a result exists. The B6 job API
becomes a second source behind the same interface later; the scan bound is
explicit, never unbounded.
"""

from __future__ import annotations

import base64
import json
import subprocess
import urllib.request
from typing import Callable, Optional

COMPLETED_STATUSES = ("result_submitted", "challenged", "challenger_wins",
                      "challenged_wide", "wide_arb_ready", "finalized", "fraud")


class ChainScanError(Exception):
    """The chain could not be queried."""


class PrismadTaskScan:
    def __init__(self, *, prismad: str, chain_id: str, rpc_urls: list[str], max_scan: int = 512,
                 runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
                 http_get: Optional[Callable[[str], dict]] = None,
                 max_consecutive_missing: int = 64):
        self.prismad = prismad
        self.chain_id = chain_id
        self.rpc_urls = rpc_urls
        self.max_scan = max_scan
        self.max_consecutive_missing = max_consecutive_missing
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

    def _task(self, task_id: int) -> dict:
        command = [self.prismad, "query", "compute", "task", "--task-id", str(task_id),
                   "--node", self.rpc, "--output", "json"]
        run = self._runner or (lambda cmd, **kw: subprocess.run(cmd, capture_output=True,
                                                                text=True, **kw))
        proc = run(command, timeout=60)
        if proc.returncode != 0:
            return {}
        try:
            raw = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {}
        blob = raw.get("task_json")
        if blob:
            return json.loads(base64.b64decode(blob))
        return raw

    def completed_tasks(self, *, start_id: int = 1, max_scan: int = 0) -> list[dict]:
        """Task documents whose status indicates a result to verify."""
        bound = min(max_scan or self.max_scan, self.max_scan)
        found: list[dict] = []
        missing_run = 0
        for task_id in range(max(start_id, 1), max(start_id, 1) + bound):
            document = self._task(task_id)
            if not document:
                missing_run += 1
                if missing_run >= self.max_consecutive_missing:
                    break
                continue
            missing_run = 0
            status = str(document.get("status", ""))
            if status in COMPLETED_STATUSES:
                found.append(document)
        return found
