"""Bundle retrieval and the frozen-backed verifier (B3-01).

- ``HttpBundleSource`` fetches a bundle from DA providers (the B4-01 surface:
  ``GET /v1/da/{task_id}/output`` returns the canonical blob as hex), trying
  providers in order, with a size bound and the task id checked before the
  file is accepted.
- ``FrozenVerifier`` adapts the frozen ``WatcherV2`` to the daemon's verifier
  interface. The verdict mapping is conservative: a report that says
  fraud/fail counts as fraud, clean/pass counts as clean, anything else is
  ``unknown`` — and the full frozen report is kept in the journal either way.
"""

from __future__ import annotations

import json
import pathlib
import urllib.error
import urllib.request
from typing import Optional

from .frozen import frozen


class RetrievalError(Exception):
    """The bundle could not be retrieved from any provider."""


class HttpBundleSource:
    def __init__(self, providers: list[str], *, timeout: float = 30.0,
                 max_bytes: int = 64 * 1024 * 1024,
                 http_get: Optional[callable] = None):
        self.providers = [provider.rstrip("/") for provider in providers]
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._http_get = http_get or self._default_get

    @staticmethod
    def _default_get(url: str, timeout: float):
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()

    def fetch(self, task: dict, destination: pathlib.Path) -> pathlib.Path:
        task_id = int(task.get("id", 0))
        errors: list[str] = []
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "bundle.hex"
        for provider in self.providers:
            url = f"{provider}/v1/da/{task_id}/output"
            try:
                raw = self._http_get(url, self.timeout)
            except (urllib.error.URLError, OSError) as exc:
                errors.append(f"{provider}: {exc}")
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"{provider}: response is not JSON: {exc}")
                continue
            if int(payload.get("task_id", -1)) != task_id:
                errors.append(f"{provider}: answered for task {payload.get('task_id')}")
                continue
            blob_hex = str(payload.get("c_hex", ""))
            if not blob_hex:
                errors.append(f"{provider}: empty bundle")
                continue
            if len(blob_hex) // 2 > self.max_bytes:
                errors.append(f"{provider}: bundle exceeds the {self.max_bytes} byte bound")
                continue
            blob = bytes.fromhex(blob_hex)
            target.write_bytes(blob)
            # the frozen watcher expects the raw bundle bytes
            bundle_path = destination / "bundle.bin"
            bundle_path.write_bytes(blob)
            target.unlink()
            return bundle_path
        raise RetrievalError("no provider returned the bundle: " + "; ".join(errors))


class FrozenVerifier:
    """Runs the frozen WatcherV2 over a downloaded bundle."""

    def __init__(self, *, test_seed: Optional[int] = None, log=lambda _m: None):
        self.test_seed = test_seed
        self.log = log

    def verify(self, bundle_path: pathlib.Path, task: dict) -> dict:
        watcher_class = frozen.watcher_class()
        watcher = watcher_class(bundle_path=pathlib.Path(bundle_path), task=task,
                                test_seed=self.test_seed)
        report = watcher.run()
        verdict_text = " ".join(str(report.get(key, "")) for key in
                                ("verdict", "status", "result", "outcome")).lower()
        if any(word in verdict_text for word in ("fraud", "fail", "invalid", "mismatch")):
            verdict = "fraud"
        elif any(word in verdict_text for word in ("clean", "pass", "ok", "honest", "consistent")):
            verdict = "clean"
        else:
            verdict = "unknown"
        return {"verdict": verdict, "detail": json.dumps(report)[:2000], "report": report}
