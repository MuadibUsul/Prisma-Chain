"""Shared plumbing for the prisma CLI command groups.

Ownership is stated honestly per command (the roadmap requires it): some
groups drive chain transactions through ``prismad``, some drive the product
daemons' own CLIs, and the API groups (B6) are HTTP clients. Nothing here
re-implements protocol logic: chain ops delegate to ``prismad``, wallet rules
to ``prisma_worker``, version metadata to ``prismad version``.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional


class CliError(Exception):
    """User-facing failure: printed, exit code 1."""


@dataclass
class GlobalConfig:
    """The operator's default endpoints and binaries (one file, reused)."""

    chain_id: str = ""
    node: str = "http://127.0.0.1:26657"
    api: str = ""                    # the Job API base (B6); empty = not configured
    prismad: str = "prismad"
    keystore: str = ""               # operator keystore (prisma_worker rules)
    faucet: str = ""                 # faucet endpoint (B7)

    @staticmethod
    def default_path() -> pathlib.Path:
        override = os.environ.get("PRISMA_CLI_CONFIG")
        if override:
            return pathlib.Path(override).expanduser()
        return pathlib.Path.home() / ".prisma" / "cli.json"

    def save(self, path: Optional[pathlib.Path] = None) -> pathlib.Path:
        path = pathlib.Path(path or self.default_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Optional[pathlib.Path] = None) -> "GlobalConfig":
        path = pathlib.Path(path or cls.default_path())
        if not path.exists():
            return cls()
        document = json.loads(path.read_text(encoding="utf-8"))
        known = {k: v for k, v in document.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.chain_id and not self.chain_id.startswith("prisma-"):
            problems.append(f"chain_id {self.chain_id!r} does not look like a prisma chain id")
        if self.node and not self.node.startswith(("http://", "https://")):
            problems.append(f"node {self.node!r} is not an HTTP(S) URL")
        if self.api and not self.api.startswith(("http://", "https://")):
            problems.append(f"api {self.api!r} is not an HTTP(S) URL")
        return problems


def http_json(method: str, url: str, *, body: Optional[dict] = None,
              headers: Optional[dict] = None, timeout: float = 15.0) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            return response.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"error": raw[:300]}
    except urllib.error.URLError as exc:
        raise CliError(f"{url}: {exc}") from exc


def prismad(config: GlobalConfig, *args: str, runner: Optional[Callable] = None) -> dict:
    """Run a prismad query/tx and return its JSON (the chain surface)."""
    command = [config.prismad, *args, "--node", config.node, "--output", "json"]
    run = runner or (lambda cmd, **kw: subprocess.run(cmd, capture_output=True, text=True, **kw))
    proc = run(command, timeout=120)
    if proc.returncode != 0:
        raise CliError(proc.stderr.strip() or f"{' '.join(args)} failed")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise CliError(f"prismad answered non-JSON: {exc}") from exc


def emit(payload: dict, as_json: bool, *, fallback: str = "") -> None:
    if as_json:
        print(json.dumps(payload, indent=1))
    else:
        if payload:
            for key, value in payload.items():
                print(f"{key:22s} {value}")
        elif fallback:
            print(fallback)
