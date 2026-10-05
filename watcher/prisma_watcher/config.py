"""Watcher daemon configuration (B3-01)."""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import asdict, dataclass, field


@dataclass
class ChainConfig:
    rpc_urls: list[str] = field(default_factory=lambda: ["http://127.0.0.1:26657"])
    chain_id: str = ""
    prismad: str = "prismad"


@dataclass
class DaemonConfig:
    account: str = ""                       # challenger account (bech32)
    keystore: str = ""
    journal_dir: str = ""
    providers: list[str] = field(default_factory=list)   # DA endpoints for retrieval
    chain: ChainConfig = field(default_factory=ChainConfig)
    interval_seconds: float = 15.0
    max_scan: int = 512

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, document: dict) -> "DaemonConfig":
        chain = ChainConfig(**{k: v for k, v in (document.get("chain") or {}).items()
                               if k in ChainConfig.__dataclass_fields__})
        known = {k: v for k, v in document.items() if k in cls.__dataclass_fields__ and k != "chain"}
        return cls(chain=chain, **known)

    def save(self, path: pathlib.Path | str) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "DaemonConfig":
        return cls.from_dict(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def default_path() -> pathlib.Path:
        override = os.environ.get("PRISMA_WATCHER_CONFIG")
        if override:
            return pathlib.Path(override).expanduser()
        return pathlib.Path.home() / ".prisma-watcher" / "config.json"

    def resolved_journal_dir(self) -> pathlib.Path:
        if self.journal_dir:
            return pathlib.Path(self.journal_dir).expanduser()
        return self.default_path().parent / "journal"
