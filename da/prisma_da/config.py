"""DA daemon configuration (B4-01)."""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class ChainConfig:
    rpc_urls: list[str] = field(default_factory=lambda: ["http://127.0.0.1:26657"])
    chain_id: str = ""
    prismad: str = "prismad"


@dataclass
class DaemonConfig:
    account: str = ""                       # provider account (bech32 prsm1…)
    listen_host: str = "127.0.0.1"
    listen_port: int = 8401
    data_dir: str = ""                      # artifact root (default: <home>/data)
    keystore: str = ""                      # provider key keystore (B2-01 rules)
    chain: ChainConfig = field(default_factory=ChainConfig)
    # B4-02 owns enforcement; recorded here so a config file is complete.
    quota_bytes: int = 0                    # 0 = unlimited (B4-02 enforces the cap)
    ttl_seconds: int = 0                    # 0 = keep until GC policy exists (B4-02)

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
        override = os.environ.get("PRISMA_DA_CONFIG")
        if override:
            return pathlib.Path(override).expanduser()
        return pathlib.Path.home() / ".prisma-da" / "config.json"

    def resolved_data_dir(self) -> pathlib.Path:
        if self.data_dir:
            return pathlib.Path(self.data_dir).expanduser()
        return self.default_path().parent / "data"
