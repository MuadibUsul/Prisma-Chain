"""Artifact storage for the DA daemon (B4-01 core; B4-02 adds the policies).

Invariants enforced here (they are the reason the chain can trust a provider):

- **Verify before storing.** The provider recomputes ``output_root`` from the
  bytes it received — with the *frozen* libraries, never a re-implementation —
  and refuses anything that does not match (the reference provider's
  ``OUTPUT_ROOT_MISMATCH`` behaviour).
- **Atomic writes.** A blob is written to ``.incoming`` and fsynced before it
  is renamed into place; a crash can never leave a half artifact that looks
  complete.
- **Verify on read.** Every served byte is re-hashed against the index; a
  mismatch quarantines the artifact and refuses to serve it.

Layout (compatible with the Phase E devnet data directories):

```text
<root>/<task_id>/output.bin   the canonical blob
<root>/<task_id>/meta.json    m, n, output_root, task_id32, assignment_id, sha256
```
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Optional

from .frozen import frozen


class StorageError(Exception):
    """Base class for storage failures."""


class RootMismatch(StorageError):
    """The received blob does not hash to the claimed output root."""


class IntegrityError(StorageError):
    """A stored artifact no longer matches its recorded content hash."""


@dataclass
class IndexEntry:
    task_id: int
    bytes: int
    output_root: str
    content_sha256: str
    stored_at_ms: int
    m: int = 0
    n: int = 0
    task_id32: str = ""
    assignment_id: str = ""
    quarantined: bool = False


@dataclass
class ArtifactIndex:
    root: pathlib.Path
    entries: dict[int, IndexEntry] = field(default_factory=dict)
    scanned: int = 0
    quarantined: int = 0

    def __post_init__(self):
        self.root = pathlib.Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- helpers ---------------------------------------------------------

    def task_dir(self, task_id: int) -> pathlib.Path:
        return self.root / str(task_id)

    @staticmethod
    def content_sha256(blob: bytes) -> str:
        return hashlib.sha256(blob).hexdigest()

    def recompute_output_root(self, blob: bytes, *, m: int, n: int, task_id32_hex: str,
                              assignment_id_hex: str) -> bytes:
        """The frozen root over the blob's 32x32 output tiles."""
        values = [int.from_bytes(blob[i:i + 4], "big", signed=True)
                  for i in range(0, len(blob), 4)]
        tiles = frozen.tensors.output_tiles(values, m, n)
        cols_c = (n + 7) // 8
        task_id32 = bytes.fromhex(task_id32_hex)
        assignment_id = bytes.fromhex(assignment_id_hex)
        leaves = [frozen.protocol.leaf_output_tile(
            task_id32, assignment_id, index // cols_c, index % cols_c,
            frozen.tensors.int32s_to_canonical(tile)) for index, tile in enumerate(tiles)]
        return frozen.protocol.merkle_root(leaves)

    def output_root_of(self, blob: bytes, meta: dict) -> bytes:
        return self.recompute_output_root(blob, m=int(meta["m"]), n=int(meta["n"]),
                                          task_id32_hex=meta["task_id32"],
                                          assignment_id_hex=meta["assignment_id"])

    # --- write path ------------------------------------------------------

    def store(self, *, task_id: int, c_hex: str, output_root_hex: str, m: int, n: int,
              task_id32_hex: str, assignment_id_hex: str) -> dict:
        blob = bytes.fromhex(c_hex)
        if len(blob) != m * n * 4:
            raise StorageError(f"blob length {len(blob)} does not match M*N*4 = {m * n * 4}")
        meta = {"m": m, "n": n, "task_id32": task_id32_hex, "assignment_id": assignment_id_hex}
        computed = self.output_root_of(blob, meta)
        if computed.hex() != output_root_hex:
            raise RootMismatch(
                f"OUTPUT_ROOT_MISMATCH: recomputed {computed.hex()[:16]}… != claimed "
                f"{output_root_hex[:16]}…; refusing to store or attest")
        directory = self.task_dir(task_id)
        incoming = self.root / ".incoming"
        incoming.mkdir(exist_ok=True)
        staged = incoming / f"{task_id}-{os.getpid()}-{time.time_ns()}.bin"
        with open(staged, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        directory.mkdir(parents=True, exist_ok=True)
        os.replace(staged, directory / "output.bin")     # atomic publish
        entry_meta = {**meta, "output_root": computed.hex(),
                      "content_sha256": self.content_sha256(blob)}
        meta_path = directory / "meta.json"
        tmp_meta = directory / "meta.json.tmp"
        tmp_meta.write_text(json.dumps(entry_meta, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp_meta, meta_path)
        self.entries[task_id] = IndexEntry(
            task_id=task_id, bytes=len(blob), output_root=computed.hex(),
            content_sha256=entry_meta["content_sha256"], stored_at_ms=int(time.time() * 1000),
            m=m, n=n, task_id32=task_id32_hex, assignment_id=assignment_id_hex)
        return {"stored": True, "bytes": len(blob), "output_root": computed.hex()}

    # --- read path -------------------------------------------------------

    def load_meta(self, task_id: int) -> Optional[dict]:
        path = self.task_dir(task_id) / "meta.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def output(self, task_id: int) -> Optional[bytes]:
        """Serve the blob, re-verifying its content hash; quarantine on mismatch."""
        path = self.task_dir(task_id) / "output.bin"
        meta = self.load_meta(task_id)
        if not path.exists() or meta is None:
            return None
        blob = path.read_bytes()
        actual = self.content_sha256(blob)
        expected = str(meta.get("content_sha256", ""))
        if expected and actual != expected:
            self.quarantine(task_id, reason=f"content sha256 {actual} != recorded {expected}")
            raise IntegrityError(
                f"artifact {task_id} failed its content hash; quarantined, refusing to serve")
        return blob

    def quarantine(self, task_id: int, *, reason: str) -> pathlib.Path:
        directory = self.task_dir(task_id)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.root / f"quarantine-{task_id}-{stamp}"
        os.replace(directory, target)
        (target / "quarantine.reason").write_text(reason + "\n", encoding="utf-8")
        entry = self.entries.get(task_id)
        if entry:
            entry.quarantined = True
        self.quarantined += 1
        return target

    # --- restart recovery -------------------------------------------------

    def rescan(self) -> dict:
        """Rebuild the in-memory index from disk and verify every artifact.

        Called at start-up: the daemon resumes serving exactly what still
        verifies, and reports what it found.
        """
        self.entries = {}
        self.scanned = 0
        self.quarantined = 0
        bad: list[dict] = []
        for directory in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if not directory.name.isdigit():
                continue
            task_id = int(directory.name)
            meta = self.load_meta(task_id)
            if meta is None:
                continue
            blob_path = directory / "output.bin"
            if not blob_path.exists():
                continue
            self.scanned += 1
            blob = blob_path.read_bytes()
            actual = self.content_sha256(blob)
            if str(meta.get("content_sha256", "")) != actual:
                self.quarantine(task_id, reason=f"rescan: content sha256 {actual} != recorded")
                bad.append({"task_id": task_id, "reason": "content hash mismatch"})
                continue
            self.entries[task_id] = IndexEntry(
                task_id=task_id, bytes=len(blob), output_root=str(meta.get("output_root", "")),
                content_sha256=actual, stored_at_ms=int(directory.stat().st_mtime * 1000),
                m=int(meta.get("m", 0)), n=int(meta.get("n", 0)),
                task_id32=str(meta.get("task_id32", "")),
                assignment_id=str(meta.get("assignment_id", "")))
        return {"scanned": self.scanned, "serving": len(self.entries), "bad": bad}

    def metrics(self) -> dict:
        return {"artifacts": len(self.entries),
                "bytes": sum(entry.bytes for entry in self.entries.values()),
                "scanned_at_start": self.scanned,
                "quarantined": self.quarantined}
