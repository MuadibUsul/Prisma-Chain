"""Retention policy: quota, TTL, safe GC and the integrity scan (B4-02).

The rules, in the order the roadmap states them:

- **quota** — a per-provider size cap with an explicit rejection (never a
  silent overwrite or eviction);
- **TTL** — a retention policy per artifact class (`output` by default);
  a TTL of 0 means *keep forever*, so a misconfigured policy cannot delete
  everything;
- **GC** — an artifact is deletable only when its TTL has expired **and** no
  live challenge references it; every deletion is reported, nothing is
  removed silently;
- **integrity scan** — a full re-hash of every artifact; a mismatch is
  quarantined (moved aside with a reason) and reported, and the artifact
  stops being served.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .storage import ArtifactIndex, IntegrityError, StorageError


class QuotaExceeded(StorageError):
    """The provider's size cap would be exceeded (explicit rejection)."""


@dataclass
class RetentionPolicy:
    quota_bytes: int = 0                      # 0 = unlimited
    ttl_seconds: int = 0                      # 0 = keep forever
    class_ttl_seconds: dict = field(default_factory=dict)   # e.g. {"challenge_relevant": 86400}


def artifact_class(meta: dict) -> str:
    return str(meta.get("artifact_class") or "output")


def ttl_for(policy: RetentionPolicy, meta: dict) -> int:
    return int(policy.class_ttl_seconds.get(artifact_class(meta), policy.ttl_seconds))


def current_bytes(index: ArtifactIndex, *, ignore_task: Optional[int] = None) -> int:
    return sum(entry.bytes for task_id, entry in index.entries.items() if task_id != ignore_task)


def enforce_quota(index: ArtifactIndex, new_bytes: int, quota_bytes: int, *,
                  ignore_task: Optional[int] = None) -> None:
    """Raise QuotaExceeded when storing ``new_bytes`` would breach the cap."""
    if quota_bytes <= 0:
        return
    used = current_bytes(index, ignore_task=ignore_task)
    if used + new_bytes > quota_bytes:
        raise QuotaExceeded(
            f"quota exceeded: {used} bytes stored + {new_bytes} new > cap {quota_bytes}; "
            "refusing the upload (no eviction happens implicitly — run `prisma-da gc`)")


def gc(index: ArtifactIndex, policy: RetentionPolicy, *,
       live_challenges: Iterable[int] = (), now_ms: Optional[int] = None,
       dry_run: bool = False, log=lambda _m: None) -> dict:
    """Delete artifacts whose TTL expired and which no live challenge needs.

    Returns a report; nothing is deleted when the TTL is 0 (keep forever) or
    when the task id appears in ``live_challenges``.
    """
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    live = {int(task_id) for task_id in live_challenges}
    deleted: list[dict] = []
    kept: list[dict] = []
    for task_id in sorted(index.entries):
        entry = index.entries[task_id]
        meta = index.load_meta(task_id) or {}
        ttl = ttl_for(policy, meta)
        age_ms = now_ms - entry.stored_at_ms
        if task_id in live:
            kept.append({"task_id": task_id, "reason": "live challenge"})
            continue
        if ttl <= 0:
            kept.append({"task_id": task_id, "reason": "ttl 0 (keep forever)"})
            continue
        if age_ms < ttl * 1000:
            kept.append({"task_id": task_id,
                         "reason": f"age {age_ms // 1000}s < ttl {ttl}s"})
            continue
        if dry_run:
            deleted.append({"task_id": task_id, "dry_run": True, "bytes": entry.bytes})
            continue
        directory = index.task_dir(task_id)
        freed = entry.bytes
        for child in sorted(directory.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            else:
                child.rmdir()
        directory.rmdir()
        index.entries.pop(task_id, None)
        deleted.append({"task_id": task_id, "bytes": freed})
        log(f"gc: deleted task {task_id} ({freed} bytes, ttl {ttl}s expired)")
    return {"deleted": deleted, "kept": kept, "dry_run": dry_run,
            "freed_bytes": sum(item.get("bytes", 0) for item in deleted)}


def integrity_scan(index: ArtifactIndex, *, log=lambda _m: None) -> dict:
    """Full re-hash of every indexed artifact; quarantine mismatches."""
    checked = 0
    bad: list[dict] = []
    for task_id in sorted(index.entries):
        checked += 1
        try:
            index.output(task_id)          # re-verifies the content hash
        except IntegrityError as exc:
            bad.append({"task_id": task_id, "reason": str(exc)})
            log(f"integrity scan: task {task_id} failed: {exc}")
            index.entries.pop(task_id, None)
    return {"checked": checked, "bad": bad, "healthy": checked - len(bad)}
