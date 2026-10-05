"""Frozen model / profile artifacts (B2-05).

A worker executes only artifacts that hash to the frozen manifest:

```text
<models dir>/<model_id>/<spec_version>/
    manifest.json     public, versioned: model/spec, digests, per-file sha256+size
    graph.json        the canonical graph document (GraphIDV2 verified on load)
    ...               any further profile artifacts, all listed in the manifest
```

Rules enforced here:

- **verify** recomputes every file hash, the GraphIDV2 from the graph
  document and the PolicyID from the descriptor; any mismatch is a hard
  error and the artifact is quarantined (`quarantine()`), never executed.
- **install** downloads resumably (`Range` on the partial file), verifies
  each byte against the manifest, and stores files in a content-addressed
  cache before publishing them into the profile directory — so several
  versions can live side by side and re-installs are free.
- **chain check** compares the local manifest against the model registered
  on chain (image/tokenizer/weights/program digests); a mismatch refuses.

Manifest schema (v1, frozen here so the worker and any future CLI agree):

```json
{
  "schema": 1,
  "model_id": "qwen3-0.6b-layer0-v2",
  "spec_version": "v1",
  "graph_id_v2": "8fb86087…",
  "policy_id": "eb9a9fef…",
  "digests": {"image": "…", "tokenizer": "…", "weights": "…", "program": "…"},
  "files": [{"path": "graph.json", "sha256": "…", "bytes": 123}]
}
```
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

from .graphid import CanonicalError, graph_id_v2_of, policy_id_of

MANIFEST_NAME = "manifest.json"
SCHEMA = 1

# The frozen identities this release executes (B2-05: only the frozen profile
# is supported; arbitrary models are explicitly out of scope).
FROZEN_GRAPH_ID_V2 = "8fb86087697e277602bb4b2088ba48eb4172c21d9c4f83914582f2cbf00d4def"
FROZEN_POLICY_ID = "eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0"


class ProfileError(Exception):
    """Base class for profile handling failures."""


class ProfileVerificationError(ProfileError):
    """A hash (or identity) mismatch: the artifact must not be executed."""


@dataclass
class ProfileManifest:
    model_id: str
    spec_version: str
    graph_id_v2: str
    policy_id: str
    files: list[dict] = field(default_factory=list)
    digests: dict = field(default_factory=dict)
    schema: int = SCHEMA

    def to_dict(self) -> dict:
        return {"schema": self.schema, "model_id": self.model_id, "spec_version": self.spec_version,
                "graph_id_v2": self.graph_id_v2, "policy_id": self.policy_id,
                "digests": dict(self.digests), "files": list(self.files)}

    @classmethod
    def from_dict(cls, document: dict) -> "ProfileManifest":
        if document.get("schema") != SCHEMA:
            raise ProfileError(f"unsupported manifest schema {document.get('schema')!r}")
        return cls(model_id=str(document["model_id"]), spec_version=str(document["spec_version"]),
                   graph_id_v2=str(document["graph_id_v2"]), policy_id=str(document["policy_id"]),
                   files=list(document.get("files", [])), digests=dict(document.get("digests", {})))


def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def profile_dir(models_dir: pathlib.Path, model_id: str, spec_version: str) -> pathlib.Path:
    return pathlib.Path(models_dir) / model_id / spec_version


def load_manifest(directory: pathlib.Path) -> ProfileManifest:
    path = pathlib.Path(directory) / MANIFEST_NAME
    try:
        return ProfileManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise ProfileError(f"no {MANIFEST_NAME} in {directory}") from exc


def quarantine(path: pathlib.Path, *, reason: str) -> pathlib.Path:
    """Move a bad artifact aside with an explicit reason; never delete evidence."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.quarantined-{stamp}")
    os.replace(path, target)
    (path.parent / f"{target.name}.reason").write_text(reason + "\n", encoding="utf-8")
    return target


def verify(directory: pathlib.Path, *, require_frozen_identity: bool = True,
           expected_model_id: Optional[str] = None,
           quarantine_on_mismatch: bool = True) -> ProfileManifest:
    """Verify a profile directory end to end. Raises on any mismatch."""
    directory = pathlib.Path(directory)
    manifest = load_manifest(directory)
    if expected_model_id and manifest.model_id != expected_model_id:
        raise ProfileVerificationError(
            f"profile is for model {manifest.model_id!r}, expected {expected_model_id!r}")
    for entry in manifest.files:
        path = directory / entry["path"]
        if not path.exists():
            raise ProfileVerificationError(f"profile file missing: {entry['path']}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            reason = (f"checksum mismatch for {entry['path']}: manifest {entry['sha256']}, "
                      f"actual {actual}")
            if quarantine_on_mismatch:
                quarantine(path, reason=reason)
                reason += f" (quarantined as {path.name}.quarantined-*)"
            raise ProfileVerificationError(reason)
        if int(entry.get("bytes", -1)) not in (-1, path.stat().st_size):
            raise ProfileVerificationError(f"size mismatch for {entry['path']}")
    graph_path = directory / "graph.json"
    if graph_path.exists():
        document = json.loads(graph_path.read_text(encoding="utf-8"))
        try:
            computed = graph_id_v2_of(document).hex()
            policy = policy_id_of(document)
        except (CanonicalError, ValueError, KeyError) as exc:
            raise ProfileVerificationError(f"graph document is not canonical: {exc}") from exc
        if computed != manifest.graph_id_v2:
            raise ProfileVerificationError(
                f"GraphIDV2 mismatch: manifest {manifest.graph_id_v2}, recomputed {computed}")
        if policy != manifest.policy_id:
            raise ProfileVerificationError(
                f"PolicyID mismatch: manifest {manifest.policy_id}, descriptor {policy}")
    if require_frozen_identity:
        if manifest.graph_id_v2 not in (FROZEN_GRAPH_ID_V2, "") or manifest.policy_id not in (
                FROZEN_POLICY_ID, ""):
            raise ProfileVerificationError(
                "profile does not carry the frozen identities "
                f"(graph {manifest.graph_id_v2[:12]}…, policy {manifest.policy_id[:12]}…); "
                "only the frozen profile may be executed")
    return manifest


def build_manifest(directory: pathlib.Path, *, model_id: str, spec_version: str,
                   digests: Optional[dict] = None, include: Optional[list[str]] = None,
                   write: bool = True) -> ProfileManifest:
    """Create a manifest for an existing artifact directory (graph.json required)."""
    directory = pathlib.Path(directory)
    graph_path = directory / "graph.json"
    document = json.loads(graph_path.read_text(encoding="utf-8"))
    paths = include or [p.name for p in sorted(directory.iterdir())
                        if p.is_file() and p.name != MANIFEST_NAME]
    files = [{"path": name, "sha256": sha256_file(directory / name),
              "bytes": (directory / name).stat().st_size} for name in paths]
    manifest = ProfileManifest(model_id=model_id, spec_version=spec_version,
                               graph_id_v2=graph_id_v2_of(document).hex(),
                               policy_id=policy_id_of(document), files=files,
                               digests=digests or {})
    if write:
        (directory / MANIFEST_NAME).write_text(json.dumps(manifest.to_dict(), indent=1) + "\n",
                                               encoding="utf-8")
    return manifest


# --- resumable install into a content-addressed cache -----------------------

@dataclass
class InstallReport:
    model_id: str
    spec_version: str
    files: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    resumed: list[str] = field(default_factory=list)


def _fetch(source: str, name: str, target: pathlib.Path, *,
           opener: Optional[Callable[[str], bytes]] = None,
           log: Callable[[str], None] = lambda _m: None) -> bool:
    """Fetch one file into ``target``; returns True when a transfer resumed."""
    if opener is not None:
        target.write_bytes(opener(name))
        return False
    offset = target.stat().st_size if target.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(f"{source.rstrip('/')}/{name}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            resumed = bool(offset) and response.status == 206
            if offset and not resumed:
                log(f"{name}: server ignored Range; restarting the transfer")
            with open(target, "ab" if resumed else "wb") as handle:
                shutil.copyfileobj(response, handle, length=1 << 20)
    except urllib.error.URLError as exc:
        raise ProfileError(f"download failed for {name}: {exc}") from exc
    if resumed:
        log(f"{name}: resumed at byte {offset}")
    return resumed


def install(source: str, models_dir: pathlib.Path, *,
            model_id: Optional[str] = None, spec_version: Optional[str] = None,
            opener: Optional[Callable[[str], bytes]] = None,
            log: Callable[[str], None] = print) -> InstallReport:
    """Install a profile from a directory path or a URL base, resumably.

    The manifest comes first (it is the single source of truth), every file is
    downloaded into the content-addressed cache and verified before it is
    published into the profile directory; a mismatch quarantines the download
    and aborts, so nothing unverified can reach an execution path.
    """
    models_dir = pathlib.Path(models_dir)
    cache = models_dir / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    source_is_dir = pathlib.Path(source).is_dir() if opener is None else False
    source_dir = pathlib.Path(source) if source_is_dir else None

    if source_is_dir:
        manifest_raw = (source_dir / MANIFEST_NAME).read_bytes()
    elif opener is not None:
        manifest_raw = opener(MANIFEST_NAME)
    else:
        try:
            manifest_raw = urllib.request.urlopen(
                f"{source.rstrip('/')}/{MANIFEST_NAME}", timeout=60).read()
        except urllib.error.URLError as exc:
            raise ProfileError(f"cannot fetch {MANIFEST_NAME}: {exc}") from exc
    manifest = ProfileManifest.from_dict(json.loads(manifest_raw))
    if model_id and manifest.model_id != model_id:
        raise ProfileError(f"source holds model {manifest.model_id!r}, expected {model_id!r}")
    if spec_version and manifest.spec_version != spec_version:
        raise ProfileError(f"source holds spec {manifest.spec_version!r}, expected {spec_version!r}")

    target_dir = profile_dir(models_dir, manifest.model_id, manifest.spec_version)
    target_dir.mkdir(parents=True, exist_ok=True)
    report = InstallReport(model_id=manifest.model_id, spec_version=manifest.spec_version)
    for entry in manifest.files:
        name, expected = entry["path"], entry["sha256"]
        blob, partial = cache / expected, cache / f"{name}.part"
        if blob.exists() and sha256_file(blob) == expected:
            log(f"{name}: cache hit")
            report.cached.append(name)
        else:
            if source_is_dir:
                shutil.copyfile(source_dir / name, partial)
            elif _fetch(source, name, partial, opener=opener, log=log):
                report.resumed.append(name)
            actual = sha256_file(partial)
            if actual != expected:
                quarantined = quarantine(partial, reason=(
                    f"checksum mismatch for {name}: manifest {expected}, actual {actual}"))
                raise ProfileVerificationError(
                    f"checksum mismatch for {name} (expected {expected[:12]}…, got {actual[:12]}…); "
                    f"download quarantined as {quarantined.name}")
            os.replace(partial, blob)
        shutil.copyfile(blob, target_dir / name)
        report.files.append(name)
    (target_dir / MANIFEST_NAME).write_bytes(manifest_raw)
    verify(target_dir, quarantine_on_mismatch=False)
    return report


def chain_check(manifest: ProfileManifest, chain_model: dict) -> tuple[bool, str]:
    """Compare a local manifest against the model registered on chain."""
    if not chain_model:
        return False, "the chain has no registered model with this id/version"
    chain_digests = {
        "image": str(chain_model.get("image_digest", "") or ""),
        "tokenizer": str(chain_model.get("tokenizer_digest", "") or ""),
        "weights": str(chain_model.get("weights_digest", "") or ""),
        "program": str(chain_model.get("program_digest", "") or ""),
    }
    problems = []
    for name, local in (manifest.digests or {}).items():
        remote = chain_digests.get(name, "")
        if local and remote and local != remote:
            problems.append(f"{name} digest: local {local[:12]}… vs chain {remote[:12]}…")
    if problems:
        return False, "; ".join(problems)
    return True, "local profile matches the registered model"
