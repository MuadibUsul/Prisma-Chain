"""Locate the frozen libraries the watcher verifies with (B3-01).

The watcher must be able to verify a bundle *by itself*: the canonical
encoder, the gemmv1 tensors/protocol helpers and the frozen WatcherV2
implementation all live in the repository tree and ship with the release.
This module finds them (``PRISMA_FROZEN_PYTHON`` first, then repository
defaults) and refuses explicitly when something is missing — no
re-implementation, no approximation.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import sys
from types import ModuleType

_EXTRA_DIRS = ("compute/canonical/python", "compute/gemmv1/python", "tools")


class FrozenLibraryMissing(Exception):
    """A frozen library required by the watcher could not be imported."""


def _candidate_dirs() -> list[pathlib.Path]:
    dirs: list[pathlib.Path] = []
    for entry in os.environ.get("PRISMA_FROZEN_PYTHON", "").split(os.pathsep):
        if entry.strip():
            root = pathlib.Path(entry)
            dirs.extend(root / extra for extra in _EXTRA_DIRS)
            dirs.append(root)
    here = pathlib.Path(__file__).resolve()
    for base in (here.parents[3], pathlib.Path.cwd(), pathlib.Path.cwd().parent):
        dirs.extend(base / extra for extra in _EXTRA_DIRS)
    return dirs


def frozen_import(name: str) -> ModuleType:
    """Import a frozen module, searching the candidate directories."""
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    for directory in _candidate_dirs():
        if directory.is_dir() and str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise FrozenLibraryMissing(
            f"{name} is not importable. The watcher verifies with the frozen libraries and "
            "the frozen WatcherV2 module; point PRISMA_FROZEN_PYTHON at the repository root "
            "(or at the directory containing compute/ and tools/) and retry."
        ) from exc


class FrozenWatcher:
    """Handles the watcher-relevant frozen modules (imported lazily)."""

    @property
    def canonical(self) -> ModuleType:
        return frozen_import("canonical_ref")

    @property
    def protocol(self) -> ModuleType:
        return frozen_import("gemmv1.protocol")

    @property
    def watcher(self) -> ModuleType:
        return frozen_import("f5c_watcher_v2")

    def watcher_class(self):
        return self.watcher.WatcherV2

    def available(self) -> tuple[bool, str]:
        try:
            self.canonical, self.protocol, self.watcher
            return True, ""
        except FrozenLibraryMissing as exc:
            return False, str(exc)


frozen = FrozenWatcher()
