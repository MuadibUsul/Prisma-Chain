"""Locate and import the frozen Python libraries the DA protocol is defined by.

The attestation preimage, the tile encoding and the Merkle construction live in
the frozen tree (``compute/canonical/python``, ``compute/gemmv1/python``). The
daemon **imports** them instead of re-implementing: a second implementation of
a frozen format is a drift risk, and the release ships the libraries.

Search order:

1. ``PRISMA_FROZEN_PYTHON`` — ``os.pathsep``-separated directories;
2. repository defaults relative to this file and to the current directory
   (``compute/canonical/python``, ``compute/gemmv1/python``).

A missing library is an explicit error naming what is missing; nothing here
falls back to an approximation.
"""

from __future__ import annotations

import os
import pathlib
import sys
from types import ModuleType
from typing import Optional


class FrozenLibraryMissing(Exception):
    """A frozen library required by the DA protocol could not be imported."""


_CANDIDATES = (
    ("canonical/python", "compute"),
    ("gemmv1/python", "compute"),
)


def _roots() -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    override = os.environ.get("PRISMA_FROZEN_PYTHON", "")
    for entry in override.split(os.pathsep):
        if entry.strip():
            roots.append(pathlib.Path(entry))
    here = pathlib.Path(__file__).resolve()
    for base in (here.parents[2], pathlib.Path.cwd(), pathlib.Path.cwd().parent):
        roots.append(base / "compute")
    return roots


def _import(name: str, *, required_by: str) -> ModuleType:
    try:
        module = sys.modules.get(name) or __import__(name)
        return module
    except ImportError:
        pass
    for root in _roots():
        for sub, prefix in _CANDIDATES:
            candidate = root.parent / prefix / sub if root.name == "compute" else root / sub
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
    try:
        return __import__(name)
    except ImportError as exc:
        raise FrozenLibraryMissing(
            f"{name} (needed by {required_by}) is not importable. The DA protocol's frozen "
            "libraries ship with the release; point PRISMA_FROZEN_PYTHON at the directory "
            "containing compute/canonical/python and compute/gemmv1/python."
        ) from exc


class _Frozen:
    """Lazy handles to the frozen modules (imported on first use)."""

    @property
    def canonical(self) -> ModuleType:
        return _import("canonical_ref", required_by="canonical encoding")

    @property
    def protocol(self) -> ModuleType:
        return _import("gemmv1.protocol", required_by="DA attestations and tiles")

    @property
    def tensors(self) -> ModuleType:
        return _import("gemmv1.tensors", required_by="tile layouts")

    @property
    def merkle(self) -> ModuleType:
        return _import("gemmv1.merkle_proofs", required_by="tile proofs")

    @property
    def cbor(self) -> ModuleType:
        return _import("gemmv1.canonical_cbor", required_by="canonical CBOR")

    def available(self) -> tuple[bool, Optional[str]]:
        try:
            self.canonical, self.protocol, self.tensors, self.merkle, self.cbor
            return True, None
        except FrozenLibraryMissing as exc:
            return False, str(exc)


frozen = _Frozen()
