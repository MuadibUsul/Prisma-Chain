"""Output data availability for GEMM v0.1.2, mirroring
compute/gemmv1/verify/availability.go (Go). The v0.1.1 ResultCommit hash is
untouched; availability is a separate versioned descriptor."""

import hashlib

from .protocol import merkle_root, task_id, leaf_output_tile
from .tensors import int32s_to_canonical, output_tiles


class AvailabilityError(ValueError):
    pass


# Sentinel failure classes. A data problem is not fraud evidence; it only
# blocks optimistic finalization.
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
OUTPUT_DATA_COMMITMENT_MISMATCH = "OUTPUT_DATA_COMMITMENT_MISMATCH"


def new_output_availability_descriptor(descriptor: dict, assignment: bytes, output_root: bytes, output_data_ref: str) -> dict:
    tid = task_id(descriptor)
    if len(assignment) != 32 or len(output_root) != 32:
        raise AvailabilityError("assignment_id/output_root must be 32 bytes")
    m, n = descriptor["m"], descriptor["n"]
    return {
        "protocol_version": "0.1.2",
        "task_id": tid,
        "assignment_id": assignment,
        "output_root": output_root,
        "output_bytes": m * n * 4,
        "output_data_ref": output_data_ref,
    }


def resolve_output(descriptor: dict, assignment: bytes, desc: dict, fetch):
    """Fetch C and verify it hashes to the committed output_root.

    fetch(desc) -> list[int32]; raise AvailabilityError(DATA_UNAVAILABLE) or
    return any error to mark the data unavailable for this window.
    """
    try:
        c = fetch(desc)
    except Exception:
        raise AvailabilityError(DATA_UNAVAILABLE)
    if c is None:
        raise AvailabilityError(DATA_UNAVAILABLE)
    m, n = descriptor["m"], descriptor["n"]
    if len(c) != m * n:
        raise AvailabilityError(OUTPUT_DATA_COMMITMENT_MISMATCH)
    cols_c = (n + 7) // 8
    tiles = output_tiles(c, m, n)
    leaves = [
        leaf_output_tile(desc["task_id"], desc["assignment_id"], idx // cols_c, idx % cols_c, int32s_to_canonical(t))
        for idx, t in enumerate(tiles)
    ]
    if merkle_root(leaves) != desc["output_root"]:
        raise AvailabilityError(OUTPUT_DATA_COMMITMENT_MISMATCH)
    return c


def can_finalize_optimistic(descriptor: dict, assignment: bytes, desc: dict, fetch) -> bool:
    """Data-unavailable or mismatched outputs must not finalize normally."""
    resolve_output(descriptor, assignment, desc, fetch)
    return True
