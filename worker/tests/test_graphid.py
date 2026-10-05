"""B2-05: the GraphIDV2 mirror must equal the authoritative implementation.

The worker verifies a profile by itself, so it carries its own canonical
encoder; these tests pin it against the frozen tree
(``compute/canonical/python/canonical_ref.py`` and ``tools/f5c_bundle_v2.py``)
and against the frozen GraphIDV2 of ``testdata/f5c_qwen3_block_v2.json``.
Any drift fails CI.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from prisma_worker import graphid

REPO = pathlib.Path(__file__).resolve().parents[2]
FROZEN_GRAPH = REPO / "testdata" / "f5c_qwen3_block_v2.json"
FROZEN_GRAPH_ID_V2 = "8fb86087697e277602bb4b2088ba48eb4172c21d9c4f83914582f2cbf00d4def"
FROZEN_POLICY_ID = "eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0"


def _authoritative():
    """Import the frozen canonical reference and the bundle tool."""
    sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))
    sys.path.insert(0, str(REPO / "tools"))
    import canonical_ref  # type: ignore
    import f5c_bundle_v2  # type: ignore

    return canonical_ref, f5c_bundle_v2


def test_encoder_matches_the_frozen_reference_on_shapes():
    canonical_ref, _ = _authoritative()
    shapes = [
        0, 1, 23, 24, 255, 256, 65535, 65536, 2**32, 2**63 - 1, -1, -(2**63),
        b"", b"\x00\x01\xff", "", "prisma", "üñîçødé",
        [], [1, 2, 3], [b"a", "b"], {},
        {"b": 1, "a": 2}, {"nested": {"x": [1, {"y": b"z"}]}},
    ]
    for shape in shapes:
        assert graphid.encode_canonical(shape) == canonical_ref.encode_canonical(shape), shape


def test_encoder_refuses_bool_and_none_like_the_reference():
    canonical_ref, _ = _authoritative()
    for bad in (True, False, None):
        with pytest.raises(graphid.CanonicalError):
            graphid.encode_canonical(bad)
        with pytest.raises(Exception):
            canonical_ref.encode_canonical(bad)


def test_graph_id_matches_the_authoritative_tool_and_the_frozen_constant():
    canonical_ref, bundle = _authoritative()
    document = json.loads(FROZEN_GRAPH.read_text(encoding="utf-8"))
    ours = graphid.graph_id_v2_of(document).hex()
    theirs = bundle.graph_id_v2_of(document).hex()
    assert ours == theirs
    assert ours == FROZEN_GRAPH_ID_V2
    assert graphid.policy_id_of(document) == FROZEN_POLICY_ID


def test_hash_bytes_matches_the_reference():
    canonical_ref, _ = _authoritative()
    assert graphid.hash_bytes(b"a", b"bc") == canonical_ref.hash_bytes(b"a", b"bc")


def test_domain_string_is_the_frozen_v2_domain():
    canonical_ref, _ = _authoritative()
    assert graphid.DOMAIN_GRAPH_V2 == canonical_ref.DOMAIN_GRAPH_V2
    assert graphid.DOMAIN_GRAPH_V2 == b"PRISMA_CANONICAL_GRAPH_V2\x00"
