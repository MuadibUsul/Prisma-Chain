"""Generate the F.5C V2 state/trail vector files (roadmap A1-02/A1-03).

Python produces the fixtures (deterministic); the Go tests recompute them.

    python tools/f5c_gen_v2_vectors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import canonical_ref as R  # noqa: E402


def _root(tag: bytes) -> bytes:
    return R.hash_bytes(b"f5c-v2-vector-" + tag)


def main() -> None:
    # --- state vectors -------------------------------------------------------
    state_cases = []
    cases = [
        ("single_input", {(0, 0): _root(b"x0")}),
        ("input_plus_nodes", {(0, 0): _root(b"x0"), (1, 0): _root(b"n0"),
                              (1, 1): _root(b"n1"), (1, 2): _root(b"n2")}),
        ("sparse_ids_sorted", {(1, 9): _root(b"n9"), (0, 2): _root(b"x2"),
                               (1, 0): _root(b"n0"), (0, 0): _root(b"x0")}),
    ]
    for name, live in cases:
        root_v2 = R.graph_state_root_v2(live)
        # V1 state root of the same logical tensors (V1 leaf/domain) for the
        # cross-version separation record
        v1_leaves = []
        for kind, index in sorted(live, key=lambda t: (t[0] << 32) | t[1]):
            v1_leaves.append(R.hash_bytes(R.DOMAIN_GRAPH_STATE, R._u32be(kind),
                                          R._u32be(index), live[(kind, index)]))
        level = v1_leaves
        while len(level) > 1:
            level = [R.hash_bytes(level[i] + (level[i + 1] if i + 1 < len(level) else level[i]))
                     for i in range(0, len(level), 2)]
        state_cases.append({
            "name": name,
            "live": [{"kind": k, "index": i, "root": live[(k, i)].hex()}
                     for (k, i) in sorted(live, key=lambda t: (t[0] << 32) | t[1])],
            "state_root_v2": root_v2.hex(),
            "state_root_v1_same_inputs": level[0].hex(),
        })
    (REPO / "testdata" / "canonical_graph_v2_state_vectors.json").write_text(
        json.dumps({"version": "f5c-state-v2-v1", "cases": state_cases}, indent=1) + "\n",
        encoding="utf-8")

    # --- trail vectors -------------------------------------------------------
    graph_id = _root(b"graph")
    trail = [R.hash_bytes(b"state-%d" % i) for i in range(5)]
    root = R.trail_root_v2(graph_id, trail)
    proofs = []
    for i in range(len(trail)):
        proofs.append({"step": i, "state_root": trail[i].hex(),
                       "siblings": [s.hex() for s in R.trail_proof_v2(graph_id, trail, i)]})
    (REPO / "testdata" / "canonical_graph_v2_trail_vectors.json").write_text(
        json.dumps({
            "version": "f5c-trail-v2-v1",
            "graph_id": graph_id.hex(),
            "trail": [s.hex() for s in trail],
            "trail_root_v2": root.hex(),
            "proofs": proofs,
        }, indent=1) + "\n", encoding="utf-8")
    print("state vectors:", len(state_cases), "| trail steps:", len(trail),
          "| trail root:", root.hex()[:16])


if __name__ == "__main__":
    main()
