"""On-demand tile trace and GEMM dispute logic mirroring compute/gemmv1
trace.go and dispute.go for the two-node E2E. Only one disputed tile ever
gets a trace."""

from .protocol import leaf_trace_state
from .merkle_proofs import build_levels, prove, verify_inclusion
from .tensors import extract_a_tile, extract_b_tile, micro_step, INT_TILE_BYTES


def build_tile_trace(a, b, m, n, k, task: bytes, assignment: bytes, tile_i: int, tile_j: int, tile_step):
    """Returns (states S0..SR, levels, root). tile_step is the micro-step
    function; use tensors.micro_step for the CPU path."""
    r_steps = (k + 7) // 8
    states = [ [0] * 64 ]
    for r in range(r_steps):
        a_tile = extract_a_tile(a, m, k, tile_i, r)
        b_tile = extract_b_tile(b, k, n, r, tile_j)
        states.append(tile_step(states[-1], a_tile, b_tile))
    leaves = [
        leaf_trace_state(task, assignment, tile_i, tile_j, step, int32s_to_bytes(s))
        for step, s in enumerate(states)
    ]
    levels = build_levels(leaves)
    return states, levels, levels[-1][0]


def int32s_to_bytes(state) -> bytes:
    from .tensors import int32s_to_canonical
    return int32s_to_canonical(state)


def trace_commit(task, assignment, tile_i, tile_j, states, levels, locked_epoch):
    """Wire form of a locked trace with endpoint proofs (no signature here;
    signatures belong to the transport/chain layer in the E2E)."""
    r_steps = len(states) - 1
    idx0, count, sib0 = prove(levels, 0)
    idxf, _, sibf = prove(levels, r_steps)
    root = levels[-1][0]
    return {
        "protocol_version": "0.1.1",
        "task_id": task.hex(),
        "assignment_id": assignment.hex(),
        "disputed_tile_i": tile_i,
        "disputed_tile_j": tile_j,
        "trace_root": root.hex(),
        "initial_state": int32s_to_bytes(states[0]).hex(),
        "initial_proof": {"index": idx0, "count": count, "siblings": [s.hex() for s in sib0]},
        "final_state": int32s_to_bytes(states[r_steps]).hex(),
        "final_proof": {"index": idxf, "count": count, "siblings": [s.hex() for s in sibf]},
        "locked_epoch": locked_epoch,
    }


def coordinator_verify_trace(task: bytes, assignment: bytes, tile_i, tile_j, r_steps, commit: dict) -> bytes:
    """Coordinator-side lock verification; returns the trace root."""
    if commit["disputed_tile_i"] != tile_i or commit["disputed_tile_j"] != tile_j:
        raise ValueError("trace commit is not bound to the disputed tile")
    root = bytes.fromhex(commit["trace_root"])
    initial = bytes.fromhex(commit["initial_state"])
    final = bytes.fromhex(commit["final_state"])
    if len(initial) != INT_TILE_BYTES or len(final) != INT_TILE_BYTES:
        raise ValueError("trace states must be 256 canonical bytes")
    if any(initial) :
        raise ValueError("S0 of a locked trace must be the zero matrix")
    leaf0 = leaf_trace_state(task, assignment, tile_i, tile_j, 0, initial)
    leaff = leaf_trace_state(task, assignment, tile_i, tile_j, r_steps, final)
    p0 = commit["initial_proof"]
    pf = commit["final_proof"]
    if not verify_inclusion(root, leaf0, p0["index"], p0["count"], [bytes.fromhex(s) for s in p0["siblings"]]):
        raise ValueError("invalid S0 inclusion proof")
    if not verify_inclusion(root, leaff, pf["index"], pf["count"], [bytes.fromhex(s) for s in pf["siblings"]]):
        raise ValueError("invalid S_R inclusion proof")
    return root


def submit_mid_state(states, levels, step: int):
    """Participant-side midpoint: (state_bytes, proof dict)."""
    state_bytes = int32s_to_bytes(states[step])
    idx, count, siblings = prove(levels, step)
    return state_bytes, {"index": idx, "count": count, "siblings": [s.hex() for s in siblings]}


def micro_step_state(low_state, a_tile, b_tile):
    return micro_step(low_state, a_tile, b_tile)
