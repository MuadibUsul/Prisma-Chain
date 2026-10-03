"""Devnet GEMM settlement E2E: honest, fraud and false-challenge scenarios.

Runs against the local Compose chain (phase-d image):

    python deploy/gemm_chain_smoke.py --scenario honest
    python deploy/gemm_chain_smoke.py --scenario fraud
    python deploy/gemm_chain_smoke.py --scenario false-challenge

Every protocol object is produced by the real compute/gemmv1 library; the
chain derives task ids, assignment ids and the canonical MAC count itself.
Transactions are signed by the container's test keyring via prismad.
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "compute" / "gemmv1" / "python"))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402

from gemmv1.protocol import (  # noqa: E402
    ArithmeticSpec, Operator, ProtocolVersion, MATRIX_ID_A, MATRIX_ID_B,
    assignment_id as protocol_assignment_id, build_matrix_roots,
    leaf_input_tile, leaf_output_tile, leaf_trace_state, merkle_root,
    task_id as protocol_task_id,
)
from gemmv1.merkle_proofs import build_levels, prove  # noqa: E402
from gemmv1.tensors import (  # noqa: E402
    INT_TILE_BYTES, canonical_to_int32s, extract_a_tile, extract_b_tile,
    int32s_to_canonical, micro_step, output_tiles, reference_gemm,
)
from gemmv1.testgen import gen_test_matrix  # noqa: E402
from gemmv1.trace import build_tile_trace  # noqa: E402

# Single-node devnet defaults; the multivalidator harness overrides these
# module constants to target a specific validator container and chain.
CHAIN_ID = os.environ.get("PRISMA_SMOKE_CHAIN_ID", "prisma-local-1")
COMPOSE = Path(os.environ.get("PRISMA_SMOKE_COMPOSE", str(REPO / "deploy" / "compose.yaml")))
SERVICE = os.environ.get("PRISMA_SMOKE_SERVICE", "chain")
# Legacy CLI commands (keys, bank) do not read client.toml, so they need
# explicit keyring flags; autocli tx/query commands resolve through the
# client.toml + default-home symlink set up by the container entrypoints.
HOME = os.environ.get("PRISMA_SMOKE_HOME", "/data")
KEYRING_FLAGS = ["--keyring-backend", "test", "--keyring-dir", HOME, "--home", HOME]
M, N, K = 16, 16, 16
SEED = 4242
CHALLENGE_WINDOW = 30
MAX_FEE = 2_000_000
MIN_BOND = 1_000_000
GAS = "2000000"


def cli(*args: str, check: bool = True) -> str:
    cmd = ["docker", "compose", "-f", str(COMPOSE), "exec", "-T", SERVICE, "prismad", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    out = proc.stdout.strip()
    if check and proc.returncode != 0:
        raise RuntimeError(f"prismad {' '.join(args[:3])} failed: {proc.stderr.strip() or out}")
    return out


def wait_inclusion(txhash: str, seconds: int = 60) -> dict:
    deadline = time.time() + seconds
    last = ""
    while time.time() < deadline:
        out = cli("query", "tx", txhash, "-o", "json", check=False)
        if out and "{" in out:
            payload = json.loads(out[out.index("{"):])
            if int(payload.get("code", 0)) != 0:
                raise RuntimeError(f"tx {txhash} failed in block: {payload.get('raw_log')}")
            return payload
        last = out
        time.sleep(1)
    raise RuntimeError(f"tx {txhash} not included: {last}")


def send_tx(msg: str, frm: str, **flags) -> dict:
    args = ["tx", "compute", msg, "--from", frm, "-b", "sync", "-y", "-o", "json",
            "--chain-id", CHAIN_ID, "--fees", "0uprsm", "--gas", GAS]
    for key, value in flags.items():
        if isinstance(value, list):
            for item in value:
                args += [f"--{key.replace('_', '-')}", item]
        else:
            args += [f"--{key.replace('_', '-')}", str(value)]
    out = cli(*args)
    payload = json.loads(out[out.index("{"):])
    if int(payload.get("code", 1)) != 0:
        raise RuntimeError(f"{msg} rejected at mempool: code={payload.get('code')} log={payload.get('raw_log')}")
    return wait_inclusion(payload["txhash"])


def max_task_id() -> int:
    """Highest chain task id visible right now (scan is id-bounded and small
    on a fresh devnet)."""
    highest = 0
    for candidate in range(1, 200):
        try:
            query_task(candidate)
            highest = candidate
        except Exception:
            if candidate > highest + 5:
                break
    return highest


def height() -> int:
    out = cli("status", "--home", "/data", check=False)
    try:
        return int(json.loads(out[out.index("{"):])["sync_info"]["latest_block_height"])
    except Exception:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:26657/status", timeout=10) as resp:
            return int(json.load(resp)["result"]["sync_info"]["latest_block_height"])


def decode_bytes_field(value):
    """Autocli renders bytes fields as base64; the payload is JSON inside."""
    if value is None or value == "":
        return None
    if isinstance(value, (bytes, bytearray)):
        return json.loads(bytes(value))
    return json.loads(base64.b64decode(value))


def query_task(task_id: int) -> dict:
    out = cli("query", "compute", "gemm-task", "--gemm-task-id", str(task_id), "-o", "json")
    payload = json.loads(out[out.index("{"):])
    task = decode_bytes_field(payload.get("task_json"))
    dispute = decode_bytes_field(payload.get("dispute_json"))
    if dispute and not dispute.get("snapshot"):
        dispute["snapshot"] = None
    return {"task": task, "dispute": dispute}


def wait_task(task_id: int, status: str, seconds: int = 60) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        state = query_task(task_id)
        if state["task"]["status"] == status:
            return state
        time.sleep(1)
    raise RuntimeError(f"task {task_id} did not reach {status}: {state['task']['status']}")


def ensure_account(name: str) -> str:
    try:
        return cli("keys", "show", name, "-a", *KEYRING_FLAGS)
    except RuntimeError:
        cli("keys", "add", name, *KEYRING_FLAGS, "--no-backup")
        return cli("keys", "show", name, "-a", *KEYRING_FLAGS)


def key_address(name: str) -> str:
    return cli("keys", "show", name, "-a", *KEYRING_FLAGS)


def fund(address: str, amount: int) -> None:
    send = ["tx", "bank", "send", "validator", address, f"{amount}uprsm",
            "--from", "validator", "-b", "sync", "-y", "-o", "json",
            "--chain-id", CHAIN_ID, "--fees", "0uprsm", "--gas", GAS, *KEYRING_FLAGS]
    out = cli(*send)
    payload = json.loads(out[out.index("{"):])
    if int(payload.get("code", 1)) != 0:
        raise RuntimeError(f"faucet rejected: {payload.get('raw_log')}")
    wait_inclusion(payload["txhash"])


def balance(address: str) -> int:
    out = cli("query", "bank", "balances", address, "-o", "json")
    payload = json.loads(out[out.index("{"):])
    for coin in payload.get("balances", []):
        if coin["denom"] == "uprsm":
            return int(coin["amount"])
    return 0


def canonical(bytes_value: bytes) -> str:
    return bytes_value.hex()


def new_ed25519():
    private = Ed25519PrivateKey.generate()
    raw_priv = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    raw_pub = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    return raw_priv.hex(), raw_pub


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def network_key_proof(account: str, network_seed_hex: str, network_pub: bytes) -> bytes:
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(network_seed_hex))
    payload = canonical_json({"chain_id": CHAIN_ID, "worker": account,
                              "network_public_key": network_pub.hex()})
    return private.sign(b"prisma:network-key-binding:v1\n" + payload)


def requester_key_proof(requester: str, private_hex: str, pub: bytes, nonce: bytes) -> bytes:
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex))
    payload = canonical_json({"chain_id": CHAIN_ID, "requester": requester,
                              "requester_protocol_pubkey": pub.hex(),
                              "requester_nonce": nonce.hex()})
    return private.sign(b"prisma:gemm-requester-key-binding:v1\n" + payload)


def build_descriptor(requester_pub: bytes, nonce: bytes, issued_epoch: int, root_a, root_b):
    return {
        "protocol_version": ProtocolVersion, "operator": Operator,
        "requester_pubkey": requester_pub, "requester_nonce": nonce,
        "issued_epoch": issued_epoch, "m": M, "n": N, "k": K,
        "matrix_a_root": root_a, "matrix_b_root": root_b,
        "arithmetic_spec": ArithmeticSpec, "tile_size": 8,
        "challenge_window": CHALLENGE_WINDOW, "max_price_per_cwu": 1000,
        "settlement_asset": "uprsm",
    }


def input_tile_proof(matrix_id: int, a, b, row: int, col: int, cols: int):
    leaves = []
    if matrix_id == MATRIX_ID_A:
        rows = (M + 7) // 8
        for i in range(rows):
            for r in range(cols):
                tile = extract_a_tile(a, M, K, i, r)
                leaves.append(leaf_input_tile(MATRIX_ID_A, i, r, bytes(v & 0xFF for v in tile)))
    else:
        rows = (K + 7) // 8
        for r in range(rows):
            for j in range(cols):
                tile = extract_b_tile(b, K, N, r, j)
                leaves.append(leaf_input_tile(MATRIX_ID_B, r, j, bytes(v & 0xFF for v in tile)))
    levels = build_levels(leaves)
    index = row * cols + col
    idx, count, siblings = prove(levels, index)
    return idx, count, siblings


def run_honest(tag: str) -> dict:
    report = {"scenario": "honest"}
    # The chain binds the proof to the bech32 address the CLI fills into
    # the signer field, so the payload must use the address, not the
    # keyring name.
    requester = key_address("validator")
    worker = ensure_account(f"gemm-w-{tag}")
    monitor_a = ensure_account(f"gemm-ma-{tag}")
    monitor_b = ensure_account(f"gemm-mb-{tag}")
    for account in (worker, monitor_a, monitor_b):
        fund(account, 10_000_000)
    for monitor in (monitor_a, monitor_b):
        # Monitors only need a plain bond; no network key is required for
        # attestations.
        send_tx("bond-worker", monitor, amount=MIN_BOND)

    worker_seed, worker_pub = new_ed25519()
    proof = network_key_proof(worker, worker_seed, worker_pub)
    send_tx("bond-worker", worker, amount=5 * MIN_BOND,
            network_public_key=canonical(worker_pub), network_key_proof=canonical(proof))

    requester_seed, requester_pub = new_ed25519()
    nonce = uuid.uuid4().bytes
    post_height = height() + 2
    descriptor = build_descriptor(requester_pub, nonce, post_height, bytes(32), bytes(32))
    del descriptor  # the chain derives task_id; local descriptor only documents shape

    a = gen_test_matrix(ord("A"), SEED, M * K)
    b = gen_test_matrix(ord("B"), SEED, K * N)
    root_a, root_b, *_ = build_matrix_roots(a, b, M, N, K)
    key_proof = requester_key_proof(requester, requester_seed, requester_pub, nonce)

    before = max_task_id()
    send_tx("post-gemm-task", "validator",
            requester_protocol_pubkey=canonical(requester_pub),
            requester_key_proof=canonical(key_proof),
            requester_nonce=canonical(nonce),
            m=M, n=N, k=K,
            matrix_a_root=root_a.hex(), matrix_b_root=root_b.hex(),
            challenge_window=CHALLENGE_WINDOW, max_price_per_cwu=1000,
            max_fee=MAX_FEE, input_data_ref="dev://e2e-matrices")
    state = wait_new_task(before)
    task_id = state["task"]["id"]
    protocol_id = state["task"]["protocol_task_id"]
    report["chain_task_id"] = task_id
    report["protocol_task_id"] = protocol_id

    assignment_nonce = uuid.uuid4().bytes
    send_tx("accept-gemm-task", worker, gemm_task_id=task_id,
            assignment_nonce=canonical(assignment_nonce))
    state = wait_task(task_id, "assigned")
    report["assignment_id"] = state["task"]["assignment_id"]

    # Honest worker output commitment.
    c = reference_gemm(a, b, M, N, K)
    tiles = output_tiles(c, M, N)
    cols_c = (N + 7) // 8
    task = state["task"]
    # Go json.Marshal renders []byte fields as base64.
    assignment = base64.b64decode(task["assignment_id"])
    p_task = base64.b64decode(task["protocol_task_id"])
    leaves = [leaf_output_tile(p_task, assignment, i // cols_c, i % cols_c, int32s_to_canonical(t))
              for i, t in enumerate(tiles)]
    output_root = merkle_root(leaves)
    result_epoch = height() + 5
    rc = {
        "protocol_version": ProtocolVersion, "task_id": p_task, "assignment_id": assignment,
        "worker_pubkey": worker_pub, "output_root": output_root,
        "canonical_mac_count": M * N * K, "completed_epoch": result_epoch,
        # The Go verifier zeroes the signature field but still encodes it
        # as an empty bstr, so the signed preimage carries the empty field.
        "worker_signature": b"",
    }
    rc_sig = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(worker_seed))
    from gemmv1.canonical_cbor import encode_canonical  # noqa: E402
    signature = rc_sig.sign(b"PRISMA_GEMM_SIG_V1\x00" + encode_canonical(rc))
    send_tx("submit-gemm-result", worker, gemm_task_id=task_id,
            output_root=output_root.hex(), output_data_ref="dev://e2e-output",
            output_bytes=M * N * 4, completed_epoch=result_epoch,
            worker_signature=signature.hex())
    state = wait_task(task_id, "result_submitted")
    report["challenge_end"] = state["task"]["challenge_end"]

    for monitor in (monitor_a, monitor_b):
        send_tx("attest-gemm-task", monitor, gemm_task_id=task_id)

    # Wait out the challenge window.
    while height() <= state["task"]["challenge_end"]:
        time.sleep(1)
    send_tx("finalize-gemm", "validator", gemm_task_id=task_id)
    final = wait_task(task_id, "finalized")
    report["receipt_id"] = final["task"]["receipt_id"]
    report["status"] = final["task"]["status"]
    report["worker_balance"] = balance(worker)
    report["requester_balance"] = balance(requester)
    return report


def wait_new_task(previous_max: int, seconds: int = 60) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        candidate = previous_max + 1
        try:
            return query_task(candidate)
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("posted task did not appear")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["honest", "fraud", "false-challenge"], required=True)
    args = parser.parse_args()
    tag = uuid.uuid4().hex[:6]
    if args.scenario == "honest":
        report = run_honest(tag)
        print(json.dumps(report, indent=2))
        print("PASS: honest GEMM task settled on chain with exactly one VWR")
        return
    if args.scenario == "fraud":
        report = run_fraud(tag)
        print(json.dumps(report, indent=2))
        print("PASS: corrupted GEMM result adjudicated on chain; ChallengerWins, no VWR")
        return
    if args.scenario == "false-challenge":
        report = run_false_challenge(tag)
        print(json.dumps(report, indent=2))
        print("PASS: false challenge rejected on chain; WorkerWins and the worker settled with challenged_worker_won")
        return
    print(f"scenario {args.scenario} not yet wired")
    sys.exit(2)



# ---------------------------------------------------------------------------
# Fraud scenario: the worker commits a corrupted tile, a challenger proves it
# through the full on-chain dispute.
# ---------------------------------------------------------------------------

def sign_object(payload: dict, private_hex: str) -> bytes:
    from gemmv1.canonical_cbor import encode_canonical

    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex))
    return private.sign(b"PRISMA_GEMM_SIG_V1\x00" + encode_canonical(payload))


def parse_snapshot_low_high(snapshot_b64: str):
    raw = base64.b64decode(snapshot_b64)
    if raw[:4] != b"GMD1":
        raise RuntimeError("unexpected dispute snapshot magic")
    offset = 4 + 1 + 32 + 32 + 4 + 4 + 32 + 32
    low = int.from_bytes(raw[offset:offset + 4], "big")
    high = int.from_bytes(raw[offset + 4:offset + 8], "big")
    return low, high


def run_fraud(tag: str) -> dict:
    report = {"scenario": "fraud"}
    requester = key_address("validator")
    worker = ensure_account(f"gemm-fw-{tag}")
    challenger = ensure_account(f"gemm-fc-{tag}")
    for account in (worker, challenger):
        fund(account, 10_000_000)

    worker_seed, worker_pub = new_ed25519()
    send_tx("bond-worker", worker, amount=5 * MIN_BOND,
            network_public_key=canonical(worker_pub),
            network_key_proof=canonical(network_key_proof(worker, worker_seed, worker_pub)))
    challenger_seed, challenger_pub = new_ed25519()
    send_tx("bond-worker", challenger, amount=5 * MIN_BOND,
            network_public_key=canonical(challenger_pub),
            network_key_proof=canonical(network_key_proof(challenger, challenger_seed, challenger_pub)))

    requester_seed, requester_pub = new_ed25519()
    nonce = uuid.uuid4().bytes
    a = gen_test_matrix(ord("A"), SEED + 1, M * K)
    b = gen_test_matrix(ord("B"), SEED + 1, K * N)
    root_a, root_b, *_ = build_matrix_roots(a, b, M, N, K)
    before = max_task_id()
    send_tx("post-gemm-task", "validator",
            requester_protocol_pubkey=canonical(requester_pub),
            requester_key_proof=canonical(requester_key_proof(requester, requester_seed, requester_pub, nonce)),
            requester_nonce=canonical(nonce),
            m=M, n=N, k=K, matrix_a_root=root_a.hex(), matrix_b_root=root_b.hex(),
            challenge_window=CHALLENGE_WINDOW, max_price_per_cwu=1000,
            max_fee=MAX_FEE, input_data_ref="dev://e2e-matrices")
    state = wait_new_task(before)
    task_id = state["task"]["id"]
    report["chain_task_id"] = task_id

    send_tx("accept-gemm-task", worker, gemm_task_id=task_id,
            assignment_nonce=canonical(uuid.uuid4().bytes))
    state = wait_task(task_id, "assigned")
    task = state["task"]
    p_task = base64.b64decode(task["protocol_task_id"])
    assignment = base64.b64decode(task["assignment_id"])

    # Honest computation, then the worker commits a corrupted tile.
    c = reference_gemm(a, b, M, N, K)
    honest_tiles = output_tiles(c, M, N)
    fraud_tiles = [list(t) for t in honest_tiles]
    fraud_tiles[0][0] += 1
    cols_c = (N + 7) // 8

    def tree_for(tiles):
        leaves = [leaf_output_tile(p_task, assignment, i // cols_c, i % cols_c, int32s_to_canonical(t))
                  for i, t in enumerate(tiles)]
        return build_levels(leaves)

    worker_levels = tree_for(fraud_tiles)
    worker_root = worker_levels[-1][0]
    result_epoch = height() + 5
    rc = {
        "protocol_version": ProtocolVersion, "task_id": p_task, "assignment_id": assignment,
        "worker_pubkey": worker_pub, "output_root": worker_root,
        "canonical_mac_count": M * N * K, "completed_epoch": result_epoch,
        "worker_signature": b"",
    }
    send_tx("submit-gemm-result", worker, gemm_task_id=task_id,
            output_root=worker_root.hex(), output_data_ref="dev://e2e-output",
            output_bytes=M * N * K * 4 // K, completed_epoch=result_epoch,
            worker_signature=sign_object(rc, worker_seed).hex())
    wait_task(task_id, "result_submitted")

    # The challenger disputes tile (0,0) with the exact canonical tile.
    idx, count, siblings = prove(worker_levels, 0)
    opened_epoch = height()
    co = {
        "protocol_version": ProtocolVersion, "task_id": p_task, "assignment_id": assignment,
        "challenger_pubkey": challenger_pub, "worker_pubkey": worker_pub,
        "disputed_tile_i": 0, "disputed_tile_j": 0,
        "worker_output_tile": int32s_to_canonical(fraud_tiles[0]),
        # The signed object carries raw bytes; hex stays on the CLI side.
        "worker_output_proof": {"index": idx, "count": count,
                                "siblings": list(siblings)},
        "challenger_output_tile": int32s_to_canonical(honest_tiles[0]),
        "challenge_bond": MAX_FEE // 100,
        "opened_epoch": opened_epoch,
        "challenger_signature": b"",
    }
    send_tx("open-gemm-challenge", challenger, gemm_task_id=task_id,
            disputed_tile_i=0, disputed_tile_j=0,
            worker_output_tile=co["worker_output_tile"].hex(),
            worker_proof_siblings=[s.hex() for s in siblings],
            worker_proof_index=idx, worker_proof_count=count,
            challenger_output_tile=co["challenger_output_tile"].hex(),
            challenge_bond=co["challenge_bond"], opened_epoch=opened_epoch,
            challenger_signature=sign_object(co, challenger_seed).hex())
    state = wait_task(task_id, "challenged")
    report["challenge_end"] = state["task"]["challenge_end"]

    # Both parties lock traces for the disputed tile only.
    honest_art = build_tile_trace(a, b, M, N, K, p_task, assignment, 0, 0, micro_step)
    states, levels = honest_art[0], honest_art[1]
    r_steps = (K + 7) // 8
    worker_states = list(states)
    worker_states[r_steps] = fraud_tiles[0]
    worker_leaves = [leaf_trace_state(p_task, assignment, 0, 0, step, int32s_to_canonical(s))
                     for step, s in enumerate(worker_states)]
    worker_levels_trace = build_levels(worker_leaves)

    def trace_commit(states, levels, party, keys_pub, locked_epoch):
        p0 = prove(levels, 0)
        pf = prove(levels, r_steps)
        return {
            "protocol_version": ProtocolVersion, "task_id": p_task, "assignment_id": assignment,
            "disputed_tile_i": 0, "disputed_tile_j": 0, "party": party,
            "trace_root": levels[-1][0],
            "initial_state": int32s_to_canonical(states[0]),
            "initial_proof": {"index": p0[0], "count": p0[1], "siblings": list(p0[2])},
            "final_state": int32s_to_canonical(states[r_steps]),
            "final_proof": {"index": pf[0], "count": pf[1], "siblings": list(pf[2])},
            "locked_epoch": locked_epoch,
            "signature": b"",
        }

    locked_epoch = height()
    for actor, party_no, states_i, levels_i, seed_i in (
        (worker, 1, worker_states, worker_levels_trace, worker_seed),
        (challenger, 2, states, levels, challenger_seed),
    ):
        tc = trace_commit(states_i, levels_i, party_no, None, locked_epoch)
        send_tx("commit-gemm-trace", actor, gemm_task_id=task_id,
                trace_root=tc["trace_root"].hex(),
                initial_state=tc["initial_state"].hex(),
                initial_proof_siblings=[s.hex() for s in tc["initial_proof"]["siblings"]],
                initial_proof_index=tc["initial_proof"]["index"],
                initial_proof_count=tc["initial_proof"]["count"],
                final_state=tc["final_state"].hex(),
                final_proof_siblings=[s.hex() for s in tc["final_proof"]["siblings"]],
                final_proof_index=tc["final_proof"]["index"],
                final_proof_count=tc["final_proof"]["count"],
                locked_epoch=locked_epoch,
                signature=sign_object(tc, seed_i).hex())

    # Bisection: the chain derives the midpoint; both parties answer it
    # from their locked traces.
    rounds = 0
    while True:
        state = query_task(task_id)
        dispute = state["dispute"]
        if dispute is None or not dispute.get("snapshot"):
            raise RuntimeError("dispute snapshot missing")
        low, high = parse_snapshot_low_high(dispute["snapshot"])
        if high - low <= 1:
            break
        mid = low + (high - low) // 2
        for actor, states_i, levels_i in (
            (worker, worker_states, worker_levels_trace),
            (challenger, states, levels),
        ):
            i, c_, sibs = prove(levels_i, mid)
            send_tx("submit-gemm-mid-state", actor, gemm_task_id=task_id,
                    state=int32s_to_canonical(states_i[mid]).hex(),
                    proof_siblings=[s.hex() for s in sibs],
                    proof_index=i, proof_count=c_)
        rounds += 1
        if rounds > 32:
            raise RuntimeError("bisection did not converge")
    report["bisection_rounds"] = rounds

    # Arbitration: one 8x8x8 micro-step witness, permissionless caller.
    state = query_task(task_id)
    low, _ = parse_snapshot_low_high(state["dispute"]["snapshot"])
    a_tile = extract_a_tile(a, M, K, 0, low)
    b_tile = extract_b_tile(b, K, N, low, 0)
    cols_a = (K + 7) // 8
    ai, acount, asibs = input_tile_proof(MATRIX_ID_A, a, b, 0, low, cols_a)
    bi, bcount, bsibs = input_tile_proof(MATRIX_ID_B, a, b, low, 0, (N + 7) // 8)
    send_tx("arbitrate-gemm", "validator", gemm_task_id=task_id,
            a_tile=bytes(v & 0xFF for v in a_tile).hex(),
            b_tile=bytes(v & 0xFF for v in b_tile).hex(),
            a_proof_tile_row=0, a_proof_tile_col=low, a_proof_tile_cols=cols_a,
            a_proof_siblings=[s.hex() for s in asibs], a_proof_count=acount,
            b_proof_tile_row=low, b_proof_tile_col=0, b_proof_tile_cols=(N + 7) // 8,
            b_proof_siblings=[s.hex() for s in bsibs], b_proof_count=bcount)
    final = wait_task(task_id, "fraud")
    report["status"] = final["task"]["status"]
    report["arbitration_step"] = low
    report["worker_balance"] = balance(worker)
    report["challenger_balance"] = balance(challenger)
    report["requester_balance"] = balance(requester)
    return report
def run_false_challenge(tag: str) -> dict:
    """Honest worker, malicious challenger: the deterministic dispute must
    end WorkerWins and the worker finalizes with challenged_worker_won."""
    report = {"scenario": "false-challenge"}
    requester = key_address("validator")
    worker = ensure_account(f"gemm-xw-{tag}")
    challenger = ensure_account(f"gemm-xc-{tag}")
    for account in (worker, challenger):
        fund(account, 10_000_000)
    worker_seed, worker_pub = new_ed25519()
    send_tx("bond-worker", worker, amount=5 * MIN_BOND,
            network_public_key=canonical(worker_pub),
            network_key_proof=canonical(network_key_proof(worker, worker_seed, worker_pub)))
    challenger_seed, challenger_pub = new_ed25519()
    send_tx("bond-worker", challenger, amount=5 * MIN_BOND,
            network_public_key=canonical(challenger_pub),
            network_key_proof=canonical(network_key_proof(challenger, challenger_seed, challenger_pub)))

    requester_seed, requester_pub = new_ed25519()
    nonce = uuid.uuid4().bytes
    a = gen_test_matrix(ord("A"), SEED + 2, M * K)
    b = gen_test_matrix(ord("B"), SEED + 2, K * N)
    root_a, root_b, *_ = build_matrix_roots(a, b, M, N, K)
    before = max_task_id()
    send_tx("post-gemm-task", "validator",
            requester_protocol_pubkey=canonical(requester_pub),
            requester_key_proof=canonical(requester_key_proof(requester, requester_seed, requester_pub, nonce)),
            requester_nonce=canonical(nonce),
            m=M, n=N, k=K, matrix_a_root=root_a.hex(), matrix_b_root=root_b.hex(),
            challenge_window=CHALLENGE_WINDOW, max_price_per_cwu=1000,
            max_fee=MAX_FEE, input_data_ref="dev://e2e-matrices")
    state = wait_new_task(before)
    task_id = state["task"]["id"]
    report["chain_task_id"] = task_id
    send_tx("accept-gemm-task", worker, gemm_task_id=task_id,
            assignment_nonce=canonical(uuid.uuid4().bytes))
    state = wait_task(task_id, "assigned")
    task = state["task"]
    p_task = base64.b64decode(task["protocol_task_id"])
    assignment = base64.b64decode(task["assignment_id"])

    c = reference_gemm(a, b, M, N, K)
    honest_tiles = output_tiles(c, M, N)
    cols_c = (N + 7) // 8
    worker_levels = build_levels([
        leaf_output_tile(p_task, assignment, i // cols_c, i % cols_c, int32s_to_canonical(t))
        for i, t in enumerate(honest_tiles)])
    worker_root = worker_levels[-1][0]
    result_epoch = height() + 5
    rc = {
        "protocol_version": ProtocolVersion, "task_id": p_task, "assignment_id": assignment,
        "worker_pubkey": worker_pub, "output_root": worker_root,
        "canonical_mac_count": M * N * K, "completed_epoch": result_epoch,
        "worker_signature": b"",
    }
    send_tx("submit-gemm-result", worker, gemm_task_id=task_id,
            output_root=worker_root.hex(), output_data_ref="dev://e2e-output",
            output_bytes=M * N * 4, completed_epoch=result_epoch,
            worker_signature=sign_object(rc, worker_seed).hex())
    wait_task(task_id, "result_submitted")

    # Malicious challenger claims a wrong tile for the honest worker.
    idx, count, siblings = prove(worker_levels, 0)
    wrong_tile = list(honest_tiles[0])
    wrong_tile[5] += 9
    opened_epoch = height()
    co = {
        "protocol_version": ProtocolVersion, "task_id": p_task, "assignment_id": assignment,
        "challenger_pubkey": challenger_pub, "worker_pubkey": worker_pub,
        "disputed_tile_i": 0, "disputed_tile_j": 0,
        "worker_output_tile": int32s_to_canonical(honest_tiles[0]),
        "worker_output_proof": {"index": idx, "count": count, "siblings": list(siblings)},
        "challenger_output_tile": int32s_to_canonical(wrong_tile),
        "challenge_bond": MAX_FEE // 100,
        "opened_epoch": opened_epoch,
        "challenger_signature": b"",
    }
    send_tx("open-gemm-challenge", challenger, gemm_task_id=task_id,
            disputed_tile_i=0, disputed_tile_j=0,
            worker_output_tile=co["worker_output_tile"].hex(),
            worker_proof_siblings=[s.hex() for s in siblings],
            worker_proof_index=idx, worker_proof_count=count,
            challenger_output_tile=co["challenger_output_tile"].hex(),
            challenge_bond=co["challenge_bond"], opened_epoch=opened_epoch,
            challenger_signature=sign_object(co, challenger_seed).hex())
    wait_task(task_id, "challenged")

    honest_art = build_tile_trace(a, b, M, N, K, p_task, assignment, 0, 0, micro_step)
    states, levels = honest_art[0], honest_art[1]
    r_steps = (K + 7) // 8
    # The lying challenger fabricates a trace ending at its wrong claim.
    fake_states = list(states)
    fake_states[r_steps] = wrong_tile
    fake_leaves = [leaf_trace_state(p_task, assignment, 0, 0, step, int32s_to_canonical(s))
                   for step, s in enumerate(fake_states)]
    fake_levels = build_levels(fake_leaves)

    def trace_commit(states_i, levels_i, party, locked_epoch):
        p0 = prove(levels_i, 0)
        pf = prove(levels_i, r_steps)
        return {
            "protocol_version": ProtocolVersion, "task_id": p_task, "assignment_id": assignment,
            "disputed_tile_i": 0, "disputed_tile_j": 0, "party": party,
            "trace_root": levels_i[-1][0],
            "initial_state": int32s_to_canonical(states_i[0]),
            "initial_proof": {"index": p0[0], "count": p0[1], "siblings": list(p0[2])},
            "final_state": int32s_to_canonical(states_i[r_steps]),
            "final_proof": {"index": pf[0], "count": pf[1], "siblings": list(pf[2])},
            "locked_epoch": locked_epoch,
            "signature": b"",
        }

    locked_epoch = height()
    for actor, party_no, states_i, levels_i, seed_i in (
        (worker, 1, states, levels, worker_seed),
        (challenger, 2, fake_states, fake_levels, challenger_seed),
    ):
        tc = trace_commit(states_i, levels_i, party_no, locked_epoch)
        send_tx("commit-gemm-trace", actor, gemm_task_id=task_id,
                trace_root=tc["trace_root"].hex(),
                initial_state=tc["initial_state"].hex(),
                initial_proof_siblings=[s.hex() for s in tc["initial_proof"]["siblings"]],
                initial_proof_index=tc["initial_proof"]["index"],
                initial_proof_count=tc["initial_proof"]["count"],
                final_state=tc["final_state"].hex(),
                final_proof_siblings=[s.hex() for s in tc["final_proof"]["siblings"]],
                final_proof_index=tc["final_proof"]["index"],
                final_proof_count=tc["final_proof"]["count"],
                locked_epoch=locked_epoch,
                signature=sign_object(tc, seed_i).hex())

    rounds = 0
    while True:
        state = query_task(task_id)
        dispute = state["dispute"]
        low, high = parse_snapshot_low_high(dispute["snapshot"])
        if high - low <= 1:
            break
        mid = low + (high - low) // 2
        for actor, states_i, levels_i in (
            (worker, states, levels),
            (challenger, fake_states, fake_levels),
        ):
            i, c_, sibs = prove(levels_i, mid)
            send_tx("submit-gemm-mid-state", actor, gemm_task_id=task_id,
                    state=int32s_to_canonical(states_i[mid]).hex(),
                    proof_siblings=[s.hex() for s in sibs],
                    proof_index=i, proof_count=c_)
        rounds += 1
        if rounds > 32:
            raise RuntimeError("bisection did not converge")
    report["bisection_rounds"] = rounds

    state = query_task(task_id)
    low, _ = parse_snapshot_low_high(state["dispute"]["snapshot"])
    a_tile = extract_a_tile(a, M, K, 0, low)
    b_tile = extract_b_tile(b, K, N, low, 0)
    cols_a = (K + 7) // 8
    ai, acount, asibs = input_tile_proof(MATRIX_ID_A, a, b, 0, low, cols_a)
    bi, bcount, bsibs = input_tile_proof(MATRIX_ID_B, a, b, low, 0, (N + 7) // 8)
    send_tx("arbitrate-gemm", "validator", gemm_task_id=task_id,
            a_tile=bytes(v & 0xFF for v in a_tile).hex(),
            b_tile=bytes(v & 0xFF for v in b_tile).hex(),
            a_proof_tile_row=0, a_proof_tile_col=low, a_proof_tile_cols=cols_a,
            a_proof_siblings=[s.hex() for s in asibs], a_proof_count=acount,
            b_proof_tile_row=low, b_proof_tile_col=0, b_proof_tile_cols=(N + 7) // 8,
            b_proof_siblings=[s.hex() for s in bsibs], b_proof_count=bcount)
    final = wait_task(task_id, "result_submitted")
    report["status_after_dispute"] = final["task"]["status"]
    report["survived_challenge"] = final["task"].get("survived_challenge")
    digest = final["task"].get("dispute_transcript_digest")
    report["transcript_digest_len"] = len(base64.b64decode(digest)) if digest else 0

    # The honest worker settles and receives challenged_worker_won.
    monitor_a = ensure_account(f"gemm-xma-{tag}")
    monitor_b = ensure_account(f"gemm-xmb-{tag}")
    for monitor in (monitor_a, monitor_b):
        fund(monitor, 5_000_000)
        send_tx("bond-worker", monitor, amount=MIN_BOND)
    for monitor in (monitor_a, monitor_b):
        send_tx("attest-gemm-task", monitor, gemm_task_id=task_id)
    while height() <= final["task"]["challenge_end"]:
        time.sleep(1)
    send_tx("finalize-gemm", "validator", gemm_task_id=task_id)
    settled = wait_task(task_id, "finalized")
    report["status"] = settled["task"]["status"]
    report["worker_balance"] = balance(worker)
    report["challenger_balance"] = balance(challenger)
    return report


if __name__ == "__main__":
    main()
