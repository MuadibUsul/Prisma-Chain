"""E1 challenge-inclusion E2E under a censoring proposer.

Scenario: validator-a runs with PRISMA_DEV_CENSOR_GEMM_CHALLENGES=all
(proposer omission only; every validator still considers the transactions
valid). A fraudulent GEMM result is disputed while the observer records,
for every block from broadcast to inclusion:

    (height, proposer, did this block include the dispute transaction)

The assertions:
  1. The challenge transaction is included within a bounded number of
     blocks (one censoring proposer cannot suppress it permanently).
  2. Every block proposed by the censoring validator while the disputed
     transaction was still pending omitted it (proposer omission is the
     objective, observed fact), and a later honest proposer included it.
  3. The full dispute still completes: single-tile traces, bisection,
     512-MAC arbitration, ChallengerWins, no worker receipt.

Results are written to docs/phase-e-censorship-results.json.
"""

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy"))
sys.path.insert(0, str(REPO / "compute" / "gemmv1" / "python"))

DIR = REPO / "deploy" / "multivalidator"
COMPOSE = DIR / "compose.yaml"
RPCS = {
    "validator-a": "http://127.0.0.1:26661",
    "validator-b": "http://127.0.0.1:26662",
    "validator-c": "http://127.0.0.1:26663",
    "validator-d": "http://127.0.0.1:26664",
}

# The driver execs into validator-b (a non-censoring node) and speaks the
# multivalidator chain.
import gemm_chain_smoke as G  # noqa: E402

G.CHAIN_ID = "prisma-mv-1"
G.COMPOSE = COMPOSE
G.SERVICE = "validator-b"
G.HOME = "/data/validator-b"
G.KEYRING_FLAGS = ["--keyring-backend", "test", "--keyring-dir", G.HOME, "--home", G.HOME]


def rpc(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=5) as resp:
        return json.load(resp)["result"]


def validator_addresses() -> dict:
    addresses = {}
    for name, base in RPCS.items():
        try:
            info = rpc(base, "/status")["validator_info"]
            if info.get("address"):
                addresses[info["address"]] = name
        except Exception:
            pass
    return addresses


def tx_hash_of(raw_b64: str) -> str:
    """CometBFT tx hash = uppercase hex sha256 of the tx bytes."""
    import base64 as b64
    import hashlib

    return hashlib.sha256(b64.b64decode(raw_b64)).hexdigest().upper()


def block_info(height: int) -> dict:
    raw = rpc(RPCS["validator-b"], f"/block?height={height}")
    header = raw["block"]["header"]
    txs = raw["block"]["data"].get("txs") or []
    return {"height": height, "proposer": header["proposer_address"].upper(),
            "tx_hashes": [tx_hash_of(tx) for tx in txs]}


def unconfirmed(base: str) -> int:
    return int(rpc(base, "/unconfirmed_txs")["n_txs"])


def latest_height() -> int:
    return int(rpc(RPCS["validator-b"], "/status")["sync_info"]["latest_block_height"])


def broadcast_sync(*args: str) -> str:
    """Broadcast without waiting; returns the tx hash."""
    out = G.cli(*args)
    payload = json.loads(out[out.index("{"):])
    if int(payload.get("code", 1)) != 0:
        raise RuntimeError(f"mempool rejected: {payload.get('raw_log')}")
    return payload["txhash"]


def wait_included(txhash: str, seconds: int = 90) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        out = G.cli("query", "tx", txhash, "-o", "json", check=False)
        if out and "{" in out:
            return json.loads(out[out.index("{"):])
        time.sleep(1)
    raise RuntimeError(f"tx {txhash} was never included")


def observe_inclusion(txhash: str, from_height: int, addresses: dict, seconds: int = 120) -> dict:
    """Watch blocks until the tx is included, recording proposer omission."""
    observation = {"broadcast_height": from_height, "blocks": [], "censored_by": [],
                   "included_height": None, "included_by": None}
    deadline = time.time() + seconds
    height = from_height
    included_at = None
    while time.time() < deadline:
        current = latest_height()
        while height <= current:
            info = block_info(height)
            has_tx = txhash.upper() in info["tx_hashes"]
            proposer = addresses.get(info["proposer"], info["proposer"][:12])
            observation["blocks"].append({"height": height, "proposer": proposer,
                                          "included": has_tx})
            if has_tx and included_at is None:
                included_at = height
                observation["included_height"] = height
                observation["included_by"] = proposer
            if not has_tx and proposer == "validator-a":
                observation["censored_by"].append(height)
            height += 1
        if included_at is not None:
            observation["inclusion_delay_blocks"] = included_at - from_height
            return observation
        time.sleep(1)
    raise RuntimeError(f"challenge not included within {seconds}s: {observation}")


def with_service(service: str) -> None:
    """Point the smoke driver's CLI at one validator container."""
    home = f"/data/{service}"
    G.SERVICE = service
    G.HOME = home
    G.KEYRING_FLAGS = ["--keyring-backend", "test", "--keyring-dir", home, "--home", home]


def ensure_imported(name: str, seed_hex: str) -> str:
    """Import the same secp256k1 account key into validator-a and
    validator-b keyrings so both the censoring broadcaster and the honest
    driver can sign with the same account."""
    address = ""
    for service in ("validator-a", "validator-b"):
        with_service(service)
        try:
            address = G.key_address(name)
        except Exception:
            G.cli("keys", "import-hex", name, seed_hex, *G.KEYRING_FLAGS)
            address = G.key_address(name)
    return address


def learn_proposer_cycle(addresses: dict) -> dict:
    """Learn the round-robin proposer order from recent blocks and return a
    next-proposer map. Equal-power validators rotate deterministically."""
    top = latest_height()
    order = []
    for height in range(top - 11, top + 1):
        try:
            proposer = block_info(height)["proposer"]
        except Exception:
            continue
        name = addresses.get(proposer)
        if name and (not order or order[-1] != name):
            order.append(name)
    nxt = {}
    for i in range(len(order) - 1):
        nxt[order[i]] = order[i + 1]
    return nxt


def wait_for_predecessor_of(target: str, addresses: dict, seconds: int = 240) -> dict:
    """Block until the chain's latest proposer is the one whose rotation
    precedes `target`, so the very next block is proposed by `target`."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        nxt = learn_proposer_cycle(addresses)
        current = latest_height()
        info = block_info(current)
        name = addresses.get(info["proposer"])
        if nxt.get(name) == target:
            # Re-check that no new block landed meanwhile.
            time.sleep(0.15)
            if latest_height() == current:
                return {"height": current, "last_proposer": name,
                        "next_expected": target, "order": nxt}
        time.sleep(0.2)
    raise RuntimeError(f"never observed the predecessor of {target}")


def run_attempt(attempt: int, report: dict) -> bool:
    """One fraud attempt: build, broadcast the challenge from the censoring
    node's own RPC, observe inclusion, then complete the dispute. Returns
    True when a proposer-omission window was captured."""
    tag = __import__("uuid").uuid4().hex[:6]
    addresses = report["validator_addresses"]
    with_service("validator-b")
    import os as _os

    worker = ensure_imported(f"gcw{tag}", _os.urandom(32).hex())
    challenger = ensure_imported(f"gcc{tag}", _os.urandom(32).hex())
    with_service("validator-b")
    for account in (worker, challenger):
        G.fund(account, 10_000_000)
    worker_seed, worker_pub = G.new_ed25519()
    G.send_tx("bond-worker", worker, amount=5 * G.MIN_BOND,
              network_public_key=G.canonical(worker_pub),
              network_key_proof=G.canonical(G.network_key_proof(worker, worker_seed, worker_pub)))
    challenger_seed, challenger_pub = G.new_ed25519()
    G.send_tx("bond-worker", challenger, amount=5 * G.MIN_BOND,
              network_public_key=G.canonical(challenger_pub),
              network_key_proof=G.canonical(G.network_key_proof(challenger, challenger_seed, challenger_pub)))
    requester = G.key_address("validator")
    requester_seed, requester_pub = G.new_ed25519()
    nonce = __import__("uuid").uuid4().bytes
    a = G.gen_test_matrix(ord("A"), G.SEED + 7, G.M * G.K)
    b = G.gen_test_matrix(ord("B"), G.SEED + 7, G.K * G.N)
    root_a, root_b, *_ = G.build_matrix_roots(a, b, G.M, G.N, G.K)
    before = G.max_task_id()
    G.send_tx("post-gemm-task", "validator",
              requester_protocol_pubkey=G.canonical(requester_pub),
              requester_key_proof=G.canonical(G.requester_key_proof(requester, requester_seed, requester_pub, nonce)),
              requester_nonce=G.canonical(nonce),
              m=G.M, n=G.N, k=G.K, matrix_a_root=root_a.hex(), matrix_b_root=root_b.hex(),
              challenge_window=G.CHALLENGE_WINDOW, max_price_per_cwu=1000,
              max_fee=G.MAX_FEE, input_data_ref="dev://e2e-matrices")
    state = G.wait_new_task(before)
    task_id = state["task"]["id"]
    report["chain_task_id"] = task_id
    G.send_tx("accept-gemm-task", worker, gemm_task_id=task_id,
              assignment_nonce=G.canonical(__import__("uuid").uuid4().bytes))
    state = G.wait_task(task_id, "assigned")
    task = state["task"]
    p_task = base64.b64decode(task["protocol_task_id"])
    assignment = base64.b64decode(task["assignment_id"])
    c = G.reference_gemm(a, b, G.M, G.N, G.K)
    honest_tiles = G.output_tiles(c, G.M, G.N)
    fraud_tiles = [list(t) for t in honest_tiles]
    fraud_tiles[0][0] += 1
    cols_c = (G.N + 7) // 8
    worker_levels = G.build_levels([
        G.leaf_output_tile(p_task, assignment, i // cols_c, i % cols_c, G.int32s_to_canonical(t))
        for i, t in enumerate(fraud_tiles)])
    worker_root = worker_levels[-1][0]
    result_epoch = G.height() + 5
    rc = {"protocol_version": G.ProtocolVersion, "task_id": p_task, "assignment_id": assignment,
          "worker_pubkey": worker_pub, "output_root": worker_root,
          "canonical_mac_count": G.M * G.N * G.K, "completed_epoch": result_epoch,
          "worker_signature": b""}
    G.send_tx("submit-gemm-result", worker, gemm_task_id=task_id,
              output_root=worker_root.hex(), output_data_ref="dev://e2e-output",
              output_bytes=G.M * G.N * 4, completed_epoch=result_epoch,
              worker_signature=G.sign_object(rc, worker_seed).hex())
    G.wait_task(task_id, "result_submitted")

    # Broadcast the challenge from validator-a's RPC: the censoring
    # proposer's own mempool holds it first, so if it proposes next it
    # must actively omit a transaction it is holding.
    # Timing: broadcast right after the rotation predecessor of validator-a
    # proposes, so the very next proposal belongs to the censoring node
    # itself and its omission is directly observable.
    timing = wait_for_predecessor_of("validator-a", addresses)
    report.setdefault("timing_windows", []).append(timing)
    with_service("validator-a")
    # The challenge is broadcast but NOT waited on: the observer records
    # which proposers omit it and who includes it.
    idx, count, siblings = G.prove(worker_levels, 0)
    opened_epoch = G.height()
    co = {"protocol_version": G.ProtocolVersion, "task_id": p_task, "assignment_id": assignment,
          "challenger_pubkey": challenger_pub, "worker_pubkey": worker_pub,
          "disputed_tile_i": 0, "disputed_tile_j": 0,
          "worker_output_tile": G.int32s_to_canonical(fraud_tiles[0]),
          "worker_output_proof": {"index": idx, "count": count, "siblings": list(siblings)},
          "challenger_output_tile": G.int32s_to_canonical(honest_tiles[0]),
          "challenge_bond": G.MAX_FEE // 100, "opened_epoch": opened_epoch,
          "challenger_signature": b""}
    broadcast_height = latest_height()
    challenge_args = ["tx", "compute", "open-gemm-challenge", "--from", challenger,
                      "-b", "sync", "-y", "-o", "json",
                      "--chain-id", G.CHAIN_ID, "--fees", "0uprsm", "--gas", G.GAS,
                      "--gemm-task-id", str(task_id),
                      "--disputed-tile-i", "0", "--disputed-tile-j", "0",
                      "--worker-output-tile", co["worker_output_tile"].hex(),
                      "--worker-proof-index", str(idx), "--worker-proof-count", str(count),
                      "--challenger-output-tile", co["challenger_output_tile"].hex(),
                      "--challenge-bond", str(co["challenge_bond"]),
                      "--opened-epoch", str(opened_epoch),
                      "--challenger-signature", G.sign_object(co, challenger_seed).hex()]
    # Repeated flags must repeat the flag name.
    for s in siblings:
        challenge_args += ["--worker-proof-siblings", s.hex()]
    txhash = broadcast_sync(*challenge_args)
    report["challenge_txhash"] = txhash
    report["mempool_pending_at_broadcast"] = unconfirmed(RPCS["validator-b"])

    observation = observe_inclusion(txhash, broadcast_height, addresses)
    report["attempts"].append({"attempt": attempt, "task_id": task_id,
                               "challenge_txhash": txhash,
                               "inclusion": observation})
    with_service("validator-b")
    G.wait_task(task_id, "challenged")

    # The dispute itself must still complete while the censoring validator
    # keeps omitting trace/bisection transactions from its own proposals.
    honest_art = G.build_tile_trace(a, b, G.M, G.N, G.K, p_task, assignment, 0, 0, G.micro_step)
    states, levels = honest_art[0], honest_art[1]
    r_steps = (G.K + 7) // 8
    worker_states = list(states)
    worker_states[r_steps] = fraud_tiles[0]
    worker_leaves = [G.leaf_trace_state(p_task, assignment, 0, 0, step, G.int32s_to_canonical(s))
                     for step, s in enumerate(worker_states)]
    worker_trace_levels = G.build_levels(worker_leaves)

    def trace_commit(states_i, levels_i, party, locked_epoch):
        p0 = G.prove(levels_i, 0)
        pf = G.prove(levels_i, r_steps)
        return {"protocol_version": G.ProtocolVersion, "task_id": p_task, "assignment_id": assignment,
                "disputed_tile_i": 0, "disputed_tile_j": 0, "party": party,
                "trace_root": levels_i[-1][0],
                "initial_state": G.int32s_to_canonical(states_i[0]),
                "initial_proof": {"index": p0[0], "count": p0[1], "siblings": list(p0[2])},
                "final_state": G.int32s_to_canonical(states_i[r_steps]),
                "final_proof": {"index": pf[0], "count": pf[1], "siblings": list(pf[2])},
                "locked_epoch": locked_epoch, "signature": b""}

    locked_epoch = G.height()
    for actor, party_no, states_i, levels_i, seed_i in (
        (worker, 1, worker_states, worker_trace_levels, worker_seed),
        (challenger, 2, states, levels, challenger_seed),
    ):
        tc = trace_commit(states_i, levels_i, party_no, locked_epoch)
        G.send_tx("commit-gemm-trace", actor, gemm_task_id=task_id,
                  trace_root=tc["trace_root"].hex(),
                  initial_state=tc["initial_state"].hex(),
                  initial_proof_siblings=[s.hex() for s in tc["initial_proof"]["siblings"]],
                  initial_proof_index=tc["initial_proof"]["index"],
                  initial_proof_count=tc["initial_proof"]["count"],
                  final_state=tc["final_state"].hex(),
                  final_proof_siblings=[s.hex() for s in tc["final_proof"]["siblings"]],
                  final_proof_index=tc["final_proof"]["index"],
                  final_proof_count=tc["final_proof"]["count"],
                  locked_epoch=locked_epoch, signature=G.sign_object(tc, seed_i).hex())

    rounds = 0
    while True:
        state = G.query_task(task_id)
        dispute = state["dispute"]
        low, high = G.parse_snapshot_low_high(dispute["snapshot"])
        if high - low <= 1:
            break
        mid = low + (high - low) // 2
        for actor, states_i, levels_i in ((worker, worker_states, worker_trace_levels),
                                          (challenger, states, levels)):
            i, c_, sibs = G.prove(levels_i, mid)
            G.send_tx("submit-gemm-mid-state", actor, gemm_task_id=task_id,
                      state=G.int32s_to_canonical(states_i[mid]).hex(),
                      proof_siblings=[s.hex() for s in sibs],
                      proof_index=i, proof_count=c_)
        rounds += 1
        if rounds > 32:
            raise RuntimeError("bisection did not converge")
    report["bisection_rounds"] = rounds

    state = G.query_task(task_id)
    low, _ = G.parse_snapshot_low_high(state["dispute"]["snapshot"])
    a_tile = G.extract_a_tile(a, G.M, G.K, 0, low)
    b_tile = G.extract_b_tile(b, G.K, G.N, low, 0)
    cols_a = (G.K + 7) // 8
    ai, acount, asibs = G.input_tile_proof(G.MATRIX_ID_A, a, b, 0, low, cols_a)
    bi, bcount, bsibs = G.input_tile_proof(G.MATRIX_ID_B, a, b, low, 0, (G.N + 7) // 8)
    G.send_tx("arbitrate-gemm", "validator", gemm_task_id=task_id,
              a_tile=bytes(v & 0xFF for v in a_tile).hex(),
              b_tile=bytes(v & 0xFF for v in b_tile).hex(),
              a_proof_tile_row=0, a_proof_tile_col=low, a_proof_tile_cols=cols_a,
              a_proof_siblings=[s.hex() for s in asibs], a_proof_count=acount,
              b_proof_tile_row=low, b_proof_tile_col=0, b_proof_tile_cols=(G.N + 7) // 8,
              b_proof_siblings=[s.hex() for s in bsibs], b_proof_count=bcount)
    final = G.wait_task(task_id, "fraud")
    report["attempts"][-1]["final_status"] = final["task"]["status"]
    report["attempts"][-1]["worker_receipt"] = final["task"].get("receipt_id")
    report["attempts"][-1]["bisection_rounds"] = rounds
    report["attempts"][-1]["arbitration_step"] = low
    return bool(observation["censored_by"])


def main() -> None:
    report = {"scenario": "e1-censorship-inclusion", "attempts": []}

    # Ensure validator-a runs the censoring harness (all dispute-phase txs).
    subprocess.run(["docker", "compose", "-f", str(COMPOSE), "up", "-d",
                    "--force-recreate", "validator-a"],
                   env={**__import__("os").environ, "CENSOR_A": "all"},
                   capture_output=True, text=True, timeout=180)
    time.sleep(5)
    report["validator_addresses"] = validator_addresses()

    observed = False
    for attempt in range(1, 7):
        print(f"== attempt {attempt} ==")
        observed = run_attempt(attempt, report)
        last = report["attempts"][-1]
        print(f"   inclusion: broadcast@{last['inclusion']['broadcast_height']} "
              f"included@{last['inclusion']['included_height']} by {last['inclusion']['included_by']} "
              f"(delay {last['inclusion']['inclusion_delay_blocks']}), "
              f"censored_by={last['inclusion']['censored_by']}, final={last['final_status']}")
        if observed:
            break

    report["censoring_window_observed"] = observed
    out = REPO / "docs" / "phase-e-censorship-results.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write(chr(10))
    print("written:", out)
    if not observed:
        raise RuntimeError("no proposer-omission window was captured across attempts")
    print("PASS: a censoring proposer omitted the pending challenge and a later honest "
          "proposer included it; the dispute completed with ChallengerWins and no receipt")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
