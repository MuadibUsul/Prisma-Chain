"""Permissionless watcher: obtain committed GEMM output from independent DA
providers, verify it against the on-chain output_root, run Freivalds, and
open a chain challenge for the localized bad tile.

The watcher never reads anything from the worker: the only data source is
the DA provider bulk endpoint. Output-root verification is mandatory and
instrumented (a fast path that skipped it would fail the tests).

    python deploy/watcher.py --task-id N --providers URL1,URL2,URL3 \
        --chain-chain-id prisma-mv-1 --compose deploy/multivalidator/compose.yaml \
        --service validator-b --keyring-home /data/validator-b --out report.json
"""

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "compute" / "gemmv1" / "python"))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey  # noqa: E402

from gemmv1.canonical_cbor import encode_canonical  # noqa: E402
from gemmv1.freivalds import (  # noqa: E402
    DeterministicSource, VerificationProfile, verify_freivalds)
from gemmv1.merkle_proofs import build_levels  # noqa: E402
from gemmv1.protocol import leaf_output_tile, merkle_root  # noqa: E402
from gemmv1.row_locator import localize_from_rows  # noqa: E402
from gemmv1.tensors import int32s_to_canonical, output_tiles  # noqa: E402
from gemmv1.testgen import gen_test_matrix  # noqa: E402


def download_c(providers, task_id: int) -> bytes:
    errors = []
    for base in providers:
        try:
            with urllib.request.urlopen(f"{base}/v1/da/{task_id}/output", timeout=10) as resp:
                return bytes.fromhex(json.load(resp)["c_hex"])
        except Exception as exc:  # a missing provider is not a verdict
            errors.append(f"{base}: {exc}")
    raise RuntimeError("no DA provider could serve the blob: " + "; ".join(errors))


def rebuild_output_root(c: bytes, m: int, n: int, task_id32: bytes, assignment_id: bytes) -> bytes:
    values = [int.from_bytes(c[i:i + 4], "big", signed=True) for i in range(0, len(c), 4)]
    tiles = output_tiles(values, m, n)
    cols_c = (n + 7) // 8
    leaves = [
        leaf_output_tile(task_id32, assignment_id, i // cols_c, i % cols_c, int32s_to_canonical(t))
        for i, t in enumerate(tiles)
    ]
    return merkle_root(leaves)


class Chain:
    """Thin prismad CLI wrapper against one validator container."""

    def __init__(self, compose: str, service: str, chain_id: str, home: str, keyring_name: str):
        self.compose, self.service, self.chain_id = compose, service, chain_id
        self.home, self.keyring_name = home, keyring_name

    def cli(self, *args, check=True):
        proc = subprocess.run(
            ["docker", "compose", "-f", self.compose, "exec", "-T", self.service, "prismad", *args],
            capture_output=True, text=True, timeout=180)
        out = proc.stdout.strip()
        if check and proc.returncode != 0:
            raise RuntimeError(f"{args[:2]} failed: {proc.stderr.strip() or out}")
        return out

    def query_task(self, task_id: int) -> dict:
        out = self.cli("query", "compute", "gemm-task", "--gemm-task-id", str(task_id), "-o", "json")
        payload = json.loads(out[out.index("{"):])
        return json.loads(base64.b64decode(payload["task_json"]))

    def query_da_status(self, task_id: int) -> dict:
        out = self.cli("query", "compute", "gemm-da-status", "--gemm-task-id", str(task_id), "-o", "json")
        return json.loads(out[out.index("{"):])

    def send(self, args, wait=True):
        proc = subprocess.run(
            ["docker", "compose", "-f", self.compose, "exec", "-T", self.service, "prismad", *args],
            capture_output=True, text=True, timeout=180)
        out = proc.stdout.strip()
        if proc.returncode != 0:
            raise RuntimeError(f"tx failed: {proc.stderr.strip() or out}")
        payload = json.loads(out[out.index("{"):])
        if int(payload.get("code", 1)) != 0:
            raise RuntimeError(f"mempool rejected: {payload.get('raw_log')}")
        if not wait:
            return payload
        deadline = time.time() + 90
        while time.time() < deadline:
            out = self.cli("query", "tx", payload["txhash"], "-o", "json", check=False)
            if out and "{" in out:
                result = json.loads(out[out.index("{"):])
                if int(result.get("code", 0)) != 0:
                    raise RuntimeError(f"tx failed in block: {result.get('raw_log')}")
                return result
            time.sleep(1)
        raise RuntimeError("tx not included")


def sign_object(payload: dict, private_hex: str) -> bytes:
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex))
    return private.sign(b"PRISMA_GEMM_SIG_V1\x00" + encode_canonical(payload))


def wait_next_proposer_is_a(rpc_base: str, out_path: str, seconds: int = 240) -> None:
    """Learn the proposer rotation from recent blocks and block until the
    rotation predecessor of validator-a has just proposed."""

    def rpc(path):
        with urllib.request.urlopen(rpc_base + path, timeout=5) as resp:
            return json.load(resp)["result"]

    addresses = {}
    for port in (26661, 26662, 26663, 26664):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as resp:
                info = json.load(resp)["result"]["validator_info"]
            if info.get("address"):
                addresses[info["address"].upper()] = port
        except Exception:
            pass
    a_address = None
    for address, port in addresses.items():
        if port == 26661:
            a_address = address
    deadline = time.time() + seconds
    last = None
    while time.time() < deadline:
        top = int(rpc("/status")["sync_info"]["latest_block_height"])
        order = []
        for height in range(top - 7, top + 1):
            proposer = rpc(f"/block?height={height}")["block"]["header"]["proposer_address"].upper()
            if not order or order[-1] != proposer:
                order.append(proposer)
        nxt = {order[i]: order[i + 1] for i in range(len(order) - 1)}
        if nxt.get(order[-1]) == a_address:
            if int(rpc("/status")["sync_info"]["latest_block_height"]) == top:
                last = {"height": top, "next_expected": order[-1], "target": "validator-a"}
                break
        time.sleep(0.2)
    if last is None:
        raise RuntimeError("never observed the rotation window before validator-a")
    print("broadcasting in the window before validator-a proposes:", last, flush=True)
    if out_path:
        Path(out_path).with_suffix(".timing.json").write_text(
            json.dumps(last) + chr(10), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--providers", required=True, help="comma-separated DA endpoints")
    parser.add_argument("--compose", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--keyring-home", required=True)
    parser.add_argument("--chain-id", required=True)
    parser.add_argument("--watcher-key-name", default="")
    parser.add_argument("--watcher-key-hex", default="", help="secp256k1 account hex imported by the caller")
    parser.add_argument("--challenger-key-hex", default="", help="Ed25519 network key of the watcher")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="")
    parser.add_argument("--rpc-b", default="http://127.0.0.1:26662",
                        help="RPC used to time the broadcast before the censoring proposer's slot")
    parser.add_argument("--wait-predecessor", action="store_true",
                        help="wait until the next proposer is validator-a, then broadcast at once")
    args = parser.parse_args()
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    report = {"task_id": args.task_id, "providers": providers}

    chain = Chain(args.compose, args.service, args.chain_id, args.keyring_home, args.watcher_key_name)
    task = chain.query_task(args.task_id)
    if task["status"] not in ("result_submitted", "challenged"):
        raise RuntimeError(f"task is not in the challenge window: {task['status']}")
    m, n, k = int(task["m"]), int(task["n"]), int(task["k"])
    p_task = base64.b64decode(task["protocol_task_id"])
    assignment = base64.b64decode(task["assignment_id"])
    output_root = base64.b64decode(task["output_root"])

    started = time.time()
    c = download_c(providers, args.task_id)
    report["bulk_download_seconds"] = round(time.time() - started, 3)
    report["output_bytes"] = len(c)

    started = time.time()
    rebuilt = rebuild_output_root(c, m, n, p_task, assignment)
    report["root_verification_seconds"] = round(time.time() - started, 3)
    report["root_verified"] = rebuilt == output_root
    if not report["root_verified"]:
        raise RuntimeError("OUTPUT_DATA_COMMITMENT_MISMATCH: DA blob does not match output_root")

    values = [int.from_bytes(c[i:i + 4], "big", signed=True) for i in range(0, len(c), 4)]
    a = gen_test_matrix(ord("A"), args.seed, m * k)
    b = gen_test_matrix(ord("B"), args.seed, k * n)
    started = time.time()
    verification = verify_freivalds(a, b, values, m, n, k,
                                    VerificationProfile(8), DeterministicSource(b"watcher"))
    report["freivalds_seconds"] = round(time.time() - started, 3)
    report["freivalds_passed"] = verification["passed"]
    if verification["passed"]:
        report["outcome"] = "no_challenge_needed"
        print(json.dumps(report, indent=2))
        return

    loc, found = localize_from_rows(a, b, values, m, n, k, verification["residual_rows"])
    if not found:
        raise RuntimeError("residual rows did not localize a bad tile")
    report["bad_row"] = loc["row"]
    report["bad_tile"] = [loc["tile_i"], loc["tile_j"]]

    # Independent recomputation of the canonical tile for the challenge.
    from gemmv1.row_locator import reference_row_gemm

    tile = []
    for r8 in range(8):
        gi = loc["tile_i"] * 8 + r8
        if gi < m:
            row = reference_row_gemm(a, gi, b, m, n, k)
        else:
            row = [0] * n
        tile.append([row[loc["tile_j"] * 8 + c8] if loc["tile_j"] * 8 + c8 < n else 0
                     for c8 in range(8)])
    challenger_tile = [tile[r][cc] for r in range(8) for cc in range(8)]

    worker_tiles = output_tiles(values, m, n)
    cols_c = (n + 7) // 8
    idx = loc["tile_i"] * cols_c + loc["tile_j"]
    levels = build_levels([
        leaf_output_tile(p_task, assignment, t // cols_c, t % cols_c, int32s_to_canonical(tt))
        for t, tt in enumerate(worker_tiles)
    ])
    from gemmv1.merkle_proofs import prove

    proof_index, proof_count, proof_siblings = prove(levels, idx)

    challenger_priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(args.challenger_key_hex))
    challenger_pub = challenger_priv.public_key().public_bytes_raw()
    opened_epoch = int(chain.cli("status", "--home", args.keyring_home).split('"latest_block_height":"')[1].split('"')[0]) \
        if False else None
    # Height from the RPC the chain wrapper already talks to:
    status = chain.cli("status", "--home", args.keyring_home, check=False)
    opened_epoch = None
    if "{" in status:
        try:
            opened_epoch = int(json.loads(status[status.index("{"):])["sync_info"]["latest_block_height"])
        except Exception:
            pass
    if opened_epoch is None:
        raise RuntimeError("could not read the chain height")

    worker_pub = base64.b64decode(task["worker_protocol_pub_key"])
    co = {
        "protocol_version": "0.1.1", "task_id": p_task, "assignment_id": assignment,
        "challenger_pubkey": challenger_pub, "worker_pubkey": worker_pub,
        "disputed_tile_i": loc["tile_i"], "disputed_tile_j": loc["tile_j"],
        "worker_output_tile": int32s_to_canonical(worker_tiles[idx]),
        "worker_output_proof": {"index": proof_index, "count": proof_count,
                                "siblings": list(proof_siblings)},
        "challenger_output_tile": int32s_to_canonical(challenger_tile),
        "challenge_bond": int(task["max_fee"]) // 100,
        "opened_epoch": opened_epoch,
        "challenger_signature": b"",
    }
    tx_args = ["tx", "compute", "open-gemm-challenge", "--from", args.watcher_key_name,
               "-b", "sync", "-y", "-o", "json", "--chain-id", args.chain_id,
               "--fees", "0uprsm", "--gas", "2000000",
               "--gemm-task-id", str(args.task_id),
               "--disputed-tile-i", str(loc["tile_i"]), "--disputed-tile-j", str(loc["tile_j"]),
               "--worker-output-tile", co["worker_output_tile"].hex(),
               "--worker-proof-index", str(proof_index), "--worker-proof-count", str(proof_count),
               "--challenger-output-tile", co["challenger_output_tile"].hex(),
               "--challenge-bond", str(co["challenge_bond"]),
               "--opened-epoch", str(opened_epoch),
               "--challenger-signature", sign_object(co, args.challenger_key_hex).hex()]
    for s in proof_siblings:
        tx_args += ["--worker-proof-siblings", s.hex()]
    if args.wait_predecessor:
        # Timing: broadcast at the last moment, so validator-a (the
        # censoring proposer in the combined devnet) proposes the very next
        # block and must actively omit a transaction it already holds.
        wait_next_proposer_is_a(args.rpc_b, args.out)
    result = chain.send(tx_args)
    print("challenge broadcast response:", json.dumps(result)[:400], flush=True)
    try:
        with urllib.request.urlopen("http://127.0.0.1:26662/unconfirmed_txs", timeout=5) as resp:
            print("validator-b mempool n_txs:", json.load(resp)["result"]["n_txs"], flush=True)
    except Exception as exc:
        print("mempool probe failed:", exc, flush=True)
    report["challenge_txhash"] = result.get("txhash")
    report["outcome"] = "challenge_opened"
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
