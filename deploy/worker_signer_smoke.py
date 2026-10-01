"""Exercise the deployed worker's isolated devnet keyring against the live chain."""

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path

from chain_smoke import CHAIN_ID, COMPOSE, balance, cli, height, rpc, wait_for
from compute_smoke import account, query_worker, submit, task_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "network"))
from prisma_network.core import Identity, ModelPin, canonical, digest
from prisma_network.server import WorkerAttestation

SIGNER = "dev-worker"
SIGNER_FLAGS = ("--home", "/signer", "--keyring-dir", "/signer",
                "--keyring-backend", "test")


def worker_cli(*args: str) -> str:
    command = ["docker", "compose", "-f", str(COMPOSE), "exec", "-T", "worker", "prismad", *args]
    result = subprocess.run(command, cwd=COMPOSE.parent.parent, capture_output=True,
                            text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "worker-side prismad command failed")
    return result.stdout.strip()


def worker_tx(*args: str) -> str:
    data = json.loads(worker_cli("tx", *args, "--from", SIGNER,
                                 "--chain-id", CHAIN_ID, "--node", "http://chain:26657",
                                 *SIGNER_FLAGS, "--fees", "0uprsm", "--gas", "2000000",
                                 "--yes", "--output", "json"))
    if int(data.get("code", 1)) != 0 or not data.get("txhash"):
        raise RuntimeError(f"worker-side transaction rejected at broadcast: {data}")
    txhash = data["txhash"]

    def included():
        response = rpc("/tx?hash=0x" + txhash)
        return response.get("result")

    result = wait_for(included, "worker-side transaction inclusion")
    if int(result["tx_result"]["code"]) != 0:
        raise RuntimeError(f"worker-side transaction failed in block: {txhash}")
    return txhash


def chain_task_for_model(model_id: str) -> tuple[int, dict]:
    latest = None
    for task_id in range(1, 513):
        try:
            task = task_state(task_id)
        except RuntimeError as exc:
            if "not found" in str(exc).lower():
                break
            raise
        if task["model_id"] == model_id:
            latest = task_id, task
    if latest is None:
        raise RuntimeError("posted signer task not found")
    return latest


def finish_task(task_id: int, api_key: str) -> None:
    task = task_state(task_id)
    if task["status"] != "pending" or task["mode"] != "lightweight":
        raise RuntimeError("task is not a pending lightweight receipt")
    suffix = uuid.uuid4().hex[:12]
    for role in ("a", "b"):
        name = f"signer-monitor-{role}-{suffix}"
        address = account(name)
        submit("bank", "send", "validator", address, "2000000uprsm", signer="validator")
        submit("compute", "bond-worker", "--amount", "1000000", signer=name)
        submit("compute", "attest-result", "--task-id", str(task_id), signer=name)
    wait_for(lambda: height() > int(task_state(task_id)["challenge_end"]),
             "signer receipt confirmation window", seconds=180)
    submit("compute", "finalize-task", "--task-id", str(task_id), signer="validator")
    settled = task_state(task_id)
    if settled["status"] != "settled" or settled["charged_fee"] != 2000:
        raise RuntimeError("deployed signer task did not settle at its two-token tariff")
    request = urllib.request.Request(f"http://127.0.0.1:8080/v1/tasks/{task_id}",
                                     headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=10) as response:
        status = json.load(response)
    if (status["chain_status"] != "settled" or not status["settlement_final"]
            or status["billing"]["charged_uprsm"] != "2000"
            or status["billing"]["refunded_uprsm"] != "8000"):
        raise RuntimeError("gateway did not observe final charge and refund")
    print(f"PASS: task {task_id} settled through two bonded monitors; "
          "gateway API reports 2000uprsm charged and 8000uprsm refunded")


def main() -> None:
    wait_for(lambda: height() >= 1, "first block")
    config = dict(line.split("=", 1) for line in (COMPOSE.parent / ".env").read_text().splitlines()
                  if line and not line.startswith("#"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--finish-task", type=int)
    options = parser.parse_args()
    if options.finish_task:
        finish_task(options.finish_task, config["PRISMA_CLIENT_API_KEY"])
        return
    worker_identity = Identity.from_seed_b64(config["PRISMA_NODE_SEED_WORKER_B64"])
    gateway_identity = Identity.from_seed_b64(config["PRISMA_NODE_SEED_GATEWAY_B64"])
    try:
        worker_address = worker_cli("keys", "show", SIGNER, "-a", *SIGNER_FLAGS)
    except RuntimeError:
        worker_cli("keys", "add", SIGNER, *SIGNER_FLAGS, "--no-backup")
        worker_address = worker_cli("keys", "show", SIGNER, "-a", *SIGNER_FLAGS)
    if not worker_address.startswith("prsm1"):
        raise RuntimeError("worker signer account is not a Prisma address")
    if balance(worker_address) < 1_100_000:
        submit("bank", "send", "validator", worker_address, "2000000uprsm", signer="validator")
    bonded = query_worker(worker_address)
    if not bonded.get("network_public_key"):
        worker_tx("compute", "bond-worker", "--amount", "1000000",
                  "--network-public-key", worker_identity.public_key.hex(),
                  "--network-key-proof", worker_identity.network_key_proof_hex(CHAIN_ID, worker_address))
    bonded = query_worker(worker_address)
    if base64.b64decode(bonded.get("network_public_key", "")) != worker_identity.public_key:
        raise RuntimeError("isolated signer account is not bound to this worker node")

    gateway_name = "deployed-gateway-signer"
    gateway_address = account(gateway_name)
    if balance(gateway_address) < 1_100_000:
        submit("bank", "send", "validator", gateway_address, "2000000uprsm", signer="validator")
    gateway_bond = query_worker(gateway_address)
    if not gateway_bond.get("network_public_key"):
        submit("compute", "bond-worker", "--amount", "1000000",
               "--network-public-key", gateway_identity.public_key.hex(),
               "--network-key-proof", gateway_identity.network_key_proof_hex(CHAIN_ID, gateway_address),
               signer=gateway_name)
    gateway_bond = query_worker(gateway_address)
    if base64.b64decode(gateway_bond.get("network_public_key", "")) != gateway_identity.public_key:
        raise RuntimeError("gateway account is not bound to gateway node key")

    pin = ModelPin(model_id=config["PRISMA_MODEL_ID"], weights_digest=config["PRISMA_WEIGHTS_DIGEST"],
                   tokenizer_digest=config["PRISMA_TOKENIZER_DIGEST"],
                   runtime_digest=config["PRISMA_RUNTIME_DIGEST"],
                   spec_version=config["PRISMA_SPEC_VERSION"])
    if pin.model_digest != config["PRISMA_MODEL_DIGEST"]:
        raise RuntimeError("local mock model pin has changed")
    try:
        raw = json.loads(cli("query", "compute", "model", "--model-id", pin.model_id,
                             "--spec-version", pin.spec_version,
                             "--home", "/data", "--output", "json"))
        registered = json.loads(base64.b64decode(raw["model_json"]))
        if (registered["version"] != pin.spec_version or
                base64.b64decode(registered["weights_digest"]).hex() != pin.weights_digest or
                base64.b64decode(registered["tokenizer_digest"]).hex() != pin.tokenizer_digest or
                base64.b64decode(registered["image_digest"]).hex() != pin.runtime_digest):
            raise RuntimeError("chain model registration conflicts with worker model pin")
    except RuntimeError as exc:
        if "not found" not in str(exc).lower():
            raise
        submit("compute", "register-model", "--model-id", pin.model_id,
               "--spec-version", pin.spec_version, "--mode", "lightweight",
               "--image-digest", pin.runtime_digest,
               "--tokenizer-digest", pin.tokenizer_digest,
               "--weights-digest", pin.weights_digest, signer="validator")

    suffix = uuid.uuid4().hex[:12]
    commitment = digest({"signer-only-synthetic": suffix})
    submit("compute", "post-task", "--mode", "lightweight", "--model-id", pin.model_id,
           "--spec-version", pin.spec_version, "--input-commitment", commitment,
           "--data-ref", "encrypted:signer-only-synthetic", "--max-fee", "10000",
           "--deadline", str(height() + 1000), "--privacy-tier", "tier0_relative",
           signer="validator")
    task_id, posted = chain_task_for_model(pin.model_id)
    if posted["status"] != "posted":
        raise RuntimeError("signer-only task was not posted")
    worker_tx("compute", "accept-task", "--task-id", str(task_id))
    output_digest = digest("synthetic signer acceptance only")
    att = WorkerAttestation(task_id=str(task_id), attempt_id="signer-" + suffix,
                            worker_node_id=worker_identity.node_id,
                            gateway_node_id=gateway_identity.node_id,
                            group_id="mock-one-stage", lease_epoch=1,
                            model_id=pin.model_id, model_digest=pin.model_digest,
                            spec_version=pin.spec_version, input_commitment=commitment,
                            output_commitment=output_digest, output_tokens=2,
                            completed_at_ms=1_700_000_000_000)
    payload = {**att.model_dump(), "mode": "lightweight",
               "worker_attestation": att.model_dump(),
               "worker_signature": worker_identity.sign("prisma:worker-receipt:v1", att.model_dump())}
    receipt = {**payload, "gateway_signature": gateway_identity.sign("prisma:gateway-receipt:v1", payload)}
    body = json.dumps({"receipt": receipt}).encode()
    request = urllib.request.Request("http://127.0.0.1:8081/v1/submit-receipt", body,
                                     {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=40) as response:
        if response.status != 200 or not json.load(response)["submitted"]:
            raise RuntimeError("deployed worker did not submit its signed receipt")
    expected_digest = hashlib.sha256(canonical(receipt)).digest()
    pending = task_state(task_id)
    if (pending["status"] != "pending" or
            base64.b64decode(pending["receipt_digest"]) != expected_digest):
        raise RuntimeError("deployed worker receipt did not enter chain pending state")
    with urllib.request.urlopen(request, timeout=40) as response:
        if response.status != 200 or not json.load(response)["submitted"]:
            raise RuntimeError("duplicate receipt was not idempotent")
    print(f"PASS: deployed worker used its isolated test keyring to submit task {task_id}; "
          "matching receipt reached chain pending state and duplicate delivery was idempotent")
    finish_task(task_id, config["PRISMA_CLIENT_API_KEY"])


if __name__ == "__main__":
    main()
