"""Exercise the local verifiable compute lifecycle against a live devnet."""

import argparse
import base64
import hashlib
import json
import subprocess
import uuid

from chain_smoke import CHAIN_ID, COMPOSE, FLAGS, balance, cli, height, rpc, wait_for

PROGRAM = [{"Op": 1, "Dst": 1, "A": 0, "B": 0, "C": 0, "Imm": 42},
           {"Op": 3, "Dst": 0, "A": 1, "B": 2, "C": 0, "Imm": 0}]
PROGRAM_HEX = json.dumps(PROGRAM, separators=(",", ":")).encode().hex()
HONEST_JOB = {"input": [], "program": [{"op": "set", "dst": 1, "imm": 42},
                                        {"op": "add", "dst": 0, "a": 1, "b": 2}]}
FRAUD_JOB = {"input": [], "program": [{"op": "set", "dst": 1, "imm": 42},
                                       {"op": "set", "dst": 0, "imm": 43}]}
EMPTY_INPUT_DIGEST = "c40db7b6e1d6624ba2d732d90d9bb2d5443c6b39283cb090a638b69117881f9c"
BOND = 1_000_000
FEE = 1_000_000


def submit(*args: str, signer: str, expected_code: int = 0) -> str:
    data = json.loads(cli("tx", *args, "--from", signer, "--chain-id", CHAIN_ID, *FLAGS,
                          "--fees", "0uprsm", "--gas", "2000000", "--yes", "--output", "json"))
    if int(data.get("code", 1)) != 0 or not data.get("txhash"):
        raise RuntimeError(f"compute transaction rejected: {data}")
    txhash = data["txhash"]

    def included():
        answer = rpc("/tx?hash=0x" + txhash)
        return answer.get("result")

    result = wait_for(included, f"transaction {txhash}")
    if int(result["tx_result"]["code"]) != expected_code:
        raise RuntimeError(f"unexpected DeliverTx result for {txhash}: {result['tx_result']}")
    return txhash


def query_model(model_id: str) -> dict:
    response = json.loads(cli("query", "compute", "model", "--model-id", model_id,
                              "--spec-version", "v1", "--home", "/data", "--output", "json"))
    return json.loads(base64.b64decode(response["model_json"]))


def query_worker(address: str) -> dict:
    return json.loads(cli("query", "compute", "worker", "--worker", address,
                          "--home", "/data", "--output", "json"))


def find_task(model_id: str) -> tuple[int, dict]:
    # ponytail: linear scan is sufficient for the bounded local devnet.
    for task_id in range(1, 513):
        try:
            response = json.loads(cli("query", "compute", "task", "--task-id", str(task_id),
                                      "--home", "/data", "--output", "json"))
        except RuntimeError as exc:
            if "not found" in str(exc).lower():
                break
            raise
        task = json.loads(base64.b64decode(response["task_json"]))
        if task["model_id"] == model_id:
            return task_id, task
    raise RuntimeError("newly posted task was not found in the first 512 task IDs")


def put_json(value: dict) -> str:
    path = "/tmp/prisma-compute-smoke-" + uuid.uuid4().hex + ".json"
    command = ["docker", "compose", "-f", str(COMPOSE), "exec", "-T", "chain",
               "sh", "-c", "cat > " + path]
    result = subprocess.run(command, cwd=COMPOSE.parent.parent, input=json.dumps(value),
                            capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "could not write VM evidence")
    return path


def vm_run(job_path: str, *options: str) -> dict:
    command = ["docker", "compose", "-f", str(COMPOSE), "exec", "-T", "chain",
               "prisma-vm", "-job", job_path, *options]
    result = subprocess.run(command, cwd=COMPOSE.parent.parent, capture_output=True,
                            text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "VM execution failed")
    return json.loads(result.stdout)


def output_digest(value: int) -> str:
    return hashlib.sha256(b"prismavm:output:v1\x00" +
                          value.to_bytes(8, "big", signed=True)).hexdigest()


def supply() -> int:
    data = json.loads(cli("query", "bank", "total-supply-of", "uprsm", "--home", "/data",
                          "--output", "json"))
    return int(data["amount"]["amount"])


def account(name: str) -> str:
    try:
        return cli("keys", "show", name, "-a", *FLAGS)
    except RuntimeError:
        cli("keys", "add", name, *FLAGS, "--no-backup")
        return cli("keys", "show", name, "-a", *FLAGS)


def bond_monitor(name: str, address: str) -> None:
    if balance(address) < BOND:
        submit("bank", "send", "validator", address, "2000000uprsm", signer="validator")
    submit("compute", "bond-worker", "--amount", str(BOND), signer=name)


def task_state(task_id: int) -> dict:
    response = json.loads(cli("query", "compute", "task", "--task-id", str(task_id),
                              "--home", "/data", "--output", "json"))
    return json.loads(base64.b64decode(response["task_json"]))


def assert_reserved_released(task_id: int, worker: str, prior: int) -> None:
    task = task_state(task_id)
    if int(query_worker(worker).get("reserved_uprsm", 0)) != prior or int(task.get("reserved_bond", 0)) != 0:
        raise RuntimeError("task bond reservation was not released")


def complete_honest(task_id: int, worker: str, requester: str, module: str,
                    worker_reserved: int, job_path: str, claim: dict) -> None:
    monitor_names = ("compute-smoke-monitor-a", "compute-smoke-monitor-b")
    monitors = [account(name) for name in monitor_names]
    for name, address in zip(monitor_names, monitors):
        bond_monitor(name, address)
    evidence_path = put_json(claim)
    for name in monitor_names:
        review = vm_run(job_path, "-review", evidence_path)
        if review.get("review_action") != "attest":
            raise RuntimeError("monitor did not independently replay and attest")
        submit("compute", "attest-result", "--task-id", str(task_id), signer=name)
    deadline = int(task_state(task_id)["challenge_end"])
    wait_for(lambda: height() > deadline, "full challenge window", seconds=180)
    before = {address: balance(address) for address in (worker, *monitors, module, requester)}
    before_supply = supply()
    submit("compute", "finalize-task", "--task-id", str(task_id), signer="validator")
    after = {address: balance(address) for address in before}
    if (task_state(task_id)["status"] != "settled" or after[worker] - before[worker] != 700_000
            or any(after[a] - before[a] != 50_000 for a in monitors)
            or before[module] - after[module] != FEE or before_supply - supply() != 200_000
            or after[requester] != before[requester]):
        raise RuntimeError("honest settlement or 20/70/5/5 fee split did not match")
    assert_reserved_released(task_id, worker, worker_reserved)
    submit("compute", "finalize-task", "--task-id", str(task_id), signer="validator",
           expected_code=1)
    if (balance(module) != after[module] or balance(worker) != after[worker]
            or supply() != before_supply - 200_000):
        raise RuntimeError("duplicate finalization changed balances or supply")
    print(f"PASS: task {task_id} settled; 200000uprsm burned, worker +700000, "
          "two monitors +50000 each; duplicate payout rejected")


def complete_fraud(task_id: int, worker: str, requester: str, module: str,
                   worker_reserved: int, honest_path: str, fraud_path: str,
                   fraudulent_claim: dict) -> None:
    challenger_name = "compute-smoke-challenger"
    challenger = account(challenger_name)
    if balance(challenger) < BOND:
        submit("bank", "send", "validator", challenger, "2000000uprsm", signer="validator")
    evidence_path = put_json(fraudulent_claim)
    review = vm_run(honest_path, "-review", evidence_path)
    if review.get("review_action") != "challenge":
        raise RuntimeError("honest replay did not detect the fraudulent trace")
    challenger_claim_path = put_json(review["challenger_claim"])
    before = {address: balance(address) for address in (worker, requester, module, challenger)}
    before_supply, before_bond = supply(), int(query_worker(worker)["bonded_uprsm"])
    submit("compute", "start-challenge", "--task-id", str(task_id),
           "--trace-claim-json", challenger_claim_path, signer=challenger_name)
    if task_state(task_id)["status"] != "challenged":
        raise RuntimeError("fraud dispute did not open")
    for name, path in (("compute-smoke-worker", fraud_path), (challenger_name, honest_path)):
        proof = vm_run(path, "-proof", "1")["proof"]
        if not proof["valid"]:
            raise RuntimeError("VM midpoint proof invalid")
        proof_path = put_json({"Siblings": [list(bytes.fromhex(item))
                                            for item in proof["siblings"]]})
        state_path = put_json(proof["state"])
        submit("compute", "challenge-midpoint", "--task-id", str(task_id),
               "--state-json", state_path, "--proof-json", proof_path, signer=name)
    after = {address: balance(address) for address in before}
    slash = min(before_bond // 10, BOND)
    challenge_bond = max(1, FEE // 100)
    if (task_state(task_id)["status"] != "refunded" or after[requester] - before[requester] != FEE
            or after[challenger] - before[challenger] != slash
            or before[module] - after[module] != FEE + slash
            or int(query_worker(worker)["bonded_uprsm"]) != before_bond - slash
            or after[worker] != before[worker] or supply() != before_supply):
        raise RuntimeError("fraud refund, challenger bond, slash, or supply did not match")
    assert_reserved_released(task_id, worker, worker_reserved)
    submit("compute", "finalize-task", "--task-id", str(task_id), signer="validator",
           expected_code=1)
    if balance(module) != after[module] or supply() != before_supply:
        raise RuntimeError("fraudulent task was paid after refund")
    print(f"PASS: task {task_id} fraud proven at one VM step; fee refunded, "
          f"worker slashed {slash}uprsm, challenger bond {challenge_bond} returned, "
          "duplicate payout rejected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("accepted", "honest", "fraud"),
                        default="accepted")
    scenario = parser.parse_args().scenario
    wait_for(lambda: height() >= 1, "first block")
    requester = cli("keys", "show", "validator", "-a", *FLAGS)
    worker = account("compute-smoke-worker")
    module = json.loads(cli("query", "auth", "module-account", "compute", "--home", "/data",
                            "--output", "json"))["account"]["value"]["address"]
    if balance(worker) < BOND:
        submit("bank", "send", "validator", worker, "2000000uprsm", signer="validator")
    model_id = "smoke-vm-" + uuid.uuid4().hex[:12]
    submit("compute", "register-model", "--model-id", model_id, "--spec-version", "v1",
           "--mode", "verifiable", "--image-digest", "11" * 32,
           "--tokenizer-digest", "22" * 32, "--weights-digest", "33" * 32,
           "--program", PROGRAM_HEX, signer="validator")
    model = query_model(model_id)
    if model["id"] != model_id or model["mode"] != "verifiable" or model["owner"] != requester:
        raise RuntimeError("registered model query did not match the transaction")
    module_before, worker_before = balance(module), balance(worker)
    worker_before_state = query_worker(worker)
    submit("compute", "bond-worker", "--amount", str(BOND), signer="compute-smoke-worker")
    worker_after_bond = query_worker(worker)
    if (int(worker_after_bond.get("bonded_uprsm", 0)) != int(worker_before_state.get("bonded_uprsm", 0)) + BOND
            or balance(worker) != worker_before - BOND or balance(module) != module_before + BOND):
        raise RuntimeError("worker bond was not escrowed exactly")
    requester_before_post = balance(requester)
    submit("compute", "post-task", "--mode", "verifiable", "--model-id", model_id,
           "--spec-version", "v1", "--input-commitment", EMPTY_INPUT_DIGEST,
           "--data-ref", "dev://public-empty-input", "--max-fee", str(FEE),
           "--deadline", str(height() + 1000), "--privacy-tier", "public", signer="validator")
    task_id, task = find_task(model_id)
    if (task["status"] != "posted" or task["requester"] != requester or task["max_fee"] != FEE
            or balance(requester) != requester_before_post - FEE
            or balance(module) != module_before + BOND + FEE):
        raise RuntimeError("posted task or fee escrow did not match")
    submit("compute", "accept-task", "--task-id", str(task_id), signer="compute-smoke-worker")
    _, accepted = find_task(model_id)
    worker_final = query_worker(worker)
    if (accepted["status"] != "accepted" or accepted["worker"] != worker
            or accepted["reserved_bond"] != BOND
            or int(worker_final.get("reserved_uprsm", 0)) != int(worker_after_bond.get("reserved_uprsm", 0)) + BOND
            or balance(module) != module_before + BOND + FEE):
        raise RuntimeError("accepted task, bond reservation, or escrow did not match")
    print(f"PASS: registered {model_id}, bonded distinct worker {worker}, "
          f"posted task {task_id}, accepted with {BOND}uprsm reserved and {FEE}uprsm fee escrowed")
    if scenario == "accepted":
        print("Task remains accepted; select --scenario honest or fraud for full lifecycle checks.")
        return

    honest_path = put_json(HONEST_JOB)
    job_path = honest_path if scenario == "honest" else put_json(FRAUD_JOB)
    canonical = vm_run(honest_path, "-claim")
    if bytes.fromhex(canonical["program_digest"]) != bytes(model["program_digest"]):
        raise RuntimeError("honest replay program does not match the registered model")
    result = vm_run(job_path, "-claim")
    if result["steps"] != 2 or result["input_digest"] != EMPTY_INPUT_DIGEST:
        raise RuntimeError("VM result did not match registered public input")
    claim = {"trace_claim": result["trace_claim"]}
    claim_path = put_json(result["trace_claim"])
    submit("compute", "submit-result", "--task-id", str(task_id),
           "--output-digest", output_digest(result["output"]),
           "--trace-root", result["trace_root"], "--trace-claim-json", claim_path,
           signer="compute-smoke-worker")
    if task_state(task_id)["status"] != "pending":
        raise RuntimeError("worker result did not enter the challenge window")
    prior_reserved = int(worker_after_bond.get("reserved_uprsm", 0))
    if scenario == "honest":
        complete_honest(task_id, worker, requester, module, prior_reserved,
                        honest_path, claim)
    else:
        complete_fraud(task_id, worker, requester, module, prior_reserved,
                       honest_path, job_path, claim)


if __name__ == "__main__":
    main()
