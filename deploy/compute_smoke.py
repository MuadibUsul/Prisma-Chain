"""Exercise local compute registration, bond, escrow, and acceptance.

This stops with an accepted task. No automated replay monitor or lightweight
receipt-based settlement is installed in this devnet.
"""

import base64
import json
import uuid

from chain_smoke import CHAIN_ID, FLAGS, balance, cli, height, rpc, wait_for

PROGRAM_HEX = "5b7b224f70223a312c22447374223a302c2241223a302c2242223a302c2243223a302c22496d6d223a34327d5d"
EMPTY_INPUT_DIGEST = "c40db7b6e1d6624ba2d732d90d9bb2d5443c6b39283cb090a638b69117881f9c"
BOND = 1_000_000
FEE = 1_000_000


def submit(*args: str, signer: str) -> str:
    data = json.loads(cli("tx", *args, "--from", signer, "--chain-id", CHAIN_ID, *FLAGS,
                          "--fees", "0uprsm", "--gas", "2000000", "--yes", "--output", "json"))
    if int(data.get("code", 1)) != 0 or not data.get("txhash"):
        raise RuntimeError(f"compute transaction rejected: {data}")
    txhash = data["txhash"]

    def included():
        answer = rpc("/tx?hash=0x" + txhash)
        return answer.get("result")

    result = wait_for(included, f"transaction {txhash}")
    if int(result["tx_result"]["code"]) != 0:
        raise RuntimeError(f"compute transaction failed in block: {txhash}")
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


def main() -> None:
    wait_for(lambda: height() >= 1, "first block")
    requester = cli("keys", "show", "validator", "-a", *FLAGS)
    try:
        worker = cli("keys", "show", "compute-smoke-worker", "-a", *FLAGS)
    except RuntimeError:
        cli("keys", "add", "compute-smoke-worker", *FLAGS, "--no-backup")
        worker = cli("keys", "show", "compute-smoke-worker", "-a", *FLAGS)
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
    print("Task remains accepted; this devnet has no automated replay monitor or payout for it.")


if __name__ == "__main__":
    main()
