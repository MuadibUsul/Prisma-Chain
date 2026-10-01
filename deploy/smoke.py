"""Exercise a local chain RPC and one unfunded mock delivery through the gateway."""

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "network"))
from prisma_network.core import verify


def request(url: str, *, body: dict | None = None, api_key: str | None = None) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = None
    if body is not None:
        data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)


def digest(value: object) -> str:
    packed = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(packed).hexdigest()


def wait_for(check, label: str, seconds: int = 60):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        try:
            return check()
        except (OSError, ValueError, KeyError, AssertionError):
            time.sleep(2)
    raise RuntimeError(f"{label} was not ready in {seconds}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-chain", action="store_true", help="check network only")
    args = parser.parse_args()
    env = dict(line.strip().split("=", 1) for line in
               (Path(__file__).resolve().parent / ".env").read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#"))
    if not args.no_chain:
        status = wait_for(lambda: request("http://127.0.0.1:26657/status"), "chain RPC")
        assert status["result"]["node_info"]["network"] == "prisma-local-1"
    gateway = "http://127.0.0.1:8080"
    privacy = wait_for(lambda: request(gateway + "/v1/privacy"), "gateway")
    assert privacy["tier"] == "tier0_relative"
    params = urllib.parse.urlencode({"model_digest": env["PRISMA_MODEL_DIGEST"],
                                     "spec_version": env["PRISMA_SPEC_VERSION"]})
    route = wait_for(lambda: request(gateway + "/v1/routes/demo-model?" + params), "mock route")
    assert route["group_id"] == "mock-one-stage"
    messages = [{"role": "user", "content": "local devnet smoke"}]
    committed = {"messages": messages, "max_tokens": 8, "temperature": 0.0}
    task_id = "dev-smoke-" + uuid.uuid4().hex
    payload = {"task": {"task_id": task_id, "mode": "lightweight", "model_id": "demo-model",
                        "model_digest": env["PRISMA_MODEL_DIGEST"], "spec_version": env["PRISMA_SPEC_VERSION"],
                        "input_commitment": digest(committed), "data_ref": "dev://local-mock",
                        "max_fee": "1000", "deadline": 1000000,
                        "privacy_tier": "tier0_relative"}, **committed}
    first = request(gateway + "/v1/inference", body=payload, api_key=env["PRISMA_CLIENT_API_KEY"])
    assert first["status"] == "provisional_delivery" and first["output"] == "MOCK"
    assert first["receipt"]["task_id"] == task_id and first["receipt"]["output_tokens"] == 1
    receipt = first["receipt"]
    keys = json.loads((Path(__file__).resolve().parent / "data" / "trusted-keys.json").read_text(encoding="utf-8"))
    gateway_sig = receipt["gateway_signature"]
    assert verify(keys[receipt["gateway_node_id"]], "prisma:gateway-receipt:v1",
                  {key: value for key, value in receipt.items() if key != "gateway_signature"}, gateway_sig)
    assert verify(keys[receipt["worker_node_id"]], "prisma:worker-receipt:v1",
                  receipt["worker_attestation"], receipt["worker_signature"])
    replay = request(gateway + "/v1/inference", body=payload, api_key=env["PRISMA_CLIENT_API_KEY"])
    assert replay["already_completed"] and replay["receipt"] == first["receipt"]
    task_status = request(gateway + "/v1/tasks/" + task_id, api_key=env["PRISMA_CLIENT_API_KEY"])
    assert task_status["chain_status"] == "unfunded_dev"
    assert task_status["delivery_status"] == "provisional_delivery"
    assert task_status["settlement_final"] is False
    print("PASS: chain RPC, Tier 0 notice, measured route, signed mock delivery, unfunded status, idempotent replay"
          if not args.no_chain else
          "PASS: Tier 0 notice, measured route, signed mock delivery, unfunded status, idempotent replay")


if __name__ == "__main__":
    main()
