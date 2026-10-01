"""The worker signer must own the chain account and retry idempotently."""

import asyncio
import base64
import hashlib
import json
from pathlib import Path

import pytest

from prisma_network.chain_tx import DevnetCliReceiptSubmitter
from prisma_network.core import Conflict, canonical


class FakeQueries:
    def __init__(self, key):
        self.key = key
        self.record = {"status": "accepted", "mode": "lightweight", "worker": "prsmworker",
                       "deadline": 50}

    async def task(self, task_id):
        assert task_id == 1
        return self.record

    async def worker(self, account):
        assert account == "prsmworker"
        return 1_000_000, self.key


def test_cli_signer_checks_account_and_deduplicates(tmp_path):
    sample = json.loads((Path(__file__).parents[1] / "examples" / "contract.json").read_text())
    receipt = sample["delivery_receipt"]
    key = base64.b64decode(sample["announcement"]["capability"]["public_key"])
    chain = FakeQueries(key)
    signer = DevnetCliReceiptSubmitter(chain, chain_id="prisma-test-1",
                                      rpc_url="http://127.0.0.1:26657",
                                      signer_name="worker", signer_home=str(tmp_path),
                                      network_public_key=key)
    calls = []

    async def chain_status():
        return 10

    async def command(*args):
        calls.append(args)
        if args[0] == "keys":
            return "prsmworker"
        receipt_path = Path(args[args.index("--receipt-json") + 1])
        assert receipt_path.read_bytes() == canonical(receipt)
        assert args[args.index("--from") + 1] == "worker"
        return '{"code":0,"txhash":"ABC123"}'

    async def included(txhash):
        assert txhash == "ABC123"
        chain.record = {**chain.record, "status": "pending",
                        "receipt_digest": base64.b64encode(hashlib.sha256(canonical(receipt)).digest()).decode()}

    signer._chain_status = chain_status
    signer._command = command
    signer._included = included
    asyncio.run(signer(receipt))
    assert [call[0] for call in calls] == ["keys", "tx"]
    assert not list(tmp_path.glob("receipt-*.json"))
    asyncio.run(signer(receipt))
    assert len(calls) == 2
    chain.record["receipt_digest"] = base64.b64encode(b"x" * 32).decode()
    with pytest.raises(Conflict, match="different receipt"):
        asyncio.run(signer(receipt))


def test_cli_signer_rejects_unbound_account_before_broadcast(tmp_path):
    sample = json.loads((Path(__file__).parents[1] / "examples" / "contract.json").read_text())
    receipt = sample["delivery_receipt"]
    key = base64.b64decode(sample["announcement"]["capability"]["public_key"])
    chain = FakeQueries(key)
    chain.record["worker"] = "prsmother"
    signer = DevnetCliReceiptSubmitter(chain, chain_id="prisma-test-1",
                                      rpc_url="http://127.0.0.1:26657",
                                      signer_name="worker", signer_home=str(tmp_path),
                                      network_public_key=key)

    async def chain_status():
        return 10

    async def command(*args):
        if args[0] == "keys":
            return "prsmworker"
        pytest.fail("unbound account reached transaction broadcast")

    signer._chain_status = chain_status
    signer._command = command
    with pytest.raises(Conflict, match="did not accept"):
        asyncio.run(signer(receipt))
