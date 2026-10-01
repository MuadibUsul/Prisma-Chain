"""Funded task authorization must reject stale or mismatched chain state."""

import asyncio
import base64
import hashlib
from copy import deepcopy

from prisma_network.chain_auth import ChainTaskAuthorizer
from prisma_network.core import ModelPin, Route, TaskEnvelope


def encoded(byte: int) -> str:
    return base64.b64encode(bytes([byte]) * 32).decode()


class FakeChain:
    def __init__(self, task: dict, model: dict):
        self.record = task
        self.registration = model
        self.block = 10
        self.bonds = {"prsmworker": 1_000_000, "prsmstage": 1_000_000}
        self.keys = {"prsmworker": bytes([0xAA]) * 32, "prsmstage": bytes([0xBB]) * 32}

    async def task(self, task_id: int) -> dict:
        assert task_id == 7
        return self.record

    async def model(self, model_id: str, spec_version: str) -> dict:
        assert (model_id, spec_version) == ("test-model", "v1")
        return self.registration

    async def worker(self, account: str) -> tuple[int, bytes]:
        return self.bonds[account], self.keys[account]

    async def height(self) -> int:
        return self.block


def fixture():
    pin = ModelPin(model_id="test-model", weights_digest="11" * 32,
                   tokenizer_digest="22" * 32, runtime_digest="33" * 32,
                   spec_version="v1")
    envelope = TaskEnvelope(task_id="7", mode="lightweight", model_id=pin.model_id,
                            model_digest=pin.model_digest, spec_version=pin.spec_version,
                            input_commitment="44" * 32, data_ref="encrypted://input",
                            max_fee="10000", deadline=20, privacy_tier="tier0_relative")
    node0 = hashlib.sha256(bytes([0xAA]) * 32).hexdigest()
    node1 = hashlib.sha256(bytes([0xBB]) * 32).hexdigest()
    route = Route(group_id="group", api_url="https://worker.example/v1/execute",
                  worker_node_id=node0, stage_node_ids=(node0, node1),
                  stage_chain_workers=("prsmworker", "prsmstage"), measured_score=1.0)
    task = {"id": 7, "status": "accepted", "worker": "prsmworker", "mode": "lightweight",
            "model_id": "test-model", "spec_version": "v1", "input_commitment": encoded(0x44),
            "data_ref": "encrypted://input", "max_fee": 10000, "deadline": 20,
            "privacy_tier": "tier0_relative"}
    model = {"id": "test-model", "version": "v1", "mode": "lightweight",
             "weights_digest": encoded(0x11), "tokenizer_digest": encoded(0x22),
             "image_digest": encoded(0x33)}
    return envelope, route, task, model


def test_authorizer_binds_task_model_workers_and_height():
    envelope, route, task, model = fixture()
    chain = FakeChain(task, model)
    authorize = ChainTaskAuthorizer(chain)
    assert asyncio.run(authorize(envelope, route))

    for changed in ({"status": "refunded"}, {"worker": "prsmother"},
                    {"input_commitment": encoded(0x55)}, {"max_fee": 9999}):
        chain.record = {**task, **changed}
        assert not asyncio.run(authorize(envelope, route)), changed
    chain.record = task
    chain.block = 21
    assert not asyncio.run(authorize(envelope, route))
    chain.block = 10
    chain.bonds["prsmstage"] = 999_999
    assert not asyncio.run(authorize(envelope, route))
    chain.bonds["prsmstage"] = 1_000_000
    chain.registration = deepcopy(model)
    chain.registration["tokenizer_digest"] = encoded(0x66)
    assert not asyncio.run(authorize(envelope, route))
    chain.registration = model
    chain.keys["prsmstage"] = bytes([0xCC]) * 32
    assert not asyncio.run(authorize(envelope, route))
