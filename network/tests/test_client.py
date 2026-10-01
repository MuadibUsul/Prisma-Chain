import asyncio
import json

import httpx
import pytest

from prisma_network import PrismaAPIError, PrismaClient, TaskEnvelope


def test_client_binds_input_commitment_and_never_sends_key_over_remote_http():
    with pytest.raises(ValueError, match="HTTPS"):
        PrismaClient("http://worker.example", "secret", allow_http_dev=True)
    messages = [{"role": "user", "content": "hello"}]
    commitment = PrismaClient.input_commitment(messages, 8, 0.0)
    task = TaskEnvelope(task_id="7", mode="lightweight", model_id="demo-model",
                        model_digest="a" * 64, spec_version="light-v1",
                        input_commitment=commitment, data_ref="encrypted:example",
                        max_fee="10000", deadline=100, privacy_tier="tier0_relative")
    calls = []

    def handle(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer secret"
        if request.url.path == "/v1/inference":
            body = json.loads(request.content)
            assert body["task"]["input_commitment"] == commitment
            assert body["messages"] == messages
            return httpx.Response(200, json={"status": "provisional_delivery", "output": "answer"})
        if request.url.path == "/v1/tasks/7":
            if sum(call.url.path == "/v1/tasks/7" for call in calls) == 1:
                return httpx.Response(503, json={"detail": "chain query unavailable"})
            return httpx.Response(200, json={"task_id": "7", "settlement_final": True,
                                             "chain_status": "settled"})
        return httpx.Response(404, json={"detail": "not found"})

    async def exercise():
        async with PrismaClient("http://127.0.0.1:8080", "secret", allow_http_dev=True,
                                transport=httpx.MockTransport(handle)) as client:
            assert (await client.infer(task, messages, max_tokens=8))["output"] == "answer"
            assert (await client.wait_for_settlement("7", poll_seconds=0.001))["chain_status"] == "settled"
            with pytest.raises(ValueError, match="input commitment"):
                await client.infer(task, [{"role": "user", "content": "changed"}], max_tokens=8)
            with pytest.raises(PrismaAPIError) as exc:
                await client.privacy()
            assert exc.value.status_code == 404
    asyncio.run(exercise())
    assert len(calls) == 4


def test_client_wait_times_out_without_final_chain_state():
    def handle(_request):
        return httpx.Response(200, json={"task_id": "dev-task", "settlement_final": False,
                                         "chain_status": "unfunded_dev"})

    async def exercise():
        async with PrismaClient("https://gateway.example", "secret",
                                transport=httpx.MockTransport(handle)) as client:
            assert (await client.task_status("dev-task"))["chain_status"] == "unfunded_dev"
            with pytest.raises(TimeoutError, match="has not settled"):
                await client.wait_for_settlement("dev-task", timeout=0.01, poll_seconds=0.005)
    asyncio.run(exercise())
