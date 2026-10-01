"""Exercise the Python SDK against the running unfunded local Compose gateway."""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "network"))
from prisma_network import PrismaClient, TaskEnvelope


async def main() -> None:
    values = dict(line.split("=", 1) for line in (Path(__file__).parent / ".env").read_text().splitlines()
                  if line and not line.startswith("#"))
    messages = [{"role": "user", "content": "local SDK smoke"}]
    task_id = "dev-sdk-" + uuid.uuid4().hex
    task = TaskEnvelope(task_id=task_id, mode="lightweight", model_id=values["PRISMA_MODEL_ID"],
                        model_digest=values["PRISMA_MODEL_DIGEST"],
                        spec_version=values["PRISMA_SPEC_VERSION"],
                        input_commitment=PrismaClient.input_commitment(messages, 8, 0.0),
                        data_ref="dev://local-sdk", max_fee="1000", deadline=1_000_000,
                        privacy_tier="tier0_relative")
    async with PrismaClient("http://127.0.0.1:8080", values["PRISMA_CLIENT_API_KEY"],
                            allow_http_dev=True) as client:
        privacy = await client.privacy()
        first = await client.infer(task, messages, max_tokens=8)
        status = await client.task_status(task_id)
        replay = await client.infer(task, messages, max_tokens=8)
    if (privacy["tier"] != "tier0_relative" or first["output"] != "MOCK"
            or first["receipt"]["task_id"] != task_id or status["chain_status"] != "unfunded_dev"
            or status["chain_submission"] != "unconfigured"
            or not replay["already_completed"] or replay["output"] is not None):
        raise RuntimeError("Python SDK did not preserve local gateway delivery semantics")
    print("PASS: Python SDK privacy, committed mock inference, unfunded status and idempotent replay")


if __name__ == "__main__":
    asyncio.run(main())
