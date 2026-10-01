"""Local devnet fixture. Never use for funded tasks or claim Qwen inference."""

import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Prisma deterministic mock model (dev only)")


class MockRequest(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int = Field(ge=1)


@app.get("/health")
def health() -> dict:
    return {"status": "mock_ready", "funded_tasks": False, "distributed_inference": False}


@app.post("/v1/chat/completions")
def completion(request: MockRequest) -> dict:
    expected = os.environ.get("PRISMA_MODEL_ID", "demo-model")
    if request.model != expected:
        raise HTTPException(400, "mock model ID mismatch")
    # A fixed one-token answer makes local receipts deterministic; it is not inference.
    return {"id": "prisma-dev-mock", "object": "chat.completion", "created": int(time.time()),
            "model": expected, "choices": [{"index": 0, "message": {"role": "assistant", "content": "MOCK"},
                                            "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1}}
