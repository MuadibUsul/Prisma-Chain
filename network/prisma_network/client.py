"""Small client for the funded Prisma gateway API."""

from __future__ import annotations

import asyncio
import re
import time
from urllib.parse import urlsplit

import httpx

from .core import TaskEnvelope, digest
from .server import InferenceRequest


class PrismaAPIError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(f"Prisma API {status_code}: {detail}")


class PrismaClient:
    """Call one gateway without storing prompts, answers or API keys on disk."""

    def __init__(self, base_url: str, api_key: str, *, allow_http_dev: bool = False,
                 timeout: float = 180, transport: httpx.AsyncBaseTransport | None = None):
        url = urlsplit(base_url)
        if (not api_key or not url.hostname or url.username or url.password or url.query
                or url.fragment or url.path not in ("", "/")):
            raise ValueError("gateway URL or API key is invalid")
        if url.scheme != "https" and not (allow_http_dev and url.scheme == "http"
                                          and url.hostname in {"localhost", "127.0.0.1", "::1"}):
            raise ValueError("gateway must use HTTPS; development HTTP is loopback-only")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.http = httpx.AsyncClient(timeout=timeout, transport=transport, trust_env=False)

    async def __aenter__(self) -> "PrismaClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _request(self, method: str, path: str, *, body: dict | None = None) -> dict:
        response = await self.http.request(method, self.base_url + path,
                                           headers=self.headers, json=body)
        if response.is_error:
            try:
                detail = response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            raise PrismaAPIError(response.status_code,
                                 detail if isinstance(detail, str) else "request failed")
        try:
            result = response.json()
        except ValueError as exc:
            raise PrismaAPIError(502, "gateway returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise PrismaAPIError(502, "gateway returned an invalid response")
        return result

    @staticmethod
    def input_commitment(messages: list[dict], max_tokens: int, temperature: float) -> str:
        return digest({"messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature})

    async def infer(self, task: TaskEnvelope, messages: list[dict], *, max_tokens: int,
                    temperature: float = 0.0) -> dict:
        request = InferenceRequest(task=task, messages=messages, max_tokens=max_tokens,
                                   temperature=temperature)
        if request.committed_input() != task.input_commitment:
            raise ValueError("messages and generation parameters do not match the chain input commitment")
        return await self._request("POST", "/v1/inference", body=request.model_dump())

    async def task_status(self, task_id: str) -> dict:
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", task_id):
            raise ValueError("task ID contains unsupported characters")
        return await self._request("GET", f"/v1/tasks/{task_id}")

    async def privacy(self) -> dict:
        return await self._request("GET", "/v1/privacy")

    async def wait_for_settlement(self, task_id: str, *, timeout: float = 180,
                                  poll_seconds: float = 2) -> dict:
        if timeout <= 0 or poll_seconds <= 0:
            raise ValueError("poll timeout and interval must be positive")
        deadline = time.monotonic() + timeout
        while True:
            try:
                status = await self.task_status(task_id)
            except PrismaAPIError as exc:
                if exc.status_code != 503:
                    raise
                status = {}
            if status.get("settlement_final") is True:
                return status
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"task {task_id} has not settled")
            await asyncio.sleep(min(poll_seconds, remaining))
