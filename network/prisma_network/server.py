"""HTTP sidecars for a Prisma gateway and pipeline-stage workers.

The gateway creates delivery evidence only. It never reports on-chain settlement.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote
from typing import Awaitable, Callable, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import Field

from .core import (Capability, Conflict, ControlPlane, Identity, ModelPin, Route, SignedCapability,
                   StrictModel, TaskEnvelope, Unavailable, digest, verify)

PRIVACY_NOTICE = (
    "Tier 0 is relative privacy: ingress can see the full input, egress can see the "
    "full output, and intermediate activations may leak information. It is not "
    "cryptographic secret sharing or a strong confidentiality guarantee."
)
PROBE_BODY = b"P" * 65_536


class InferenceRequest(StrictModel):
    task: TaskEnvelope
    messages: list[dict] = Field(min_length=1, max_length=256)
    max_tokens: int = Field(ge=1, le=8192)
    temperature: float = Field(ge=0, le=2)

    def committed_input(self) -> str:
        return digest({"messages": self.messages, "max_tokens": self.max_tokens,
                       "temperature": self.temperature})


class ExecutionRequest(InferenceRequest):
    attempt_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    lease_epoch: int = Field(ge=1)
    gateway_node_id: str
    group_id: str


class SignedExecution(StrictModel):
    request: ExecutionRequest
    signature: str


class WorkerAttestation(StrictModel):
    task_id: str
    attempt_id: str
    worker_node_id: str
    gateway_node_id: str
    group_id: str
    lease_epoch: int
    model_id: str
    model_digest: str
    spec_version: str
    input_commitment: str
    output_commitment: str
    output_tokens: int = Field(ge=0)
    completed_at_ms: int
    status: Literal["delivered"] = "delivered"


class WorkerResponse(StrictModel):
    output: str
    attestation: WorkerAttestation
    signature: str


def _require_api_key(expected: str, authorization: str | None) -> None:
    if not expected or authorization != f"Bearer {expected}":
        raise HTTPException(401, "invalid API key")


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, Conflict):
        return HTTPException(409, str(exc))
    if isinstance(exc, Unavailable):
        return HTTPException(503, str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(400, str(exc))
    return HTTPException(502, "upstream compute service failed")


async def _default_worker_call(url: str, request: SignedExecution) -> WorkerResponse:
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(url, json=request.model_dump())
    response.raise_for_status()
    return WorkerResponse.model_validate(response.json())


async def _default_model_call(vllm_base_url: str, model_id: str, request: ExecutionRequest,
                              api_key: str | None) -> tuple[str, int]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(vllm_base_url.rstrip("/") + "/v1/chat/completions", headers=headers,
                                     json={"model": model_id, "messages": request.messages,
                                           "max_tokens": request.max_tokens,
                                           "temperature": request.temperature, "stream": False})
    response.raise_for_status()
    data = response.json()
    output = data["choices"][0]["message"]["content"]
    count = data["usage"]["completion_tokens"]
    if not isinstance(output, str) or not isinstance(count, int) or count < 0:
        raise ValueError("invalid vLLM response or missing token usage")
    return output, count


async def _default_lease_lookup(control_url: str, task_id: str) -> dict:
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(control_url.rstrip("/") + "/v1/leases/" + quote(task_id, safe=""))
    response.raise_for_status()
    return response.json()


def create_gateway_app(
    plane: ControlPlane,
    identity: Identity,
    api_key: str,
    *,
    task_authorizer: Callable[[TaskEnvelope, Route], Awaitable[bool]] | None = None,
    token_counter: Callable[[str], Awaitable[int]] | None = None,
    allow_unfunded_dev_tasks: bool = False,
    worker_call: Callable[[str, SignedExecution], Awaitable[WorkerResponse]] = _default_worker_call,
    gossip_peers: tuple[str, ...] = (),
) -> FastAPI:
    async def refresh() -> None:
        async with httpx.AsyncClient(timeout=5) as client:
            while True:
                for signed in plane.announcements():
                    c = signed.capability
                    started = time.monotonic()
                    try:
                        response = await client.get(c.probe_url)
                        response.raise_for_status()
                        if len(response.content) < 4096:
                            raise ValueError("short probe response")
                        elapsed = max(time.monotonic() - started, 0.001)
                        plane.probe(c.node_id, ok=True, latency_ms=elapsed * 1000,
                                    bandwidth_mbps=len(response.content) * 8 / elapsed / 1_000_000,
                                    queue_depth=int(response.headers.get("x-queue-depth", "0")),
                                    expected_sequence=c.sequence)
                    except (httpx.HTTPError, ValueError):
                        plane.probe(c.node_id, ok=False, latency_ms=0, bandwidth_mbps=0,
                                    expected_sequence=c.sequence)
                for peer in gossip_peers:
                    try:
                        response = await client.get(peer.rstrip("/") + "/v1/announcements")
                        response.raise_for_status()
                        for item in response.json():
                            try:
                                plane.announce(SignedCapability.model_validate(item))
                            except (ValueError, Conflict):
                                pass
                    except (httpx.HTTPError, ValueError):
                        pass
                await asyncio.sleep(5)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        background = asyncio.create_task(refresh())
        try:
            yield
        finally:
            background.cancel()
            try:
                await background
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="Prisma compute gateway", lifespan=lifespan)

    @app.get("/v1/privacy")
    def privacy() -> dict:
        return {"tier": "tier0_relative", "notice": PRIVACY_NOTICE}

    @app.post("/v1/announcements", status_code=202)
    def announce(body: SignedCapability) -> dict:
        try:
            plane.announce(body)
        except (ValueError, Conflict) as exc:
            raise _http_error(exc) from exc
        return {"accepted": True}

    @app.get("/v1/announcements")
    def announcements() -> list[dict]:
        return [item.model_dump() for item in plane.announcements()]

    @app.get("/v1/routes/{model_id}")
    def route(model_id: str, model_digest: str, spec_version: str) -> dict:
        try:
            return vars(plane.route(model_id, model_digest, spec_version))
        except Unavailable as exc:
            raise _http_error(exc) from exc

    @app.get("/v1/leases/{task_id}")
    def lease(task_id: str) -> dict:
        result = plane.lease(task_id)
        if not result:
            raise HTTPException(404, "unknown task lease")
        return result

    @app.get("/v1/receipts/{task_id}")
    def receipt(task_id: str, authorization: str | None = Header(default=None)) -> dict:
        _require_api_key(api_key, authorization)
        result = plane.receipt(task_id)
        if not result:
            raise HTTPException(404, "no delivery receipt")
        return result

    @app.post("/v1/inference")
    async def inference(body: InferenceRequest, authorization: str | None = Header(default=None)) -> dict:
        _require_api_key(api_key, authorization)
        if body.task.mode != "lightweight":
            raise HTTPException(400, "LLM inference requires lightweight mode")
        if body.committed_input() != body.task.input_commitment:
            raise HTTPException(400, "input commitment mismatch")
        if not allow_unfunded_dev_tasks:
            if task_authorizer is None:
                raise HTTPException(503, "chain task authorizer is not configured")
            if token_counter is None:
                raise HTTPException(503, "pinned tokenizer counter is not configured")
            if not plane.bound_worker_accounts:
                raise HTTPException(503, "node-to-chain-account bindings are not configured")
        previous = plane.receipt(body.task.task_id)
        if previous:
            for field in ("mode", "model_id", "model_digest", "spec_version", "input_commitment"):
                if previous.get(field) != getattr(body.task, field):
                    raise HTTPException(409, "retry envelope conflicts with existing delivery receipt")
            return {"status": "provisional_delivery", "already_completed": True,
                    "output": None, "receipt": previous, "privacy_notice": PRIVACY_NOTICE}
        try:
            route = plane.route(body.task.model_id, body.task.model_digest, body.task.spec_version)
            if not allow_unfunded_dev_tasks and not await task_authorizer(body.task, route):
                raise HTTPException(403, "chain escrow, model or worker does not match request")
            lease = plane.acquire(body.task.task_id, route.group_id, identity.node_id)
            attempt_id = str(uuid.uuid4())
            plane.start_attempt(body.task.task_id, route.group_id, identity.node_id,
                                lease["epoch"], attempt_id)
        except HTTPException:
            raise
        except (ValueError, Conflict, Unavailable) as exc:
            raise _http_error(exc) from exc
        request = ExecutionRequest(**body.model_dump(), attempt_id=attempt_id,
                                   lease_epoch=lease["epoch"], gateway_node_id=identity.node_id,
                                   group_id=route.group_id)
        signed = SignedExecution(request=request, signature=identity.sign("prisma:execution:v1", request.model_dump()))
        stop = asyncio.Event()

        async def renew() -> None:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=10)
                except asyncio.TimeoutError:
                    try:
                        updated = plane.acquire(body.task.task_id, route.group_id, identity.node_id)
                        if updated["epoch"] != lease["epoch"]:
                            stop.set()
                    except (Conflict, ValueError):
                        stop.set()

        heartbeat = asyncio.create_task(renew())
        try:
            worker = await worker_call(route.api_url, signed)
            att = worker.attestation
            expected = {"task_id": body.task.task_id, "attempt_id": attempt_id,
                        "worker_node_id": route.worker_node_id, "gateway_node_id": identity.node_id,
                        "group_id": route.group_id, "lease_epoch": lease["epoch"],
                        "model_id": body.task.model_id, "model_digest": body.task.model_digest,
                        "spec_version": body.task.spec_version,
                        "input_commitment": body.task.input_commitment,
                        "output_commitment": digest(worker.output)}
            if any(getattr(att, key) != value for key, value in expected.items()):
                raise ValueError("worker attestation does not match delivery")
            if att.output_tokens > body.max_tokens:
                raise ValueError("worker token count exceeds requested maximum")
            worker_key = plane.trusted_keys[route.worker_node_id]
            if not verify(worker_key, "prisma:worker-receipt:v1", att.model_dump(), worker.signature):
                raise ValueError("invalid worker receipt signature")
            if token_counter is not None and await token_counter(worker.output) != att.output_tokens:
                raise ValueError("worker token count disagrees with pinned tokenizer")
            if not allow_unfunded_dev_tasks and not await task_authorizer(body.task, route):
                raise Conflict("chain task changed during inference")
            if not plane.check_fence(body.task.task_id, route.group_id, identity.node_id, lease["epoch"]):
                raise Conflict("lease expired during inference")
            payload = {"task_id": body.task.task_id, "attempt_id": attempt_id,
                       "mode": body.task.mode, "model_id": body.task.model_id,
                       "model_digest": body.task.model_digest, "spec_version": body.task.spec_version,
                       "input_commitment": body.task.input_commitment,
                       "output_commitment": att.output_commitment, "output_tokens": att.output_tokens,
                       "gateway_node_id": identity.node_id, "worker_node_id": route.worker_node_id,
                       "group_id": route.group_id, "lease_epoch": lease["epoch"],
                       "completed_at_ms": att.completed_at_ms, "status": "delivered",
                       "worker_attestation": att.model_dump(), "worker_signature": worker.signature}
            receipt = {**payload, "gateway_signature": identity.sign("prisma:gateway-receipt:v1", payload)}
            plane.commit_receipt(body.task.task_id, attempt_id, route.group_id, identity.node_id,
                                 lease["epoch"], receipt)
            return {"status": "provisional_delivery", "already_completed": False,
                    "output": worker.output, "receipt": receipt, "privacy_notice": PRIVACY_NOTICE}
        except Exception as exc:
            plane.finish_attempt(attempt_id, False)
            raise _http_error(exc) from exc
        finally:
            stop.set()
            await heartbeat

    return app


def create_worker_app(
    identity: Identity,
    group_id: str,
    pin: ModelPin,
    gateway_keys: dict[str, str],
    *,
    model_call: Callable[[ExecutionRequest], Awaitable[tuple[str, int]]],
    lease_lookup: Callable[[str], Awaitable[dict]],
    readiness: Callable[[], Awaitable[bool]] | None = None,
    lifespan=None,
) -> FastAPI:
    app = FastAPI(title="Prisma compute worker", lifespan=lifespan)
    active = 0

    @app.get("/v1/privacy")
    def privacy() -> dict:
        return {"tier": "tier0_relative", "notice": PRIVACY_NOTICE}

    @app.get("/v1/probe")
    async def probe() -> Response:
        if readiness is not None and not await readiness():
            raise HTTPException(503, "model backend is not ready")
        return Response(PROBE_BODY, media_type="application/octet-stream", headers={"X-Queue-Depth": str(active)})

    @app.post("/v1/execute")
    async def execute(body: SignedExecution) -> WorkerResponse:
        nonlocal active
        request = body.request
        pub = gateway_keys.get(request.gateway_node_id)
        if not pub or not verify(pub, "prisma:execution:v1", request.model_dump(), body.signature):
            raise HTTPException(401, "invalid gateway signature")
        if (request.group_id != group_id or request.task.model_id != pin.model_id
                or request.task.model_digest != pin.model_digest
                or request.task.spec_version != pin.spec_version
                or request.task.mode != "lightweight"
                or request.committed_input() != request.task.input_commitment):
            raise HTTPException(400, "task does not match pinned model/input")

        async def fenced() -> bool:
            lease = await lease_lookup(request.task.task_id)
            return (lease.get("group_id") == group_id and lease.get("owner_id") == request.gateway_node_id
                    and lease.get("epoch") == request.lease_epoch
                    and lease.get("expires_at_ms", 0) > int(time.time() * 1000))

        try:
            if not await fenced():
                raise Conflict("stale gateway fence")
            active += 1
            try:
                output, tokens = await model_call(request)
            finally:
                active -= 1
            if not isinstance(tokens, int) or not 0 <= tokens <= request.max_tokens:
                raise ValueError("model token count exceeds requested maximum")
            if not await fenced():
                raise Conflict("lease expired before worker attestation")
            att = WorkerAttestation(task_id=request.task.task_id, attempt_id=request.attempt_id,
                                    worker_node_id=identity.node_id,
                                    gateway_node_id=request.gateway_node_id, group_id=group_id,
                                    lease_epoch=request.lease_epoch, model_id=pin.model_id,
                                    model_digest=pin.model_digest, spec_version=pin.spec_version,
                                    input_commitment=request.task.input_commitment,
                                    output_commitment=digest(output), output_tokens=tokens,
                                    completed_at_ms=int(time.time() * 1000))
            return WorkerResponse(output=output, attestation=att,
                                  signature=identity.sign("prisma:worker-receipt:v1", att.model_dump()))
        except Exception as exc:
            raise _http_error(exc) from exc

    return app


def _keys_from_env() -> dict[str, str]:
    with open(os.environ["PRISMA_TRUSTED_KEYS_FILE"], encoding="utf-8") as handle:
        return json.load(handle)


def _bindings_from_env() -> dict[str, str]:
    path = os.environ.get("PRISMA_NODE_ACCOUNT_BINDINGS_FILE")
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _verified_pin_from_env() -> ModelPin:
    pin = ModelPin(model_id=os.environ["PRISMA_MODEL_ID"],
                   weights_digest=os.environ["PRISMA_WEIGHTS_DIGEST"],
                   tokenizer_digest=os.environ["PRISMA_TOKENIZER_DIGEST"],
                   runtime_digest=os.environ["PRISMA_RUNTIME_DIGEST"],
                   spec_version=os.environ["PRISMA_SPEC_VERSION"])
    if pin.model_digest != os.environ["PRISMA_MODEL_DIGEST"]:
        raise ValueError("configured model digest is not the registry bundle digest")
    if os.environ.get("PRISMA_DEV_SKIP_FILE_HASH") == "1":
        return pin
    root = Path(os.environ["PRISMA_MODEL_DIR"]).resolve()
    with open(os.environ["PRISMA_MODEL_FILES_FILE"], encoding="utf-8") as handle:
        files = json.load(handle)
    for kind, expected_digest in (("weights", pin.weights_digest), ("tokenizer", pin.tokenizer_digest)):
        entries = files[kind]
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"{kind} file manifest must be a nonempty path-to-SHA256 map")
        for relative, expected_file_digest in entries.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or len(expected_file_digest) != 64:
                raise ValueError(f"invalid {kind} artifact path or digest")
            hasher = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != expected_file_digest:
                raise ValueError(f"{kind} artifact hash mismatch: {relative}")
        if digest(entries) != expected_digest:
            raise ValueError(f"{kind} artifact set digest mismatch")
    return pin


def gateway_app_from_env() -> FastAPI:
    """Run with `uvicorn prisma_network.server:gateway_app_from_env --factory`."""
    identity = Identity.from_seed_b64(os.environ["PRISMA_NODE_SEED_B64"])
    plane = ControlPlane(os.environ.get("PRISMA_DB", "prisma-network.sqlite3"), _keys_from_env(),
                         set(os.environ["PRISMA_ALLOWED_NODE_HOSTS"].split(",")),
                         bound_worker_accounts=_bindings_from_env(),
                         allow_http=os.environ.get("PRISMA_DEV_HTTP") == "1")
    authorizer = None
    if os.environ.get("PRISMA_CHAIN_GRPC_ADDR") and os.environ.get("PRISMA_CHAIN_RPC_URL"):
        from .chain_auth import ChainTaskAuthorizer, GrpcChainQueries
        authorizer = ChainTaskAuthorizer(GrpcChainQueries(
            os.environ["PRISMA_CHAIN_GRPC_ADDR"], os.environ["PRISMA_CHAIN_RPC_URL"],
            insecure_dev=os.environ.get("PRISMA_DEV_HTTP") == "1"))
    return create_gateway_app(plane, identity, os.environ["PRISMA_CLIENT_API_KEY"],
                              task_authorizer=authorizer,
                              allow_unfunded_dev_tasks=os.environ.get("PRISMA_DEV_UNFUNDED") == "1",
                              gossip_peers=tuple(filter(None, os.environ.get("PRISMA_GOSSIP_PEERS", "").split(","))))


def worker_app_from_env() -> FastAPI:
    """Stage-0 sidecar for an already-running, pinned multi-node vLLM cluster."""
    identity = Identity.from_seed_b64(os.environ["PRISMA_NODE_SEED_B64"])
    if os.environ["PRISMA_STAGE_INDEX"] != "0":
        raise ValueError("worker_app_from_env requires stage index 0")
    pin = _verified_pin_from_env()
    trusted = _keys_from_env()
    vllm_url = os.environ["PRISMA_VLLM_URL"]
    control_url = os.environ["PRISMA_CONTROL_URL"]

    async def call(request: ExecutionRequest) -> tuple[str, int]:
        return await _default_model_call(vllm_url, pin.model_id, request, os.environ.get("PRISMA_VLLM_API_KEY"))

    async def lookup(task_id: str) -> dict:
        return await _default_lease_lookup(control_url, task_id)

    async def ready() -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(vllm_url.rstrip("/") + "/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    app = create_worker_app(identity, os.environ["PRISMA_GROUP_ID"], pin, trusted,
                            model_call=call, lease_lookup=lookup, readiness=ready,
                            lifespan=_announcer_lifespan(identity, pin))
    return app


def stage_app_from_env() -> FastAPI:
    """Probe/announcement sidecar for a non-head pipeline stage."""
    identity = Identity.from_seed_b64(os.environ["PRISMA_NODE_SEED_B64"])
    if int(os.environ["PRISMA_STAGE_INDEX"]) <= 0:
        raise ValueError("stage_app_from_env requires a non-head stage")
    pin = _verified_pin_from_env()
    health_url = os.environ["PRISMA_STAGE_HEALTH_URL"]
    app = FastAPI(title="Prisma pipeline stage", lifespan=_announcer_lifespan(identity, pin))

    @app.get("/v1/probe")
    async def probe() -> Response:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(health_url)
            if response.status_code != 200:
                raise HTTPException(503, "pipeline stage is not ready")
        except httpx.HTTPError as exc:
            raise HTTPException(503, "pipeline stage is not reachable") from exc
        return Response(PROBE_BODY, media_type="application/octet-stream", headers={"X-Queue-Depth": "0"})

    return app


def _announcer_lifespan(identity: Identity, pin: ModelPin):
    """Renew a signed 45-second capability every 20 seconds."""
    async def announce_forever() -> None:
        control = os.environ["PRISMA_CONTROL_URL"].rstrip("/")
        public = os.environ["PRISMA_PUBLIC_URL"].rstrip("/")
        stage = int(os.environ["PRISMA_STAGE_INDEX"])
        sequence = 0
        async with httpx.AsyncClient(timeout=5) as client:
            while True:
                now = int(time.time() * 1000)
                sequence = max(sequence + 1, now)
                cap = Capability(node_id=identity.node_id, public_key=identity.public_key_b64,
                                 sequence=sequence, issued_at_ms=now, expires_at_ms=now + 45_000,
                                 group_id=os.environ["PRISMA_GROUP_ID"],
                                 chain_worker=os.environ.get("PRISMA_CHAIN_WORKER", "dev-worker"),
                                 stage_index=stage,
                                 stage_count=int(os.environ["PRISMA_STAGE_COUNT"]),
                                 model_id=pin.model_id, model_digest=pin.model_digest,
                                 weights_digest=pin.weights_digest,
                                 tokenizer_digest=pin.tokenizer_digest,
                                 runtime_digest=pin.runtime_digest, spec_version=pin.spec_version,
                                 probe_url=public + "/v1/probe",
                                 api_url=public + "/v1/execute" if stage == 0 else None,
                                 gpu_count=int(os.environ["PRISMA_GPU_COUNT"]),
                                 vram_mb=int(os.environ["PRISMA_VRAM_MB"]))
                signed = SignedCapability(capability=cap,
                                          signature=identity.sign("prisma:capability:v1", cap.model_dump()))
                try:
                    response = await client.post(control + "/v1/announcements", json=signed.model_dump())
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    logging.getLogger(__name__).warning("capability announcement failed: %s", exc)
                await asyncio.sleep(20)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(announce_forever())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    return lifespan
