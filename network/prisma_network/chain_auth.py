"""Fail-closed read-only bridge to the chain's compute Query gRPC service."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Protocol

import httpx

from .core import ModelPin, Route, SignedCapability, TaskEnvelope, Unavailable, verify


def _varint(number: int) -> bytes:
    if not 0 <= number < 2**64:
        raise ValueError("protobuf uint64 out of range")
    data = bytearray()
    while number >= 128:
        data.append((number & 127) | 128)
        number >>= 7
    data.append(number)
    return bytes(data)


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            if value >= 2**64:
                raise ValueError("protobuf uint64 overflow")
            return value, offset
    raise ValueError("protobuf varint too long")


def _length_field(number: int, value: bytes) -> bytes:
    return _varint(number << 3 | 2) + _varint(len(value)) + value


def _response_field(data: bytes, wanted: int, wire: int) -> bytes | int:
    position = 0
    result: bytes | int | None = None
    while position < len(data):
        tag, position = _read_varint(data, position)
        field, kind = tag >> 3, tag & 7
        if kind == 0:
            value, position = _read_varint(data, position)
        elif kind == 2:
            length, position = _read_varint(data, position)
            if position + length > len(data):
                raise ValueError("truncated protobuf field")
            value = data[position:position + length]
            position += length
        elif kind in (1, 5):
            size = 8 if kind == 1 else 4
            if position + size > len(data):
                raise ValueError("truncated protobuf field")
            value = data[position:position + size]
            position += size
        else:
            raise ValueError("unsupported protobuf wire type")
        if field == wanted:
            if kind != wire:
                raise ValueError("unexpected protobuf response field type")
            result = value
    if result is None:
        raise ValueError("missing protobuf response field")
    return result


class ChainQueries(Protocol):
    async def task(self, task_id: int) -> dict: ...
    async def model(self, model_id: str, spec_version: str) -> dict: ...
    async def worker(self, account: str) -> tuple[int, bytes]: ...
    async def height(self) -> int: ...


class GrpcChainQueries:
    """Uses the checked-in `prisma.compute.v1.Query` proto without codegen."""

    def __init__(self, grpc_address: str, rpc_url: str, *, insecure_dev: bool = False):
        try:
            import grpc
        except ImportError as exc:
            raise RuntimeError("install prisma-network[chain] to query the chain") from exc
        if not insecure_dev and not rpc_url.startswith("https://"):
            raise ValueError("chain RPC must use HTTPS outside local devnet")
        self.grpc = grpc
        self.address = grpc_address
        self.rpc_url = rpc_url.rstrip("/")
        self.insecure_dev = insecure_dev

    async def _call(self, method: str, request: bytes) -> bytes:
        if self.insecure_dev:
            channel = self.grpc.aio.insecure_channel(self.address)
        else:
            channel = self.grpc.aio.secure_channel(self.address, self.grpc.ssl_channel_credentials())
        async with channel:
            call = channel.unary_unary("/prisma.compute.v1.Query/" + method,
                                       request_serializer=lambda x: x, response_deserializer=lambda x: x)
            return await call(request, timeout=5)

    async def task(self, task_id: int) -> dict:
        raw = await self._call("Task", b"\x08" + _varint(task_id))
        return json.loads(_response_field(raw, 1, 2))

    async def model(self, model_id: str, spec_version: str) -> dict:
        request = _length_field(1, model_id.encode()) + _length_field(2, spec_version.encode())
        raw = await self._call("Model", request)
        return json.loads(_response_field(raw, 1, 2))

    async def worker(self, account: str) -> tuple[int, bytes]:
        raw = await self._call("Worker", _length_field(1, account.encode()))
        return int(_response_field(raw, 1, 0)), bytes(_response_field(raw, 2, 2))

    async def height(self) -> int:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            response = await client.get(self.rpc_url + "/status")
        response.raise_for_status()
        return int(response.json()["result"]["sync_info"]["latest_block_height"])


class ChainAnnouncementAuthorizer:
    """Admit a signed node only while its chain account owns a live bond and key."""

    MIN_BOND_UPRSM = 1_000_000

    def __init__(self, queries: ChainQueries):
        self.queries = queries

    async def __call__(self, signed: SignedCapability) -> bool:
        cap = signed.capability
        try:
            public_key = base64.b64decode(cap.public_key, validate=True)
        except ValueError as exc:
            raise ValueError("invalid announcement public key") from exc
        if (len(public_key) != 32 or hashlib.sha256(public_key).hexdigest() != cap.node_id
                or not verify(cap.public_key, "prisma:capability:v1", cap.model_dump(), signed.signature)):
            raise ValueError("invalid signed node announcement")
        try:
            bonded, chain_key = await self.queries.worker(cap.chain_worker)
        except Exception as exc:
            raise Unavailable("chain worker admission query failed") from exc
        return bonded >= self.MIN_BOND_UPRSM and chain_key == public_key


def _digest_hex(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("chain digest is not base64")
    raw = base64.b64decode(value, validate=True)
    if len(raw) != 32:
        raise ValueError("chain digest length is not 32")
    return raw.hex()


class ChainTaskAuthorizer:
    MIN_BOND_UPRSM = 1_000_000

    def __init__(self, queries: ChainQueries):
        self.queries = queries

    async def __call__(self, envelope: TaskEnvelope, route: Route) -> bool:
        try:
            task_id = int(envelope.task_id)
            if task_id <= 0 or str(task_id) != envelope.task_id:
                return False
            task = await self.queries.task(task_id)
            model = await self.queries.model(envelope.model_id, envelope.spec_version)
            height = await self.queries.height()
            pin = ModelPin(model_id=model["id"], weights_digest=_digest_hex(model["weights_digest"]),
                           tokenizer_digest=_digest_hex(model["tokenizer_digest"]),
                           runtime_digest=_digest_hex(model["image_digest"]),
                           spec_version=model["version"])
            if (task["id"] != task_id or task["status"] != "accepted"
                    or task["worker"] != route.stage_chain_workers[0]
                    or task["mode"] != envelope.mode or model["mode"] != envelope.mode
                    or task["model_id"] != envelope.model_id or pin.model_id != envelope.model_id
                    or task["spec_version"] != envelope.spec_version or pin.spec_version != envelope.spec_version
                    or pin.model_digest != envelope.model_digest
                    or _digest_hex(task["input_commitment"]) != envelope.input_commitment
                    or task["data_ref"] != envelope.data_ref
                    or int(task["max_fee"]) != int(envelope.max_fee)
                    or int(task["deadline"]) != envelope.deadline or height > envelope.deadline
                    or task["privacy_tier"] != envelope.privacy_tier):
                return False
            if len(route.stage_node_ids) != len(route.stage_chain_workers) or not route.stage_node_ids:
                return False
            for node_id, account in zip(route.stage_node_ids, route.stage_chain_workers):
                bonded, network_key = await self.queries.worker(account)
                if (bonded < self.MIN_BOND_UPRSM or len(network_key) != 32
                        or hashlib.sha256(network_key).hexdigest() != node_id):
                    return False
            return True
        except (KeyError, TypeError, ValueError) as exc:
            raise Unavailable("malformed chain task/model query response") from exc
        except Exception as exc:
            raise Unavailable("chain task authorization query failed") from exc
