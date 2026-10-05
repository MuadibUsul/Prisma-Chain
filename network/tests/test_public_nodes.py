import asyncio
import socket
import time

import httpx
import pytest

from prisma_network.core import ControlPlane, Identity
from prisma_network.node_http import PublicNodeHTTP
from prisma_network.server import create_gateway_app

from test_network import capability


def test_public_admission_rejects_private_literal_and_dev_http(tmp_path):
    worker = Identity.generate()
    now = int(time.time() * 1000)
    with pytest.raises(ValueError, match="HTTPS"):
        ControlPlane(":memory:", {}, set(), allow_http=True, public_node_admission=True)
    plane = ControlPlane(str(tmp_path / "public.db"), {worker.node_id: worker.public_key_b64},
                         set(), public_node_admission=True, clock_ms=lambda: now)
    signed = capability(worker, 0, 1, now)
    signed.capability.probe_url = "https://127.0.0.1/v1/probe"
    signed.capability.api_url = "https://127.0.0.1/v1/execute"
    signed.signature = worker.sign("prisma:capability:v1", signed.capability.model_dump())
    with pytest.raises(ValueError, match="not public"):
        plane.announce(signed)
    signed.capability.probe_url = "https://node.example/v1/probe"
    signed.capability.api_url = "https://node.example/v1/execute"
    signed.signature = worker.sign("prisma:capability:v1", signed.capability.model_dump())
    plane.announce(signed)
    assert plane.announcements()[0].capability.node_id == worker.node_id
    with pytest.raises(ValueError, match="protected transport"):
        create_gateway_app(plane, Identity.generate(), "secret")
    with pytest.raises(ValueError, match="funded tasks"):
        create_gateway_app(plane, Identity.generate(), "secret",
                           public_node_http=PublicNodeHTTP(), allow_unfunded_dev_tasks=True)


def test_public_request_pins_checked_dns_and_preserves_tls_name(monkeypatch):
    observed = []

    def resolve(_host, port, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port))]

    def respond(request):
        observed.append(request)
        return httpx.Response(200, content=b"ready")

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    client = PublicNodeHTTP(transport=httpx.MockTransport(respond))
    response = asyncio.run(client.request("GET", "https://node.example:8443/v1/probe"))
    assert response.content == b"ready"
    assert str(observed[0].url) == "https://8.8.8.8:8443/v1/probe"
    assert observed[0].headers["host"] == "node.example:8443"
    assert observed[0].extensions["sni_hostname"] == "node.example"


@pytest.mark.parametrize("answers", [
    ["127.0.0.1"], ["169.254.169.254"], ["8.8.8.8", "10.0.0.1"], []
])
def test_public_request_never_connects_to_private_dns_answer(monkeypatch, answers):
    def resolve(_host, port, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in answers]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    called = []
    client = PublicNodeHTTP(transport=httpx.MockTransport(
        lambda request: (called.append(request), httpx.Response(200))[1]))
    with pytest.raises(ValueError, match="non-public"):
        asyncio.run(client.request("POST", "https://node.example/v1/execute", json={"x": 1}))
    assert called == []


def test_public_request_bounds_response(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda _host, port, **_kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port))])
    client = PublicNodeHTTP(transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, content=b"A" * 64)))
    with pytest.raises(ValueError, match="size limit"):
        asyncio.run(client.request("GET", "https://node.example/v1/probe", max_response_bytes=32))
