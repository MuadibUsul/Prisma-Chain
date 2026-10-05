"""HTTPS transport for bonded public nodes with DNS answers pinned per request."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

import httpx


class PublicNodeHTTP:
    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None):
        self.transport = transport

    async def request(self, method: str, url: str, *, json: object = None,
                      timeout: float = 5, max_response_bytes: int = 1_048_576) -> httpx.Response:
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.fragment or parsed.query):
            raise ValueError("public node request requires an HTTPS URL without credentials")
        host = parsed.hostname
        port = parsed.port or 443
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            addresses = await asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
            ips = [ipaddress.ip_address(entry[4][0]) for entry in addresses]
        else:
            ips = [literal]
        if not ips or any(not ip.is_global for ip in ips):
            raise ValueError("public node resolves to a non-public address")
        # Connect to the checked address. Keep the original host for the HTTP
        # authority and TLS certificate/SNI. A new client prevents reuse of a
        # connection authenticated for another hostname at the same IP.
        target = str(httpx.URL(url).copy_with(host=str(ips[0])))
        named_host = f"[{host}]" if ":" in host else host
        authority = named_host if port == 443 else f"{named_host}:{port}"
        async with httpx.AsyncClient(timeout=timeout, trust_env=False,
                                     transport=self.transport, follow_redirects=False) as client:
            async with client.stream(method, target, json=json, headers={"Host": authority},
                                     extensions={"sni_hostname": host}) as response:
                chunks = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > max_response_bytes:
                        raise ValueError("public node response exceeds size limit")
                    chunks.append(chunk)
                return httpx.Response(response.status_code, headers=response.headers,
                                      content=b"".join(chunks), request=response.request)
