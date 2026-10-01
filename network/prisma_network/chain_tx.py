"""Worker-owned devnet signer for canonical lightweight delivery receipts."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path

import httpx

from .chain_auth import ChainQueries
from .core import Conflict, Unavailable, canonical


class DevnetCliReceiptSubmitter:
    """Sign with an isolated test keyring; never share it with a gateway."""

    def __init__(self, queries: ChainQueries, *, chain_id: str, rpc_url: str,
                 signer_name: str, signer_home: str, network_public_key: bytes,
                 binary: str = "prismad"):
        if not chain_id or not signer_name or len(network_public_key) != 32:
            raise ValueError("chain, signer and network key are required")
        if not rpc_url.startswith("http://"):
            raise ValueError("development signer requires an HTTP devnet RPC")
        self.queries = queries
        self.chain_id = chain_id
        self.rpc_url = rpc_url.rstrip("/")
        self.signer_name = signer_name
        self.home = Path(signer_home).resolve()
        self.public_key = network_public_key
        self.binary = binary
        self.lock = asyncio.Lock()

    async def _command(self, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            self.binary, *args, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            raise Unavailable("chain signer command timed out") from None
        if process.returncode:
            raise Unavailable("chain signer command failed: " +
                              stderr.decode(errors="replace")[-500:])
        return stdout.decode().strip()

    async def _chain_status(self) -> int:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            response = await client.get(self.rpc_url + "/status")
        response.raise_for_status()
        status = response.json()["result"]
        if status["node_info"]["network"] != self.chain_id:
            raise Conflict("chain signer RPC is on a different chain")
        return int(status["sync_info"]["latest_block_height"])

    async def _included(self, txhash: str) -> None:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            for _ in range(30):
                response = await client.get(self.rpc_url + "/tx", params={"hash": "0x" + txhash})
                data = response.json()
                if data.get("result") is not None:
                    if int(data["result"]["tx_result"]["code"]) != 0:
                        raise Conflict("chain rejected the signed receipt transaction")
                    return
                await asyncio.sleep(1)
        raise Unavailable("receipt transaction inclusion was not observed")

    async def __call__(self, receipt: dict) -> None:
        task_text = receipt.get("task_id")
        if (not isinstance(task_text, str) or not task_text.isascii() or
                not task_text.isdecimal() or not 0 < int(task_text) < 2**64 or
                str(int(task_text)) != task_text):
            raise ValueError("receipt task ID must be a canonical uint64")
        tokens = receipt.get("output_tokens")
        output = receipt.get("output_commitment")
        if (type(tokens) is not int or not 0 < tokens < 2**64 or
                not isinstance(output, str) or len(output) != 64):
            raise ValueError("receipt has invalid billable output")
        try:
            bytes.fromhex(output)
        except ValueError as exc:
            raise ValueError("receipt output commitment is not hexadecimal") from exc
        evidence = canonical(receipt)
        if len(evidence) > 8192:
            raise ValueError("signed receipt exceeds chain limit")
        receipt_hash = hashlib.sha256(evidence).digest()

        async with self.lock:
            height = await self._chain_status()
            task = await self.queries.task(int(task_text))
            if task["status"] in {"pending", "settled"}:
                if base64.b64decode(task["receipt_digest"], validate=True) != receipt_hash:
                    raise Conflict("chain has a different receipt for this task")
                return
            if (task["status"] != "accepted" or task["mode"] != "lightweight"
                    or height > int(task["deadline"])):
                raise Conflict("chain task is not available for a lightweight result")
            address = await self._command("keys", "show", self.signer_name, "-a",
                                          "--home", str(self.home), "--keyring-dir", str(self.home),
                                          "--keyring-backend", "test")
            if task["worker"] != address:
                raise Conflict("signer account did not accept this task")
            bonded, network_key = await self.queries.worker(address)
            if bonded < 1_000_000 or network_key != self.public_key:
                raise Conflict("signer account is not bonded to this network key")

            self.home.mkdir(parents=True, exist_ok=True)
            descriptor, filename = tempfile.mkstemp(prefix="receipt-", suffix=".json", dir=self.home)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(evidence)
                broadcast = await self._command(
                    "tx", "compute", "submit-result", "--task-id", task_text,
                    "--output-digest", output, "--output-tokens", str(tokens),
                    "--receipt-digest", receipt_hash.hex(), "--receipt-json", filename,
                    "--from", self.signer_name, "--chain-id", self.chain_id,
                    "--node", self.rpc_url, "--home", str(self.home),
                    "--keyring-dir", str(self.home), "--keyring-backend", "test",
                    "--fees", "0uprsm", "--gas", "2000000", "--yes", "--output", "json")
                tx = json.loads(broadcast)
                if int(tx.get("code", 1)) != 0 or not tx.get("txhash"):
                    raise Conflict("chain rejected the receipt at broadcast")
                await self._included(tx["txhash"])
            finally:
                Path(filename).unlink(missing_ok=True)
            observed = await self.queries.task(int(task_text))
            if (observed["status"] not in {"pending", "settled"} or
                    base64.b64decode(observed["receipt_digest"], validate=True) != receipt_hash):
                raise Unavailable("included receipt does not match chain task state")
