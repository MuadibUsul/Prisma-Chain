"""B2-08: upload, attestation verification and quorum — including the
provider-down drills (1/3 and 2/3 offline). The fake providers sign with the
frozen encoder (compute/canonical/python/canonical_ref.py), so verification is
checked against the authoritative preimage, not a re-implementation."""

from __future__ import annotations

import base64
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from prisma_worker import da

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))
import canonical_ref as R  # type: ignore  # noqa: E402

TASK_ID = 7
ASSIGNMENT_HEX = "cd" * 32
OUTPUT_ROOT_HEX = "ef" * 32
M, N, K = 4, 4, 4
AVAILABLE_UNTIL = 5000
ATTESTED_HEIGHT = 1000


def _sign_attestation(seed: bytes, *, output_root_hex=OUTPUT_ROOT_HEX,
                      protocol_version=da.DA_PROTOCOL_VERSION) -> tuple[dict, bytes]:
    key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
    pubkey = key.public_key().public_bytes_raw()
    canonical = {
        "protocol_version": protocol_version,
        "task_id": TASK_ID.to_bytes(8, "big"),
        "assignment_id": bytes.fromhex(ASSIGNMENT_HEX),
        "output_root": bytes.fromhex(output_root_hex),
        "provider_account": b"prsm1provider",
        "provider_pub_key": pubkey,
        "output_bytes": M * N * 4,
        "available_until_height": AVAILABLE_UNTIL,
        "attested_height": ATTESTED_HEIGHT,
        "signature": b"",
    }
    preimage = da.DA_SIGN_DOMAIN + R.encode_canonical(canonical)
    signature = key.sign(preimage)
    wire = {
        "ProtocolVersion": canonical["protocol_version"],
        "TaskID": base64.b64encode(canonical["task_id"]).decode(),
        "AssignmentID": base64.b64encode(canonical["assignment_id"]).decode(),
        "OutputRoot": base64.b64encode(canonical["output_root"]).decode(),
        "ProviderAccount": base64.b64encode(canonical["provider_account"]).decode(),
        "ProviderPubKey": base64.b64encode(pubkey).decode(),
        "OutputBytes": canonical["output_bytes"],
        "AvailableUntilHeight": canonical["available_until_height"],
        "AttestedHeight": canonical["attested_height"],
        "Signature": base64.b64encode(signature).decode(),
    }
    return wire, pubkey


class FakeProviderServer:
    """Configurable DA provider: healthy, transiently failing, store-only, bad-signing."""

    def __init__(self, name: str, seed: bytes, *, fail_store: bool = False,
                 fail_attest: bool = False, fail_attestures: int = 0,
                 bad_signature: bool = False, wrong_root: bool = False,
                 protocol_version: str = da.DA_PROTOCOL_VERSION,
                 advertise_key: bool = True):
        self.name, self.seed = name, seed
        self.fail_store, self.fail_attest = fail_store, fail_attest
        self.fail_attestures_remaining = fail_attestures
        self.bad_signature, self.wrong_root = bad_signature, wrong_root
        self.protocol_version = protocol_version
        self.advertise_key = advertise_key
        self.store_calls = 0
        self.attest_calls = 0
        self.server = None
        self.pubkey = ed25519.Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()

    # --- HTTP ---

    def start(self) -> str:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _json(self, code, payload):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                if self.path == "/health":
                    info = {"provider": outer.name, "tasks": [TASK_ID]}
                    if outer.advertise_key:
                        info["pubkey"] = base64.b64encode(outer.pubkey).decode()
                    self._json(200, info)
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/store":
                    outer.store_calls += 1
                    if outer.fail_store:
                        self._json(400, {"error": "provider is not accepting data"})
                        return
                    self._json(200, {"ok": True, "task_id": body.get("task_id")})
                elif self.path == "/attest":
                    outer.attest_calls += 1
                    if outer.fail_attest:
                        self._json(500, {"error": "cannot attest"})
                        return
                    if outer.fail_attestures_remaining > 0:
                        outer.fail_attestures_remaining -= 1
                        self._json(503, {"error": "temporarily unavailable"})
                        return
                    wire, _ = _sign_attestation(
                        outer.seed,
                        output_root_hex=("00" * 32) if outer.wrong_root else OUTPUT_ROOT_HEX,
                        protocol_version=outer.protocol_version)
                    if outer.bad_signature:
                        wire["Signature"] = base64.b64encode(b"\x00" * 64).decode()
                    self._json(200, {"attestation": wire,
                                     "attestation_json": json.dumps(wire).encode().hex()})
                else:
                    self._json(404, {"error": "not found"})

            def log_message(self, *args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        if self.server:
            self.server.shutdown()


@pytest.fixture()
def providers():
    servers: list[FakeProviderServer] = []

    def make(name: str, **kw) -> tuple[da.DAProvider, FakeProviderServer]:
        server = FakeProviderServer(name, bytes([len(name)]) * 32, **kw)
        servers.append(server)
        return da.DAProvider(url=server.start(), name=name), server

    yield make
    for server in servers:
        server.stop()


def quorum_call(provider_list, **kw):
    return da.ensure_quorum(provider_list, quorum=kw.pop("quorum", 2), bundle_hex="ab" * 64,
                            output_root_hex=OUTPUT_ROOT_HEX, m=M, n=N, k=K, task_id=TASK_ID,
                            assignment_id_hex=ASSIGNMENT_HEX, available_until=AVAILABLE_UNTIL,
                            attested_height=ATTESTED_HEIGHT, log=lambda _m: None, **kw)


def test_all_three_providers_verify(providers):
    plist = [providers("a")[0], providers("b")[0], providers("c")[0]]
    result = quorum_call(plist, quorum=3)
    assert len(result["verified"]) == 3
    assert all("verified" in outcome for outcome in result["outcomes"].values())


def test_one_provider_offline_still_reaches_quorum(providers):
    offline = da.DAProvider(url="http://127.0.0.1:1", name="a")  # nothing listens there
    b, _ = providers("b")
    c, _ = providers("c")
    result = quorum_call([offline, b, c], retries=1, backoff_seconds=0)
    assert len(result["verified"]) == 2
    assert "failed" in result["outcomes"]["a"], result["outcomes"]


def test_two_providers_offline_never_reaches_quorum(providers):
    a, _ = providers("a")
    offline_b = da.DAProvider(url="http://127.0.0.1:1", name="b")
    offline_c = da.DAProvider(url="http://127.0.0.1:2", name="c")
    with pytest.raises(da.QuorumNotReached) as excinfo:
        quorum_call([a, offline_b, offline_c], retries=1, backoff_seconds=0)
    assert "quorum 2 not reached" in str(excinfo.value)
    assert set(excinfo.value.outcomes) == {"a", "b", "c"}


def test_upload_success_without_a_verified_attestation_is_not_storage(providers):
    a, server_a = providers("a")
    b, server_b = providers("b", fail_attest=True)
    with pytest.raises(da.QuorumNotReached) as excinfo:
        quorum_call([a, b], quorum=2, retries=0, backoff_seconds=0)
    assert server_a.store_calls == 1 and server_b.store_calls == 1, "both uploads happened"
    assert "failed" in excinfo.value.outcomes["b"], excinfo.value.outcomes
    assert len(excinfo.value.outcomes) == 2


def test_bad_signature_provider_does_not_count(providers):
    a, _ = providers("a")
    b, _ = providers("b")
    bad, _ = providers("c", bad_signature=True)
    with pytest.raises(da.QuorumNotReached):
        quorum_call([a, b, bad], quorum=3, retries=0, backoff_seconds=0)


def test_wrong_output_root_attestation_is_refused(providers):
    a, _ = providers("a")
    wrong, _ = providers("b", wrong_root=True)
    with pytest.raises(da.QuorumNotReached):
        quorum_call([a, wrong], quorum=2, retries=0, backoff_seconds=0)


def test_wrong_protocol_version_is_refused(providers):
    a, _ = providers("a")
    stale, _ = providers("b", protocol_version="DA_REPLICA_V0")
    with pytest.raises(da.QuorumNotReached):
        quorum_call([a, stale], quorum=2, retries=0, backoff_seconds=0)


def test_transient_failure_is_retried_with_backoff(providers):
    a, server_a = providers("a")
    flaky, server_flaky = providers("b", fail_attestures=1)
    sleeps: list[float] = []
    result = quorum_call([a, flaky], quorum=2, retries=2, backoff_seconds=0.5,
                         sleep=sleeps.append)
    assert len(result["verified"]) == 2
    assert server_flaky.attest_calls == 2 and sleeps and sleeps[0] > 0


def test_provider_without_an_advertised_key_is_refused(providers):
    a, _ = providers("a")
    no_key, _ = providers("b", advertise_key=False)
    with pytest.raises(da.QuorumNotReached):
        quorum_call([a, no_key], quorum=2, retries=0, backoff_seconds=0)


def test_quorum_is_validated_against_the_provider_count(providers):
    a, _ = providers("a")
    with pytest.raises(da.DAError, match="exceeds the provider count"):
        quorum_call([a], quorum=2, retries=0, backoff_seconds=0)
    with pytest.raises(da.DAError, match="at least 1"):
        quorum_call([a], quorum=0, retries=0, backoff_seconds=0)


def test_store_refusal_is_reported_with_the_provider_reason(providers):
    refusing, _ = providers("a", fail_store=True)
    with pytest.raises(da.QuorumNotReached) as excinfo:
        quorum_call([refusing], quorum=1, retries=0, backoff_seconds=0)
    assert "store refused" in excinfo.value.outcomes["a"]
