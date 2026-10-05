"""B2-03: chain client, join sequence and heartbeat (unit level, no live chain).

The live integration run lives in deploy/worker_join_smoke.py; these tests
pin the decision logic and the failure modes that must never degrade into
silent success.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from prisma_worker import chain as chain_mod
from prisma_worker import gpu
from prisma_worker.identity import WorkerIdentity
from prisma_worker.join import JoinConfig, JoinError, heartbeat, join, load_state

RPC = "http://127.0.0.1:26657"


def capability(supported: bool = True, cc: str = "8.6") -> gpu.CapabilityReport:
    return gpu.CapabilityReport(
        gpu_model="NVIDIA A40" if supported else "NVIDIA GeForce RTX 2060 SUPER",
        compute_capability=cc, vram_bytes=1024, driver_version="550", cuda_runtime="12.8",
        backend=gpu.BACKEND, backend_supported=supported,
        status="CAPABILITY_SUPPORTED" if supported else "CAPABILITY_UNSUPPORTED",
        reason="proven" if supported else "compute capability 7.5 is not a proven target",
        supported_profiles=[gpu.CANONICAL_PROFILE] if supported else [])


class FakeClient:
    """Stands in for ChainClient with scripted responses."""

    def __init__(self, *, chain_id="prisma-testnet-1", bonded=0, balance=0, bound_key=b"",
                 valid_chain=True, frozen=True, rpc=RPC):
        self.chain_id = chain_id
        self.bonded = bonded
        self.balance = balance
        self.bound_key = bound_key
        self.valid_chain = valid_chain
        self.frozen = frozen
        self.active_rpc = rpc
        self.bond_calls: list[dict] = []

    def validate_chain_id(self):
        if not self.valid_chain:
            raise chain_mod.ChainMismatchError(
                f"refusing to join chain 'prisma-other-1': worker is configured for {self.chain_id!r}")
        return self.chain_id

    def check_frozen_surface(self, address):
        if not self.frozen:
            raise chain_mod.ProtocolUnsupportedError("the chain does not expose the frozen compute surface")
        return chain_mod.FrozenSurface(
            worker_query_fields=["bonded_uprsm", "network_public_key", "reserved_uprsm"],
            bonded_uprsm=self.bonded, network_public_key_b64=(
                __import__("base64").b64encode(self.bound_key).decode() if self.bound_key else ""))

    def balance_uprsm(self, address):
        return self.balance

    def bond_worker(self, *, amount_uprsm, network_public_key_hex, network_key_proof_hex,
                    account_scalar_hex):
        self.bond_calls.append({"amount": amount_uprsm, "key": network_public_key_hex,
                                "proof": network_key_proof_hex})
        self.bonded += amount_uprsm
        self.bound_key = bytes.fromhex(network_public_key_hex)
        return "TXHASH"

    def query_worker(self, address):
        import base64

        return {"bonded_uprsm": str(self.bonded),
                "network_public_key": base64.b64encode(self.bound_key).decode(),
                "reserved_uprsm": "0"}


@pytest.fixture()
def identity() -> WorkerIdentity:
    return WorkerIdentity(protocol_seed=bytes(range(32)), account_scalar=bytes([0x11] * 32))


def _config(tmp_path: pathlib.Path, **kw) -> JoinConfig:
    return JoinConfig(chain=chain_mod.ChainConfig(rpc_urls=[RPC], chain_id="prisma-testnet-1"),
                      state_path=tmp_path / "state.json", **kw)


def test_join_happy_path_bonds_registers_and_writes_state(tmp_path, identity):
    client = FakeClient(balance=5_000_000)
    state = join(_config(tmp_path), identity, client=client, probe_fn=capability, log=lambda _: None)
    assert client.bond_calls and client.bond_calls[0]["amount"] == 1_000_000
    assert client.bond_calls[0]["key"] == identity.protocol_public_hex
    assert client.bond_calls[0]["proof"] == identity.network_key_proof_hex("prisma-testnet-1",
                                                                          identity.account_address)
    assert state["bonded_uprsm"] == 1_000_000
    assert state["capability"]["status"] == "CAPABILITY_SUPPORTED"
    assert load_state(tmp_path / "state.json")["account_address"] == identity.account_address


def test_join_is_idempotent_when_already_bonded(tmp_path, identity):
    client = FakeClient(bonded=1_000_000, bound_key=identity.protocol_public)
    join(_config(tmp_path), identity, client=client, probe_fn=capability, log=lambda _: None)
    assert client.bond_calls == []


def test_join_refuses_unsupported_capability_without_reason(tmp_path, identity):
    with pytest.raises(JoinError, match="refusing to join with CAPABILITY_UNSUPPORTED"):
        join(_config(tmp_path), identity, client=FakeClient(balance=5_000_000),
             probe_fn=lambda: capability(False, "7.5"), log=lambda _: None)


def test_join_records_an_explicit_override(tmp_path, identity):
    config = _config(tmp_path, allow_unsupported_reason="lab benchmark only, SM75")
    state = join(config, identity, client=FakeClient(balance=5_000_000),
                 probe_fn=lambda: capability(False, "7.5"), log=lambda _: None)
    assert state["allow_unsupported_reason"] == "lab benchmark only, SM75"
    assert state["capability"]["status"] == "CAPABILITY_UNSUPPORTED"


def test_join_refuses_wrong_chain(tmp_path, identity):
    with pytest.raises(chain_mod.ChainMismatchError, match="refusing to join"):
        join(_config(tmp_path), identity, client=FakeClient(valid_chain=False),
             probe_fn=capability, log=lambda _: None)


def test_join_refuses_chain_without_frozen_surface(tmp_path, identity):
    with pytest.raises(chain_mod.ProtocolUnsupportedError):
        join(_config(tmp_path), identity, client=FakeClient(frozen=False),
             probe_fn=capability, log=lambda _: None)


def test_join_insufficient_balance_is_actionable(tmp_path, identity):
    with pytest.raises(chain_mod.InsufficientBalanceError) as excinfo:
        join(_config(tmp_path), identity, client=FakeClient(balance=10),
             probe_fn=capability, log=lambda _: None)
    message = str(excinfo.value)
    assert identity.account_address in message and "Fund the account" in message and "--bond" in message


def test_join_topup_uses_the_missing_amount(tmp_path, identity):
    client = FakeClient(bonded=400_000, balance=5_000_000)
    join(_config(tmp_path), identity, client=client, probe_fn=capability, log=lambda _: None)
    assert client.bond_calls[0]["amount"] == 600_000  # 1_000_000 target - 400_000 bonded


def test_join_refuses_to_claim_success_when_the_chain_disagrees(tmp_path, identity):
    class LyingClient(FakeClient):
        def bond_worker(self, **kw):
            return "TXHASH"  # pretends to bond but never records the key

    with pytest.raises(JoinError, match="does not show this worker's protocol key"):
        join(_config(tmp_path), identity, client=LyingClient(balance=5_000_000),
             probe_fn=capability, log=lambda _: None)


def test_heartbeat_detects_capability_change_and_marks_announcement(tmp_path, identity):
    join(_config(tmp_path), identity, client=FakeClient(balance=5_000_000),
         probe_fn=capability, log=lambda _: None)
    client = FakeClient(bonded=1_000_000, bound_key=identity.protocol_public)
    state = heartbeat(_config(tmp_path), identity, client=client,
                      probe_fn=lambda: capability(False, "7.5"), iterations=1,
                      log=lambda _: None)
    assert state["capability"]["compute_capability"] == "7.5"
    assert state["gateway_announcement_pending"] is True
    assert "capability_changed_at_ms" in state


def test_heartbeat_flags_chain_side_registration_loss(tmp_path, identity):
    join(_config(tmp_path), identity, client=FakeClient(balance=5_000_000),
         probe_fn=capability, log=lambda _: None)
    client = FakeClient(bonded=1_000_000, bound_key=b"\x00" * 32)  # someone else's key
    state = heartbeat(_config(tmp_path), identity, client=client, probe_fn=capability,
                      iterations=1, log=lambda _: None)
    assert state["gateway_announcement_pending"] is True


# --- chain.py unit level ---------------------------------------------------


def test_status_fails_over_with_explicit_logging():
    logs: list[str] = []

    def http_get(url: str) -> str:
        if "dead" in url:
            raise OSError("connection refused")
        return json.dumps({"result": {"node_info": {"network": "prisma-testnet-1"}}})

    client = chain_mod.ChainClient(
        chain_mod.ChainConfig(rpc_urls=["http://dead:26657", "http://alive:26657"],
                             chain_id="prisma-testnet-1"),
        runner=lambda *a, **k: "{}", http_get=http_get, log=logs.append)
    assert client.validate_chain_id() == "prisma-testnet-1"
    assert client.active_rpc == "http://alive:26657"
    assert any("dead" in line and "failed" in line for line in logs)


def test_all_endpoints_down_is_one_explicit_error():
    client = chain_mod.ChainClient(
        chain_mod.ChainConfig(rpc_urls=["http://dead:26657"], chain_id="x"),
        runner=lambda *a, **k: "{}", http_get=lambda url: (_ for _ in ()).throw(OSError("refused")))
    with pytest.raises(chain_mod.ChainError, match="no reachable RPC endpoint"):
        client.status()


def test_bond_worker_command_shape_and_keyring_cleanup():
    commands: list[list[str]] = []

    def runner(cmd, timeout=60.0):
        commands.append(cmd)
        if "keys" in cmd:
            return "{}"
        return json.dumps({"code": 0, "txhash": "ABCD"})

    client = chain_mod.ChainClient(
        chain_mod.ChainConfig(rpc_urls=[RPC], chain_id="prisma-testnet-1"),
        runner=runner, http_get=lambda url: json.dumps({"result": {"tx_result": {"code": 0}}}))
    txhash = client.bond_worker(amount_uprsm=1_000_000, network_public_key_hex="ab" * 32,
                                network_key_proof_hex="cd" * 64, account_scalar_hex="11" * 32)
    assert txhash == "ABCD"
    tx = [c for c in commands if "bond-worker" in c][0]
    assert "--network-public-key" in tx and "--network-key-proof" in tx
    assert tx[tx.index("--amount") + 1] == "1000000"
    assert tx[tx.index("--chain-id") + 1] == "prisma-testnet-1"
    keyring_dir = pathlib.Path(tx[tx.index("--keyring-dir") + 1])
    assert not keyring_dir.exists(), "temporary keyring must be shredded after the transaction"


def test_check_frozen_surface_accepts_a_fresh_worker():
    """proto3 omits zero values: a brand-new worker answers {} and must pass."""
    client = chain_mod.ChainClient(
        chain_mod.ChainConfig(rpc_urls=[RPC], chain_id="x"),
        runner=lambda *a, **k: "{}",
        http_get=lambda url: json.dumps({"result": {"node_info": {"network": "x"}}}))
    surface = client.check_frozen_surface("prsm1abc")
    assert surface.bonded_uprsm == 0 and surface.network_public_key_b64 == ""


def test_check_frozen_surface_rejects_a_chain_without_the_module():
    def failing_runner(*_a, **_k):
        raise chain_mod.ChainError("unknown command \"worker\" for \"prismad query compute\"")

    client = chain_mod.ChainClient(
        chain_mod.ChainConfig(rpc_urls=[RPC], chain_id="x"),
        runner=failing_runner,
        http_get=lambda url: json.dumps({"result": {"node_info": {"network": "x"}}}))
    with pytest.raises(chain_mod.ProtocolUnsupportedError, match="frozen compute surface"):
        client.check_frozen_surface("prsm1abc")


def test_query_worker_uses_the_worker_flag():
    commands: list[list[str]] = []

    def runner(cmd, timeout=60.0):
        commands.append(cmd)
        return json.dumps({"bonded_uprsm": "1", "network_public_key": "", "reserved_uprsm": "0"})

    client = chain_mod.ChainClient(
        chain_mod.ChainConfig(rpc_urls=[RPC], chain_id="x"), runner=runner,
        http_get=lambda url: json.dumps({"result": {"node_info": {"network": "x"}}}))
    client.query_worker("prsm1abc")
    cmd = commands[-1]
    assert cmd[cmd.index("worker") + 1] == "--worker"       # flag, not a positional arg
    assert cmd[cmd.index("--worker") + 1] == "prsm1abc"


def test_chain_identity_is_checked_before_the_gpu_probe(tmp_path, identity):
    """Wrong chain must fail even when the GPU would also refuse."""
    calls: list[str] = []

    def probe():
        calls.append("probe")
        return capability(False, "7.5")

    with pytest.raises(chain_mod.ChainMismatchError):
        join(_config(tmp_path), identity, client=FakeClient(valid_chain=False),
             probe_fn=probe, log=lambda _: None)
    assert calls == [], "the GPU must not be probed before the chain is validated"
