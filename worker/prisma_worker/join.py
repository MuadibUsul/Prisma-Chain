"""Worker join / bond / register / heartbeat (B2-03).

What the frozen chain allows (checked against chain/x/compute/keeper.go):

- ``MsgBondWorker`` requires a non-zero amount (``invalid bond amount`` for
  zero), so bonding and re-registration are always a bond top-up of >= 1.
- A bonded network key **cannot be replaced** (``bonded network key cannot
  be replaced``). Key rotation therefore means a *fresh account* with a new
  bond; the capability refresh is not a chain transaction at all.

What this module therefore does:

- ``join``: capability gate -> chain-id gate -> frozen-surface gate ->
  balance gate -> one bond/register transaction -> verify the chain shows
  our protocol key -> write the worker state file.
- ``heartbeat``: periodic liveness (status + chain-id) and capability
  re-probe; on a capability change it updates the local state and marks the
  gateway announcement as pending (the announcement itself belongs to the
  network layer, not the chain).
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import chain as chain_mod
from . import gpu
from .identity import WorkerIdentity

STATE_SCHEMA = 1


class JoinError(Exception):
    """The join sequence cannot proceed; the message says what to change."""


@dataclass
class JoinConfig:
    chain: chain_mod.ChainConfig
    bond_amount_uprsm: int = 1_000_000  # frozen MinBond
    state_path: Optional[pathlib.Path] = None
    allow_unsupported_reason: str = ""


def _write_state(path: pathlib.Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_state(path: pathlib.Path) -> dict:
    try:
        document = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise JoinError(f"no worker state at {path}; run `prisma-worker join` first") from exc
    if document.get("schema") != STATE_SCHEMA:
        raise JoinError(f"unsupported worker state schema {document.get('schema')!r}")
    return document


def join(config: JoinConfig, identity: WorkerIdentity, *,
         client: Optional[chain_mod.ChainClient] = None,
         probe_fn: Callable[[], gpu.CapabilityReport] = gpu.probe,
         log: Callable[[str], None] = print) -> dict:
    client = client or chain_mod.ChainClient(config.chain, log=log)

    # Chain identity first: never probe or announce into the wrong chain.
    chain_id = client.validate_chain_id()
    log(f"chain {chain_id} reachable at {client.active_rpc}")

    report = probe_fn()
    if not report.backend_supported and not config.allow_unsupported_reason:
        raise JoinError(
            f"refusing to join with {report.status}: {report.reason}. "
            "Pass --allow-unsupported REASON to override; the reason is recorded in the state file.")
    if not report.backend_supported:
        log(f"WARNING joining with {report.status} because: {config.allow_unsupported_reason}")

    address = identity.account_address
    surface = client.check_frozen_surface(address)
    log(f"frozen compute surface present: {surface.worker_query_fields}")

    bond_target = max(config.bond_amount_uprsm, 1)
    bonded = int(surface.bonded_uprsm)
    outcome = {
        "schema": STATE_SCHEMA,
        "chain_id": chain_id,
        "account_address": address,
        "node_id": identity.node_id,
        "protocol_public_b64": identity.protocol_public_b64,
        "capability": report.to_dict(),
        "allow_unsupported_reason": config.allow_unsupported_reason,
        "joined_at_ms": int(time.time() * 1000),
        "bond_txhash": "",
    }

    registered = chain_mod.decode_network_key(surface.network_public_key_b64) == identity.protocol_public
    if registered and bonded >= bond_target:
        log(f"already bonded ({bonded} uprsm) and registered; nothing to submit")
    else:
        needed = max(bond_target - bonded, 1)  # the chain refuses amount 0
        balance = client.balance_uprsm(address)
        log(f"balance {balance} uprsm, bonded {bonded} uprsm, need {needed} more to bond")
        if balance < needed:
            raise chain_mod.InsufficientBalanceError(
                f"account {address} has {balance} uprsm but {needed} uprsm is required to bond "
                f"(bond target {bond_target}). Fund the account first (testnet faucet or a transfer), "
                "or lower --bond.")
        txhash = client.bond_worker(
            amount_uprsm=needed,
            network_public_key_hex=identity.protocol_public_hex,
            network_key_proof_hex=identity.network_key_proof_hex(chain_id, address),
            account_scalar_hex=identity.account_scalar.hex())
        outcome["bond_txhash"] = txhash
        log(f"bond/register transaction included: {txhash}")

    final = client.query_worker(address)
    bound_key = chain_mod.decode_network_key(final.get("network_public_key", ""))
    if bound_key != identity.protocol_public:
        raise JoinError(
            "the chain does not show this worker's protocol key after the transaction; "
            "refusing to report a successful join (check the transaction log)")
    outcome.update({
        "bonded_uprsm": int(final.get("bonded_uprsm", 0)),
        "reserved_uprsm": int(final.get("reserved_uprsm", 0)),
        "gateway_announcement_pending": True,
    })
    if config.state_path:
        _write_state(pathlib.Path(config.state_path), outcome)
        log(f"worker state written to {config.state_path}")
    return outcome


def heartbeat(config: JoinConfig, identity: WorkerIdentity, *,
              client: Optional[chain_mod.ChainClient] = None,
              probe_fn: Callable[[], gpu.CapabilityReport] = gpu.probe,
              interval_seconds: float = 30.0,
              iterations: Optional[int] = None,
              sleep: Callable[[float], None] = time.sleep,
              log: Callable[[str], None] = print) -> dict:
    """Liveness + capability refresh loop (bounded by ``iterations`` for tests)."""
    if not config.state_path:
        raise JoinError("heartbeat requires a state path (run join first)")
    state = load_state(pathlib.Path(config.state_path))
    client = client or chain_mod.ChainClient(config.chain, log=log)
    previous = state.get("capability", {}).get("compute_capability")
    ticks = 0
    while iterations is None or ticks < iterations:
        ticks += 1
        chain_id = client.validate_chain_id()
        if chain_id != state["chain_id"]:
            raise chain_mod.ChainMismatchError(
                f"the endpoint now serves chain {chain_id!r}, state says {state['chain_id']!r}")
        report = probe_fn()
        if report.compute_capability != previous:
            log(f"capability changed {previous!r} -> {report.compute_capability!r}; "
                "updating state and marking the gateway announcement pending "
                "(the frozen chain cannot replace a bonded key)")
            state["capability"] = report.to_dict()
            state["gateway_announcement_pending"] = True
            state["capability_changed_at_ms"] = int(time.time() * 1000)
            previous = report.compute_capability
        surface = client.check_frozen_surface(identity.account_address)
        bound = chain_mod.decode_network_key(surface.network_public_key_b64)
        if bound != identity.protocol_public:
            log("the chain no longer shows our protocol key; re-run `prisma-worker join`")
            state["gateway_announcement_pending"] = True
        if config.state_path:
            _write_state(pathlib.Path(config.state_path), state)
        log(f"heartbeat ok (chain {chain_id}, bonded {surface.bonded_uprsm} uprsm)")
        if iterations is None or ticks < iterations:
            sleep(interval_seconds)
    return state
