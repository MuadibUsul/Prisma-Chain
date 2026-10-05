"""Identity tests, anchored to the chain's own tooling.

The address vector was produced by the node itself:

    prismad keys import-hex vec 1111…11 --keyring-backend test --home <tmp>
    prismad keys show vec -a  ->  prsm1l3e9pgs3mmwuwrh95fecme0s0qtn28806jn3dq
    prismad keys show vec -p  ->  A081W9y3zAr3KO88zrlhXZBoS7Wyyl+FmrDwtwQHWHGq

so any drift between this package and the chain's address format fails here.
"""

from __future__ import annotations

import base64
import json

import pytest
from prisma_network.core import Identity as NetworkIdentity

from prisma_worker.identity import (
    SECP256K1_ORDER,
    WorkerIdentity,
    bech32_decode,
    bech32_encode,
    node_id,
)

VECTOR_SCALAR = bytes([0x11] * 32)
VECTOR_ADDRESS = "prsm1l3e9pgs3mmwuwrh95fecme0s0qtn28806jn3dq"
VECTOR_PUBKEY_B64 = "A081W9y3zAr3KO88zrlhXZBoS7Wyyl+FmrDwtwQHWHGq"


def test_account_address_matches_chain_tooling():
    identity = WorkerIdentity(protocol_seed=bytes(range(32)), account_scalar=VECTOR_SCALAR)
    assert identity.account_address == VECTOR_ADDRESS
    assert identity.account_public_b64 == VECTOR_PUBKEY_B64


def test_node_id_is_sha256_of_the_protocol_public_key():
    identity = WorkerIdentity(protocol_seed=bytes([7] * 32), account_scalar=VECTOR_SCALAR)
    assert identity.node_id == node_id(identity.protocol_public)
    assert len(identity.node_id) == 64


def test_bech32_roundtrip_and_checksum():
    hrp, decoded = bech32_decode(VECTOR_ADDRESS)
    assert hrp == "prsm"
    assert len(decoded) == 20  # cosmos account address hash length
    # Re-encoding the same 20 bytes reproduces the exact address.
    from prisma_worker.identity import account_address

    assert bech32_decode(account_address(
        WorkerIdentity(protocol_seed=bytes(32), account_scalar=VECTOR_SCALAR)
        .account_public_compressed))[1] == decoded
    with pytest.raises(ValueError):
        bech32_decode(VECTOR_ADDRESS[:-1] + ("q" if VECTOR_ADDRESS[-1] != "q" else "p"))


def test_protocol_signature_matches_the_network_package():
    seed = bytes(range(32))
    worker = WorkerIdentity(protocol_seed=seed, account_scalar=VECTOR_SCALAR)
    network = NetworkIdentity.from_seed_b64(base64.b64encode(seed).decode())
    assert worker.protocol_public_b64 == network.public_key_b64
    assert worker.node_id == network.node_id
    payload = {"b": 2, "a": 1}
    assert worker.sign_protocol("prisma:test:v1", payload) == network.sign("prisma:test:v1", payload)
    assert (worker.network_key_proof_hex("prisma-testnet-1", worker.account_address)
            == network.network_key_proof_hex("prisma-testnet-1", worker.account_address))


def test_public_record_never_contains_secrets():
    identity = WorkerIdentity.generate()
    serialized = json.dumps(identity.public_record())
    for secret in identity.secret_material():
        assert secret not in serialized
    assert identity.protocol_public_b64 in serialized
    assert identity.account_address in serialized


def test_rejects_invalid_key_material():
    with pytest.raises(ValueError):
        WorkerIdentity(protocol_seed=b"short", account_scalar=VECTOR_SCALAR)
    with pytest.raises(ValueError):
        WorkerIdentity(protocol_seed=bytes(32), account_scalar=bytes(32))  # scalar 0
    out_of_range = (SECP256K1_ORDER + 1).to_bytes(32, "big")
    with pytest.raises(ValueError):
        WorkerIdentity(protocol_seed=bytes(32), account_scalar=out_of_range)


def test_import_hex_accepts_prefixed_and_bare():
    hex_seed = "ab" * 32
    a = WorkerIdentity.from_protocol_hex(hex_seed, VECTOR_SCALAR)
    b = WorkerIdentity.from_protocol_hex("0x" + hex_seed, VECTOR_SCALAR)
    assert a.protocol_public == b.protocol_public
