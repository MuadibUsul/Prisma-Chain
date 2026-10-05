"""Keystore behaviour: encryption, permissions, rotation, failure modes."""

from __future__ import annotations

import base64
import json
import os
import pathlib

import pytest

from prisma_worker.identity import WorkerIdentity
from prisma_worker.keystore import (
    KeystoreError,
    KeystorePermissionsError,
    UnrecoverableKeystoreError,
    load,
    public_info,
    rotate,
    save,
)

PASSPHRASE = "correct horse battery staple"


@pytest.fixture()
def keystore_path(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "worker-key.json"


def test_roundtrip(keystore_path):
    identity = WorkerIdentity.generate()
    save(keystore_path, identity, PASSPHRASE)
    loaded = load(keystore_path, PASSPHRASE)
    assert loaded.protocol_seed == identity.protocol_seed
    assert loaded.account_scalar == identity.account_scalar
    assert loaded.account_address == identity.account_address


def test_secrets_are_not_stored_in_the_clear(keystore_path):
    identity = WorkerIdentity.generate()
    save(keystore_path, identity, PASSPHRASE)
    raw = keystore_path.read_text(encoding="utf-8")
    for secret in identity.secret_material():
        assert secret not in raw
    document = json.loads(raw)
    assert document["cipher"] == "AESGCM"
    assert document["kdf"]["name"] == "scrypt"


def test_wrong_passphrase_is_explicitly_unrecoverable(keystore_path):
    save(keystore_path, WorkerIdentity.generate(), PASSPHRASE)
    with pytest.raises(UnrecoverableKeystoreError) as excinfo:
        load(keystore_path, "wrong passphrase")
    assert "no recovery path" in str(excinfo.value)


def test_tampered_ciphertext_is_rejected(keystore_path):
    save(keystore_path, WorkerIdentity.generate(), PASSPHRASE)
    document = json.loads(keystore_path.read_text(encoding="utf-8"))
    raw = bytearray(base64.b64decode(document["ciphertext_b64"]))
    raw[0] ^= 0x01
    document["ciphertext_b64"] = base64.b64encode(bytes(raw)).decode()
    keystore_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(UnrecoverableKeystoreError):
        load(keystore_path, PASSPHRASE)


def test_weakened_kdf_is_refused(keystore_path):
    save(keystore_path, WorkerIdentity.generate(), PASSPHRASE)
    document = json.loads(keystore_path.read_text(encoding="utf-8"))
    document["kdf"]["n"] = 2 ** 8  # an attacker-edited weak KDF must not load
    keystore_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(KeystoreError) as excinfo:
        load(keystore_path, PASSPHRASE)
    assert "weakened" in str(excinfo.value) or "unsupported" in str(excinfo.value)


@pytest.mark.skipif(os.name != "posix", reason="Windows file modes are not meaningful")
def test_group_readable_keystore_is_refused(keystore_path):
    save(keystore_path, WorkerIdentity.generate(), PASSPHRASE)
    os.chmod(keystore_path, 0o644)
    with pytest.raises(KeystorePermissionsError):
        load(keystore_path, PASSPHRASE)


def test_public_info_needs_no_passphrase(keystore_path):
    identity = WorkerIdentity.generate()
    save(keystore_path, identity, PASSPHRASE)
    record = public_info(keystore_path)
    assert record["account_address"] == identity.account_address
    assert record["node_id"] == identity.node_id


def test_rotate_replaces_the_identity(keystore_path):
    first = WorkerIdentity.generate()
    save(keystore_path, first, PASSPHRASE)
    old_public, new_identity = rotate(keystore_path, PASSPHRASE)
    assert old_public["account_address"] == first.account_address
    assert new_identity.account_address != first.account_address
    loaded = load(keystore_path, PASSPHRASE)
    assert loaded.protocol_seed == new_identity.protocol_seed


def test_no_temporary_file_is_left_behind(keystore_path):
    save(keystore_path, WorkerIdentity.generate(), PASSPHRASE)
    leftovers = list(keystore_path.parent.glob("*.tmp"))
    assert leftovers == []


def test_missing_keystore_has_a_clear_error(tmp_path):
    with pytest.raises(KeystoreError) as excinfo:
        load(tmp_path / "absent.json", PASSPHRASE)
    assert "identity init" in str(excinfo.value)
