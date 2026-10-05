"""B2-01f: no key material may reach any log path."""

from __future__ import annotations

import io
import logging

from prisma_worker.identity import WorkerIdentity
from prisma_worker.redact import SecretRedactingFilter, clear_registered_secrets, install, register_secret


def _capture(logger: logging.Logger) -> io.StringIO:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    return stream


def test_registered_secrets_are_scrubbed_everywhere():
    logger = logging.getLogger("prisma_worker.test.registered")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    install(logger)
    stream = _capture(logger)
    try:
        identity = WorkerIdentity.generate()
        for secret in identity.secret_material():
            register_secret(secret)
        logger.info("seed is %s", identity.protocol_seed.hex())
        logger.info(f"inline seed {identity.protocol_seed.hex()} in an f-string")
        logger.info("scalar " + identity.account_scalar.hex())
        logger.info("base64 form %s", identity.protocol_public_b64)  # public: must survive
        try:
            raise RuntimeError(f"failure mentioning {identity.protocol_seed.hex()}")
        except RuntimeError:
            logger.exception("caught")
        output = stream.getvalue()
        for secret in identity.secret_material():
            assert secret not in output, f"secret leaked: {secret[:8]}…"
        assert "«redacted" in output
        assert identity.protocol_public_b64 in output
    finally:
        clear_registered_secrets()


def test_unregistered_key_shaped_hex_is_still_scrubbed():
    logger = logging.getLogger("prisma_worker.test.unregistered")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    install(logger)
    stream = _capture(logger)
    try:
        unknown_key = "deadbeef" * 8  # 64 hex chars, never registered
        logger.warning("raw key %s", unknown_key)
        output = stream.getvalue()
        assert unknown_key not in output
        assert "«redacted-hex»" in output
    finally:
        clear_registered_secrets()


def test_short_values_are_not_registered():
    register_secret("abcd")
    logger = logging.getLogger("prisma_worker.test.short")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    install(logger)
    stream = _capture(logger)
    try:
        logger.info("value abcd stays")
        assert "abcd stays" in stream.getvalue()
    finally:
        clear_registered_secrets()


def test_filter_is_a_logging_filter():
    assert isinstance(SecretRedactingFilter(), logging.Filter)


def test_cli_init_never_prints_secrets(tmp_path, capsys, monkeypatch):
    import os

    from prisma_worker.cli import main

    keystore = tmp_path / "k.json"
    # An empty passphrase is refused (no key material written at all).
    monkeypatch.setenv("PRISMA_WORKER_PASSPHRASE", "")
    monkeypatch.setattr("getpass.getpass", lambda *a, **k: "")
    assert main(["identity", "init", "--keystore", str(keystore), "--json"]) != 0
    assert not keystore.exists()

    monkeypatch.setenv("PRISMA_WORKER_PASSPHRASE", "test passphrase")
    code = main(["identity", "init", "--keystore", str(keystore), "--json"])
    assert code == 0
    out = capsys.readouterr().out
    assert "account_address" in out and "node_id" in out
    assert "seed" not in out.lower()
    # The keystore exists and holds no plaintext secret.
    raw = keystore.read_text(encoding="utf-8")
    assert "protocol_seed_b64" not in raw
    assert os.environ.get("PRISMA_WORKER_PASSPHRASE") == "test passphrase"
