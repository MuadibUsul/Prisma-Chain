"""Funded usage counts delivered text with a locally pinned tokenizer."""

import asyncio
import hashlib
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from prisma_network.core import ModelPin, digest
from prisma_network.server import _pin_from_env, _token_counter_from_env, _verified_pin_from_env


def test_pinned_tokenizer_counts_and_rejects_tampering(tmp_path, monkeypatch):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"test weights")
    files = {
        "weights": {"weights.bin": hashlib.sha256(weights_path.read_bytes()).hexdigest()},
        "tokenizer": {"tokenizer.json": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(files), encoding="utf-8")
    pin = ModelPin(model_id="Qwen/test", weights_digest=digest(files["weights"]),
                   tokenizer_digest=digest(files["tokenizer"]), runtime_digest="a" * 64,
                   spec_version="light-v1")
    for key, value in {
        "PRISMA_MODEL_ID": pin.model_id, "PRISMA_MODEL_DIGEST": pin.model_digest,
        "PRISMA_WEIGHTS_DIGEST": pin.weights_digest,
        "PRISMA_TOKENIZER_DIGEST": pin.tokenizer_digest,
        "PRISMA_RUNTIME_DIGEST": pin.runtime_digest,
        "PRISMA_SPEC_VERSION": pin.spec_version,
        "PRISMA_MODEL_DIR": str(tmp_path), "PRISMA_MODEL_FILES_FILE": str(manifest_path),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("PRISMA_DEV_SKIP_FILE_HASH", raising=False)

    assert _verified_pin_from_env() == _pin_from_env() == pin
    count = _token_counter_from_env(pin)
    assert asyncio.run(count("hello world")) == 2
    assert asyncio.run(count("hello")) == 1
    tokenizer_path.write_bytes(tokenizer_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="tokenizer artifact hash mismatch"):
        _token_counter_from_env(pin)
    monkeypatch.setenv("PRISMA_DEV_SKIP_FILE_HASH", "1")
    with pytest.raises(ValueError, match="cannot skip tokenizer hash"):
        _token_counter_from_env(pin)
