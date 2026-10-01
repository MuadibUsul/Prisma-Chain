"""Generate local-only gateway/worker identities and mock model pin."""

import base64
import hashlib
import json
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "network"))
from prisma_network.core import Identity, ModelPin  # noqa: E402


def seed() -> tuple[str, Identity]:
    encoded = base64.b64encode(secrets.token_bytes(32)).decode()
    return encoded, Identity.from_seed_b64(encoded)


def main() -> None:
    target = Path(__file__).resolve().parent
    env = target / ".env"
    trusted = target / "data" / "trusted-keys.json"
    if env.exists() or trusted.exists():
        raise SystemExit("Devnet identity exists; remove deploy/.env and deploy/data to regenerate it")
    gateway_seed, gateway = seed()
    worker_seed, worker = seed()
    pin = ModelPin(
        model_id="demo-model",
        weights_digest=hashlib.sha256(b"prisma-dev-mock-weights-v1").hexdigest(),
        tokenizer_digest=hashlib.sha256(b"prisma-dev-mock-tokenizer-v1").hexdigest(),
        runtime_digest=hashlib.sha256(b"prisma-dev-mock-runtime-v1").hexdigest(),
        spec_version="light-dev-v1",
    )
    trusted.parent.mkdir(parents=True, exist_ok=True)
    trusted.write_text(json.dumps({gateway.node_id: gateway.public_key_b64,
                                   worker.node_id: worker.public_key_b64}, indent=2) + "\n", encoding="utf-8")
    values = {
        "PRISMA_NODE_SEED_GATEWAY_B64": gateway_seed,
        "PRISMA_NODE_SEED_WORKER_B64": worker_seed,
        "PRISMA_CLIENT_API_KEY": secrets.token_urlsafe(32),
        "PRISMA_MODEL_ID": pin.model_id,
        "PRISMA_MODEL_DIGEST": pin.model_digest,
        "PRISMA_WEIGHTS_DIGEST": pin.weights_digest,
        "PRISMA_TOKENIZER_DIGEST": pin.tokenizer_digest,
        "PRISMA_RUNTIME_DIGEST": pin.runtime_digest,
        "PRISMA_SPEC_VERSION": pin.spec_version,
    }
    env.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
    env.chmod(0o600)
    print("Created deploy/.env and deploy/data/trusted-keys.json for local development only")


if __name__ == "__main__":
    main()
