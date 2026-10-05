"""Phase F.1 model manifest: download the pinned Qwen3 checkpoint and
record every identity that matters.

    python tools/f1_model_manifest.py [--probe-only]

Writes docs/phase-f1-model-manifest.json with the repository, the exact
revision, the license, and the SHA256 of every downloaded file. Never
uses trust_remote_code and never resolves a floating revision.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("PRISMA_MODEL_DIR", str(REPO_ROOT / "models" / "qwen3-0.6b-base")))

REPO_ID = "Qwen/Qwen3-0.6B-Base"
REVISION = "57ca99e94acb83175495aa2c6b6b0cc498170924"
MANIFEST = REPO_ROOT / "docs" / "phase-f1-model-manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-only", action="store_true",
                        help="only fetch the metadata and small config files")
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    try:
        info = api.model_info(REPO_ID, revision=REVISION, files_metadata=True)
    except Exception as exc:  # noqa: BLE001
        print(f"MODEL_REVISION_UNAVAILABLE: {exc}")
        sys.exit(2)
    if info.sha != REVISION:
        print(f"MODEL_REVISION_UNAVAILABLE: resolved sha {info.sha} != pinned {REVISION}")
        sys.exit(2)
    license_name = ""
    card = info.card_data
    if card is not None:
        license_name = str(getattr(card, "license", "") or "")
    if not license_name:
        for tag in info.tags or []:
            if tag.startswith("license:"):
                license_name = tag.split(":", 1)[1]

    files = sorted((sib.rfilename, sib.size) for sib in info.siblings)
    print("repo files:")
    for name, size in files:
        print(f"  {name} ({size if size is not None else '?'} bytes)")

    MODE = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "")
    _ = MODE
    if args.probe_only:
        for name in ("config.json", "tokenizer.json", "generation_config.json"):
            try:
                path = hf_hub_download(REPO_ID, name, revision=REVISION,
                                       local_dir=MODEL_DIR)
                print("fetched:", path)
            except Exception as exc:  # noqa: BLE001
                print(f"  (skipped {name}: {exc})")

    manifest_files = {}
    if not args.probe_only:
        # Files already on disk (e.g. a checkpoint fetched with curl and
        # hash-verified against the pinned blob etag) are trusted as-is;
        # only missing files are fetched.
        missing = [name for name, _size in files if not (MODEL_DIR / name).exists()]
        if missing:
            snapshot_download(REPO_ID, revision=REVISION, local_dir=MODEL_DIR,
                              allow_patterns=missing, max_workers=4)
        for root, _dirs, names in os.walk(MODEL_DIR):
            for name in sorted(names):
                path = Path(root) / name
                if path.name.startswith("."):
                    continue
                rel = str(path.relative_to(MODEL_DIR)).replace("\\", "/")
                manifest_files[rel] = {"bytes": path.stat().st_size,
                                       "sha256": sha256_file(path)}

    import huggingface_hub
    try:
        import torch
        torch_version = torch.__version__
    except Exception:  # noqa: BLE001
        torch_version = "NOT INSTALLED"
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:  # noqa: BLE001
        transformers_version = "NOT INSTALLED"

    manifest = {
        "repository": REPO_ID,
        "revision": REVISION,
        "revision_source": "explicit pin (never main/latest/floating tags)",
        "license": license_name or "see repository",
        "trust_remote_code": False,
        "files": manifest_files,
        "huggingface_hub": huggingface_hub.__version__,
        "transformers": transformers_version,
        "torch": torch_version,
        "repo_file_list": [{"name": n, "bytes": s} for n, s in files],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("written:", MANIFEST, "files:", len(manifest_files))


if __name__ == "__main__":
    main()
