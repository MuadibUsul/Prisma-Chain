"""F.5A fresh data splits (EXPERIMENTAL / NON-PROTOCOL).

64 / 64 / 64 disjoint sequences that overlap with ZERO of the F.1
(calibration, eval) and F.2A (calibration, tuning, heldout) token
sequences; the overlap check is exhaustive and recorded.

    python tools/f3a_generate_splits.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

MODEL_DIR = Path(os.environ.get("PRISMA_MODEL_DIR",
                                str(REPO / "models" / "qwen3-0.6b-base")))
TESTDATA = REPO / "testdata"
SEQ_LEN = 16
SEED = 20261007
SPLIT_SIZES = {"calibration": 64, "selection": 64, "heldout": 64}

ADJS = ["lacquered", "ink-black", "windless", "bread", "sun-cracked", "foreign",
        "feathered", "gilded", "plain", "drowned", "spare", "quiet"]
NOUNS = ["apothecary", "sawyer", "tanner", "miller", "chandler",
         "cutler", "cooper", "fletcher", "cordwainer", "brewer",
         "thatcher", "wright"]
ACTS = ["barrelled", "stowed", "tallied", "oiled", "stamped",
        "sorted", "wrapped", "sealed", "loaded", "roped",
        "brushed", "counted"]
THINGS = ["the flour sacks", "a varnish pot", "the dye records",
          "two bushels", "the wool bales", "an iron rake",
          "the hide bundles", "a stone jar", "the grain dockets",
          "three kegs", "the wax tablets", "a plumb line"]
ENDS = ["before the fair opened.", "through the lunch hour.", "beside the millrace.",
        "while the bread cooled.", "near the river gate.",
        "after the last cart.", "past the drying sheds.",
        "under an opal sky."]


def make_texts(count: int, offset: int) -> list[str]:
    texts = []
    for i in range(count):
        n = offset + i
        texts.append(
            f"Chit {n:04d}: the {ADJS[n % 12]} {NOUNS[(n * 5 + 1) % 12]} "
            f"{ACTS[(n * 7 + 2) % 12]} {THINGS[(n * 11 + 5) % 12]} "
            f"{ENDS[(n * 13 + 3) % 8]} The notes continue in the same careful manner.")
    return texts


def token_sets(path: Path):
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {tuple(case["token_ids"]) for case in data["cases"]}


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)

    previous = {}
    for name in ("qwen3_f1_calibration", "qwen3_f1_eval",
                 "qwen3_f2a_calibration", "qwen3_f2a_tuning", "qwen3_f2a_heldout",
                 "qwen3_f3a_calibration", "qwen3_f3a_selection", "qwen3_f3a_heldout",
                 "qwen3_f4a_calibration", "qwen3_f4a_selection", "qwen3_f4a_heldout"):
        previous[name] = token_sets(TESTDATA / f"{name}.json")
    prev_all = set().union(*previous.values()) if previous else set()

    manifest = {"generation_seed": SEED, "seq_len": SEQ_LEN,
                "overlap_checks": {k: 0 for k in previous}, "files": {}}
    offset = 0
    produced = set()
    for name, count in SPLIT_SIZES.items():
        cases = []
        for text in make_texts(count, offset):
            ids = tuple(tokenizer(text, add_special_tokens=False)["input_ids"][:SEQ_LEN])
            if len(ids) < SEQ_LEN:
                raise ValueError(f"short text: {text!r}")
            if ids in prev_all or ids in produced:
                raise ValueError(f"token overlap with previous data: {ids}")
            produced.add(ids)
            cases.append({"token_ids": list(ids), "position_ids": list(range(SEQ_LEN)),
                          "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
        offset += count
        payload = {**{k: manifest[k] for k in ("generation_seed", "seq_len")},
                   "split": name, "size": count, "cases": cases}
        path = TESTDATA / f"qwen3_f5a_{name}.json"
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ids_set = {tuple(c["token_ids"]) for c in cases}
        for prev_name, prev_ids in previous.items():
            manifest["overlap_checks"][prev_name] += len(ids_set & prev_ids)
        manifest["files"][f"qwen3_f5a_{name}.json"] = {
            "sha256": digest, "count": count, "seq_len": SEQ_LEN}
        print(f"{name}: {count} cases sha256={digest[:16]}")

    if any(v != 0 for v in manifest["overlap_checks"].values()):
        raise SystemExit(f"nonzero overlap: {manifest['overlap_checks']}")
    (REPO / "docs" / "phase-f5a-data-manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print("overlap = 0 against all previous splits; manifest written")


if __name__ == "__main__":
    main()
