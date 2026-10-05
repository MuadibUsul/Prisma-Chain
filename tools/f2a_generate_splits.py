"""F.2A fresh data splits (EXPERIMENTAL / NON-PROTOCOL).

Creates three disjoint, frozen token sets that the F.1 evaluation set
NEVER touches:

    testdata/qwen3_f2a_calibration.json   64 sequences (scale derivation)
    testdata/qwen3_f2a_tuning.json        32 sequences (policy selection)
    testdata/qwen3_f2a_heldout.json       64 sequences (final gate, locked)

Texts are generated deterministically from a recorded seed over a phrase
bank, then tokenized once by the pinned tokenizer; the frozen artifacts
carry token ids, position ids, per-file SHA256, the generation seed and
the tokenizer revision. Heldout is never read without --unlock-heldout
(see f2a_groupwise_search.py).

    python tools/f2a_generate_splits.py
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
SEED = 20261004
SPLIT_SIZES = {"calibration": 64, "tuning": 32, "heldout": 64}

SUBJECTS = ["The engineer", "A distant probe", "The historian", "Our gardener",
            "The pilot", "A curious student", "The pharmacist", "The climber",
            "An analyst", "The violinist", "The mechanic", "A young physicist",
            "The librarian", "A field biologist", "The architect", "A chess coach"]
VERBS = ["measured", "rebuilt", "annotated", "calibrated", "catalogued",
         "rewired", "sketched", "verified", "assembled", "questioned",
         "mapped", "sharpened"]
OBJECTS = ["the storm drains", "a rusted turbine", "the monthly ledger",
           "an unfamiliar argument", "the greenhouse vents", "a broken compass",
           "the annotated map", "a fragile manuscript", "the frequency drift",
           "an old theorem", "the orchard rows", "a silent oscillator"]
TAILS = ["before the frost arrived.", "during the long outage.",
         "while the observatory slept.", "after the second trial.",
         "under a pale morning sky.", "without any formal notice.",
         "near the abandoned station.", "past the quiet river bend."]


def make_texts(count: int, offset: int) -> list[str]:
    texts = []
    for i in range(count):
        n = offset + i
        subj = SUBJECTS[n % len(SUBJECTS)]
        verb = VERBS[(n * 7 + 3) % len(VERBS)]
        obj = OBJECTS[(n * 5 + 1) % len(OBJECTS)]
        tail = TAILS[(n * 11 + 2) % len(TAILS)]
        texts.append(f"Item {n:03d}: {subj} {verb} {obj} {tail} "
                     f"Further remarks continue about the very same subject matter.")
    return texts


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    revision = ""
    try:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download("Qwen/Qwen3-0.6B-Base", "tokenizer.json",
                                   revision="57ca99e94acb83175495aa2c6b6b0cc498170924",
                                   local_dir=str(MODEL_DIR))
        revision = hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001
        revision = "local-file"

    offset = 0
    meta = {"generation_seed": SEED, "seq_len": SEQ_LEN,
            "tokenizer_file_sha256": revision,
            "note": "F.2A splits are disjoint from the F.1 calibration/eval sets"}
    for name, count in SPLIT_SIZES.items():
        cases = []
        for text in make_texts(count, offset):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) < SEQ_LEN:
                raise ValueError(f"too short: {text!r}")
            cases.append({"token_ids": ids[:SEQ_LEN],
                          "position_ids": list(range(SEQ_LEN)),
                          "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
        offset += count
        payload = {**meta, "split": name, "size": count, "cases": cases}
        path = TESTDATA / f"qwen3_f2a_{name}.json"
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.name}: {count} cases sha256={digest[:16]}")
        meta[f"{name}_sha256"] = digest
    (TESTDATA / "qwen3_f2a_splits_meta.json").write_text(json.dumps(meta, indent=1) + "\n",
                                                         encoding="utf-8")
    print("meta written")


if __name__ == "__main__":
    main()
