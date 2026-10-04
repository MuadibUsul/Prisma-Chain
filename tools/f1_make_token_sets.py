"""Freeze the calibration and evaluation token sets (Phase F.1 §19-20).

The corpus is written here, tokenized ONCE by the pinned tokenizer, and
then frozen as token ids + sequence lengths + position ids, so later
tokenizer version changes cannot drift the gate inputs. Calibration and
evaluation paragraphs are disjoint.

    python tools/f1_make_token_sets.py
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen3-0.6b-base"
TESTDATA = Path(__file__).resolve().parents[1] / "testdata"

SEQ_LEN = 16

CALIBRATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "In a distant galaxy, astronomers observed a supernova in 1987.",
    "She sells seashells by the seashore every summer morning.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The committee approved the budget after a lengthy debate.",
    "f(x) = a*x^2 + b*x + c is a quadratic function of x.",
    "def add(a, b): return a + b  # simple integer addition",
    "Photosynthesis converts carbon dioxide and water into glucose.",
    "The train arrived at platform nine, exactly on schedule.",
    "Mount Everest is the highest mountain above sea level.",
    "He played the violin with remarkable precision and feeling.",
    "The novel explores themes of memory, loss, and identity.",
    "An array of integers can be sorted with a stable algorithm.",
    "Quantum mechanics describes nature at the smallest scales.",
    "The recipe calls for two cups of flour and one egg.",
    "Economic growth depends on productivity and investment.",
]

EVAL_TEXTS = [
    "Birds migrate south for the winter to find warmer climates.",
    "The telescope revealed rings around the distant planet.",
    "A function is continuous if small changes yield small outputs.",
    "The orchestra tuned their instruments before the concert.",
    "Irrigation canals brought water to the dry farmland.",
    "She calculated the derivative of the polynomial by hand.",
    "The library archived newspapers from the last century.",
    "Machine learning models learn patterns from training data.",
    "The bridge withstood the storm without any structural damage.",
    "Volcanoes form where tectonic plates collide or separate.",
    "He measured the voltage across the resistor with a meter.",
    "The garden blooms with tulips every April and May.",
    "Cryptography protects messages with mathematical hardness.",
    "The pilot adjusted the heading to avoid the turbulence.",
    "A binary tree stores keys in sorted order for fast lookup.",
    "The museum displayed ancient pottery from the Bronze Age.",
]


def tokenize(texts, tokenizer, suffix):
    encoded = []
    for base in texts:
        text = base + suffix
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < SEQ_LEN:
            raise ValueError(f"text too short for seq {SEQ_LEN}: {text!r} ({len(ids)})")
        encoded.append({"text_sha256_hint": len(text), "token_ids": ids[:SEQ_LEN]})
    return encoded


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    import hashlib

    def pack(texts, suffix):
        full = [t + suffix for t in texts]
        cases = tokenize(texts, tokenizer, suffix)
        return {
            "seq_len": SEQ_LEN,
            "cases": [
                {
                    "token_ids": case["token_ids"],
                    "position_ids": list(range(SEQ_LEN)),
                    "text_sha256": hashlib.sha256(full[i].encode()).hexdigest(),
                }
                for i, case in enumerate(cases)
            ],
        }

    TESTDATA.mkdir(parents=True, exist_ok=True)
    for name, texts, suffix in (
        ("qwen3_f1_calibration", CALIBRATION_TEXTS,
         " It continues with several more carefully chosen words here."),
        ("qwen3_f1_eval", EVAL_TEXTS,
         " The passage goes on with additional distinct vocabulary afterwards now."),
    ):
        path = TESTDATA / f"{name}.json"
        path.write_text(json.dumps(pack(texts, suffix), indent=1) + "\n", encoding="utf-8")
        print("written:", path)


if __name__ == "__main__":
    main()
