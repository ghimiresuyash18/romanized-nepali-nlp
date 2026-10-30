"""Unit tests for Romanized Nepali spelling variant mining."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from romanized_nepali_nlp.variant_mining import (
    VariantMiner,
    save_candidates,
)


def test_mine_candidates_elongation():
    miner = VariantMiner()
    vocab = Counter({
        "ramro": 100,
        "ramrooo": 15,
        "dherai": 50,
        "dheraiii": 8,
        "kura": 20,
    })

    candidates = miner.mine_candidates(vocab, min_canonical_freq=10)
    surfaces = {c.surface_form for c in candidates}

    assert "ramrooo" in surfaces
    assert "dheraiii" in surfaces
    c = next(c for c in candidates if c.surface_form == "ramrooo")
    assert c.canonical_form == "ramro"
    assert c.category == "elongation"
    assert c.confidence >= 0.90


def test_mine_candidates_phonetic():
    miner = VariantMiner()
    vocab = Counter({
        "cha": 200,
        "chha": 40,
        "bhayo": 150,
        "vayo": 30,
    })

    candidates = miner.mine_candidates(vocab, min_canonical_freq=10)
    pairs = {(c.surface_form, c.canonical_form) for c in candidates}

    assert ("chha", "cha") in pairs
    assert ("vayo", "bhayo") in pairs


def test_save_candidates(tmp_path: Path):
    miner = VariantMiner()
    vocab = Counter({"ramro": 50, "ramrooo": 10})
    candidates = miner.mine_candidates(vocab, min_canonical_freq=10)

    out_file = tmp_path / "candidates.jsonl"
    save_candidates(candidates, out_file)

    assert out_file.exists()
    with open(out_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == len(candidates)
    assert rows[0]["surface_form"] == "ramrooo"
