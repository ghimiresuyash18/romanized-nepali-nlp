"""Leakage-safe, reproducible train/validation/test splitting.

Task 16: deterministic splits, group-aware deduplication, stratification,
split manifest, and pre-training leakage check.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from romanized_nepali_nlp.dataset import DatasetRecord
from romanized_nepali_nlp.quality import DuplicateReport, detect_duplicates

SPLIT_NAMES = ("train", "validation", "test")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SplitConfig:
    """Configuration for a leakage-safe dataset split."""

    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15
    seed: int = 42
    stratify_by_label: bool = True
    # If True, all members of a duplicate group land in the same split
    group_by_duplicates: bool = True

    def __post_init__(self) -> None:
        total = round(self.train + self.validation + self.test, 6)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total} "
                f"(train={self.train}, validation={self.validation}, test={self.test})"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "seed": self.seed,
            "stratify_by_label": self.stratify_by_label,
            "group_by_duplicates": self.group_by_duplicates,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SplitConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Result and manifest
# ---------------------------------------------------------------------------


@dataclass
class SplitResult:
    """Outcome of a split operation."""

    train: List[DatasetRecord]
    validation: List[DatasetRecord]
    test: List[DatasetRecord]
    config: SplitConfig
    dataset_hash: str  # SHA-256 of sorted record IDs to version the input dataset

    def by_name(self, name: str) -> List[DatasetRecord]:
        return {"train": self.train, "validation": self.validation, "test": self.test}[name]

    def to_manifest(self) -> Dict[str, Any]:
        """Return a JSON-serialisable split manifest (Task 16, subtask 5)."""
        label_dist = {}
        for split_name in SPLIT_NAMES:
            recs = self.by_name(split_name)
            from collections import Counter

            label_dist[split_name] = dict(Counter(r.label for r in recs))

        return {
            "dataset_hash": self.dataset_hash,
            "config": self.config.to_dict(),
            "counts": {
                "train": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
            },
            "label_distribution": label_dist,
            "record_ids": {
                "train": [r.id for r in self.train],
                "validation": [r.id for r in self.validation],
                "test": [r.id for r in self.test],
            },
        }

    def save_manifest(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(self.to_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dataset_hash(records: Sequence[DatasetRecord]) -> str:
    """Stable hash of sorted record IDs to fingerprint the dataset version."""
    joined = "\n".join(sorted(r.id for r in records))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _assign_splits(
    items: List[Any],
    ratios: Tuple[float, float, float],
    rng: random.Random,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Shuffle and split a list into (train, validation, test) by ratios."""
    rng.shuffle(items)
    n = len(items)
    n_train = round(ratios[0] * n)
    n_val = round(ratios[1] * n)
    return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split(
    records: Sequence[DatasetRecord],
    config: Optional[SplitConfig] = None,
    dup_report: Optional[DuplicateReport] = None,
) -> SplitResult:
    """Produce a leakage-safe, reproducible train/validation/test split.

    Algorithm:
      1. Deduplicate (if group_by_duplicates=True): build groups where all
         members of a duplicate group are treated as a single unit.
      2. Stratify by label (if stratify_by_label=True).
      3. Apply deterministic shuffle with the configured seed.
      4. Assign split; duplicate-group members always land together.
    """
    config = config or SplitConfig()
    records = list(records)
    rng = random.Random(config.seed)
    ratios = (config.train, config.validation, config.test)
    dataset_hash = _dataset_hash(records)

    if config.group_by_duplicates:
        if dup_report is None:
            dup_report = detect_duplicates(records)
        # Compute transitive closure using union-find across overlapping duplicate groups
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                if rx < ry:
                    parent[ry] = rx
                else:
                    parent[rx] = ry

        for grp in dup_report.raw_groups + dup_report.normalized_groups:
            if grp.record_ids:
                first = grp.record_ids[0]
                for rid in grp.record_ids[1:]:
                    union(first, rid)

        # Group units: records with same connected component stay together
        from collections import defaultdict

        unit_map: Dict[str, List[DatasetRecord]] = defaultdict(list)
        for rec in records:
            rep = find(rec.id)
            unit_map[rep].append(rec)
        units: List[List[DatasetRecord]] = list(unit_map.values())
    else:
        # Each record is its own unit
        units = [[r] for r in records]

    # Stratify by label of the first record in each unit (representative)
    if config.stratify_by_label:
        label_units: Dict[Optional[str], List[List[DatasetRecord]]] = {}
        for unit in units:
            lbl = unit[0].label
            label_units.setdefault(lbl, []).append(unit)

        train_units: List[List[DatasetRecord]] = []
        val_units: List[List[DatasetRecord]] = []
        test_units: List[List[DatasetRecord]] = []

        for lbl_units in label_units.values():
            tr, va, te = _assign_splits(lbl_units, ratios, rng)
            train_units.extend(tr)
            val_units.extend(va)
            test_units.extend(te)
    else:
        train_units, val_units, test_units = _assign_splits(units, ratios, rng)

    def _flatten_and_tag(
        unit_list: List[List[DatasetRecord]], split_name: str
    ) -> List[DatasetRecord]:
        flat = []
        for unit in unit_list:
            for rec in unit:
                flat.append(replace(rec, split=split_name))
        return flat

    return SplitResult(
        train=_flatten_and_tag(train_units, "train"),
        validation=_flatten_and_tag(val_units, "validation"),
        test=_flatten_and_tag(test_units, "test"),
        config=config,
        dataset_hash=dataset_hash,
    )


def check_leakage(result: SplitResult) -> List[str]:
    """Return a list of record IDs that appear in more than one split.

    An empty list means the split is leak-free. Raises nothing; callers
    decide whether to treat leakage as fatal.
    """
    seen: Dict[str, List[str]] = {}
    for split_name in SPLIT_NAMES:
        for rec in result.by_name(split_name):
            seen.setdefault(rec.id, []).append(split_name)

    leaked = [rid for rid, splits in seen.items() if len(splits) > 1]
    return sorted(leaked)


def split_balance_report(result: SplitResult) -> str:
    """Return a Markdown table comparing label distributions across splits."""
    from collections import Counter

    labels = sorted({r.label for r in result.train + result.validation + result.test if r.label})
    lines = ["## Split Balance Report", ""]
    header = "| Label | Train | Validation | Test |"
    sep = "|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    counters = {
        "train": Counter(r.label for r in result.train),
        "validation": Counter(r.label for r in result.validation),
        "test": Counter(r.label for r in result.test),
    }
    for lbl in labels:
        tr = counters["train"][lbl]
        va = counters["validation"][lbl]
        te = counters["test"][lbl]
        lines.append(f"| {lbl} | {tr} | {va} | {te} |")

    totals = (len(result.train), len(result.validation), len(result.test))
    lines.append(f"| **Total** | {totals[0]} | {totals[1]} | {totals[2]} |")
    lines.append("")
    lines.append(f"Dataset hash: `{result.dataset_hash}`  ")
    lines.append(f"Seed: `{result.config.seed}`")
    return "\n".join(lines)
