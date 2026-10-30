"""External corpus ingestion, profiling, quality triage, and leakage control.

Phase 5.5: Integration pipeline for external Romanized Nepali corpora
(specifically Boredoom17/Nepali-Flow-Roman).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from romanized_nepali_nlp.dataset import DatasetRecord
from romanized_nepali_nlp.normalization import normalize_whitespace

RE_REPEATS = re.compile(r"([a-zA-Z])\1{2,}")

URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
MENTION_REGEX = re.compile(r"@\w+", re.IGNORECASE)
HASHTAG_REGEX = re.compile(r"#\w+", re.IGNORECASE)
EMOJI_REGEX = re.compile(r"[\U00010000-\U0010ffff\u2600-\u27bf]|(:\)|:\(|;-\)|;\)|:D|:P|<3)")


def compute_latin_ratio(text: str) -> float:
    """Calculate ratio of Latin alphabet letters to total alphabetical characters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    latin_count = sum(
        1
        for c in letters
        if (65 <= ord(c) <= 90)
        or (97 <= ord(c) <= 122)
        or (0x00C0 <= ord(c) <= 0x024F)
        or (0x1E00 <= ord(c) <= 0x1EFF)
    )
    return latin_count / len(letters)


def compute_non_ascii_ratio(text: str) -> float:
    """Calculate ratio of non-ASCII characters to total string length."""
    if not text:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text)


@dataclass
class CorpusManifest:
    """Provenance and metadata for an external corpus snapshot."""

    dataset: str
    revision: str
    downloaded_at: str
    rows_downloaded: int
    source_license_field: str
    upstream_attribution: str
    local_sha256: str
    processing_version: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> CorpusManifest:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**data)


@dataclass
class RowProfile:
    """Structural profile of a single raw corpus record."""

    id: str
    raw_text: str
    char_count: int
    word_count: int
    latin_ratio: float
    non_ascii_ratio: float
    has_url: bool
    has_email: bool
    has_mention: bool
    has_hashtag: bool
    emoji_count: int
    has_repeats: bool
    is_empty: bool
    partition: str
    triage_reason: str
    raw_hash: str
    upstream_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def profile_and_triage_row(
    raw_row: Dict[str, Any],
    index: int,
    id_prefix: str = "nfr",
) -> RowProfile:
    """Profile structural characteristics and assign a quality partition."""
    text = str(raw_row.get("text") or "")
    text_clean = text.strip()
    is_empty = not bool(text_clean)
    char_count = len(text)
    words = text_clean.split()
    word_count = len(words)
    latin_rat = compute_latin_ratio(text)
    non_ascii_rat = compute_non_ascii_ratio(text)

    has_url = bool(URL_REGEX.search(text))
    has_email = bool(EMAIL_REGEX.search(text))
    has_mention = bool(MENTION_REGEX.search(text))
    has_hashtag = bool(HASHTAG_REGEX.search(text))
    emojis = EMOJI_REGEX.findall(text)
    emoji_count = len(emojis)
    has_repeats = bool(RE_REPEATS.search(text))
    raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # Upstream metadata preservation
    meta = {
        "upstream_source": raw_row.get("source"),
        "upstream_domain": raw_row.get("domain"),
        "upstream_script": raw_row.get("script"),
        "upstream_lang": raw_row.get("lang"),
        "date_collected": raw_row.get("date_collected"),
        "upstream_license": raw_row.get("license"),
        "upstream_row_index": index,
    }

    # Triage Decision Logic
    partition = "roman_clean"
    reason = "Target Romanized Nepali text"

    if is_empty:
        partition = "reject_or_review"
        reason = "Empty or whitespace-only content"
    elif word_count < 2 and char_count < 3:
        partition = "reject_or_review"
        reason = "Single letter or uninformative non-word"
    elif latin_rat < 0.30:
        partition = "reject_or_review"
        reason = (
            f"Low Latin alphabet ratio ({latin_rat:.2f} < 0.30); predominantly non-Latin script"
        )
    elif char_count > 4000 or word_count > 500:
        partition = "reject_or_review"
        reason = "Excessive text length (>4000 chars); pasted lyrics or spam boilerplate"
    elif (
        has_repeats
        or emoji_count >= 5
        or has_url
        or (word_count > 0 and emoji_count / word_count > 0.5)
    ):
        partition = "roman_noisy"
        reason = "Noisy formatting: repeated characters, URL presence, or high emoji density"
    elif any(
        w.lower() in {"the", "and", "this", "video", "subscribe", "channel", "song", "like"}
        for w in words
    ):
        partition = "roman_mixed"
        reason = "Code-switched Romanized Nepali with English vocabulary"

    rec_id = f"{id_prefix}-{index:07d}"
    return RowProfile(
        id=rec_id,
        raw_text=text,
        char_count=char_count,
        word_count=word_count,
        latin_ratio=round(latin_rat, 4),
        non_ascii_ratio=round(non_ascii_rat, 4),
        has_url=has_url,
        has_email=has_email,
        has_mention=has_mention,
        has_hashtag=has_hashtag,
        emoji_count=emoji_count,
        has_repeats=has_repeats,
        is_empty=is_empty,
        partition=partition,
        triage_reason=reason,
        raw_hash=raw_hash,
        upstream_meta=meta,
    )


def row_to_dataset_record(profile: RowProfile) -> DatasetRecord:
    """Convert RowProfile to standard DatasetRecord."""
    return DatasetRecord(
        id=profile.id,
        text=profile.raw_text,
        label=None,
        source="Nepali-Flow-Roman",
        split=profile.partition,
        metadata={
            "triage_partition": profile.partition,
            "triage_reason": profile.triage_reason,
            "char_count": profile.char_count,
            "word_count": profile.word_count,
            "latin_ratio": profile.latin_ratio,
            **profile.upstream_meta,
        },
    )


def audit_corpus_against_benchmark(
    profiles: Sequence[RowProfile],
    benchmark_records: Sequence[DatasetRecord],
) -> Tuple[List[RowProfile], List[Dict[str, Any]]]:
    """Identify and quarantine any corpus records that duplicate benchmark splits (Task Step 7).

    Returns (safe_profiles, quarantined_records).
    """
    bench_raw_hashes: Set[str] = set()
    bench_norm_hashes: Set[str] = set()
    bench_map: Dict[str, str] = {}

    for r in benchmark_records:
        r_hash = hashlib.sha256(r.text.encode("utf-8")).hexdigest()
        bench_raw_hashes.add(r_hash)
        bench_map[r_hash] = r.id

        norm_t, _ = normalize_whitespace(r.text.lower())
        n_hash = hashlib.sha256(norm_t.encode("utf-8")).hexdigest()
        bench_norm_hashes.add(n_hash)
        bench_map[n_hash] = r.id

    safe_profiles: List[RowProfile] = []
    quarantined: List[Dict[str, Any]] = []

    for prof in profiles:
        r_hash = hashlib.sha256(prof.raw_text.encode("utf-8")).hexdigest()
        norm_t, _ = normalize_whitespace(prof.raw_text.lower())
        n_hash = hashlib.sha256(norm_t.encode("utf-8")).hexdigest()

        matched_bench_id = bench_map.get(r_hash) or bench_map.get(n_hash)
        if matched_bench_id:
            # Overlap detected: force into reject_or_review to protect benchmark
            prof.partition = "reject_or_review"
            prof.triage_reason = f"Benchmark overlap with ID {matched_bench_id}"
            quarantined.append(
                {
                    "corpus_id": prof.id,
                    "text": prof.raw_text,
                    "matched_benchmark_id": matched_bench_id,
                    "quarantine_reason": "benchmark_test_leakage_protection",
                }
            )
        else:
            safe_profiles.append(prof)

    return safe_profiles, quarantined


def process_corpus_file(
    input_path: Path | str,
    output_dir: Path | str,
    benchmark_records: Optional[Sequence[DatasetRecord]] = None,
    manifest: Optional[CorpusManifest] = None,
) -> Dict[str, Any]:
    """Ingest, profile, audit, and partition external corpus file (Phase 5.5)."""
    in_p = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    if in_p.suffix.lower() == ".jsonl":
        with open(in_p, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    rows.append(json.loads(line_str))
    elif in_p.suffix.lower() == ".csv":
        import csv

        with open(in_p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    else:
        # Plain text lines
        with open(in_p, "r", encoding="utf-8") as f:
            for line in f:
                rows.append({"text": line.rstrip("\r\n")})

    profiles = [profile_and_triage_row(r, i) for i, r in enumerate(rows)]

    quarantined: List[Dict[str, Any]] = []
    if benchmark_records:
        _, quarantined = audit_corpus_against_benchmark(profiles, benchmark_records)

    # Save manifest if provided
    if manifest:
        manifest.save(out_dir / "manifest.json")

    # Write profile JSONL
    profile_path = out_dir / "profile.jsonl"
    with open(profile_path, "w", encoding="utf-8") as f:
        for p in profiles:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    # Partition buckets
    partitions: Dict[str, List[RowProfile]] = {
        "roman_clean": [],
        "roman_mixed": [],
        "roman_noisy": [],
        "reject_or_review": [],
    }
    for p in profiles:
        partitions.setdefault(p.partition, []).append(p)

    for part_name, part_profiles in partitions.items():
        part_file = out_dir / f"{part_name}.jsonl"
        with open(part_file, "w", encoding="utf-8") as f:
            for p in part_profiles:
                rec = row_to_dataset_record(p)
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    # Write quarantined leakage records
    quar_path = out_dir / "leakage_quarantine.jsonl"
    with open(quar_path, "w", encoding="utf-8") as f:
        for q in quarantined:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Counts
    summary = {
        "total_rows": len(profiles),
        "partitions": {k: len(v) for k, v in partitions.items()},
        "leakage_quarantined": len(quarantined),
    }

    tot_prof = len(profiles)
    pct_clean = (len(partitions["roman_clean"]) / tot_prof * 100) if tot_prof else 0.0
    pct_mixed = (len(partitions["roman_mixed"]) / tot_prof * 100) if tot_prof else 0.0
    pct_noisy = (len(partitions["roman_noisy"]) / tot_prof * 100) if tot_prof else 0.0
    pct_reject = (len(partitions["reject_or_review"]) / tot_prof * 100) if tot_prof else 0.0

    # Generate Markdown profile report
    report_lines = [
        "# External Corpus Quality Profile & Triage Report",
        "",
        f"- **Source File**: `{in_p.resolve()}`",
        f"- **Total Rows Ingested**: **{len(profiles)}**",
        f"- **Benchmark Overlap Quarantined**: **{len(quarantined)}**",
        "",
        "## Quality Partitions Breakdown",
        "",
        "| Partition | Count | Percentage | Description |",
        "|---|---|---|---|",
        (
            f"| `roman_clean` | **{len(partitions['roman_clean'])}** | "
            f"{pct_clean:.1f}% | "
            "Likely target Romanized Nepali text |"
        ),
        (
            f"| `roman_mixed` | **{len(partitions['roman_mixed'])}** | "
            f"{pct_mixed:.1f}% | "
            "Code-switched with English vocabulary |"
        ),
        (
            f"| `roman_noisy` | **{len(partitions['roman_noisy'])}** | "
            f"{pct_noisy:.1f}% | "
            "Noisy formatting: repeated characters or emojis |"
        ),
        (
            f"| `reject_or_review` | **{len(partitions['reject_or_review'])}** | "
            f"{pct_reject:.1f}% | "
            "Non-Latin script, empty, or benchmark overlap |"
        ),
        "",
        "```mermaid",
        "xychart-beta",
        '    title "Corpus Partition Distribution"',
        '    x-axis ["roman_clean", "roman_mixed", "roman_noisy", "reject_or_review"]',
        f'    y-axis "Count" 0 --> {max(len(v) for v in partitions.values()) + 2}',
        (
            "    bar ["
            f"{len(partitions['roman_clean'])}, "
            f"{len(partitions['roman_mixed'])}, "
            f"{len(partitions['roman_noisy'])}, "
            f"{len(partitions['reject_or_review'])}"
            "]"
        ),
        "```",
    ]
    (out_dir / "profile_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return summary
