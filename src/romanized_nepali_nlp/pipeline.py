"""Composable, deterministic preprocessing pipeline for Romanized Nepali text."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from romanized_nepali_nlp.normalization import (
    mask_protected_spans,
    normalize_casing,
    normalize_unicode,
    normalize_whitespace,
    reduce_repetitions,
    restore_protected_spans,
)
from romanized_nepali_nlp.tokenizer import Tokenizer
from romanized_nepali_nlp.types import PreprocessingResult, ProtectedSpan
from romanized_nepali_nlp.variants import VariantNormalizer

VALID_FAILURE_POLICIES = {"fail_fast", "skip_with_warning", "keep_original"}


@dataclass
class PipelineConfig:
    """Configuration options for text preprocessing."""

    unicode_form: str = "NFC"
    strip_zero_width: bool = True
    collapse_whitespace: bool = True
    preserve_newlines: bool = True
    lowercase: bool = False
    reduce_repetitions: bool = False
    max_repeats: int = 2
    normalize_variants: bool = True
    variant_policy: str = "conservative"
    protected_modes: Dict[str, str] = field(
        default_factory=lambda: {
            "URL": "preserve",
            "EMAIL": "preserve",
            "MENTION": "preserve",
            "HASHTAG": "preserve",
            "EMOJI": "preserve",
            "NUMBER": "preserve",
        }
    )
    failure_policy: str = "skip_with_warning"  # 'fail_fast', 'skip_with_warning', 'keep_original'

    def __post_init__(self) -> None:
        if self.failure_policy not in VALID_FAILURE_POLICIES:
            raise ValueError(
                f"Invalid failure_policy: {self.failure_policy!r}. "
                f"Must be one of {sorted(VALID_FAILURE_POLICIES)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        return cls(**data)

    def fingerprint(self) -> str:
        """Compute deterministic SHA-256 hash of the configuration."""
        canonical_json = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]

    def save(self, path: Path | str) -> None:
        """Persist configuration to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path | str) -> PipelineConfig:
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class PreprocessingPipeline:
    """Orchestrates ordered normalization steps while strictly preserving raw text."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.tokenizer = tokenizer or Tokenizer(validate_spans=False)
        self.variant_normalizer: Optional[VariantNormalizer] = None

        if self.config.normalize_variants:
            self.variant_normalizer = VariantNormalizer(policy=self.config.variant_policy)

    def preprocess(self, text: str) -> PreprocessingResult:
        """Run configured preprocessing pipeline on a single text string."""
        raw_text = text
        steps: List[str] = []
        warnings: List[str] = []
        protected_spans: List[ProtectedSpan] = []

        if not text:
            return PreprocessingResult(
                raw_text=raw_text,
                normalized_text="",
                steps=steps,
                warnings=warnings,
                protected_spans=protected_spans,
                metadata={"pipeline_fingerprint": self.config.fingerprint()},
            )

        try:
            curr_text = text

            # 1. Mask protected spans (URLs, Emojis, Mentions, etc.)
            curr_text, protected_spans, placeholder_map = mask_protected_spans(
                curr_text,
                tokenizer=self.tokenizer,
                modes=self.config.protected_modes,
            )
            if placeholder_map:
                steps.append("protect_special_tokens")

            # 2. Unicode normalization
            if self.config.unicode_form:
                curr_text, uni_steps, uni_warn = normalize_unicode(
                    curr_text,
                    form=self.config.unicode_form,
                    strip_zero_width=self.config.strip_zero_width,
                )
                steps.extend(uni_steps)
                warnings.extend(uni_warn)

            # 3. Repetition reduction (optional)
            if self.config.reduce_repetitions:
                curr_text, rep_changed = reduce_repetitions(
                    curr_text,
                    max_repeats=self.config.max_repeats,
                )
                if rep_changed:
                    steps.append("reduce_repetitions")

            # 4. Spelling variants normalization
            applied_variants: List[Dict[str, Any]] = []
            if self.variant_normalizer is not None:
                curr_text, applied_variants = self.variant_normalizer.normalize_text(curr_text)
                if applied_variants:
                    steps.append(f"normalize_variants_{self.config.variant_policy}")

            # 5. Casing normalization (optional, disabled by default)
            if self.config.lowercase:
                curr_text, case_changed = normalize_casing(curr_text, lowercase=True)
                if case_changed:
                    steps.append("lowercase")

            # 6. Whitespace normalization
            if self.config.collapse_whitespace:
                curr_text, ws_changed = normalize_whitespace(
                    curr_text,
                    preserve_newlines=self.config.preserve_newlines,
                )
                if ws_changed:
                    steps.append("normalize_whitespace")

            # 7. Restore or format protected spans
            if placeholder_map:
                curr_text, protected_spans = restore_protected_spans(
                    curr_text,
                    placeholder_map,
                    modes=self.config.protected_modes,
                    return_spans=True,
                )
                steps.append("restore_special_tokens")

            metadata: Dict[str, Any] = {
                "pipeline_fingerprint": self.config.fingerprint(),
                "char_delta": len(curr_text) - len(raw_text),
                "applied_variants_count": len(applied_variants),
            }
            if applied_variants:
                metadata["applied_variants"] = [
                    {"from": v["original"], "to": v["canonical"]} for v in applied_variants
                ]

            return PreprocessingResult(
                raw_text=raw_text,
                normalized_text=curr_text,
                steps=steps,
                warnings=warnings,
                protected_spans=protected_spans,
                metadata=metadata,
            )

        except Exception as exc:
            if self.config.failure_policy == "fail_fast":
                raise exc
            if self.config.failure_policy == "skip_with_warning":
                warnings.append(f"Preprocessing error: {exc}")
                return PreprocessingResult(
                    raw_text=raw_text,
                    normalized_text=raw_text,
                    steps=["fallback_keep_original"],
                    warnings=warnings,
                    protected_spans=[],
                    metadata={"error": str(exc)},
                )
            # Default keep_original
            return PreprocessingResult(
                raw_text=raw_text,
                normalized_text=raw_text,
                steps=["fallback_keep_original"],
                warnings=[str(exc)],
                protected_spans=[],
                metadata={"error": str(exc)},
            )

    def preprocess_batch(self, texts: List[str]) -> List[PreprocessingResult]:
        """Process a list of texts sequentially."""
        return [self.preprocess(t) for t in texts]
