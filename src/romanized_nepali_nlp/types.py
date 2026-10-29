"""Core data types and contracts for the Romanized Nepali NLP toolkit.

All span offsets (start and end) use Python Unicode code-point positions,
ensuring text[start:end] == token.text for all Unicode and emoji characters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TokenCategory(str, Enum):
    """Supported token classification categories."""

    WORD = "WORD"
    ENGLISH_WORD = "ENGLISH_WORD"
    URL = "URL"
    EMAIL = "EMAIL"
    MENTION = "MENTION"
    HASHTAG = "HASHTAG"
    NUMBER = "NUMBER"
    EMOJI = "EMOJI"
    PUNCTUATION = "PUNCTUATION"
    UNKNOWN = "UNKNOWN"


class Severity(str, Enum):
    """Validation issue severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ProtectedSpan:
    """Span of text marked to be protected during text normalization."""

    start: int
    end: int
    category: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProtectedSpan:
        return cls(
            start=int(data["start"]),
            end=int(data["end"]),
            category=str(data["category"]),
            text=str(data["text"]),
        )


@dataclass
class Token:
    """Individual token with code-point offsets and category."""

    text: str
    category: TokenCategory
    start: int
    end: int
    normalized_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["category"] = self.category.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Token:
        category = TokenCategory(data["category"])
        return cls(
            text=data["text"],
            category=category,
            start=int(data["start"]),
            end=int(data["end"]),
            normalized_text=data.get("normalized_text"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class PreprocessingResult:
    """Result of running text normalization while preserving raw inputs."""

    raw_text: str
    normalized_text: str
    steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    protected_spans: List[ProtectedSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "steps": list(self.steps),
            "warnings": list(self.warnings),
            "protected_spans": [span.to_dict() for span in self.protected_spans],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PreprocessingResult:
        protected = [ProtectedSpan.from_dict(span) for span in data.get("protected_spans", [])]
        return cls(
            raw_text=data["raw_text"],
            normalized_text=data["normalized_text"],
            steps=list(data.get("steps", [])),
            warnings=list(data.get("warnings", [])),
            protected_spans=protected,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class DatasetRecord:
    """Standardized representation of a single corpus or dataset record."""

    id: str
    text: str
    label: Optional[str] = None
    source: Optional[str] = None
    split: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetRecord:
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            label=data.get("label"),
            source=data.get("source"),
            split=data.get("split"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ValidationIssue:
    """Issue detected during dataset validation or quality check."""

    severity: Severity
    message: str
    row_id: Optional[str] = None
    field: Optional[str] = None
    category: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["severity"] = self.severity.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationIssue:
        return cls(
            severity=Severity(data["severity"]),
            message=str(data["message"]),
            row_id=data.get("row_id"),
            field=data.get("field"),
            category=data.get("category", "general"),
        )


@dataclass
class Prediction:
    """Model output for a given record, including probabilities and model version."""

    record_id: str
    predicted_label: str
    class_scores: Dict[str, float]
    gold_label: Optional[str] = None
    model_version: str = "0.1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Prediction:
        return cls(
            record_id=str(data["record_id"]),
            predicted_label=str(data["predicted_label"]),
            class_scores={k: float(v) for k, v in data.get("class_scores", {}).items()},
            gold_label=data.get("gold_label"),
            model_version=str(data.get("model_version", "0.1.0")),
            metadata=dict(data.get("metadata", {})),
        )
