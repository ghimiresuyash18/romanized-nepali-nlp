"""Sentiment baseline models for Romanized Nepali text.

Tasks 18 and 19: Majority baseline, word-level TF-IDF, character n-gram TF-IDF,
model serialization, and feature inspection.
"""

from __future__ import annotations

import pickle
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from romanized_nepali_nlp.dataset import ALLOWED_LABELS, DatasetRecord
from romanized_nepali_nlp.pipeline import PipelineConfig, PreprocessingPipeline


@dataclass
class SentimentModelConfig:
    """Configuration for a sentiment model."""

    model_type: str = "char"  # 'majority', 'word', or 'char'
    analyzer: str = "char_wb"  # 'word', 'char', or 'char_wb'
    ngram_range: Tuple[int, int] = (2, 5)
    min_df: int = 1
    max_features: Optional[int] = None
    class_weight: Optional[str] = "balanced"  # 'balanced' or None
    C: float = 1.0
    seed: int = 42
    preprocess: bool = True
    pipeline_config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.model_type not in {"majority", "word", "char"}:
            raise ValueError(f"Unknown model_type: '{self.model_type}'")
        if self.model_type == "word":
            self.analyzer = "word"
            if self.ngram_range == (2, 5):
                self.ngram_range = (1, 2)
        elif self.model_type == "char":
            if self.analyzer not in {"char", "char_wb"}:
                self.analyzer = "char_wb"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ngram_range"] = list(self.ngram_range)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SentimentModelConfig:
        data = dict(data)
        if "ngram_range" in data and isinstance(data["ngram_range"], list):
            data["ngram_range"] = tuple(data["ngram_range"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Prediction:
    """Output prediction for a single input record."""

    id: str
    predicted_label: str
    scores: Dict[str, float]
    gold_label: Optional[str] = None
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "gold_label": self.gold_label,
            "predicted_label": self.predicted_label,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "raw_text": self.raw_text,
        }


class SentimentModel:
    """Unified wrapper for majority, word, and character TF-IDF sentiment models."""

    def __init__(self, config: Optional[SentimentModelConfig] = None) -> None:
        self.config = config or SentimentModelConfig()
        self.classes_: List[str] = sorted(list(ALLOWED_LABELS))
        self.vectorizer_: Optional[TfidfVectorizer] = None
        self.classifier_: Optional[LogisticRegression] = None
        self.majority_label_: Optional[str] = None
        self.class_priors_: Dict[str, float] = {}
        self.dataset_hash_: Optional[str] = None
        self.training_counts_: Dict[str, int] = {}
        self.pipeline: Optional[PreprocessingPipeline] = None

        if self.config.preprocess:
            p_cfg = (
                PipelineConfig(**self.config.pipeline_config)
                if self.config.pipeline_config
                else PipelineConfig()
            )
            self.pipeline = PreprocessingPipeline(p_cfg)

    def _prepare_text(self, text: str) -> str:
        if self.pipeline:
            res = self.pipeline.preprocess(text)
            return res.normalized_text
        return text

    def fit(
        self,
        records: Sequence[DatasetRecord],
        dataset_hash: Optional[str] = None,
    ) -> SentimentModel:
        """Fit model on labeled training records."""
        labeled_records = [r for r in records if r.label and r.label in ALLOWED_LABELS]
        if not labeled_records:
            raise ValueError("No valid labeled records provided for training.")

        self.dataset_hash_ = dataset_hash
        counts = Counter(r.label for r in labeled_records)
        self.training_counts_ = dict(counts)
        self.majority_label_ = max(counts, key=counts.get)  # type: ignore[arg-type]
        total = len(labeled_records)
        self.class_priors_ = {c: counts.get(c, 0) / total for c in self.classes_}

        if self.config.model_type == "majority":
            return self

        # Extract texts and labels
        texts = [self._prepare_text(r.text) for r in labeled_records]
        y = [r.label for r in labeled_records]

        # Fit vectorizer
        self.vectorizer_ = TfidfVectorizer(
            analyzer=self.config.analyzer,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_features=self.config.max_features,
            sublinear_tf=True,
        )
        X = self.vectorizer_.fit_transform(texts)

        # Fit linear classifier
        self.classifier_ = LogisticRegression(
            C=self.config.C,
            class_weight=self.config.class_weight,
            random_state=self.config.seed,
            max_iter=1000,
        )
        self.classifier_.fit(X, y)
        self.classes_ = list(self.classifier_.classes_)
        return self

    def predict_records(self, records: Sequence[DatasetRecord]) -> List[Prediction]:
        """Predict labels and class probability scores for dataset records."""
        predictions: List[Prediction] = []
        if not records:
            return predictions

        if self.config.model_type == "majority":
            assert self.majority_label_ is not None
            for r in records:
                predictions.append(
                    Prediction(
                        id=r.id,
                        gold_label=r.label,
                        predicted_label=self.majority_label_,
                        scores=dict(self.class_priors_),
                        raw_text=r.text,
                    )
                )
            return predictions

        assert self.vectorizer_ is not None and self.classifier_ is not None
        prepared_texts = [self._prepare_text(r.text) for r in records]
        X = self.vectorizer_.transform(prepared_texts)
        probs = self.classifier_.predict_proba(X)
        pred_labels = self.classifier_.predict(X)

        for r, pred_label, prob_dist in zip(records, pred_labels, probs):
            scores = {cls_name: float(prob) for cls_name, prob in zip(self.classes_, prob_dist)}
            predictions.append(
                Prediction(
                    id=r.id,
                    gold_label=r.label,
                    predicted_label=str(pred_label),
                    scores=scores,
                    raw_text=r.text,
                )
            )
        return predictions

    def predict_texts(
        self,
        texts: Sequence[str],
        ids: Optional[Sequence[str]] = None,
    ) -> List[Prediction]:
        """Predict labels and scores for raw strings."""
        records = [
            DatasetRecord(id=ids[i] if ids else f"text_{i}", text=t) for i, t in enumerate(texts)
        ]
        return self.predict_records(records)

    def get_top_features(self, n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Inspect top predictive features per class (Task 19, subtask 7).

        Filters private or sensitive identifiers.
        """
        if self.config.model_type == "majority" or not self.classifier_ or not self.vectorizer_:
            return {}

        feature_names = self.vectorizer_.get_feature_names_out()
        results: Dict[str, List[Tuple[str, float]]] = {}

        # Sensitive patterns check (phone numbers, emails, handles)
        sensitive_pattern = re.compile(r"(@|\b\d{7,}\b|https?://)", re.IGNORECASE)
        is_binary = len(self.classes_) == 2

        for class_idx, class_name in enumerate(self.classes_):
            if is_binary:
                # Binary logistic regression produces coef_ of shape (1, n_features) for classes_[1]
                coefs = -self.classifier_.coef_[0] if class_idx == 0 else self.classifier_.coef_[0]
            else:
                coefs = self.classifier_.coef_[class_idx]
            sorted_indices = coefs.argsort()[::-1]

            top_features: List[Tuple[str, float]] = []
            for idx in sorted_indices:
                feat = feature_names[idx]
                if sensitive_pattern.search(feat):
                    continue
                top_features.append((str(feat), round(float(coefs[idx]), 4)))
                if len(top_features) >= n:
                    break
            results[class_name] = top_features
        return results

    def save(self, path: Path | str) -> None:
        """Save model configuration and trained state to disk.

        Security note: Models are serialized using pickle. Only load model files
        from trusted sources.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "classes": self.classes_,
            "vectorizer": self.vectorizer_,
            "classifier": self.classifier_,
            "majority_label": self.majority_label_,
            "class_priors": self.class_priors_,
            "dataset_hash": self.dataset_hash_,
            "training_counts": self.training_counts_,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path | str) -> SentimentModel:
        """Load saved model artifact from disk.

        Security note: Never load model files from untrusted sources;
        pickle can execute arbitrary code during deserialization.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict) or "config" not in payload:
            raise ValueError(f"Invalid or corrupted model artifact: {path}")

        cfg = SentimentModelConfig.from_dict(payload["config"])
        model = cls(cfg)
        model.classes_ = payload["classes"]
        model.vectorizer_ = payload["vectorizer"]
        model.classifier_ = payload["classifier"]
        model.majority_label_ = payload["majority_label"]
        model.class_priors_ = payload["class_priors"]
        model.dataset_hash_ = payload["dataset_hash"]
        model.training_counts_ = payload.get("training_counts", {})
        return model


def compute_learning_curve(
    train_records: Sequence[DatasetRecord],
    val_records: Sequence[DatasetRecord],
    config: Optional[SentimentModelConfig] = None,
    train_fractions: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Experiment: measure train and validation performance across sizes (Task 18, subtask 7)."""
    import random

    from sklearn.metrics import f1_score

    fractions = train_fractions or [0.2, 0.4, 0.6, 0.8, 1.0]
    labeled_train = [r for r in train_records if r.label]
    labeled_val = [r for r in val_records if r.label]

    if not labeled_train or not labeled_val:
        return []

    curve: List[Dict[str, Any]] = []
    base_cfg = config or SentimentModelConfig(model_type="char")

    # Group by label to ensure representative classes across fractions
    by_label: Dict[str, List[DatasetRecord]] = {}
    for r in labeled_train:
        by_label.setdefault(r.label, []).append(r)

    rng = random.Random(base_cfg.seed)
    for lbl in by_label:
        rng.shuffle(by_label[lbl])

    for frac in fractions:
        subset: List[DatasetRecord] = []
        for lbl, recs in by_label.items():
            k = max(1, int(round(len(recs) * frac)))
            subset.extend(recs[:k])

        if len(set(r.label for r in subset)) < 2:
            continue

        model = SentimentModel(base_cfg).fit(subset)

        # Train macro-F1
        train_preds = model.predict_records(subset)
        train_gold = [p.gold_label for p in train_preds]
        train_pred = [p.predicted_label for p in train_preds]
        train_f1 = float(f1_score(train_gold, train_pred, average="macro", zero_division=0))

        # Val macro-F1
        val_preds = model.predict_records(labeled_val)
        val_gold = [p.gold_label for p in val_preds]
        val_pred = [p.predicted_label for p in val_preds]
        val_f1 = float(f1_score(val_gold, val_pred, average="macro", zero_division=0))

        curve.append(
            {
                "fraction": frac,
                "train_samples": len(subset),
                "train_macro_f1": round(train_f1, 4),
                "val_macro_f1": round(val_f1, 4),
            }
        )
    return curve
