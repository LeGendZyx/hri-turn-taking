from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hri_turn_taking.legacy.data.dataset import WindowedDataset, load_windowed_dataset
from hri_turn_taking.legacy.data.schema import DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    metrics: dict[str, float]
    model_type: str
    n_train: int
    n_test: int
    positive_rate: float


def train_trp_classifier(
    *,
    dataset_path: str | Path,
    model_path: str | Path,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
    model_type: str = "random_forest",
    window_size: int = 1,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> TrainingResult:
    dataset = load_windowed_dataset(
        dataset_path,
        feature_columns=feature_columns or DEFAULT_FEATURE_COLUMNS,
        target_column=target_column,
        window_size=window_size,
    )
    train_idx, test_idx = _train_test_indices(dataset, test_size=test_size, random_seed=random_seed)

    pipeline = _build_classifier(model_type=model_type, random_seed=random_seed)
    pipeline.fit(dataset.x[train_idx], dataset.y[train_idx])

    predictions = pipeline.predict(dataset.x[test_idx])
    probabilities = _positive_probabilities(pipeline, dataset.x[test_idx])
    metrics = _classification_metrics(dataset.y[test_idx], predictions, probabilities)

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "model_type": model_type,
        "feature_columns": feature_columns or DEFAULT_FEATURE_COLUMNS,
        "window_size": window_size,
        "window_feature_names": dataset.feature_names,
        "target_column": target_column,
        "metrics": metrics,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    joblib.dump(artifact, output_path)

    return TrainingResult(
        model_path=output_path,
        metrics=metrics,
        model_type=model_type,
        n_train=int(train_idx.size),
        n_test=int(test_idx.size),
        positive_rate=float(np.mean(dataset.y)),
    )


def _train_test_indices(
    dataset: WindowedDataset,
    *,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be within (0, 1)")

    indices = np.arange(dataset.y.shape[0])
    unique_groups = np.unique(dataset.groups)
    if unique_groups.size > 1:
        train_groups, test_groups = train_test_split(
            unique_groups,
            test_size=test_size,
            random_state=random_seed,
        )
        train_idx = indices[np.isin(dataset.groups, train_groups)]
        test_idx = indices[np.isin(dataset.groups, test_groups)]
        return train_idx, test_idx

    stratify = dataset.y if np.unique(dataset.y).size == 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def _build_classifier(model_type: str, random_seed: int) -> Pipeline:
    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_seed,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    if model_type == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(random_state=random_seed)
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            "Unsupported model_type. Use one of: random_forest, "
            "hist_gradient_boosting, logistic_regression"
        )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )


def _positive_probabilities(pipeline: Pipeline, x: np.ndarray) -> np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return pipeline.predict_proba(x)[:, 1]
    if hasattr(classifier, "decision_function"):
        scores = pipeline.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    return pipeline.predict(x).astype(float)


def _classification_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    if np.unique(y_true).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return metrics
