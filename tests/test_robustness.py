
"""
Sprint 17 - MLOps Final Project
Tests: tests/test_robustness.py

Tests for src/models/robustness.py.

Architecture:
    models/reference/
        best_model_pipeline.joblib
        best_model_metadata.json

    models/runs/<execution_id>/
        Isolated training executions

These tests use temporary artifacts and synthetic datasets.

Run:
    python -m pytest tests/test_robustness.py -v
"""

from argparse import Namespace
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models import robustness


# ============================================================
# 1. Fixtures
# ============================================================

@pytest.fixture
def synthetic_data():
    """
    Create deterministic binary classification data.

    The target is based on a combination of two features.
    """

    rng = np.random.default_rng(42)

    X = pd.DataFrame({
        "age": rng.integers(20, 65, size=120),
        "hours_per_week": rng.integers(
            20, 60, size=120
        ),
    })

    signal = (
        0.12 * X["age"]
        + 0.15 * X["hours_per_week"]
    )

    y = pd.Series(
        (signal > signal.median()).astype(int),
        name="income",
    )

    return X, y


@pytest.fixture
def trained_model(synthetic_data):
    """Train a small classifier for isolated tests."""

    X, y = synthetic_data

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X, y)

    return model


@pytest.fixture
def reference_artifacts(
    tmp_path,
    trained_model,
):
    """
    Create temporary reference model artifacts.

    No files in the project's models/reference directory
    are accessed or modified.
    """

    reference_dir = (
        tmp_path / "models" / "reference"
    )
    reference_dir.mkdir(parents=True)

    model_path = (
        reference_dir / "best_model_pipeline.joblib"
    )
    metadata_path = (
        reference_dir / "best_model_metadata.json"
    )

    joblib.dump(trained_model, model_path)

    metadata = {
        "model_family": "logistic_regression",
        "preprocessing_variant": "baseline",
        "decision_threshold": 0.40,
        "model_sha256": robustness.file_sha256(
            model_path
        ),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "directory": reference_dir,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


@pytest.fixture
def scenario_datasets(
    tmp_path,
    synthetic_data,
):
    """
    Generate four labeled evaluation datasets.

    All scenarios contain the same number of rows.
    The feature distributions differ across scenarios.
    """

    X, y = synthetic_data

    drift_dir = tmp_path / "data" / "drift"

    rng = np.random.default_rng(123)

    shifts = {
        "control": 0,
        "mild": 2,
        "moderate": 6,
        "severe": 12,
    }

    paths = {}

    for scenario, shift in shifts.items():
        scenario_dir = drift_dir / scenario
        scenario_dir.mkdir(parents=True)

        X_current = X.copy()

        if shift:
            X_current["age"] = (
                X_current["age"]
                + rng.integers(
                    -shift,
                    shift + 1,
                    size=len(X_current),
                )
            )

        features_path = (
            scenario_dir / "X_test.csv"
        )
        target_path = (
            scenario_dir / "y_test.csv"
        )

        X_current.to_csv(
            features_path,
            index=False,
        )

        y.to_frame().to_csv(
            target_path,
            index=False,
        )

        paths[scenario] = (
            features_path,
            target_path,
        )

    return paths


@pytest.fixture
def robustness_args(
    tmp_path,
    reference_artifacts,
    scenario_datasets,
):
    """Build command-line arguments for orchestration."""

    args = {
        "model": reference_artifacts["model_path"],
        "metadata": reference_artifacts[
            "metadata_path"
        ],
        "output_dir": (
            tmp_path / "reports" / "robustness"
        ),
    }

    for scenario in robustness.SCENARIOS:
        features, target = scenario_datasets[
            scenario
        ]

        args[scenario] = features
        args[f"{scenario}_target"] = target

    return Namespace(**args)


# ============================================================
# 2. Configuration
# ============================================================

def test_default_model_paths_use_reference():
    """Default artifacts must come from models/reference."""

    assert robustness.DEFAULT_MODEL == (
        robustness.ROOT
        / "models"
        / "reference"
        / "best_model_pipeline.joblib"
    )

    assert robustness.DEFAULT_METADATA == (
        robustness.ROOT
        / "models"
        / "reference"
        / "best_model_metadata.json"
    )


def test_scenario_configuration():
    """All required scenarios must be configured."""

    assert robustness.SCENARIOS == (
        "control",
        "mild",
        "moderate",
        "severe",
    )


# ============================================================
# 3. SHA-256 integrity
# ============================================================

def test_file_sha256_matches_hashlib(tmp_path):
    """File fingerprint must match the standard SHA-256."""

    import hashlib

    path = tmp_path / "sample.bin"
    content = b"Sprint 17 robustness test"

    path.write_bytes(content)

    expected = hashlib.sha256(
        content
    ).hexdigest()

    assert robustness.file_sha256(path) == expected


def test_file_sha256_detects_changes(tmp_path):
    """Changing a file must change its fingerprint."""

    path = tmp_path / "sample.bin"

    path.write_bytes(b"original")
    original_hash = robustness.file_sha256(path)

    path.write_bytes(b"modified")
    modified_hash = robustness.file_sha256(path)

    assert original_hash != modified_hash


# ============================================================
# 4. Reference model loading
# ============================================================

def test_load_reference_model(
    reference_artifacts,
):
    """Load a valid model, metadata, and threshold."""

    model, metadata, threshold = (
        robustness.load_reference_model(
            reference_artifacts["model_path"],
            reference_artifacts["metadata_path"],
        )
    )

    assert callable(model.predict_proba)
    assert threshold == pytest.approx(0.40)
    assert metadata["preprocessing_variant"] == (
        "baseline"
    )


def test_load_reference_preserves_files(
    reference_artifacts,
):
    """Loading must not modify reference artifacts."""

    model_path = reference_artifacts["model_path"]
    metadata_path = reference_artifacts[
        "metadata_path"
    ]

    before = {
        "model": robustness.file_sha256(model_path),
        "metadata": robustness.file_sha256(
            metadata_path
        ),
    }

    robustness.load_reference_model(
        model_path,
        metadata_path,
    )

    after = {
        "model": robustness.file_sha256(model_path),
        "metadata": robustness.file_sha256(
            metadata_path
        ),
    }

    assert before == after


def test_missing_reference_model(
    tmp_path,
    reference_artifacts,
):
    """Missing model files must be rejected."""

    missing_model = tmp_path / "missing.joblib"

    with pytest.raises(
        FileNotFoundError,
        match="Reference model not found",
    ):
        robustness.load_reference_model(
            missing_model,
            reference_artifacts["metadata_path"],
        )


def test_missing_reference_metadata(
    tmp_path,
    reference_artifacts,
):
    """Missing metadata files must be rejected."""

    missing_metadata = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError,
        match="Reference metadata not found",
    ):
        robustness.load_reference_model(
            reference_artifacts["model_path"],
            missing_metadata,
        )


def test_reference_model_hash_mismatch(
    reference_artifacts,
):
    """A model with an incorrect hash must be rejected."""

    metadata_path = reference_artifacts[
        "metadata_path"
    ]

    metadata = reference_artifacts[
        "metadata"
    ].copy()

    metadata["model_sha256"] = "0" * 64

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="SHA-256 mismatch",
    ):
        robustness.load_reference_model(
            reference_artifacts["model_path"],
            metadata_path,
        )


def test_missing_decision_threshold(
    reference_artifacts,
):
    """Reference metadata must define a threshold."""

    metadata = reference_artifacts[
        "metadata"
    ].copy()

    metadata.pop("decision_threshold")

    metadata_path = reference_artifacts[
        "metadata_path"
    ]

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="decision_threshold",
    ):
        robustness.load_reference_model(
            reference_artifacts["model_path"],
            metadata_path,
        )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_invalid_decision_threshold(
    reference_artifacts,
    threshold,
):
    """Thresholds must be finite and within [0, 1]."""

    metadata = reference_artifacts[
        "metadata"
    ].copy()

    metadata["decision_threshold"] = threshold

    metadata_path = reference_artifacts[
        "metadata_path"
    ]

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid decision threshold",
    ):
        robustness.load_reference_model(
            reference_artifacts["model_path"],
            metadata_path,
        )


def test_model_without_predict_proba(
    tmp_path,
):
    """Models without predict_proba must be rejected."""

    model_path = tmp_path / "invalid.joblib"
    metadata_path = tmp_path / "metadata.json"

    joblib.dump({"not": "a classifier"}, model_path)

    metadata = {
        "decision_threshold": 0.5,
        "model_sha256": robustness.file_sha256(
            model_path
        ),
    }

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        TypeError,
        match="predict_proba",
    ):
        robustness.load_reference_model(
            model_path,
            metadata_path,
        )


# ============================================================
# 5. Dataset loading
# ============================================================

def test_load_dataset(
    scenario_datasets,
):
    """Load a valid labeled evaluation dataset."""

    features, target = scenario_datasets[
        "control"
    ]

    X, y = robustness.load_dataset(
        features,
        target,
    )

    assert len(X) == len(y)
    assert X.shape[1] == 2
    assert set(y.unique()) == {0, 1}
    assert pd.api.types.is_integer_dtype(y)


def test_missing_feature_dataset(
    tmp_path,
    scenario_datasets,
):
    """Missing feature files must be rejected."""

    _, target = scenario_datasets["control"]

    with pytest.raises(
        FileNotFoundError,
        match="Feature dataset not found",
    ):
        robustness.load_dataset(
            tmp_path / "missing.csv",
            target,
        )


def test_missing_target_dataset(
    tmp_path,
    scenario_datasets,
):
    """Missing target files must be rejected."""

    features, _ = scenario_datasets["control"]

    with pytest.raises(
        FileNotFoundError,
        match="Target dataset not found",
    ):
        robustness.load_dataset(
            features,
            tmp_path / "missing.csv",
        )


def test_feature_target_row_mismatch(
    tmp_path,
    synthetic_data,
):
    """Feature and target row counts must match."""

    X, y = synthetic_data

    features = tmp_path / "X.csv"
    target = tmp_path / "y.csv"

    X.to_csv(features, index=False)
    y.iloc[:-1].to_frame().to_csv(
        target, index=False
    )

    with pytest.raises(
        ValueError,
        match="row counts",
    ):
        robustness.load_dataset(
            features,
            target,
        )


def test_multiple_target_columns_rejected(
    tmp_path,
    synthetic_data,
):
    """Target files must contain exactly one column."""

    X, y = synthetic_data

    features = tmp_path / "X.csv"
    target = tmp_path / "y.csv"

    X.to_csv(features, index=False)

    pd.DataFrame({
        "income": y,
        "extra": y,
    }).to_csv(target, index=False)

    with pytest.raises(
        ValueError,
        match="exactly one column",
    ):
        robustness.load_dataset(
            features,
            target,
        )


def test_invalid_target_labels_rejected(
    tmp_path,
    synthetic_data,
):
    """Only binary 0/1 target labels are accepted."""

    X, y = synthetic_data

    features = tmp_path / "X.csv"
    target = tmp_path / "y.csv"

    X.to_csv(features, index=False)

    invalid_y = y.copy()
    invalid_y.iloc[0] = 2

    invalid_y.to_frame().to_csv(
        target, index=False
    )

    with pytest.raises(
        ValueError,
        match="binary labels",
    ):
        robustness.load_dataset(
            features,
            target,
        )


def test_single_class_target_rejected(
    tmp_path,
    synthetic_data,
):
    """Both target classes are required for ROC-AUC."""

    X, _ = synthetic_data

    features = tmp_path / "X.csv"
    target = tmp_path / "y.csv"

    X.to_csv(features, index=False)

    pd.DataFrame({
        "income": np.zeros(
            len(X), dtype=int
        ),
    }).to_csv(target, index=False)

    with pytest.raises(
        ValueError,
        match="both classes",
    ):
        robustness.load_dataset(
            features,
            target,
        )


def test_missing_target_values_rejected(
    tmp_path,
    synthetic_data,
):
    """Missing target labels must be rejected."""

    X, y = synthetic_data

    features = tmp_path / "X.csv"
    target = tmp_path / "y.csv"

    X.to_csv(features, index=False)

    invalid_y = y.astype(float).copy()
    invalid_y.iloc[0] = np.nan

    invalid_y.to_frame().to_csv(
        target, index=False
    )

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        robustness.load_dataset(
            features,
            target,
        )


# ============================================================
# 6. Model evaluation
# ============================================================

def test_evaluate_model_returns_metrics(
    trained_model,
    synthetic_data,
):
    """Evaluation must return all required metrics."""

    X, y = synthetic_data

    metrics = robustness.evaluate_model(
        trained_model,
        X,
        y,
        threshold=0.40,
    )

    expected = {
        "rows",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
    }

    assert set(metrics) == expected
    assert metrics["rows"] == len(y)

    for metric in expected - {
        "rows", "confusion_matrix"
    }:
        assert 0 <= metrics[metric] <= 1


def test_evaluation_uses_saved_threshold(
    trained_model,
    synthetic_data,
):
    """Predictions must use the supplied threshold."""

    X, y = synthetic_data

    threshold = 0.40

    probabilities = trained_model.predict_proba(X)

    positive_index = np.flatnonzero(
        trained_model.classes_ == 1
    )[0]

    positive_probabilities = probabilities[
        :, positive_index
    ]

    predictions = (
        positive_probabilities >= threshold
    ).astype(int)

    metrics = robustness.evaluate_model(
        trained_model,
        X,
        y,
        threshold,
    )

    assert metrics["f1"] == pytest.approx(
        f1_score(y, predictions)
    )

    assert metrics["precision"] == pytest.approx(
        precision_score(
            y, predictions, zero_division=0
        )
    )

    assert metrics["recall"] == pytest.approx(
        recall_score(
            y, predictions, zero_division=0
        )
    )


def test_evaluation_matches_sklearn(
    trained_model,
    synthetic_data,
):
    """Metrics must match independent sklearn calculations."""

    X, y = synthetic_data

    threshold = 0.40

    probabilities = trained_model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = robustness.evaluate_model(
        trained_model,
        X,
        y,
        threshold,
    )

    assert metrics["accuracy"] == pytest.approx(
        accuracy_score(y, predictions)
    )

    assert metrics[
        "balanced_accuracy"
    ] == pytest.approx(
        balanced_accuracy_score(y, predictions)
    )

    assert metrics["roc_auc"] == pytest.approx(
        roc_auc_score(y, probabilities)
    )

    assert metrics["pr_auc"] == pytest.approx(
        average_precision_score(y, probabilities)
    )


def test_evaluation_confusion_matrix(
    trained_model,
    synthetic_data,
):
    """Confusion matrix must contain all observations."""

    X, y = synthetic_data

    metrics = robustness.evaluate_model(
        trained_model,
        X,
        y,
        threshold=0.40,
    )

    matrix = metrics["confusion_matrix"]

    assert len(matrix) == 2
    assert all(len(row) == 2 for row in matrix)

    assert sum(
        sum(row) for row in matrix
    ) == len(y)


def test_threshold_changes_predictions(
    trained_model,
    synthetic_data,
):
    """Different thresholds should affect classification."""

    X, y = synthetic_data

    low = robustness.evaluate_model(
        trained_model,
        X,
        y,
        threshold=0.10,
    )

    high = robustness.evaluate_model(
        trained_model,
        X,
        y,
        threshold=0.90,
    )

    assert (
        low["confusion_matrix"]
        != high["confusion_matrix"]
    )


def test_evaluation_rejects_missing_classes(
    trained_model,
    synthetic_data,
):
    """The classifier must expose its class labels."""

    X, y = synthetic_data

    class MissingClassesModel:
        def __init__(self, original):
            self.original = original

        def predict_proba(self, features):
            return self.original.predict_proba(
                features
            )

    model = MissingClassesModel(trained_model)

    with pytest.raises(
        ValueError,
        match="classes_",
    ):
        robustness.evaluate_model(
            model,
            X,
            y,
            threshold=0.5,
        )


def test_evaluation_rejects_nonfinite_probabilities(
    synthetic_data,
):
    """NaN probabilities must not produce metrics."""

    X, y = synthetic_data

    class InvalidProbabilityModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, features):
            return np.full(
                (len(features), 2),
                np.nan,
            )

    with pytest.raises(
        ValueError,
        match="non-finite probabilities",
    ):
        robustness.evaluate_model(
            InvalidProbabilityModel(),
            X,
            y,
            threshold=0.5,
        )


# ============================================================
# 7. Robustness comparison
# ============================================================

@pytest.fixture
def example_results():
    """Provide controlled metrics for comparison tests."""

    def metrics(f1, accuracy):
        return {
            "rows": 100,
            "accuracy": accuracy,
            "balanced_accuracy": accuracy,
            "precision": f1,
            "recall": f1,
            "f1": f1,
            "roc_auc": accuracy,
            "pr_auc": accuracy,
            "confusion_matrix": [
                [40, 10],
                [10, 40],
            ],
        }

    return {
        "control": metrics(0.80, 0.90),
        "mild": metrics(0.78, 0.89),
        "moderate": metrics(0.70, 0.85),
        "severe": metrics(0.50, 0.75),
    }


def test_compare_to_control(
    example_results,
):
    """Metric deltas must be relative to control."""

    comparison = robustness.compare_to_control(
        example_results
    )

    assert comparison[
        "control"
    ]["f1"]["delta"] == pytest.approx(0.0)

    assert comparison[
        "mild"
    ]["f1"]["delta"] == pytest.approx(-0.02)

    assert comparison[
        "moderate"
    ]["f1"]["delta"] == pytest.approx(-0.10)

    assert comparison[
        "severe"
    ]["f1"]["delta"] == pytest.approx(-0.30)


def test_relative_degradation(
    example_results,
):
    """Relative degradation must use control as baseline."""

    comparison = robustness.compare_to_control(
        example_results
    )

    expected = (0.80 - 0.50) / 0.80

    assert comparison[
        "severe"
    ]["f1"]["relative_degradation"] == (
        pytest.approx(expected)
    )


def test_improvement_has_negative_degradation(
    example_results,
):
    """Metric improvement must produce negative degradation."""

    example_results["mild"]["f1"] = 0.85

    comparison = robustness.compare_to_control(
        example_results
    )

    assert comparison[
        "mild"
    ]["f1"]["delta"] > 0

    assert comparison[
        "mild"
    ]["f1"]["relative_degradation"] < 0


def test_zero_control_metric(
    example_results,
):
    """Avoid division by zero in relative degradation."""

    example_results["control"]["f1"] = 0.0

    comparison = robustness.compare_to_control(
        example_results
    )

    assert comparison[
        "severe"
    ]["f1"]["relative_degradation"] is None


# ============================================================
# 8. Report generation
# ============================================================

def test_save_reports_creates_files(
    tmp_path,
    example_results,
):
    """JSON and CSV reports must be created."""

    comparison = robustness.compare_to_control(
        example_results
    )

    output_dir = tmp_path / "reports"

    json_path, csv_path = robustness.save_reports(
        example_results,
        comparison,
        {
            "model_family": "logistic_regression",
            "preprocessing_variant": "baseline",
            "model_sha256": "test-hash",
        },
        output_dir,
        threshold=0.40,
    )

    assert json_path.is_file()
    assert csv_path.is_file()


def test_json_report_structure(
    tmp_path,
    example_results,
):
    """JSON reports must include model and scenario data."""

    comparison = robustness.compare_to_control(
        example_results
    )

    json_path, _ = robustness.save_reports(
        example_results,
        comparison,
        {
            "model_family": "logistic_regression",
            "preprocessing_variant": "baseline",
            "model_sha256": "test-hash",
        },
        tmp_path / "reports",
        threshold=0.40,
    )

    report = json.loads(
        json_path.read_text(encoding="utf-8")
    )

    assert "reference_model" in report
    assert "scenarios" in report
    assert "comparison_to_control" in report

    assert report[
        "reference_model"
    ]["decision_threshold"] == pytest.approx(0.40)

    assert set(report["scenarios"]) == set(
        robustness.SCENARIOS
    )


def test_csv_report_structure(
    tmp_path,
    example_results,
):
    """CSV reports must contain all scenarios and metrics."""

    comparison = robustness.compare_to_control(
        example_results
    )

    _, csv_path = robustness.save_reports(
        example_results,
        comparison,
        {},
        tmp_path / "reports",
        threshold=0.40,
    )

    frame = pd.read_csv(csv_path)

    assert len(frame) == 4

    assert list(frame["scenario"]) == list(
        robustness.SCENARIOS
    )

    assert "f1" in frame.columns
    assert "roc_auc" in frame.columns
    assert "f1_delta" in frame.columns
    assert "f1_relative_degradation" in (
        frame.columns
    )


# ============================================================
# 9. End-to-end orchestration
# ============================================================

def test_run_robustness_end_to_end(
    robustness_args,
):
    """
    Execute the complete robustness workflow.

    Uses:
        - Temporary reference model
        - Four synthetic datasets
        - Temporary report directory
    """

    results, comparison = (
        robustness.run_robustness(
            robustness_args
        )
    )

    assert set(results) == set(
        robustness.SCENARIOS
    )

    assert set(comparison) == set(
        robustness.SCENARIOS
    )

    for scenario in robustness.SCENARIOS:
        assert results[scenario]["rows"] == 120

    assert (
        robustness_args.output_dir
        / "robustness_report.json"
    ).is_file()

    assert (
        robustness_args.output_dir
        / "robustness_summary.csv"
    ).is_file()


def test_run_robustness_preserves_reference(
    robustness_args,
):
    """Evaluation must not change reference artifacts."""

    model_path = robustness_args.model
    metadata_path = robustness_args.metadata

    before_model = robustness.file_sha256(
        model_path
    )

    before_metadata = robustness.file_sha256(
        metadata_path
    )

    robustness.run_robustness(
        robustness_args
    )

    assert robustness.file_sha256(
        model_path
    ) == before_model

    assert robustness.file_sha256(
        metadata_path
    ) == before_metadata


def test_run_robustness_uses_saved_threshold(
    robustness_args,
    synthetic_data,
):
    """
    The orchestration must use the threshold stored
    in reference metadata.
    """

    results, _ = robustness.run_robustness(
        robustness_args
    )

    model, _, threshold = (
        robustness.load_reference_model(
            robustness_args.model,
            robustness_args.metadata,
        )
    )

    X, y = synthetic_data

    expected = robustness.evaluate_model(
        model,
        X,
        y,
        threshold,
    )

    assert results["control"]["f1"] == (
        pytest.approx(expected["f1"])
    )


def test_run_robustness_rejects_missing_scenario(
    robustness_args,
    tmp_path,
):
    """Missing scenario datasets must stop evaluation."""

    robustness_args.severe = (
        tmp_path / "missing_severe.csv"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Feature dataset not found",
    ):
        robustness.run_robustness(
            robustness_args
        )
