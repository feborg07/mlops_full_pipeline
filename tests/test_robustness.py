
"""
Sprint 17 - Unit Tests for Model Robustness Evaluation

Tests:
    - Model and metadata loading
    - Target validation
    - Dataset loading and validation
    - Classification metrics
    - Fixed decision threshold
    - Performance degradation
    - Four-scenario robustness evaluation
    - CLI parsing and main workflow

All test artifacts are temporary.
No production datasets or models are modified.
"""

import json
import sys

from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

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

class MockClassifier:
    """
    Deterministic binary classifier for robustness tests.

    Uses the 'score' feature as the positive-class probability.
    No training is required.
    """

    def __init__(self):
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = np.array(["score"])

    def predict_proba(self, X):
        scores = X["score"].to_numpy(dtype=float)

        return np.column_stack([
            1.0 - scores,
            scores,
        ])


@pytest.fixture
def model():
    return MockClassifier()


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "score": [0.1, 0.8, 0.4, 0.9, 0.3, 0.7],
    })

    y = pd.Series(
        [0, 1, 0, 1, 1, 0],
        name="income",
    )

    return X, y


@pytest.fixture
def metadata():
    return {
        "decision_threshold": 0.5,
        "positive_class": 1,
        "preprocessing_variant": "baseline",
        "model_family": "hist_gradient_boosting",
    }


@pytest.fixture
def model_files(tmp_path, model, metadata):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"

    joblib.dump(model, model_path)

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    return model_path, metadata_path


@pytest.fixture
def dataset_files(tmp_path, sample_data):
    X, y = sample_data

    features_path = tmp_path / "X_test.csv"
    target_path = tmp_path / "y_test.csv"

    X.to_csv(features_path, index=False)
    y.to_frame().to_csv(target_path, index=False)

    return features_path, target_path


@pytest.fixture
def scenario_paths(tmp_path):
    """
    Create four labeled scenarios with identical row counts
    and controlled feature perturbations.
    """

    scores = {
        "control": [0.1, 0.8, 0.4, 0.9],
        "mild": [0.2, 0.75, 0.45, 0.85],
        "moderate": [0.4, 0.6, 0.55, 0.7],
        "severe": [0.8, 0.2, 0.9, 0.1],
    }

    targets = [0, 1, 0, 1]

    paths = {}

    for scenario, values in scores.items():
        directory = tmp_path / scenario
        directory.mkdir()

        features_path = directory / "X_test.csv"
        target_path = directory / "y_test.csv"

        pd.DataFrame({
            "score": values,
        }).to_csv(features_path, index=False)

        pd.DataFrame({
            "income": targets,
        }).to_csv(target_path, index=False)

        paths[scenario] = {
            "features": features_path,
            "target": target_path,
        }

    return paths


# ============================================================
# 2. Model Loading
# ============================================================

def test_load_model_success(model_files):
    model_path, metadata_path = model_files

    loaded_model, loaded_metadata = robustness.load_model(
        model_path,
        metadata_path,
    )

    assert isinstance(loaded_model, MockClassifier)
    assert loaded_metadata["decision_threshold"] == 0.5
    assert loaded_metadata["positive_class"] == 1


def test_load_model_missing_file(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="Model not found",
    ):
        robustness.load_model(
            tmp_path / "missing.joblib",
            tmp_path / "metadata.json",
        )


def test_load_model_missing_metadata(model_files, tmp_path):
    model_path, _ = model_files

    with pytest.raises(
        FileNotFoundError,
        match="Model metadata not found",
    ):
        robustness.load_model(
            model_path,
            tmp_path / "missing.json",
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [-0.1, 1.1, np.nan, np.inf, -np.inf],
)
def test_load_model_rejects_invalid_threshold(
    model_files,
    metadata,
    invalid_threshold,
):
    model_path, metadata_path = model_files

    metadata["decision_threshold"] = invalid_threshold

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid decision threshold",
    ):
        robustness.load_model(
            model_path,
            metadata_path,
        )


def test_load_model_rejects_wrong_positive_class(
    model_files,
    metadata,
):
    model_path, metadata_path = model_files

    metadata["positive_class"] = 0

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Expected positive class 1",
    ):
        robustness.load_model(
            model_path,
            metadata_path,
        )



def test_load_model_rejects_missing_predict_proba(
    monkeypatch,
    model_files,
):
    model_path, metadata_path = model_files

    # Create an object with classes_ but no predict_proba().
    class InvalidModel:
        classes_ = np.array([0, 1])

    invalid_model = InvalidModel()

    # Replace deserialization with the invalid model.
    monkeypatch.setattr(
        robustness.joblib,
        "load",
        lambda path: invalid_model,
    )

    with pytest.raises(
        TypeError,
        match="Model must implement predict_proba",
    ):
        robustness.load_model(
            model_path,
            metadata_path,
        )


def test_load_model_rejects_missing_classes(
    monkeypatch,
    model_files,
):
    model_path, metadata_path = model_files

    # Simulate a model that implements predict_proba()
    # but does not expose fitted classes_.
    class InvalidModel:
        def predict_proba(self, X):
            return np.zeros((len(X), 2))

    invalid_model = InvalidModel()

    # Return the invalid model without serializing it.
    monkeypatch.setattr(
        robustness.joblib,
        "load",
        lambda path: invalid_model,
    )

    with pytest.raises(
        TypeError,
        match="Model must expose fitted classes_",
    ):
        robustness.load_model(
            model_path,
            metadata_path,
        )

def test_load_model_rejects_wrong_classes(
    tmp_path,
    metadata,
):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"

    invalid_model = MockClassifier()
    invalid_model.classes_ = np.array([1, 2])

    joblib.dump(invalid_model, model_path)

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unexpected model classes",
    ):
        robustness.load_model(
            model_path,
            metadata_path,
        )


# ============================================================
# 3. Target Validation
# ============================================================

@pytest.mark.parametrize(
    "values, expected",
    [
        ([0, 1, 0], [0, 1, 0]),
        (["0", "1", "0"], [0, 1, 0]),
        (["<=50K", ">50K"], [0, 1]),
        ([" <=50K ", " >50K. "], [0, 1]),
    ],
)
def test_validate_target_valid_labels(values, expected):
    result = robustness.validate_target(
        pd.Series(values)
    )

    assert result.tolist() == expected
    assert pd.api.types.is_integer_dtype(result)


@pytest.mark.parametrize(
    "values",
    [
        ["unknown", ">50K"],
        ["<=50K", None],
        [0, 2],
        [-1, 1],
        [0, np.nan],
    ],
)
def test_validate_target_rejects_invalid_labels(values):
    with pytest.raises(ValueError):
        robustness.validate_target(
            pd.Series(values)
        )


def test_validate_target_does_not_modify_input():
    original = pd.Series(
        ["<=50K", ">50K."],
        name="income",
    )

    original_copy = original.copy(deep=True)

    robustness.validate_target(original)

    pd.testing.assert_series_equal(
        original,
        original_copy,
    )


# ============================================================
# 4. Dataset Loading
# ============================================================

def test_load_dataset_separate_files(dataset_files, sample_data):
    features_path, target_path = dataset_files
    expected_X, expected_y = sample_data

    X, y = robustness.load_dataset(
        features_path,
        target_path,
    )

    pd.testing.assert_frame_equal(
        X,
        expected_X,
    )

    pd.testing.assert_series_equal(
        y,
        expected_y,
    )


def test_load_dataset_combined_file(tmp_path, sample_data):
    X, y = sample_data

    combined = X.copy()
    combined["income"] = y

    path = tmp_path / "combined.csv"
    combined.to_csv(path, index=False)

    loaded_X, loaded_y = robustness.load_dataset(path)

    pd.testing.assert_frame_equal(
        loaded_X,
        X,
    )

    pd.testing.assert_series_equal(
        loaded_y,
        y,
    )


def test_load_dataset_missing_features(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="Features file not found",
    ):
        robustness.load_dataset(
            tmp_path / "missing.csv"
        )


def test_load_dataset_missing_target(dataset_files, tmp_path):
    features_path, _ = dataset_files

    with pytest.raises(
        FileNotFoundError,
        match="Target file not found",
    ):
        robustness.load_dataset(
            features_path,
            tmp_path / "missing_target.csv",
        )


def test_load_dataset_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"

    pd.DataFrame({
        "score": pd.Series(dtype=float),
        "income": pd.Series(dtype=int),
    }).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Empty dataset",
    ):
        robustness.load_dataset(path)


def test_load_dataset_rejects_missing_target_column(
    dataset_files,
):
    features_path, _ = dataset_files

    with pytest.raises(
        ValueError,
        match="Target 'income' missing",
    ):
        robustness.load_dataset(features_path)


def test_load_dataset_rejects_duplicate_target(
    tmp_path,
    sample_data,
):
    X, y = sample_data

    combined = X.copy()
    combined["income"] = y

    features_path = tmp_path / "features.csv"
    target_path = tmp_path / "target.csv"

    combined.to_csv(features_path, index=False)
    y.to_frame().to_csv(target_path, index=False)

    with pytest.raises(
        ValueError,
        match="already contains the target",
    ):
        robustness.load_dataset(
            features_path,
            target_path,
        )


def test_load_dataset_rejects_multiple_target_columns(
    tmp_path,
    sample_data,
):
    X, y = sample_data

    features_path = tmp_path / "features.csv"
    target_path = tmp_path / "target.csv"

    X.to_csv(features_path, index=False)

    pd.DataFrame({
        "income": y,
        "extra": y,
    }).to_csv(target_path, index=False)

    with pytest.raises(
        ValueError,
        match="exactly one column",
    ):
        robustness.load_dataset(
            features_path,
            target_path,
        )


def test_load_dataset_rejects_mismatched_rows(
    tmp_path,
    sample_data,
):
    X, y = sample_data

    features_path = tmp_path / "features.csv"
    target_path = tmp_path / "target.csv"

    X.to_csv(features_path, index=False)
    y.iloc[:-1].to_frame().to_csv(
        target_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="row counts do not match",
    ):
        robustness.load_dataset(
            features_path,
            target_path,
        )


def test_load_dataset_rejects_no_features(tmp_path):
    path = tmp_path / "target_only.csv"

    pd.DataFrame({
        "income": [0, 1, 0],
    }).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="no feature columns",
    ):
        robustness.load_dataset(path)


def test_load_dataset_preserves_row_order(
    tmp_path,
):
    X = pd.DataFrame({
        "score": [0.9, 0.1, 0.7, 0.2],
    })

    y = pd.Series(
        [1, 0, 1, 0],
        name="income",
    )

    features_path = tmp_path / "features.csv"
    target_path = tmp_path / "target.csv"

    X.to_csv(features_path, index=False)
    y.to_frame().to_csv(target_path, index=False)

    loaded_X, loaded_y = robustness.load_dataset(
        features_path,
        target_path,
    )

    assert loaded_X["score"].tolist() == [
        0.9, 0.1, 0.7, 0.2
    ]

    assert loaded_y.tolist() == [1, 0, 1, 0]


def test_load_dataset_does_not_modify_files(dataset_files):
    features_path, target_path = dataset_files

    features_before = features_path.read_bytes()
    target_before = target_path.read_bytes()

    robustness.load_dataset(
        features_path,
        target_path,
    )

    assert features_path.read_bytes() == features_before
    assert target_path.read_bytes() == target_before


# ============================================================
# 5. Classification Metrics
# ============================================================

def test_evaluate_model_correct_metrics(model, sample_data):
    X, y = sample_data

    result = robustness.evaluate_model(
        model,
        X,
        y,
        threshold=0.5,
    )

    probabilities = X["score"].to_numpy()
    predictions = (probabilities >= 0.5).astype(int)

    assert result["n_samples"] == len(y)

    assert result["accuracy"] == pytest.approx(
        accuracy_score(y, predictions)
    )

    assert result["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(y, predictions)
    )

    assert result["precision"] == pytest.approx(
        precision_score(y, predictions)
    )

    assert result["recall"] == pytest.approx(
        recall_score(y, predictions)
    )

    assert result["f1"] == pytest.approx(
        f1_score(y, predictions)
    )

    assert result["roc_auc"] == pytest.approx(
        roc_auc_score(y, probabilities)
    )

    assert result["pr_auc"] == pytest.approx(
        average_precision_score(y, probabilities)
    )


def test_evaluate_model_confusion_matrix(model, sample_data):
    X, y = sample_data

    result = robustness.evaluate_model(
        model,
        X,
        y,
        threshold=0.5,
    )

    assert result["tn"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tp"] == 2

    assert sum(
        result[key]
        for key in ("tn", "fp", "fn", "tp")
    ) == len(y)


def test_evaluate_model_uses_fixed_threshold(
    model,
    sample_data,
):
    X, y = sample_data

    low_threshold = robustness.evaluate_model(
        model,
        X,
        y,
        threshold=0.3,
    )

    high_threshold = robustness.evaluate_model(
        model,
        X,
        y,
        threshold=0.8,
    )

    assert low_threshold["tp"] > high_threshold["tp"]
    assert low_threshold["fp"] > high_threshold["fp"]


def test_evaluate_model_rejects_empty_data(model):
    X = pd.DataFrame({
        "score": [],
    })

    y = pd.Series(dtype=int)

    with pytest.raises(
        ValueError,
        match="Empty or misaligned",
    ):
        robustness.evaluate_model(
            model,
            X,
            y,
            threshold=0.5,
        )


def test_evaluate_model_rejects_misaligned_data(
    model,
    sample_data,
):
    X, y = sample_data

    with pytest.raises(
        ValueError,
        match="Empty or misaligned",
    ):
        robustness.evaluate_model(
            model,
            X,
            y.iloc[:-1],
            threshold=0.5,
        )


def test_evaluate_model_rejects_wrong_feature_schema(
    model,
    sample_data,
):
    X, y = sample_data

    X = X.rename(columns={
        "score": "wrong_name",
    })

    with pytest.raises(
        ValueError,
        match="expected feature names and ordering",
    ):
        robustness.evaluate_model(
            model,
            X,
            y,
            threshold=0.5,
        )


def test_evaluate_model_rejects_nonfinite_probabilities(
    model,
    sample_data,
):
    X, y = sample_data

    X = X.copy()
    X.loc[0, "score"] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite probabilities",
    ):
        robustness.evaluate_model(
            model,
            X,
            y,
            threshold=0.5,
        )


@pytest.mark.parametrize(
    "target_value",
    [0, 1],
)
def test_evaluate_model_single_class(
    model,
    sample_data,
    target_value,
):
    X, _ = sample_data

    y = pd.Series(
        [target_value] * len(X)
    )

    with pytest.warns(
        UserWarning,
        match="y_pred contains classes not in y_true",
    ):
        result = robustness.evaluate_model(
            model,
            X,
            y,
            threshold=0.5,
        )

    assert np.isnan(result["roc_auc"])
    assert np.isnan(result["pr_auc"])
    assert np.isfinite(result["f1"])

def test_evaluate_model_does_not_fit(
    sample_data,
):
    X, y = sample_data

    model = MockClassifier()
    model.fit = MagicMock(
        side_effect=AssertionError(
            "Model must not be fitted."
        )
    )

    robustness.evaluate_model(
        model,
        X,
        y,
        threshold=0.5,
    )

    model.fit.assert_not_called()


# ============================================================
# 6. Performance Degradation
# ============================================================

@pytest.mark.parametrize(
    "baseline, current, expected",
    [
        (0.8, 0.6, 25.0),
        (0.8, 0.9, -12.5),
        (0.8, 0.8, 0.0),
        (1.0, 0.0, 100.0),
    ],
)
def test_calculate_degradation(
    baseline,
    current,
    expected,
):
    result = robustness.calculate_degradation(
        baseline,
        current,
    )

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "baseline, current",
    [
        (0.0, 0.5),
        (np.nan, 0.5),
        (0.5, np.nan),
        (np.inf, 0.5),
        (0.5, np.inf),
    ],
)
def test_calculate_degradation_invalid_values(
    baseline,
    current,
):
    result = robustness.calculate_degradation(
        baseline,
        current,
    )

    assert np.isnan(result)


# ============================================================
# 7. Four-Scenario Evaluation
# ============================================================

def test_evaluate_robustness_all_scenarios(
    model,
    scenario_paths,
):
    results = robustness.evaluate_robustness(
        model=model,
        threshold=0.5,
        dataset_paths=scenario_paths,
    )

    assert isinstance(results, pd.DataFrame)

    assert list(results.index) == list(
        robustness.SCENARIOS
    )

    assert results.index.name == "scenario"

    assert (results["n_samples"] == 4).all()


def test_evaluate_robustness_calculates_degradation(
    model,
    scenario_paths,
):
    results = robustness.evaluate_robustness(
        model=model,
        threshold=0.5,
        dataset_paths=scenario_paths,
    )

    assert results.loc[
        "control", "f1_delta"
    ] == pytest.approx(0.0)

    assert results.loc[
        "control", "f1_degradation_pct"
    ] == pytest.approx(0.0)

    assert results.loc[
        "severe", "f1"
    ] < results.loc[
        "control", "f1"
    ]

    assert results.loc[
        "severe", "f1_degradation_pct"
    ] > 0


def test_evaluate_robustness_all_metric_columns(
    model,
    scenario_paths,
):
    results = robustness.evaluate_robustness(
        model=model,
        threshold=0.5,
        dataset_paths=scenario_paths,
    )

    for metric in robustness.METRIC_NAMES:

        assert metric in results.columns

        assert f"{metric}_delta" in results.columns

        assert (
            f"{metric}_degradation_pct"
            in results.columns
        )


def test_evaluate_robustness_missing_scenario(
    model,
    scenario_paths,
):
    del scenario_paths["severe"]

    with pytest.raises(
        ValueError,
        match="Missing scenarios",
    ):
        robustness.evaluate_robustness(
            model=model,
            threshold=0.5,
            dataset_paths=scenario_paths,
        )


def test_evaluate_robustness_preserves_files(
    model,
    scenario_paths,
):
    files = []

    for paths in scenario_paths.values():
        files.extend([
            paths["features"],
            paths["target"],
        ])

    before = {
        path: path.read_bytes()
        for path in files
    }

    robustness.evaluate_robustness(
        model=model,
        threshold=0.5,
        dataset_paths=scenario_paths,
    )

    for path in files:
        assert path.read_bytes() == before[path]


def test_evaluate_robustness_uses_same_model(
    model,
    scenario_paths,
):
    model.predict_proba = MagicMock(
        wraps=model.predict_proba
    )

    robustness.evaluate_robustness(
        model=model,
        threshold=0.5,
        dataset_paths=scenario_paths,
    )

    assert model.predict_proba.call_count == 4


# ============================================================
# 8. CLI
# ============================================================

def test_parse_args_requires_all_scenarios(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["robustness.py"],
    )

    with pytest.raises(SystemExit) as exc:
        robustness.parse_args()

    assert exc.value.code == 2


def test_parse_args_accepts_dataset_paths(monkeypatch):
    argv = ["robustness.py"]

    for scenario in robustness.SCENARIOS:
        argv.extend([
            f"--{scenario}",
            f"data/drift/{scenario}/X_test.csv",
            f"--{scenario}-target",
            f"data/drift/{scenario}/y_test.csv",
        ])

    monkeypatch.setattr(
        sys,
        "argv",
        argv,
    )

    args = robustness.parse_args()

    for scenario in robustness.SCENARIOS:

        assert getattr(
            args,
            scenario,
        ) == Path(
            f"data/drift/{scenario}/X_test.csv"
        )

        assert getattr(
            args,
            f"{scenario}_target",
        ) == Path(
            f"data/drift/{scenario}/y_test.csv"
        )


def test_main_runs_complete_workflow(
    monkeypatch,
    model_files,
    scenario_paths,
    capsys,
):
    model_path, metadata_path = model_files

    argv = [
        "robustness.py",
        "--model",
        str(model_path),
        "--metadata",
        str(metadata_path),
    ]

    for scenario, paths in scenario_paths.items():
        argv.extend([
            f"--{scenario}",
            str(paths["features"]),
            f"--{scenario}-target",
            str(paths["target"]),
        ])

    monkeypatch.setattr(
        sys,
        "argv",
        argv,
    )

    results = robustness.main()

    output = capsys.readouterr().out

    assert isinstance(results, pd.DataFrame)

    assert len(results) == 4

    assert (
        "SPRINT 17 - MODEL ROBUSTNESS UNDER DATA DRIFT"
        in output
    )

    assert "CLASSIFICATION METRICS" in output
    assert "PERFORMANCE DEGRADATION" in output

    for scenario in robustness.SCENARIOS:
        assert scenario in output
