
"""
Sprint 17 - MLOps Final Project
Module: tests/test_train.py

Automated tests for src/models/train.py.

The suite uses the shared split_three_data fixture from
tests/conftest.py.

Coverage:
    1. Dataset loading and validation
    2. Model candidate generation
    3. Logistic Regression configuration
    4. Decision threshold optimization
    5. Classification metrics
    6. Validation-based model ranking
    7. Artifact hashing
    8. MLflow configuration
    9. Quick training integration
    10. Full training integration
    11. Model serialization and inference
    12. Input artifact preservation
    13. MLflow experiment tracking

Run:
    python -m pytest tests/test_train.py -v
"""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models import train as training


# ============================================================
# 1. Fixtures
# ============================================================

@pytest.fixture
def training_workspace(tmp_path, split_three_data):
    """
    Reuse synthetic Adult Income data from conftest.py.

    Create temporary:
        - Train, validation, and test CSV files
        - Fitted preprocessing artifacts
        - Model output directories
        - MLflow tracking database

    The temporary environment simulates the project's
    actual preprocessing outputs without using real data.
    """

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_three_data

    # Use numeric predictors for a lightweight training fixture.
    numeric_columns = (
        X_train.select_dtypes(include="number")
        .columns.tolist()
    )

    assert numeric_columns, (
        "The shared fixture must contain numeric predictors."
    )

    # Preserve the three partitions created by conftest.py.
    raw = {
        "train": (
            X_train[numeric_columns]
            .reset_index(drop=True)
        ),
        "val": (
            X_val[numeric_columns]
            .reset_index(drop=True)
        ),
        "test": (
            X_test[numeric_columns]
            .reset_index(drop=True)
        ),
    }

    targets = {
        "train": (
            pd.Series(y_train, name="income")
            .reset_index(drop=True)
            .astype(int)
        ),
        "val": (
            pd.Series(y_val, name="income")
            .reset_index(drop=True)
            .astype(int)
        ),
        "test": (
            pd.Series(y_test, name="income")
            .reset_index(drop=True)
            .astype(int)
        ),
    }

    assert set(targets["train"]) == {0, 1}
    assert set(targets["val"]) == {0, 1}

    # Temporary project directories.
    processed_dir = (
        tmp_path / "data" / "processed"
    )

    preprocessor_dir = (
        tmp_path / "models" / "preprocessors"
    )

    output_dir = (
        tmp_path / "models" / "training"
    )

    preprocessor_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Generate saved artifacts for every preprocessing variant.
    for variant in training.VARIANTS:

        variant_dir = processed_dir / variant

        variant_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Fit the synthetic preprocessor on training data only.
        scaler = StandardScaler()

        scaler.fit(raw["train"])

        joblib.dump(
            scaler,
            preprocessor_dir / f"{variant}.joblib",
        )

        # Transform all splits using the same fitted scaler.
        for split in ("train", "val", "test"):

            transformed = pd.DataFrame(
                scaler.transform(raw[split]),
                columns=numeric_columns,
            )

            transformed.to_csv(
                variant_dir / f"X_{split}.csv",
                index=False,
            )

            targets[split].to_frame().to_csv(
                variant_dir / f"y_{split}.csv",
                index=False,
            )

    # Simulate command-line arguments.
    args = Namespace(
        processed_dir=processed_dir,
        preprocessor_dir=preprocessor_dir,
        output_dir=output_dir,
        mlflow_db=tmp_path / "tracking.db",
        mlflow_artifacts=tmp_path / "mlartifacts",
        experiment="pytest-adult-income-training",
        quick=True,
    )

    return {
        "args": args,
        "raw": raw,
        "targets": targets,
    }


@pytest.fixture
def fast_candidates(monkeypatch):
    """
    Replace the production model sweep with one lightweight
    Logistic Regression candidate.

    This reduces execution time for integration tests while
    preserving the actual training orchestration.
    """

    def factory(quick=False):

        estimator = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        )

        return [
            (
                "logistic_regression",
                {"C": 1.0},
                estimator,
            )
        ]

    monkeypatch.setattr(
        training,
        "candidates",
        factory,
    )


@pytest.fixture
def isolated_mlflow():
    """
    Restore the original MLflow tracking URI after each test.

    MLflow tracking configuration is process-global.
    """

    previous_uri = mlflow.get_tracking_uri()

    yield

    if mlflow.active_run() is not None:
        mlflow.end_run()

    mlflow.set_tracking_uri(previous_uri)


# ============================================================
# 2. Test helpers
# ============================================================

def load(workspace, variant="baseline"):
    """Load a saved preprocessing variant."""

    args = workspace["args"]

    return training.load_variant(
        args.processed_dir,
        args.preprocessor_dir,
        variant,
    )


def csv_path(workspace, filename, variant="baseline"):
    """Return the path of a saved dataset partition."""

    return (
        workspace["args"].processed_dir
        / variant
        / filename
    )


def read_summary(workspace, quick=True):
    """Load a generated training summary."""

    filename = (
        "quick_training_summary.json"
        if quick
        else "training_summary.json"
    )

    path = (
        workspace["args"].output_dir
        / filename
    )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


# ============================================================
# 3. Dataset loading and validation tests
# ============================================================

def test_load_variant_reads_saved_splits(training_workspace):
    """Verify that all saved dataset partitions load correctly."""

    dataset = load(training_workspace)

    for split in ("train", "val", "test"):

        X = getattr(dataset, f"X_{split}")
        y = getattr(dataset, f"y_{split}")

        assert len(X) == len(
            training_workspace["raw"][split]
        )

        assert len(X) == len(y)

        assert list(X.columns) == list(
            dataset.X_train.columns
        )

        assert set(y.unique()).issubset({0, 1})

    assert dataset.preprocessor_path.is_file()


def test_loading_preserves_saved_rows(training_workspace):
    """Loading must not remove, shuffle, or regenerate rows."""

    expected = pd.read_csv(
        csv_path(
            training_workspace,
            "X_train.csv",
        )
    )

    actual = load(training_workspace).X_train

    pd.testing.assert_frame_equal(
        actual,
        expected,
    )


@pytest.mark.parametrize(
    "filename",
    [
        "X_train.csv",
        "y_train.csv",
        "X_val.csv",
        "y_val.csv",
        "X_test.csv",
        "y_test.csv",
    ],
)
def test_missing_partition_is_rejected(
    training_workspace,
    filename,
):
    """Missing saved partitions must raise FileNotFoundError."""

    path = csv_path(
        training_workspace,
        filename,
    )

    path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Required partition missing",
    ):
        load(training_workspace)


def test_missing_preprocessor_is_rejected(training_workspace):
    """A missing fitted preprocessor must be rejected."""

    path = (
        training_workspace["args"].preprocessor_dir
        / "baseline.joblib"
    )

    path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Missing fitted preprocessor",
    ):
        load(training_workspace)


def test_missing_feature_is_rejected(training_workspace):
    """Processed feature matrices must not contain NaN."""

    path = csv_path(
        training_workspace,
        "X_train.csv",
    )

    df = pd.read_csv(path)

    df.iloc[0, 0] = np.nan

    df.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Missing processed features",
    ):
        load(training_workspace)


def test_non_numeric_feature_is_rejected(training_workspace):
    """
    Non-numeric processed features must be rejected.

    Convert the column to object dtype before inserting
    a string to maintain compatibility with recent pandas.
    """

    path = csv_path(
        training_workspace,
        "X_train.csv",
    )

    df = pd.read_csv(path)

    column = df.columns[0]

    # Explicit dtype conversion avoids pandas TypeError.
    df[column] = df[column].astype(object)

    df.loc[0, column] = "not-a-number"

    df.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Non-numeric features",
    ):
        load(training_workspace)


def test_infinite_feature_is_rejected(training_workspace):
    """Infinite feature values must be rejected."""

    path = csv_path(
        training_workspace,
        "X_train.csv",
    )

    df = pd.read_csv(path)

    df.iloc[0, 0] = np.inf

    df.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Non-finite features",
    ):
        load(training_workspace)


def test_invalid_target_is_rejected(training_workspace):
    """Target values must contain only binary labels."""

    path = csv_path(
        training_workspace,
        "y_train.csv",
    )

    df = pd.read_csv(path)

    df.iloc[0, 0] = 2

    df.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Target must contain only 0/1",
    ):
        load(training_workspace)


def test_multiple_target_columns_are_rejected(
    training_workspace,
):
    """Target files must contain exactly one column."""

    path = csv_path(
        training_workspace,
        "y_train.csv",
    )

    df = pd.read_csv(path)

    df["extra"] = df.iloc[:, 0]

    df.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Expected exactly one target column",
    ):
        load(training_workspace)

def test_feature_order_mismatch_is_rejected(
    training_workspace,
):
    """Feature order must match across all dataset partitions."""

    path = csv_path(
        training_workspace,
        "X_val.csv",
    )

    df = pd.read_csv(path)

    df = df[df.columns[::-1]]

    df.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Feature columns/order differ",
    ):
        load(training_workspace)


def test_feature_target_row_mismatch_is_rejected(
    training_workspace,
):
    """Feature and target row counts must be identical."""

    path = csv_path(
        training_workspace,
        "y_train.csv",
    )

    df = pd.read_csv(path)

    df.iloc[:-1].to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Empty or misaligned",
    ):
        load(training_workspace)


# ============================================================
# 4. Model candidate tests
# ============================================================

def test_full_candidate_count_and_families():
    """Full mode must generate 18 candidates across four families."""

    specs = training.candidates(quick=False)

    assert len(specs) == 18

    families = {
        family
        for family, _, _ in specs
    }

    assert families == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "extra_trees",
    }


def test_quick_candidate_count():
    """Quick mode must generate four candidates."""

    specs = training.candidates(quick=True)

    assert len(specs) == 4


def test_logistic_regression_configuration():
    """
    Logistic Regression candidates must use a pipeline
    with StandardScaler followed by LogisticRegression.
    """

    logistic_models = [
        model
        for family, _, model in training.candidates(
            quick=False
        )
        if family == "logistic_regression"
    ]

    assert len(logistic_models) == 6

    for model in logistic_models:

        assert isinstance(
            model,
            Pipeline,
        )

        assert isinstance(
            model.steps[0][1],
            StandardScaler,
        )

        classifier = model.steps[-1][1]

        assert isinstance(
            classifier,
            LogisticRegression,
        )

        assert classifier.max_iter == 5000


# ============================================================
# 5. Threshold optimization tests
# ============================================================

def test_optimal_threshold_maximizes_validation_f1():
    """Selected threshold must maximize validation F1."""

    y = pd.Series([
        0, 0, 0, 1, 1, 1,
    ])

    probabilities = np.array([
        0.10,
        0.20,
        0.40,
        0.35,
        0.70,
        0.90,
    ])

    precision, recall, thresholds = (
        precision_recall_curve(
            y,
            probabilities,
        )
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-15,
        )
    )

    expected = thresholds[
        np.argmax(f1_scores)
    ]

    actual = training.optimal_threshold(
        y,
        probabilities,
    )

    assert actual == pytest.approx(expected)


# ============================================================
# 6. Evaluation metrics tests
# ============================================================

def test_evaluate_perfect_predictions():
    """Perfect predictions must produce perfect metrics."""

    y = pd.Series([
        0, 0, 1, 1,
    ])

    probabilities = np.array([
        0.1, 0.2, 0.8, 0.9,
    ])

    metrics = training.evaluate(
        y,
        probabilities,
        0.5,
    )

    for name in (
        "f1",
        "precision",
        "recall",
        "accuracy",
        "pr_auc",
        "roc_auc",
    ):
        assert metrics[name] == pytest.approx(1.0)

    assert metrics["confusion_matrix"] == [
        [2, 0],
        [0, 2],
    ]


def test_evaluate_known_confusion_matrix():
    """Verify metrics against a known confusion matrix."""

    y = pd.Series([
        0, 0, 1, 1,
    ])

    probabilities = np.array([
        0.1, 0.7, 0.8, 0.3,
    ])

    metrics = training.evaluate(
        y,
        probabilities,
        0.5,
    )

    assert metrics["confusion_matrix"] == [
        [1, 1],
        [1, 1],
    ]

    for name in (
        "f1",
        "precision",
        "recall",
        "accuracy",
    ):
        assert metrics[name] == pytest.approx(0.5)


# ============================================================
# 7. Validation-based ranking tests
# ============================================================

def test_ranking_uses_validation_f1_first():
    """Validation F1 must have the highest ranking priority."""

    model_a = {
        "val_f1": 0.80,
        "val_pr_auc": 0.60,
        "val_roc_auc": 0.60,
    }

    model_b = {
        "val_f1": 0.70,
        "val_pr_auc": 0.90,
        "val_roc_auc": 0.90,
    }

    assert training.rank(
        model_a
    ) > training.rank(
        model_b
    )


def test_ranking_tiebreakers():
    """PR-AUC and ROC-AUC must resolve validation F1 ties."""

    model_a = {
        "val_f1": 0.80,
        "val_pr_auc": 0.90,
        "val_roc_auc": 0.80,
    }

    model_b = {
        "val_f1": 0.80,
        "val_pr_auc": 0.80,
        "val_roc_auc": 0.99,
    }

    model_c = {
        "val_f1": 0.80,
        "val_pr_auc": 0.90,
        "val_roc_auc": 0.70,
    }

    assert training.rank(
        model_a
    ) > training.rank(
        model_b
    )

    assert training.rank(
        model_a
    ) > training.rank(
        model_c
    )


# ============================================================
# 8. Artifact hashing tests
# ============================================================

def test_file_sha256(tmp_path):
    """Verify correct SHA-256 artifact hashing."""

    path = tmp_path / "artifact.txt"

    content = b"mlops-artifact"

    path.write_bytes(content)

    expected = hashlib.sha256(
        content
    ).hexdigest()

    actual = training.file_sha256(path)

    assert actual == expected


# ============================================================
# 9. MLflow configuration tests
# ============================================================

def test_configure_mlflow_creates_experiment(
    tmp_path,
    isolated_mlflow,
):
    """MLflow must create a local tracking experiment."""

    database = tmp_path / "tracking.db"

    artifacts = tmp_path / "artifacts"

    experiment_name = "pytest-configuration"

    training.configure_mlflow(
        experiment_name,
        database,
        artifacts,
    )

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    assert experiment is not None

    assert artifacts.is_dir()


# ============================================================
# 10. Quick training integration tests
# ============================================================

def test_quick_training_creates_artifacts(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
):
    """Quick training must save its expected model artifacts."""

    args = training_workspace["args"]

    training.train(args)

    assert (
        args.output_dir / "sweep_leaderboard.csv"
    ).is_file()

    summary = read_summary(
        training_workspace,
        quick=True,
    )

    assert summary["quick"] is True

    assert summary["candidate_count"] == len(
        training.VARIANTS
    )

    for variant in training.VARIANTS:

        assert (
            args.output_dir
            / variant
            / "model_pipeline.joblib"
        ).is_file()

        assert (
            args.output_dir
            / variant
            / "decision_threshold.json"
        ).is_file()

        # Quick mode must not evaluate held-out test data.
        assert "test" not in summary["winners"][variant]


def test_quick_training_ranks_by_validation(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
):
    """Preferred variant must be selected using validation metrics."""

    training.train(
        training_workspace["args"]
    )

    summary = read_summary(
        training_workspace,
        quick=True,
    )

    expected = max(
        training.VARIANTS,
        key=lambda variant: training.rank(
            summary["winners"][variant]["validation"]
        ),
    )

    assert (
        summary["preferred_variant_by_validation"]
        == expected
    )


def test_quick_training_leaderboard(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
):
    """Quick training must generate a valid leaderboard."""

    training.train(
        training_workspace["args"]
    )

    leaderboard = pd.read_csv(
        training_workspace["args"].output_dir
        / "sweep_leaderboard.csv"
    )

    assert len(leaderboard) == len(
        training.VARIANTS
    )

    assert set(
        leaderboard["variant"]
    ) == set(training.VARIANTS)

    assert (
        leaderboard["status"] == "FINISHED"
    ).all()

    assert leaderboard["val_f1"].notna().all()


# ============================================================
# 11. Full training integration tests
# ============================================================

def test_full_training_evaluates_held_out_test(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
):
    """Full training must evaluate winners on held-out test data."""

    args = training_workspace["args"]

    args.quick = False

    training.train(args)

    summary = read_summary(
        training_workspace,
        quick=False,
    )

    assert summary["quick"] is False

    for variant in training.VARIANTS:

        winner = summary["winners"][variant]

        assert "validation" in winner
        assert "test" in winner

        assert 0 <= winner["test"]["f1"] <= 1

        assert len(
            winner["test"]["confusion_matrix"]
        ) == 2

        threshold_path = (
            args.output_dir
            / variant
            / "decision_threshold.json"
        )

        saved_threshold = json.loads(
            threshold_path.read_text(
                encoding="utf-8"
            )
        )["threshold"]

        expected_threshold = (
            winner["validation"]["optimal_threshold"]
        )

        assert saved_threshold == pytest.approx(
            expected_threshold
        )


# ============================================================
# 12. Saved model inference tests
# ============================================================

def test_saved_pipeline_predicts_raw_inputs(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
):
    """Saved model pipeline must accept raw numeric inputs."""

    args = training_workspace["args"]

    training.train(args)

    model_path = (
        args.output_dir
        / "baseline"
        / "model_pipeline.joblib"
    )

    model = joblib.load(model_path)

    assert isinstance(
        model,
        Pipeline,
    )

    # Verify the updated three-stage inference pipeline.
    assert list(model.named_steps) == [
        "preprocess",
        "feature_names",
        "classifier",
    ]

    raw = training_workspace["raw"]["test"]

    probabilities = model.predict_proba(raw)

    assert probabilities.shape == (
        len(raw),
        2,
    )

    assert np.isfinite(
        probabilities
    ).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        1.0,
    )

def test_saved_pipeline_matches_processed_predictions(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
):
    """
    Raw-input pipeline predictions must match predictions
    made using saved processed features.
    """

    args = training_workspace["args"]

    training.train(args)

    model = joblib.load(
        args.output_dir
        / "baseline"
        / "model_pipeline.joblib"
    )

    raw = training_workspace["raw"]["test"]

    processed = pd.read_csv(
        csv_path(
            training_workspace,
            "X_test.csv",
        )
    )

    raw_probabilities = model.predict_proba(raw)

    processed_probabilities = (
        model.named_steps["classifier"]
        .predict_proba(processed)
    )

    np.testing.assert_allclose(
        raw_probabilities,
        processed_probabilities,
        rtol=1e-6,
        atol=1e-6,
    )


# ============================================================
# 13. Input artifact preservation tests
# ============================================================

def test_training_preserves_input_files(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
):
    """
    Training must not modify saved dataset partitions
    or fitted preprocessing artifacts.
    """

    args = training_workspace["args"]

    paths = (
        sorted(
            args.processed_dir.rglob("*.csv")
        )
        +
        sorted(
            args.preprocessor_dir.glob("*.joblib")
        )
    )

    hashes_before = {
        path: training.file_sha256(path)
        for path in paths
    }

    training.train(args)

    hashes_after = {
        path: training.file_sha256(path)
        for path in paths
    }

    assert hashes_before == hashes_after


# ============================================================
# 14. MLflow integration tests
# ============================================================

def test_training_logs_mlflow_runs(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
):
    """Training must record validation metrics in MLflow."""

    args = training_workspace["args"]

    training.train(args)

    experiment = mlflow.get_experiment_by_name(
        args.experiment
    )

    assert experiment is not None

    runs = mlflow.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ]
    )

    assert len(runs) == len(
        training.VARIANTS
    )

    assert "metrics.val_f1" in runs.columns

    assert (
        runs["metrics.val_f1"].notna()
    ).all()



# ============================================================
# 15. Dataset fingerprint and Git traceability tests
# ============================================================

def test_dataset_fingerprints_match_saved_files(
    training_workspace,
):
    """Every dataset fingerprint must match its saved CSV."""

    args = training_workspace["args"]

    fingerprints = training.dataset_fingerprints(
        args.processed_dir,
        "baseline",
    )

    expected_files = {
        f"{kind}_{split}.csv"
        for split in ("train", "val", "test")
        for kind in ("X", "y")
    }

    assert set(fingerprints) == expected_files

    for filename, digest in fingerprints.items():
        path = csv_path(
            training_workspace,
            filename,
        )

        assert digest == training.file_sha256(path)
        assert len(digest) == 64


def test_dataset_fingerprint_detects_changes(
    training_workspace,
):
    """Changing a dataset must change its fingerprint."""

    args = training_workspace["args"]

    before = training.dataset_fingerprints(
        args.processed_dir,
        "baseline",
    )

    path = csv_path(
        training_workspace,
        "X_train.csv",
    )

    with path.open("a", encoding="utf-8") as file:
        file.write("\n")

    after = training.dataset_fingerprints(
        args.processed_dir,
        "baseline",
    )

    assert (
        before["X_train.csv"]
        != after["X_train.csv"]
    )

    for filename in before:
        if filename != "X_train.csv":
            assert before[filename] == after[filename]


def test_git_revision_returns_commit(monkeypatch):
    """Git revision must return the current commit hash."""

    expected = "a" * 40

    def fake_check_output(*args, **kwargs):
        assert args[0] == [
            "git",
            "rev-parse",
            "HEAD",
        ]
        return expected + "\n"

    monkeypatch.setattr(
        training.subprocess,
        "check_output",
        fake_check_output,
    )

    assert training.git_revision() == expected


def test_git_revision_handles_missing_git(monkeypatch):
    """Training must work outside a Git repository."""

    def fake_check_output(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(
        training.subprocess,
        "check_output",
        fake_check_output,
    )

    assert training.git_revision() is None


# ============================================================
# 16. Atomic metadata writing tests
# ============================================================

def test_atomic_json_write(tmp_path):
    """Metadata must be written as valid JSON."""

    path = tmp_path / "models" / "metadata.json"

    payload = {
        "model_family": "logistic_regression",
        "decision_threshold": 0.42,
        "model_sha256": "a" * 64,
    }

    training.atomic_json_write(path, payload)

    assert path.is_file()

    actual = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert actual == payload

    # No temporary files should remain.
    assert not list(
        path.parent.glob("*.tmp")
    )


def test_atomic_json_write_replaces_existing_file(
    tmp_path,
):
    """Writing new metadata must replace the old contents."""

    path = tmp_path / "metadata.json"

    training.atomic_json_write(
        path,
        {"version": 1},
    )

    training.atomic_json_write(
        path,
        {"version": 2},
    )

    actual = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert actual == {"version": 2}


# ============================================================
# 17. Saved model artifact validation tests
# ============================================================

def test_verify_saved_model_accepts_valid_pipeline(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
):
    """A correctly serialized model must pass validation."""

    args = training_workspace["args"]

    training.train(args)

    model_path = (
        args.output_dir
        / "baseline"
        / "model_pipeline.joblib"
    )

    threshold_path = (
        args.output_dir
        / "baseline"
        / "decision_threshold.json"
    )

    threshold = json.loads(
        threshold_path.read_text(encoding="utf-8")
    )["threshold"]

    data = load(training_workspace)

    saved_model = joblib.load(model_path)

    classifier = saved_model.named_steps["classifier"]

    # Must complete without raising an exception.
    training.verify_saved_model(
        model_path,
        data,
        classifier,
        threshold,
    )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1],
)
def test_verify_saved_model_rejects_invalid_threshold(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
    threshold,
):
    """Decision thresholds must remain between zero and one."""

    args = training_workspace["args"]

    training.train(args)

    model_path = (
        args.output_dir
        / "baseline"
        / "model_pipeline.joblib"
    )

    data = load(training_workspace)

    classifier = (
        joblib.load(model_path)
        .named_steps["classifier"]
    )

    with pytest.raises(
        ValueError,
        match="Invalid decision threshold",
    ):
        training.verify_saved_model(
            model_path,
            data,
            classifier,
            threshold,
        )



def test_verify_saved_model_detects_prediction_mismatch(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
):
    """A saved classifier must match the expected trained model."""

    import copy

    args = training_workspace["args"]

    training.train(args)

    model_path = (
        args.output_dir
        / "baseline"
        / "model_pipeline.joblib"
    )

    data = load(training_workspace)

    saved_model = joblib.load(model_path)

    classifier = saved_model.named_steps["classifier"]

    # Create an independent copy of the trained classifier.
    altered_classifier = copy.deepcopy(classifier)

    # Change its learned coefficients to produce different
    # probabilities without changing its interface.
    altered_classifier.coef_ = (
        altered_classifier.coef_.copy()
    )

    altered_classifier.coef_[:] = 0.0

    altered_classifier.intercept_ = (
        altered_classifier.intercept_.copy()
    )

    altered_classifier.intercept_[:] = 10.0

    # The saved model and altered classifier should disagree.
    with pytest.raises(
        ValueError,
        match="Serialized classifier differs",
    ):
        training.verify_saved_model(
            model_path,
            data,
            altered_classifier,
            0.5,
        )


# ============================================================
# 18. Promoted model metadata integration tests
# ============================================================

def test_promoted_model_has_traceable_metadata(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
):
    """A full training run must record artifact provenance."""

    args = training_workspace["args"]
    args.quick = False

    training.train(args)

    model_path = (
        args.output_dir.parent
        / "best_model_pipeline.joblib"
    )

    metadata_path = (
        args.output_dir.parent
        / "best_model_metadata.json"
    )

    assert model_path.is_file()
    assert metadata_path.is_file()

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    required_fields = {
        "model_family",
        "preprocessing_variant",
        "decision_threshold",
        "positive_class",
        "validation_f1",
        "validation_pr_auc",
        "validation_roc_auc",
        "mlflow_run_id",
        "git_commit",
        "dataset_sha256",
        "preprocessor_sha256",
        "model_filename",
        "model_sha256",
    }

    assert required_fields.issubset(metadata)

    assert metadata["model_filename"] == model_path.name

    assert metadata["model_sha256"] == (
        training.file_sha256(model_path)
    )

    assert 0 <= metadata["decision_threshold"] <= 1

    assert metadata["positive_class"] == 1

    variant = metadata["preprocessing_variant"]

    assert variant in training.VARIANTS

    expected_fingerprints = (
        training.dataset_fingerprints(
            args.processed_dir,
            variant,
        )
    )

    assert (
        metadata["dataset_sha256"]
        == expected_fingerprints
    )

    preprocessor_path = (
        args.preprocessor_dir
        / f"{variant}.joblib"
    )

    assert metadata["preprocessor_sha256"] == (
        training.file_sha256(preprocessor_path)
    )

    git_commit = metadata["git_commit"]

    assert (
        git_commit is None
        or (
            isinstance(git_commit, str)
            and len(git_commit) == 40
        )
    )


def test_promoted_artifacts_are_logged_to_mlflow(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
):
    """The selected MLflow run must contain promotion artifacts."""

    args = training_workspace["args"]
    args.quick = False

    training.train(args)

    metadata_path = (
        args.output_dir.parent
        / "best_model_metadata.json"
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    run_id = metadata["mlflow_run_id"]

    client = mlflow.tracking.MlflowClient()

    run = client.get_run(run_id)

    assert run.info.status == "FINISHED"

    assert run.data.tags["promoted"] == "true"

    assert (
        run.data.tags["model_sha256"]
        == metadata["model_sha256"]
    )

    artifacts = client.list_artifacts(
        run_id,
        "promotion",
    )

    artifact_names = {
        artifact.path.split("/")[-1]
        for artifact in artifacts
    }

    assert "best_model_pipeline.joblib" in artifact_names

    assert "best_model_metadata.json" in artifact_names


def test_quick_training_does_not_replace_promoted_model(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
):
    """Quick experiments must preserve the existing promoted model."""

    args = training_workspace["args"]

    args.quick = False
    training.train(args)

    models_dir = args.output_dir.parent

    model_path = (
        models_dir / "best_model_pipeline.joblib"
    )

    metadata_path = (
        models_dir / "best_model_metadata.json"
    )

    model_hash_before = training.file_sha256(
        model_path
    )

    metadata_hash_before = training.file_sha256(
        metadata_path
    )

    args.quick = True
    training.train(args)

    assert training.file_sha256(model_path) == (
        model_hash_before
    )

    assert training.file_sha256(metadata_path) == (
        metadata_hash_before
    )
