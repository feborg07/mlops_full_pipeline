
"""
Sprint 17 - MLOps Final Project
Module: tests/test_train.py

Tests for isolated training executions, candidate models,
MLflow tracking, and DVC reference-model protection.

Run:
    python -m pytest tests/test_train.py -v
"""

from __future__ import annotations

import copy
import hashlib
import json
from argparse import Namespace
from pathlib import Path

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
    """Create synthetic, saved preprocessing partitions."""

    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_three_data
    )

    numeric_columns = (
        X_train.select_dtypes(include="number")
        .columns.tolist()
    )
    assert numeric_columns

    raw = {
        "train": X_train[numeric_columns].reset_index(drop=True),
        "val": X_val[numeric_columns].reset_index(drop=True),
        "test": X_test[numeric_columns].reset_index(drop=True),
    }

    targets = {
        name: pd.Series(y, name="income")
        .reset_index(drop=True)
        .astype(int)
        for name, y in (
            ("train", y_train),
            ("val", y_val),
            ("test", y_test),
        )
    }

    assert set(targets["train"]) == {0, 1}
    assert set(targets["val"]) == {0, 1}

    processed_dir = tmp_path / "data" / "processed"
    preprocessor_dir = tmp_path / "models" / "preprocessors"
    output_dir = tmp_path / "models" / "runs"
    reference_dir = tmp_path / "models" / "reference"

    preprocessor_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)

    for variant in training.VARIANTS:
        variant_dir = processed_dir / variant
        variant_dir.mkdir(parents=True)

        scaler = StandardScaler().fit(raw["train"])
        joblib.dump(
            scaler,
            preprocessor_dir / f"{variant}.joblib",
        )

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
        "reference_dir": reference_dir,
    }


@pytest.fixture
def fast_candidates(monkeypatch):
    """Use one inexpensive classifier per variant."""

    def factory(quick=False):
        return [
            (
                "logistic_regression",
                {"C": 1.0},
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42,
                ),
            )
        ]

    monkeypatch.setattr(training, "candidates", factory)


@pytest.fixture
def isolated_mlflow():
    """Restore process-global MLflow tracking configuration."""

    previous_uri = mlflow.get_tracking_uri()
    yield

    if mlflow.active_run() is not None:
        mlflow.end_run()

    mlflow.set_tracking_uri(previous_uri)


# ============================================================
# 2. Helpers
# ============================================================

def load(workspace, variant="baseline"):
    args = workspace["args"]
    return training.load_variant(
        args.processed_dir,
        args.preprocessor_dir,
        variant,
    )


def csv_path(workspace, filename, variant="baseline"):
    return (
        workspace["args"].processed_dir
        / variant
        / filename
    )


def execution_directories(workspace):
    """Return directories created by training invocations."""

    root = workspace["args"].output_dir
    if not root.exists():
        return []

    return sorted(
        path for path in root.iterdir()
        if path.is_dir()
    )


def run_training(workspace, quick=True):
    """
    Execute training and identify its new output directory.

    The function deliberately does not depend on train()
    returning a value.
    """

    args = workspace["args"]
    args.quick = quick

    before = set(execution_directories(workspace))
    training.train(args)
    after = set(execution_directories(workspace))

    created = after - before
    assert len(created) == 1

    run_dir = created.pop()
    assert run_dir.parent == args.output_dir.resolve()

    summary_name = (
        "quick_training_summary.json"
        if quick else "training_summary.json"
    )

    summary = json.loads(
        (run_dir / summary_name).read_text(
            encoding="utf-8"
        )
    )

    assert summary["quick"] is quick
    assert summary["execution_id"] == run_dir.name
    assert Path(summary["output_dir"]) == run_dir

    return run_dir, summary


def variant_model(run_dir, variant="baseline"):
    return run_dir / variant / "model_pipeline.joblib"


def variant_threshold(run_dir, variant="baseline"):
    return run_dir / variant / "decision_threshold.json"


def reference_hashes(workspace):
    """Fingerprint every file in the reference directory."""

    reference = workspace["reference_dir"]

    return {
        str(path.relative_to(reference)):
            training.file_sha256(path)
        for path in reference.rglob("*")
        if path.is_file()
    }


def create_reference_artifacts(workspace):
    """Simulate a DVC-recovered reference model."""

    reference = workspace["reference_dir"]

    (reference / "best_model_pipeline.joblib").write_bytes(
        b"existing-reference-model"
    )

    (reference / "best_model_metadata.json").write_text(
        '{"reference": true}',
        encoding="utf-8",
    )

    return reference_hashes(workspace)


# ============================================================
# 3. Dataset loading and validation
# ============================================================

def test_load_variant_reads_saved_splits(training_workspace):
    dataset = load(training_workspace)

    for split in ("train", "val", "test"):
        X = getattr(dataset, f"X_{split}")
        y = getattr(dataset, f"y_{split}")

        assert len(X) == len(
            training_workspace["raw"][split]
        )
        assert len(X) == len(y)
        assert list(X.columns) == list(dataset.X_train.columns)
        assert set(y.unique()).issubset({0, 1})

    assert dataset.preprocessor_path.is_file()


def test_loading_preserves_saved_rows(training_workspace):
    expected = pd.read_csv(
        csv_path(training_workspace, "X_train.csv")
    )
    actual = load(training_workspace).X_train

    pd.testing.assert_frame_equal(actual, expected)


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
    training_workspace, filename
):
    csv_path(training_workspace, filename).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Required partition missing",
    ):
        load(training_workspace)


def test_missing_preprocessor_is_rejected(training_workspace):
    (
        training_workspace["args"].preprocessor_dir
        / "baseline.joblib"
    ).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Missing fitted preprocessor",
    ):
        load(training_workspace)


def test_missing_feature_is_rejected(training_workspace):
    path = csv_path(training_workspace, "X_train.csv")
    df = pd.read_csv(path)
    df.iloc[0, 0] = np.nan
    df.to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Missing processed features",
    ):
        load(training_workspace)


def test_non_numeric_feature_is_rejected(training_workspace):
    path = csv_path(training_workspace, "X_train.csv")
    df = pd.read_csv(path)

    column = df.columns[0]
    df[column] = df[column].astype(object)
    df.loc[0, column] = "not-a-number"
    df.to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Non-numeric features",
    ):
        load(training_workspace)


def test_infinite_feature_is_rejected(training_workspace):
    path = csv_path(training_workspace, "X_train.csv")
    df = pd.read_csv(path)
    df.iloc[0, 0] = np.inf
    df.to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Non-finite features",
    ):
        load(training_workspace)


def test_invalid_target_is_rejected(training_workspace):
    path = csv_path(training_workspace, "y_train.csv")
    df = pd.read_csv(path)
    df.iloc[0, 0] = 2
    df.to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Target must contain only 0/1",
    ):
        load(training_workspace)


def test_multiple_target_columns_are_rejected(
    training_workspace,
):
    path = csv_path(training_workspace, "y_train.csv")
    df = pd.read_csv(path)
    df["extra"] = df.iloc[:, 0]
    df.to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Expected exactly one target column",
    ):
        load(training_workspace)


def test_feature_order_mismatch_is_rejected(
    training_workspace,
):
    path = csv_path(training_workspace, "X_val.csv")
    df = pd.read_csv(path)
    df[df.columns[::-1]].to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Feature columns/order differ",
    ):
        load(training_workspace)


def test_feature_target_row_mismatch_is_rejected(
    training_workspace,
):
    path = csv_path(training_workspace, "y_train.csv")
    df = pd.read_csv(path)
    df.iloc[:-1].to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Empty or misaligned",
    ):
        load(training_workspace)


# ============================================================
# 4. Model candidate generation
# ============================================================

def test_full_candidate_count_and_families():
    specs = training.candidates(quick=False)

    assert len(specs) == 18
    assert {family for family, _, _ in specs} == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "extra_trees",
    }


def test_quick_candidate_count():
    assert len(training.candidates(quick=True)) == 4


def test_logistic_regression_configuration():
    models = [
        model
        for family, _, model in training.candidates(False)
        if family == "logistic_regression"
    ]

    assert len(models) == 6

    for model in models:
        assert isinstance(model, Pipeline)
        assert isinstance(model.steps[0][1], StandardScaler)
        assert isinstance(
            model.steps[-1][1],
            LogisticRegression,
        )
        assert model.steps[-1][1].max_iter == 5000


# ============================================================
# 5. Thresholds, metrics, and ranking
# ============================================================

def test_optimal_threshold_maximizes_validation_f1():
    y = pd.Series([0, 0, 0, 1, 1, 1])
    probabilities = np.array(
        [0.10, 0.20, 0.40, 0.35, 0.70, 0.90]
    )

    precision, recall, thresholds = (
        precision_recall_curve(y, probabilities)
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-15,
        )
    )

    expected = thresholds[np.argmax(f1)]

    assert training.optimal_threshold(
        y, probabilities
    ) == pytest.approx(expected)


def test_evaluate_perfect_predictions():
    metrics = training.evaluate(
        pd.Series([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        0.5,
    )

    for name in (
        "f1", "precision", "recall",
        "accuracy", "pr_auc", "roc_auc",
    ):
        assert metrics[name] == pytest.approx(1.0)

    assert metrics["confusion_matrix"] == [
        [2, 0], [0, 2]
    ]


def test_evaluate_known_confusion_matrix():
    metrics = training.evaluate(
        pd.Series([0, 0, 1, 1]),
        np.array([0.1, 0.7, 0.8, 0.3]),
        0.5,
    )

    assert metrics["confusion_matrix"] == [
        [1, 1], [1, 1]
    ]

    for name in ("f1", "precision", "recall", "accuracy"):
        assert metrics[name] == pytest.approx(0.5)


def test_ranking_uses_validation_f1_first():
    a = {
        "val_f1": 0.80,
        "val_pr_auc": 0.60,
        "val_roc_auc": 0.60,
    }
    b = {
        "val_f1": 0.70,
        "val_pr_auc": 0.90,
        "val_roc_auc": 0.90,
    }

    assert training.rank(a) > training.rank(b)


def test_ranking_tiebreakers():
    a = {
        "val_f1": 0.80,
        "val_pr_auc": 0.90,
        "val_roc_auc": 0.80,
    }
    b = {
        "val_f1": 0.80,
        "val_pr_auc": 0.80,
        "val_roc_auc": 0.99,
    }
    c = {
        "val_f1": 0.80,
        "val_pr_auc": 0.90,
        "val_roc_auc": 0.70,
    }

    assert training.rank(a) > training.rank(b)
    assert training.rank(a) > training.rank(c)


# ============================================================
# 6. Hashing, Git, and metadata utilities
# ============================================================

def test_file_sha256(tmp_path):
    path = tmp_path / "artifact.txt"
    content = b"mlops-artifact"
    path.write_bytes(content)

    assert training.file_sha256(path) == (
        hashlib.sha256(content).hexdigest()
    )


def test_dataset_fingerprints_match_saved_files(
    training_workspace,
):
    args = training_workspace["args"]

    fingerprints = training.dataset_fingerprints(
        args.processed_dir, "baseline"
    )

    expected = {
        f"{kind}_{split}.csv"
        for split in ("train", "val", "test")
        for kind in ("X", "y")
    }

    assert set(fingerprints) == expected

    for filename, digest in fingerprints.items():
        path = csv_path(training_workspace, filename)
        assert digest == training.file_sha256(path)
        assert len(digest) == 64


def test_dataset_fingerprint_detects_changes(
    training_workspace,
):
    args = training_workspace["args"]

    before = training.dataset_fingerprints(
        args.processed_dir, "baseline"
    )

    path = csv_path(training_workspace, "X_train.csv")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    after = training.dataset_fingerprints(
        args.processed_dir, "baseline"
    )

    assert before["X_train.csv"] != after["X_train.csv"]

    for filename in before:
        if filename != "X_train.csv":
            assert before[filename] == after[filename]


def test_git_revision_returns_commit(monkeypatch):
    expected = "a" * 40

    def fake_check_output(*args, **kwargs):
        assert args[0] == ["git", "rev-parse", "HEAD"]
        return expected + "\n"

    monkeypatch.setattr(
        training.subprocess,
        "check_output",
        fake_check_output,
    )

    assert training.git_revision() == expected


def test_git_revision_handles_missing_git(monkeypatch):
    def fake_check_output(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(
        training.subprocess,
        "check_output",
        fake_check_output,
    )

    assert training.git_revision() is None


def test_atomic_json_write(tmp_path):
    path = tmp_path / "models" / "metadata.json"
    payload = {
        "model_family": "logistic_regression",
        "decision_threshold": 0.42,
        "model_sha256": "a" * 64,
    }

    training.atomic_json_write(path, payload)

    assert json.loads(path.read_text()) == payload
    assert not list(path.parent.glob("*.tmp"))


def test_atomic_json_write_replaces_existing_file(tmp_path):
    path = tmp_path / "metadata.json"

    training.atomic_json_write(path, {"version": 1})
    training.atomic_json_write(path, {"version": 2})

    assert json.loads(path.read_text()) == {"version": 2}


# ============================================================
# 7. MLflow configuration
# ============================================================

def test_configure_mlflow_creates_experiment(
    tmp_path, isolated_mlflow
):
    database = tmp_path / "tracking.db"
    artifacts = tmp_path / "artifacts"

    training.configure_mlflow(
        "pytest-configuration",
        database,
        artifacts,
    )

    assert mlflow.get_experiment_by_name(
        "pytest-configuration"
    ) is not None

    assert artifacts.is_dir()


# ============================================================
# 8. Quick training
# ============================================================

def test_quick_training_creates_artifacts(
    training_workspace, fast_candidates, isolated_mlflow
):
    run_dir, summary = run_training(training_workspace)

    assert (run_dir / "sweep_leaderboard.csv").is_file()
    assert summary["candidate_count"] == len(
        training.VARIANTS
    )

    for variant in training.VARIANTS:
        assert variant_model(run_dir, variant).is_file()
        assert variant_threshold(run_dir, variant).is_file()
        assert "test" not in summary["winners"][variant]

    # Quick training does not package an overall candidate.
    assert not (run_dir / "best_model_pipeline.joblib").exists()
    assert not (run_dir / "best_model_metadata.json").exists()


def test_quick_training_ranks_by_validation(
    training_workspace, fast_candidates, isolated_mlflow
):
    _, summary = run_training(training_workspace)

    expected = max(
        training.VARIANTS,
        key=lambda variant: training.rank(
            summary["winners"][variant]["validation"]
        ),
    )

    assert summary["preferred_variant_by_validation"] == expected


def test_quick_training_leaderboard(
    training_workspace, fast_candidates, isolated_mlflow
):
    run_dir, _ = run_training(training_workspace)

    leaderboard = pd.read_csv(
        run_dir / "sweep_leaderboard.csv"
    )

    assert len(leaderboard) == len(training.VARIANTS)
    assert set(leaderboard["variant"]) == set(
        training.VARIANTS
    )
    assert (leaderboard["status"] == "FINISHED").all()
    assert leaderboard["val_f1"].notna().all()


# ============================================================
# 9. Full training
# ============================================================

def test_full_training_evaluates_held_out_test(
    training_workspace, fast_candidates, isolated_mlflow
):
    run_dir, summary = run_training(
        training_workspace, quick=False
    )

    for variant in training.VARIANTS:
        winner = summary["winners"][variant]

        assert "validation" in winner
        assert "test" in winner
        assert 0 <= winner["test"]["f1"] <= 1
        assert len(winner["test"]["confusion_matrix"]) == 2

        threshold = json.loads(
            variant_threshold(run_dir, variant).read_text()
        )["threshold"]

        assert threshold == pytest.approx(
            winner["validation"]["optimal_threshold"]
        )

    assert (run_dir / "best_model_pipeline.joblib").is_file()
    assert (run_dir / "best_model_metadata.json").is_file()


# ============================================================
# 10. Saved pipeline inference
# ============================================================

def test_saved_pipeline_predicts_raw_inputs(
    training_workspace, fast_candidates, isolated_mlflow
):
    run_dir, _ = run_training(training_workspace)
    model = joblib.load(variant_model(run_dir))

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == [
        "preprocess",
        "feature_names",
        "classifier",
    ]

    raw = training_workspace["raw"]["test"]
    probabilities = model.predict_proba(raw)

    assert probabilities.shape == (len(raw), 2)
    assert np.isfinite(probabilities).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1), 1.0
    )


def test_saved_pipeline_matches_processed_predictions(
    training_workspace, fast_candidates, isolated_mlflow
):
    run_dir, _ = run_training(training_workspace)
    model = joblib.load(variant_model(run_dir))

    raw = training_workspace["raw"]["test"]
    processed = pd.read_csv(
        csv_path(training_workspace, "X_test.csv")
    )

    np.testing.assert_allclose(
        model.predict_proba(raw),
        model.named_steps["classifier"].predict_proba(
            processed
        ),
        rtol=1e-6,
        atol=1e-6,
    )


# ============================================================
# 11. Serialized model verification
# ============================================================

def test_verify_saved_model_accepts_valid_pipeline(
    training_workspace, fast_candidates, isolated_mlflow
):
    run_dir, _ = run_training(training_workspace)

    model_path = variant_model(run_dir)
    threshold = json.loads(
        variant_threshold(run_dir).read_text()
    )["threshold"]

    classifier = joblib.load(
        model_path
    ).named_steps["classifier"]

    training.verify_saved_model(
        model_path,
        load(training_workspace),
        classifier,
        threshold,
    )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_verify_saved_model_rejects_invalid_threshold(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
    threshold,
):
    run_dir, _ = run_training(training_workspace)
    model_path = variant_model(run_dir)

    classifier = joblib.load(
        model_path
    ).named_steps["classifier"]

    with pytest.raises(
        ValueError,
        match="Invalid decision threshold",
    ):
        training.verify_saved_model(
            model_path,
            load(training_workspace),
            classifier,
            threshold,
        )


def test_verify_saved_model_detects_prediction_mismatch(
    training_workspace, fast_candidates, isolated_mlflow
):
    run_dir, _ = run_training(training_workspace)
    model_path = variant_model(run_dir)

    classifier = joblib.load(
        model_path
    ).named_steps["classifier"]

    altered = copy.deepcopy(classifier)
    altered.coef_ = altered.coef_.copy()
    altered.coef_[:] = 0.0
    altered.intercept_ = altered.intercept_.copy()
    altered.intercept_[:] = 10.0

    with pytest.raises(
        ValueError,
        match="Serialized classifier differs",
    ):
        training.verify_saved_model(
            model_path,
            load(training_workspace),
            altered,
            0.5,
        )


# ============================================================
# 12. Input artifact preservation
# ============================================================

def test_training_preserves_input_files(
    training_workspace, fast_candidates, isolated_mlflow
):
    args = training_workspace["args"]

    paths = (
        sorted(args.processed_dir.rglob("*.csv"))
        + sorted(args.preprocessor_dir.glob("*.joblib"))
    )

    before = {
        path: training.file_sha256(path)
        for path in paths
    }

    run_training(training_workspace)

    after = {
        path: training.file_sha256(path)
        for path in paths
    }

    assert before == after


# ============================================================
# 13. MLflow integration
# ============================================================

def test_training_logs_mlflow_runs(
    training_workspace, fast_candidates, isolated_mlflow
):
    _, summary = run_training(training_workspace)
    args = training_workspace["args"]

    experiment = mlflow.get_experiment_by_name(
        args.experiment
    )
    assert experiment is not None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    assert len(runs) == len(training.VARIANTS)
    assert "metrics.val_f1" in runs.columns
    assert runs["metrics.val_f1"].notna().all()

    expected_ids = {
        summary["winners"][variant]["validation"]["run_id"]
        for variant in training.VARIANTS
    }

    assert set(runs["run_id"]) == expected_ids


# ============================================================
# 14. Run isolation
# ============================================================

def test_each_execution_has_unique_directory(
    training_workspace, fast_candidates, isolated_mlflow
):
    first_dir, first = run_training(training_workspace)
    second_dir, second = run_training(training_workspace)

    assert first_dir != second_dir
    assert first["execution_id"] != second["execution_id"]
    assert first_dir.is_dir()
    assert second_dir.is_dir()
    assert len(execution_directories(training_workspace)) == 2


def test_second_execution_preserves_first_artifacts(
    training_workspace, fast_candidates, isolated_mlflow
):
    first_dir, _ = run_training(training_workspace)

    before = {
        str(path.relative_to(first_dir)):
            training.file_sha256(path)
        for path in first_dir.rglob("*")
        if path.is_file()
    }

    run_training(training_workspace)

    after = {
        str(path.relative_to(first_dir)):
            training.file_sha256(path)
        for path in first_dir.rglob("*")
        if path.is_file()
    }

    assert before == after


def test_quick_and_full_runs_are_isolated(
    training_workspace, fast_candidates, isolated_mlflow
):
    quick_dir, quick_summary = run_training(
        training_workspace, quick=True
    )
    full_dir, full_summary = run_training(
        training_workspace, quick=False
    )

    assert quick_dir != full_dir
    assert quick_summary["quick"] is True
    assert full_summary["quick"] is False

    assert not (quick_dir / "best_model_pipeline.joblib").exists()
    assert (full_dir / "best_model_pipeline.joblib").is_file()


# ============================================================
# 15. Candidate metadata and provenance
# ============================================================

def test_full_run_candidate_has_traceable_metadata(
    training_workspace, fast_candidates, isolated_mlflow
):
    run_dir, summary = run_training(
        training_workspace, quick=False
    )

    model_path = run_dir / "best_model_pipeline.joblib"
    metadata_path = run_dir / "best_model_metadata.json"

    assert model_path.is_file()
    assert metadata_path.is_file()

    metadata = json.loads(metadata_path.read_text())

    required = {
        "model_family",
        "preprocessing_variant",
        "decision_threshold",
        "positive_class",
        "validation_f1",
        "validation_pr_auc",
        "validation_roc_auc",
        "mlflow_run_id",
        "execution_id",
        "git_commit",
        "dataset_sha256",
        "preprocessor_sha256",
        "source_model_path",
        "model_filename",
        "model_sha256",
    }

    assert required.issubset(metadata)
    assert metadata["execution_id"] == run_dir.name
    assert metadata["model_filename"] == model_path.name
    assert metadata["model_sha256"] == (
        training.file_sha256(model_path)
    )
    assert 0 <= metadata["decision_threshold"] <= 1
    assert metadata["positive_class"] == 1

    variant = metadata["preprocessing_variant"]

    assert variant == summary[
        "preferred_variant_by_validation"
    ]

    assert metadata["dataset_sha256"] == (
        training.dataset_fingerprints(
            training_workspace["args"].processed_dir,
            variant,
        )
    )

    preprocessor = (
        training_workspace["args"].preprocessor_dir
        / f"{variant}.joblib"
    )

    assert metadata["preprocessor_sha256"] == (
        training.file_sha256(preprocessor)
    )

    assert metadata["mlflow_run_id"] == (
        summary["winners"][variant]["validation"]["run_id"]
    )

    git_commit = metadata["git_commit"]
    assert git_commit is None or (
        isinstance(git_commit, str)
        and len(git_commit) == 40
    )

    assert Path(summary["candidate_model"]) == model_path
    assert Path(summary["candidate_metadata"]) == metadata_path


def test_candidate_artifacts_are_logged_to_mlflow(
    training_workspace, fast_candidates, isolated_mlflow
):
    run_dir, _ = run_training(
        training_workspace, quick=False
    )

    metadata = json.loads(
        (run_dir / "best_model_metadata.json").read_text()
    )

    client = mlflow.tracking.MlflowClient()
    run_id = metadata["mlflow_run_id"]
    run = client.get_run(run_id)

    assert run.info.status == "FINISHED"
    assert run.data.tags["candidate_for_promotion"] == "true"
    assert run.data.tags["training_execution_id"] == (
        metadata["execution_id"]
    )
    assert run.data.tags["model_sha256"] == (
        metadata["model_sha256"]
    )

    # Candidate selection must not mark the model as promoted.
    assert run.data.tags.get("promoted") != "true"

    artifacts = client.list_artifacts(
        run_id, "candidate"
    )

    names = {
        Path(artifact.path).name
        for artifact in artifacts
    }

    assert "best_model_pipeline.joblib" in names
    assert "best_model_metadata.json" in names


# ============================================================
# 16. DVC reference-model protection
# ============================================================

@pytest.mark.parametrize("quick", [True, False])
def test_training_preserves_reference_artifacts(
    training_workspace,
    fast_candidates,
    isolated_mlflow,
    quick,
):
    """Neither training mode may alter existing reference files."""

    before = create_reference_artifacts(
        training_workspace
    )

    run_dir, _ = run_training(
        training_workspace, quick=quick
    )

    assert reference_hashes(training_workspace) == before
    assert run_dir.is_dir()


def test_repeated_training_preserves_reference(
    training_workspace, fast_candidates, isolated_mlflow
):
    before = create_reference_artifacts(
        training_workspace
    )

    run_training(training_workspace, quick=False)
    run_training(training_workspace, quick=True)

    assert reference_hashes(training_workspace) == before


def test_full_training_does_not_automatically_promote(
    training_workspace, fast_candidates, isolated_mlflow
):
    reference = training_workspace["reference_dir"]

    run_dir, summary = run_training(
        training_workspace, quick=False
    )

    # The candidate exists only inside its execution directory.
    assert (run_dir / "best_model_pipeline.joblib").is_file()
    assert (run_dir / "best_model_metadata.json").is_file()

    assert "candidate_model" in summary
    assert "candidate_metadata" in summary

    # No reference artifacts are created automatically.
    assert not (
        reference / "best_model_pipeline.joblib"
    ).exists()
    assert not (
        reference / "best_model_metadata.json"
    ).exists()

    # No legacy root-level promotion either.
    models_dir = training_workspace["args"].output_dir.parent

    assert not (
        models_dir / "best_model_pipeline.joblib"
    ).exists()
    assert not (
        models_dir / "best_model_metadata.json"
    ).exists()


def test_training_rejects_reference_as_output_directory(
    training_workspace,
):
    """
    Explicitly targeting the protected reference directory
    must fail before training begins.
    """

    args = training_workspace["args"]

    # The production guard protects ROOT/models/reference.
    args.output_dir = training.ROOT / "models" / "reference"

    with pytest.raises(
        ValueError,
        match="separate from models/reference",
    ):
        training.train(args)
