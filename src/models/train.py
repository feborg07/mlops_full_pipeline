
"""
Sprint 17 - MLOps Final Project
Module: src/models/train.py

Train machine learning classifiers using previously saved
preprocessing partitions.

Usage:
    python -m src.models.train --quick
    python -m src.models.train

Required inputs for each preprocessing variant:

    data/processed/<variant>/X_train.csv
    data/processed/<variant>/y_train.csv
    data/processed/<variant>/X_val.csv
    data/processed/<variant>/y_val.csv
    data/processed/<variant>/X_test.csv
    data/processed/<variant>/y_test.csv

    models/preprocessors/<variant>.joblib

The script:
    - Never reloads raw datasets.
    - Never recreates dataset splits.
    - Never refits saved preprocessing transformers.
    - Fits candidate models using training data only.
    - Selects models and thresholds using validation data.
    - Evaluates winners using held-out test data.
    - Tracks experiments using MLflow.
    - Preserves feature names in deployable pipelines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import os
import tempfile
import warnings

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
)


# ============================================================
# 1. Configuration
# ============================================================

LOG = logging.getLogger("training")

ROOT = Path(__file__).resolve().parents[2]

VARIANTS = (
    "baseline",
    "robust",
    "interactions",
)

SEED = 42


# ============================================================
# 2. Dataset container
# ============================================================

@dataclass
class Dataset:
    """Store previously generated dataset partitions."""

    X_train: pd.DataFrame
    y_train: pd.Series

    X_val: pd.DataFrame
    y_val: pd.Series

    X_test: pd.DataFrame
    y_test: pd.Series

    preprocessor_path: Path


# ============================================================
# 3. Load saved preprocessing partitions
# ============================================================

def load_variant(
    processed_dir: Path,
    preprocessor_dir: Path,
    variant: str,
) -> Dataset:
    """
    Load partitions generated during preprocessing.

    No splitting, cleaning, or preprocessing is performed.
    """

    folder = processed_dir / variant
    matrices = {}

    for split in ("train", "val", "test"):

        x_path = folder / f"X_{split}.csv"
        y_path = folder / f"y_{split}.csv"

        for path in (x_path, y_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required partition missing: {path}. "
                    "Run preprocessing first."
                )

        X = pd.read_csv(x_path)
        y_frame = pd.read_csv(y_path)

        if y_frame.shape[1] != 1:
            raise ValueError(
                f"Expected exactly one target column: {y_path}"
            )

        y = y_frame.iloc[:, 0]

        if X.empty or len(X) != len(y):
            raise ValueError(
                f"Empty or misaligned data: {variant}/{split}"
            )

        if X.columns.duplicated().any():
            raise ValueError(
                f"Duplicate feature columns: {variant}/{split}"
            )

        if X.isna().any().any():
            raise ValueError(
                f"Missing processed features: {variant}/{split}"
            )

        if not all(
            pd.api.types.is_numeric_dtype(dtype)
            for dtype in X.dtypes
        ):
            raise ValueError(
                f"Non-numeric features: {variant}/{split}"
            )

        if not np.isfinite(
            X.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Non-finite features: {variant}/{split}"
            )

        if (
            y.isna().any()
            or not set(y.unique()).issubset({0, 1})
        ):
            raise ValueError(
                f"Target must contain only 0/1: {variant}/{split}"
            )

        matrices[split] = (
            X,
            y.astype(int),
        )

    # Validate consistent feature columns and ordering.
    train_columns = matrices["train"][0].columns

    for split in ("val", "test"):

        if not train_columns.equals(
            matrices[split][0].columns
        ):
            raise ValueError(
                f"Feature columns/order differ: {variant}/{split}"
            )

    # Training and validation require both classes.
    for split in ("train", "val"):

        if set(matrices[split][1].unique()) != {0, 1}:
            raise ValueError(
                f"Both target classes required: {variant}/{split}"
            )

    preprocessor_path = (
        preprocessor_dir / f"{variant}.joblib"
    )

    if not preprocessor_path.is_file():
        raise FileNotFoundError(
            f"Missing fitted preprocessor: {preprocessor_path}"
        )

    return Dataset(
        *matrices["train"],
        *matrices["val"],
        *matrices["test"],
        preprocessor_path=preprocessor_path,
    )


# ============================================================
# 4. Model candidates and hyperparameters
# ============================================================

def candidates(quick: bool = False):
    """
    Generate model candidates.

    Full mode:
        Logistic Regression: 6
        Random Forest: 4
        HistGradientBoosting: 4
        Extra Trees: 4

        Total: 18 candidates per variant.
        Total: 54 experiments across three variants.

    Quick mode:
        One candidate per model family.
        Total: 12 experiments.

    Logistic Regression receives an additional StandardScaler
    fitted exclusively on the training partition.
    """

    specs = []

    # Logistic Regression
    for c, weight in product(
        (0.1, 1.0, 10.0),
        (None, "balanced"),
    ):

        params = {
            "C": c,
            "class_weight": weight,
        }

        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c,
                class_weight=weight,
                max_iter=5000,
                solver="lbfgs",
                tol=1e-4,
                random_state=SEED,
            ),
        )

        specs.append(
            ("logistic_regression", params, estimator)
        )

    # Random Forest
    for depth, leaf in product(
        (None, 15),
        (1, 5),
    ):

        params = {
            "max_depth": depth,
            "min_samples_leaf": leaf,
        }

        estimator = RandomForestClassifier(
            n_estimators=150,
            n_jobs=-1,
            random_state=SEED,
            **params,
        )

        specs.append(
            ("random_forest", params, estimator)
        )

    # Histogram Gradient Boosting
    for rate, leaves in product(
        (0.05, 0.1),
        (15, 31),
    ):

        params = {
            "learning_rate": rate,
            "max_leaf_nodes": leaves,
        }

        estimator = HistGradientBoostingClassifier(
            max_iter=150,
            random_state=SEED,
            **params,
        )

        specs.append(
            ("hist_gradient_boosting", params, estimator)
        )

    # Extra Trees
    for depth, leaf in product(
        (None, 15),
        (1, 5),
    ):

        params = {
            "max_depth": depth,
            "min_samples_leaf": leaf,
        }

        estimator = ExtraTreesClassifier(
            n_estimators=150,
            n_jobs=-1,
            random_state=SEED,
            **params,
        )

        specs.append(
            ("extra_trees", params, estimator)
        )

    if quick:

        families = (
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "extra_trees",
        )

        return [
            next(
                spec for spec in specs
                if spec[0] == family
            )
            for family in families
        ]

    return specs


# ============================================================
# 5. Decision threshold optimization
# ============================================================

def optimal_threshold(
    y: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """
    Find the threshold maximizing validation F1.

    The test set is never used for threshold optimization.
    """

    precision, recall, thresholds = (
        precision_recall_curve(y, probabilities)
    )

    if not len(thresholds):
        return 0.5

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-15,
        )
    )

    return float(
        thresholds[int(np.argmax(f1))]
    )


# ============================================================
# 6. Model evaluation
# ============================================================

def evaluate(
    y: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """Calculate metrics using a fixed decision threshold."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    return {
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "pr_auc": (
            float(average_precision_score(y, probabilities))
            if y.nunique() == 2
            else None
        ),
        "roc_auc": (
            float(roc_auc_score(y, probabilities))
            if y.nunique() == 2
            else None
        ),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# 7. Validation ranking
# ============================================================

def rank(row: dict):
    """
    Rank candidates using validation metrics.

    Priority:
        1. F1
        2. PR-AUC
        3. ROC-AUC
    """

    return (
        row["val_f1"],
        row["val_pr_auc"],
        row["val_roc_auc"],
    )


# ============================================================
# 8. Artifact fingerprint
# ============================================================

def file_sha256(path: Path) -> str:
    """Calculate SHA-256 for any local file artifact."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def git_revision() -> str | None:
    """Return the source revision when running inside a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def dataset_fingerprints(processed_dir: Path, variant: str) -> dict[str, str]:
    """Hash the exact saved train/validation/test inputs used by training."""
    folder = processed_dir / variant
    return {
        f"{kind}_{split}.csv": file_sha256(folder / f"{kind}_{split}.csv")
        for split in ("train", "val", "test")
        for kind in ("X", "y")
    }


def verify_saved_model(path: Path, data: Dataset, classifier, threshold: float) -> None:
    """Reload serialized pipeline and check compatibility with saved features."""
    pipeline = joblib.load(path)  # Only load artifacts created by trusted code.
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("Promoted pipeline does not implement predict_proba")
    # The deployable pipeline accepts raw columns; the training classifier
    # accepts transformed columns. Use the fitted preprocessor's input schema.
    preprocessor = pipeline.named_steps["preprocess"]
    if not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError("Preprocessor does not expose raw feature names")
    raw_names = list(preprocessor.feature_names_in_)
    if not raw_names:
        raise ValueError("Empty raw feature schema")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Invalid decision threshold")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Saved model is missing or empty")
    # Checking raw predictions requires raw examples; the processed CSVs
    # cannot be passed through preprocessing again. Validate serialization
    # against the stored fitted estimator and its transformed input instead.
    saved_classifier = pipeline.named_steps["classifier"]
    sample = data.X_val.iloc[: min(10, len(data.X_val))]
    actual = saved_classifier.predict_proba(sample)
    expected = classifier.predict_proba(sample)
    if actual.shape != (len(sample), 2):
        raise ValueError("Invalid prediction probability shape")
    if not np.isfinite(actual).all() or not np.allclose(actual.sum(axis=1), 1.0):
        raise ValueError("Invalid predicted probabilities")
    if not np.allclose(actual, expected, atol=1e-10, rtol=1e-10):
        raise ValueError("Serialized classifier differs from trained winner")


def atomic_json_write(path: Path, payload: dict) -> None:
    """Write metadata via a temporary file on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


# ============================================================
# 9. MLflow configuration
# ============================================================

def configure_mlflow(
    experiment: str,
    db: Path,
    artifacts: Path,
):
    """Configure local MLflow tracking using SQLite."""

    db.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracking_uri = (
        f"sqlite:///{db.resolve().as_posix()}"
    )

    mlflow.set_tracking_uri(tracking_uri)

    existing = mlflow.get_experiment_by_name(
        experiment
    )

    if existing is None:
        mlflow.create_experiment(
            experiment,
            artifact_location=(
                artifacts.resolve().as_uri()
            ),
        )

    mlflow.set_experiment(experiment)


# ============================================================
# 10. Deployable pipeline construction
# ============================================================

def build_deployable_pipeline(
    fitted_preprocessor,
    classifier,
    feature_names,
    sample_features,
):
    """
    Assemble an inference pipeline without refitting models.

    The fitted preprocessor produces a NumPy array.
    The classifier was fitted with named DataFrame columns.

    FunctionTransformer restores those column names only after
    preprocessing, avoiding feature-name warnings.

    The adapter does not alter numerical feature values.
    """

    feature_names = list(feature_names)

    # Verify the classifier's expected feature names.
    if hasattr(classifier, "feature_names_in_"):

        classifier_names = list(
            classifier.feature_names_in_
        )

        if classifier_names != feature_names:
            raise ValueError(
                "Classifier feature names do not match "
                "the fitted preprocessor output."
            )

    # Convert NumPy output to a named DataFrame.
    # pandas.DataFrame is importable and joblib-compatible.
    adapter = FunctionTransformer(
        pd.DataFrame,
        kw_args={"columns": feature_names},
        validate=False,
        feature_names_out="one-to-one",
    )

    # Fit only the stateless adapter's metadata.
    # The preprocessor and classifier are NOT refitted.
    adapter.fit(sample_features)

    deployable = Pipeline([
        ("preprocess", fitted_preprocessor),
        ("feature_names", adapter),
        ("classifier", classifier),
    ])

    return deployable


# ============================================================
# 11. Main training workflow
# ============================================================

def train(args):
    """
    Run model selection across preprocessing configurations.

    Training:
        Fit candidates using training data.

    Validation:
        Optimize thresholds and select winners.

    Testing:
        Evaluate each winning model once.

    Existing preprocessing artifacts are never modified.
    """

    processed = args.processed_dir.resolve()
    preprocessor_dir = args.preprocessor_dir.resolve()
    output = args.output_dir.resolve()

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    db = args.mlflow_db.resolve()
    artifacts = args.mlflow_artifacts.resolve()

    # Load existing datasets.
    datasets = {
        variant: load_variant(
            processed,
            preprocessor_dir,
            variant,
        )
        for variant in VARIANTS
    }

    # Configure MLflow.
    configure_mlflow(
        args.experiment,
        db,
        artifacts,
    )

    leaderboard = []
    winners = {}

    specifications = candidates(args.quick)

    LOG.info(
        "Training %s candidates across %s variants",
        len(specifications) * len(VARIANTS),
        len(VARIANTS),
    )

    # --------------------------------------------------------
    # Train candidate models
    # --------------------------------------------------------

    for variant, data in datasets.items():

        best = None
        best_model = None

        for family, params, prototype in specifications:

            run_name = (
                f"{variant}-{family}-{len(leaderboard):03d}"
            )

            with mlflow.start_run(
                run_name=run_name
            ) as run:

                mlflow.set_tags({
                    "preprocessing_variant": variant,
                    "model_family": family,
                    "data_source": (
                        "saved_preprocessing_partitions"
                    ),
                    "mode": (
                        "quick" if args.quick else "full"
                    ),
                })

                mlflow.log_params({
                    "variant": variant,
                    "model": family,
                    **{
                        f"model_{key}": str(value)
                        for key, value in params.items()
                    },
                })

                try:

                    # Fit using training data only.
                    model = clone(prototype)

                    with warnings.catch_warnings(
                        record=True
                    ) as caught_warnings:

                        warnings.simplefilter(
                            "always",
                            ConvergenceWarning,
                        )

                        model.fit(
                            data.X_train,
                            data.y_train,
                        )

                    convergence_warnings = [
                        str(warning.message)
                        for warning in caught_warnings
                        if issubclass(
                            warning.category,
                            ConvergenceWarning,
                        )
                    ]

                    mlflow.log_metric(
                        "convergence_warning_count",
                        len(convergence_warnings),
                    )

                    mlflow.set_tag(
                        "converged",
                        not bool(convergence_warnings),
                    )

                    if convergence_warnings:

                        LOG.warning(
                            "%s / %s: %s convergence warning(s)",
                            variant,
                            family,
                            len(convergence_warnings),
                        )

                        mlflow.set_tag(
                            "convergence_warning",
                            convergence_warnings[0][:500],
                        )

                    # Validation predictions.
                    probabilities = (
                        model.predict_proba(
                            data.X_val
                        )[:, 1]
                    )

                    threshold = optimal_threshold(
                        data.y_val,
                        probabilities,
                    )

                    metrics = evaluate(
                        data.y_val,
                        probabilities,
                        threshold,
                    )

                    default_f1 = f1_score(
                        data.y_val,
                        probabilities >= 0.5,
                        zero_division=0,
                    )

                    # Log validation metrics.
                    mlflow.log_metrics({
                        f"val_{key}": value
                        for key, value in metrics.items()
                        if (
                            key != "confusion_matrix"
                            and value is not None
                        )
                    })

                    mlflow.log_metric(
                        "val_f1_default_050",
                        float(default_f1),
                    )

                    mlflow.log_metric(
                        "optimal_threshold",
                        threshold,
                    )

                    row = {
                        "run_id": run.info.run_id,
                        "variant": variant,
                        "model": family,
                        "parameters": json.dumps(
                            params,
                            sort_keys=True,
                        ),
                        "optimal_threshold": threshold,
                        "val_f1_default_050": float(
                            default_f1
                        ),
                        **{
                            f"val_{key}": value
                            for key, value in metrics.items()
                            if key != "confusion_matrix"
                        },
                        "convergence_warning_count": len(
                            convergence_warnings
                        ),
                        "status": "FINISHED",
                    }

                    # Select best validation candidate.
                    if (
                        best is None
                        or rank(row) > rank(best)
                    ):
                        best = row
                        best_model = model

                except Exception as exc:

                    LOG.exception(
                        "Candidate failed: %s / %s / %s",
                        variant,
                        family,
                        params,
                    )

                    mlflow.set_tag(
                        "candidate_error",
                        str(exc)[:500],
                    )

                    row = {
                        "run_id": run.info.run_id,
                        "variant": variant,
                        "model": family,
                        "parameters": json.dumps(
                            params,
                            sort_keys=True,
                        ),
                        "status": "FAILED",
                        "error": str(exc),
                    }

                leaderboard.append(row)

        if best is None or best_model is None:
            raise RuntimeError(
                f"All candidates failed for {variant}"
            )

        winners[variant] = (
            best,
            best_model,
            data,
        )

    # --------------------------------------------------------
    # Save experiment leaderboard
    # --------------------------------------------------------

    leaderboard_frame = pd.DataFrame(
        leaderboard
    )

    leaderboard_frame.sort_values(
        [
            "status",
            "val_f1",
            "val_pr_auc",
            "val_roc_auc",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        na_position="last",
    ).to_csv(
        output / "sweep_leaderboard.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Select preferred configuration using validation
    # --------------------------------------------------------

    preferred = max(
        winners,
        key=lambda variant: rank(
            winners[variant][0]
        ),
    )

    summary = {
        "preferred_variant_by_validation": preferred,
        "quick": args.quick,
        "candidate_count": len(leaderboard),
        "successful_candidates": sum(
            row["status"] == "FINISHED"
            for row in leaderboard
        ),
        "winners": {},
    }

    # --------------------------------------------------------
    # Package and evaluate winning models
    # --------------------------------------------------------

    for variant, (
        best,
        classifier,
        data,
    ) in winners.items():

        variant_dir = output / variant

        variant_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Load the original fitted preprocessor.
        fitted_preprocessor = joblib.load(
            data.preprocessor_path
        )

        # Verify feature names and ordering.
        feature_names = list(
            data.X_train.columns
        )

        if hasattr(
            fitted_preprocessor,
            "get_feature_names_out",
        ):

            expected = list(
                fitted_preprocessor.get_feature_names_out()
            )

            if expected != feature_names:
                raise ValueError(
                    "Saved preprocessor feature names "
                    f"do not match CSV columns for {variant}"
                )

        # Build corrected deployable pipeline.
        deployable = build_deployable_pipeline(
            fitted_preprocessor=fitted_preprocessor,
            classifier=classifier,
            feature_names=feature_names,
            sample_features=data.X_train,
        )

        # Do not call deployable.fit().
        # Both preprocessing and classifier are already fitted.

        model_path = (
            variant_dir / "model_pipeline.joblib"
        )

        joblib.dump(
            deployable,
            model_path,
        )

        # Save the selected validation threshold.
        threshold_path = (
            variant_dir / "decision_threshold.json"
        )

        threshold_path.write_text(
            json.dumps(
                {
                    "threshold": best["optimal_threshold"],
                    "positive_class": 1,
                    "variant": variant,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        winner_info = {
            "validation": best,
            "model_path": str(model_path),
            "preprocessor_sha256": file_sha256(
                data.preprocessor_path
            ),
        }

        # ----------------------------------------------------
        # Evaluate on held-out test data
        # ----------------------------------------------------

        if not args.quick:

            test_probabilities = (
                classifier.predict_proba(
                    data.X_test
                )[:, 1]
            )

            test_metrics = evaluate(
                data.y_test,
                test_probabilities,
                best["optimal_threshold"],
            )

            winner_info["test"] = test_metrics

            with mlflow.start_run(
                run_name=f"final-test-{variant}"
            ):

                mlflow.set_tags({
                    "preprocessing_variant": variant,
                    "stage": "final_test",
                    "selected_run_id": best["run_id"],
                    "selected_by": "validation_f1",
                })

                mlflow.log_metrics({
                    f"test_{key}": value
                    for key, value in test_metrics.items()
                    if (
                        key != "confusion_matrix"
                        and value is not None
                    )
                })

                mlflow.log_metric(
                    "decision_threshold",
                    best["optimal_threshold"],
                )

                mlflow.log_artifact(
                    str(model_path),
                    artifact_path="model",
                )

                mlflow.log_artifact(
                    str(threshold_path),
                    artifact_path="model",
                )

        summary["winners"][variant] = winner_info

        LOG.info(
            "Winner %s: %s, validation F1=%.4f%s",
            variant,
            best["model"],
            best["val_f1"],
            (
                f", test F1={winner_info['test']['f1']:.4f}"
                if not args.quick
                else ""
            ),
        )

    # --------------------------------------------------------
    # Promote overall validation winner for inference
    # --------------------------------------------------------

    # Quick runs must not overwrite a promoted full-run model.
    if not args.quick:

        best_info, _, _ = winners[preferred]

        source_model = (
            output / preferred / "model_pipeline.joblib"
        )

        source_threshold = (
            output / preferred / "decision_threshold.json"
        )

        if (
            not source_model.is_file()
            or not source_threshold.is_file()
        ):
            raise FileNotFoundError(
                "Winning model or threshold artifact is missing."
            )

        # Copy the exact serialized winning pipeline.
        models_dir = output.parent

        models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        promoted_model = (
            models_dir / "best_model_pipeline.joblib"
        )

        promoted_metadata = (
            models_dir / "best_model_metadata.json"
        )

        # Validate the winning serialized model before touching the promoted copy.
        selected_data = winners[preferred][2]
        selected_classifier = winners[preferred][1]
        threshold_info = json.loads(source_threshold.read_text(encoding="utf-8"))
        threshold = float(threshold_info["threshold"])
        verify_saved_model(source_model, selected_data, selected_classifier, threshold)

        # Record exact input and code fingerprints. Hashes are identifiers,
        # not backups: durable historical copies still need artifact storage.
        metadata = {
            "model_family": best_info["model"],
            "preprocessing_variant": preferred,
            "decision_threshold": threshold,
            "positive_class": threshold_info["positive_class"],
            "validation_f1": best_info["val_f1"],
            "validation_pr_auc": best_info["val_pr_auc"],
            "validation_roc_auc": best_info["val_roc_auc"],
            "mlflow_run_id": best_info["run_id"],
            "git_commit": git_revision(),
            "dataset_sha256": dataset_fingerprints(processed, preferred),
            "preprocessor_sha256": file_sha256(selected_data.preprocessor_path),
            "source_model_path": str(source_model),
            "model_filename": promoted_model.name,
            "model_sha256": file_sha256(source_model),
        }

        # Publish a validated copy and metadata. Neither step retrains the model.
        # Both writes are atomic individually; not a transactional pair.
        with tempfile.NamedTemporaryFile(
            dir=models_dir, prefix=".best_model_", suffix=".joblib",
            delete=False,
        ) as handle:
            temporary_model = Path(handle.name)
        try:
            shutil.copy2(source_model, temporary_model)
            if file_sha256(temporary_model) != metadata["model_sha256"]:
                raise ValueError("Model checksum changed during copying")
            verify_saved_model(temporary_model, selected_data, selected_classifier, threshold)
            os.replace(temporary_model, promoted_model)
        finally:
            temporary_model.unlink(missing_ok=True)

        atomic_json_write(promoted_metadata, metadata)

        # Keep a copy of the promoted metadata in the selected MLflow run.
        # This run already exists and contains its validation metrics.
        with mlflow.start_run(run_id=best_info["run_id"]):
            mlflow.set_tag("promoted", "true")
            mlflow.set_tag("model_sha256", metadata["model_sha256"])
            if metadata["git_commit"]:
                mlflow.set_tag("git_commit", metadata["git_commit"])
            mlflow.log_artifact(str(promoted_metadata), artifact_path="promotion")
            mlflow.log_artifact(str(promoted_model), artifact_path="promotion")

        summary["promoted_model"] = str(
            promoted_model
        )

        summary["promoted_metadata"] = str(
            promoted_metadata
        )

        LOG.info(
            "Promoted best model: %s",
            promoted_model,
        )

        LOG.info(
            "Promoted model metadata: %s",
            promoted_metadata,
        )

    # --------------------------------------------------------
    # Save training summary
    # --------------------------------------------------------

    summary_path = output / (
        "quick_training_summary.json"
        if args.quick
        else "training_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    LOG.info(
        "Preferred configuration by validation: %s",
        preferred,
    )

    LOG.info(
        "Saved leaderboard and summary under %s",
        output,
    )


# ============================================================
# 12. Command-line arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data/processed",
    )

    parser.add_argument(
        "--preprocessor-dir",
        type=Path,
        default=ROOT / "models/preprocessors",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models/training",
    )

    parser.add_argument(
        "--mlflow-db",
        type=Path,
        default=ROOT / "mlflow.db",
    )

    parser.add_argument(
        "--mlflow-artifacts",
        type=Path,
        default=ROOT / "mlartifacts",
    )

    parser.add_argument(
        "--experiment",
        default="adult-income-saved-splits",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run 4 candidates per variant "
            "without test evaluation."
        ),
    )

    parser.add_argument(
        "--show-mlflow-command",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# 13. Entry point
# ============================================================

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()

    if args.show_mlflow_command:

        uri = (
            "sqlite:///"
            f"{args.mlflow_db.resolve().as_posix()}"
        )

        print(
            f'mlflow ui --backend-store-uri "{uri}" '
            "--port 5000"
        )

        return

    train(args)


if __name__ == "__main__":
    main()
