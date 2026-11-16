
"""
Sprint 17 - MLOps Final Project
Module: src/models/robustness.py

Evaluate the robustness of the reference model against
control and drifted datasets.

Architecture:
    models/reference/
        best_model_pipeline.joblib
        best_model_metadata.json

    models/runs/<execution_id>/
        Training candidates (not used by default)

The reference model is managed separately through DVC.
This module only reads model artifacts.

Run:
    python -m src.models.robustness

Or specify datasets explicitly:
    python -m src.models.robustness \
        --control data/drift/control/X_test.csv \
        --control-target data/drift/control/y_test.csv \
        --mild data/drift/mild/X_test.csv \
        --mild-target data/drift/mild/y_test.csv \
        --moderate data/drift/moderate/X_test.csv \
        --moderate-target data/drift/moderate/y_test.csv \
        --severe data/drift/severe/X_test.csv \
        --severe-target data/drift/severe/y_test.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REFERENCE_DIR = ROOT / "models" / "reference"

DEFAULT_MODEL = (
    REFERENCE_DIR / "best_model_pipeline.joblib"
)

DEFAULT_METADATA = (
    REFERENCE_DIR / "best_model_metadata.json"
)

DEFAULT_DRIFT_DIR = ROOT / "data" / "drift"

DEFAULT_REPORT_DIR = ROOT / "reports" / "robustness"

SCENARIOS = (
    "control",
    "mild",
    "moderate",
    "severe",
)

LOGGER = logging.getLogger(__name__)


# ============================================================
# 2. File integrity
# ============================================================

def file_sha256(path: Path) -> str:
    """Calculate a file's SHA-256 fingerprint."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# 3. Load reference model
# ============================================================

def load_reference_model(
    model_path: Path = DEFAULT_MODEL,
    metadata_path: Path = DEFAULT_METADATA,
):
    """
    Load the reference model and its metadata.

    Validate:
        - Required files exist
        - Decision threshold is valid
        - Serialized model supports predict_proba
        - Model SHA-256 matches metadata, if provided

    Returns:
        model, metadata, threshold
    """

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Reference model not found: {model_path}\n"
            "Run 'dvc pull' to restore DVC-tracked artifacts."
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Reference metadata not found: {metadata_path}\n"
            "Run 'dvc pull' to restore DVC-tracked artifacts."
        )

    with metadata_path.open(
        "r", encoding="utf-8"
    ) as handle:
        metadata = json.load(handle)

    if not isinstance(metadata, dict):
        raise ValueError(
            "Reference metadata must be a JSON object."
        )

    if "decision_threshold" not in metadata:
        raise ValueError(
            "Reference metadata is missing decision_threshold."
        )

    threshold = float(metadata["decision_threshold"])

    if not np.isfinite(threshold) or not (
        0.0 <= threshold <= 1.0
    ):
        raise ValueError(
            f"Invalid decision threshold: {threshold}"
        )

    # Verify artifact integrity before deserializing.
    actual_hash = file_sha256(model_path)

    expected_hash = metadata.get("model_sha256")

    if expected_hash is not None:
        if actual_hash != expected_hash:
            raise ValueError(
                "Reference model SHA-256 mismatch. "
                "The model differs from its metadata."
            )

    # Joblib files must originate from a trusted source.
    model = joblib.load(model_path)

    if not callable(
        getattr(model, "predict_proba", None)
    ):
        raise TypeError(
            "Reference model must implement predict_proba()."
        )

    LOGGER.info(
        "Loaded reference model: %s", model_path
    )
    LOGGER.info(
        "Decision threshold: %.6f", threshold
    )
    LOGGER.info(
        "Model SHA-256: %s", actual_hash
    )

    return model, metadata, threshold


# ============================================================
# 4. Load evaluation datasets
# ============================================================

def load_dataset(
    features_path: Path,
    target_path: Path,
):
    """
    Load a labeled evaluation dataset.

    Input features must be raw model inputs, not already
    transformed preprocessing matrices.
    """

    features_path = Path(features_path)
    target_path = Path(target_path)

    if not features_path.is_file():
        raise FileNotFoundError(
            f"Feature dataset not found: {features_path}"
        )

    if not target_path.is_file():
        raise FileNotFoundError(
            f"Target dataset not found: {target_path}"
        )

    X = pd.read_csv(features_path)
    target_frame = pd.read_csv(target_path)

    if X.empty or target_frame.empty:
        raise ValueError(
            "Evaluation datasets must not be empty."
        )

    if target_frame.shape[1] != 1:
        raise ValueError(
            "Target dataset must contain exactly one column."
        )

    y = target_frame.iloc[:, 0]

    if len(X) != len(y):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    if X.columns.duplicated().any():
        raise ValueError(
            "Duplicate feature column names are not allowed."
        )

    if y.isna().any():
        raise ValueError(
            "Target dataset contains missing values."
        )

    # All evaluation datasets must use binary target labels.
    labels = set(y.unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            "Targets must contain binary labels 0 and 1."
        )

    if len(labels) != 2:
        raise ValueError(
            "Evaluation targets must contain both classes."
        )

    return X, y.astype(int)


# ============================================================
# 5. Model evaluation
# ============================================================

def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> dict:
    """
    Evaluate the model with its saved decision threshold.

    Positive class:
        1 = income >50K
    """

    probabilities = np.asarray(
        model.predict_proba(X)
    )

    if (
        probabilities.ndim != 2
        or probabilities.shape != (len(X), 2)
    ):
        raise ValueError(
            "Expected predict_proba output with two columns."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model returned non-finite probabilities."
        )

    # Explicitly select the positive class by its label.
    classes = getattr(model, "classes_", None)

    if classes is None:
        raise ValueError(
            "Model does not expose classes_."
        )

    classes = np.asarray(classes)
    positive_indices = np.flatnonzero(classes == 1)

    if len(positive_indices) != 1:
        raise ValueError(
            "Model must contain positive class 1."
        )

    positive_probabilities = probabilities[
        :, positive_indices[0]
    ]

    predictions = (
        positive_probabilities >= threshold
    ).astype(int)

    return {
        "rows": int(len(y)),
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(
                y, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y, positive_probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                y, positive_probabilities
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y, predictions, labels=[0, 1]
            ).tolist()
        ),
    }


# ============================================================
# 6. Robustness comparison
# ============================================================

def compare_to_control(
    results: dict,
) -> dict:
    """
    Compare drifted scenarios against control.

    Metric delta:
        scenario metric - control metric

    Relative degradation:
        (control metric - scenario metric)
        / control metric

    Negative degradation means improvement.
    """

    control = results["control"]

    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    )

    comparison = {}

    for scenario in SCENARIOS:
        current = results[scenario]

        comparison[scenario] = {}

        for metric in metric_names:
            baseline = control[metric]
            observed = current[metric]

            delta = observed - baseline

            relative_degradation = (
                (baseline - observed) / baseline
                if baseline != 0
                else None
            )

            comparison[scenario][metric] = {
                "delta": float(delta),
                "relative_degradation": (
                    float(relative_degradation)
                    if relative_degradation is not None
                    else None
                ),
            }

    return comparison


# ============================================================
# 7. Save reports
# ============================================================

def save_reports(
    results: dict,
    comparison: dict,
    metadata: dict,
    output_dir: Path,
    threshold: float,
):
    """Save machine-readable JSON and tabular CSV reports."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "reference_model": {
            "model_family": metadata.get(
                "model_family"
            ),
            "preprocessing_variant": metadata.get(
                "preprocessing_variant"
            ),
            "model_sha256": metadata.get(
                "model_sha256"
            ),
            "decision_threshold": threshold,
        },
        "scenarios": results,
        "comparison_to_control": comparison,
    }

    json_path = output_dir / "robustness_report.json"
    csv_path = output_dir / "robustness_summary.csv"

    with json_path.open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)

    rows = []

    for scenario in SCENARIOS:
        metrics = results[scenario]

        row = {
            "scenario": scenario,
            **{
                key: value
                for key, value in metrics.items()
                if key != "confusion_matrix"
            },
            "f1_delta": comparison[
                scenario
            ]["f1"]["delta"],
            "f1_relative_degradation": comparison[
                scenario
            ]["f1"]["relative_degradation"],
        }

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        csv_path, index=False
    )

    LOGGER.info("JSON report: %s", json_path)
    LOGGER.info("CSV report: %s", csv_path)

    return json_path, csv_path


# ============================================================
# 8. Orchestration
# ============================================================

def run_robustness(args):
    """Evaluate the reference model across all scenarios."""

    model, metadata, threshold = load_reference_model(
        args.model,
        args.metadata,
    )

    dataset_paths = {
        "control": (
            args.control,
            args.control_target,
        ),
        "mild": (
            args.mild,
            args.mild_target,
        ),
        "moderate": (
            args.moderate,
            args.moderate_target,
        ),
        "severe": (
            args.severe,
            args.severe_target,
        ),
    }

    results = {}

    for scenario, (features, target) in (
        dataset_paths.items()
    ):
        X, y = load_dataset(features, target)

        metrics = evaluate_model(
            model, X, y, threshold
        )

        results[scenario] = metrics

        LOGGER.info(
            "%-8s | rows=%5d | F1=%.4f | "
            "ROC-AUC=%.4f",
            scenario.upper(),
            metrics["rows"],
            metrics["f1"],
            metrics["roc_auc"],
        )

    comparison = compare_to_control(results)

    save_reports(
        results,
        comparison,
        metadata,
        args.output_dir,
        threshold,
    )

    print()
    print("=" * 68)
    print("SPRINT 17 - MODEL ROBUSTNESS EVALUATION")
    print("=" * 68)
    print(
        f"{'SCENARIO':<12}"
        f"{'F1':>10}"
        f"{'F1 DELTA':>12}"
        f"{'ROC-AUC':>12}"
        f"{'PR-AUC':>12}"
    )
    print("-" * 68)

    for scenario in SCENARIOS:
        metrics = results[scenario]
        f1_delta = comparison[
            scenario
        ]["f1"]["delta"]

        print(
            f"{scenario.upper():<12}"
            f"{metrics['f1']:>10.4f}"
            f"{f1_delta:>12.4f}"
            f"{metrics['roc_auc']:>12.4f}"
            f"{metrics['pr_auc']:>12.4f}"
        )

    print("-" * 68)
    print(f"Reference model: {args.model}")
    print(f"Reports: {args.output_dir}")

    return results, comparison


# ============================================================
# 9. Command-line interface
# ============================================================

def parse_args():
    """Parse model, dataset, and output paths."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate reference model robustness under drift."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Reference model joblib path.",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Reference model metadata JSON path.",
    )

    for scenario in SCENARIOS:
        parser.add_argument(
            f"--{scenario}",
            type=Path,
            default=(
                DEFAULT_DRIFT_DIR
                / scenario
                / "X_test.csv"
            ),
            help=f"{scenario} feature dataset.",
        )

        parser.add_argument(
            f"--{scenario}-target",
            type=Path,
            default=(
                DEFAULT_DRIFT_DIR
                / scenario
                / "y_test.csv"
            ),
            help=f"{scenario} target dataset.",
        )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for robustness reports.",
    )

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()
    run_robustness(args)


if __name__ == "__main__":
    main()
