
"""
Sprint 17 - Model Robustness Evaluation Under Data Drift

Evaluate the promoted best model across four scenarios:
    - control
    - mild
    - moderate
    - severe

The model and decision threshold remain fixed.

This module:
    1. Loads the promoted model and metadata.
    2. Loads labeled evaluation datasets.
    3. Generates predictions without retraining.
    4. Calculates classification metrics.
    5. Measures performance degradation relative to control.

No datasets, reports, or model artifacts are modified.
"""

from __future__ import annotations

import argparse
import json
import logging

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# 1. Configuration
# ============================================================

LOG = logging.getLogger("robustness")

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/best_model_pipeline.joblib"

METADATA_PATH = ROOT / "models/best_model_metadata.json"

SCENARIOS = (
    "control",
    "mild",
    "moderate",
    "severe",
)

TARGET_COLUMN = "income"

METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
)


# ============================================================
# 2. Load the promoted model and metadata
# ============================================================

def load_model(
    model_path: Path,
    metadata_path: Path,
):
    """
    Load the fitted deployment pipeline and its metadata.

    The metadata contains the decision threshold selected
    during validation.

    No model fitting is performed.
    """

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}"
        )

    LOG.info("Loading model: %s", model_path)

    # Only deserialize trusted joblib artifacts.
    model = joblib.load(model_path)

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Model must implement predict_proba()."
        )

    if not hasattr(model, "classes_"):
        raise TypeError(
            "Model must expose fitted classes_."
        )

    # Training uses binary targets 0 and 1.
    if set(model.classes_) != {0, 1}:
        raise ValueError(
            f"Unexpected model classes: {model.classes_}"
        )

    threshold = float(metadata["decision_threshold"])

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(
            f"Invalid decision threshold: {threshold}"
        )

    if int(metadata["positive_class"]) != 1:
        raise ValueError(
            "Expected positive class 1 in metadata."
        )

    LOG.info(
        "Selected model: %s / %s",
        metadata["preprocessing_variant"],
        metadata["model_family"],
    )

    LOG.info(
        "Decision threshold: %.4f",
        threshold,
    )

    return model, metadata


# ============================================================
# 3. Target validation
# ============================================================

def validate_target(y: pd.Series) -> pd.Series:
    """
    Validate the binary Adult Income target.

    Accept numeric 0/1 or original Adult Income labels.

    The positive class is always 1 (>50K).
    """

    y = y.copy()

    # Convert original string labels when necessary.
    if (
        pd.api.types.is_string_dtype(y)
        or y.dtype == object
    ):
        labels = (
            y.astype("string")
            .str.strip()
            .str.rstrip(".")
        )

        mapping = {
            "<=50K": 0,
            ">50K": 1,
            "0": 0,
            "1": 1,
        }

        y = labels.map(mapping)

    if y.isna().any():
        raise ValueError(
            "Target contains missing or invalid labels."
        )

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only classes 0 and 1."
        )

    return y.astype(int)


# ============================================================
# 4. Dataset loading
# ============================================================

def load_dataset(
    features_path: Path,
    target_path: Path | None = None,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load an evaluation dataset.

    Supported formats:

    A. One CSV containing features and income labels.

    B. Two CSV files:
       - One containing original input features.
       - One containing the corresponding target labels.

    For separate files, row ordering must match exactly.

    This function never preprocesses or reshuffles data.
    """

    features_path = Path(features_path)

    if not features_path.is_file():
        raise FileNotFoundError(
            f"Features file not found: {features_path}"
        )

    df = pd.read_csv(features_path)

    if df.empty:
        raise ValueError(
            f"Empty dataset: {features_path}"
        )

    if df.columns.duplicated().any():
        raise ValueError(
            f"Duplicate feature columns: {features_path}"
        )

    if target_path is None:

        if target_column not in df.columns:
            raise ValueError(
                f"Target '{target_column}' missing from "
                f"{features_path}. Provide a target CSV."
            )

        y = df[target_column].copy()

        X = df.drop(columns=[target_column])

    else:

        target_path = Path(target_path)

        if not target_path.is_file():
            raise FileNotFoundError(
                f"Target file not found: {target_path}"
            )

        if target_column in df.columns:
            raise ValueError(
                "Features file already contains the target. "
                "Do not provide a separate target file."
            )

        target_df = pd.read_csv(target_path)

        if target_df.shape[1] != 1:
            raise ValueError(
                "Target CSV must contain exactly one column."
            )

        X = df
        y = target_df.iloc[:, 0].copy()

    if X.shape[1] == 0:
        raise ValueError("Dataset has no feature columns.")

    if len(X) != len(y):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    y = validate_target(y)

    # Reset indices without changing row ordering.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    LOG.info(
        "Loaded %s: %d rows, %d features",
        features_path.name,
        len(X),
        X.shape[1],
    )

    return X, y


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
    Calculate classification metrics using the fixed
    validation-selected decision threshold.

    No fit(), fit_transform(), or threshold optimization
    is performed.
    """

    if len(X) == 0 or len(X) != len(y):
        raise ValueError(
            "Empty or misaligned evaluation data."
        )

    # Verify feature schema when available.
    if hasattr(model, "feature_names_in_"):

        expected = list(model.feature_names_in_)
        actual = list(X.columns)

        if expected != actual:
            raise ValueError(
                "Input features do not match the model's "
                "expected feature names and ordering."
            )

    # Locate the positive class probability column.
    positive_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(X)[
        :, positive_index
    ]

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model returned non-finite probabilities."
        )

    # Apply the original validation-selected threshold.
    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Use fixed label ordering in the confusion matrix.
    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "n_samples": int(len(y)),
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # Ranking metrics require both target classes.
    if y.nunique() == 2:

        metrics["roc_auc"] = float(
            roc_auc_score(
                y,
                probabilities,
            )
        )

        metrics["pr_auc"] = float(
            average_precision_score(
                y,
                probabilities,
            )
        )

    return metrics


# ============================================================
# 6. Performance degradation
# ============================================================

def calculate_degradation(
    baseline: float,
    current: float,
) -> float:
    """
    Calculate relative degradation as a percentage.

    Positive: performance decreased.
    Negative: performance improved.
    Zero: unchanged.
    """

    if (
        not np.isfinite(baseline)
        or not np.isfinite(current)
        or baseline == 0
    ):
        return np.nan

    return float(
        100 * (baseline - current) / baseline
    )


# ============================================================
# 7. Evaluate all drift scenarios
# ============================================================

def evaluate_robustness(
    model,
    threshold: float,
    dataset_paths: dict,
) -> pd.DataFrame:
    """
    Evaluate the same model against every drift scenario.

    The control dataset defines baseline performance.

    No retraining or model selection occurs.
    """

    missing = set(SCENARIOS) - set(dataset_paths)

    if missing:
        raise ValueError(
            f"Missing scenarios: {sorted(missing)}"
        )

    results = {}

    for scenario in SCENARIOS:

        LOG.info(
            "Evaluating scenario: %s",
            scenario.upper(),
        )

        paths = dataset_paths[scenario]

        X, y = load_dataset(
            features_path=paths["features"],
            target_path=paths.get("target"),
        )

        results[scenario] = evaluate_model(
            model=model,
            X=X,
            y=y,
            threshold=threshold,
        )

    results_df = pd.DataFrame.from_dict(
        results,
        orient="index",
    )

    results_df.index.name = "scenario"

    baseline = results_df.loc["control"]

    # Calculate both absolute and relative metric changes.
    for metric in METRIC_NAMES:

        results_df[f"{metric}_delta"] = (
            results_df[metric] - baseline[metric]
        )

        results_df[f"{metric}_degradation_pct"] = [
            calculate_degradation(
                baseline[metric],
                value,
            )
            for value in results_df[metric]
        ]

    return results_df


# ============================================================
# 8. Command-line interface
# ============================================================

def parse_args():
    """
    Accept model paths and evaluation datasets.

    Dataset paths are explicit because monitoring outputs
    may use different filenames or directory structures.
    """

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_PATH,
    )

    # Each scenario requires a feature CSV.
    # A separate target CSV is optional.
    for scenario in SCENARIOS:

        parser.add_argument(
            f"--{scenario}",
            type=Path,
            required=True,
            help=f"{scenario} features CSV",
        )

        parser.add_argument(
            f"--{scenario}-target",
            type=Path,
            default=None,
            help=f"Optional {scenario} target CSV",
        )

    return parser.parse_args()


# ============================================================
# 9. Main workflow
# ============================================================

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()

    # Load the promoted model once.
    model, metadata = load_model(
        model_path=args.model,
        metadata_path=args.metadata,
    )

    threshold = float(
        metadata["decision_threshold"]
    )

    # Build the scenario-to-file mapping.
    dataset_paths = {
        scenario: {
            "features": getattr(args, scenario),
            "target": getattr(
                args,
                f"{scenario}_target",
            ),
        }
        for scenario in SCENARIOS
    }

    results = evaluate_robustness(
        model=model,
        threshold=threshold,
        dataset_paths=dataset_paths,
    )

    # Display the core classification metrics.
    columns = [
        "n_samples",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    degradation_columns = [
        "f1_delta",
        "f1_degradation_pct",
        "recall_degradation_pct",
        "roc_auc_degradation_pct",
    ]

    print("\n" + "=" * 72)
    print("SPRINT 17 - MODEL ROBUSTNESS UNDER DATA DRIFT")
    print("=" * 72)

    print(
        f"Model: {metadata['model_family']}"
    )

    print(
        f"Variant: {metadata['preprocessing_variant']}"
    )

    print(
        f"Decision threshold: {threshold:.4f}"
    )

    print("\nCLASSIFICATION METRICS")
    print("-" * 72)

    print(
        results[columns]
        .round(4)
        .to_string()
    )

    print("\nPERFORMANCE DEGRADATION")
    print("-" * 72)

    print(
        results[degradation_columns]
        .round(4)
        .to_string()
    )

    print("=" * 72)

    return results


if __name__ == "__main__":
    main()
