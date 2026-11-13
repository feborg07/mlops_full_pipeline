"""Sprint 17: reproducible Adult Income model sweep with MLflow.

Run from the repository root::

    python -m src.models.train
    python -m src.models.train --quick

Requires: pip install mlflow scikit-learn pandas joblib

Every candidate is selected using VALIDATION ROC-AUC. The untouched TEST set
is evaluated exactly once after refitting the winner on train + validation.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data.preprocess import (
    PreprocessingConfig,
    build_preprocessor,
    clean_data,
    create_train_val_test_split,
    encode_target,
    load_and_concatenate_data,
    split_features_target,
    validate_data,
)

LOGGER = logging.getLogger(__name__)


def preprocessing_variants(base: PreprocessingConfig) -> dict[str, PreprocessingConfig]:
    """Three controlled feature-processing alternatives; identical splits."""
    return {
        "baseline": base,
        "robust": replace(base, numerical_scaling="robust"),
        "interactions": replace(base, interaction_features=True),
    }


def candidate_specs(seed: int, quick: bool = False):
    """Yield (model name, hyperparameters, unfitted estimator) for the sweep."""
    grids = {
        "logistic_regression": [
            {"C": c, "class_weight": weight}
            for c, weight in product((0.1, 1.0, 10.0), (None, "balanced"))
        ],
        "random_forest": [
            {"max_depth": depth, "min_samples_leaf": leaf}
            for depth, leaf in product((None, 15), (1, 5))
        ],
        "hist_gradient_boosting": [
            {"learning_rate": rate, "max_leaf_nodes": leaves}
            for rate, leaves in product((0.05, 0.1), (15, 31))
        ],
    }
    for model_name, combinations in grids.items():
        for params in (combinations[:1] if quick else combinations):
            if model_name == "logistic_regression":
                estimator = LogisticRegression(
                    max_iter=2000, solver="lbfgs", random_state=seed, **params
                )
            elif model_name == "random_forest":
                estimator = RandomForestClassifier(
                    n_estimators=150, random_state=seed, n_jobs=-1, **params
                )
            else:
                estimator = HistGradientBoostingClassifier(
                    max_iter=150, random_state=seed, **params
                )
            yield model_name, params, estimator


def classification_metrics(model, X, y) -> dict[str, float]:
    """Compute probability-based ROC-AUC and threshold-0.5 classification metrics."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y, predictions)),
    }


def load_raw_dataset(raw_dir: Path):
    paths = [raw_dir / "adult.data", raw_dir / "adult.test"]
    df = clean_data(load_and_concatenate_data(paths), remove_duplicates=False)
    validate_data(df)
    return split_features_target(encode_target(df))


def train_sweep(
    raw_dir: Path,
    output_dir: Path,
    tracking_uri: str,
    experiment_name: str = "adult-income-classification",
    quick: bool = False,
) -> dict:
    """Run experiments, select by validation, refit and test the winner once."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = PreprocessingConfig()
    X, y = load_raw_dataset(raw_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
        X, y,
        test_size=base_config.test_size,
        validation_size=base_config.validation_size,
        random_state=base_config.random_state,
        duplicate_aware=base_config.duplicate_aware_split,
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    results = []
    best = None
    for variant_name, config in preprocessing_variants(base_config).items():
        # Fit preprocessing ONLY on training rows, never on validation/test.
        preprocessor = build_preprocessor(config)
        X_train_encoded = preprocessor.fit_transform(X_train, y_train)
        X_val_encoded = preprocessor.transform(X_val)
        LOGGER.info("Variant %s: %s encoded features", variant_name, X_train_encoded.shape[1])

        for model_name, params, estimator in candidate_specs(config.random_state, quick):
            run_name = f"{variant_name}__{model_name}__{len(results)+1:03d}"
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.set_tags({"stage": "sweep", "preprocessing_variant": variant_name,
                                 "model_family": model_name})
                mlflow.log_params({
                    "variant": variant_name, "model": model_name,
                    "test_size": config.test_size,
                    "validation_size": config.validation_size,
                    "random_state": config.random_state,
                    "duplicate_aware_split": config.duplicate_aware_split,
                    **{f"model__{k}": str(v) for k, v in params.items()},
                })
                mlflow.log_dict(asdict(config), "preprocessing_config.json")
                try:
                    estimator.fit(X_train_encoded, y_train)
                    metrics = classification_metrics(estimator, X_val_encoded, y_val)
                    mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
                    row = {"run_id": run.info.run_id, "variant": variant_name,
                           "model": model_name, "parameters": json.dumps(params, sort_keys=True),
                           **{f"val_{k}": v for k, v in metrics.items()}, "status": "FINISHED"}
                    results.append(row)
                    # Strict > makes ties deterministic (first encountered wins).
                    if best is None or metrics["roc_auc"] > best["val_roc_auc"]:
                        best = {**row, "config": config, "params": params,
                                "estimator_class": type(estimator)}
                    LOGGER.info("%-43s validation ROC-AUC %.4f", run_name, metrics["roc_auc"])
                except Exception as exc:
                    # Record a failed candidate without aborting the entire sweep.
                    mlflow.set_tag("candidate_error", str(exc)[:500])
                    results.append({"run_id": run.info.run_id, "variant": variant_name,
                                    "model": model_name, "parameters": json.dumps(params),
                                    "status": "FAILED", "error": str(exc)})
                    LOGGER.exception("Candidate failed: %s", run_name)

    leaderboard = pd.DataFrame(results)
    leaderboard = leaderboard.sort_values(
        ["val_roc_auc", "val_f1"], ascending=[False, False], na_position="last"
    ) if not leaderboard.empty and "val_roc_auc" in leaderboard else leaderboard
    leaderboard.to_csv(output_dir / "sweep_leaderboard.csv", index=False)
    if best is None:
        raise RuntimeError("Every sweep candidate failed; see MLflow and sweep_leaderboard.csv")

    # Refit both preprocessing and classifier on all development observations.
    # The original test set remains untouched until this point.
    X_dev = pd.concat([X_train, X_val], axis=0)
    y_dev = pd.concat([y_train, y_val], axis=0)
    final_estimator = best["estimator_class"](
        **_final_model_params(best["model"], best["params"], base_config.random_state)
    )
    final_pipeline = Pipeline([
        ("preprocess", build_preprocessor(best["config"])),
        ("classifier", final_estimator),
    ])
    final_pipeline.fit(X_dev, y_dev)
    test_metrics = classification_metrics(final_pipeline, X_test, y_test)
    matrix = confusion_matrix(y_test, final_pipeline.predict(X_test), labels=[0, 1])
    model_path = output_dir / "best_model.joblib"
    joblib.dump(final_pipeline, model_path)

    with mlflow.start_run(run_name="FINAL__refit_and_test") as run:
        mlflow.set_tags({"stage": "final_test", "selected_sweep_run_id": best["run_id"],
                         "preprocessing_variant": best["variant"], "model_family": best["model"]})
        mlflow.log_params({"variant": best["variant"], "model": best["model"],
                           **{f"model__{k}": str(v) for k, v in best["params"].items()}})
        mlflow.log_metric("selected_val_roc_auc", best["val_roc_auc"])
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_dict(asdict(best["config"]), "preprocessing_config.json")
        mlflow.log_dict({"labels": [0, 1], "matrix": matrix.tolist()}, "confusion_matrix.json")
        mlflow.log_artifact(str(leaderboard_path := output_dir / "sweep_leaderboard.csv"))
        mlflow.sklearn.log_model(final_pipeline, name="model")
        final_run_id = run.info.run_id

    summary = {
        "winner": {"variant": best["variant"], "model": best["model"],
                   "parameters": best["params"], "sweep_run_id": best["run_id"]},
        "validation_roc_auc": best["val_roc_auc"],
        "final_test_metrics": test_metrics,
        "test_confusion_matrix": matrix.tolist(),
        "final_mlflow_run_id": final_run_id,
        "completed_candidates": sum(r["status"] == "FINISHED" for r in results),
        "failed_candidates": sum(r["status"] == "FAILED" for r in results),
        "dataset_rows": {"train": len(X_train), "validation": len(X_val),
                         "test": len(X_test)},
        "model_path": str(model_path),
        "leaderboard_path": str(leaderboard_path),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _final_model_params(model_name: str, params: dict, seed: int) -> dict:
    if model_name == "logistic_regression":
        return {"max_iter": 2000, "solver": "lbfgs", "random_state": seed, **params}
    if model_name == "random_forest":
        return {"n_estimators": 150, "random_state": seed, "n_jobs": -1, **params}
    return {"max_iter": 150, "random_state": seed, **params}


def main() -> None:
    parser = argparse.ArgumentParser(description="Adult Income MLflow training sweep")
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--raw-dir", type=Path, default=project_root / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=project_root / "models/training")
    parser.add_argument("--tracking-uri", default=(project_root / "mlruns").resolve().as_uri())
    parser.add_argument("--experiment", default="adult-income-classification")
    parser.add_argument("--quick", action="store_true", help="One parameter set per model and variant")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = train_sweep(args.raw_dir, args.output_dir, args.tracking_uri,
                          experiment_name=args.experiment, quick=args.quick)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
