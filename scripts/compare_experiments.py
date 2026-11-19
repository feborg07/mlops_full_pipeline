"""Compare Sprint 17 training candidates using the MLflow tracking store.

Usage (from the repository root):
    python -m scripts.compare_experiments
    python -m scripts.compare_experiments --top 10
    python -m scripts.compare_experiments --variant baseline
    python -m scripts.compare_experiments --execution-id <FULL_EXECUTION_ID>
    python -m scripts.compare_experiments --mode all  # include smoke checks
    python -m scripts.compare_experiments --output reports/mlflow_comparison.csv

By default, only full-training runs are compared. Quick runs are smoke checks,
not eligible for promotion; inspect them only with --mode quick or --mode all.
The primary ranking metric is validation F1 (metrics.val_f1). This script
never selects models using held-out test metrics and never promotes models.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mlflow
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
METRIC = "metrics.val_f1"
EXECUTION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def load_settings(config_path: Path) -> tuple[str, Path, list[str]]:
    """Read experiment, tracking database, and variant order from YAML."""
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training YAML must contain a mapping.")
    try:
        experiment = config["mlflow"]["experiment"]
        db_path = Path(config["paths"]["mlflow_db"])
        variants = config["variants"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Training YAML requires mlflow.experiment, paths.mlflow_db, and variants."
        ) from exc
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("mlflow.experiment must be a nonempty string.")
    if not str(db_path).strip():
        raise ValueError("paths.mlflow_db must be a nonempty path.")
    if (not isinstance(variants, list) or not variants
            or any(not isinstance(v, str) or not v.strip() for v in variants)
            or len(set(variants)) != len(variants)):
        raise ValueError("variants must be a nonempty list of unique names.")
    # Relative paths follow the same convention as the training runner: repo root.
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    return experiment, db_path.resolve(), variants


def rank_candidates(
    runs: pd.DataFrame,
    *,
    mode: str | None = None,
    variant: str | None = None,
    execution_id: str | None = None,
    variant_order: list[str] | None = None,
) -> pd.DataFrame:
    """Select successful candidate runs with valid F1, sorted best first.

    The training pipeline logs separate 'final_test' records. Requiring a
    candidate family and variant excludes these and unrelated MLflow runs.
    """
    columns = [
        "run_id", "start_time", "tags.model_family", "tags.preprocessing_variant",
        "tags.mode", METRIC, "metrics.val_pr_auc", "metrics.val_roc_auc",
        "params.config_sha256", "params.data_version",
        "tags.training_execution_id", "tags.candidate_index",
    ]
    if runs.empty or METRIC not in runs.columns:
        return pd.DataFrame(columns=columns)

    result = runs.copy()
    if "status" in result.columns:
        result = result.loc[result["status"] == "FINISHED"]
    for required in ("tags.model_family", "tags.preprocessing_variant"):
        if required not in result.columns:
            return pd.DataFrame(columns=columns)
        result = result.loc[result[required].notna() & result[required].astype(str).ne("")]

    if mode is not None:
        if "tags.mode" not in result.columns:
            return pd.DataFrame(columns=columns)
        result = result.loc[result["tags.mode"] == mode]
    if variant is not None:
        result = result.loc[result["tags.preprocessing_variant"] == variant]
    if execution_id is not None:
        if "tags.training_execution_id" not in result.columns:
            return pd.DataFrame(columns=columns)
        result = result.loc[result["tags.training_execution_id"] == execution_id]

    result[METRIC] = pd.to_numeric(result[METRIC], errors="coerce")
    result = result.loc[result[METRIC].between(0, 1, inclusive="both")]
    for tie_breaker in ("metrics.val_pr_auc", "metrics.val_roc_auc"):
        if tie_breaker not in result.columns:
            result[tie_breaker] = float("nan")
        result[tie_breaker] = pd.to_numeric(result[tie_breaker], errors="coerce")

    # train.py chooses the first candidate when all three metrics tie.
    # First means: YAML variant order, then candidate creation order.
    order = variant_order or ["baseline", "robust", "interactions"]
    priority = {name: index for index, name in enumerate(order)}
    result["_variant_priority"] = (
        result["tags.preprocessing_variant"].map(priority).fillna(len(order))
    )
    if "tags.candidate_index" not in result.columns:
        result["tags.candidate_index"] = pd.NA
    result["_candidate_priority"] = pd.to_numeric(
        result["tags.candidate_index"], errors="coerce"
    )
    if "start_time" not in result.columns:
        result["start_time"] = pd.NaT
    if "run_id" not in result.columns:
        result["run_id"] = ""
    result = result.sort_values(
        by=[METRIC, "metrics.val_pr_auc", "metrics.val_roc_auc",
            "_variant_priority", "_candidate_priority", "start_time", "run_id"],
        ascending=[False, False, False, True, True, True, True],
        kind="stable",
        na_position="last",
    )
    return result.drop(columns=["_variant_priority", "_candidate_priority"]).reset_index(drop=True)


def compare(
    *,
    config_path: Path,
    top: int = 10,
    mode: str = "full",
    variant: str | None = None,
    execution_id: str | None = None,
    output: Path | None = None,
) -> pd.DataFrame:
    """Query MLflow, display a leaderboard and optionally export it."""
    if top < 1:
        raise ValueError("--top must be at least 1")
    if execution_id is not None and not EXECUTION_ID_RE.fullmatch(execution_id):
        raise ValueError("--execution-id must be a 32-character lowercase hex ID")
    experiment_name, db_path, variant_order = load_settings(config_path)
    if not db_path.is_file():
        raise FileNotFoundError(
            f"MLflow tracking database not found: {db_path}. "
            "Run training first (python -m src.models.train)."
        )
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name!r}")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="pandas",
        max_results=100000,
    )
    ranked = rank_candidates(
        runs,
        mode=None if mode == "all" else mode,
        variant=variant,
        execution_id=execution_id,
        variant_order=variant_order,
    )
    if ranked.empty:
        raise ValueError("No completed training candidates with valid validation F1 were found.")

    display_columns = [
        "run_id", "tags.model_family", "tags.preprocessing_variant",
        "tags.mode", METRIC, "metrics.val_pr_auc", "metrics.val_roc_auc",
        "params.config_sha256", "params.data_version",
        "tags.training_execution_id",
    ]
    display_columns = [col for col in display_columns if col in ranked.columns]
    print(f"Experiment: {experiment_name}")
    print(f"Tracking DB: {db_path}")
    print(f"Valid candidates: {len(ranked)}")
    if execution_id is not None:
        print(f"Execution ID: {execution_id}")
    print(f"Best run: {ranked.iloc[0]['run_id']}")
    print(f"Best validation F1: {ranked.iloc[0][METRIC]:.4f}")
    print("\nLeaderboard:")
    print(ranked.loc[:, display_columns].head(top).to_string(index=False))

    winner = ranked.iloc[0]
    params = {
        key.removeprefix("params."): value
        for key, value in winner.items()
        if key.startswith("params.") and pd.notna(value)
    }
    print("\nWinning run parameters:")
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}")

    if output is not None:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(output, index=False)
        print(f"\nExported complete leaderboard: {output}")
    return ranked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yml")
    parser.add_argument("--top", type=int, default=10, help="Number of rows to display")
    parser.add_argument(
        "--mode", choices=("full", "quick", "all"), default="full",
        help="Training runs to compare (default: full; quick is only a smoke check)",
    )
    parser.add_argument("--variant", choices=("baseline", "robust", "interactions"))
    parser.add_argument(
        "--execution-id", type=str,
        help="Limit comparison to one full training execution (32 hex characters)",
    )
    parser.add_argument("--output", type=Path, help="Optional CSV leaderboard output")
    args = parser.parse_args()
    try:
        compare(
            config_path=args.config,
            top=args.top,
            mode=args.mode,
            variant=args.variant,
            execution_id=args.execution_id,
            output=args.output,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
