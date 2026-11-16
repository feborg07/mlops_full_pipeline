
"""
Sprint 17 - MLOps Final Project
Module: src/data/monitoring.py

Detect covariate drift using Evidently 0.7.23.

Reference:
    data/drift/control/X_test.csv

Current datasets:
    data/drift/mild/X_test.csv
    data/drift/moderate/X_test.csv
    data/drift/severe/X_test.csv

Outputs:
    reports/drift/<scenario>_drift.html
    reports/drift/<scenario>_drift.json
    reports/drift/drift_summary.json

Exit codes:
    0 - No critical drift
    1 - Critical drift detected
    2 - Monitoring execution error

This module does not train models or evaluate
prediction performance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.data.preprocess import FEATURE_COLUMNS


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

LOGGER = logging.getLogger(__name__)

SCENARIOS = (
    "mild",
    "moderate",
    "severe",
)

NUMERICAL_FEATURES = (
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
)

CATEGORICAL_FEATURES = (
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
)

# Use the same bounded distance measure for both
# numerical and categorical features.
#
# Jensen-Shannon distance:
#   0 = identical distributions
#   larger values = greater distribution difference
#
# A feature is considered drifted when its
# Jensen-Shannon distance exceeds 0.10.

DRIFT_METHOD = "jensenshannon"
FEATURE_DRIFT_THRESHOLD = 0.10

DRIFT_WARNING = 0.20
DRIFT_CRITICAL = 0.35


@dataclass(frozen=True)
class MonitoringConfig:
    """Configuration for the monitoring pipeline."""

    drift_dir: Path = ROOT / "data/drift"
    report_dir: Path = ROOT / "reports/drift"

    warning_threshold: float = DRIFT_WARNING
    critical_threshold: float = DRIFT_CRITICAL

    feature_threshold: float = FEATURE_DRIFT_THRESHOLD

    def __post_init__(self):
        if not (
            0 <= self.warning_threshold
            < self.critical_threshold
            <= 1
        ):
            raise ValueError(
                "Thresholds must satisfy: "
                "0 <= warning < critical <= 1."
            )

        if not 0 < self.feature_threshold < 1:
            raise ValueError(
                "Feature drift threshold must be between 0 and 1."
            )


# ============================================================
# 2. Data validation and loading
# ============================================================

def validate_monitoring_data(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> None:
    """
    Validate feature schemas and dataset integrity.

    Drift monitoring requires the same feature columns,
    but does not require equal row counts.
    """

    expected = set(FEATURE_COLUMNS)

    for name, frame in (
        ("reference", reference),
        ("current", current),
    ):
        if frame.empty:
            raise ValueError(
                f"{name} dataset cannot be empty."
            )

        if not frame.columns.is_unique:
            raise ValueError(
                f"{name} contains duplicate column names."
            )

        if set(frame.columns) != expected:
            missing = sorted(
                expected - set(frame.columns)
            )

            unexpected = sorted(
                set(frame.columns) - expected
            )

            raise ValueError(
                f"Invalid {name} schema. "
                f"Missing: {missing}. "
                f"Unexpected: {unexpected}."
            )

        # Reject nonnumeric values in numerical features.
        for column in NUMERICAL_FEATURES:
            converted = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            invalid = (
                frame[column].notna()
                & converted.isna()
            )

            if invalid.any():
                raise ValueError(
                    f"{name}.{column} contains "
                    "nonnumeric values."
                )


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load one raw-feature dataset from CSV."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Monitoring dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"Monitoring dataset is empty: {path}"
        )

    return df


def prepare_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize column order and feature types.

    The original DataFrame is not modified.
    """

    result = frame.loc[
        :,
        FEATURE_COLUMNS,
    ].copy(deep=True)

    for column in NUMERICAL_FEATURES:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

    for column in CATEGORICAL_FEATURES:
        result[column] = (
            result[column]
            .astype("object")
            .where(
                result[column].notna(),
                None,
            )
        )

    return result


# ============================================================
# 3. Evidently dataset preparation
# ============================================================

def create_evidently_dataset(
    frame: pd.DataFrame,
    definition: DataDefinition,
) -> Dataset:
    """
    Convert a pandas DataFrame to an Evidently Dataset.

    Explicit feature typing prevents numeric variables
    from being interpreted as categorical merely
    because they have few distinct values.
    """

    return Dataset.from_pandas(
        frame,
        data_definition=definition,
    )


def create_data_definition() -> DataDefinition:
    """Define the 14 original Adult Income feature types."""

    return DataDefinition(
        numerical_columns=list(NUMERICAL_FEATURES),
        categorical_columns=list(CATEGORICAL_FEATURES),
    )


# ============================================================
# 4. Alert classification
# ============================================================

def classify_drift(
    drift_share: float,
    warning_threshold: float = DRIFT_WARNING,
    critical_threshold: float = DRIFT_CRITICAL,
) -> str:
    """
    Convert drifted-feature share into an alert level.

    Critical: share > 35%
    Warning:  share > 20%
    Normal:   otherwise
    """

    if not 0 <= drift_share <= 1:
        raise ValueError(
            "Drift share must be between 0 and 1."
        )

    if not 0 <= warning_threshold < critical_threshold <= 1:
        raise ValueError(
            "Invalid warning/critical thresholds."
        )

    if drift_share >= critical_threshold:
        return "critical"

    if drift_share > warning_threshold:
        return "warning"

    return "normal"


# ============================================================
# 5. Extract Evidently results
# ============================================================

def extract_drift_metrics(
    snapshot: Any,
    expected_features: int,
) -> dict:
    """
    Extract drift statistics from an Evidently snapshot.

    Uses the DriftedColumnsCount metric.

    Fail explicitly if the expected metric is absent.
    Never silently interpret missing metrics as zero drift.
    """

    result = snapshot.dict()

    metrics = result.get("metrics")

    if not isinstance(metrics, list):
        raise ValueError(
            "Evidently snapshot has no metrics list."
        )

    matches = [
        metric
        for metric in metrics
        if (
            isinstance(metric, dict)
            and "DriftedColumnsCount"
            in str(metric.get("metric_name", ""))
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one DriftedColumnsCount "
            f"metric; found {len(matches)}."
        )

    value = matches[0].get("value")

    if not isinstance(value, dict):
        raise ValueError(
            "DriftedColumnsCount returned an invalid value."
        )

    if "count" not in value or "share" not in value:
        raise ValueError(
            "DriftedColumnsCount must contain "
            "'count' and 'share'."
        )

    count = int(value["count"])
    share = float(value["share"])

    if not 0 <= count <= expected_features:
        raise ValueError(
            "Invalid drifted feature count."
        )

    if not 0 <= share <= 1:
        raise ValueError(
            "Invalid drift share."
        )

    calculated_share = count / expected_features

    if abs(share - calculated_share) > 1e-6:
        raise ValueError(
            "Evidently drift share is inconsistent "
            "with the monitored feature count."
        )

    return {
        "total_features": expected_features,
        "drifted_features": count,
        "drift_share": share,
    }


# ============================================================
# 6. Run one drift comparison
# ============================================================

def run_drift_analysis(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    scenario: str,
    config: MonitoringConfig,
) -> dict:
    """
    Run Evidently drift detection for one scenario.

    Save an HTML report and a complete Evidently JSON
    snapshot, then return a compact summary.
    """

    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unsupported scenario: {scenario}"
        )

    validate_monitoring_data(
        reference,
        current,
    )

    reference = prepare_features(reference)
    current = prepare_features(current)

    definition = create_data_definition()

    reference_dataset = create_evidently_dataset(
        reference,
        definition,
    )

    current_dataset = create_evidently_dataset(
        current,
        definition,
    )

    # Explicitly configure the statistical method.
    #
    # Using Jensen-Shannon distance consistently avoids
    # mixing p-values with distance-based drift scores.

    preset = DataDriftPreset(
        columns=list(FEATURE_COLUMNS),
        num_method=DRIFT_METHOD,
        cat_method=DRIFT_METHOD,
        num_threshold=config.feature_threshold,
        cat_threshold=config.feature_threshold,
        include_tests=False,
    )

    report = Report(
        metrics=[preset],
        include_tests=False,
    )

    # Evidently 0.7.x:
    # First argument = current dataset.
    # Second argument = reference dataset.

    snapshot = report.run(
        current_dataset,
        reference_dataset,
    )

    statistics = extract_drift_metrics(
        snapshot,
        expected_features=len(FEATURE_COLUMNS),
    )

    status = classify_drift(
        statistics["drift_share"],
        warning_threshold=config.warning_threshold,
        critical_threshold=config.critical_threshold,
    )

    config.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    html_path = (
        config.report_dir
        / f"{scenario}_drift.html"
    )

    json_path = (
        config.report_dir
        / f"{scenario}_drift.json"
    )

    # Save the complete Evidently report.
    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    summary = {
        "scenario": scenario,
        "reference_rows": len(reference),
        "current_rows": len(current),
        **statistics,
        "status": status,
        "html_report": str(html_path),
        "json_report": str(json_path),
    }

    LOGGER.info(
        "%s: %d/%d drifted features (%.2f%%) - %s",
        scenario.upper(),
        statistics["drifted_features"],
        statistics["total_features"],
        statistics["drift_share"] * 100,
        status.upper(),
    )

    return summary


# ============================================================
# 7. Run all scenarios
# ============================================================

def run_monitoring_pipeline(
    config: MonitoringConfig,
) -> dict:
    """
    Compare all generated drift scenarios with control.

    Returns a combined summary with overall status.
    """

    reference_path = (
        config.drift_dir
        / "control"
        / "X_test.csv"
    )

    reference = load_dataset(reference_path)

    results = {}

    for scenario in SCENARIOS:
        current_path = (
            config.drift_dir
            / scenario
            / "X_test.csv"
        )

        current = load_dataset(current_path)

        results[scenario] = run_drift_analysis(
            reference=reference,
            current=current,
            scenario=scenario,
            config=config,
        )

    statuses = [
        result["status"]
        for result in results.values()
    ]

    if "critical" in statuses:
        overall_status = "critical"
    elif "warning" in statuses:
        overall_status = "warning"
    else:
        overall_status = "normal"

    summary = {
        "monitoring_type": "covariate_drift",
        "model_variant": "baseline",
        "method": DRIFT_METHOD,
        "feature_threshold": config.feature_threshold,
        "warning_threshold": config.warning_threshold,
        "critical_threshold": config.critical_threshold,
        "total_features": len(FEATURE_COLUMNS),
        "overall_status": overall_status,
        "scenarios": results,
    }

    config.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        config.report_dir
        / "drift_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return summary


# ============================================================
# 8. Command-line interface
# ============================================================

def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Monitor synthetic Adult Income covariate "
            "drift using Evidently."
        )
    )

    parser.add_argument(
        "--drift-dir",
        type=Path,
        default=ROOT / "data/drift",
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "reports/drift",
    )

    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=DRIFT_WARNING,
    )

    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=DRIFT_CRITICAL,
    )

    parser.add_argument(
        "--feature-threshold",
        type=float,
        default=FEATURE_DRIFT_THRESHOLD,
    )

    return parser.parse_args(argv)


# ============================================================
# 9. Main execution
# ============================================================

def main(
    argv: list[str] | None = None,
) -> int:
    """
    Execute monitoring and return a CI-compatible exit code.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        args = parse_args(argv)

        config = MonitoringConfig(
            drift_dir=args.drift_dir,
            report_dir=args.report_dir,
            warning_threshold=args.warning_threshold,
            critical_threshold=args.critical_threshold,
            feature_threshold=args.feature_threshold,
        )

        summary = run_monitoring_pipeline(config)

        print("\n" + "=" * 55)
        print("SPRINT 17 - DATA DRIFT MONITORING")
        print("=" * 55)

        for scenario, result in summary["scenarios"].items():
            print(
                f"{scenario.upper():10s} | "
                f"{result['drifted_features']:2d}/"
                f"{result['total_features']:2d} features | "
                f"{result['drift_share']:.2%} | "
                f"{result['status'].upper()}"
            )

        print("-" * 55)

        print(
            "OVERALL STATUS:",
            summary["overall_status"].upper(),
        )

        print(
            "Reports:",
            config.report_dir,
        )

        if summary["overall_status"] == "critical":
            return 1

        return 0

    except Exception:
        LOGGER.exception(
            "Data drift monitoring failed."
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
