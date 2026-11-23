
"""
Sprint 17 - MLOps Final Project
Module: tests/test_monitoring.py

Tests for Evidently 0.7.23 data drift monitoring.

Coverage:
    1. Monitoring configuration
    2. Dataset loading and validation
    3. Feature preparation
    4. Evidently dataset creation
    5. Alert classification
    6. Drift metric extraction
    7. Drift analysis integration
    8. Report serialization
    9. Monitoring pipeline
    10. CLI exit codes

Synthetic fixtures avoid dependencies on real datasets,
trained models, MLflow, or external services.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evidently import Dataset

from src.data import monitoring as mon
from src.data.preprocess import FEATURE_COLUMNS


# ============================================================
# 1. Fixtures
# ============================================================

@pytest.fixture
def reference_data():
    """
    Create a synthetic Adult Income feature dataset.

    Include variation in numerical and categorical
    features to support meaningful drift comparisons.
    """

    rng = np.random.default_rng(42)
    n = 200

    df = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "fnlwgt": rng.integers(10000, 400000, n),
        "education-num": rng.integers(1, 17, n),
        "capital-gain": rng.choice(
            [0, 1000, 5000, 10000],
            size=n,
            p=[0.8, 0.1, 0.07, 0.03],
        ),
        "capital-loss": rng.choice(
            [0, 500, 1500],
            size=n,
            p=[0.9, 0.07, 0.03],
        ),
        "hours-per-week": rng.integers(20, 65, n),
        "workclass": rng.choice(
            ["Private", "Self-emp-not-inc", "Federal-gov"],
            size=n,
        ),
        "education": rng.choice(
            ["HS-grad", "Bachelors", "Masters"],
            size=n,
        ),
        "marital-status": rng.choice(
            ["Never-married", "Married-civ-spouse"],
            size=n,
        ),
        "occupation": rng.choice(
            ["Sales", "Exec-managerial", "Craft-repair"],
            size=n,
        ),
        "relationship": rng.choice(
            ["Husband", "Not-in-family", "Own-child"],
            size=n,
        ),
        "race": rng.choice(
            ["White", "Black", "Asian-Pac-Islander"],
            size=n,
        ),
        "sex": rng.choice(
            ["Male", "Female"],
            size=n,
        ),
        "native-country": rng.choice(
            ["United-States", "Mexico", "Canada"],
            size=n,
        ),
    })

    return df.loc[:, FEATURE_COLUMNS]


@pytest.fixture
def current_data(reference_data):
    """Create a clearly shifted current dataset."""

    df = reference_data.copy(deep=True)

    df["age"] = np.clip(
        df["age"] + 15,
        17,
        90,
    )

    df["hours-per-week"] = np.clip(
        df["hours-per-week"] + 20,
        1,
        99,
    )

    df["occupation"] = "Exec-managerial"

    return df


@pytest.fixture
def config(tmp_path):
    """Create monitoring configuration with temporary paths."""

    return mon.MonitoringConfig(
        drift_dir=tmp_path / "data" / "drift",
        report_dir=tmp_path / "reports" / "drift",
    )


@pytest.fixture
def scenario_files(reference_data, current_data, config):
    """
    Create a complete synthetic drift directory.

    Control uses the reference dataset.
    Other scenarios use deterministic transformations.
    """

    datasets = {
        "control": reference_data,
        "mild": reference_data.copy(deep=True),
        "moderate": current_data.copy(deep=True),
        "severe": current_data.copy(deep=True),
    }

    for scenario, df in datasets.items():
        directory = config.drift_dir / scenario

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            directory / "X_test.csv",
            index=False,
        )

    return config


@pytest.fixture
def fake_snapshot():
    """
    Build a small snapshot-like object for testing
    metric extraction without running Evidently.
    """

    def factory(count, share, total=14):
        # Deterministic per-column metrics matching Evidently 0.7.23 JSON.
        scores = [0.2 if i < min(count, total) else 0.0 for i in range(total)]
        metrics = [{
            "metric_name": "DriftedColumnsCount(columns=None)",
            "value": {"count": count, "share": share},
        }]
        metrics += [{
            "metric_name": f"ValueDrift(column={column})",
            "config": {
                "type": "evidently:metric_v2:ValueDrift",
                "column": column,
                "threshold": 0.1,
            },
            "value": score,
        } for column, score in zip(FEATURE_COLUMNS, scores)]
        return SimpleNamespace(dict=lambda: {"metrics": metrics})

    return factory


# ============================================================
# 2. Configuration tests
# ============================================================

def test_default_thresholds():
    config = mon.MonitoringConfig()

    assert config.warning_threshold == 0.20
    assert config.critical_threshold == 0.35
    assert config.feature_threshold == 0.10


@pytest.mark.parametrize(
    "warning,critical",
    [
        (-0.1, 0.35),
        (0.20, 1.1),
        (0.35, 0.35),
        (0.40, 0.35),
        (1.0, 1.0),
    ],
)
def test_invalid_alert_thresholds(
    tmp_path,
    warning,
    critical,
):
    with pytest.raises(ValueError):
        mon.MonitoringConfig(
            drift_dir=tmp_path,
            report_dir=tmp_path,
            warning_threshold=warning,
            critical_threshold=critical,
        )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 0, 1, 1.1],
)
def test_invalid_feature_threshold(
    tmp_path,
    threshold,
):
    with pytest.raises(ValueError):
        mon.MonitoringConfig(
            drift_dir=tmp_path,
            report_dir=tmp_path,
            feature_threshold=threshold,
        )


# ============================================================
# 3. Dataset loading tests
# ============================================================

def test_load_dataset(reference_data, tmp_path):
    path = tmp_path / "reference.csv"

    reference_data.to_csv(path, index=False)

    loaded = mon.load_dataset(path)

    pd.testing.assert_frame_equal(
        loaded,
        reference_data,
        check_dtype=False,
    )


def test_load_dataset_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        mon.load_dataset(
            tmp_path / "missing.csv"
        )


def test_load_dataset_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"

    pd.DataFrame(
        columns=FEATURE_COLUMNS
    ).to_csv(path, index=False)

    with pytest.raises(ValueError):
        mon.load_dataset(path)


# ============================================================
# 4. Dataset validation tests
# ============================================================

def test_valid_monitoring_data(
    reference_data,
    current_data,
):
    mon.validate_monitoring_data(
        reference_data,
        current_data,
    )


def test_validation_allows_different_row_counts(
    reference_data,
):
    current = reference_data.iloc[:100].copy()

    mon.validate_monitoring_data(
        reference_data,
        current,
    )


def test_validation_allows_reordered_columns(
    reference_data,
):
    current = reference_data[
        list(reversed(reference_data.columns))
    ]

    mon.validate_monitoring_data(
        reference_data,
        current,
    )


@pytest.mark.parametrize(
    "dataset",
    ["reference", "current"],
)
def test_validation_rejects_missing_feature(
    reference_data,
    dataset,
):
    invalid = reference_data.drop(
        columns=["age"]
    )

    reference = (
        invalid
        if dataset == "reference"
        else reference_data
    )

    current = (
        invalid
        if dataset == "current"
        else reference_data
    )

    with pytest.raises(ValueError):
        mon.validate_monitoring_data(
            reference,
            current,
        )


def test_validation_rejects_extra_feature(reference_data):
    invalid = reference_data.copy()

    invalid["unexpected"] = 1

    with pytest.raises(ValueError):
        mon.validate_monitoring_data(
            reference_data,
            invalid,
        )


def test_validation_rejects_empty_dataset(reference_data):
    empty = reference_data.iloc[0:0]

    with pytest.raises(ValueError):
        mon.validate_monitoring_data(
            reference_data,
            empty,
        )


def test_validation_rejects_nonnumeric_feature(reference_data):
    invalid = reference_data.copy()

    invalid["age"] = invalid["age"].astype(object)

    invalid.loc[0, "age"] = "invalid"

    with pytest.raises(ValueError):
        mon.validate_monitoring_data(
            reference_data,
            invalid,
        )


def test_validation_rejects_duplicate_columns(reference_data):
    invalid = reference_data.copy()

    invalid.columns = [
        "age" if col == "fnlwgt" else col
        for col in invalid.columns
    ]

    with pytest.raises(ValueError):
        mon.validate_monitoring_data(
            reference_data,
            invalid,
        )


# ============================================================
# 5. Feature preparation tests
# ============================================================

def test_prepare_features_preserves_schema(reference_data):
    prepared = mon.prepare_features(
        reference_data
    )

    assert list(prepared.columns) == FEATURE_COLUMNS


def test_prepare_features_reorders_columns(reference_data):
    reversed_df = reference_data[
        list(reversed(reference_data.columns))
    ]

    prepared = mon.prepare_features(
        reversed_df
    )

    assert list(prepared.columns) == FEATURE_COLUMNS


def test_prepare_features_preserves_row_count(reference_data):
    prepared = mon.prepare_features(
        reference_data
    )

    assert len(prepared) == len(reference_data)


def test_prepare_features_does_not_mutate_input(reference_data):
    original = reference_data.copy(deep=True)

    mon.prepare_features(reference_data)

    pd.testing.assert_frame_equal(
        reference_data,
        original,
    )


def test_prepare_features_numeric_types(reference_data):
    prepared = mon.prepare_features(
        reference_data
    )

    for column in mon.NUMERICAL_FEATURES:
        assert pd.api.types.is_numeric_dtype(
            prepared[column]
        )


def test_prepare_features_preserves_missing_values(reference_data):
    df = reference_data.copy()

    df.loc[0:9, "workclass"] = np.nan

    prepared = mon.prepare_features(df)

    assert prepared["workclass"].isna().sum() == 10


# ============================================================
# 6. Evidently integration tests
# ============================================================

def test_data_definition_has_expected_features():
    assert len(mon.NUMERICAL_FEATURES) == 6
    assert len(mon.CATEGORICAL_FEATURES) == 8

    assert set(
        mon.NUMERICAL_FEATURES
    ).isdisjoint(
        mon.CATEGORICAL_FEATURES
    )

    assert (
        set(mon.NUMERICAL_FEATURES)
        | set(mon.CATEGORICAL_FEATURES)
    ) == set(FEATURE_COLUMNS)


def test_create_evidently_dataset(reference_data):
    definition = mon.create_data_definition()

    prepared = mon.prepare_features(
        reference_data
    )

    dataset = mon.create_evidently_dataset(
        prepared,
        definition,
    )

    assert isinstance(dataset, Dataset)


# ============================================================
# 7. Alert classification tests
# ============================================================

@pytest.mark.parametrize(
    "share,expected",
    [
        (0.00, "normal"),
        (0.10, "normal"),
        (0.20, "normal"),
        (0.200001, "warning"),
        (0.214285714, "warning"),
        (0.30, "warning"),
        (0.35, "critical"),
        (0.350001, "critical"),
        (0.357142857, "critical"),
        (0.50, "critical"),
        (1.00, "critical"),
    ],
)
def test_classify_drift_boundaries(
    share,
    expected,
):
    assert mon.classify_drift(
        share,
        warning_threshold=0.20,
        critical_threshold=0.35,
    ) == expected


@pytest.mark.parametrize(
    "share",
    [-0.1, 1.1],
)
def test_classify_drift_rejects_invalid_share(share):
    with pytest.raises(ValueError):
        mon.classify_drift(share)


def test_classify_drift_custom_thresholds():
    assert mon.classify_drift(
        0.50,
        warning_threshold=0.30,
        critical_threshold=0.60,
    ) == "warning"


# ============================================================
# 8. Evidently metric extraction tests
# ============================================================

def test_extract_valid_metrics(fake_snapshot):
    snapshot = fake_snapshot(
        count=5,
        share=5 / 14,
    )

    result = mon.extract_drift_metrics(
        snapshot,
        expected_features=14,
    )

    assert result["total_features"] == 14
    assert result["drifted_features"] == 5
    assert result["drifted_feature_names"] == FEATURE_COLUMNS[:5]

    assert result["drift_share"] == pytest.approx(
        5 / 14
    )


def test_extract_zero_drift(fake_snapshot):
    snapshot = fake_snapshot(
        count=0,
        share=0,
    )

    result = mon.extract_drift_metrics(
        snapshot,
        expected_features=14,
    )

    assert result["drifted_features"] == 0
    assert result["drift_share"] == 0
    assert result["drifted_feature_names"] == []


def test_extract_rejects_inconsistent_share(fake_snapshot):
    snapshot = fake_snapshot(
        count=5,
        share=0.10,
    )

    with pytest.raises(ValueError):
        mon.extract_drift_metrics(
            snapshot,
            expected_features=14,
        )


def test_extract_rejects_invalid_count(fake_snapshot):
    snapshot = fake_snapshot(
        count=20,
        share=1.0,
    )

    with pytest.raises(ValueError):
        mon.extract_drift_metrics(
            snapshot,
            expected_features=14,
        )


def test_extract_rejects_missing_metric():
    snapshot = SimpleNamespace(
        dict=lambda: {
            "metrics": []
        }
    )

    with pytest.raises(ValueError):
        mon.extract_drift_metrics(
            snapshot,
            expected_features=14,
        )


def test_extract_rejects_missing_values():
    snapshot = SimpleNamespace(
        dict=lambda: {
            "metrics": [
                {
                    "metric_name": "DriftedColumnsCount",
                    "value": {},
                }
            ]
        }
    )

    with pytest.raises(ValueError):
        mon.extract_drift_metrics(
            snapshot,
            expected_features=14,
        )



def test_extract_real_evidently_snapshot():
    """Regression test against the supplied Evidently 0.7.23 JSON structure."""
    from pathlib import Path
    # Fixture generated in-memory to avoid depending on an external report file.
    metrics = [{"metric_name": "DriftedColumnsCount", "value": {"count": 3, "share": 3 / 14}}]
    drifted = {"education-num", "hours-per-week", "occupation"}
    for column in FEATURE_COLUMNS:
        metrics.append({
            "config": {"type": "evidently:metric_v2:ValueDrift", "column": column, "threshold": 0.1},
            "value": 0.2 if column in drifted else 0.0,
        })
    result = mon.extract_drift_metrics(SimpleNamespace(dict=lambda: {"metrics": metrics}), 14)
    assert set(result["drifted_feature_names"]) == drifted

# ============================================================
# 9. Drift analysis integration tests
# ============================================================

def test_run_drift_analysis_creates_reports(
    reference_data,
    current_data,
    config,
):
    result = mon.run_drift_analysis(
        reference=reference_data,
        current=current_data,
        scenario="moderate",
        config=config,
    )

    assert result["scenario"] == "moderate"
    assert len(result["drifted_feature_names"]) == result["drifted_features"]
    assert {"age", "hours-per-week", "occupation"}.issubset(set(result["drifted_feature_names"]))

    assert result["total_features"] == 14

    assert (
        config.report_dir / "moderate_drift.html"
    ).is_file()

    assert (
        config.report_dir / "moderate_drift.json"
    ).is_file()


def test_identical_distributions_have_zero_drift(
    reference_data,
    config,
):
    result = mon.run_drift_analysis(
        reference=reference_data,
        current=reference_data.copy(),
        scenario="mild",
        config=config,
    )

    assert result["drifted_features"] == 0
    assert result["drift_share"] == pytest.approx(0)
    assert result["status"] == "normal"
    assert result["drifted_feature_names"] == []


def test_run_drift_analysis_rejects_invalid_scenario(
    reference_data,
    config,
):
    with pytest.raises(ValueError):
        mon.run_drift_analysis(
            reference=reference_data,
            current=reference_data,
            scenario="unknown",
            config=config,
        )


def test_run_drift_analysis_does_not_mutate_inputs(
    reference_data,
    current_data,
    config,
):
    original_reference = reference_data.copy(deep=True)
    original_current = current_data.copy(deep=True)

    mon.run_drift_analysis(
        reference_data,
        current_data,
        scenario="severe",
        config=config,
    )

    pd.testing.assert_frame_equal(
        reference_data,
        original_reference,
    )

    pd.testing.assert_frame_equal(
        current_data,
        original_current,
    )


# ============================================================
# 10. Monitoring pipeline tests
# ============================================================

def test_pipeline_generates_all_reports(scenario_files):
    summary = mon.run_monitoring_pipeline(
        scenario_files
    )

    assert set(summary["scenarios"]) == set(
        mon.SCENARIOS
    )

    for scenario in mon.SCENARIOS:
        assert (
            scenario_files.report_dir
            / f"{scenario}_drift.html"
        ).is_file()

        assert (
            scenario_files.report_dir
            / f"{scenario}_drift.json"
        ).is_file()

    assert (
        scenario_files.report_dir
        / "drift_summary.json"
    ).is_file()


def test_pipeline_summary_is_valid_json(scenario_files):
    summary = mon.run_monitoring_pipeline(
        scenario_files
    )

    path = (
        scenario_files.report_dir
        / "drift_summary.json"
    )

    saved = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert saved == summary

    assert saved["model_variant"] == "baseline"
    assert saved["monitoring_type"] == "covariate_drift"


def test_pipeline_is_reproducible(scenario_files):
    first = mon.run_monitoring_pipeline(
        scenario_files
    )

    second = mon.run_monitoring_pipeline(
        scenario_files
    )

    for scenario in mon.SCENARIOS:
        assert (
            first["scenarios"][scenario]["drifted_features"]
            == second["scenarios"][scenario]["drifted_features"]
        )

        assert (
            first["scenarios"][scenario]["drift_share"]
            == second["scenarios"][scenario]["drift_share"]
        )


def test_pipeline_rejects_missing_scenario(
    scenario_files,
):
    missing_path = (
        scenario_files.drift_dir
        / "moderate"
        / "X_test.csv"
    )

    missing_path.unlink()

    with pytest.raises(FileNotFoundError):
        mon.run_monitoring_pipeline(
            scenario_files
        )


# ============================================================
# 11. CLI tests
# ============================================================

def test_parse_args_defaults():
    args = mon.parse_args([])

    assert args.warning_threshold == 0.20
    assert args.critical_threshold == 0.35
    assert args.feature_threshold == 0.10


def test_parse_args_custom_thresholds():
    args = mon.parse_args([
        "--warning-threshold", "0.15",
        "--critical-threshold", "0.50",
        "--feature-threshold", "0.12",
    ])

    assert args.warning_threshold == 0.15
    assert args.critical_threshold == 0.50
    assert args.feature_threshold == 0.12


@pytest.mark.parametrize(
    "status,expected_exit_code",
    [
        ("normal", 0),
        ("warning", 0),
        ("critical", 1),
    ],
)
def test_main_exit_codes(
    monkeypatch,
    tmp_path,
    status,
    expected_exit_code,
):
    """
    Mock the monitoring pipeline to isolate CLI behavior.

    A critical alert must return exit code 1.
    """

    monkeypatch.setattr(
        mon,
        "run_monitoring_pipeline",
        lambda config: {
            "overall_status": status,
            "scenarios": {
                "mild": {
                    "drifted_features": 0,
                    "drifted_feature_names": [],
                    "total_features": 14,
                    "drift_share": 0.0,
                    "status": "normal",
                },
                "moderate": {
                    "drifted_features": 3,
                    "drifted_feature_names": ["education-num", "hours-per-week", "occupation"],
                    "total_features": 14,
                    "drift_share": 3 / 14,
                    "status": "warning",
                },
                "severe": {
                    "drifted_features": 5,
                    "drifted_feature_names": ["age", "education-num", "hours-per-week", "workclass", "occupation"],
                    "total_features": 14,
                    "drift_share": 5 / 14,
                    "status": status,
                },
            },
        },
    )

    exit_code = mon.main([
        "--drift-dir",
        str(tmp_path / "drift"),
        "--report-dir",
        str(tmp_path / "reports"),
    ])

    assert exit_code == expected_exit_code


def test_main_returns_error_code_on_exception(
    monkeypatch,
    tmp_path,
):
    """
    Unexpected monitoring failures must return code 2,
    rather than being reported as normal drift.
    """

    def raise_error(config):
        raise RuntimeError(
            "Simulated Evidently failure"
        )

    monkeypatch.setattr(
        mon,
        "run_monitoring_pipeline",
        raise_error,
    )

    exit_code = mon.main([
        "--drift-dir",
        str(tmp_path / "drift"),
        "--report-dir",
        str(tmp_path / "reports"),
    ])

    assert exit_code == 2
