
"""
Sprint 17 - MLOps Final Project
Module: tests/test_drift_generator.py

Unit and integration tests for synthetic covariate drift.

Tests are deterministic and do not require the actual
Adult Income dataset, MLflow, or Evidently.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import FunctionTransformer

from src.data import drift_generator as dg
from src.data.preprocess import (
    FEATURE_COLUMNS,
    PreprocessingConfig,
    build_preprocessor,
)


# ============================================================
# 1. Fixtures
# ============================================================

@pytest.fixture
def reference_data():
    """
    Create realistic synthetic Adult Income features.

    Include sufficient categorical diversity for
    category replacement and severity tests.
    """

    n = 200

    rng = np.random.default_rng(123)

    df = pd.DataFrame({
        "age": rng.integers(20, 65, n),
        "fnlwgt": rng.integers(10000, 400000, n),
        "education-num": rng.integers(2, 13, n),
        "capital-gain": np.zeros(n, dtype=int),
        "capital-loss": np.zeros(n, dtype=int),
        "hours-per-week": rng.integers(20, 60, n),
        "workclass": rng.choice(
            ["Private", "Self-emp-not-inc", "Federal-gov"],
            size=n,
            p=[0.5, 0.3, 0.2],
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
            ["Exec-managerial", "Sales", "Craft-repair"],
            size=n,
            p=[0.2, 0.4, 0.4],
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
def reference_with_missing(reference_data):
    """Add missing categorical values to the reference."""

    df = reference_data.copy(deep=True)

    df.loc[0:9, "workclass"] = np.nan
    df.loc[10:19, "occupation"] = np.nan
    df.loc[20:29, "native-country"] = np.nan

    return df


@pytest.fixture
def labels(reference_data):
    """Create deterministic binary target labels."""

    return pd.Series(
        np.arange(len(reference_data)) % 2,
        name="income",
    )


@pytest.fixture
def rng():
    """Return a fresh random generator."""

    return np.random.default_rng(42)


# ============================================================
# 2. Configuration tests
# ============================================================

def test_default_configuration():
    config = dg.DriftConfig()

    assert config.random_state == 42
    assert config.test_size == 0.20
    assert config.validation_size == 0.20
    assert config.duplicate_aware_split is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"random_state": -1},
        {"random_state": 2**32},
        {"random_state": True},
        {"test_size": 0},
        {"test_size": 1},
        {"validation_size": 0},
        {"validation_size": 1},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        dg.DriftConfig(**kwargs)


# ============================================================
# 3. Fingerprint tests
# ============================================================

def test_dataframe_fingerprint_is_deterministic(reference_data):
    first = dg.dataframe_sha256(reference_data)
    second = dg.dataframe_sha256(reference_data.copy())

    assert first == second
    assert len(first) == 64


def test_dataframe_fingerprint_detects_value_change(reference_data):
    modified = reference_data.copy()

    modified.loc[0, "age"] += 1

    assert (
        dg.dataframe_sha256(reference_data)
        != dg.dataframe_sha256(modified)
    )


def test_dataframe_fingerprint_detects_row_order(reference_data):
    reordered = reference_data.iloc[::-1].reset_index(drop=True)

    assert (
        dg.dataframe_sha256(reference_data)
        != dg.dataframe_sha256(reordered)
    )


def test_dataframe_fingerprint_detects_column_order(reference_data):
    reordered = reference_data[
        list(reversed(reference_data.columns))
    ]

    assert (
        dg.dataframe_sha256(reference_data)
        != dg.dataframe_sha256(reordered)
    )


# ============================================================
# 4. Numerical drift tests
# ============================================================

@pytest.mark.parametrize(
    "column,shift",
    [
        ("age", 5),
        ("education-num", 2),
        ("hours-per-week", 10),
    ],
)
def test_numeric_drift_changes_selected_rows(
    reference_data,
    column,
    shift,
):
    fraction = 0.25

    drifted = dg.apply_numeric_drift(
        reference_data,
        column=column,
        fraction=fraction,
        shift=shift,
        rng=np.random.default_rng(42),
    )

    changed = (
        reference_data[column] != drifted[column]
    ).sum()

    expected = round(len(reference_data) * fraction)

    assert changed == expected


@pytest.mark.parametrize(
    "column,bounds",
    list(dg.NUMERIC_BOUNDS.items()),
)
def test_numeric_drift_respects_bounds(
    reference_data,
    column,
    bounds,
):
    drifted = dg.apply_numeric_drift(
        reference_data,
        column=column,
        fraction=1.0,
        shift=1000,
        rng=np.random.default_rng(42),
    )

    minimum, maximum = bounds

    assert drifted[column].between(
        minimum,
        maximum,
    ).all()


def test_numeric_drift_preserves_missing_values(reference_data):
    df = reference_data.copy()

    df.loc[0:9, "age"] = np.nan

    drifted = dg.apply_numeric_drift(
        df,
        column="age",
        fraction=0.5,
        shift=5,
        rng=np.random.default_rng(42),
    )

    pd.testing.assert_series_equal(
        df["age"].isna(),
        drifted["age"].isna(),
    )


def test_numeric_drift_zero_fraction(reference_data, rng):
    drifted = dg.apply_numeric_drift(
        reference_data,
        column="age",
        fraction=0,
        shift=10,
        rng=rng,
    )

    pd.testing.assert_frame_equal(
        reference_data,
        drifted,
    )


def test_numeric_drift_does_not_mutate_input(reference_data, rng):
    original = reference_data.copy(deep=True)

    dg.apply_numeric_drift(
        reference_data,
        column="age",
        fraction=0.5,
        shift=10,
        rng=rng,
    )

    pd.testing.assert_frame_equal(
        reference_data,
        original,
    )


@pytest.mark.parametrize(
    "fraction",
    [-0.1, 1.1],
)
def test_numeric_drift_rejects_invalid_fraction(
    reference_data,
    fraction,
    rng,
):
    with pytest.raises(ValueError):
        dg.apply_numeric_drift(
            reference_data,
            column="age",
            fraction=fraction,
            shift=5,
            rng=rng,
        )


def test_numeric_drift_rejects_unsupported_column(
    reference_data,
    rng,
):
    with pytest.raises(ValueError):
        dg.apply_numeric_drift(
            reference_data,
            column="fnlwgt",
            fraction=0.5,
            shift=5,
            rng=rng,
        )


# ============================================================
# 5. Categorical drift tests
# ============================================================

def test_categorical_drift_increases_target_category(
    reference_data,
    rng,
):
    original_count = (
        reference_data["occupation"] == "Exec-managerial"
    ).sum()

    drifted = dg.apply_categorical_drift(
        reference_data,
        column="occupation",
        fraction=0.30,
        target_category="Exec-managerial",
        rng=rng,
    )

    new_count = (
        drifted["occupation"] == "Exec-managerial"
    ).sum()

    assert new_count > original_count


def test_categorical_drift_changes_expected_eligible_rows(
    reference_data,
    rng,
):
    eligible = (
        reference_data["workclass"] != "Private"
    ).sum()

    fraction = 0.25

    drifted = dg.apply_categorical_drift(
        reference_data,
        column="workclass",
        fraction=fraction,
        target_category="Private",
        rng=rng,
    )

    changed = (
        reference_data["workclass"] != drifted["workclass"]
    ).sum()

    assert changed == round(eligible * fraction)


def test_categorical_drift_preserves_missing_values(
    reference_with_missing,
    rng,
):
    drifted = dg.apply_categorical_drift(
        reference_with_missing,
        column="workclass",
        fraction=0.50,
        target_category="Private",
        rng=rng,
    )

    pd.testing.assert_series_equal(
        reference_with_missing["workclass"].isna(),
        drifted["workclass"].isna(),
    )


def test_categorical_drift_zero_fraction(reference_data, rng):
    drifted = dg.apply_categorical_drift(
        reference_data,
        column="workclass",
        fraction=0,
        target_category="Private",
        rng=rng,
    )

    pd.testing.assert_frame_equal(
        reference_data,
        drifted,
    )


def test_categorical_drift_does_not_mutate_input(
    reference_data,
    rng,
):
    original = reference_data.copy(deep=True)

    dg.apply_categorical_drift(
        reference_data,
        column="occupation",
        fraction=0.50,
        target_category="Exec-managerial",
        rng=rng,
    )

    pd.testing.assert_frame_equal(
        reference_data,
        original,
    )


@pytest.mark.parametrize(
    "fraction",
    [-0.1, 1.1],
)
def test_categorical_drift_rejects_invalid_fraction(
    reference_data,
    fraction,
    rng,
):
    with pytest.raises(ValueError):
        dg.apply_categorical_drift(
            reference_data,
            column="workclass",
            fraction=fraction,
            target_category="Private",
            rng=rng,
        )


def test_categorical_drift_rejects_unknown_category(
    reference_data,
    rng,
):
    with pytest.raises(ValueError):
        dg.apply_categorical_drift(
            reference_data,
            column="workclass",
            fraction=0.50,
            target_category="Nonexistent",
            rng=rng,
        )


def test_categorical_drift_rejects_unknown_column(
    reference_data,
    rng,
):
    with pytest.raises(ValueError):
        dg.apply_categorical_drift(
            reference_data,
            column="unknown",
            fraction=0.50,
            target_category="Private",
            rng=rng,
        )


# ============================================================
# 6. Scenario generation tests
# ============================================================

@pytest.mark.parametrize(
    "scenario",
    dg.SCENARIOS,
)
def test_scenario_preserves_shape(
    reference_data,
    scenario,
):
    drifted = dg.generate_drift_scenario(
        reference_data,
        scenario=scenario,
    )

    assert drifted.shape == reference_data.shape


@pytest.mark.parametrize(
    "scenario",
    dg.SCENARIOS,
)
def test_scenario_preserves_schema(
    reference_data,
    scenario,
):
    drifted = dg.generate_drift_scenario(
        reference_data,
        scenario=scenario,
    )

    assert list(drifted.columns) == list(
        reference_data.columns
    )

    assert drifted.dtypes.equals(
        reference_data.dtypes
    )


def test_control_scenario_is_identical(reference_data):
    control = dg.generate_drift_scenario(
        reference_data,
        scenario="control",
    )

    pd.testing.assert_frame_equal(
        reference_data,
        control,
    )


@pytest.mark.parametrize(
    "scenario",
    dg.SCENARIOS,
)
def test_scenario_does_not_mutate_reference(
    reference_data,
    scenario,
):
    original = reference_data.copy(deep=True)

    dg.generate_drift_scenario(
        reference_data,
        scenario=scenario,
    )

    pd.testing.assert_frame_equal(
        reference_data,
        original,
    )


@pytest.mark.parametrize(
    "scenario",
    dg.SCENARIOS,
)
def test_scenario_is_reproducible(
    reference_data,
    scenario,
):
    first = dg.generate_drift_scenario(
        reference_data,
        scenario=scenario,
        random_state=42,
    )

    second = dg.generate_drift_scenario(
        reference_data,
        scenario=scenario,
        random_state=42,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_different_seeds_produce_different_drift(reference_data):
    first = dg.generate_drift_scenario(
        reference_data,
        scenario="moderate",
        random_state=42,
    )

    second = dg.generate_drift_scenario(
        reference_data,
        scenario="moderate",
        random_state=99,
    )

    assert not first.equals(second)


def test_scenario_rejects_unknown_name(reference_data):
    with pytest.raises(ValueError):
        dg.generate_drift_scenario(
            reference_data,
            scenario="extreme",
        )


def test_scenario_rejects_empty_reference(reference_data):
    empty = reference_data.iloc[0:0]

    with pytest.raises(ValueError):
        dg.generate_drift_scenario(
            empty,
            scenario="mild",
        )


def test_scenario_rejects_missing_features(reference_data):
    invalid = reference_data.drop(columns=["age"])

    with pytest.raises(ValueError):
        dg.generate_drift_scenario(
            invalid,
            scenario="mild",
        )


def test_scenario_reorders_features_to_canonical_order(
    reference_data,
):
    reordered = reference_data[
        list(reversed(reference_data.columns))
    ]

    drifted = dg.generate_drift_scenario(
        reordered,
        scenario="mild",
    )

    assert list(drifted.columns) == FEATURE_COLUMNS


# ============================================================
# 7. Drift severity tests
# ============================================================

@pytest.mark.parametrize(
    "column",
    [
        "age",
        "education-num",
        "hours-per-week",
        "workclass",
        "occupation",
    ],
)
def test_drift_severity_increases(
    reference_data,
    column,
):
    fractions = []

    for scenario in (
        "mild",
        "moderate",
        "severe",
    ):
        drifted = dg.generate_drift_scenario(
            reference_data,
            scenario=scenario,
        )

        changes = dg.summarize_changes(
            reference_data,
            drifted,
        )

        fractions.append(
            changes[column]["changed_fraction"]
        )

    assert fractions[0] < fractions[1] < fractions[2]


@pytest.mark.parametrize(
    "scenario",
    ["mild", "moderate", "severe"],
)
def test_only_configured_features_change(
    reference_data,
    scenario,
):
    drifted = dg.generate_drift_scenario(
        reference_data,
        scenario=scenario,
    )

    changes = dg.summarize_changes(
        reference_data,
        drifted,
    )

    allowed = {
        "age",
        "education-num",
        "hours-per-week",
        "workclass",
        "occupation",
    }

    for column, statistics in changes.items():
        if column not in allowed:
            assert statistics["changed_rows"] == 0


@pytest.mark.parametrize(
    "scenario",
    dg.SCENARIOS,
)
def test_scenario_preserves_missingness(
    reference_with_missing,
    scenario,
):
    drifted = dg.generate_drift_scenario(
        reference_with_missing,
        scenario=scenario,
    )

    pd.testing.assert_frame_equal(
        reference_with_missing.isna(),
        drifted.isna(),
    )


# ============================================================
# 8. Change summary tests
# ============================================================

def test_change_summary_control(reference_data):
    changes = dg.summarize_changes(
        reference_data,
        reference_data.copy(),
    )

    assert all(
        stats["changed_rows"] == 0
        for stats in changes.values()
    )


def test_change_summary_detects_single_change(reference_data):
    modified = reference_data.copy()

    modified.loc[0, "age"] += 1

    changes = dg.summarize_changes(
        reference_data,
        modified,
    )

    assert changes["age"]["changed_rows"] == 1

    assert changes["age"]["changed_fraction"] == pytest.approx(
        1 / len(reference_data)
    )


def test_change_summary_handles_missing_values(
    reference_with_missing,
):
    changes = dg.summarize_changes(
        reference_with_missing,
        reference_with_missing.copy(),
    )

    assert all(
        stats["changed_rows"] == 0
        for stats in changes.values()
    )


def test_change_summary_rejects_column_mismatch(reference_data):
    modified = reference_data.drop(columns=["age"])

    with pytest.raises(ValueError):
        dg.summarize_changes(
            reference_data,
            modified,
        )


def test_change_summary_rejects_index_mismatch(reference_data):
    modified = reference_data.copy()

    modified.index = np.arange(
        1,
        len(modified) + 1,
    )

    with pytest.raises(ValueError):
        dg.summarize_changes(
            reference_data,
            modified,
        )


# ============================================================
# 9. Serialization tests
# ============================================================

def test_save_creates_all_scenario_files(
    reference_data,
    labels,
    tmp_path,
):
    dg.save_drift_scenarios(
        reference_data,
        labels,
        output_dir=tmp_path,
    )

    for scenario in dg.SCENARIOS:
        assert (
            tmp_path / scenario / "X_test.csv"
        ).is_file()

        assert (
            tmp_path / scenario / "y_test.csv"
        ).is_file()

    assert (
        tmp_path / "drift_metadata.json"
    ).is_file()


def test_saved_labels_are_identical(
    reference_data,
    labels,
    tmp_path,
):
    dg.save_drift_scenarios(
        reference_data,
        labels,
        output_dir=tmp_path,
    )

    for scenario in dg.SCENARIOS:
        saved = pd.read_csv(
            tmp_path / scenario / "y_test.csv"
        )["income"]

        pd.testing.assert_series_equal(
            saved,
            labels,
        )


def test_saved_control_matches_reference(
    reference_data,
    labels,
    tmp_path,
):
    dg.save_drift_scenarios(
        reference_data,
        labels,
        output_dir=tmp_path,
    )

    control = pd.read_csv(
        tmp_path / "control" / "X_test.csv"
    )

    pd.testing.assert_frame_equal(
        control,
        reference_data,
        check_dtype=False,
    )


def test_metadata_contains_all_scenarios(
    reference_data,
    labels,
    tmp_path,
):
    metadata = dg.save_drift_scenarios(
        reference_data,
        labels,
        output_dir=tmp_path,
    )

    saved_metadata = json.loads(
        (tmp_path / "drift_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert saved_metadata == metadata

    assert metadata["model_variant"] == "baseline"

    assert set(metadata["scenarios"]) == set(
        dg.SCENARIOS
    )

    assert metadata["reference_rows"] == len(
        reference_data
    )


def test_save_rejects_mismatched_target_length(
    reference_data,
    labels,
    tmp_path,
):
    with pytest.raises(ValueError):
        dg.save_drift_scenarios(
            reference_data,
            labels.iloc[:-1],
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [-1, 2, np.nan],
)
def test_save_rejects_invalid_targets(
    reference_data,
    labels,
    tmp_path,
    invalid_value,
):
    invalid = labels.astype(float).copy()

    invalid.iloc[0] = invalid_value

    with pytest.raises(ValueError):
        dg.save_drift_scenarios(
            reference_data,
            invalid,
            output_dir=tmp_path,
        )


# ============================================================
# 10. Reference verification integration tests
# ============================================================

@pytest.fixture
def baseline_artifacts(
    reference_data,
    labels,
    tmp_path,
):
    """
    Create a fitted Baseline preprocessor and matching
    transformed test artifacts.

    Uses synthetic data only.
    """

    config = PreprocessingConfig()

    preprocessor = build_preprocessor(config)

    # Fit on synthetic training data.
    preprocessor.fit(reference_data)

    processed = preprocessor.transform(reference_data)

    columns = preprocessor.get_feature_names_out()

    processed_df = pd.DataFrame(
        processed,
        columns=columns,
    )

    processed_dir = tmp_path / "processed" / "baseline"

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_df.to_csv(
        processed_dir / "X_test.csv",
        index=False,
    )

    labels.to_frame(name="income").to_csv(
        processed_dir / "y_test.csv",
        index=False,
    )

    preprocessor_path = (
        tmp_path / "baseline.joblib"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    return {
        "processed_dir": processed_dir,
        "preprocessor_path": preprocessor_path,
        "X_test": reference_data,
        "y_test": labels,
    }


def test_reference_verification_passes(baseline_artifacts):
    result = dg.verify_reference_partition(
        baseline_artifacts["X_test"],
        baseline_artifacts["y_test"],
        processed_dir=baseline_artifacts["processed_dir"],
        preprocessor_path=baseline_artifacts["preprocessor_path"],
    )

    assert result["verified"] is True
    assert result["labels_match"] is True
    assert result["features_match"] is True


def test_reference_verification_rejects_changed_labels(
    baseline_artifacts,
):
    changed_labels = baseline_artifacts["y_test"].copy()

    changed_labels.iloc[0] = 1 - changed_labels.iloc[0]

    with pytest.raises(ValueError):
        dg.verify_reference_partition(
            baseline_artifacts["X_test"],
            changed_labels,
            processed_dir=baseline_artifacts["processed_dir"],
            preprocessor_path=baseline_artifacts["preprocessor_path"],
        )


def test_reference_verification_rejects_changed_features(
    baseline_artifacts,
):
    changed = baseline_artifacts["X_test"].copy()

    changed.loc[0, "age"] += 5

    with pytest.raises(ValueError):
        dg.verify_reference_partition(
            changed,
            baseline_artifacts["y_test"],
            processed_dir=baseline_artifacts["processed_dir"],
            preprocessor_path=baseline_artifacts["preprocessor_path"],
        )


def test_reference_verification_rejects_missing_artifacts(
    baseline_artifacts,
    tmp_path,
):
    with pytest.raises(FileNotFoundError):
        dg.verify_reference_partition(
            baseline_artifacts["X_test"],
            baseline_artifacts["y_test"],
            processed_dir=tmp_path / "missing",
            preprocessor_path=baseline_artifacts["preprocessor_path"],
        )
