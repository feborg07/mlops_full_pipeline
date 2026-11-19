"""Validate the real DVC-managed UCI Adult Income dataset.

Unlike the synthetic preprocessing fixtures, these tests read the raw files
retrieved by ``dvc pull data/raw.dvc``. Run them from the repository root:

    python -m pytest tests/test_data_validation.py -v

CI must download the raw data *before* running pytest.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.preprocess import (
    CATEGORICAL_FEATURES,
    COLUMN_NAMES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    clean_data,
    load_and_concatenate_data,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_FILES = (RAW_DIR / "adult.data", RAW_DIR / "adult.test")


@pytest.fixture(scope="module")
def adult_data() -> pd.DataFrame:
    """Load and normalize the actual dataset once for this test module."""
    missing = [str(path) for path in RAW_FILES if not path.is_file()]
    if missing:
        pytest.fail(
            "DVC training inputs are missing: "
            + ", ".join(missing)
            + ". Run: dvc pull data/raw.dvc"
        )

    raw = load_and_concatenate_data(RAW_FILES)
    return clean_data(raw)


def test_expected_columns_are_present(adult_data: pd.DataFrame) -> None:
    """Validate the 14-feature Adult schema and its target column."""
    assert all(column in adult_data.columns for column in COLUMN_NAMES)
    assert len(COLUMN_NAMES) == 15
    assert TARGET_COLUMN in adult_data.columns
    assert "source_file" in adult_data.columns  # Loader provenance metadata.
    assert set(adult_data["source_file"]) == {"adult.data", "adult.test"}


def test_dataset_size_and_feature_types(adult_data: pd.DataFrame) -> None:
    """The real dataset must meet the sprint's tabular-data requirements."""
    assert len(adult_data) == 48_842
    assert len(adult_data) >= 1_000
    assert len(COLUMN_NAMES) - 1 >= 8
    assert NUMERICAL_FEATURES and CATEGORICAL_FEATURES
    for column in NUMERICAL_FEATURES:
        assert pd.api.types.is_numeric_dtype(adult_data[column]), column
    for column in CATEGORICAL_FEATURES:
        assert not pd.api.types.is_numeric_dtype(adult_data[column]), column


def test_target_contains_only_expected_labels(adult_data: pd.DataFrame) -> None:
    """Both original files must normalize to the same two income classes."""
    labels = adult_data[TARGET_COLUMN]
    assert labels.notna().all()
    assert set(labels.unique()) == {"<=50K", ">50K"}
    assert not labels.str.endswith(".").any()


def test_numerical_features_have_expected_ranges(adult_data: pd.DataFrame) -> None:
    """Check the documented bounds of the unmodified UCI Adult data."""
    ranges = {
        "age": (17, 90),
        "fnlwgt": (1, None),
        "education-num": (1, 16),
        "capital-gain": (0, 99_999),
        "capital-loss": (0, 99_999),
        "hours-per-week": (1, 99),
    }
    for column, (minimum, maximum) in ranges.items():
        values = adult_data[column]
        assert values.notna().all(), f"Unexpected missing numeric values: {column}"
        assert (values >= minimum).all(), f"Values below expected minimum: {column}"
        if maximum is not None:
            assert (values <= maximum).all(), f"Values above expected maximum: {column}"


def test_missing_categorical_values_are_preserved(adult_data: pd.DataFrame) -> None:
    """Missing values must remain visible for later categorical imputation."""
    for column in ("workclass", "occupation", "native-country"):
        assert adult_data[column].isna().any(), column
    assert not adult_data[TARGET_COLUMN].isna().any()
