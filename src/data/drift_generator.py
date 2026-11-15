
"""
Sprint 17 - MLOps Final Project
Module: src/data/drift_generator.py

Generate reproducible synthetic covariate drift for the
winning Adult Income model's held-out test partition.

Scenarios:
    control
    mild
    moderate
    severe

The generator:
    - Reconstructs the original raw test partition.
    - Verifies it against saved baseline test artifacts.
    - Applies controlled numerical and categorical changes.
    - Preserves labels, rows, and feature schema.
    - Saves datasets and reproducibility metadata.

No model training or preprocessing fitting occurs here.

IMPORTANT:
    Synthetic feature perturbations with fixed labels are
    robustness stress tests, not observations of genuine
    real-world concept drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import (
    FEATURE_COLUMNS,
    PreprocessingConfig,
    clean_data,
    create_train_val_test_split,
    encode_target,
    load_and_concatenate_data,
    split_features_target,
    validate_data,
)


# ============================================================
# 1. Global configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SCENARIOS = (
    "control",
    "mild",
    "moderate",
    "severe",
)

# Valid ranges for numerical Adult Income features.
NUMERIC_BOUNDS = {
    "age": (17, 90),
    "education-num": (1, 16),
    "hours-per-week": (1, 99),
}

# Scenario-specific perturbation parameters.
#
# fraction:
#     Fraction of eligible observations selected.
#
# shift:
#     Additive change applied to selected numerical values.
#
# categorical_fraction:
#     Fraction of eligible categorical observations replaced.

SCENARIO_SETTINGS = {
    "control": {
        "fraction": 0.00,
        "categorical_fraction": 0.00,
        "shifts": {},
    },
    "mild": {
        "fraction": 0.10,
        "categorical_fraction": 0.10,
        "shifts": {
            "age": 3,
            "hours-per-week": 3,
            "education-num": 1,
        },
    },
    "moderate": {
        "fraction": 0.30,
        "categorical_fraction": 0.30,
        "shifts": {
            "age": 10,
            "hours-per-week": 10,
            "education-num": 3,
        },
    },
    "severe": {
        "fraction": 0.60,
        "categorical_fraction": 0.60,
        "shifts": {
            "age": 20,
            "hours-per-week": 25,
            "education-num": 6,
        },
    },
}


@dataclass(frozen=True)
class DriftConfig:
    """Configuration for drift generation and split verification."""

    random_state: int = 42
    test_size: float = 0.20
    validation_size: float = 0.20
    duplicate_aware_split: bool = True

    def __post_init__(self):
        # Validate using the project's existing configuration.
        PreprocessingConfig(
            random_state=self.random_state,
            test_size=self.test_size,
            validation_size=self.validation_size,
            duplicate_aware_split=self.duplicate_aware_split,
        )


# ============================================================
# 2. Fingerprinting
# ============================================================

def dataframe_sha256(df: pd.DataFrame) -> str:
    """
    Calculate a deterministic fingerprint of DataFrame
    content, column order, and data types.

    Row order is significant.
    """

    digest = hashlib.sha256()

    schema = {
        "columns": list(df.columns),
        "dtypes": [str(dtype) for dtype in df.dtypes],
        "rows": len(df),
    }

    digest.update(
        json.dumps(
            schema,
            sort_keys=True,
        ).encode("utf-8")
    )

    hashes = pd.util.hash_pandas_object(
        df.reset_index(drop=True),
        index=False,
    )

    digest.update(
        hashes.to_numpy(dtype="uint64").tobytes()
    )

    return digest.hexdigest()


# ============================================================
# 3. Load and reconstruct raw test partition
# ============================================================

def load_reference_test_data(
    raw_paths: list[str | Path],
    config: DriftConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reconstruct the raw held-out test partition using
    the original preprocessing split configuration.

    Duplicate observations are preserved.
    """

    raw = load_and_concatenate_data(raw_paths)

    cleaned = clean_data(
        raw,
        remove_duplicates=False,
    )

    validate_data(cleaned)

    encoded = encode_target(cleaned)

    X, y = split_features_target(encoded)

    _, _, X_test, _, _, y_test = (
        create_train_val_test_split(
            X,
            y,
            test_size=config.test_size,
            validation_size=config.validation_size,
            random_state=config.random_state,
            duplicate_aware=config.duplicate_aware_split,
        )
    )

    return (
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


# ============================================================
# 4. Verify against saved baseline artifacts
# ============================================================

def verify_reference_partition(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    processed_dir: str | Path,
    preprocessor_path: str | Path,
) -> dict:
    """
    Verify that reconstructed raw test data corresponds
    to the saved Baseline preprocessing artifacts.

    Checks:
        - Target values and ordering.
        - Transformed feature names and ordering.
        - Transformed numerical values.

    Raises an exception if verification fails.
    """

    processed_dir = Path(processed_dir)
    preprocessor_path = Path(preprocessor_path)

    x_path = processed_dir / "X_test.csv"
    y_path = processed_dir / "y_test.csv"

    for path in (
        x_path,
        y_path,
        preprocessor_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required baseline artifact missing: {path}"
            )

    saved_X = pd.read_csv(x_path)
    saved_y_frame = pd.read_csv(y_path)

    if saved_y_frame.shape[1] != 1:
        raise ValueError(
            "Saved y_test.csv must contain exactly one column."
        )

    saved_y = saved_y_frame.iloc[:, 0]

    if len(X_test) != len(saved_X):
        raise ValueError(
            "Reconstructed test row count does not match "
            "the saved baseline test partition."
        )

    if not np.array_equal(
        y_test.to_numpy(),
        saved_y.to_numpy(),
    ):
        raise ValueError(
            "Reconstructed test labels do not match saved "
            "baseline labels or row ordering."
        )

    # Load the fitted Baseline preprocessor.
    preprocessor = joblib.load(preprocessor_path)

    # Transform raw features without fitting.
    transformed = preprocessor.transform(X_test)

    expected_columns = list(
        preprocessor.get_feature_names_out()
    )

    if expected_columns != list(saved_X.columns):
        raise ValueError(
            "Transformed feature names or ordering differ "
            "from the saved baseline test dataset."
        )

    transformed_values = np.asarray(
        transformed,
        dtype=float,
    )

    saved_values = saved_X.to_numpy(dtype=float)

    if transformed_values.shape != saved_values.shape:
        raise ValueError(
            "Transformed feature matrix dimensions differ."
        )

    # CSV serialization may introduce very small
    # floating-point differences.
    if not np.allclose(
        transformed_values,
        saved_values,
        rtol=1e-7,
        atol=1e-9,
        equal_nan=True,
    ):
        raise ValueError(
            "Reconstructed raw test data does not reproduce "
            "the saved baseline transformed features."
        )

    return {
        "verified": True,
        "rows": len(X_test),
        "processed_features": saved_X.shape[1],
        "labels_match": True,
        "features_match": True,
    }


# ============================================================
# 5. Numerical drift
# ============================================================

def apply_numeric_drift(
    df: pd.DataFrame,
    column: str,
    fraction: float,
    shift: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Add a numerical shift to randomly selected observations.

    Values are clipped to valid Adult Income ranges.
    Missing values are preserved.
    """

    if column not in NUMERIC_BOUNDS:
        raise ValueError(
            f"Unsupported numerical feature: {column}"
        )

    if not 0 <= fraction <= 1:
        raise ValueError(
            "fraction must be between 0 and 1."
        )

    result = df.copy(deep=True)

    values = pd.to_numeric(
        result[column],
        errors="coerce",
    )

    eligible = np.flatnonzero(
        values.notna().to_numpy()
    )

    count = int(round(len(eligible) * fraction))

    if count == 0:
        return result

    selected = rng.choice(
        eligible,
        size=count,
        replace=False,
    )

    lower, upper = NUMERIC_BOUNDS[column]

    modified = np.clip(
        values.iloc[selected].to_numpy(dtype=float) + shift,
        lower,
        upper,
    )

    result.iloc[
        selected,
        result.columns.get_loc(column),
    ] = modified.astype(int)

    return result


# ============================================================
# 6. Categorical drift
# ============================================================

def apply_categorical_drift(
    df: pd.DataFrame,
    column: str,
    fraction: float,
    target_category: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Increase the prevalence of a selected category.

    Only nonmissing rows belonging to other categories
    are eligible for replacement.
    """

    if column not in df.columns:
        raise ValueError(
            f"Unknown categorical feature: {column}"
        )

    if not 0 <= fraction <= 1:
        raise ValueError(
            "fraction must be between 0 and 1."
        )

    result = df.copy(deep=True)

    categories = set(
        result[column].dropna().unique()
    )

    if target_category not in categories:
        raise ValueError(
            f"Target category '{target_category}' "
            f"does not exist in {column}."
        )

    eligible = np.flatnonzero(
        (
            result[column].notna()
            & result[column].ne(target_category)
        ).to_numpy()
    )

    # Fraction refers to eligible rows, not all rows.
    count = int(round(len(eligible) * fraction))

    if count == 0:
        return result

    selected = rng.choice(
        eligible,
        size=count,
        replace=False,
    )

    result.iloc[
        selected,
        result.columns.get_loc(column),
    ] = target_category

    return result


# ============================================================
# 7. Generate one drift scenario
# ============================================================

def generate_drift_scenario(
    reference: pd.DataFrame,
    scenario: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic drift without modifying reference data.

    All scenarios are independently reproducible.
    """

    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    if reference.empty:
        raise ValueError(
            "Reference dataset cannot be empty."
        )

    if list(reference.columns) != FEATURE_COLUMNS:
        # FEATURE_COLUMNS is the canonical preprocessing order.
        # We accept a different order but require the same schema.
        if set(reference.columns) != set(FEATURE_COLUMNS):
            raise ValueError(
                "Reference feature schema does not match "
                "the Adult Income dataset."
            )

        reference = reference.loc[:, FEATURE_COLUMNS]

    drifted = reference.copy(deep=True)

    if scenario == "control":
        return drifted

    settings = SCENARIO_SETTINGS[scenario]

    offsets = {
        "mild": 101,
        "moderate": 202,
        "severe": 303,
    }

    rng = np.random.default_rng(
        random_state + offsets[scenario]
    )

    # Apply numerical drift.
    for column, shift in settings["shifts"].items():
        drifted = apply_numeric_drift(
            drifted,
            column=column,
            fraction=settings["fraction"],
            shift=shift,
            rng=rng,
        )

    # Apply categorical drift.
    drifted = apply_categorical_drift(
        drifted,
        column="workclass",
        fraction=settings["categorical_fraction"],
        target_category="Private",
        rng=rng,
    )

    drifted = apply_categorical_drift(
        drifted,
        column="occupation",
        fraction=settings["categorical_fraction"],
        target_category="Exec-managerial",
        rng=rng,
    )

    return drifted


# ============================================================
# 8. Summarize observed feature changes
# ============================================================

def summarize_changes(
    reference: pd.DataFrame,
    drifted: pd.DataFrame,
) -> dict:
    """
    Calculate how many values actually changed.

    This does not perform statistical drift detection.
    """

    if not reference.index.equals(drifted.index):
        raise ValueError(
            "Reference and drifted indices differ."
        )

    if not reference.columns.equals(drifted.columns):
        raise ValueError(
            "Reference and drifted columns differ."
        )

    summary = {}

    for column in reference.columns:
        original = reference[column]
        modified = drifted[column]

        equal = original.eq(modified).fillna(False)

        both_missing = (
            original.isna()
            & modified.isna()
        )

        changed = ~(equal | both_missing)

        summary[column] = {
            "changed_rows": int(changed.sum()),
            "changed_fraction": float(changed.mean()),
        }

    return summary


# ============================================================
# 9. Save all drift scenarios
# ============================================================

def save_drift_scenarios(
    reference: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | Path,
    random_state: int = 42,
    verification: dict | None = None,
) -> dict:
    """
    Save the control, mild, moderate, and severe datasets.

    Target labels are preserved for paired stress testing.
    """

    if len(reference) != len(y_test):
        raise ValueError(
            "Reference and target lengths do not match."
        )

    reference = reference.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    if (
        y_test.isna().any()
        or not set(y_test.unique()).issubset({0, 1})
    ):
        raise ValueError(
            "Target labels must contain only 0 and 1."
        )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "model_variant": "baseline",
        "random_state": random_state,
        "reference_rows": len(reference),
        "reference_sha256": dataframe_sha256(reference),
        "reference_verification": verification,
        "scenario_settings": SCENARIO_SETTINGS,
        "scenarios": {},
    }

    for scenario in SCENARIOS:
        drifted = generate_drift_scenario(
            reference,
            scenario=scenario,
            random_state=random_state,
        )

        scenario_dir = output_dir / scenario

        scenario_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        drifted.to_csv(
            scenario_dir / "X_test.csv",
            index=False,
        )

        y_test.to_frame(name="income").to_csv(
            scenario_dir / "y_test.csv",
            index=False,
        )

        metadata["scenarios"][scenario] = {
            "rows": len(drifted),
            "sha256": dataframe_sha256(drifted),
            "feature_changes": summarize_changes(
                reference,
                drifted,
            ),
        }

    metadata_path = (
        output_dir / "drift_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metadata


# ============================================================
# 10. Command-line interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic drift for the winning "
            "Adult Income Baseline model."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data/raw",
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data/processed/baseline",
    )

    parser.add_argument(
        "--preprocessor-path",
        type=Path,
        default=ROOT / "models/preprocessors/baseline.joblib",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/drift",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
    )

    return parser.parse_args()


# ============================================================
# 11. Main execution
# ============================================================

def main():
    args = parse_args()

    config = DriftConfig(
        random_state=args.random_state,
        test_size=args.test_size,
        validation_size=args.validation_size,
    )

    raw_paths = [
        args.raw_dir / "adult.data",
        args.raw_dir / "adult.test",
    ]

    # Reconstruct the original held-out test partition.
    X_test, y_test = load_reference_test_data(
        raw_paths,
        config=config,
    )

    # Verify against the saved winning preprocessing variant.
    verification = verify_reference_partition(
        X_test,
        y_test,
        processed_dir=args.processed_dir,
        preprocessor_path=args.preprocessor_path,
    )

    print(
        "Reference partition verified:",
        verification,
    )

    # Generate and save the drift scenarios.
    metadata = save_drift_scenarios(
        X_test,
        y_test,
        output_dir=args.output_dir,
        random_state=config.random_state,
        verification=verification,
    )

    print("\nSynthetic drift generation completed.\n")

    for scenario, details in metadata["scenarios"].items():
        changes = {
            column: round(
                values["changed_fraction"], 4
            )
            for column, values in details[
                "feature_changes"
            ].items()
            if values["changed_rows"] > 0
        }

        print(f"{scenario.upper()}:")
        print(f"  Rows: {details['rows']}")
        print(f"  Changed features: {changes}")


if __name__ == "__main__":
    main()
