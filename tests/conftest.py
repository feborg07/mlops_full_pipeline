"""
Sprint 17 - MLOps Final Project
File: tests/conftest.py

Purpose
-------
Provide reusable pytest fixtures for testing the configurable
Adult Income preprocessing pipeline.

Test coverage supported by these fixtures:
1. Raw dataset loading and concatenation.
2. Data cleaning and missing-value normalization.
3. Target encoding and feature separation.
4. Stratified and duplicate-aware splitting.
5. Numerical and categorical preprocessing.
6. Rare-category grouping.
7. Capital gain/loss feature engineering.
8. Alternative education representations.
9. Optional feature interactions.
10. Fitted preprocessing and serialization.
11. End-to-end preprocessing integration.

All observations are synthetic and are not sampled from
the actual Adult Income dataset.
"""

# ============================================================
# 1. Imports
# ============================================================

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    CATEGORICAL_FEATURES,
    COLUMN_NAMES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    PreprocessingConfig,
    build_preprocessor,
    create_train_test_split,
    encode_target,
    split_features_target,
)

# ============================================================
# 2. Base preprocessing configuration
# ============================================================


@pytest.fixture
def default_config():
    """
    Return the default EDA-driven preprocessing configuration.

    The configuration is immutable because PreprocessingConfig
    uses frozen=True.
    """

    return PreprocessingConfig()


@pytest.fixture
def alternative_configs(default_config):
    """
    Return several configurations for parameterized tests.

    dataclasses.replace() creates a new configuration without
    modifying the original frozen dataclass.
    """

    return {
        "default": default_config,
        "most_frequent": replace(
            default_config,
            categorical_imputation="most_frequent",
        ),
        "no_rare_grouping": replace(
            default_config,
            rare_category_frequency=None,
        ),
        "robust_scaling": replace(
            default_config,
            numerical_scaling="robust",
        ),
        "no_scaling": replace(
            default_config,
            numerical_scaling="none",
        ),
        "original_capital": replace(
            default_config,
            capital_transform="original",
        ),
        "both_capital": replace(
            default_config,
            capital_transform="both",
        ),
        "no_capital_indicators": replace(
            default_config,
            capital_indicators=False,
        ),
        "categorical_education": replace(
            default_config,
            education_representation="education",
        ),
        "both_education": replace(
            default_config,
            education_representation="both",
        ),
        "interactions": replace(
            default_config,
            interaction_features=True,
        ),
        "standard_split": replace(
            default_config,
            duplicate_aware_split=False,
        ),
    }


# ============================================================
# 3. Synthetic Adult Income dataset
# ============================================================


@pytest.fixture
def sample_adult_df():
    """
    Return a small, valid Adult Income DataFrame.

    Contains:
    - All 15 required Adult Income columns.
    - Both target classes.
    - Different numerical values.
    - Several categorical levels.
    - Capital gain/loss values.
    - Source-file metadata.

    The dataset is deterministic and suitable for unit tests.
    """

    # One person per line, in COLUMN_NAMES order.
    # fmt: off
    records = [
        [25, "Private", 120000, "Bachelors", 13, "Never-married", "Sales", "Not-in-family", "White", "Male", 0, 0, 40, "United-States", "<=50K"],
        [45, "Self-emp-inc", 180000, "Masters", 14, "Married-civ-spouse", "Exec-managerial", "Husband", "White", "Male", 15000, 0, 55, "United-States", ">50K"],
        [32, "Private", 140000, "HS-grad", 9, "Divorced", "Craft-repair", "Unmarried", "Black", "Female", 0, 0, 35, "Mexico", "<=50K"],
        [51, "Federal-gov", 210000, "Doctorate", 16, "Married-civ-spouse", "Prof-specialty", "Wife", "White", "Female", 99999, 0, 45, "United-States", ">50K"],
        [28, "Private", 130000, "Some-college", 10, "Never-married", "Adm-clerical", "Own-child", "White", "Female", 0, 0, 30, "Philippines", "<=50K"],
        [39, "Local-gov", 175000, "Bachelors", 13, "Married-civ-spouse", "Tech-support", "Husband", "Asian-Pac-Islander", "Male", 5000, 0, 42, "India", ">50K"],
        [22, "Private", 115000, "HS-grad", 9, "Never-married", "Handlers-cleaners", "Own-child", "White", "Male", 0, 0, 25, "United-States", "<=50K"],
        [57, "Self-emp-not-inc", 190000, "Masters", 14, "Married-civ-spouse", "Sales", "Husband", "White", "Male", 0, 1902, 60, "Canada", ">50K"],
        [34, "Private", 155000, "Assoc-voc", 11, "Divorced", "Machine-op-inspct", "Not-in-family", "Black", "Female", 0, 0, 40, "United-States", "<=50K"],
        [48, "State-gov", 205000, "Bachelors", 13, "Married-civ-spouse", "Prof-specialty", "Wife", "White", "Female", 12000, 0, 45, "Germany", ">50K"],
        [19, "Private", 105000, "Some-college", 10, "Never-married", "Other-service", "Own-child", "White", "Male", 0, 0, 20, "Mexico", "<=50K"],
        [62, "Self-emp-inc", 230000, "Doctorate", 16, "Married-civ-spouse", "Exec-managerial", "Husband", "White", "Male", 20000, 0, 50, "United-States", ">50K"],
    ]
    # fmt: on

    df = pd.DataFrame(
        records,
        columns=COLUMN_NAMES,
    )

    df["source_file"] = "synthetic.data"

    return df


# ============================================================
# 4. Dirty dataset
# ============================================================


@pytest.fixture
def dirty_adult_df(sample_adult_df):
    """
    Return a dataset containing common data-quality issues.

    Includes:
    - Surrounding whitespace.
    - Missing-value marker "?".
    - Actual NaN values.
    - Numeric values stored as strings.
    - A target label with a trailing period.
    - One duplicate observation.

    Used to test clean_data().
    """

    df = sample_adult_df.copy(deep=True)

    # Allow mixed numeric and string values.
    df["age"] = df["age"].astype("object")

    # Whitespace around categorical values.
    df.loc[0, "workclass"] = " Private "

    # Textual missing-value marker.
    df.loc[1, "occupation"] = "?"

    # Actual missing value.
    df.loc[2, "native-country"] = np.nan

    # Numeric value stored as a string.
    df.loc[3, "age"] = "51"

    # Target formatting from adult.test.
    df.loc[4, "income"] = "<=50K."

    # Add an exact duplicate observation.
    duplicate = df.iloc[[0]].copy()

    df = pd.concat(
        [df, duplicate],
        ignore_index=True,
    )

    return df


# ============================================================
# 5. Dataset with missing categorical values
# ============================================================


@pytest.fixture
def missing_values_df(sample_adult_df):
    """
    Return a dataset containing missing values in the
    three categorical features identified during EDA.

    These are:
    - workclass
    - occupation
    - native-country
    """

    df = sample_adult_df.copy(deep=True)

    df.loc[0, "workclass"] = np.nan
    df.loc[1, "occupation"] = np.nan
    df.loc[2, "native-country"] = np.nan

    # Include overlapping missingness.
    df.loc[3, "workclass"] = np.nan
    df.loc[3, "occupation"] = np.nan

    return df


# ============================================================
# 6. Dataset with rare countries
# ============================================================


@pytest.fixture
def rare_country_df(sample_adult_df):
    """
    Return a larger synthetic dataset for testing
    infrequent-category grouping.

    The country distribution is intentionally unbalanced:

    - United-States: 90 observations
    - Mexico: 7 observations
    - Germany: 2 observations
    - Portugal: 1 observation

    With min_frequency=0.05:
    - United-States receives its own category.
    - Mexico receives its own category.
    - Germany and Portugal are grouped as infrequent.

    The generated observations have unique age/fnlwgt
    combinations so they are not identical records.
    """

    records = []

    countries = ["United-States"] * 90 + ["Mexico"] * 7 + ["Germany"] * 2 + ["Portugal"]

    for index, country in enumerate(countries):
        # Cycle through the existing synthetic records.
        record = sample_adult_df.iloc[index % len(sample_adult_df)].copy()

        record["native-country"] = country

        # Make each observation distinguishable.
        record["age"] = 20 + (index % 60)
        record["fnlwgt"] = 100000 + index * 100

        records.append(record)

    return pd.DataFrame(records).reset_index(drop=True)


# ============================================================
# 7. Dataset containing duplicate feature vectors
# ============================================================


@pytest.fixture
def duplicate_groups_df(sample_adult_df):
    """
    Return a dataset containing repeated observations.

    The fixture contains:
    - The original 12 observations.
    - Three additional copies of observation 0.
    - Two additional copies of observation 1.

    Expected total:
        12 + 3 + 2 = 17 observations.

    Duplicate-aware splitting should preserve all 17
    observations while keeping identical feature vectors
    within the same partition.
    """

    df = sample_adult_df.copy(deep=True)

    duplicates = pd.concat(
        [df.iloc[[0]].copy() for _ in range(3)]
        + [df.iloc[[1]].copy() for _ in range(2)],
        ignore_index=True,
    )

    return pd.concat(
        [df, duplicates],
        ignore_index=True,
    )


# ============================================================
# 8. Dataset with conflicting labels
# ============================================================


@pytest.fixture
def conflicting_labels_df(sample_adult_df):
    """
    Return two observations with identical predictor
    values but different income labels.

    This fixture verifies that duplicate-aware splitting
    groups by predictor values, not by target labels.

    It also helps distinguish duplicate observations
    from duplicate feature vectors.
    """

    df = sample_adult_df.copy(deep=True)

    conflicting_record = df.iloc[[0]].copy()

    conflicting_record["income"] = ">50K"

    return pd.concat(
        [df, conflicting_record],
        ignore_index=True,
    )


# ============================================================
# 9. Dataset with unseen categorical values
# ============================================================


@pytest.fixture
def unseen_category_df(sample_adult_df):
    """
    Return a dataset containing a category that does
    not appear in the standard training fixture.

    This supports testing:
        OneHotEncoder(handle_unknown=...)

    The unseen country is synthetic.
    """

    df = sample_adult_df.iloc[[0]].copy()

    df["native-country"] = "Neverland"

    df["occupation"] = "New-Occupation"

    return df.reset_index(drop=True)


# ============================================================
# 10. Dataset with numerical missing values
# ============================================================


@pytest.fixture
def numerical_missing_df(sample_adult_df):
    """
    Return a dataset containing numerical missing values.

    Supports testing median/mean imputation and
    numerical transformations.
    """

    df = sample_adult_df.copy(deep=True)

    df.loc[0, "age"] = np.nan
    df.loc[1, "hours-per-week"] = np.nan
    df.loc[2, "fnlwgt"] = np.nan
    df.loc[3, "capital-gain"] = np.nan

    return df


# ============================================================
# 11. Dataset for capital feature engineering
# ============================================================


@pytest.fixture
def capital_features_df(sample_adult_df):
    """
    Return observations with representative capital values.

    Includes:
    - Zero capital gain/loss.
    - Positive capital gain.
    - Positive capital loss.
    - Maximum observed capital gain (99,999).

    Useful for testing:
    - log1p transformation.
    - Positive-value indicators.
    - Original vs transformed representations.
    """

    df = sample_adult_df.copy(deep=True)

    df.loc[0, "capital-gain"] = 0
    df.loc[0, "capital-loss"] = 0

    df.loc[1, "capital-gain"] = 5000
    df.loc[1, "capital-loss"] = 0

    df.loc[2, "capital-gain"] = 0
    df.loc[2, "capital-loss"] = 1902

    df.loc[3, "capital-gain"] = 99999
    df.loc[3, "capital-loss"] = 0

    return df


# ============================================================
# 12. Dataset for education representation
# ============================================================


@pytest.fixture
def education_features_df(sample_adult_df):
    """
    Return data for testing education representations.

    The pipeline can use:
    - education-num only
    - education only
    - both

    This fixture preserves the original education columns.
    """

    return sample_adult_df.copy(deep=True)


# ============================================================
# 13. Dataset for interaction features
# ============================================================


@pytest.fixture
def interaction_features_df(sample_adult_df):
    """
    Return data for testing optional interactions.

    Expected interactions include:
    - age * education-num
    - occupation-specific working hours
    """

    return sample_adult_df.copy(deep=True)


# ============================================================
# 14. Encoded Adult Income dataset
# ============================================================


@pytest.fixture
def encoded_adult_df(sample_adult_df):
    """
    Return a dataset with numerical target labels.

    <=50K -> 0
    >50K  -> 1
    """

    return encode_target(sample_adult_df)


# ============================================================
# 15. Feature matrix and target vector
# ============================================================


@pytest.fixture
def features_target(encoded_adult_df):
    """
    Return X and y from the encoded dataset.

    X contains the 14 original predictors.
    y contains binary target labels.
    """

    return split_features_target(encoded_adult_df)


# ============================================================
# 16. Train/test split
# ============================================================


@pytest.fixture
def split_data(features_target):
    """
    Return a deterministic stratified train/test split.

    Uses ordinary stratified splitting for this fixture
    to avoid group-level limitations on very small
    synthetic datasets.

    Duplicate-aware splitting is tested separately.
    """

    X, y = features_target

    return create_train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        duplicate_aware=False,
    )


# ============================================================
# 17. Unfitted preprocessor
# ============================================================


@pytest.fixture
def preprocessor(default_config):
    """
    Return an unfitted preprocessing pipeline using
    the default EDA-driven configuration.
    """

    return build_preprocessor(default_config)


# ============================================================
# 18. Fitted preprocessor
# ============================================================


@pytest.fixture
def fitted_preprocessor(
    preprocessor,
    split_data,
):
    """
    Fit the default preprocessor using training data only.

    Returns the fitted pipeline and the split datasets.
    """

    X_train, X_test, y_train, y_test = split_data

    preprocessor.fit(
        X_train,
        y_train,
    )

    return {
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ============================================================
# 19. Temporary raw Adult Income files
# ============================================================


@pytest.fixture
def sample_raw_files(
    tmp_path,
    sample_adult_df,
):
    """
    Create temporary adult.data and adult.test files.

    These files simulate the structure of the original
    UCI Adult Income datasets.

    adult.test includes:
    - A metadata line beginning with "|".
    - Income labels ending with a period.

    pytest automatically cleans up tmp_path.
    """

    raw_dir = tmp_path / "raw"

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = raw_dir / "adult.data"
    test_path = raw_dir / "adult.test"

    # Split synthetic observations into two files.
    train_df = sample_adult_df.iloc[:8].copy()

    test_df = sample_adult_df.iloc[8:].copy()

    # Remove metadata before writing.
    train_df = train_df[COLUMN_NAMES]

    test_df = test_df[COLUMN_NAMES]

    # Match the target-label formatting in adult.test.
    test_df["income"] = test_df["income"].astype(str) + "."

    # Save without headers or row indices.
    train_df.to_csv(
        train_path,
        index=False,
        header=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
        header=False,
    )

    # Prepend the UCI test-file metadata line.
    test_content = test_path.read_text(encoding="utf-8")

    test_path.write_text(
        "|1x3 Cross validator\n" + test_content,
        encoding="utf-8",
    )

    return [
        train_path,
        test_path,
    ]


# ============================================================
# 20. Temporary output directory
# ============================================================


@pytest.fixture
def output_dir(tmp_path):
    """
    Return a temporary directory for processed CSV files.
    """

    directory = tmp_path / "processed"

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


# ============================================================
# 21. Temporary preprocessor artifact path
# ============================================================


@pytest.fixture
def preprocessor_path(tmp_path):
    """
    Return a temporary path for serialized preprocessing.
    """

    return tmp_path / "models" / "preprocessor.joblib"


# ============================================================
# 22. Temporary configuration artifact path
# ============================================================


@pytest.fixture
def config_path(tmp_path):
    """
    Return a temporary JSON configuration path.
    """

    return tmp_path / "config" / "preprocessing_config.json"


# ============================================================
# 23. Numerical feature reference
# ============================================================


@pytest.fixture
def numerical_features():
    """
    Return a copy of the original numerical feature names.
    """

    return NUMERICAL_FEATURES.copy()


# ============================================================
# 24. Categorical feature reference
# ============================================================


@pytest.fixture
def categorical_features():
    """
    Return a copy of the original categorical feature names.
    """

    return CATEGORICAL_FEATURES.copy()


@pytest.fixture(
    params=list(
        product(
            ("original", "log1p", "both"),
            ("education-num", "education", "both"),
            (False, True),
            (False, True),
        )
    ),
    ids=lambda values: "-".join(map(str, values)),
)
def feature_name_config(request, default_config):
    """Exercise all capital/education/indicator/interaction combinations."""
    capital, education, indicators, interactions = request.param
    return replace(
        default_config,
        capital_transform=capital,
        education_representation=education,
        capital_indicators=indicators,
        interaction_features=interactions,
    )


@pytest.fixture
def missing_occupation_features(sample_adult_df):
    """Nullable strings expose missing-category comparison errors."""
    X = sample_adult_df[FEATURE_COLUMNS].copy()
    X["occupation"] = X["occupation"].astype("string")
    X.loc[X.index[:2], "occupation"] = pd.NA
    return X