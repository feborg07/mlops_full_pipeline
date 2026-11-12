"""
Sprint 17 - MLOps Final Project
Module: src/data/preprocess.py

Purpose
-------
Provide a configurable, reproducible preprocessing pipeline for
the UCI Adult Income dataset.

Architecture
------------
1. Load and concatenate raw datasets.
2. Clean and validate the data.
3. Encode the binary target.
4. Split observations into training and testing partitions.
5. Build a configurable scikit-learn preprocessing transformer.
6. Fit transformations using training data only.
7. Save processed datasets and preprocessing artifacts.

EDA-driven options
------------------
- Categorical missing-value imputation
- Rare-category grouping
- Numerical scaling
- Capital gain/loss transformations
- Capital presence indicators
- Education feature representation
- Selected feature interactions
- Duplicate-aware train/test splitting

Important
---------
The preprocessor is designed to be embedded inside a scikit-learn
Pipeline during model cross-validation.

Do not fit a preprocessor on the complete training dataset before
performing cross-validation. Doing so would allow validation-fold
information to influence fitted transformations.

Duplicate records are preserved by default.
"""

# ============================================================
# 1. Imports
# ============================================================

# Standard library
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path

# Model serialization
import joblib

# Data manipulation
import numpy as np
import pandas as pd

# Scikit-learn transformers and pipeline tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

# ============================================================
# 2. Global constants
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

TARGET_COLUMN = "income"

NUMERICAL_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

TARGET_MAPPING = {
    "<=50K": 0,
    ">50K": 1,
}

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

CAPITAL_COLUMNS = [
    "capital-gain",
    "capital-loss",
]


# ============================================================
# 3. Preprocessing configuration
# ============================================================


def _check_choice(name: str, value, choices: tuple[str, ...]) -> None:
    """
    Raise ValueError when a setting is not one of its allowed values.

    The error message lists the options in order, for example:

        numerical_scaling must be 'standard', 'robust', or 'none'.

    Parameters
    ----------
    name:
        Name of the configuration field being checked.

    value:
        Value supplied for that field.

    choices:
        Allowed values, in the order shown in the error message.
    """

    if value in choices:
        return

    quoted = [f"'{choice}'" for choice in choices]

    # Two options read "'a' or 'b'"; three or more read "'a', 'b', or 'c'".
    if len(quoted) == 2:
        options = " or ".join(quoted)
    else:
        options = ", ".join(quoted[:-1]) + ", or " + quoted[-1]

    raise ValueError(f"{name} must be {options}.")


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Configuration controlling the preprocessing workflow.

    Each configuration can be used as a separate MLflow
    experiment during model development.

    Parameters
    ----------
    categorical_imputation:
        "constant" or "most_frequent".

    missing_category:
        Label used when categorical_imputation="constant".

    rare_category_frequency:
        Minimum frequency required for a category to remain
        separate in OneHotEncoder.

        None disables infrequent-category grouping.

        A float between 0 and 1 represents a proportion of
        the fitting dataset.

    numerical_imputation:
        Numerical missing-value strategy.

    numerical_scaling:
        "standard", "robust", or "none".

    capital_transform:
        "original", "log1p", or "both".

    capital_indicators:
        Add binary features indicating positive capital
        gain and capital loss.

    education_representation:
        "education-num", "education", or "both".

    interaction_features:
        Add selected EDA-motivated interaction features.

    duplicate_aware_split:
        Keep identical predictor vectors in the same partition
        without deleting observations.

    test_size:
        Proportion reserved for final testing.

    random_state:
        Random seed used for reproducibility.
    """

    categorical_imputation: str = "constant"
    missing_category: str = "Unknown"

    rare_category_frequency: float | None = 0.01

    numerical_imputation: str = "median"
    numerical_scaling: str = "standard"

    capital_transform: str = "log1p"
    capital_indicators: bool = True

    education_representation: str = "education-num"

    interaction_features: bool = False

    duplicate_aware_split: bool = True

    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE

    def __post_init__(self):
        """
        Validate configuration immediately after creation.
        """

        _check_choice(
            "categorical_imputation",
            self.categorical_imputation,
            ("constant", "most_frequent"),
        )

        if (
            not isinstance(self.missing_category, str)
            or not self.missing_category.strip()
        ):
            raise ValueError("missing_category must be a nonempty string.")

        frequency = self.rare_category_frequency

        if frequency is not None and (
            not isinstance(frequency, Real)
            or isinstance(frequency, bool)
            or not 0 < frequency < 1
        ):
            raise ValueError(
                "rare_category_frequency must be None or a number between 0 and 1."
            )

        _check_choice(
            "numerical_imputation",
            self.numerical_imputation,
            ("median", "mean"),
        )
        _check_choice(
            "numerical_scaling",
            self.numerical_scaling,
            ("standard", "robust", "none"),
        )
        _check_choice(
            "capital_transform",
            self.capital_transform,
            ("original", "log1p", "both"),
        )
        _check_choice(
            "education_representation",
            self.education_representation,
            ("education-num", "education", "both"),
        )

        if (
            isinstance(self.test_size, bool)
            or not isinstance(self.test_size, Real)
            or not 0 < self.test_size < 1
        ):
            raise ValueError("test_size must be between 0 and 1.")

        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, Integral)
            or not 0 <= self.random_state <= 2**32 - 1
        ):
            raise ValueError(
                "random_state must be an integer between 0 and "
                "2**32 - 1, excluding bool."
            )

        # Normalize NumPy scalar values for JSON serialization.
        object.__setattr__(self, "random_state", int(self.random_state))
        object.__setattr__(self, "test_size", float(self.test_size))
        if frequency is not None:
            object.__setattr__(self, "rare_category_frequency", float(frequency))


# ============================================================
# 4. Load and concatenate raw datasets
# ============================================================


def load_and_concatenate_data(
    paths: Sequence[str | Path],
) -> pd.DataFrame:
    """
    Load multiple Adult Income files and concatenate them.

    Handles:
    - adult.data
    - adult.test
    - The metadata line beginning with "|" in adult.test
    - Missing values represented by "?"
    - Source-file tracking

    Returns
    -------
    pd.DataFrame
        Combined dataset containing all observations.
    """

    if not paths:
        raise ValueError("The paths list cannot be empty.")

    dataframes = []

    for path in paths:
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(
            path,
            header=None,
            names=COLUMN_NAMES,
            sep=",",
            skipinitialspace=True,
            na_values=["?", ""],
            comment="|",
        )

        # Remove completely empty records.
        df = df.dropna(how="all")

        # Preserve the origin of each observation.
        df["source_file"] = path.name

        dataframes.append(df)

    combined_df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    return combined_df


# ============================================================
# 5. Clean dataset
# ============================================================


def clean_data(
    df: pd.DataFrame,
    remove_duplicates: bool = False,
) -> pd.DataFrame:
    """
    Apply deterministic cleaning operations.

    Operations
    ----------
    1. Copy the input DataFrame.
    2. Remove surrounding whitespace.
    3. Normalize missing-value markers.
    4. Remove trailing periods from target labels.
    5. Convert numerical columns.
    6. Remove records without a valid target value.
    7. Optionally remove duplicate observations.

    Duplicate records are preserved by default.
    """

    cleaned = df.copy(deep=True)

    # Normalize string columns.
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Normalize textual missing-value markers.
    cleaned = cleaned.replace(
        ["?", "", "nan"],
        np.nan,
    )

    # Normalize income labels from adult.test.
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype("string").str.rstrip(".")

    # Convert all configured numerical features.
    for column in NUMERICAL_FEATURES:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    # A supervised observation requires a target label.
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])

    # Optional duplicate removal.
    # Source-file metadata must not define uniqueness.
    if remove_duplicates:
        cleaned = cleaned.drop_duplicates(
            subset=COLUMN_NAMES,
        )

    return cleaned.reset_index(drop=True)


# ============================================================
# 6. Validate dataset
# ============================================================


def validate_data(
    df: pd.DataFrame,
) -> None:
    """
    Validate the dataset before modeling.

    Checks:
    - Required columns exist.
    - Dataset is not empty.
    - Target contains only supported labels.
    - Both target classes are represented.
    """

    missing_columns = set(COLUMN_NAMES) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("The dataset is empty.")

    target_values = set(df[TARGET_COLUMN].dropna().unique())

    invalid_labels = target_values - set(TARGET_MAPPING)

    if invalid_labels:
        raise ValueError(f"Unexpected target labels: {sorted(invalid_labels)}")

    if target_values != set(TARGET_MAPPING):
        raise ValueError("The dataset must contain both income classes.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("The target column contains missing values.")


# ============================================================
# 7. Encode binary target
# ============================================================


def encode_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode income as a binary numerical target.

    <=50K -> 0
    >50K  -> 1
    """

    encoded = df.copy(deep=True)

    if TARGET_COLUMN not in encoded.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    target = encoded[TARGET_COLUMN]
    if target.isna().any():
        raise ValueError("Target contains missing values.")

    invalid = target.loc[~target.isin(TARGET_MAPPING)]
    if not invalid.empty:
        raise ValueError(f"Unexpected target labels: {invalid.unique().tolist()}")

    encoded[TARGET_COLUMN] = encoded[TARGET_COLUMN].map(TARGET_MAPPING).astype("int64")

    return encoded


# ============================================================
# 8. Separate features and target
# ============================================================


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate predictor features and target labels.

    The source_file metadata column is excluded from
    the model input.
    """

    X = df[FEATURE_COLUMNS].copy()

    y = df[TARGET_COLUMN].copy()

    return X, y


# ============================================================
# 9. Duplicate-aware train/test splitting
# ============================================================


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    duplicate_aware: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Split data into training and testing partitions.

    When duplicate_aware=True, identical complete feature
    records are assigned to the same partition.

    Duplicate records are not deleted.

    The split operates on unique feature groups and then
    expands those groups back into individual observations.

    Limitations
    -----------
    Group-level stratification approximates, but does not
    guarantee, observation-level class proportions.

    Groups are defined using complete predictor values.
    Identical feature vectors with different target labels
    are still assigned to the same partition.
    """

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if not X.index.equals(y.index):
        raise ValueError("X and y must have matching indices.")

    # Reuse the same validation for direct calls and the full workflow.
    split_config = PreprocessingConfig(
        test_size=test_size,
        random_state=random_state,
    )
    test_size = split_config.test_size
    random_state = split_config.random_state

    if not duplicate_aware:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    # Reset positional indices for reliable group expansion.
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)

    # Factorize complete feature rows into group IDs.
    #
    # Missing values are treated as equivalent within
    # the grouping operation.
    group_index = pd.MultiIndex.from_frame(X_reset)

    group_codes, _ = pd.factorize(
        group_index,
        sort=False,
    )

    # Construct a table linking each observation to
    # its feature-group ID and target label.
    grouping_df = pd.DataFrame(
        {
            "group": group_codes,
            "target": y_reset.to_numpy(),
        }
    )

    # Calculate group sizes and positive-class rates.
    group_summary = (
        grouping_df.groupby("group", sort=False)
        .agg(
            group_size=("target", "size"),
            positive_rate=("target", "mean"),
        )
        .reset_index()
    )

    # Assign each group to a class for approximate
    # group-level stratification.
    group_summary["stratum"] = (group_summary["positive_rate"] >= 0.5).astype(int)

    # Stratification requires at least two groups
    # in each stratum.
    stratum_counts = group_summary["stratum"].value_counts()

    can_stratify = (
        len(stratum_counts) == 2
        and stratum_counts.min() >= 2
        and int(np.ceil(len(group_summary) * test_size)) >= 2
        and len(group_summary) - int(np.ceil(len(group_summary) * test_size)) >= 2
    )

    group_train, group_test = train_test_split(
        group_summary["group"],
        test_size=test_size,
        random_state=random_state,
        stratify=(group_summary["stratum"] if can_stratify else None),
    )

    # Expand selected groups to original observations.
    train_mask = grouping_df["group"].isin(group_train).to_numpy()

    test_mask = grouping_df["group"].isin(group_test).to_numpy()

    X_train = X.iloc[np.flatnonzero(train_mask)].copy()

    X_test = X.iloc[np.flatnonzero(test_mask)].copy()

    y_train = y.iloc[np.flatnonzero(train_mask)].copy()

    y_test = y.iloc[np.flatnonzero(test_mask)].copy()

    # Verify that all observations were assigned.
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise RuntimeError("Train/test split lost observations or assigned them twice.")

    return X_train, X_test, y_train, y_test


# ============================================================
# 10. EDA-driven feature engineering
# ============================================================


class AdultFeatureEngineer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Create optional EDA-driven features.

    This transformer is compatible with scikit-learn
    Pipeline and cross-validation.

    Feature transformations are deterministic and do not
    learn population statistics.

    Parameters
    ----------
    capital_transform:
        "original", "log1p", or "both".

    capital_indicators:
        Whether to add positive-capital indicators.

    education_representation:
        Which education representation to retain.

    interaction_features:
        Whether to create selected interaction features.
    """

    def __init__(
        self,
        capital_transform="log1p",
        capital_indicators=True,
        education_representation="education-num",
        interaction_features=False,
    ):

        self.capital_transform = capital_transform

        self.capital_indicators = capital_indicators

        self.education_representation = education_representation

        self.interaction_features = interaction_features

    def fit(self, X, y=None):
        """
        Validate input columns and record input schema.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("AdultFeatureEngineer requires a DataFrame.")

        missing = set(FEATURE_COLUMNS) - set(X.columns)

        if missing:
            raise ValueError(f"Missing input features: {sorted(missing)}")

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )

        self.n_features_in_ = X.shape[1]

        # Derive names from the actual transformation, preserving column order.
        self.output_features_ = np.asarray(
            self._transform_frame(X).columns, dtype=object
        )

        return self

    def get_feature_names_out(self, input_features=None):
        """Return output names and validate names propagated by Pipeline."""
        check_is_fitted(self, "output_features_")
        if input_features is not None and not np.array_equal(
            np.asarray(input_features, dtype=object), self.feature_names_in_
        ):
            raise ValueError("input_features must match feature_names_in_.")
        return self.output_features_.copy()

    def transform(self, X):
        """
        Apply deterministic feature transformations.
        """

        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("AdultFeatureEngineer must be fitted first.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("AdultFeatureEngineer requires a DataFrame.")

        missing = set(self.feature_names_in_) - set(X.columns)

        if missing:
            raise ValueError(f"Missing input features: {sorted(missing)}")

        return self._transform_frame(X)

    def _transform_frame(self, X):
        """Build a DataFrame without invoking scikit-learn output wrapping."""
        transformed = X.loc[:, self.feature_names_in_].copy(deep=True)

        # ----------------------------------------------------
        # Capital gain/loss feature engineering
        # ----------------------------------------------------

        for column in CAPITAL_COLUMNS:
            values = pd.to_numeric(
                transformed[column],
                errors="coerce",
            )

            # Negative capital values are not expected
            # in the Adult Income dataset.
            if values.lt(0).any():
                raise ValueError(f"{column} contains negative values.")

            if self.capital_indicators:
                transformed[f"{column}-positive"] = (values > 0).astype("int64")

            if self.capital_transform in {
                "log1p",
                "both",
            }:
                transformed[f"{column}-log1p"] = np.log1p(values)

        # Remove original capital features when only
        # the logarithmic representation is requested.
        if self.capital_transform == "log1p":
            transformed = transformed.drop(columns=CAPITAL_COLUMNS)

        # ----------------------------------------------------
        # Education representation
        # ----------------------------------------------------

        if self.education_representation == "education-num":
            transformed = transformed.drop(columns=["education"])

        elif self.education_representation == "education":
            transformed = transformed.drop(columns=["education-num"])

        # ----------------------------------------------------
        # Selected interaction features
        # ----------------------------------------------------

        if self.interaction_features:
            # Age × educational attainment.
            #
            # education-num is an ordered numerical
            # representation of education level.
            transformed["age-education-interaction"] = pd.to_numeric(
                X["age"],
                errors="coerce",
            ) * pd.to_numeric(
                X["education-num"],
                errors="coerce",
            )

        return transformed


# ============================================================
# 11. Custom occupation-hours interaction transformer
# ============================================================


class OccupationHoursTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Create occupation-specific working-hours features.

    The transformer learns occupation categories only
    from the fitting dataset.

    Each category receives an interaction column:

        occupation_indicator * hours_per_week

    Unseen occupations produce zeros across the learned
    occupation-specific columns.
    """

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("OccupationHoursTransformer requires a DataFrame.")

        missing = {"occupation", "hours-per-week"} - set(X.columns)
        if missing:
            raise ValueError(f"Missing input features: {sorted(missing)}")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]

        self.occupations_ = sorted(
            X["occupation"].dropna().astype(str).unique().tolist()
        )

        self.output_features_ = np.asarray(
            list(self.feature_names_in_)
            + [
                f"occupation-hours-interaction-{index}"
                for index in range(len(self.occupations_))
            ],
            dtype=object,
        )

        return self

    def get_feature_names_out(self, input_features=None):
        """Return input names followed by the learned interaction names."""
        check_is_fitted(self, "output_features_")
        if input_features is not None and not np.array_equal(
            np.asarray(input_features, dtype=object), self.feature_names_in_
        ):
            raise ValueError("input_features must match feature_names_in_.")
        return self.output_features_.copy()

    def transform(self, X):

        if not hasattr(self, "occupations_"):
            raise RuntimeError("OccupationHoursTransformer must be fitted first.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("OccupationHoursTransformer requires a DataFrame.")
        missing = set(self.feature_names_in_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing input features: {sorted(missing)}")

        transformed = X.loc[:, self.feature_names_in_].copy(deep=True)

        hours = pd.to_numeric(
            transformed["hours-per-week"],
            errors="coerce",
        )

        for index, occupation in enumerate(self.occupations_):
            column_name = f"occupation-hours-interaction-{index}"

            transformed[column_name] = (
                transformed["occupation"].eq(occupation).fillna(False).astype("float64")
                * hours
            )

        return transformed


# ============================================================
# 12. Determine transformed feature groups
# ============================================================


def get_feature_groups(
    config: PreprocessingConfig,
) -> tuple[list[str], list[str]]:
    """
    Determine numerical and categorical columns after
    EDA-driven feature engineering.
    """

    numerical = NUMERICAL_FEATURES.copy()

    categorical = CATEGORICAL_FEATURES.copy()

    # Education representation.
    if config.education_representation == "education-num":
        categorical.remove("education")

    elif config.education_representation == "education":
        numerical.remove("education-num")

    # Capital representation.
    if config.capital_transform == "log1p":
        for column in CAPITAL_COLUMNS:
            numerical.remove(column)

    if config.capital_transform in {"log1p", "both"}:
        for column in CAPITAL_COLUMNS:
            numerical.append(f"{column}-log1p")

    # Binary capital indicators.
    if config.capital_indicators:
        for column in CAPITAL_COLUMNS:
            numerical.append(f"{column}-positive")

    # Interaction features.
    if config.interaction_features:
        numerical.append("age-education-interaction")

        # Dynamically select occupation-specific interactions.
        # These columns are generated during pipeline fitting.

    return numerical, categorical


# ============================================================
# 13. Build configurable preprocessing pipeline
# ============================================================
class NumericalFeatureSelector:
    """
    Select numerical features, including dynamically
    generated occupation-hours interactions.

    A top-level class is compatible with joblib serialization.
    """

    def __init__(self, feature_names, include_interactions=False):
        self.feature_names = list(feature_names)
        self.include_interactions = include_interactions

    def __call__(self, X):
        selected = list(self.feature_names)

        if self.include_interactions:
            selected.extend(
                column
                for column in X.columns
                if column.startswith("occupation-hours-interaction-")
            )

        return selected


def build_preprocessor(
    config: PreprocessingConfig | None = None,
) -> Pipeline:
    """
    Build the complete feature preprocessing pipeline.

    Pipeline stages
    ---------------
    1. EDA-driven feature engineering.
    2. Optional occupation-hours interactions.
    3. Numerical imputation and scaling.
    4. Categorical imputation and one-hot encoding.

    The returned pipeline is unfitted.

    It should be combined with a classifier for
    cross-validation.
    """

    if config is None:
        config = PreprocessingConfig()

    numerical_features, categorical_features = get_feature_groups(config)

    # --------------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------------

    numerical_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy=config.numerical_imputation,
                keep_empty_features=True,
            ),
        )
    ]

    if config.numerical_scaling == "standard":
        numerical_steps.append(("scaler", StandardScaler()))

    elif config.numerical_scaling == "robust":
        numerical_steps.append(("scaler", RobustScaler()))

    numerical_pipeline = Pipeline(steps=numerical_steps)

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    if config.categorical_imputation == "constant":
        categorical_imputer = SimpleImputer(
            strategy="constant",
            fill_value=config.missing_category,
            keep_empty_features=True,
        )

    else:
        categorical_imputer = SimpleImputer(
            strategy="most_frequent",
            keep_empty_features=True,
        )

    categorical_encoder = OneHotEncoder(
        handle_unknown=(
            "infrequent_if_exist"
            if config.rare_category_frequency is not None
            else "ignore"
        ),
        min_frequency=config.rare_category_frequency,
        sparse_output=False,
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", categorical_imputer),
            ("encoder", categorical_encoder),
        ]
    )

    # --------------------------------------------------------
    # ColumnTransformer
    # --------------------------------------------------------

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NumericalFeatureSelector(
                    feature_names=numerical_features,
                    include_interactions=config.interaction_features,
                ),
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # --------------------------------------------------------
    # Complete preprocessing pipeline
    # --------------------------------------------------------

    steps = [
        (
            "feature_engineering",
            AdultFeatureEngineer(
                capital_transform=config.capital_transform,
                capital_indicators=config.capital_indicators,
                education_representation=(config.education_representation),
                interaction_features=config.interaction_features,
            ),
        )
    ]

    if config.interaction_features:
        steps.append(
            (
                "occupation_interactions",
                OccupationHoursTransformer(),
            )
        )

    steps.append(("column_transformer", column_transformer))

    return Pipeline(steps=steps)


# ============================================================
# 14. Transform training and testing datasets
# ============================================================


def transform_datasets(
    preprocessor: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit the preprocessor on training data only.

    Then apply the fitted transformations to testing data.

    This function is appropriate for generating a fixed
    train/test representation.

    For model cross-validation, use the unfitted
    preprocessor inside a classifier Pipeline instead.
    """

    train_array = preprocessor.fit_transform(X_train)

    test_array = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    train_processed = pd.DataFrame(
        train_array,
        columns=feature_names,
        index=X_train.index,
    )

    test_processed = pd.DataFrame(
        test_array,
        columns=feature_names,
        index=X_test.index,
    )

    return train_processed, test_processed


# ============================================================
# 15. Save processed datasets
# ============================================================


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str | Path,
) -> None:
    """
    Save the transformed training and testing datasets.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train.to_csv(
        output_dir / "X_train.csv",
        index=False,
    )

    X_test.to_csv(
        output_dir / "X_test.csv",
        index=False,
    )

    y_train.to_csv(
        output_dir / "y_train.csv",
        index=False,
    )

    y_test.to_csv(
        output_dir / "y_test.csv",
        index=False,
    )


# ============================================================
# 16. Save fitted preprocessor
# ============================================================


def save_preprocessor(
    preprocessor: Pipeline,
    output_path: str | Path,
) -> None:
    """
    Serialize a fitted preprocessing pipeline using joblib.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        preprocessor,
        output_path,
    )


# ============================================================
# 17. Save preprocessing configuration
# ============================================================


def save_config(
    config: PreprocessingConfig,
    output_path: str | Path,
) -> None:
    """
    Save the preprocessing configuration as JSON.

    This makes the experiment settings reproducible
    and easy to inspect or log with MLflow.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(config),
            file,
            indent=4,
        )


# ============================================================
# 18. Run complete preprocessing workflow
# ============================================================


def run_preprocessing_pipeline(
    paths: Sequence[str | Path],
    output_dir: str | Path,
    preprocessor_path: str | Path,
    config: PreprocessingConfig | None = None,
    remove_duplicates: bool = False,
) -> dict:
    """
    Execute the complete preprocessing workflow.

    Returns
    -------
    dict
        Summary of the preprocessing execution.
    """

    if config is None:
        config = PreprocessingConfig()

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    raw_df = load_and_concatenate_data(paths)

    raw_rows = len(raw_df)

    # --------------------------------------------------------
    # Clean and validate
    # --------------------------------------------------------

    cleaned_df = clean_data(
        raw_df,
        remove_duplicates=remove_duplicates,
    )

    validate_data(cleaned_df)

    cleaned_rows = len(cleaned_df)

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    encoded_df = encode_target(cleaned_df)

    # --------------------------------------------------------
    # Separate predictors and target
    # --------------------------------------------------------

    X, y = split_features_target(encoded_df)

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        duplicate_aware=config.duplicate_aware_split,
    )

    # --------------------------------------------------------
    # Build and fit preprocessor
    # --------------------------------------------------------

    preprocessor = build_preprocessor(config)

    X_train_processed, X_test_processed = transform_datasets(
        preprocessor,
        X_train,
        X_test,
    )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    save_processed_data(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        output_dir,
    )

    save_preprocessor(
        preprocessor,
        preprocessor_path,
    )

    save_config(
        config,
        Path(output_dir) / "preprocessing_config.json",
    )

    # --------------------------------------------------------
    # Execution summary
    # --------------------------------------------------------

    summary = {
        "raw_rows": raw_rows,
        "cleaned_rows": cleaned_rows,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "original_features": X.shape[1],
        "processed_features": X_train_processed.shape[1],
        "training_positive_rate": float(y_train.mean()),
        "testing_positive_rate": float(y_test.mean()),
        "test_size": config.test_size,
        "random_state": config.random_state,
        "duplicate_aware_split": config.duplicate_aware_split,
        "remove_duplicates": remove_duplicates,
        "configuration": asdict(config),
    }

    return summary


# ============================================================
# 19. Main execution
# ============================================================


def main():
    """
    Execute preprocessing using the default EDA-driven
    configuration.

    Run from the project root:

        python -m src.data.preprocess
    """

    # Resolve project root from this module's location.
    project_root = Path(__file__).resolve().parents[2]

    raw_paths = [
        project_root / "data/raw/adult.data",
        project_root / "data/raw/adult.test",
    ]

    output_dir = project_root / "data/processed"

    preprocessor_path = project_root / "models/preprocessor.joblib"

    # EDA-driven starting configuration.
    config = PreprocessingConfig(
        categorical_imputation="constant",
        missing_category="Unknown",
        rare_category_frequency=0.01,
        numerical_imputation="median",
        numerical_scaling="standard",
        capital_transform="log1p",
        capital_indicators=True,
        education_representation="education-num",
        interaction_features=False,
        duplicate_aware_split=True,
        test_size=0.20,
        random_state=42,
    )

    summary = run_preprocessing_pipeline(
        paths=raw_paths,
        output_dir=output_dir,
        preprocessor_path=preprocessor_path,
        config=config,
        remove_duplicates=False,
    )

    print("\nPreprocessing completed successfully.\n")

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # Keep serialized custom transformers importable in a fresh Python process.
    # Also allow `python src/data/preprocess.py` by adding the project root.
    import sys

    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.data.preprocess import main as run_main

    run_main()