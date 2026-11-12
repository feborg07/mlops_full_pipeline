"""
Sprint 17 - MLOps Final Project
File: tests/test_preprocess.py

Unit and integration tests for the configurable Adult Income
preprocessing pipeline.

Run from the project root:

    python -m pytest tests/test_preprocess.py -v

Coverage:

    python -m pytest tests/test_preprocess.py -v \
        --cov=src.data.preprocess \
        --cov-report=term-missing
"""

# ============================================================
# 1. Imports
# ============================================================

import json
from dataclasses import asdict, replace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from src.data.preprocess import (
    COLUMN_NAMES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_MAPPING,
    AdultFeatureEngineer,
    OccupationHoursTransformer,
    PreprocessingConfig,
    build_preprocessor,
    clean_data,
    create_train_test_split,
    encode_target,
    get_feature_groups,
    load_and_concatenate_data,
    run_preprocessing_pipeline,
    save_config,
    save_preprocessor,
    save_processed_data,
    split_features_target,
    transform_datasets,
    validate_data,
)

# ============================================================
# 2. Helper functions
# ============================================================


def get_column_transformer(preprocessor):
    """
    Return the ColumnTransformer from the preprocessing
    pipeline.
    """

    return preprocessor.named_steps["column_transformer"]


def get_numerical_pipeline(preprocessor):
    """
    Return the fitted or unfitted numerical pipeline.
    """

    column_transformer = get_column_transformer(preprocessor)

    return column_transformer.named_transformers_["numerical"]


def get_categorical_pipeline(preprocessor):
    """
    Return the fitted categorical pipeline.
    """

    column_transformer = get_column_transformer(preprocessor)

    return column_transformer.named_transformers_["categorical"]


def get_feature_names(preprocessor):
    """
    Return the final feature names after preprocessing.
    """

    return preprocessor.get_feature_names_out().tolist()


def get_feature_group_ids(X):
    """
    Assign an integer identifier to each unique feature
    vector.

    Identical feature vectors receive identical IDs,
    including rows containing equivalent missing values.
    """

    index = pd.MultiIndex.from_frame(X.reset_index(drop=True))

    codes, _ = pd.factorize(
        index,
        sort=False,
    )

    return codes


def get_interaction_columns(columns):
    """
    Return the occupation-specific working-hours columns.
    """

    return [
        column
        for column in columns
        if column.startswith("occupation-hours-interaction-")
    ]


def split(X, y, duplicate_aware=False):
    """
    Split X and y with the settings shared by most tests:
    25% test data and random_state=42.
    """

    return create_train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        duplicate_aware=duplicate_aware,
    )


def assert_groups_disjoint(X_train, X_test):
    """
    Assert that no feature vector appears in both partitions.
    """

    ids = get_feature_group_ids(pd.concat([X_train, X_test]))

    assert set(ids[: len(X_train)]).isdisjoint(ids[len(X_train) :])


# ============================================================
# 3. Configuration tests
# ============================================================


class TestPreprocessingConfig:
    def test_default_configuration(self):
        """
        Default configuration should reflect the selected
        EDA-driven starting assumptions.
        """

        config = PreprocessingConfig()

        assert config.categorical_imputation == "constant"
        assert config.missing_category == "Unknown"
        assert config.rare_category_frequency == 0.01

        assert config.numerical_imputation == "median"
        assert config.numerical_scaling == "standard"

        assert config.capital_transform == "log1p"
        assert config.capital_indicators is True

        assert config.education_representation == "education-num"
        assert config.interaction_features is False

        assert config.duplicate_aware_split is True
        assert config.test_size == 0.20
        assert config.random_state == 42

    @pytest.mark.parametrize(
        "parameter,value",
        [
            ("categorical_imputation", "invalid"),
            ("missing_category", ""),
            ("rare_category_frequency", -0.1),
            ("rare_category_frequency", 1.5),
            ("numerical_imputation", "mode"),
            ("numerical_scaling", "invalid"),
            ("capital_transform", "sqrt"),
            ("education_representation", "invalid"),
            ("test_size", 0),
            ("test_size", 1),
            ("random_state", 3.5),
        ],
    )
    def test_invalid_configuration(self, parameter, value):
        """
        Invalid configuration values should fail immediately.
        """

        with pytest.raises(ValueError):
            PreprocessingConfig(**{parameter: value})

    def test_configuration_is_immutable(self, default_config):
        """
        Frozen dataclasses prevent accidental reassignment.
        """

        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            default_config.random_state = 100

    def test_configuration_can_be_converted_to_dictionary(self, default_config):
        """
        Configuration must be serializable for experiment
        tracking and reproducibility.
        """

        config_dict = asdict(default_config)

        assert isinstance(config_dict, dict)
        assert config_dict["random_state"] == 42
        assert config_dict["missing_category"] == "Unknown"

    def test_alternative_configurations(self, alternative_configs):
        """
        All provided alternative configurations should be
        valid PreprocessingConfig instances.
        """

        for config in alternative_configs.values():
            assert isinstance(config, PreprocessingConfig)


# ============================================================
# 4. Data loading tests
# ============================================================


class TestDataLoading:
    def test_load_and_concatenate_data(self, sample_raw_files):
        """
        Both raw files should be loaded and concatenated.
        """

        df = load_and_concatenate_data(sample_raw_files)

        assert len(df) == 12
        assert set(COLUMN_NAMES).issubset(df.columns)
        assert "source_file" in df.columns

    def test_source_file_tracking(self, sample_raw_files):
        """
        Source metadata should identify the original file.
        """

        df = load_and_concatenate_data(sample_raw_files)

        assert set(df["source_file"]) == {"adult.data", "adult.test"}
        assert df["source_file"].value_counts().to_dict() == {
            "adult.data": 8,
            "adult.test": 4,
        }

    def test_test_file_metadata_is_skipped(self, sample_raw_files):
        """
        The adult.test metadata line should not become
        a dataset observation.
        """

        df = load_and_concatenate_data(sample_raw_files)

        assert len(df) == 12
        assert not df["age"].astype(str).str.contains("Cross validator").any()

    def test_missing_file_raises_error(self, tmp_path):
        """
        Missing raw files should raise FileNotFoundError.
        """

        with pytest.raises(FileNotFoundError):
            load_and_concatenate_data([tmp_path / "missing.data"])

    def test_empty_paths_raise_error(self):
        """
        At least one input path is required.
        """

        with pytest.raises(ValueError):
            load_and_concatenate_data([])


# ============================================================
# 5. Data cleaning tests
# ============================================================


class TestDataCleaning:
    @pytest.fixture
    def cleaned(self, dirty_adult_df):
        """
        Return the dirty fixture after default cleaning.
        """

        return clean_data(dirty_adult_df)

    def test_clean_data_preserves_row_count(self, cleaned):
        """
        Cleaning should preserve duplicate records by default.
        """

        assert len(cleaned) == 13

    def test_clean_data_removes_whitespace(self, cleaned):
        """
        Surrounding whitespace should be removed.
        """

        assert cleaned.loc[0, "workclass"] == "Private"

    def test_clean_data_normalizes_missing_markers(self, cleaned):
        """
        Question marks should become missing values.
        """

        assert pd.isna(cleaned.loc[1, "occupation"])

    def test_clean_data_preserves_missing_values(self, cleaned):
        """
        Missing categorical values should remain missing
        until the fitted preprocessing pipeline handles them.
        """

        assert pd.isna(cleaned.loc[2, "native-country"])

    def test_clean_data_converts_numerical_values(self, cleaned):
        """
        Numeric strings should be converted to numeric types.
        """

        assert pd.api.types.is_numeric_dtype(cleaned["age"])
        assert cleaned.loc[3, "age"] == 51

    def test_clean_data_normalizes_target_labels(self, cleaned):
        """
        Trailing periods in adult.test labels should be removed.
        """

        assert cleaned.loc[4, "income"] == "<=50K"

    def test_optional_duplicate_removal(self, dirty_adult_df):
        """
        Duplicate removal should only occur when requested.
        """

        assert len(clean_data(dirty_adult_df, remove_duplicates=True)) == 12

    def test_clean_data_does_not_modify_original(self, dirty_adult_df):
        """
        Cleaning should not mutate the input DataFrame.
        """

        original = dirty_adult_df.copy(deep=True)

        clean_data(dirty_adult_df)

        pd.testing.assert_frame_equal(dirty_adult_df, original)


# ============================================================
# 6. Dataset validation tests
# ============================================================


class TestDataValidation:
    def test_valid_dataset(self, sample_adult_df):
        """
        A valid dataset should pass validation.
        """

        validate_data(sample_adult_df)

    def test_missing_required_column(self, sample_adult_df):
        """
        Missing predictor columns should raise ValueError.
        """

        df = sample_adult_df.drop(columns=["occupation"])

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data(df)

    def test_empty_dataset(self, sample_adult_df):
        """
        An empty dataset should be rejected.
        """

        with pytest.raises(ValueError, match="empty"):
            validate_data(sample_adult_df.iloc[0:0])

    def test_invalid_target_label(self, sample_adult_df):
        """
        Unexpected target labels should be rejected.
        """

        df = sample_adult_df.copy()
        df.loc[0, TARGET_COLUMN] = "INVALID"

        with pytest.raises(ValueError, match="Unexpected target labels"):
            validate_data(df)

    def test_single_target_class(self, sample_adult_df):
        """
        A supervised binary dataset must contain both classes.
        """

        df = sample_adult_df.copy()
        df[TARGET_COLUMN] = "<=50K"

        with pytest.raises(ValueError, match="both income classes"):
            validate_data(df)


# ============================================================
# 7. Target encoding tests
# ============================================================


class TestTargetEncoding:
    def test_encode_target(self, sample_adult_df):
        """
        Target labels should be converted to 0 and 1.
        """

        encoded = encode_target(sample_adult_df)

        assert set(encoded[TARGET_COLUMN]) == {0, 1}

    def test_target_mapping(self, sample_adult_df):
        """
        Income labels should follow the defined mapping.
        """

        encoded = encode_target(sample_adult_df)

        for original, numerical in TARGET_MAPPING.items():
            mask = sample_adult_df[TARGET_COLUMN] == original

            assert (encoded.loc[mask, TARGET_COLUMN] == numerical).all()

    def test_encode_target_preserves_original(self, sample_adult_df):
        """
        Target encoding should not modify the source data.
        """

        original = sample_adult_df.copy(deep=True)

        encode_target(sample_adult_df)

        pd.testing.assert_frame_equal(sample_adult_df, original)


# ============================================================
# 8. Feature and target separation tests
# ============================================================


class TestFeatureTargetSplit:
    def test_feature_count(self, encoded_adult_df):
        """
        The original model input should contain 14 predictors.
        """

        X, _y = split_features_target(encoded_adult_df)

        assert X.shape[1] == 14
        assert list(X.columns) == FEATURE_COLUMNS

    @pytest.mark.parametrize("column", [TARGET_COLUMN, "source_file"])
    def test_non_predictors_excluded_from_features(self, column, encoded_adult_df):
        """
        Income must not appear in the predictor matrix.

        Source-file metadata must not become a predictor.

        Leakage prevention (section 15):
        The income target must never enter the preprocessing
        feature matrix.
        Source-file metadata must not influence predictions.
        """

        X, _y = split_features_target(encoded_adult_df)

        assert column not in X.columns

    def test_target_values(self, encoded_adult_df):
        """
        The target vector should contain binary labels.
        """

        _X, y = split_features_target(encoded_adult_df)

        assert set(y.unique()) == {0, 1}


# ============================================================
# 9. Train/test splitting tests
# ============================================================


class TestTrainTestSplit:
    def test_standard_split_preserves_all_rows(self, features_target):
        """
        Standard splitting should not lose observations.
        """

        X, y = features_target

        X_train, X_test, y_train, y_test = split(X, y)

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

    def test_standard_split_is_reproducible(self, features_target):
        """
        The same random seed should reproduce the same split.
        """

        X, y = features_target

        for a, b in zip(split(X, y), split(X, y)):
            if isinstance(a, pd.DataFrame):
                pd.testing.assert_frame_equal(a, b)
            else:
                pd.testing.assert_series_equal(a, b)

    def test_standard_split_is_stratified(self, features_target):
        """
        The class proportions should remain approximately
        balanced across the two partitions.
        """

        X, y = features_target

        _X_train, _X_test, y_train, y_test = split(X, y)

        # Calculate the positive-class proportion in the full dataset.
        overall_rate = y.mean()

        # Allow for rounding because the synthetic dataset is small.
        assert abs(y_train.mean() - overall_rate) <= 1 / len(y_train)
        assert abs(y_test.mean() - overall_rate) <= 1 / len(y_test)

    def test_duplicate_aware_split_preserves_rows(self, duplicate_groups_df):
        """
        Duplicate-aware splitting must preserve all records.
        """

        X, y = split_features_target(encode_target(duplicate_groups_df))

        X_train, X_test, _y_train, _y_test = split(X, y, duplicate_aware=True)

        assert len(X_train) + len(X_test) == 17

    @pytest.mark.parametrize(
        "fixture_name", ["duplicate_groups_df", "conflicting_labels_df"]
    )
    def test_duplicate_aware_split_prevents_group_overlap(self, fixture_name, request):
        """
        Identical feature vectors must not cross the
        train/test boundary.

        conflicting_labels_df case:
        Identical feature vectors should stay together
        even when their target labels differ.
        """

        df = request.getfixturevalue(fixture_name)

        X, y = split_features_target(encode_target(df))

        X_train, X_test, _y_train, _y_test = split(X, y, duplicate_aware=True)

        assert_groups_disjoint(X_train, X_test)

    def test_mismatched_lengths_raise_error(self, features_target):
        """
        X and y must have equal numbers of observations.
        """

        X, y = features_target

        with pytest.raises(ValueError):
            create_train_test_split(X, y.iloc[:-1])

    def test_mismatched_indices_raise_error(self, features_target):
        """
        X and y indices must match.
        """

        X, y = features_target

        y = y.copy()
        y.index = y.index + 100

        with pytest.raises(ValueError):
            create_train_test_split(X, y)


# ============================================================
# 10. Feature engineering tests
# ============================================================


class TestFeatureEngineering:
    def test_log1p_capital_transform(self, capital_features_df):
        """
        Logarithmic capital features should equal log1p
        of the original values.
        """

        X = capital_features_df[FEATURE_COLUMNS]

        transformed = AdultFeatureEngineer(
            capital_transform="log1p",
            capital_indicators=True,
        ).fit_transform(X)

        for column in ("capital-gain", "capital-loss"):
            np.testing.assert_allclose(
                transformed[f"{column}-log1p"],
                np.log1p(X[column]),
            )

    @pytest.mark.parametrize(
        "capital_transform,present,absent",
        [
            (
                "log1p",
                ["capital-gain-log1p", "capital-loss-log1p"],
                ["capital-gain", "capital-loss"],
            ),
            (
                "original",
                ["capital-gain", "capital-loss"],
                ["capital-gain-log1p"],
            ),
            (
                "both",
                [
                    "capital-gain",
                    "capital-gain-log1p",
                    "capital-loss",
                    "capital-loss-log1p",
                ],
                [],
            ),
        ],
    )
    def test_capital_representations(
        self, capital_transform, present, absent, capital_features_df
    ):
        """
        "log1p" should replace the original capital columns.

        Original capital values should be retained when
        capital_transform="original".

        The 'both' configuration should retain original
        and logarithmic capital features.
        """

        X = capital_features_df[FEATURE_COLUMNS]

        transformed = AdultFeatureEngineer(
            capital_transform=capital_transform,
        ).fit_transform(X)

        for column in present:
            assert column in transformed.columns

        for column in absent:
            assert column not in transformed.columns

    def test_capital_presence_indicators(self, capital_features_df):
        """
        Positive-capital indicators should contain 0 or 1.
        """

        X = capital_features_df[FEATURE_COLUMNS]

        transformed = AdultFeatureEngineer(capital_indicators=True).fit_transform(X)

        gain_indicator = transformed["capital-gain-positive"]
        loss_indicator = transformed["capital-loss-positive"]

        assert set(gain_indicator.unique()).issubset({0, 1})
        assert set(loss_indicator.unique()).issubset({0, 1})

        assert gain_indicator.iloc[0] == 0
        assert gain_indicator.iloc[1] == 1

        assert loss_indicator.iloc[0] == 0
        assert loss_indicator.iloc[2] == 1

    def test_capital_indicators_can_be_disabled(self, capital_features_df):
        """
        Disabling capital indicators should remove the
        additional binary features.
        """

        X = capital_features_df[FEATURE_COLUMNS]

        transformed = AdultFeatureEngineer(capital_indicators=False).fit_transform(X)

        assert "capital-gain-positive" not in transformed.columns
        assert "capital-loss-positive" not in transformed.columns

    def test_negative_capital_values_raise_error(self, capital_features_df):
        """
        Negative capital values should be rejected because
        log1p is configured for nonnegative capital values.
        """

        X = capital_features_df[FEATURE_COLUMNS].copy()
        X.loc[X.index[0], "capital-gain"] = -100

        with pytest.raises(ValueError, match="negative"):
            AdultFeatureEngineer().fit_transform(X)

    @pytest.mark.parametrize(
        "representation,present,absent",
        [
            ("education-num", ["education-num"], ["education"]),
            ("education", ["education"], ["education-num"]),
            ("both", ["education", "education-num"], []),
        ],
    )
    def test_education_representations(
        self, representation, present, absent, education_features_df
    ):
        """
        representation="education-num":
        education-num should be retained while the text
        education feature is removed.

        representation="education":
        Text education should be retained while
        education-num is removed.

        representation="both":
        Both education features should remain available.
        """

        X = education_features_df[FEATURE_COLUMNS]

        transformed = AdultFeatureEngineer(
            education_representation=representation,
        ).fit_transform(X)

        for column in present:
            assert column in transformed.columns

        for column in absent:
            assert column not in transformed.columns

    def test_age_education_interaction(self, interaction_features_df):
        """
        The age-education interaction should equal the
        product of age and education-num.
        """

        X = interaction_features_df[FEATURE_COLUMNS]

        transformed = AdultFeatureEngineer(interaction_features=True).fit_transform(X)

        expected = X["age"] * X["education-num"]

        np.testing.assert_allclose(transformed["age-education-interaction"], expected)

    def test_feature_engineer_requires_fit(self, sample_adult_df):
        """
        transform() should not be used before fit().
        """

        with pytest.raises(RuntimeError):
            AdultFeatureEngineer().transform(sample_adult_df[FEATURE_COLUMNS])

    def test_feature_engineering_does_not_mutate_input(self, sample_adult_df):
        """
        Feature engineering should not modify the input.
        """

        X = sample_adult_df[FEATURE_COLUMNS].copy()
        original = X.copy(deep=True)

        AdultFeatureEngineer().fit_transform(X)

        pd.testing.assert_frame_equal(X, original)


# ============================================================
# 11. Occupation-hours interaction tests
# ============================================================


class TestOccupationHoursTransformer:
    def test_occupation_hours_interactions(self, interaction_features_df):
        """
        Occupation-specific interaction columns should
        be generated.
        """

        X = interaction_features_df[FEATURE_COLUMNS]

        transformed = OccupationHoursTransformer().fit_transform(X)

        assert len(get_interaction_columns(transformed.columns)) > 0

    def test_occupation_interaction_values(self, interaction_features_df):
        """
        Each interaction should equal:

            occupation_indicator * hours_per_week
        """

        X = interaction_features_df[FEATURE_COLUMNS]

        transformer = OccupationHoursTransformer()
        transformed = transformer.fit_transform(X)

        for index, occupation in enumerate(transformer.occupations_):
            column = f"occupation-hours-interaction-{index}"

            expected = (
                X["occupation"].eq(occupation).astype(float) * X["hours-per-week"]
            )

            np.testing.assert_allclose(transformed[column], expected)

    def test_unseen_occupation_interactions(
        self, interaction_features_df, unseen_category_df
    ):
        """
        An unseen occupation should produce zero values
        across all learned occupation-specific interactions.
        """

        X_train = interaction_features_df[FEATURE_COLUMNS]
        X_test = unseen_category_df[FEATURE_COLUMNS]

        transformer = OccupationHoursTransformer().fit(X_train)
        transformed = transformer.transform(X_test)

        interaction_columns = get_interaction_columns(transformed.columns)

        assert (transformed[interaction_columns] == 0).all().all()


# ============================================================
# 12. Feature-group configuration tests
# ============================================================


class TestFeatureGroups:
    def test_default_feature_groups(self, default_config):
        """
        The default configuration should use numerical
        education and logarithmic capital features.
        """

        numerical, categorical = get_feature_groups(default_config)

        assert "education-num" in numerical
        assert "education" not in categorical

        assert "capital-gain-log1p" in numerical
        assert "capital-loss-log1p" in numerical

        assert "capital-gain" not in numerical
        assert "capital-loss" not in numerical

    def test_original_capital_feature_groups(self, default_config):
        """
        Original capital features should remain numerical.
        """

        config = replace(default_config, capital_transform="original")

        numerical, _categorical = get_feature_groups(config)

        assert "capital-gain" in numerical
        assert "capital-loss" in numerical

    def test_categorical_education_feature_groups(self, default_config):
        """
        Categorical education should appear in the
        categorical feature list.
        """

        config = replace(default_config, education_representation="education")

        numerical, categorical = get_feature_groups(config)

        assert "education" in categorical
        assert "education-num" not in numerical


# ============================================================
# 13. Preprocessor construction tests
# ============================================================


class TestBuildPreprocessor:
    def test_build_returns_pipeline(self, default_config):
        """
        The preprocessor should be a scikit-learn Pipeline.
        """

        assert isinstance(build_preprocessor(default_config), Pipeline)

    @pytest.mark.parametrize(
        "scaling,scaler_class",
        [
            ("standard", StandardScaler),
            ("robust", RobustScaler),
            ("none", None),
        ],
    )
    def test_numerical_scaling(
        self, scaling, scaler_class, default_config, features_target
    ):
        """
        "standard": The default numerical scaler should be StandardScaler.

        "robust": RobustScaler should be selected when configured.

        "none": Scaling should be optional.
        """

        X, y = features_target

        preprocessor = build_preprocessor(
            replace(default_config, numerical_scaling=scaling)
        )
        preprocessor.fit(X, y)

        steps = get_numerical_pipeline(preprocessor).named_steps

        if scaler_class is None:
            assert "scaler" not in steps
        else:
            assert isinstance(steps["scaler"], scaler_class)

    def test_categorical_encoder(self, fitted_preprocessor):
        """
        Categorical features should use OneHotEncoder.
        """

        categorical_pipeline = get_categorical_pipeline(
            fitted_preprocessor["preprocessor"]
        )

        assert isinstance(categorical_pipeline.named_steps["encoder"], OneHotEncoder)

    @pytest.mark.parametrize("strategy", ["constant", "most_frequent"])
    def test_categorical_imputation(self, strategy, missing_values_df, default_config):
        """
        Missing categorical values should be replaced
        with the configured Unknown category.

        Most-frequent imputation should be selectable.
        """

        X = missing_values_df[FEATURE_COLUMNS]

        preprocessor = build_preprocessor(
            replace(default_config, categorical_imputation=strategy)
        )
        preprocessor.fit(X)

        imputer = get_categorical_pipeline(preprocessor).named_steps["imputer"]

        assert imputer.strategy == strategy

        if strategy == "constant":
            assert imputer.fill_value == "Unknown"

    def test_processed_features_have_no_missing_values(
        self, numerical_missing_df, default_config
    ):
        """
        Numerical imputation should eliminate missing values
        from the transformed output.
        """

        X = numerical_missing_df[FEATURE_COLUMNS]

        transformed = build_preprocessor(default_config).fit_transform(X)

        assert np.isfinite(transformed).all()

    def test_train_test_feature_dimensions_match(self, fitted_preprocessor):
        """
        Training and testing data should produce identical
        feature dimensions.
        """

        preprocessor = fitted_preprocessor["preprocessor"]

        train_transformed = preprocessor.transform(fitted_preprocessor["X_train"])
        test_transformed = preprocessor.transform(fitted_preprocessor["X_test"])

        assert train_transformed.shape[1] == test_transformed.shape[1]

    def test_unseen_categories_are_handled(
        self, fitted_preprocessor, unseen_category_df
    ):
        """
        Unseen categories should not cause transformation
        errors.
        """

        preprocessor = fitted_preprocessor["preprocessor"]

        transformed = preprocessor.transform(unseen_category_df[FEATURE_COLUMNS])

        assert transformed.shape[0] == 1
        assert np.isfinite(transformed).all()

    def test_input_dataframe_not_modified(self, sample_adult_df, default_config):
        """
        Preprocessing should not mutate the input DataFrame.
        """

        X = sample_adult_df[FEATURE_COLUMNS].copy()
        original = X.copy(deep=True)

        build_preprocessor(default_config).fit_transform(X)

        pd.testing.assert_frame_equal(X, original)


# ============================================================
# 14. Rare-category encoding tests
# ============================================================


class TestRareCategoryEncoding:
    def test_rare_countries_are_grouped(self, rare_country_df, default_config):
        """
        Countries below the frequency threshold should
        be grouped into an infrequent category.
        """

        config = replace(default_config, rare_category_frequency=0.05)

        preprocessor = build_preprocessor(config).fit(rare_country_df[FEATURE_COLUMNS])

        feature_names = get_feature_names(preprocessor)

        assert "native-country_infrequent_sklearn" in feature_names
        assert "native-country_United-States" in feature_names
        assert "native-country_Mexico" in feature_names
        assert "native-country_Portugal" not in feature_names

    def test_no_rare_grouping(self, rare_country_df, default_config):
        """
        Disabling rare grouping should retain separate
        country categories.
        """

        config = replace(default_config, rare_category_frequency=None)

        preprocessor = build_preprocessor(config).fit(rare_country_df[FEATURE_COLUMNS])

        feature_names = get_feature_names(preprocessor)

        assert "native-country_Portugal" in feature_names
        assert "native-country_Germany" in feature_names

    def test_unseen_country_is_handled(
        self, rare_country_df, unseen_category_df, default_config
    ):
        """
        Unseen countries should not cause an error.
        """

        config = replace(default_config, rare_category_frequency=0.05)

        preprocessor = build_preprocessor(config).fit(rare_country_df[FEATURE_COLUMNS])

        transformed = preprocessor.transform(unseen_category_df[FEATURE_COLUMNS])

        assert transformed.shape[0] == 1
        assert np.isfinite(transformed).all()


# ============================================================
# 15. Leakage prevention tests
# ============================================================

# Target and source_file exclusion from the feature matrix is
# checked in section 8 (test_non_predictors_excluded_from_features).


class TestLeakagePrevention:
    def test_scaler_is_fitted_on_training_data_only(self, split_data, default_config):
        """
        The scaler's learned mean must reflect training
        observations, not the combined train/test dataset.
        """

        X_train, _X_test, y_train, _y_test = split_data

        preprocessor = build_preprocessor(default_config).fit(X_train, y_train)

        scaler = get_numerical_pipeline(preprocessor).named_steps["scaler"]

        # Age is the first numerical feature in the
        # default feature-group configuration.
        expected_age_mean = X_train["age"].mean()

        assert scaler.mean_[0] == pytest.approx(expected_age_mean)

    def test_test_data_does_not_change_fitted_statistics(
        self, split_data, default_config
    ):
        """
        Transforming testing data should not change the
        scaler statistics learned from training data.
        """

        X_train, X_test, y_train, _y_test = split_data

        preprocessor = build_preprocessor(default_config).fit(X_train, y_train)

        scaler = get_numerical_pipeline(preprocessor).named_steps["scaler"]

        original_means = scaler.mean_.copy()

        # Introduce an extreme value in the testing set.
        X_test_modified = X_test.copy()
        X_test_modified["age"] = 100000

        preprocessor.transform(X_test_modified)

        np.testing.assert_array_equal(scaler.mean_, original_means)


# ============================================================
# 16. Transformation output tests
# ============================================================


class TestTransformDatasets:
    @pytest.fixture
    def processed(self, split_data, default_config):
        """
        Return the split data and its transformed train/test
        DataFrames from transform_datasets().
        """

        X_train, X_test, _y_train, _y_test = split_data

        train_processed, test_processed = transform_datasets(
            build_preprocessor(default_config),
            X_train,
            X_test,
        )

        return X_train, X_test, train_processed, test_processed

    def test_transform_datasets_returns_dataframes(self, processed):
        """
        The transformation function should return two
        pandas DataFrames.
        """

        _X_train, _X_test, train_processed, test_processed = processed

        assert isinstance(train_processed, pd.DataFrame)
        assert isinstance(test_processed, pd.DataFrame)

    def test_transformed_columns_match(self, processed):
        """
        Training and testing datasets must have the
        same transformed feature names.
        """

        _X_train, _X_test, train_processed, test_processed = processed

        assert list(train_processed.columns) == list(test_processed.columns)

    def test_transformed_indices_are_preserved(self, processed):
        """
        Transformed DataFrames should preserve the
        original partition indices.
        """

        X_train, X_test, train_processed, test_processed = processed

        assert train_processed.index.equals(X_train.index)
        assert test_processed.index.equals(X_test.index)

    def test_transformed_data_contains_finite_values(self, processed):
        """
        The transformed feature matrices should contain
        no NaN or infinite values.
        """

        _X_train, _X_test, train_processed, test_processed = processed

        assert np.isfinite(train_processed.to_numpy()).all()
        assert np.isfinite(test_processed.to_numpy()).all()


# ============================================================
# 17. Artifact serialization tests
# ============================================================


class TestArtifactSerialization:
    def test_save_processed_data(self, split_data, default_config, output_dir):
        """
        Processed datasets should be saved as four CSV files.
        """

        X_train, X_test, y_train, y_test = split_data

        X_train_processed, X_test_processed = transform_datasets(
            build_preprocessor(default_config),
            X_train,
            X_test,
        )

        save_processed_data(
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
            output_dir,
        )

        for filename in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
            assert (output_dir / filename).is_file()

    def test_save_and_reload_preprocessor(
        self, fitted_preprocessor, preprocessor_path
    ):
        """
        A serialized preprocessor should produce the same
        transformations after reloading.
        """

        preprocessor = fitted_preprocessor["preprocessor"]
        X_test = fitted_preprocessor["X_test"]

        expected = preprocessor.transform(X_test)

        save_preprocessor(preprocessor, preprocessor_path)

        restored = joblib.load(preprocessor_path)

        np.testing.assert_allclose(restored.transform(X_test), expected)

    def test_save_configuration(self, default_config, config_path):
        """
        The saved JSON configuration should match the
        dataclass configuration.
        """

        save_config(default_config, config_path)

        assert config_path.is_file()

        with config_path.open("r", encoding="utf-8") as file:
            saved = json.load(file)

        assert saved == asdict(default_config)


# ============================================================
# 18. End-to-end integration tests
# ============================================================


class TestPreprocessingIntegration:
    @pytest.fixture
    def summary(self, sample_raw_files, output_dir, preprocessor_path, default_config):
        """
        Run the complete pipeline once on the synthetic raw
        files and return its summary dictionary.
        """

        return run_preprocessing_pipeline(
            paths=sample_raw_files,
            output_dir=output_dir,
            preprocessor_path=preprocessor_path,
            config=default_config,
            remove_duplicates=False,
        )

    def test_complete_preprocessing_pipeline(self, summary, preprocessor_path):
        """
        The complete pipeline should execute successfully
        using synthetic raw Adult Income files.
        """

        assert isinstance(summary, dict)

        assert summary["raw_rows"] == 12
        assert summary["cleaned_rows"] == 12

        assert summary["training_rows"] + summary["testing_rows"] == 12

        assert summary["original_features"] == 14
        assert summary["processed_features"] > 0

        assert preprocessor_path.is_file()

    def test_complete_pipeline_saves_all_artifacts(
        self, summary, output_dir, preprocessor_path
    ):
        """
        The end-to-end pipeline should save processed
        datasets, configuration, and preprocessor.
        """

        expected_files = [
            output_dir / "X_train.csv",
            output_dir / "X_test.csv",
            output_dir / "y_train.csv",
            output_dir / "y_test.csv",
            output_dir / "preprocessing_config.json",
            preprocessor_path,
        ]

        for path in expected_files:
            assert path.is_file()

    def test_pipeline_preserves_all_observations(self, summary):
        """
        Without explicit duplicate removal, the pipeline
        should preserve all valid observations.
        """

        assert (
            summary["training_rows"] + summary["testing_rows"]
            == summary["cleaned_rows"]
        )

    def test_pipeline_is_reproducible(self, sample_raw_files, tmp_path, default_config):
        """
        Identical configurations and input data should
        produce identical processed datasets.
        """

        output_a = tmp_path / "run_a"
        output_b = tmp_path / "run_b"

        for output in (output_a, output_b):
            run_preprocessing_pipeline(
                paths=sample_raw_files,
                output_dir=output,
                preprocessor_path=output / "preprocessor.joblib",
                config=default_config,
            )

        for filename in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
            pd.testing.assert_frame_equal(
                pd.read_csv(output_a / filename),
                pd.read_csv(output_b / filename),
            )


# ============================================================
# 19. Known architecture requirement
# ============================================================


class TestInteractionPipelineIntegration:
    def test_occupation_interactions_reach_model_features(
        self, interaction_features_df, default_config
    ):
        """
        When interaction_features=True, occupation-specific
        working-hours features must appear in the final
        transformed feature matrix.

        """

        config = replace(default_config, interaction_features=True)

        preprocessor = build_preprocessor(config).fit(
            interaction_features_df[FEATURE_COLUMNS]
        )

        interaction_names = get_interaction_columns(get_feature_names(preprocessor))

        assert len(interaction_names) > 0, (
            "Occupation-hours interactions were generated "
            "but did not reach the final feature matrix."
        )


# ============================================================
# 20. Regression tests for feature names, inputs and serialization
# ============================================================


class TestPreprocessingRegressions:
    @pytest.mark.parametrize("seed", [0, 42, np.int32(42), np.int64(42), 2**32 - 1])
    def test_integer_seeds_are_json_serializable(self, seed, tmp_path):
        config = PreprocessingConfig(random_state=seed)
        assert type(config.random_state) is int
        path = tmp_path / "config.json"
        save_config(config, path)
        assert json.loads(path.read_text())["random_state"] == int(seed)

    @pytest.mark.parametrize(
        "seed", [True, False, np.bool_(True), -1, 2**32, 42.5, "42", None]
    )
    def test_invalid_seeds_fail_explicitly(self, seed, features_target):
        with pytest.raises(ValueError, match="random_state"):
            PreprocessingConfig(random_state=seed)
        X, y = features_target
        with pytest.raises(ValueError, match="random_state"):
            create_train_test_split(X, y, random_state=seed)

    @pytest.mark.parametrize("label", ["unknown", "<=50k", 2, None, pd.NA, np.nan])
    def test_invalid_target_encoding(self, label, sample_adult_df):
        df = sample_adult_df.copy()
        # Use a general column type so non-text labels (e.g. 2) can be
        # inserted; pandas 3 text columns refuse integer values.
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(object)
        df.loc[df.index[0], TARGET_COLUMN] = label
        original = df.copy(deep=True)
        with pytest.raises(ValueError, match="Unexpected target labels|missing values"):
            encode_target(df)
        pd.testing.assert_frame_equal(df, original)

    def test_missing_target_column(self, sample_adult_df):
        with pytest.raises(ValueError, match="Missing target column"):
            encode_target(sample_adult_df.drop(columns=TARGET_COLUMN))

    def test_full_pipeline_names_match_matrix(
        self,
        feature_name_config,
        features_target,
    ):
        X, y = features_target
        preprocessor = build_preprocessor(feature_name_config)
        transformed = preprocessor.fit_transform(X, y)
        names = preprocessor.get_feature_names_out()
        assert len(names) == transformed.shape[1]
        assert len(names) == len(set(names))
        assert "occupation-hours" not in names
        assert np.isfinite(transformed).all()
        step_input = X
        for step_name in ("feature_engineering", "occupation_interactions"):
            if step_name in preprocessor.named_steps:
                step = preprocessor.named_steps[step_name]
                actual = step.transform(step_input)
                np.testing.assert_array_equal(
                    step.get_feature_names_out(), actual.columns
                )
                step_input = actual

    @pytest.mark.parametrize(
        "transformer_class", [AdultFeatureEngineer, OccupationHoursTransformer]
    )
    def test_feature_names_require_fit_and_validate_input(
        self, transformer_class, features_target
    ):
        from sklearn.exceptions import NotFittedError

        X, _ = features_target
        transformer = transformer_class()
        with pytest.raises(NotFittedError):
            transformer.get_feature_names_out()
        transformer.fit(X)
        np.testing.assert_array_equal(
            transformer.get_feature_names_out(X.columns),
            transformer.get_feature_names_out(),
        )
        with pytest.raises(ValueError, match="input_features"):
            transformer.get_feature_names_out(list(reversed(X.columns)))
        # Caller mutation must not change stored output names.
        names = transformer.get_feature_names_out()
        names[0] = "changed"
        assert transformer.get_feature_names_out()[0] != "changed"

    def test_duplicate_hours_column_absent(self, features_target):
        X, _ = features_target
        engineered = AdultFeatureEngineer(interaction_features=True).fit_transform(X)
        assert "occupation-hours" not in engineered.columns
        numerical, _ = get_feature_groups(
            PreprocessingConfig(interaction_features=True)
        )
        assert "occupation-hours" not in numerical

    def test_nullable_missing_occupations_produce_zero_interactions(
        self, missing_occupation_features
    ):
        X = missing_occupation_features
        transformer = OccupationHoursTransformer().fit(X)
        transformed = transformer.transform(X)
        names = get_interaction_columns(transformed)
        assert names
        assert (transformed.loc[X.index[:2], names] == 0).all().all()
        assert not transformed[names].isna().any().any()

    def test_occupation_vocabulary_uses_only_fit_data(self, features_target):
        X, _ = features_target
        transformer = OccupationHoursTransformer().fit(X)
        learned = list(transformer.occupations_)
        future = X.iloc[:1].copy()
        future["occupation"] = "Never-seen-occupation"
        transformed = transformer.transform(future)
        names = get_interaction_columns(transformed)
        assert (transformed[names] == 0).all().all()
        assert transformer.occupations_ == learned

    @pytest.mark.parametrize("interactions", [False, True])
    def test_serialization_preserves_full_pipeline_names(
        self,
        interactions,
        features_target,
        tmp_path,
    ):
        X, y = features_target
        preprocessor = build_preprocessor(
            PreprocessingConfig(interaction_features=interactions)
        )
        expected = preprocessor.fit_transform(X, y)
        path = tmp_path / "preprocessor.joblib"
        save_preprocessor(preprocessor, path)
        restored = joblib.load(path)
        np.testing.assert_allclose(restored.transform(X), expected)
        np.testing.assert_array_equal(
            restored.get_feature_names_out(),
            preprocessor.get_feature_names_out(),
        )

    @pytest.mark.parametrize("interactions", [False, True])
    def test_pandas_output_and_reordered_input(self, interactions, features_target):
        X, y = features_target
        preprocessor = build_preprocessor(
            PreprocessingConfig(interaction_features=interactions),
        ).set_output(transform="pandas")
        transformed = preprocessor.fit_transform(X, y)
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.index.equals(X.index)
        np.testing.assert_array_equal(
            transformed.columns, preprocessor.get_feature_names_out()
        )
        reordered = preprocessor.transform(X.loc[:, list(reversed(X.columns))])
        pd.testing.assert_frame_equal(transformed, reordered)

    def test_pipeline_can_be_cloned_inside_cross_validation(self, features_target):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        X, y = features_target
        model = Pipeline(
            [
                (
                    "preprocess",
                    build_preprocessor(PreprocessingConfig(interaction_features=True)),
                ),
                ("classifier", LogisticRegression(max_iter=500)),
            ]
        )
        scores = cross_val_score(model, X, y, cv=3)
        assert len(scores) == 3
        assert np.isfinite(scores).all()
        assert not hasattr(
            model.named_steps["preprocess"].named_steps["feature_engineering"],
            "feature_names_in_",
        )

    def test_small_group_split_avoids_impossible_stratification(self, features_target):
        X, y = features_target
        # Two groups per stratum, but only one test group: stratification is impossible.
        X, y = X.iloc[:4], y.iloc[:4]
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.25)
        assert len(X_test) == 1
        assert len(X_train) + len(X_test) == len(X)
        assert X_train.index.equals(y_train.index)
        assert X_test.index.equals(y_test.index)

    @pytest.mark.parametrize("corruption", ["dropped", "overlapping"])
    def test_split_guard_detects_invalid_group_assignment(
        self, monkeypatch, features_target, corruption
    ):
        import src.data.preprocess as module

        X, y = features_target

        def corrupt_split(groups, **kwargs):
            values = groups.to_numpy()
            if corruption == "dropped":
                return values[:-2], values[-1:]
            return values[:-1], values[-2:]

        monkeypatch.setattr(module, "train_test_split", corrupt_split)
        with pytest.raises(RuntimeError, match="lost observations|assigned them twice"):
            create_train_test_split(X, y)

    def test_cli_artifact_loads_in_a_fresh_process(self, sample_raw_files, tmp_path):
        import shutil
        import subprocess
        import sys
        from pathlib import Path

        import src.data.preprocess as module

        project = tmp_path / "cli_project"
        module_dir = project / "src" / "data"
        module_dir.mkdir(parents=True)
        shutil.copyfile(module.__file__, module_dir / "preprocess.py")
        raw_dir = project / "data" / "raw"
        raw_dir.mkdir(parents=True)
        for path in sample_raw_files:
            shutil.copyfile(path, raw_dir / Path(path).name)
        run = subprocess.run(
            [sys.executable, "-m", "src.data.preprocess"],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        # Reload in a separate interpreter so __main__ cannot hide broken imports.
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import joblib; "
                    "p=joblib.load('models/preprocessor.joblib'); "
                    "assert len(p.get_feature_names_out()) > 0"
                ),
            ],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert check.returncode == 0, check.stdout + check.stderr
        assert (project / "data/processed/X_train.csv").is_file()