"""
Data preprocessing pipeline for the Adult Income classification project.

Responsibilities:
- Load and validate the raw dataset.
- Clean missing values and duplicate records.
- Encode the binary target
- Split data into training and test sets.
- Build a reusable scikit-learn preprocessing pipeline.
- Fit transformations using training data only.
- Save processed datasets and the fitted preprocessor.
"""

# ============================================================
# Imports
# ============================================================

# Path: Cross-platform file and directory management
from pathlib import Path
# Sequence: Type hint for ordered collections of file paths.
from collections.abc import Sequence
# joblib: Save and load fitted preprocessing objects.
import joblib
# numpy: Numerical operations and array manipulation.
import numpy as np
# Load, concatenate, clean, and manipulate datasets
import pandas as pd
# ColumnTransformer: Apply difference transformations to numerical and categorical columns.
from sklearn.compose import ColumnTransformer
# Replace missing values using statistical strategies such as median or most frequent.
from sklearn.impute import SimpleImputer
# Chain preprocessing steps into a single object
from sklearn.pipeline import Pipeline
# Convert categorical variables into numerical indicator columns
# Standardize numerical features.
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Create reproducible train/test partitions
from sklearn.model_selection import train_test_split

# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Column names defined by the Adult Income dataset
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

# Adult Income contains two target classes.
# adult.test includes a trailing period in the labels.
TARGET_MAPPING = {
    "<=50K": 0,
    ">50K": 1
}

# ============================================================
# 1. Load and concatenate datasets
# ============================================================

def load_and_concatenate_data(
        paths: Sequence[str | Path]
) -> pd.DataFrame:
    """
    Load multiple Adult Income files and concatenate them.

    Parameters
    ----------
    paths : Sequence[str | Path]
        List of file paths to load.

        Example:
        [
            "data/raw/adult.data",
            "data/raw/adult.test"
        ]

    Returns
    -------
    pd.DataFrame
        Concatenated dataset containing all valid records.

    Notes
    -----
    - Files are concatenated vertically.
    - Original column names are assigned automatically
    - The header line in adult.test is ignored (see example)
    - Missing values represented by '?' are converted to NaN.
    - Original row indices are reset.
    """

    # Check parameter
    if not paths:
        raise ValueError("The paths list cannot be empty")

    datasets = []

    for path in paths:

        # Convert the supplied path to a Path object.
        file_path = Path(path)

        # Verify file exists.
        if not file_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Read the file using the Adult Income schema.
        # comment='|' ignores the metadata header in adult.test.
        # skipinitialspace removes whitespace after delimiters.
        df = pd.read_csv(
            file_path,
            header=None,
            names=COLUMN_NAMES,
            sep=",",
            skipinitialspace=True,
            na_values=["?", ""],
            comment="|"
        )

        # Remove rows that contain no meaningful data.
        df = df.dropna(how="all")

        # Preserve the source filename for traceability
        # This column will be removed before model training.
        df["source_file"] = file_path.name

        datasets.append(df)

    # Combine datasets vertically into one DataFrame.
    combine_df = pd.concat(
        datasets,
        axis=0,
        ignore_index=True,
    )

    return combine_df

# ============================================================
# 2. Clean dataset
# ============================================================

def clean_data(
        df: pd.DataFrame,
        remove_duplicates: bool = False,
) -> pd.DataFrame:
    """
    Clean the combined Adult Income dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw concatenated dataset.

    remove_duplicates : bool, default=False
        Whether to remove exact duplicate records.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    cleaned_df = df.copy()

    # Remove leading and trailing whitespace from string columns without converting NaN strings.
    string_columns = cleaned_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in string_columns:
        cleaned_df[column] = cleaned_df[column].str.strip()

    # Replace missing-value markers with actual NaN values
    cleaned_df = cleaned_df.replace(
        ["?", "", "nan"],
        np.nan,
    )

    # Remove the trailing period from target labels.
    cleaned_df[TARGET_COLUMN] = (
        cleaned_df[TARGET_COLUMN]
        .srt.rstrip(".")
    )

    # Convert expected numerical columns to numeric types. Invalid values are converted to NaN
    for column in NUMERICAL_FEATURES:
        cleaned_df[column] = pd.to_numeric(
            cleaned_df[column],
            errors="coerce",
        )

    # Remove rows without a valid target. Target values should not be imputed
    cleaned_df = cleaned_df.dropna(
        subset=[TARGET_COLUMN]
    )

    # Optionally remove exact duplicates.
    # Exclude source_file from the duplicate comparison
    if remove_duplicates:

        comparison_columns = [
            column
            for column in COLUMN_NAMES
            if column in cleaned_df.columns
        ]

        cleaned_df = cleaned_df.drop_duplicates(
            subset=comparison_columns,
            keep="first"
        )

    return cleaned_df.reset_index(drop=True)

# ============================================================
# 3. Validate dataset
# ============================================================

def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the dataset before preprocessing.

    Raises
    ------
    ValueError
        If required columns are missing or target values are invalid
    """

    # Verify that all expected columns exist.
    missing_columns = set(COLUMN_NAMES) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Verify that target labels are valid.
    invalid_targets = set(
        df[TARGET_COLUMN].dropna().unique()
    ) - set(TARGET_MAPPING.keys())

    if invalid_targets:
        raise ValueError(
            f"Unexpected target labels: {invalid_targets}"
        )

    # Verify that the dataset is not empty
    if df.empty:
        raise ValueError(
            "The dataset contains no valid records."
        )

    # Verify that both target classes are represented.
    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "The dataset must contain both income classes"
        )

# ============================================================
# 4. Encode target variable
# ============================================================

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the income target as a binary variable.

    <=50K -> 0
    <50K  -> 1
    """

    encoded_df = df.copy()

    encoded_df[TARGET_COLUMN] = (
        encoded_df[TARGET_COLUMN]
        .map(TARGET_COLUMN)
        .astype("int64")
    )

    return encoded_df

# ============================================================
# 5. Split features and target
# ============================================================

def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate predictor variables from the target.

    Returns
    -------

    X : pd.DataFrame
        Predictor variables

    y : pd.Series
        Binary target variable
    """

    # Explicitly select the model features.
    # This prevents source_file and other metadata from entering the training pipeline
    feature_columns = (
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    )

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y

# ============================================================
# 6. Create train/test split
# ============================================================

def create_train_test_split(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series
]:

    """
    Create a reproducible stratified train/test split.

    Stratification preserves approximately the same target class proportions in both datasets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test

# ============================================================
# 7. Build preprocessing transformer
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Create preprocessing pipelines for numerical and categorical features.

    Numerical pipeline:
        1. Impute missing values using the median.
        2. Standardize features

    Categorical pipeline:
        1. Impute missing values using the most frequent category.
        2. Apply one-hot encoding
    
    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing transformer
    """

    # Numerical preprocessing pipeline
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            )
        ]
    )

    # Categorical preprocessing pipeline
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )
            )
        ]
    )

    # Apply the appropriate pipeline to each feature group.
    preprocessor = ColumnTransformer(
        transformer=[
           ( 
               "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor

# ============================================================
# 8. Transform training and testing datasets
# ============================================================

def transform_datasets(
        preprocessor: ColumnTransformer,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit the preprocessor on training data and transform both training and testing datasets.

    Important
    ---------
    The preprocessor is fitted only on training data to prevent data leakage.
    """
    # Learn imputation values, scaling statistics, and categorical mappings from training data.
    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    # Apply the learned transformations to testing data.
    X_test_transformed = preprocessor.transform(
        X_test
    )

    # Retrieve transformed feature names.
    features_names = preprocessor.get_feature_names_out()

    # Convert the transformed arrays into DataFrames
    X_train_processed = pd.DataFrame(
        X_train_transformed,
        columns=features_names,
        index=X_train.index,
    )

    X_test_processed = pd.DataFrame(
        X_test_transformed,
        columns=features_names,
        index=X_test.index,
    )

    return X_train_processed, X_test_processed

def save_processed_data(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str | Path,
) -> None:
    """
    Save processed training and testing datasets as CSV.
    """

    output_path = Path(output_dir)

    # Create the output directory if it does not exist.
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save feature matrices.
    X_train.to_csv(
        output_path / "X_train.csv",
        index=False
    )

    X_train.to_frame(name=TARGET_COLUMN).to_csv(
        output_path / "y_train.csv",
        index=False,
    )

# ============================================================
# 10. Save fitted preprocessing transformer
# ============================================================

def save_preprocessor(
        preprocessor: ColumnTransformer,
        output_path: str | Path,
) -> None:
    """
    Save the fitted preprocessing transformer.

    The saved object contains:
    - Numerical imputation statistics.
    - Feature scaling parameters.
    - Categorical imputation values.
    - One-hot encoding mappings.
    """

    file_path = Path(output_path)

    # Ensure the destination directory exists.
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Serialize the fitted preprocessing object.
    joblib.dump(
        preprocessor,
        file_path,
    )

# ============================================================
# 11. Execute complete preprocessing pipeline
# ============================================================

def run_preprocessing_pipeline(
        paths: Sequence[str | Path],
        output_dir: str | Path,
        preprocessor_path: str | Path,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        remove_duplicates: bool = False
) -> dict:
    """
    Execute the complete preprocessing workflow.

    Parameters
    ----------
    paths: Sequence[str | Path]
        List of raw dataset file paths.

    output_dir: str | Path
        Directory for processed datasets.
    
    preprocessor_path : str | Path
        Destination for the fitted transformer.
    
    test_size : float
        Fraction of data reserved for testing
    
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Summary of preprocessing results
    """

    # Step 1: Load and concatenate raw files.
    df = load_and_concatenate_data(paths)

    raw_rows = len(df)

    # Step 2: Clean the combined dataset.
    df = clean_data(
        df,
        remove_duplicates=remove_duplicates,
    )

    # Step 3: Validate the cleaned dataset
    validate_data(df)

    # Step 4: Encode the binary target
    df = encode_target(df)

    # Step 5: Separate features and target.
    X, y = split_features_target(df)

    # Step 6: Create stratified train/test partitions
    X_train, X_test, y_train, y_test = (
        create_train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    # Step 7: Create the preprocessing transformer
    preprocessor = build_preprocessor()

    # Step 8: Fit on training data and transform both sets.
    X_train_processed, X_test_processed = (
        transform_datasets(
            preprocessor,
            X_train,
            y_test,
        )
    )

    # Step 9: Save processed datasets.
    save_processed_data(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        output_dir,
    )

    # Step 10: Save the fitted preprocessing transformer.
    save_preprocessor(
        preprocessor,
        preprocessor_path,
    )

    # Step 11: Return a summary for logging and validation
    summary = {
        "raw_rows": raw_rows,
        "cleaned_rows": len(df),
        "training_rows": len(X_train_processed),
        "testing_rows": len(X_test_processed),
        "original_features": X.shape[1],
        "processed_feature": X_train_processed.shape[1],
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "random_state": random_state,
        "test_size": test_size,
    }

    return summary

# ============================================================
# 12. Main execution
# ============================================================

def main() -> None:
    """
    Run preprocessing using the project'directory structure
    """

    # Resolve project root relative to this script:
    # project_root/src/data/preprocess.py
    project_root = Path(__file__).resolve().parents[2]

    # Define the raw input files.
    raw_paths = [
        project_root / "data/raw/adult.data",
        project_root / "data/raw/adult.test",
    ]

    # Define output location
    processed_dir = project_root / "data/processed"

    preprocessor_path = (
        project_root / "models/preprocessor.joblib"
    )

    # Execute the complete preprocessing workflow
    summary = run_preprocessing_pipeline(
        paths=raw_paths,
        output_dir=processed_dir,
        preprocessor_path=preprocessor_path,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        remove_duplicates=False
    )

    # Display a concise execution summary
    print("\nPreprocessing completed successfully.\n")

    for key, value in summary.items():
        print(f"{key}: {value}")

# Execute only when file is run directly.
# Importing its functions from another module will not
# automatically execute the preprocessor pipeline
if __name__ == "__main__":
    main()