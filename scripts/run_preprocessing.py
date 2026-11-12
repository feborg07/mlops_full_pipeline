from pathlib import Path

from src.data.preprocess import (
    PreprocessingConfig,
    run_preprocessing_pipeline,
)

# Project directories
ROOT = Path(__file__).resolve().parents[1]

RAW_PATHS = [
    ROOT / "data/raw/adult.data",
    ROOT / "data/raw/adult.test",
]

# Define preprocessing configurations
CONFIGS = {
    "baseline": PreprocessingConfig(),

    "robust": PreprocessingConfig(
        numerical_scaling="robust",
        capital_transform="both",
        interaction_features=False,
    ),

    "interactions": PreprocessingConfig(
        numerical_scaling="standard",
        capital_transform="log1p",
        interaction_features=True,
    ),
}


def main():
    for name, config in CONFIGS.items():

        print(f"\nRunning configuration: {name}")

        summary = run_preprocessing_pipeline(
            paths=RAW_PATHS,
            output_dir=ROOT / "data/processed" / name,
            preprocessor_path=(
                ROOT / "models/preprocessors" / f"{name}.joblib"
            ),
            config=config,
            remove_duplicates=False,
        )

        print(f"Completed: {name}")
        print(f"Training rows: {summary['training_rows']}")
        print(f"Testing rows: {summary['testing_rows']}")
        print(f"Features: {summary['processed_features']}")


if __name__ == "__main__":
    main()