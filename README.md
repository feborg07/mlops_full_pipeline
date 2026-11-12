## Installation and Setup

This project uses **Git** for source code versioning and **DVC (Data Version Control)** for dataset versioning. The datasets are stored in a public-read Amazon S3 bucket.

The project can be cloned and its datasets downloaded **without an AWS account or AWS credentials**.

### 1. Prerequisites

Ensure the following tools are installed:

- Python 3.11
- Git
- pip
- DVC with Amazon S3 support

### 2. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/USERNAME/mlops_full_pipeline.git
cd mlops_full_pipeline
```

Replace `USERNAME` with the repository owner's GitHub username.

### 3. Create a Virtual Environment

Create an isolated Python environment:

```bash
python -m venv .venv
```

Activate the environment.

**Windows (Git Bash):**

```bash
source .venv/Scripts/activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

Install the project's Python dependencies:

```bash
pip install -r requirements.txt
```

Ensure DVC's S3 integration is available:

```bash
pip install "dvc[s3]"
```

### 5. Download the Dataset

The raw dataset is tracked by DVC and stored remotely in Amazon S3.

Download it using:

```bash
dvc pull
```

DVC automatically retrieves the correct dataset

-------

## Preprocessing Pipeline and Testing

This project includes a configurable preprocessing pipeline for the **UCI Adult Income dataset**, along with automated unit tests implemented using **pytest**.

The preprocessing module handles data loading, cleaning, validation, feature engineering, missing-value imputation, categorical encoding, numerical scaling, and dataset splitting.

Before running the pipeline on the actual dataset, it is recommended to execute the unit tests to verify that the preprocessing components behave as expected.

### 1. Run the Preprocessing Unit Tests

The preprocessing tests are located in the `tests/` directory:

```text
tests/
├── conftest.py
└── test_preprocessing.py
```

The `conftest.py` file provides reusable pytest fixtures, while `test_preprocessing.py` verifies the preprocessing functionality.

From the project root, execute:

```bash
python -m pytest tests/test_preprocessing.py -v
```

The `-v` flag enables verbose output, displaying the result of each test individually.

To run the entire project's test suite:

```bash
python -m pytest tests/ -v
```

**Expected result:** All tests should pass before proceeding. A successful test run confirms that the tested components behave as expected, but does not guarantee that the complete pipeline will run successfully on the real dataset.

### 2. Run the Preprocessing Pipeline on Actual Data

After passing the unit tests, execute the preprocessing pipeline using the raw Adult Income dataset.

First, ensure the raw data has been downloaded using DVC:

```bash
dvc pull
```

The required files are:

```text
data/raw/
├── adult.data
└── adult.test
```

The preprocessing runner is located at:

```text
scripts/run_preprocessing.py
```

It supports three configurations:

| Configuration | Description |
|---|---|
| `baseline` | Standard numerical scaling, logarithmic capital transformations, and categorical encoding |
| `robust` | Robust numerical scaling and retention of both original and logarithmic capital features |
| `interactions` | Standard preprocessing with additional feature interactions |

**Run the baseline configuration:**

```bash
python -m scripts.run_preprocessing --config baseline
```

**Run the robust configuration:**

```bash
python -m scripts.run_preprocessing --config robust
```

**Run the interactions configuration:**

```bash
python -m scripts.run_preprocessing --config interactions
```

**Run all configurations:**

```bash
python -m scripts.run_preprocessing
```

These commands assume the runner has been updated to process the `--config` argument. In the original runner implementation, the argument is ignored and all three configurations execute.

### 3. Generated Datasets and Artifacts

The pipeline automatically creates the necessary output directories and saves the processed datasets separately for each configuration.

```text
data/processed/
├── baseline/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   └── preprocessing_config.json
├── robust/
│   └── ...
└── interactions/
    └── ...

models/preprocessors/
├── baseline.joblib
├── robust.joblib
└── interactions.joblib
```

Each configuration generates:

- `X_train.csv` — Preprocessed training features.
- `X_test.csv` — Preprocessed testing features.
- `y_train.csv` — Binary training labels.
- `y_test.csv` — Binary testing labels.
- `preprocessing_config.json` — Configuration parameters used for the run.
- `<configuration>.joblib` — Serialized, fitted preprocessing transformer.

The pipeline preserves duplicate observations by default and uses a duplicate-aware splitting strategy to keep identical feature records within the same partition.

**Note:** The current implementation combines `adult.data` and `adult.test` before creating a new train/test split. It does not preserve the original UCI test partition.

### 4. Expected Pipeline Results

The following results were obtained using the Adult Income dataset with 48,842 observations:

| Configuration | Training Rows | Testing Rows | Features |
|---|---:|---:|---:|
| Baseline | 39,078 | 9,764 | 53 |
| Robust | 39,078 | 9,764 | 55 |
| Interactions | 39,078 | 9,764 | 68 |

All three configurations completed successfully.

The differences in feature counts reflect the selected feature transformations and engineering options.

### 5. Data Versioning

Raw datasets are managed using **DVC** and stored remotely in Amazon S3.

Generated preprocessing datasets are stored locally in `data/processed/`, while fitted preprocessing artifacts are stored in `models/preprocessors/`.

These generated files are not automatically uploaded to S3. They must first be registered as DVC-tracked outputs and then pushed to the configured remote.

The preprocessing configurations can subsequently be integrated into a `dvc.yaml` pipeline to automate reproducible dataset generation and dependency tracking.

### 6. Reproducibility and Data Leakage Prevention

The preprocessing workflow uses a fixed random seed (`random_state=42`) and a 20% test split.

Learned transformations, including imputation, scaling, and categorical encoding, are fitted using only the training partition and subsequently applied to the testing partition.

For model cross-validation, the unfitted preprocessor should be included inside a scikit-learn `Pipeline` alongside the classifier. This ensures preprocessing is refitted independently within each training fold, preventing validation-data leakage.