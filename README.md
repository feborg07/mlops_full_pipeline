# Sprint 17 — End-to-End MLOps Pipeline

An end-to-end machine learning operations (MLOps) project using the **UCI Adult Income dataset**.

The project demonstrates dataset versioning, exploratory data analysis, automated preprocessing, unit testing, model experimentation, experiment tracking, and reproducible machine learning workflows.

The objective is not solely to maximize predictive performance, but to develop a structured and maintainable machine learning system.

## Project Overview

The project uses the Adult Income dataset to predict whether an individual's annual income exceeds $50,000.

**Dataset characteristics:**

| Property | Value |
|---|---|
| Dataset | UCI Adult Income |
| Observations | 48,842 |
| Original features | 14 |
| Target | `income` |
| Problem type | Binary classification |
| Positive class | `>50K` |
| Random seed | 42 |

**Technologies:**

- Python 3.11
- Git and GitHub
- DVC and Amazon S3
- pandas and NumPy
- scikit-learn
- pytest
- MLflow
- joblib
- GitHub Actions (planned)
- Evidently (planned)

---

## 1. Installation and Setup

This project uses **Git** for source code versioning and **DVC (Data Version Control)** for dataset versioning.

The raw datasets are stored in an Amazon S3 bucket configured for public read access.

The repository and its raw datasets can be downloaded **without an AWS account or AWS credentials**.

### 1.1 Prerequisites

Ensure the following tools are available:

- Python 3.11
- Git
- pip
- Internet connection for dependency installation and DVC downloads

DVC with S3 support will be installed in the project environment.

### 1.2 Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/USERNAME/mlops_full_pipeline.git
cd mlops_full_pipeline
```

Replace `USERNAME` and the repository name with the actual GitHub repository URL.

All subsequent commands should be executed from the project root.

### 1.3 Create a Virtual Environment

Create an isolated Python environment:

```bash
python -m venv .venv
```

Activate the environment.

**Windows (Git Bash):**

```bash
source .venv/Scripts/activate
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 1.4 Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project's Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Ensure DVC's S3 integration is installed:

```bash
python -m pip install "dvc[s3]"
```

### 1.5 Download the Raw Dataset

The raw dataset is tracked by DVC and stored remotely in Amazon S3.

Download it using:

```bash
dvc pull
```

DVC retrieves the dataset version referenced by the repository's DVC metadata.

The required files include:

```text
data/raw/
├── adult.data
└── adult.test
```

Additional files may be present in the DVC-tracked raw data directory.

**Note:** The S3 remote permits anonymous downloads of the configured dataset objects. Uploading or modifying remote data requires appropriate AWS permissions.

---

## 2. Preprocessing Pipeline and Testing

The project includes a configurable preprocessing pipeline for the **UCI Adult Income dataset**, with automated unit tests implemented using **pytest**.

The preprocessing module handles:

- Loading and combining raw datasets
- Data cleaning and validation
- Missing-value handling
- Feature engineering
- Numerical transformations and scaling
- Categorical encoding
- Train/validation/test splitting
- Serialization of fitted preprocessing transformers

Learned preprocessing transformations are fitted using the training partition and applied to the validation and test partitions.

### 2.1 Run Preprocessing Unit Tests

The preprocessing tests are located in the `tests/` directory.

```text
tests/
├── conftest.py
└── test_preprocessing.py
```

The `conftest.py` file provides reusable pytest fixtures.

The `test_preprocessing.py` module tests preprocessing functionality and expected behavior.

Run the preprocessing tests:

```bash
python -m pytest tests/test_preprocessing.py -v
```

The `-v` flag enables verbose output.

Run the entire test suite:

```bash
python -m pytest tests/ -v
```

**Expected result:** All implemented tests should pass.

Passing unit tests verifies the tested behavior but does not guarantee that the complete pipeline will execute successfully on the full dataset.

### 2.2 Run the Preprocessing Pipeline

Before running preprocessing, download the raw data:

```bash
dvc pull
```

The preprocessing runner is located at:

```text
scripts/run_preprocessing.py
```

The pipeline supports three configurations:

| Configuration | Description |
|---|---|
| `baseline` | Standard numerical scaling, logarithmic capital transformations, and categorical encoding |
| `robust` | Robust numerical scaling and retention of original and logarithmic capital features |
| `interactions` | Standard preprocessing with additional engineered feature interactions |

**Run all configurations:**

```bash
python -m scripts.run_preprocessing
```

If the installed runner supports configuration selection, individual configurations can be executed using:

**Baseline:**

```bash
python -m scripts.run_preprocessing --config baseline
```

**Robust:**

```bash
python -m scripts.run_preprocessing --config robust
```

**Interactions:**

```bash
python -m scripts.run_preprocessing --config interactions
```

**Important:** Individual configuration commands require the runner to implement the `--config` argument. If the argument is accepted but ignored, all configurations may execute.

### 2.3 Dataset Splitting Strategy

The preprocessing workflow combines the original UCI `adult.data` and `adult.test` files before creating new dataset partitions.

The original UCI train/test separation is therefore not preserved.

The combined dataset contains 48,842 observations.

The workflow uses three partitions:

| Partition | Percentage | Purpose |
|---|---:|---|
| Training | 60% | Fit preprocessing transformations and machine learning models |
| Validation | 20% | Select models, hyperparameters, and classification thresholds |
| Testing | 20% | Evaluate selected models on held-out data |

The splitting procedure uses a fixed random seed (`random_state=42`) for reproducibility.

Duplicate observations are preserved by default because identical records may represent different individuals.

The preprocessing workflow uses a duplicate-aware splitting strategy intended to prevent identical feature records from appearing across different partitions.

### 2.4 Generated Datasets and Artifacts

The pipeline generates processed datasets separately for each configuration.

```text
data/processed/
├── baseline/
│   ├── X_train.csv
│   ├── y_train.csv
│   ├── X_val.csv
│   ├── y_val.csv
│   ├── X_test.csv
│   ├── y_test.csv
│   └── preprocessing_config.json
├── robust/
│   ├── X_train.csv
│   ├── y_train.csv
│   ├── X_val.csv
│   ├── y_val.csv
│   ├── X_test.csv
│   ├── y_test.csv
│   └── preprocessing_config.json
└── interactions/
    ├── X_train.csv
    ├── y_train.csv
    ├── X_val.csv
    ├── y_val.csv
    ├── X_test.csv
    ├── y_test.csv
    └── preprocessing_config.json

models/preprocessors/
├── baseline.joblib
├── robust.joblib
└── interactions.joblib
```

**Generated files:**

| Artifact | Description |
|---|---|
| `X_train.csv` | Transformed training features |
| `y_train.csv` | Training target labels |
| `X_val.csv` | Transformed validation features |
| `y_val.csv` | Validation target labels |
| `X_test.csv` | Transformed testing features |
| `y_test.csv` | Testing target labels |
| `preprocessing_config.json` | Preprocessing configuration metadata |
| `<variant>.joblib` | Serialized fitted preprocessing transformer |

The three configurations produce different feature representations of the same classification problem.

The generated processed datasets and fitted preprocessors are used directly by the model-training module.

### 2.5 Reproducibility and Leakage Prevention

The preprocessing workflow separates data partitioning from learned feature transformations.

The intended guarantees are:

1. Data partitions are generated before learned transformations are fitted.
2. Imputation, scaling, and encoding parameters are learned from training data only.
3. Validation and test observations are transformed using the fitted training preprocessor.
4. The model-training module consumes saved partitions rather than recreating them.
5. The held-out test set is not used to select models or classification thresholds.

For future cross-validation experiments, an unfitted preprocessing transformer should be included within each cross-validation fold to prevent validation leakage.

---

## 3. Model Training and Experiment Tracking

The project includes a machine learning training module using **scikit-learn** and **MLflow**.

The training module is located at:

```text
src/models/train.py
```

It uses the processed datasets and fitted preprocessing artifacts generated during the previous stage.

**The training module does not reload the raw Adult Income dataset, regenerate dataset partitions, or refit the saved preprocessing transformers.**

### 3.1 Machine Learning Models

Four classification algorithms are evaluated:

| Model | Description |
|---|---|
| Logistic Regression | Linear probabilistic classification model |
| Random Forest | Ensemble of decision trees using bootstrap aggregation |
| HistGradientBoosting | Histogram-based gradient-boosted decision trees |
| Extra Trees | Ensemble of highly randomized decision trees |

Each algorithm is evaluated using multiple hyperparameter configurations.

Logistic Regression includes an additional model-specific `StandardScaler` to improve numerical optimization.

This scaler is fitted exclusively using the training partition.

### 3.2 Experiment Configurations

The full training workflow evaluates:

| Model | Candidates per preprocessing variant |
|---|---:|
| Logistic Regression | 6 |
| Random Forest | 4 |
| HistGradientBoosting | 4 |
| Extra Trees | 4 |
| **Total** | **18** |

Because three preprocessing configurations are evaluated:

**18 candidates × 3 configurations = 54 training experiments**

The experiment search is intentionally limited in scope because the project's primary objective is to demonstrate an end-to-end MLOps workflow rather than exhaustive hyperparameter optimization.

### 3.3 Run a Quick Training Experiment

A quick mode is available to verify that the training workflow executes successfully.

```bash
python -m src.models.train --quick
```

Quick mode:

- Evaluates one candidate from each model family.
- Runs across all three preprocessing configurations.
- Trains 12 candidate models.
- Calculates validation metrics.
- Selects a validation winner for each configuration.
- Saves training artifacts.
- Does not evaluate the held-out test sets.

Quick mode is intended for development and pipeline verification.

### 3.4 Run the Full Training Experiment

Execute the complete experiment sweep:

```bash
python -m src.models.train
```

The full workflow:

1. Loads the previously generated training, validation, and test partitions.
2. Trains 54 candidate models.
3. Generates validation probabilities.
4. Optimizes classification thresholds using validation data.
5. Calculates validation metrics.
6. Selects the best candidate for each preprocessing configuration.
7. Evaluates the three configuration winners on their held-out test sets.
8. Saves the trained model pipelines and selected thresholds.
9. Records experiment metrics and metadata using MLflow.

### 3.5 Model Selection Strategy

Model selection is based on validation performance.

The ranking criteria are:

1. F1 score
2. Average precision (PR-AUC)
3. ROC-AUC

The F1 score is the primary selection metric because the dataset has an imbalanced target distribution.

The positive class (`>50K`) represents approximately 24% of observations.

For each candidate, the decision threshold is optimized using the validation partition.

The selected threshold is saved with the winning model's artifacts.

**Important:** The held-out test set is not used to optimize thresholds or select the preferred configuration.

Although the workflow reports test results for each configuration winner, the preferred configuration is determined by validation metrics.

### 3.6 Model Evaluation Metrics

The training workflow calculates:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of positive predictions that are correct |
| Recall | Proportion of actual positive cases identified |
| F1 score | Harmonic mean of precision and recall |
| Average precision | Summarizes precision–recall performance across thresholds |
| ROC-AUC | Measures ranking discrimination across thresholds |
| Confusion matrix | Counts true negatives, false positives, false negatives, and true positives |

The classification threshold is optimized to maximize validation F1.

Threshold-independent ranking metrics are also recorded to provide a broader evaluation of model behavior.

### 3.7 Training Results

The completed 54-candidate experiment selected **HistGradientBoosting** as the winning model family for all three preprocessing configurations.

| Configuration | Validation F1 | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|
| Baseline | **0.7337** | 0.7150 | **0.9247** |
| Robust | **0.7337** | 0.7150 | **0.9247** |
| Interactions | 0.7314 | **0.7192** | 0.9240 |

**Preferred configuration:** `baseline`

The baseline configuration achieved the highest validation F1, tied with robust preprocessing.

The interactions configuration achieved a slightly higher test F1, but this result was not used to change the validation-based model selection.

The differences between configurations are small and do not establish a statistically significant performance advantage.

### 3.8 Generated Training Artifacts

The training workflow saves its results under:

```text
models/training/
├── sweep_leaderboard.csv
├── training_summary.json
├── quick_training_summary.json
├── baseline/
│   ├── model_pipeline.joblib
│   └── decision_threshold.json
├── robust/
│   ├── model_pipeline.joblib
│   └── decision_threshold.json
└── interactions/
    ├── model_pipeline.joblib
    └── decision_threshold.json
```

**Artifacts:**

| File | Description |
|---|---|
| `sweep_leaderboard.csv` | Candidate experiment results and validation metrics |
| `training_summary.json` | Full experiment summary, winning models, and test metrics |
| `quick_training_summary.json` | Summary of the most recent quick experiment |
| `model_pipeline.joblib` | Fitted preprocessor combined with the selected fitted classifier |
| `decision_threshold.json` | Validation-optimized classification threshold |

The saved model pipelines are intended to support inference from raw feature records.

End-to-end raw-input inference should be tested before treating these artifacts as production-ready.

### 3.9 MLflow Experiment Tracking

MLflow records the machine learning experiments.

The local tracking configuration uses:

```text
mlflow.db
mlartifacts/
```

The SQLite database stores experiment metadata and metrics, while the artifact directory stores logged files.

To display the MLflow UI command:

```bash
python -m src.models.train --show-mlflow-command
```

The command prints the appropriate MLflow launch command for the configured tracking database.

Run the displayed command, then open:

```text
http://127.0.0.1:5000
```

The MLflow interface allows users to:

- Compare candidate experiments.
- Inspect model hyperparameters.
- Review validation and test metrics.
- Identify winning configurations.
- Inspect convergence diagnostics.
- Access logged model artifacts.

**Note:** Local MLflow runs are not automatically shared when cloning the GitHub repository. A reviewer can regenerate the experiment history by executing the training workflow.

---

## 4. Data and Artifact Versioning

### 4.1 Source Code Versioning

Git tracks the project's source code and configuration files.

GitHub provides the remote repository for collaboration and code review.

Typical commands include:

```bash
git status
git add .
git commit -m "feat: implement model training and MLflow tracking"
git push
```

Generated artifacts should be reviewed against `.gitignore` before staging changes.

### 4.2 Dataset Versioning with DVC

DVC manages the raw datasets stored remotely in Amazon S3.

The DVC metadata records which dataset version should be downloaded.

To retrieve the dataset:

```bash
dvc pull
```

To inspect the configured remotes:

```bash
dvc remote list
```

### 4.3 Generated Artifacts

The preprocessing and training stages generate additional files locally.

These include:

- Processed train/validation/test datasets
- Fitted preprocessing transformers
- Trained classification pipelines
- Decision threshold configurations
- Experiment leaderboards and summaries
- MLflow tracking data

These files are **not automatically uploaded to S3 or tracked by DVC** merely because they are generated.

The current workflow expects reviewers to regenerate these artifacts by running preprocessing and training.

Generated datasets and model artifacts can be added to DVC tracking later if persistent artifact versioning is required.

A future `dvc.yaml` pipeline could also formalize stage dependencies and provide reproducible execution through DVC.

---

## 5. Reproducing the Complete Workflow

After cloning the repository, a reviewer can reproduce the current implemented workflow by following these steps.

### Step 1 — Activate the Environment

```bash
source .venv/Scripts/activate
```

The example above is for Windows Git Bash. Use the activation command appropriate for the operating system.

### Step 2 — Install Dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install "dvc[s3]"
```

### Step 3 — Download Raw Data

```bash
dvc pull
```

### Step 4 — Run Unit Tests

```bash
python -m pytest tests/ -v
```

### Step 5 — Generate Processed Datasets

```bash
python -m scripts.run_preprocessing
```

This generates the train, validation, and test partitions and saves the fitted preprocessing transformers.

### Step 6 — Verify Training with Quick Mode

```bash
python -m src.models.train --quick
```

This checks that model training and validation work using the generated preprocessing artifacts.

### Step 7 — Execute the Full Model Training

```bash
python -m src.models.train
```

This evaluates the full candidate search and produces the final training summary.

### Step 8 — Inspect the Experiment Results

```bash
python -m src.models.train --show-mlflow-command
```

Run the printed MLflow UI command and inspect the experiment results.

**Reproducibility note:** The random seed and saved preprocessing configuration support reproducibility. Exact numerical results can also depend on dependency versions, numerical libraries, and execution environment.

---

## 6. Remaining MLOps Development

The following components are planned or require additional verification.

### 6.1 Model Inference Verification

Verify that serialized model pipelines can accept raw feature records and reproduce predictions obtained from the corresponding transformed datasets.

### 6.2 Automated Testing

Extend the test suite to cover:

- Saved dataset validation
- Model-training functionality
- Classification threshold selection
- Model artifact loading
- Inference consistency
- Experiment reproducibility checks

### 6.3 CI/CD with GitHub Actions

Implement automated workflows to execute relevant tests and code-quality checks when repository changes are pushed or pull requests are opened.

### 6.4 Data Drift Monitoring with Evidently

Implement data drift detection using Evidently to compare reference and current datasets.

The planned monitoring workflow will generate drift reports and apply configurable warning and failure thresholds.

### 6.5 Pipeline Automation

Evaluate integrating the preprocessing and training stages into a formal DVC pipeline with explicit dependencies and outputs.

---

## 7. Project Scope and Limitations

The project prioritizes MLOps engineering practices over exhaustive predictive modeling.

Current limitations include:

- A restricted model and hyperparameter search.
- Validation-based threshold optimization without nested cross-validation.
- Locally stored experiment tracking and generated model artifacts.
- Pending end-to-end verification of saved inference pipelines.
- CI/CD and drift monitoring not yet fully integrated.
- No deployed model-serving endpoint.

The Adult Income dataset is used as an educational benchmark. Its demographic attributes, historical context, and potential biases make it unsuitable for consequential real-world decisions without additional fairness, validity, and deployment assessments.

The intended final deliverable is a reproducible machine learning workflow integrating version control, data preparation, model training, automated testing, experiment tracking, and monitoring.