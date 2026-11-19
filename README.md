# Sprint 17 — End-to-End MLOps Pipeline

**Reproducible machine learning workflow for the UCI Adult Income classification dataset.** This educational project focuses on the *lifecycle* of a model—not merely fitting a classifier—including versioned data, reproducible preprocessing, experiment tracking, isolated training executions, explicit model promotion, drift detection, robustness evaluation, and automated tests.

> **Project status (November 18–19, 2026):** The Sprint 17 rubric components are implemented: Git/DVC versioning, YAML-driven training, MLflow model logging and programmatic experiment comparison, real-data and model validation tests, two-job GitHub Actions CI, Evidently monitoring, and a written drift analysis. The latest **local full orchestration** passed **476 tests (11 third-party warnings)**, evaluated **54 candidates**, compared precisely those 54 MLflow runs, and completed drift and robustness evaluation. The updated automatic **GitHub Actions CI passed**, as reported by the repository owner. Earlier manual GitHub Actions quick/full orchestration runs succeeded, but the newest execution-ID-filtered orchestrator has been verified **locally**, not yet confirmed in a fresh remote full run. Reference-model promotion remains manual; there is no production prediction API or automatic deployment.

## Contents
- [Quick Start — Run the Complete Pipeline](#quick-start--run-the-complete-pipeline)
1. [Project at a glance](#1-project-at-a-glance)
2. [Architecture and responsibilities](#2-architecture-and-responsibilities)
3. [Installation and reviewer quick start](#3-installation-and-reviewer-quick-start)
4. [Dataset and exploratory analysis](#4-dataset-and-exploratory-analysis)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Training and MLflow](#6-training-and-mlflow)
7. [Candidate validation and explicit promotion](#7-candidate-validation-and-explicit-promotion)
8. [DVC and artifact versioning](#8-dvc-and-artifact-versioning)
9. [Data drift monitoring](#9-data-drift-monitoring)
10. [Model robustness under drift](#10-model-robustness-under-drift)
11. [Testing, CI, and pipeline orchestration](#11-testing-ci-and-pipeline-orchestration)
12. [Reproducing the full workflow](#12-reproducing-the-full-workflow)
13. [Command reference](#13-command-reference)
14. [Scope, limitations, and next steps](#14-scope-limitations-and-next-steps)

## Quick Start — Run the Complete Pipeline

After cloning the repository and installing the dependencies ([installation instructions](#3-installation-and-reviewer-quick-start)), execute the complete MLOps workflow from the repository root:

**Quick mode — 12 model candidates under the current default YAML (recommended for verification):**

```bash
python -m scripts.run_pipeline --quick --pull-dvc
```

**Full mode — 54 model candidates under the current default YAML:**

```bash
python -m scripts.run_pipeline --full --pull-dvc
```

Both commands execute automated tests (including actual-dataset and model-validation tests), preprocessing, model training with MLflow tracking, synthetic drift generation, Evidently monitoring, and reference-model robustness evaluation. **Quick mode** trains a representative configuration per family as a smoke check; it does not evaluate the held-out test set or package a promotable candidate. **Full mode** additionally compares only the current execution's completed MLflow candidates with `scripts/compare_experiments.py` and exports `reports/mlflow_comparison.csv`. Training reads **`configs/config.yml`** by default; changing its grids can change candidate counts. The `--pull-dvc` option restores both the raw dataset and the promoted reference model.

**Important:** Model promotion is intentionally excluded from automated orchestration and requires explicit authorization.

For detailed execution instructions, GitHub Actions orchestration, and model promotion, see [Section 12 — Reproducing the Full Workflow](#12-reproducing-the-full-workflow).

## 1. Project at a glance

| Property | Value |
|---|---|
| Dataset | [UCI Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult) |
| Task | Binary classification: annual income `>50K` versus `<=50K` |
| Observations | 48,842 combined from `adult.data` and `adult.test` |
| Predictors | 14 original input columns; `income` is the target |
| Positive class | `>50K` (encoded as `1` for evaluation) |
| Approximate class distribution | 76.1% `<=50K`; 23.9% `>50K` |
| Seed | 42 |
| Python | 3.11 (developed with 3.11.4) |
| Primary selection metric | Validation F1 for the positive class (`selection_metric: val_f1`) |
| Training configuration | [`configs/config.yml`](configs/config.yml) — YAML-defined algorithms, grids, seed, paths, and MLflow experiment |
| Search method | Cartesian-product hyperparameter enumeration (not `GridSearchCV` or Optuna) |
| Selected preprocessing variant | `baseline` |
| Selected model family | `HistGradientBoostingClassifier` |
| Selected validation F1 | 0.7337 |
| Selected held-out test F1 | 0.7150 |
| Latest local test run | **476 passed, 11 third-party warnings** (November 18, 2026, local time) |
| Automatic GitHub Actions CI | **Passed**, per latest owner report; two dependent jobs with minimum validation F1 = 0.65 |
| Manual GitHub Actions orchestration | Earlier quick/full runs successful; latest orchestrator changes still require fresh remote verification |
| Verified full-run candidates | **54 / 54** in the latest locally completed full orchestration |
| Latest local full orchestration | `21ed6abc583e4bee87433acd8f9517da` — training and MLflow comparison both selected `baseline` |

**Core tools:** Python, pandas, NumPy, scikit-learn, PyYAML, pytest, Git/GitHub, DVC with an Amazon S3 remote, MLflow (SQLite tracking), joblib, Evidently, and GitHub Actions.

### Implementation status

| Capability | Status |
|---|---|
| Git source control and DVC-backed raw data | Implemented |
| Public-read S3 downloads for DVC artifacts | Implemented |
| EDA and preprocessing variants | Implemented |
| External YAML training configuration and grid enumeration | Implemented and tested |
| Quick-mode smoke checks (no test-set evaluation or candidate packaging) | Implemented and tested |
| Real-DVC-data validation and deterministic model-validation tests | Implemented (5 + 3 tests) |
| Automated preprocessing, training, monitoring, and robustness tests | Implemented; 476-test local suite passing |
| MLflow experiment tracking and candidate isolation | Implemented |
| Programmatic MLflow comparison, deterministic ties, execution filtering | Implemented; 54/54 compared in latest local full run |
| Explicit, validated model promotion | Implemented |
| DVC-versioned promoted reference model and metadata | Implemented |
| Synthetic drift scenarios and Evidently monitoring | Implemented |
| Robustness evaluation under drift | Implemented |
| GitHub Actions **CI** (`ci.yml`: test → train; F1 gate ≥ 0.65) | Implemented; recent CI success reported |
| Shared Python orchestration (`scripts/run_pipeline.py`) | Latest full mode verified locally, including MLflow comparison |
| Manual GitHub Actions orchestration | Earlier quick/full runs successful; YAML comparison step needs synchronization with orchestrator |
| Full-mode candidate sweep | Latest local orchestration completed 54/54, selecting `baseline` |
| GitHub Actions evaluation artifacts | Training, drift, and robustness reports inspected |
| Production model-serving endpoint / automatic deployment | Out of scope |

## 2. Architecture and responsibilities

The project separates **source control**, **artifact versioning**, **training configuration**, **experiment tracking**, and **workflow execution**. These systems solve different problems and are not interchangeable.

```text
Git / GitHub             DVC + public-read S3             MLflow (SQLite)
  code, tests, YAML        raw dataset, reference model    params, metrics, models
           |                          |                           ^
           +--------------------------+---------------------------+
                                      |
                        scripts/run_pipeline.py
                        (one shared orchestrator)
                                      |
                      optional DVC pull -> pytest
                       including real-data tests
                                      |
                      preprocessing (three variants)
                                      |
        configs/config.yml -> src/models/train.py -> MLflow candidates
                                 quick / full
                                      |
          FULL ONLY: scripts/compare_experiments.py
             --execution-id <CURRENT_EXECUTION_ID>
                  -> reports/mlflow_comparison.csv
                                      |
                        synthetic drift generation
                                      |
                        Evidently drift monitoring
                                      |
                     reference-model robustness
                                      |
                       generated reports / artifacts

Automatic GitHub Actions CI: tests (with DVC data) -> quick train -> F1 gate
Manual GitHub Actions orchestration: DVC pull -> scripts/run_pipeline.py
Model promotion: separate, explicitly authorized; never automatic
```

**Division of responsibility:**

- **Git:** versions code, documentation, `configs/config.yml`, tests, workflow definitions, and `.dvc` pointer files.
- **DVC:** versions selected large/binary artifacts and retrieves their exact contents from S3. In this project, DVC is **not** the workflow orchestrator.
- **`configs/config.yml` (PyYAML):** is the default **model-training** configuration. It defines the four model families' fixed parameters and Cartesian-product grids, the selected preprocessing variants, seed, supported selection metric, training output paths, and MLflow experiment settings. It is not a replacement for preprocessing configuration.
- **`PreprocessingConfig`:** remains in `src/data/preprocess.py`; it configures data cleaning, engineering, and partitioning independently of training YAML.
- **MLflow:** stores candidate models, hyperparameters, validation metrics, configuration/data fingerprints, and per-candidate execution-ID tags. Local SQLite history is not automatically shared with GitHub clones.
- **`scripts/compare_experiments.py`:** queries runs with `mlflow.search_runs()`, excludes quick smoke checks by default, applies deterministic validation-based ranking, supports `--execution-id`, and optionally exports a CSV.
- **Python modules:** implement preprocessing, model training, drift detection, robustness evaluation, and promotion.
- **`scripts/run_pipeline.py`:** is the shared orchestrator for tests, preprocessing, training, **full-mode-only execution-scoped MLflow comparison**, drift generation, monitoring, and robustness evaluation. It uses the committed `configs/config.yml` and does not expose a custom `--config` forwarding flag.
- **GitHub Actions CI (`ci.yml`):** automatically runs on push/PR to `main`: a `test` job downloads the DVC raw data and runs pytest; the dependent `train` job performs quick training and checks validation F1 against a minimum threshold.
- **Manual GitHub Actions orchestration:** restores raw and reference DVC artifacts, invokes `scripts/run_pipeline.py`, and uploads reports; no mode promotes models. The last provided workflow YAML still has a standalone full-mode comparison step, now redundant with the orchestrator; remove it to keep a single source of execution logic.

### Main project paths

```text
.github/workflows/ci.yml                    Continuous integration
.github/workflows/ml_pipeline_orchestration.yml  Manual pipeline execution
configs/config.yml                          Version-controlled training settings
scripts/run_pipeline.py                     Local / GitHub Actions orchestrator
scripts/compare_experiments.py              MLflow search_runs comparison and CSV export
scripts/run_preprocessing.py                Preprocessing entry point
src/data/preprocess.py                      Loading, splitting, transformations
src/data/drift_generator.py                Synthetic drift scenario generator
src/data/monitoring.py                     Evidently drift detection
src/models/train.py                        YAML-driven model sweep, validation, MLflow
src/models/promote.py                      Explicit candidate validation/promotion
src/models/robustness.py                   Evaluation under synthetic drift
tests/test_data_validation.py              Five actual DVC-data checks
tests/test_model_validation.py             Three lightweight model checks
tests/test_compare_experiments.py          MLflow leaderboard and filtering tests
tests/                                     Remaining pytest test suite
data/raw/                                  DVC-retrieved Adult data
data/processed/<variant>/                 Generated train/validation/test partitions
data/drift/<scenario>/                    Generated control/mild/moderate/severe data
models/preprocessors/                     Fitted preprocessing artifacts
models/runs/<execution_id>/               Isolated candidate experiment outputs
models/reference/                         DVC-tracked promoted model and metadata
models/reference.dvc                      Git-tracked pointer for reference directory
reports/drift/                            Generated Evidently reports
reports/robustness/                       Generated robustness reports
reports/mlflow_comparison.csv              Full-mode, execution-scoped leaderboard
mlflow.db                                 Local MLflow SQLite tracking store
mlartifacts/                              Local MLflow artifacts
```

Generated paths may not exist in a fresh clone until their corresponding stage runs. The YAML settings are committed to Git; they are not a DVC data artifact.

## 3. Installation and reviewer quick start

### 3.1 Prerequisites

- Python 3.11, Git, pip, and internet access.
- Access to the project's GitHub repository.
- No AWS credentials are required **for the configured public-read DVC downloads**. Writing to the S3 remote requires separate permissions.

### 3.2 Clone the repository

```bash
git clone https://github.com/feborg07/mlops_full_pipeline mlops-full-pipeline
cd mlops-full-pipeline
```

All commands below assume the terminal is in the repository root.

### 3.3 Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows Git Bash:

```bash
source .venv/Scripts/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3.4 Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "dvc[s3]"
```

The last command enables DVC's S3 support. The CI workflow explicitly installs `dvc[s3]==3.67.1`. **PyYAML** is required by the trainer, comparison script, and orchestrator; the MLflow model-logging path also requests installed versions of `scikit-learn`, `numpy`, `scipy`, `pandas`, `cloudpickle`, and `skops` for model dependency metadata. All runtime dependencies should be declared with pinned versions in `requirements.txt`. A successful CI installation is stronger evidence than the developer machine's preinstalled packages; check this list if changing dependency versions.

### 3.5 Retrieve DVC-managed artifacts

```bash
dvc pull data/raw.dvc models/reference.dvc
```

Explicit targets are used because the earlier GitHub Actions run with a bare `dvc pull` restored raw data but did not restore `models/reference/`. The underlying reason for that behavior has not yet been established. The command retrieves the raw Adult files and the promoted reference model for the checked-out Git revision, provided the S3 objects remain available.

Expected raw inputs include:

```text
data/raw/adult.data
data/raw/adult.test
```

The reference directory contains:

```text
models/reference/best_model_pipeline.joblib
models/reference/best_model_metadata.json
```

### 3.6 Fast verification

```bash
python -m pytest tests/ -q
dvc status
```

The latest complete local run passed **476 tests with 11 nonfatal dependency deprecation warnings** during full orchestration on November 18, 2026 (local time). **Run `dvc pull data/raw.dvc` before pytest:** `tests/test_data_validation.py` intentionally loads the actual Adult files and fails if they are absent. GitHub Actions CI retrieves that dataset in the `test` job. These results are evidence of the checked-out implementation, not a guarantee across other revisions or environments.

> **Security:** `joblib`/pickle model files can execute code when loaded. Only load model artifacts from a trusted repository and storage source. Public readability does not imply that arbitrary external artifacts are safe.

## 4. Dataset and Exploratory Analysis

The project includes a comprehensive Exploratory Data Analysis (EDA) covering data quality, feature distributions, statistical associations, predictor relationships, feature redundancy, and potential feature interactions.

**Complete analysis:** [Exploratory Data Analysis Notebook](notebooks/01_eda.ipynb)

The notebook contains the Python code, statistical tests, visualizations, interpretations, and preprocessing recommendations supporting the findings summarized below.

### 4.1 Dataset Source and Structure

The project uses the [UCI Adult (Census Income) dataset](https://archive.ics.uci.edu/dataset/2/adult), combining:

| Source | Observations |
|---|---:|
| `adult.data` | 32,561 |
| `adult.test` | 16,281 |
| **Combined dataset** | **48,842** |

The dataset contains **14 predictors** (6 numerical and 8 categorical) and one binary target, `income`.

The first line of `adult.test` is skipped, trailing periods are removed from its income labels, and `?` values are interpreted as missing.

The original UCI train/test division is not reused. The project creates its own train, validation, and held-out test partitions.

The classification target is:

- `<=50K`: Annual income of $50,000 or less.
- `>50K`: Annual income exceeding $50,000 (positive class).

### 4.2 Data Quality Assessment

#### Duplicate records

The EDA identified:

- **52 duplicate occurrences** (0.11% of the dataset).
- **101 observations** belonging to 49 distinct duplicate groups.
- Most duplicate groups contain two observations; three groups contain three.

Because the dataset does not provide unique individual identifiers, identical records cannot automatically be classified as erroneous duplicates.

**Decision:** Retain all observations. The implemented preprocessing pipeline uses duplicate-aware partitioning logic to reduce the risk of identical observations appearing across different partitions.

#### Missing values

| Feature | Missing count | Missing percentage |
|---|---:|---:|
| `workclass` | 2,799 | 5.73% |
| `occupation` | 2,809 | 5.75% |
| `native-country` | 857 | 1.75% |

Overall, **3,620 observations (7.41%)** contain at least one missing value.

Further investigation identified two important patterns:

1. All 2,799 observations with missing `workclass` also have missing `occupation`.
2. Approximately 9.5% of observations with missing employment information belong to the high-income class, compared with approximately 24.8% among observations with recorded employment information.

These findings suggest that missing employment information may contain predictive information.

**Decision:** Preserve missing observations and represent missing categorical values using a dedicated `Unknown` category rather than deleting rows or replacing missing values with the most frequent category.

### 4.3 Feature Distributions and Class Imbalance

#### Numerical distributions

The six numerical predictors exhibit different distributional characteristics:

- `age`: Moderately right-skewed, with a mean of 38.64 years.
- `fnlwgt`: Right-skewed, reflecting substantial variation in census sampling weights.
- `education-num`: Ordinal educational attainment, with a median of 10.
- `hours-per-week`: Concentrated around 40 hours, with a maximum of 99.
- `capital-gain`: Extremely right-skewed and predominantly zero.
- `capital-loss`: Strongly right-skewed and predominantly zero.

The financial features require special treatment because conventional scaling alone does not address their extreme zero inflation and skewness.

#### Categorical distributions

Categorical predictors exhibit substantial frequency imbalances and varying cardinality.

`native-country` has the highest cardinality, with **41 categories**, heavily dominated by `United-States`.

Other features, including `occupation` and `workclass`, contain rare categories that may produce sparse or unstable encoded representations.

These findings support categorical encoding with appropriate handling of infrequent and previously unseen categories.

#### Target distribution

| Income class | Observations | Percentage |
|---|---:|---:|
| `<=50K` | 37,155 | 76.1% |
| `>50K` | 11,687 | 23.9% |

The majority-to-minority ratio is approximately **3.18:1**.

A classifier that always predicts `<=50K` would achieve approximately 76.1% accuracy while failing to identify any positive-class observations.

**Modeling implications:**

- Use stratified partitioning to preserve class proportions.
- Prioritize positive-class F1 for model selection.
- Report precision, recall, ROC-AUC, and PR-AUC as complementary evaluation metrics.
- Avoid relying on accuracy alone.

### 4.4 Statistical Associations with Income

The EDA used statistical tests and effect-size measures to quantify relationships between individual predictors and the binary income target.

These are observational associations, not causal effects or direct measures of model feature importance.

#### Numerical predictors

Point-biserial correlation and Mann–Whitney U tests were applied to the six numerical predictors.

| Feature | Point-biserial r | Rank-biserial |
|---|---:|---:|
| `education-num` | 0.3326 | 0.4325 |
| `age` | 0.2304 | 0.3645 |
| `hours-per-week` | 0.2277 | 0.3437 |
| `capital-gain` | 0.2230 | 0.1794 |
| `capital-loss` | 0.1476 | 0.0684 |
| `fnlwgt` | -0.0063 | -0.0080 |

Five numerical predictors showed statistically significant associations with income (p < 0.0001). `fnlwgt` showed negligible marginal association and was not statistically significant.

**Key interpretation:** Educational attainment exhibits the strongest numerical association with income, followed by age and working hours. Financial variables also contain useful statistical signals despite their highly skewed distributions.

A weak marginal association does not necessarily imply that a feature lacks multivariate predictive value.

#### Categorical predictors

Chi-square tests of independence and Cramér's V were used to evaluate categorical associations.

| Feature | Cramér's V |
|---|---:|
| `relationship` | 0.4545 |
| `marital-status` | 0.4483 |
| `education` | 0.3659 |
| `occupation` | 0.3500 |
| `sex` | 0.2146 |
| `workclass` | 0.1816 |
| `race` | 0.0999 |
| `native-country` | 0.0962 |

All eight categorical predictors showed statistically significant associations with income (p < 0.05), although their effect sizes varied.

Household relationships, marital status, education, and occupation exhibited the strongest categorical associations.

**Important distinction:** Statistical significance does not establish predictive usefulness, independence, or causality. Final feature and model decisions must be supported by validation performance.

### 4.5 Predictor Relationships, Redundancy, and Interactions

#### Numerical correlations

Pearson and Spearman correlation matrices showed weak pairwise correlations among numerical predictors.

- Maximum absolute Pearson correlation: **0.144**.
- Maximum absolute Spearman correlation: **0.164**.

The strongest numerical relationship was between `education-num` and `hours-per-week`, but its magnitude remained weak.

No numerical feature was excluded solely because of pairwise correlation.

#### Categorical relationships

Several categorical predictors exhibited moderate to strong associations with one another.

| Feature pair | Cramér's V |
|---|---:|
| `relationship` × `sex` | 0.65 |
| `marital-status` × `relationship` | 0.49 |
| `marital-status` × `sex` | 0.46 |
| `occupation` × `sex` | 0.42 |
| `occupation` × `workclass` | 0.40 |
| `race` × `native-country` | 0.40 |

These relationships indicate overlapping information, particularly among household and employment characteristics.

However, statistical association does not imply complete redundancy, and the features were retained for initial modeling.

#### Education redundancy

The notebook confirmed a **one-to-one mapping between all 16 `education` categories and their corresponding `education-num` values**.

Both variables therefore encode the same educational attainment information.

**Implemented decision:** Retain `education-num` and exclude the textual `education` representation from the preprocessing feature set.

#### Feature interaction investigation

Three combinations were explored:

| Interaction | Main observation | Small subgroups (n < 30) |
|---|---|---:|
| Education × Age | Higher education is associated with higher income rates, particularly at ages 31–60 | 4 / 64 |
| Occupation × Working Hours | Income rates vary across occupations and working-hour groups | 6 / 60 |
| Marital Status × Education | Income rates differ across combinations of marital status and educational attainment | 34 / 104 |

The subgroup differences motivated investigation of engineered interaction features.

However, these descriptive patterns do not establish statistically significant interactions or guarantee improved predictive performance.

**Implemented decision:** Evaluate interaction features through a separate `interactions` preprocessing variant and compare performance against the baseline using validation data.

### 4.6 Capital Gain and Loss Investigation

The financial variables received additional analysis because of their extreme skewness and unusually high proportions of zero values.

#### Zero inflation

| Statistic | `capital-gain` | `capital-loss` |
|---|---:|---:|
| Zero observations | 44,807 | 46,560 |
| Zero percentage | 91.74% | 95.33% |
| Positive observations | 4,035 | 2,282 |
| Unique positive values | 122 | 98 |

The zero values represent reported absence of capital gains or losses and should not be treated as missing data.

#### Association with income

| Financial condition | High-income rate |
|---|---:|
| Positive capital gain | 61.73% |
| Zero capital gain | 20.52% |
| Positive capital loss | 50.13% |
| Zero capital loss | 22.64% |

Positive capital gains and losses are associated with substantially higher proportions of high-income observations.

The notebook also established that **no observation contains both a positive capital gain and a positive capital loss**.

#### Maximum capital gain

The maximum recorded capital gain is **$99,999**, appearing in 244 observations (0.50% of the dataset).

All 244 observations belong to the `>50K` income class.

This concentration may reflect top-coding or a special reporting convention, but its cause was not independently verified. The value was not automatically classified as erroneous or removed.

**Implemented feature engineering:**

- Preserve the zero/nonzero distinction using binary indicators.
- Apply `log1p` transformations to reduce the scale of positive financial values.
- Retain the original capital variables alongside their transformations in the `robust` variant.
- Compare alternative representations through validation rather than selecting transformations solely from exploratory associations.

### 4.7 EDA Conclusions and Preprocessing Decisions

The exploratory analysis directly informed the project's preprocessing and evaluation design.

| EDA finding | Implemented decision |
|---|---|
| Missing employment information is associated with income | Preserve rows and impute missing categorical values as `Unknown` |
| Identical records may represent different individuals | Retain duplicates and use duplicate-aware partitioning |
| Target distribution is imbalanced (3.18:1) | Stratified partitions and positive-class F1 selection |
| Financial variables are highly zero-inflated | `log1p` transformations and binary positive-value indicators |
| Some categorical predictors have rare levels | One-hot encoding with infrequent/unseen category handling |
| `education` and `education-num` are redundant | Retain `education-num` and exclude textual `education` |
| Numerical predictors have weak pairwise correlations | Avoid correlation-based feature removal |
| Subgroup analysis suggests potential interactions | Evaluate a separate `interactions` variant |
| Numerical distributions contain skewness and extreme values | Compare standard and robust scaling strategies |

The original notebook proposed several alternative preprocessing and evaluation approaches. The final implementation uses three explicit preprocessing variants (`baseline`, `robust`, and `interactions`) and dedicated training, validation, and held-out test partitions.

All learned preprocessing parameters are fitted on training data only. Validation data is used for candidate selection and decision-threshold optimization, while held-out test data is reserved for evaluating selected configurations.

**Overall conclusion:** The EDA identified meaningful associations between income and education, age, employment characteristics, household relationships, and financial variables. It also revealed missing-data patterns, class imbalance, feature redundancy, rare categories, and opportunities for feature engineering.

These findings informed the implemented pipeline, but predictive performance and feature usefulness are ultimately determined through model validation rather than exploratory statistics alone.

The dataset contains sensitive demographic attributes and historical socioeconomic patterns. Statistical associations are not causal conclusions, and the model is not suitable for consequential individual decisions without additional fairness, validity, and governance assessments.

## 5. Data Preprocessing

The preprocessing pipeline converts the raw Adult Income dataset into reproducible, model-ready training, validation, and test datasets.

Its design follows the findings documented in [Section 4 — Dataset and Exploratory Analysis](#4-dataset-and-exploratory-analysis), with particular attention to missing values, duplicate preservation, class imbalance, skewed financial variables, categorical encoding, and feature interactions.

The implementation uses **scikit-learn Pipelines and ColumnTransformers**, configurable preprocessing strategies, duplicate-aware data partitioning, and serialized fitted transformers.

### 5.1 Execution and Configuration

The primary entry point is `scripts/run_preprocessing.py`.

```bash
# Generate all three preprocessing variants
python -m scripts.run_preprocessing

# Generate a specific variant
python -m scripts.run_preprocessing --config baseline
python -m scripts.run_preprocessing --config robust
python -m scripts.run_preprocessing --config interactions
```

The preprocessing runner:

1. Loads and combines the raw Adult Income files.
2. Validates the dataset structure and target labels.
3. Creates reproducible train/validation/test partitions.
4. Applies the selected feature-engineering configuration.
5. Fits preprocessing transformations using training data.
6. Transforms validation and test data using the fitted pipeline.
7. Saves processed datasets, fitted transformers, and configuration metadata.

Preprocessing behavior is controlled through the `PreprocessingConfig` dataclass in `src/data/preprocess.py`.

Configuration options include imputation strategies, numerical scaling, rare-category handling, financial transformations, education representation, interaction features, duplicate-aware splitting, partition proportions, and random seed.

The standard preprocessing configuration uses `random_state=42`. **This Python dataclass is distinct from `configs/config.yml`, which controls the downstream model-training search.** Editing the YAML training seed or model grid does not change the already saved preprocessing partitions; rerun preprocessing if its configuration or input data changes.

### 5.2 Data Loading, Cleaning, and Validation

The pipeline combines the original UCI Adult Income files:

| File | Observations |
|---|---:|
| `adult.data` | 32,561 |
| `adult.test` | 16,281 |
| **Total** | **48,842** |

The loading and cleaning process includes:

- Reading both files using the expected Adult Income column schema.
- Skipping the introductory line in `adult.test`.
- Normalizing income labels by removing trailing periods.
- Interpreting `?` as missing values.
- Validating expected columns and target values.
- Encoding the binary target as `0` for `<=50K` and `1` for `>50K`.
- Preserving duplicate observations rather than deleting them.

The project does not reuse the original UCI train/test division. Instead, both files are combined before generating project-specific partitions.

### 5.3 Duplicate-Aware Train/Validation/Test Splitting

A central design requirement is to preserve legitimate duplicate observations while reducing the risk of identical predictor records appearing in different partitions.

A conventional random split may place identical observations in both training and evaluation data, potentially producing overly optimistic performance estimates.

The implemented splitting algorithm addresses this through **group-aware partitioning**.

#### Splitting algorithm

1. Identify groups of observations with identical complete raw predictor vectors.
2. Assign each group an identifier without deleting any observations.
3. Calculate group sizes and positive-class proportions.
4. Assign each group a stratification label based on its positive-class proportion.
5. Split groups into development and test partitions.
6. Split development groups into training and validation partitions.
7. Expand the selected groups back into individual observations.

Groups are constructed from predictor values, not target labels. Consequently, identical predictors remain together even when their corresponding target labels differ.

The process uses two sequential splits:

- Approximately 20% of the combined dataset is reserved for testing.
- Approximately 20% of the remaining development dataset is reserved for validation.

This produces approximate proportions of **64% training, 16% validation, and 20% testing**.

#### Recorded partition sizes

| Partition | Observations | Approximate share |
|---|---:|---:|
| Training | 31,263 | 64.0% |
| Validation | 7,815 | 16.0% |
| Test | 9,764 | 20.0% |
| **Total** | **48,842** | **100%** |

The same partition assignments are used across all three preprocessing variants, allowing meaningful comparisons between configurations.

**Stratification limitation:** Because splitting operates on groups rather than individual observations, exact partition sizes and class proportions are not guaranteed. Group-level stratification approximates the original class distribution while prioritizing duplicate separation.

#### Partition responsibilities

| Partition | Purpose |
|---|---|
| Training | Fit imputers, encoders, scalers, and classifiers |
| Validation | Compare candidates and optimize decision thresholds |
| Test | Evaluate selected configurations without using test results for selection |

No validation or test observations are used to fit learned preprocessing parameters.

### 5.4 Feature Engineering and Transformations

The pipeline applies transformations motivated by the exploratory analysis.

#### Missing-value treatment

Categorical missing values are replaced with the dedicated category `Unknown`.

Numerical missing values are handled using median imputation.

This preserves observations and allows missing categorical information to remain distinguishable from recorded categories.

#### Numerical scaling

Two scaling strategies are evaluated:

- **StandardScaler:** Centers features around their training mean and scales by their training standard deviation.
- **RobustScaler:** Centers features using the training median and scales according to the interquartile range, reducing sensitivity to extreme observations.

The choice of scaler depends on the preprocessing variant.

#### Capital gain and loss transformations

The EDA identified extreme zero inflation and right-skewness in `capital-gain` and `capital-loss`.

The pipeline supports:

- Original financial values.
- `log1p` transformations.
- Binary indicators for positive financial values.

The logarithmic transformation is defined as:

\[
x_{\text{log}}=\ln(1+x)
\]

This transformation preserves zero values while compressing the scale of positive values.

Binary indicators identify whether the original capital value is greater than zero:

```text
capital-gain-positive
capital-loss-positive
```

The baseline and interactions variants use logarithmic representations and positive-value indicators. The robust variant additionally retains the original capital values.

#### Education representation

The EDA confirmed that `education` and `education-num` encode the same educational attainment information.

The standard preprocessing configurations retain the numerical `education-num` representation and exclude the textual `education` feature.

#### Categorical encoding

Categorical predictors are transformed using `OneHotEncoder`.

The standard configuration uses:

```python
OneHotEncoder(
    handle_unknown="infrequent_if_exist",
    min_frequency=0.01,
    sparse_output=False,
)
```

Categories representing less than 1% of the fitting observations are grouped into an infrequent-category representation where applicable.

Previously unseen categories are handled through the encoder's configured unknown-category behavior.

The encoder is fitted only on training data, ensuring that validation and test categories do not influence the learned encoding.

#### Interaction features

The `interactions` variant introduces additional features motivated by EDA subgroup analysis.

Two principal interaction mechanisms are implemented:

1. **Age × Education:** Multiplies age by numerical educational attainment to represent their combined relationship.
2. **Occupation × Working Hours:** Generates occupation-specific working-hour interaction features using the custom `OccupationHoursTransformer`.

The occupation-specific transformer learns its occupation categories during fitting and creates corresponding interaction columns.

These interactions are evaluated experimentally rather than assumed to improve predictive performance.

### 5.5 Preprocessing Variants

Three configurations are evaluated using the same underlying observations and partition assignments.

| Configuration | Numerical scaling | Financial representation | Interactions | Output features |
|---|---|---|---|---:|
| `baseline` | StandardScaler | `log1p` + positive indicators | No | 53 |
| `robust` | RobustScaler | Original + `log1p` + positive indicators | No | 55 |
| `interactions` | StandardScaler | `log1p` + positive indicators | Yes | 68 |

All three configurations use median numerical imputation, categorical `Unknown` imputation, one-hot encoding, and duplicate-aware splitting.

**Baseline:** Provides the standard EDA-informed representation without additional interaction features.

**Robust:** Investigates whether robust scaling and preservation of original financial values improve predictive performance.

**Interactions:** Investigates whether engineered relationships between age, education, occupation, and working hours improve predictive performance.

The configurations are compared through downstream model validation, not selected solely from exploratory statistics.

### 5.6 Pipeline Architecture and Leakage Prevention

The implementation uses modular scikit-learn components to separate feature engineering from learned preprocessing transformations.

The principal components include:

- `AdultFeatureEngineer` for deterministic feature transformations.
- `OccupationHoursTransformer` for occupation-specific interaction features.
- `SimpleImputer` for missing-value treatment.
- `StandardScaler` or `RobustScaler` for numerical scaling.
- `OneHotEncoder` for categorical encoding.
- `ColumnTransformer` for applying transformations to appropriate feature groups.
- `Pipeline` for combining the stages into a reusable preprocessing object.

The overall processing flow is:

```text
Raw Adult Income files
          |
          v
Data loading and validation
          |
          v
Duplicate-aware train/validation/test split
          |
          v
Feature engineering
          |
          v
Numerical and categorical transformations
          |
          v
Model-ready feature matrices
          |
          v
Saved CSV files and fitted preprocessors
```

**Leakage prevention:** Learned preprocessing parameters are fitted exclusively on the training partition. The fitted transformations are then applied to validation and test partitions without refitting.

This ensures that evaluation datasets do not influence the preprocessing statistics or categorical vocabulary learned during training.

The training module consumes the saved partitions and fitted preprocessing artifacts rather than independently recreating the data split.

### 5.7 Generated Artifacts and Reproducibility

Each preprocessing variant produces six CSV files containing model-ready features and corresponding target labels.

```text
data/processed/
├── baseline/
│   ├── X_train.csv
│   ├── X_val.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_val.csv
│   ├── y_test.csv
│   └── preprocessing_config.json
├── robust/
│   └── ...
└── interactions/
    └── ...
```

Fitted preprocessing pipelines are saved separately:

```text
models/preprocessors/
├── baseline.joblib
├── robust.joblib
└── interactions.joblib
```

The serialized pipelines preserve the fitted transformations needed to apply the same feature representation to new observations.

Configuration JSON files record the preprocessing settings associated with each variant.

The saved outputs were inspected for structural consistency:

- All three variants contain the expected training, validation, and test partitions.
- Feature dimensions match the recorded configurations: 53, 55, and 68.
- Feature matrices differ between preprocessing variants.
- Target labels remain consistent across variants for corresponding partitions.
- No missing values remain in the saved processed CSV files.

Processed datasets and fitted transformers are generated artifacts. They are not automatically assumed to be Git- or DVC-versioned and can be regenerated through the preprocessing runner.

**Reproducibility boundaries:** Fixed seeds, saved configurations, and serialized transformers support repeatable preprocessing. Exact numerical equality across different library versions or execution environments is not guaranteed.

The resulting datasets provide a consistent foundation for the model-training and experiment-tracking procedures described in [Section 6 — Training and MLflow](#6-training-and-mlflow).

## 6. Training and MLflow

The training pipeline is implemented primarily in `src/models/train.py`. It evaluates multiple machine learning algorithms and preprocessing configurations, tracks experiments with MLflow, selects candidates using validation performance, and produces reproducible model artifacts.

**The primary objective of this project is to demonstrate a complete MLOps lifecycle, not to exhaustively optimize predictive performance.** Consequently, the training experiment uses a deliberately limited hyperparameter search to validate the architecture while controlling computational cost.

### 6.1 Training Pipeline Architecture

The training module consumes the previously generated datasets and fitted preprocessors from Section 5.

For each preprocessing variant, it loads:

```text
data/processed/<variant>/
    X_train.csv
    y_train.csv
    X_val.csv
    y_val.csv
    X_test.csv
    y_test.csv

models/preprocessors/<variant>.joblib
```

The training module does not reload the raw dataset, recreate data partitions, or refit the saved preprocessing transformers.

The workflow follows these stages:

1. **Load and validate inputs:** Read the saved feature matrices, target labels, and fitted preprocessor for each variant.
2. **Load training settings and generate candidates:** Validate `configs/config.yml` (or a CLI-supplied YAML path), then enumerate the Cartesian product of each model family’s grid.
3. **Fit candidates:** Train each candidate using its training partition.
4. **Generate validation probabilities:** Estimate positive-class probabilities on validation observations.
5. **Optimize decision thresholds:** Find the threshold maximizing validation F1 for each candidate.
6. **Evaluate and rank candidates:** Calculate validation metrics and select the strongest configuration within each variant.
7. **Evaluate variant winners:** In full mode, evaluate each variant winner using its held-out test partition.
8. **Select the preferred candidate:** Compare variant winners using validation results, not test performance.
9. **Save artifacts:** Always write isolated experiment summaries and leaderboards; only **full mode** packages deployable pipelines and a promotable, unpromoted candidate plus metadata.
10. **Track experiments:** Record candidate parameters, metrics, configuration SHA-256, data fingerprints, and an execution ID for every candidate in MLflow.

Training and promotion are deliberately separate operations. A successful training execution does not automatically replace the reference model.

### 6.2 Model Families, YAML Configuration, and Hyperparameter Search

**Source of truth:** [`configs/config.yml`](configs/config.yml). Earlier versions of this project hard-coded model grids in `src/models/train.py`; the current implementation loads and validates them using **PyYAML**. The training module retains Python implementations of candidate construction, validation-F1 ranking, threshold selection, MLflow tracking, and artifact serialization.

The YAML contains these top-level sections:

| YAML key | Purpose |
|---|---|
| `seed` | Random seed passed to model constructors as `random_state` (default `42`) |
| `variants` | Which saved preprocessing variants to train (`baseline`, `robust`, `interactions`) |
| `selection_metric` | Declares selection by validation F1; currently only `val_f1` is accepted |
| `paths` | Default processed-data, preprocessor, run-output, MLflow database, and artifact paths |
| `mlflow.experiment` | Default MLflow experiment name (`adult-income-saved-splits`) |
| `models.<family>.fixed` | Estimator constructor settings held constant within that family's sweep |
| `models.<family>.grid` | Lists of parameter values combined through a Cartesian product in full mode |

An excerpt of the actual configuration illustrates the distinction between fixed values and search values:

```yaml
seed: 42
variants: [baseline, robust, interactions]
selection_metric: val_f1

models:
  hist_gradient_boosting:
    fixed:
      max_iter: 150
    grid:
      learning_rate: [0.05, 0.1]
      max_leaf_nodes: [15, 31]
```

This is an **excerpt**, not a standalone replacement for `config.yml`. The actual file includes the required `paths`, `mlflow`, and all four model families. Its loader currently requires **exactly** `logistic_regression`, `random_forest`, `hist_gradient_boosting`, and `extra_trees`, with nonempty grids and no overlap between fixed and swept parameters. Unsupported variants or a selection metric other than `val_f1` are rejected. The training config is therefore editable but **not an arbitrary plug-in algorithm framework**.

#### Why these model families?

| Algorithm | Main characteristics | Role in experiment |
|---|---|---|
| Logistic Regression | Linear, interpretable probabilistic classifier | Establishes a simpler baseline |
| Random Forest | Ensemble of independently constructed decision trees | Captures nonlinear relationships and interactions |
| HistGradientBoosting | Sequentially constructed boosted trees using histogram-based splits | Evaluates gradient boosting on tabular data |
| Extra Trees | Ensemble of highly randomized decision trees | Tests whether additional tree randomization improves generalization |

#### Default YAML search space

| Algorithm | `grid` values | Configurations per variant |
|---|---|---:|
| Logistic Regression | `C`: 0.1, 1.0, 10.0; `class_weight`: null, balanced | 6 |
| Random Forest | `max_depth`: null, 15; `min_samples_leaf`: 1, 5 | 4 |
| HistGradientBoosting | `learning_rate`: 0.05, 0.1; `max_leaf_nodes`: 15, 31 | 4 |
| Extra Trees | `max_depth`: null, 15; `min_samples_leaf`: 1, 5 | 4 |
| **Total** | | **18** |

The current `fixed` settings are:

- Logistic Regression: `solver: lbfgs`, `max_iter: 5000`, `tol: 0.0001`.
- Random Forest: `n_estimators: 150`, `n_jobs: -1`.
- HistGradientBoosting: `max_iter: 150`.
- Extra Trees: `n_estimators: 150`, `n_jobs: -1`.
- All four families receive `random_state` from YAML `seed` (currently `42`).

Logistic Regression is wrapped in an additional `StandardScaler` that is fitted using training data only. This scaling is part of the classifier candidate, not a refit of the saved preprocessing transformer.

The default search is repeated across three preprocessing variants:

| Variant | Candidates |
|---|---:|
| `baseline` | 18 |
| `robust` | 18 |
| `interactions` | 18 |
| **Default full experiment** | **54** |

Candidate counts are **derived from the YAML**, not fixed in the trainer. For a model family, the number of full-mode configurations is the product of its grid-list lengths. The total candidate count is that sum multiplied by the number of selected variants. Changing these lists or `variants` will therefore change how many candidates are trained, and can change selected hyperparameters and metrics.

This is a **manually implemented exhaustive grid enumeration**, not an invocation of `GridSearchCV`, `RandomizedSearchCV`, or Optuna. An external optimizer is a possible future improvement, not an implemented feature.

### 6.3 Quick and Full Execution Modes

The training module supports two modes. Both use the same YAML and training/validation splits. **Quick is a smoke check of training and MLflow integration, not a smaller model-selection contest.**

**Quick mode** is intended for development, testing, and orchestration verification:

```bash
python -m src.models.train --quick
```

Under the current YAML, quick mode trains the **first hyperparameter combination of each of the four model families** across three preprocessing variants: **4 × 3 = 12 candidates**. This checks that model construction, fitting, validation, threshold calculation, and MLflow logging work. **If any quick candidate fails, the smoke check fails**, even if other candidates succeed. It does **not** evaluate the held-out test set, serialize deployable per-variant pipelines, write decision-threshold files, or create a promotable overall candidate. It does still log fitted estimators to MLflow and writes a quick summary/leaderboard for diagnostics.

**Full mode** executes the complete Cartesian-product sweep:

```bash
python -m src.models.train
```

With the current YAML, this produces **18 × 3 = 54 candidate evaluations**, selects each variant winner using validation metrics, evaluates the variant winners on the held-out test set, and saves the preferred unpromoted candidate with metadata and per-variant model artifacts. Counts change when the YAML search space changes. **`scripts.run_pipeline --full` also invokes the MLflow comparison script for that specific execution**; direct `src.models.train` does not invoke the separate comparison automatically.

**Use an alternative training YAML file** without editing `train.py`:

```bash
python -m src.models.train --config configs/config.yml --quick
python -m src.models.train --config configs/config.yml
# Replace the path with your own complete, valid YAML file if needed.
```

The CLI also supports overrides for YAML defaults: `--processed-dir`, `--preprocessor-dir`, `--output-dir`, `--mlflow-db`, `--mlflow-artifacts`, and `--experiment`. When omitted, the trainer reads the corresponding settings from YAML. Relative configured paths are resolved against the repository root.

**Orchestration boundary:** `scripts/run_pipeline.py` invokes the training module without its own `--config` forwarding flag, so local and GitHub Actions orchestration currently use the committed default `configs/config.yml`. To reproduce a custom experiment with an alternative YAML file, invoke `src.models.train` directly after preprocessing, or extend the orchestrator explicitly. A model-training YAML edit does not automatically rerun or modify preprocessing configuration.

An earlier GitHub Actions full execution completed **54/54 candidates**. The latest full **local orchestrator** run on November 18, 2026 also completed 54 candidates, compared precisely those 54 MLflow records, and selected `baseline` consistently in both training and the comparison report. **476 tests passed** in that local pipeline. The automatic CI workflow has also passed since the rubric additions (owner report). A new manual GitHub Actions full run of the most recent execution-ID-aware orchestrator has not yet been supplied.

### 6.4 Validation, Model Ranking, and Decision Threshold Optimization

The project uses **positive-class F1** as its primary model-selection metric. The YAML explicitly declares `selection_metric: val_f1`; the current loader rejects other metric names, and the ranking implementation remains in Python.

The positive class represents annual income greater than $50,000.

F1 combines precision and recall:

\[
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

F1 was selected because the target is imbalanced, with approximately 24% positive observations, and accuracy alone can obscure poor positive-class detection.

#### Decision threshold optimization

Binary classifiers commonly apply a probability threshold of 0.5. However, this threshold is not necessarily optimal for F1.

For each trained candidate, the pipeline:

1. Predicts positive-class probabilities on validation data.
2. Uses `precision_recall_curve()` to obtain candidate decision thresholds.
3. Calculates F1 for those thresholds.
4. Selects the threshold maximizing validation F1.
5. Evaluates the candidate using that threshold.

The selected threshold is stored alongside the model and reused for subsequent evaluation and inference.

The preferred baseline model uses:

```text
Decision threshold: 0.3713238849891957
```

This means an observation is classified as positive when its predicted probability is approximately 37.13% or greater, rather than the conventional 50%.

Threshold optimization is performed exclusively on validation data.

#### Ranking policy

Candidates are ranked using the following priority:

1. Validation F1.
2. Validation average precision, recorded as `pr_auc`.
3. Validation ROC-AUC.

The secondary metrics serve as tie-breakers. When all three metrics tie, **training keeps the first candidate encountered** (YAML preprocessing-variant order and then candidate enumeration order). The standalone MLflow comparison applies the same deterministic order, so identical `baseline` and `robust` metrics no longer produce inconsistent winners within one execution.

The implementation calculates:

- F1.
- Precision.
- Recall.
- Accuracy.
- ROC-AUC.
- Average precision (reported as `pr_auc`).
- Confusion matrix.

Average precision summarizes precision–recall performance across thresholds; it is not identical to trapezoidal precision–recall AUC.

#### Held-out test evaluation

The test partition is reserved for evaluating selected candidates.

In full mode, the winner from each preprocessing variant is evaluated on its test partition using the threshold previously optimized on validation data.

The preferred variant is selected according to validation performance, **not the highest observed test F1**.

This distinction prevents test results from becoming an additional hyperparameter-selection mechanism.

### 6.5 Experimental Results

The recorded experiments using the **current default hyperparameter grid** selected `HistGradientBoostingClassifier` as the strongest model family within all three preprocessing variants. The YAML-driven local full run reported the same preferred family and variant as the earlier experiment; the numeric values below are the recorded benchmark results, not guaranteed outcomes after future YAML edits.

| Variant | Winning algorithm | Validation F1 | Test F1 | Test ROC-AUC |
|---|---|---:|---:|---:|
| **Baseline** | HistGradientBoosting | **0.7337** | 0.7150 | **0.9247** |
| Robust | HistGradientBoosting | **0.7337** | 0.7150 | **0.9247** |
| Interactions | HistGradientBoosting | 0.7314 | **0.7192** | 0.9240 |

The preferred configuration is **baseline + HistGradientBoosting**, with:

```text
learning_rate = 0.1
max_leaf_nodes = 15
```

Although the interactions variant achieved a slightly higher test F1, it was not selected because its validation F1 was lower.

This preserves the predefined selection policy and avoids selecting configurations using held-out test results.

The selected baseline model achieved:

| Metric | Validation | Test |
|---|---:|---:|
| F1 | 0.7337 | 0.7150 |
| Precision | 0.6942 | 0.7597 |
| Recall | 0.7781 | 0.6753 |
| Accuracy | 0.8649 | 0.8667 |
| ROC-AUC | 0.9304 | 0.9247 |

These results represent the recorded performance of a limited experimental search, not a claim that the selected model is globally optimal.

### 6.6 MLflow Experiment Tracking

MLflow provides local experiment tracking for the training pipeline. The default experiment and storage paths are configured through `configs/config.yml` and can be overridden with CLI flags.

Each candidate evaluation is associated with an MLflow run containing information such as:

- Algorithm name, preprocessing variant, and quick/full mode.
- **`training_execution_id` and `candidate_index` tags on every candidate run** (new executions), enabling deterministic filtering of a single full sweep.
- Swept hyperparameters (logged with `model_` prefixes) and fixed hyperparameters (logged with `fixed_` prefixes).
- Training seed and a hash of the candidate's training feature CSV (`data_version`).
- **`config_sha256`: SHA-256 of the exact training YAML file used**, allowing runs to be related back to a particular configuration revision.
- Validation metrics, default-threshold F1, and validation-optimized decision threshold.
- Convergence warning count and a `converged` tag for eligible estimator warnings.
- An MLflow-logged fitted sklearn estimator and related artifacts.

The trainer logs explicit model `pip_requirements` for key installed packages rather than allowing automatic package inference to scan unrelated dependencies. It also passes a narrowly defined allow-list of sklearn tree types where applicable. These settings affect **MLflow model logging**; the chosen deployable pipeline is independently packaged as a `joblib` file in the isolated execution directory.

The project uses a local SQLite tracking database, `mlflow.db`, and a separately configured artifact directory (`mlartifacts/` by default).

To obtain the configured MLflow UI command:

```bash
python -m src.models.train --show-mlflow-command
```

Execute the printed command and open the displayed local address, commonly:

```text
http://127.0.0.1:5000
```

The UI allows candidate runs to be compared without relying exclusively on console output. **A new training execution creates new MLflow run IDs even when its YAML content is unchanged.** The YAML fingerprint helps identify matching configuration content; it does not guarantee numerically identical model artifacts across platforms or dependency versions.

**Reproducibility limitation:** The local MLflow database and historical execution artifacts are not automatically distributed with a fresh GitHub clone. Reviewers can generate their own experiment history by running the trainer or shared pipeline.

#### Programmatic MLflow experiment comparison (rubric requirement)

The repository includes [`scripts/compare_experiments.py`](scripts/compare_experiments.py), which queries the configured experiment programmatically using **`mlflow.search_runs()`**. The query includes only completed candidate runs with a valid `metrics.val_f1`, excluding separate final-test records. It uses the same ranking priorities as training: validation F1, validation average precision, validation ROC-AUC, YAML variant order, and candidate order. **Held-out test metrics are not used for ranking.**

```bash
# Compare completed FULL-training candidates (default; excludes quick checks)
python -m scripts.compare_experiments

# Inspect smoke-check history explicitly, or inspect all eligible runs
python -m scripts.compare_experiments --mode quick
python -m scripts.compare_experiments --mode all

# Compare only the 54 candidates from a specific completed full execution
python -m scripts.compare_experiments --mode full --execution-id 21ed6abc583e4bee87433acd8f9517da

# Save a complete full-mode leaderboard as CSV
python -m scripts.compare_experiments --output reports/mlflow_comparison.csv
```

Without an `--execution-id`, the script can compare **multiple historical full runs** in the same local MLflow store; repeated configurations may appear more than once. Passing an execution ID limits the query to the runs bearing that tag. Candidate-level execution tags were introduced after the initial historical experiments, so filtering older runs can produce an incomplete subset. A fresh full run with the updated trainer provides complete execution coverage.

The **latest local full orchestration automatically** captures its newly created `models/runs/<execution_id>/` directory, verifies the full summary, runs the comparison with `--execution-id`, and exports `reports/mlflow_comparison.csv`. The observed execution `21ed6abc583e4bee87433acd8f9517da` compared **54 candidates**, selected MLflow run `d1e97e93b090407ea77490358e96ec82`, and matched the trainer's preferred `baseline`/HistGradientBoosting result (validation F1 0.7337).

The comparison is analysis only: **it never retrains, selects from held-out test metrics, or promotes a model**. Quick mode intentionally skips this full-execution comparison in the shared orchestrator.

### 6.7 Model Artifacts, Isolation, and Reproducibility

Each training invocation creates an isolated execution directory identified by a unique 32-character hexadecimal `execution_id`. That identifier is also attached to every new candidate's MLflow run; the shared full orchestrator uses it to isolate the run-specific comparison.

The generated structure is:

```text
models/runs/<execution_id>/
    sweep_leaderboard.csv
    training_summary.json        # full mode; records config_sha256
    best_model_pipeline.joblib
    best_model_metadata.json
    baseline/
    robust/
    interactions/
```

The full-execution directory contains:

- **`sweep_leaderboard.csv`:** Candidate-level performance information for comparing configurations.
- **`training_summary.json`:** Execution status, candidate counts, successful candidate count, YAML `config_sha256`, selected winners, and evaluation results.
- **`best_model_pipeline.joblib`:** Serialized preferred model with its fitted preprocessing pipeline.
- **`best_model_metadata.json`:** Model family, preprocessing variant, decision threshold, validation performance, source execution, and artifact provenance information. The YAML hash is recorded separately in the training summary and MLflow candidate parameters; it is **not currently stored directly in candidate metadata**.
- **Variant directories:** Additional variant-specific experiment outputs.

**Quick executions** instead produce `quick_training_summary.json` and `sweep_leaderboard.csv`, without deployable per-variant pipelines, decision-threshold JSON files, or a promotable candidate. The full-output tree above does not apply unchanged to quick mode. Both modes keep their MLflow candidate runs for inspection.

#### Deployable pipeline construction

The selected model artifact combines:

1. The fitted preprocessing pipeline from Section 5.
2. A feature-name compatibility adapter.
3. The trained classifier.

This allows the assembled model to process observations using the original input schema rather than requiring callers to manually apply the preprocessing transformations.

The implementation verifies feature compatibility and checks that the serialized classifier reproduces the fitted classifier's predicted probabilities on a sample of transformed validation observations.

The fitted preprocessing components are reused without refitting.

#### Model provenance and integrity

The saved metadata includes identifiers and fingerprints such as:

- Training execution ID.
- MLflow run ID.
- Git revision, when available.
- Preprocessing variant and model configuration.
- Input dataset SHA-256 hashes.
- Fitted preprocessor SHA-256 hash.
- Serialized model SHA-256 hash.
- Training YAML SHA-256 in the execution summary and MLflow candidate params (not in the current candidate metadata JSON).
- Optimized decision threshold.

These values support artifact traceability and integrity verification.

For the **earlier, pre-YAML-refactor** verified full GitHub Actions execution:

```text
Execution ID:
ef47ebedf90145199cba3ead91314b90

Candidates:
54 successful / 54 total

Selected MLflow run:
5ac7f6fa3ae4492e803d9a4e9c0da083

Git revision:
dd9af07544a58252ff9f308edf648affb88a3728
```

The recorded model SHA-256 is:

```text
f646ea469538c276b7e3a4f80e359214b6563631c3de9aceac4b67540bd98308
```

SHA-256 verification establishes file integrity relative to the recorded fingerprint; it does not independently guarantee model correctness or bit-for-bit reproducibility across different environments. The historical Git commit and candidate hash shown above identify the **earlier GitHub Actions execution**, not the new local YAML-driven run.

**Training isolation:** Newly generated models remain in `models/runs/<execution_id>/`. The training module does not overwrite the DVC-tracked reference model.

Candidate validation and explicit promotion are described in Section 7.

### 6.8 Model Optimization Scope and Future Improvements

**The primary objective of this project was to implement a complete, reproducible MLOps pipeline rather than maximize predictive performance.**

Model experimentation was deliberately limited to four implemented algorithm families, three default preprocessing variants, and **54 candidates under the current YAML**. The candidate count and hyperparameters can now be revised in `configs/config.yml` without editing the candidate-generation Python code.

This was sufficient to demonstrate training automation, hyperparameter comparison, MLflow tracking, validation-based selection, threshold optimization, held-out evaluation, artifact serialization, and explicit model promotion.

The selected model therefore represents a functional experimental baseline rather than an exhaustively optimized classifier.

Potential improvements include:

1. **Targeted hyperparameter tuning:** Expand the `configs/config.yml` grid around HistGradientBoosting and the strongest preprocessing variants. Explore learning rate, tree complexity, regularization, minimum leaf size, and boosting iterations. Keep a record of each YAML revision and its SHA-256 for comparison.

2. **Automated hyperparameter optimization:** Integrate Optuna, `RandomizedSearchCV`, or `GridSearchCV`. The current YAML defines grid inputs, but the trainer does **not** yet call any of those optimizers. Optuna could explore larger, conditional parameter spaces using adaptive sampling and trial pruning; exhaustive grids provide systematic coverage of a predefined space at potentially greater computational cost.

3. **Expanded algorithm comparison:** Evaluate additional models such as XGBoost, LightGBM, and CatBoost, including their preprocessing requirements and computational tradeoffs.

4. **Group-aware cross-validation:** Introduce `StratifiedGroupKFold` within the development data to estimate model performance across multiple folds while preventing identical predictor groups from appearing in both training and validation folds. Keep the held-out test partition outside optimization. Threshold selection would require out-of-fold predictions or a separate validation strategy.

5. **Additional feature engineering:** Investigate alternative interactions, feature selection, categorical encoding, and feature importance analysis. Reassess whether the robust or interactions variants improve under more extensive tuning.

6. **Probability calibration and decision analysis:** Examine calibration methods and optimize decision thresholds for alternative objectives, such as recall, precision, or application-specific misclassification costs.

7. **Fairness and subgroup evaluation:** Analyze predictive performance across demographic groups and investigate potential disparities, considering the historical and observational limitations of the Adult Income dataset.

8. **Computational efficiency:** Explore early stopping, parallel experimentation, adaptive trial allocation, and optimization strategies that reduce training cost without compromising evaluation integrity.

Future experiments should retain the existing MLOps safeguards:

- Fit learned preprocessing only on training observations.
- Preserve duplicate-group separation between training and validation folds.
- Use validation or cross-validation results for model selection.
- Keep the held-out test set outside hyperparameter and threshold optimization.
- Record YAML configuration fingerprints, candidate settings, metrics, and artifacts in MLflow.
- Preserve isolated execution outputs and model provenance.
- Require explicit validation and promotion before replacing the reference model.

The current architecture provides a foundation for these improvements without requiring a redesign of the entire MLOps workflow.

## 7. Candidate Validation and Explicit Promotion

The project separates **model training, candidate validation, and reference model promotion** into distinct operations.

Training produces an isolated candidate model, but that candidate does not automatically become the project's official reference model.

The promotion process is implemented in:

```text
src/models/promote.py
```

This separation prevents newly trained models from replacing the existing reference model without validation and explicit authorization.

### Understanding the Promotion Lifecycle

The workflow consists of five stages:

1. **Training:** A full training execution evaluates model configurations and saves the preferred candidate in `models/runs/<execution_id>/`.
2. **Candidate validation:** The promotion module verifies the candidate's provenance, integrity, and structural compatibility.
3. **Explicit promotion:** A user authorizes replacement of the current reference model using `--confirm`.
4. **Reference model update:** The validated candidate becomes the model stored in `models/reference/`.
5. **Artifact versioning:** The repository owner can use DVC and Git to preserve and publish the updated reference model.

Three terms are important:

- **Candidate model:** The model selected by a completed training execution but not yet adopted as the reference.
- **Validated candidate:** A candidate that has passed the promotion module's integrity and consistency checks.
- **Reference model:** The officially selected model used by downstream project workflows, including robustness evaluation.

Promotion does not deploy a model to a production API or external service. It updates the project's designated reference artifacts.

### 7.1 Candidate Validation (Dry Run)

Before replacing the reference model, the candidate must pass a series of validation checks.

Run:

```bash
python -m src.models.promote --run-id <FULL_EXECUTION_ID>
```

Replace `<FULL_EXECUTION_ID>` with the identifier generated by a completed full training execution.

The identifier is a 32-character lowercase hexadecimal string identifying the training execution, not an individual algorithm or MLflow run.

For example:

```bash
python -m src.models.promote --run-id ef47ebedf90145199cba3ead91314b90
```

Without `--confirm`, the command performs **validation only**. It does not modify the existing reference model.

#### Validation checks

The implementation verifies the following conditions:

| Validation | Purpose |
|---|---|
| Execution ID format | Rejects invalid identifiers and directory traversal attempts |
| Candidate artifacts exist | Confirms the model and metadata files are present |
| Execution ID consistency | Ensures candidate metadata and training summary identify the requested execution |
| Full execution | Rejects quick-mode training outputs |
| Training completion | Requires a positive candidate count and all candidates reported successful |
| Preferred variant consistency | Confirms the candidate matches the variant selected during training |
| MLflow run consistency | Checks the recorded winner's `FINISHED` status and matching MLflow run ID |
| SHA-256 integrity | Confirms the serialized model matches its recorded fingerprint |
| Decision threshold | Requires a finite threshold between 0 and 1 |
| Model compatibility | Confirms the loaded model supports `predict_proba()` and includes positive class `1` |

These checks address several potential failure conditions, including incomplete training executions, inconsistent metadata, corrupted model files, and incompatible prediction interfaces.

**Important limitations:**

- Validation does not retrain the model or independently recalculate its predictive performance.
- MLflow consistency is checked against the saved training summary, not by querying the live MLflow tracking database.
- SHA-256 confirms file integrity relative to the recorded hash; it does not establish predictive correctness.
- Loading serialized `joblib` artifacts can execute untrusted code. Only trusted model artifacts should be validated.

#### Expected output

A successful dry run returns a JSON response similar to:

```json
{
  "execution_id": "<FULL_EXECUTION_ID>",
  "candidate_sha256": "<SHA256>",
  "validated": true,
  "promoted": false
}
```

The result indicates that validation succeeded but the reference model was not changed.

### 7.2 Explicit Model Promotion

After inspecting the experiment results and validating the candidate, the user can explicitly authorize promotion.

Run:

```bash
python -m src.models.promote --run-id <FULL_EXECUTION_ID> --confirm
```

The `--confirm` argument is mandatory for reference replacement.

This is a deliberate safeguard: a successful training execution should not automatically replace an existing reference model simply because it achieved the highest validation score in a new experiment.

#### What happens during promotion?

The promotion module performs the following operations:

1. Resolves and validates the selected candidate.
2. Creates the reference directory if necessary.
3. Backs up the existing reference model and metadata.
4. Constructs updated reference metadata.
5. Copies the candidate model to a temporary file.
6. Verifies that the copied model's SHA-256 matches the original candidate.
7. Replaces the reference model file.
8. Writes the updated metadata through a temporary file.
9. Reports successful promotion.

The resulting reference directory contains:

```text
models/reference/
    best_model_pipeline.joblib
    best_model_metadata.json
```

The original candidate remains in its isolated execution directory, preserving the distinction between experimental artifacts and the adopted reference model.

#### Promotion metadata

The reference metadata retains the candidate's existing training information, including the selected algorithm, preprocessing variant, evaluation results, decision threshold, and provenance fingerprints.

Promotion adds or updates:

| Field | Description |
|---|---|
| `execution_id` | Source training execution |
| `model_sha256` | Fingerprint of the promoted model |
| `promoted_at` | UTC timestamp of promotion |
| `promotion_source` | Original candidate execution directory |
| `promotion_type` | `explicit` |

These fields help identify which experiment produced the reference model and when it was adopted.

#### Failure handling and rollback

The implementation includes a backup-and-rollback mechanism.

Before replacement, existing reference artifacts are copied to a temporary backup directory.

If an exception occurs during the replacement process, the module attempts to restore the previous reference files.

However, there is an important technical limitation:

**The model and metadata are two separate files, so promotion is not a single atomic transaction.**

Each replacement uses atomic file operations where supported, but an abrupt process termination between replacements could leave the model and metadata inconsistent.

The rollback mechanism therefore provides protection against handled exceptions, not a guarantee of transactional recovery from every possible failure.

#### Expected output

A successful promotion returns a response similar to:

```json
{
  "execution_id": "<FULL_EXECUTION_ID>",
  "candidate_sha256": "<SHA256>",
  "validated": true,
  "promoted": true,
  "reference_model": "<ABSOLUTE_PATH_TO_REFERENCE_MODEL>"
}
```

At this point, the model has been promoted **locally**.

It has not necessarily been uploaded to the DVC remote or committed as a new Git revision.

### 7.3 Persisting the Reference Model with DVC and Git

The promotion module updates the local reference artifacts. Versioning and publishing those artifacts are separate operations.

The project uses:

- **Git:** To track the DVC pointer file, `configs/config.yml`, and source code, associating model versions and experimental settings with repository revisions.
- **DVC:** To track the reference directory and identify its exact artifact contents.
- **Amazon S3:** To store DVC-managed artifact data remotely.

The reference model and metadata are tracked together through:

```text
models/reference.dvc
```

This directory-level tracking keeps both artifacts associated with the same DVC-tracked reference version.

#### Owner-authorized publishing workflow

After promoting a new reference model, the repository owner can run:

```bash
# Update the DVC-tracked reference directory
dvc add models/reference

# Upload new or changed DVC objects to S3
dvc push

# Commit the updated DVC pointer
git add models/reference.dvc
git commit -m "feat: update promoted reference model"

# Publish the Git revision
git push
```

This sequence ensures that the Git revision identifies the appropriate DVC-tracked model artifacts.

The binary model itself should not be committed directly to Git.

#### S3 Access Restrictions and Cost Control

**The project's S3 remote is intentionally configured for public-read access with restricted write permissions.**

Reviewers can retrieve the versioned datasets and reference model using:

```bash
dvc pull
```

The configured public-read objects can be downloaded without AWS credentials.

However, uploading new artifacts requires authorized AWS credentials and appropriate S3 write permissions.

Therefore:

```bash
dvc push
```

will fail for reviewers who do not have those permissions whenever an upload is required.

This restriction is intentional and serves two purposes:

1. **Artifact integrity:** Prevents unauthorized users from modifying or adding objects to the project's remote artifact storage.
2. **Cost control:** Prevents unauthorized uploads and uncontrolled storage growth that could generate additional charges for the repository owner.

Public-read access does not eliminate all S3 costs. Downloads may still generate request and data-transfer charges.

**Reviewers do not need S3 write permissions to reproduce the local MLOps workflow.**

They can download the required artifacts, run preprocessing and training, evaluate new candidates, validate them, and promote a model locally.

Only publishing new artifacts to the owner's S3 remote requires additional authorization.

#### Retrieving a versioned reference model

A reviewer can restore a reference model associated with a specific Git revision:

```bash
git checkout <COMMIT_OR_TAG>
dvc pull models/reference.dvc
```

Git selects the DVC pointer associated with that revision, while DVC retrieves the corresponding model and metadata from S3.

Successful recovery depends on the referenced objects remaining available in remote storage.

### 7.4 Design Rationale

The explicit promotion mechanism implements a controlled separation between experimentation and reference model management.

Its primary objectives are:

- **Prevent accidental replacement:** Training never automatically overwrites the reference model.
- **Validate artifact consistency:** Candidate files must pass integrity, provenance, and compatibility checks.
- **Preserve experiment isolation:** Training outputs remain associated with their original execution identifiers.
- **Support traceability:** Reference metadata identifies the source execution and promotion event.
- **Provide failure recovery:** Backups and rollback handling reduce the risk of losing the previous reference during handled failures.
- **Enable versioned recovery:** Git and DVC associate published reference models with identifiable repository revisions.
- **Control remote access:** S3 write permissions remain restricted to authorized users.

This design establishes a basic model governance mechanism suitable for an educational MLOps project.

It does not constitute a complete production model registry, approval workflow, or deployment system, but it provides the necessary separation between **training a model, validating its artifacts, adopting it as the reference, and publishing that reference for reproducibility**.

## 8. DVC and artifact versioning

### 8.1 Retrieve a specific version

```bash
git checkout <COMMIT_OR_TAG>
dvc pull data/raw.dvc models/reference.dvc
dvc status
```

The Git revision selects the `.dvc` pointers; DVC retrieves the corresponding content from the remote. Exact recovery depends on those objects still existing in S3.

### 8.2 Inspect the remote and local status

```bash
dvc remote list
dvc status
git status
```

The configured S3 remote supports anonymous **reads** of designated objects. Uploading requires suitable credentials/permissions. Never commit AWS access keys or other secrets.

### 8.3 What is and is not versioned

| Artifact | Storage/versioning policy |
|---|---|
| Python source, tests, README, workflows, `configs/config.yml` | Git |
| Raw Adult data | DVC + S3 |
| Promoted `models/reference/` directory | DVC + S3; pointer in Git |
| Per-execution training candidates | Generated locally; not automatically promoted or uploaded |
| Processed datasets / fitted preprocessors | Generated by preprocessing; check individual DVC tracking before assuming persistence |
| Local MLflow SQLite database/artifacts | Local; not automatically shared |
| Drift and robustness reports | Generated locally; not automatically versioned |
| `reports/mlflow_comparison.csv` | Generated by full orchestration; not automatically Git/DVC-versioned; uploaded in manual GitHub Actions artifact bundle when present |

**DVC does not orchestrate the stages.** The Python script `scripts/run_pipeline.py` defines stage order, and GitHub Actions provides remote execution infrastructure. `configs/config.yml` is an ordinary Git-versioned training configuration, **not** a DVC `dvc.yaml` pipeline definition. There is no requirement to introduce `dvc.yaml` solely to use DVC for artifact versioning.

## 9. Data drift monitoring

### 9.1 Synthetic scenarios

The project generates controlled feature shifts to evaluate detection and model sensitivity:

| Scenario | Approximate intervention level | Purpose |
|---|---|---|
| `control` | No injected drift | Baseline comparison |
| `mild` | Small shifts | Low-severity perturbation |
| `moderate` | Intermediate shifts | Warning-level change |
| `severe` | Large shifts | Critical-level change |

Generated labeled scenarios are stored under `data/drift/<scenario>/`, typically as `X_test.csv` and `y_test.csv`. These scenarios are synthetic stress tests, not observed production traffic.

### 9.2 Run monitoring

```bash
python -m src.data.monitoring
```

Monitoring uses Evidently to compare feature distributions and reports the share of drifted columns. The configured status policy is:

| Share of drifted columns | Status |
|---|---|
| `<= 20%` | `NORMAL` |
| `> 20%` and `< 35%` | `WARNING` |
| `>= 35%` | `CRITICAL` |

### 9.3 Recorded monitoring outcome

| Scenario | Drifted columns | Share | Status |
|---|---:|---:|---|
| Mild | 0 / 14 | 0.00% | NORMAL |
| Moderate | 3 / 14 | 21.43% | WARNING |
| Severe | 5 / 14 | 35.71% | CRITICAL |

The recorded overall status was **CRITICAL** because the severe scenario crossed the configured threshold. Uploaded Evidently reports were inspected; severe drift affected `age`, `education-num`, `hours-per-week`, `workclass`, and `occupation`. Evidently used Jensen–Shannon distance with a per-feature threshold of 0.1 in the inspected severe report. Its report-level `drift_share=0.5` setting is distinct from the project alert policy (WARNING above 20%, CRITICAL at or above 35%). Reports are generated under `reports/drift/`.

**Interpretation:** Drift is a change in input distribution, not direct proof that predictive performance has degraded. Robustness evaluation addresses the latter question using labeled synthetic scenarios.

### 9.4 Drift analysis and recommended operational response

This subsection addresses the Sprint 17 rubric's three monitoring questions. It describes a **controlled synthetic test**, not actual production traffic.

**Which features drifted, and why?** In the inspected severe report, `age`, `education-num`, `hours-per-week`, `workclass`, and `occupation` drifted; **5/14 features (35.71%)** crossed Evidently's per-column criteria. The moderate scenario flagged **3/14 (21.43%)**, while mild flagged none. The generator deliberately perturbs ages, education, hours worked, workclass, and occupation at increasingly large intervention rates. The known cause in this experiment is **synthetic feature modification**; the reports do not establish a real demographic or economic change in a production population.

**Would the drift affect model performance?** It plausibly would, especially because age, education, working hours, and employment fields are predictive features. The independently recorded labeled robustness evaluation shows the promoted reference model's **F1 decreasing from 0.7150 (control) to 0.7124 (mild), 0.6807 (moderate), and 0.5525 (severe)**. The severe change is **−0.1625 absolute F1 (approximately −22.7% relative to control)**. This directly demonstrates sensitivity to these *particular* synthetic shifts; distribution drift alone would not prove such degradation without labels.

**What action is recommended?**

| Scenario | Monitoring status | Recommended action |
|---|---|---|
| Mild | NORMAL; 0/14 features | **Continue monitoring.** F1 changed by only −0.0026 in this test; no retraining is justified solely by this result. |
| Moderate | WARNING; 3/14 features | **Investigate** which upstream distributions and data sources changed; monitor stability and check labeled validation performance. Consider retraining if the shift persists and harms performance. |
| Severe | CRITICAL; 5/14 features | **Investigate immediately; plan retraining or model rollback if this occurred in production.** Verify data quality and target labels, train on representative updated data, compare against the reference on a protected evaluation set, and require explicit promotion after validation. |

In this project, severe drift is **deliberate**; its exit code `1` is recorded as an alert while the orchestration continues to robustness analysis. It does not automatically trigger retraining or promotion. In a real system, an equivalent alert would require investigation and a governed recovery decision rather than blindly retraining on unknown-quality data.

## 10. Model robustness under drift

### 10.1 Purpose

Robustness evaluation loads the **promoted reference model** and its saved decision threshold, applies it to labeled control and drifted datasets, and compares predictive metrics against the control scenario.

It reports accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and PR-AUC, as applicable. Robustness analysis is separate from drift detection: a feature can drift without causing a large metric change, and a metric can degrade even when a chosen drift detector does not flag a shift.

### 10.2 Run the evaluation

After generating the drift datasets, run:

```bash
python -m src.models.robustness \
  --control data/drift/control/X_test.csv \
  --control-target data/drift/control/y_test.csv \
  --mild data/drift/mild/X_test.csv \
  --mild-target data/drift/mild/y_test.csv \
  --moderate data/drift/moderate/X_test.csv \
  --moderate-target data/drift/moderate/y_test.csv \
  --severe data/drift/severe/X_test.csv \
  --severe-target data/drift/severe/y_test.csv
```

The example uses Bash line continuation (`\`). On PowerShell, use a single line or replace the continuation syntax appropriately.

### 10.3 Recorded robustness results

| Scenario | F1 (positive class) | Change versus control |
|---|---:|---:|
| Control | 0.7150 | — |
| Mild | 0.7124 | −0.0026 |
| Moderate | 0.6807 | −0.0343 |
| Severe | 0.5525 | −0.1625 |

These values are from the inspected JSON and CSV robustness artifacts, each covering four scenarios of 9,764 observations. Severe drift reduced F1 by approximately **22.7% relative to control**. The robustness report references the **promoted DVC reference model** (SHA-256 prefix `d5254757`), not any model trained by the current orchestration. The hash prefix `f646ea46` identified an **earlier** unpromoted GitHub Actions candidate, not necessarily the latest local candidate. This distinction is intentional: orchestration never automatically promotes a candidate. The synthetic scenarios demonstrate stress-test sensitivity, not production performance. Reports are written under `reports/robustness/` and are not currently committed to Git.

## 11. Testing, CI, and pipeline orchestration

### 11.1 Automated testing (pytest)

The repository includes unit and integration tests for preprocessing, training, configuration, MLflow comparison, model serialization and promotion, drift generation, Evidently monitoring, robustness, plus **rubric-specific validation of the real DVC data and a small classifier**.

```bash
# Restore raw data first: the real-data tests intentionally require these files
# (skip the pull if they are already present)
dvc pull data/raw.dvc

# Run the complete test suite from the repository root
python -m pytest tests/ -q --tb=short

# Review the rubric-specific test modules
python -m pytest tests/test_data_validation.py -v
python -m pytest tests/test_model_validation.py -v
python -m pytest tests/test_compare_experiments.py -v

# Focused training, YAML, and MLflow checks
python -m pytest tests/test_train.py -v
```

**Real-dataset validation:** `tests/test_data_validation.py` has **five checks** against the actual normalized Adult data from `adult.data` and `adult.test`: expected columns/provenance, sample count and numeric/categorical types, target values, numeric bounds, and retained categorical missingness. These tests use the established `load_and_concatenate_data()` and `clean_data()` functions instead of duplicating the loader. They fail explicitly when the required DVC data is absent; synthetic fixtures alone would not satisfy the dataset-validation rubric.

**Model validation:** `tests/test_model_validation.py` has **three lightweight checks** using a deterministic sample and fitted logistic regression: prediction type/shape, **held-out F1 ≥ 0.80 on the known synthetic test set**, and valid probability outputs. This stable unit-test threshold is distinct from the **actual-data CI threshold of validation F1 ≥ 0.65**, checked after quick candidate training.

**Experiment/training tests:** These cover YAML hyperparameter propagation, candidate generation, execution-ID MLflow tags, quick-mode failure on any failed candidate, comparison filtering and deterministic tie-breaking, and full-mode candidate preservation. Quick mode is never allowed to promote or evaluate held-out test performance.

**Latest recorded full local orchestration result:**

```text
476 passed, 11 warnings in 153.74s (0:02:33)
```

The 11 warnings are nonfatal third-party deprecations from Evidently/NumPy (2), Litestar (8), and MLflow/SQLAlchemy (1). The complete full orchestration, including these tests, finished successfully on November 18, 2026 (local time). Passing tests do not guarantee production validity or fairness.

### 11.2 Automatic CI: two dependent GitHub Actions jobs

`.github/workflows/ci.yml` is the rubric-mandated **automatic** CI workflow. It triggers on `push` to `main`, `pull_request` targeting `main`, and optional manual dispatch. It has two dependent jobs:

| Job | Actions and failure conditions |
|---|---|
| **`test`** | Check out the code, install Python 3.11 and dependencies including `dvc[s3]`, **pull `data/raw.dvc`**, then run `python -m pytest tests/ -v`. Any failed test fails the job. |
| **`train`** (`needs: test`) | Download the DVC raw dataset again on the separate runner, run preprocessing, train **quick-mode** candidates, then check MLflow validation F1 against `MIN_VALIDATION_F1=0.65`. A failed training subprocess, missing metrics, or insufficient best F1 causes a nonzero exit status. |

This training job is deliberately an **integration/performance gate**, not full model selection. It does not evaluate the held-out test data, create a promotable candidate, or update the DVC reference. Its artifacts are uploaded for inspection with a 14-day retention setting.

**Status:** The repository owner reported that the revised CI workflow passed after adding the rubric-specific tests. No CI run ID or URL is embedded here. The separate manual orchestration workflow is not a substitute for this required two-job CI design.

**Current limitation:** The CI threshold check calls `mlflow.search_runs(search_all_experiments=True)` without an explicit execution-ID filter. This is adequate only if the metrics it reads are from the intended CI training; when reusing MLflow storage or editing the workflow, ensure the checking process points to the same tracking store as the trainer. A more restrictive execution-ID/experiment filter would prevent unrelated historical scores from satisfying the gate.

### 11.3 Shared Python pipeline orchestration

`scripts/run_pipeline.py` defines the execution order for both local runs and the manually dispatched GitHub workflow:

1. Optionally run `dvc pull data/raw.dvc models/reference.dvc` with `--pull-dvc` (GitHub Actions normally pulls artifacts before invocation).
2. Validate reference model artifacts; execute **the full pytest suite**, including actual-data tests.
3. Run `python -m scripts.run_preprocessing` for the selected preprocessing variants.
4. Run `python -m src.models.train --quick` (smoke check) or `python -m src.models.train` (full YAML sweep).
5. **Full mode only:** detect the one newly created training directory, verify its `training_summary.json`, call `python -m scripts.compare_experiments --mode full --execution-id <CURRENT_EXECUTION_ID> --output reports/mlflow_comparison.csv`.
6. Generate labeled synthetic drift datasets with `src.data.drift_generator`.
7. Run Evidently monitoring (`src.data.monitoring`). Exit `1` indicates the expected **CRITICAL** synthetic drift alert and is recorded; unexpected errors still fail the pipeline.
8. Run `src.models.robustness` against the **existing promoted reference model**, not the newly trained candidate.

The comparison step belongs inside the **shared Python orchestrator**, not exclusively in GitHub Actions. Full orchestration therefore produces an execution-scoped leaderboard locally as well. Quick mode skips this step entirely. The script prints stage durations, exit codes, detected drift severity, and, in full mode, the training execution ID and report path.

The reference model is never automatically promoted or uploaded to S3. The orchestrator uses the committed default `configs/config.yml`; direct `src.models.train --config ...` is available for custom-YAML experiments.

### 11.4 Manual GitHub Actions orchestration

`.github/workflows/ml_pipeline_orchestration.yml` provides a **manually dispatched** `training_mode` selector (`quick` or `full`) and an Ubuntu runner. It installs dependencies, restores `data/raw.dvc` and `models/reference.dvc`, verifies these artifacts, and invokes the shared orchestrator:

```bash
python -m scripts.run_pipeline --quick
# or
python -m scripts.run_pipeline --full
```

It preserves generated drift/robustness reports, training-run artifacts, and (for full mode) `reports/mlflow_comparison.csv` as downloadable GitHub Actions artifacts (14-day retention). Promotion and S3 writes are disabled. Earlier manual quick and full runs succeeded on Ubuntu, but they predated the latest local execution-scoped comparison integration.

**Workflow synchronization pending:** The last supplied version of the manual workflow still invokes `compare_experiments.py` **again after `run_pipeline.py`** on full runs. The updated Python orchestrator already performs the execution-ID-filtered comparison, so the extra GitHub step is redundant and can overwrite the report with an unfiltered historical leaderboard. **Remove only that standalone comparison step**; preserve the CSV artifact upload and other workflow stages. No new manual full-mode GitHub Actions execution after this final synchronization has been supplied.

## 12. Reproducing the full workflow

### 12.1 One-command local orchestration

After cloning, installing Python 3.11 dependencies, and pulling the versioned input artifacts if necessary (Section 3), run from the repository root:

```bash
# Quick full-pipeline verification with DVC restoration (12 smoke candidates)
python -m scripts.run_pipeline --quick --pull-dvc

# Full pipeline with DVC restoration (54 candidates plus scoped comparison)
python -m scripts.run_pipeline --full --pull-dvc

# Reuse already downloaded DVC artifacts locally
python -m scripts.run_pipeline --quick
python -m scripts.run_pipeline --full
```

`--quick` is the default training mode. The candidate counts come from the YAML grid; these 12/54 figures apply to the current configuration. Both modes run pytest, preprocessing, synthetic drift generation, monitoring, and reference-model robustness; quick is a training smoke check, while full creates an unpromoted candidate and an **execution-scoped MLflow comparison CSV**. The intentionally severe synthetic drift produces a CRITICAL alert but does not prevent the labeled robustness analysis.

**Latest verified local full execution (Windows 11, November 18, 2026 local / November 19 UTC):**

| Evidence | Observed result |
|---|---|
| Full pytest suite | **476 passed, 11 warnings** |
| Preprocessing | `baseline` 53 features; `robust` 55; `interactions` 68 |
| Training | **54 candidates**, three variant winners |
| Execution ID | `21ed6abc583e4bee87433acd8f9517da` |
| Preferred candidate | `baseline` + `HistGradientBoostingClassifier` |
| Validation F1 / held-out test F1 | **0.7337 / 0.7150** |
| MLflow comparison | **54 eligible candidates**, same `baseline` winner; exported `reports/mlflow_comparison.csv` |
| Drift monitoring | Mild NORMAL; moderate WARNING; severe CRITICAL (expected) |
| Robustness | Promoted-reference F1: control 0.7150; severe 0.5525 |
| Final orchestration status | **SUCCESS**; seven executed stages, approximately 439.67 seconds of stage time |

The current full-run winner was **not promoted**. The robustness stage continued using the existing DVC-promoted reference (SHA-256 prefix `d5254757`), exactly as intended.

### 12.2 GitHub Actions execution

**Automatic grading CI:** Every push to `main` or pull request targeting `main` triggers `.github/workflows/ci.yml`: the DVC-aware `test` job must pass before the quick `train` job and the minimum validation-F1 gate run. The repository owner has reported a successful revised CI run.

**Optional full orchestration:** Open the repository's **Actions** tab, select **MLOps Pipeline Orchestration**, click **Run workflow**, then choose `quick` or `full`. The workflow uses Ubuntu 24.04, restores the raw dataset and promoted reference from DVC, and runs the same Python orchestrator used locally. Download the reports from the run's artifacts.

Earlier quick and full manual GitHub Actions runs completed successfully. The latest **full comparison step was verified locally**, but a new manually dispatched full run following the Python integration and YAML synchronization has not been reported. Before rerunning, remove the workflow's redundant standalone comparison stage (Section 11.4).

### 12.3 Optional candidate review and promotion

Full training stores a selected **unpromoted** candidate in `models/runs/<execution_id>/`. Review the `training_summary.json`, the MLflow comparison CSV, the validation/test metrics, and the candidate metadata before authorizing any reference replacement:

```bash
# Validate this execution without modifying the reference
python -m src.models.promote --run-id 21ed6abc583e4bee87433acd8f9517da

# Only with explicit authorization: promote the candidate LOCALLY
python -m src.models.promote --run-id 21ed6abc583e4bee87433acd8f9517da --confirm
```

These commands are examples for a run that exists on the originating machine; reviewers must substitute their own completed full-training execution IDs. Promotion is **not** part of CI or orchestration, and local promotion does not imply deployment.

**Optional — authorized repository owner only**, after actual promotion:

```bash
dvc add models/reference
dvc push
git add models/reference.dvc
git commit -m "feat: update promoted reference model"
git push origin main
```

S3 uploads require the repository owner's authorized write credentials; readers can pull the public-read artifacts without them.

### 12.4 Reviewing execution evidence

Inspect an orchestration run's logs and artifact bundle. For a **full** run, verify `models/runs/<execution_id>/training_summary.json`, `best_model_metadata.json`, `sweep_leaderboard.csv`, and `reports/mlflow_comparison.csv`. The training summary identifies the validation-preferred variant and records `config_sha256`; the comparison report should contain only the runs associated with that **same execution ID**, not all historical MLflow data. Inspect `reports/drift/` for the HTML/JSON drift reports and `reports/robustness/` for labeled performance degradation.

Distinguish these two historical executions:

| Execution | Location and implementation | Role |
|---|---|---|
| `ef47ebedf90145199cba3ead91314b90` | Earlier GitHub Actions full run, pre-YAML comparison updates | Demonstrates successful earlier remote orchestration (54/54) |
| `21ed6abc583e4bee87433acd8f9517da` | Latest verified local Windows full orchestration | Demonstrates 476 passing tests, current-execution MLflow comparison (54/54), drift and robustness |

**Artifact policy:** GitHub Actions artifacts are retained for 14 days. Raw data and the promoted reference are persistently identified by Git/DVC pointers and available S3 objects. Local MLflow history and new unpromoted candidates are generated separately and are not automatically uploaded to DVC or adopted as the reference.

### Reproducibility boundaries

- Git revision, pinned dependencies, DVC pointer hashes, fixed seeds, data/preprocessor SHA-256 and YAML `config_sha256` support traceability but not guaranteed bitwise identical models across platforms.
- Every training execution has a unique ID; every updated candidate run carries that ID as an MLflow tag. Old runs may lack complete tags.
- Full-mode comparison must be scoped by execution ID to reproduce the selected winner from a particular run; comparing all historical runs is a different analysis.
- Quick mode validates training infrastructure, but does not evaluate the held-out test set or produce a promotable model.
- Full mode evaluates variant winners on the held-out test set without using those test scores for choosing the preferred variant.
- The promoted reference is explicitly approved and DVC-tracked; it does not automatically change when a new candidate is trained.
- The pipeline is sequential automation, not a production model-serving system or distributed scheduler.

## 13. Command reference

| Task | Command |
|---|---|
| Restore raw data and promoted reference | `dvc pull data/raw.dvc models/reference.dvc` |
| Run complete tests | `python -m pytest tests/ -q --tb=short` |
| Run actual-data validation tests | `python -m pytest tests/test_data_validation.py -v` |
| Run model-validation tests | `python -m pytest tests/test_model_validation.py -v` |
| Run MLflow comparison tests | `python -m pytest tests/test_compare_experiments.py -v` |
| Run local orchestrator (quick, DVC pull) | `python -m scripts.run_pipeline --quick --pull-dvc` |
| Run local orchestrator (full, DVC pull) | `python -m scripts.run_pipeline --full --pull-dvc` |
| Run orchestration without pulling DVC again | `python -m scripts.run_pipeline --full` |
| Run preprocessing for all variants | `python -m scripts.run_preprocessing` |
| Run a single preprocessing variant | `python -m scripts.run_preprocessing --config baseline` |
| Quick training with default YAML | `python -m src.models.train --quick` |
| Full training with default YAML | `python -m src.models.train` |
| Train with specified YAML | `python -m src.models.train --config configs/config.yml` |
| Compare completed full MLflow runs (default) | `python -m scripts.compare_experiments` |
| Compare one full training execution | `python -m scripts.compare_experiments --execution-id <FULL_EXECUTION_ID>` |
| Inspect quick-mode MLflow history | `python -m scripts.compare_experiments --mode quick` |
| Inspect all eligible MLflow records | `python -m scripts.compare_experiments --mode all` |
| Export full MLflow leaderboard | `python -m scripts.compare_experiments --output reports/mlflow_comparison.csv` |
| Display configured MLflow UI command | `python -m src.models.train --show-mlflow-command` |
| Generate labeled drift scenarios | `python -m src.data.drift_generator` |
| Run Evidently drift monitoring | `python -m src.data.monitoring` |
| Run robustness evaluation | See [Section 10](#10-model-robustness-under-drift) |
| Dry-run candidate validation | `python -m src.models.promote --run-id <FULL_EXECUTION_ID>` |
| Explicitly promote a candidate | `python -m src.models.promote --run-id <FULL_EXECUTION_ID> --confirm` |
| Inspect DVC metadata/status | `dvc status` |
| Publish a new reference to DVC (authorized owner) | `dvc add models/reference && dvc push` |

**Different meanings of `--config`:** `scripts.run_preprocessing --config baseline` selects a named preprocessing variant defined in Python; `src.models.train --config configs/config.yml` selects a **training YAML file path**. The shared orchestrator has no `--config` forwarding flag; it uses the committed default training YAML.

**Comparison behavior:** `scripts.compare_experiments` defaults to `--mode full`. Without `--execution-id`, it includes eligible candidates from all historical full runs; the full orchestrator always supplies the newly created execution ID so its exported report is scoped correctly.

## 14. Scope, limitations, and next steps

### Implemented engineering controls and rubric coverage

- **Git/DVC:** Version-controlled source, documentation, YAML, DVC pointers, raw dataset, and approved reference model in public-read S3.
- **Configurable training:** Four model families, three preprocessing variants, YAML-fixed and swept hyperparameters, deterministic validation-based ranking, and configuration/data/model SHA-256 fingerprints.
- **MLflow:** Logged parameters, metrics, fitted sklearn models, run execution IDs, and a **`mlflow.search_runs()` comparison script**. The latest full local orchestration compared all 54 candidates from the same execution.
- **Testing:** Preprocessing tests, **five real-dataset validation tests**, **three lightweight model-validation tests**, comparison tests, and integration safeguards; **476 passed, 11 dependency warnings** in the latest recorded local complete run.
- **Automatic two-job CI:** `test` then dependent `train`, with DVC data pulls, quick-mode integration training, and an **F1 ≥ 0.65** validation threshold. Recent successful CI execution reported by the owner.
- **Monitoring:** Evidently drift reports, configurable NORMAL/WARNING/CRITICAL thresholds, and explicit written drift analysis with operational recommendations in Section 9.4.
- **Robustness:** Labeled synthetic stress tests with F1 and ranking-metric degradation compared to the existing promoted reference.
- **Governance:** Separated candidate training, validation and manual promotion; DVC-backed promoted reference and integrity checks.
- **Shared orchestration:** Local full-mode execution verified with tests, training, execution-scoped MLflow comparison, synthetic drift monitoring, and reference-model robustness. GitHub's manually dispatched orchestration uses the same Python runner.

### Limitations

1. **No production serving or deployment:** The implementation is the foundation for production operations, not a deployed prediction endpoint or online inference system.
2. **Limited model search:** The current YAML enumerates 54 candidate configurations; there is no Optuna, randomized optimization, or nested/group cross-validation.
3. **Synthetic drift:** Drifted datasets are deliberately altered test data, not observed deployment traffic. The measured robustness degradation does not establish live production risk or trigger automatic retraining.
4. **Local experiment history:** The MLflow SQLite database and original run artifacts are not automatically distributed with a Git clone. Old candidate runs may lack complete execution-ID tags.
5. **Candidate/reference separation:** The latest full-run model is not automatically promoted. File replacements during explicit promotion are not transactionally atomic across the model and metadata files.
6. **Fairness and external validity:** Adult Income contains sensitive demographic features and historical socioeconomic patterns. No production fairness certification or domain-transfer assessment was performed.
7. **Cross-platform limits:** Local full orchestration was verified on Windows 11. Earlier quick/full manual orchestration completed on Ubuntu, but the latest execution-scoped comparison integration still needs an updated full remote run. Reproducibility does not guarantee byte-identical artifacts across environments.
8. **Temporary GitHub Actions artifacts:** The workflow retains uploaded outputs for 14 days, and the run-local MLflow database is not automatically persisted in Git/DVC.
9. **DVC restoration behavior:** The tested workflow uses explicit raw/reference `.dvc` targets because an earlier untargeted pull did not restore the reference on a runner; the underlying cause was not conclusively established.
10. **Constrained YAML flexibility:** The loader accepts four specific sklearn model families and `selection_metric: val_f1`. New algorithms or metrics require code changes.
11. **CI metric provenance:** The CI performance gate queries MLflow runs and should remain tied to the current trainer's tracking store when storage/caching changes; execution filtering would make the gate more robust against historical runs.
12. **Workflow synchronization:** The last provided manual GitHub workflow includes a now-redundant second full-mode comparison step. Remove that step to preserve the execution-ID-filtered report produced by the Python orchestrator.

### Final maintenance and submission checks

- **Synchronize the manual GitHub workflow:** remove only the redundant standalone `compare_experiments.py` invocation; continue uploading `reports/mlflow_comparison.csv` and existing drift/robustness/training artifacts.
- **Verify the newest revision remotely:** run manual quick/full GitHub Actions workflows after synchronization. The automatic two-job CI already passed following the rubric-specific changes, according to the owner.
- **Update the technical PDF:** it predates the recent YAML, comparison, test/CI, and latest 476-test full orchestration changes.
- **Confirm dependency pins in `requirements.txt`:** verify PyYAML, scikit-learn, MLflow, `skops`, and DVC/S3 dependencies in a clean installation; do not rely solely on packages installed on the local development machine.
- **Keep submission evidence:** repository URL, green CI run, successful manual orchestration run, experiment comparison, drift HTML reports, robustness findings, and explicit drift analysis.
- **Preserve approval boundaries:** no automated candidate promotion or unauthorized DVC/S3 publishing during tests or orchestration.

No production model-serving API or automatic model deployment is included in this project.

---

**Dataset attribution:** UCI Machine Learning Repository, Adult (Census Income) dataset. Refer to the [dataset record](https://archive.ics.uci.edu/dataset/2/adult) for source details and citation guidance.
