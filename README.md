# Sprint 17 — End-to-End MLOps Pipeline

**Reproducible machine learning workflow for the UCI Adult Income classification dataset.** This educational project focuses on the *lifecycle* of a model—not merely fitting a classifier—including versioned data, reproducible preprocessing, experiment tracking, isolated training executions, explicit model promotion, drift detection, robustness evaluation, and automated tests.

> **Project status (November 2026):** Core modeling, explicit promotion, DVC/S3 storage, monitoring, robustness evaluation, automated tests, and GitHub Actions orchestration are implemented. The shared Python orchestrator has passed a local Windows quick-mode run and **both quick-mode and full-mode GitHub Actions runs**. The full remote execution completed **54/54 training candidates**. There is no production prediction API or automatic deployment.

## Contents

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

## Technical Report

The complete project methodology, architecture,
experimental results, drift monitoring, robustness
evaluation, and conclusions are documented in:

#### __[Sprint 17 — Final Technical Report](reports/Sprint_17_Final_Technical_Report.pdf)__
---

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
| Primary selection metric | Validation F1 for the positive class |
| Selected preprocessing variant | `baseline` |
| Selected model family | `HistGradientBoostingClassifier` |
| Selected validation F1 | 0.7337 |
| Selected held-out test F1 | 0.7150 |
| Latest local test run | **451 passed, 11 warnings** (November 2026) |
| Remote orchestration | **Quick and full GitHub Actions runs successful** |
| Verified full-run candidates | **54 / 54 successful** |

**Core tools:** Python, pandas, NumPy, scikit-learn, pytest, Git/GitHub, DVC with an Amazon S3 remote, MLflow (SQLite tracking), joblib, Evidently, and GitHub Actions.

### Implementation status

| Capability | Status |
|---|---|
| Git source control and DVC-backed raw data | Implemented |
| Public-read S3 downloads for DVC artifacts | Implemented |
| EDA and preprocessing variants | Implemented |
| Automated preprocessing, training, monitoring, and robustness tests | Implemented |
| MLflow experiment tracking and candidate isolation | Implemented |
| Explicit, validated model promotion | Implemented |
| DVC-versioned promoted reference model and metadata | Implemented |
| Synthetic drift scenarios and Evidently monitoring | Implemented |
| Robustness evaluation under drift | Implemented |
| GitHub Actions **CI** workflow (`ci.yml`) | Implemented |
| Shared local Python orchestration (`scripts/run_pipeline.py`) | Quick mode verified on Windows |
| GitHub Actions workflow invoking the shared orchestrator | Quick and full modes verified on Ubuntu |
| Full-mode candidate sweep | 54/54 candidates successful in GitHub Actions |
| GitHub Actions evaluation artifacts | Training, drift, and robustness reports inspected |
| Production model-serving endpoint / automatic deployment | Out of scope |

## 2. Architecture and responsibilities

The project separates **source control**, **artifact versioning**, **experiment tracking**, and **workflow execution**. These systems solve different problems and are not interchangeable.

```text
Git / GitHub                 DVC + public-read S3           MLflow
  code, tests, workflows       raw data, reference model     local experiment tracking
          |                              |                           |
          +------------------------------+---------------------------+
                                         |
                          scripts/run_pipeline.py
                              (shared orchestrator)
                                         |
                     optional DVC pull -> pytest tests
                                         |
                        preprocessing (three variants)
                                         |
                        quick or full candidate training
                                         |
                           synthetic drift generation
                                         |
                         Evidently drift monitoring
                                         |
                       reference model robustness checks
                                         |
                       generated reports / run artifacts

Local: python -m scripts.run_pipeline --quick --pull-dvc
GitHub Actions: orchestration workflow -> run_pipeline.py
Promotion: separate, explicit promote.py action; never automatic
```

**Division of responsibility:**

- **Git:** versions code, documentation, tests, configuration, and `.dvc` pointer files.
- **DVC:** versions selected large/binary artifacts and retrieves their exact contents from S3. In this project, DVC is **not** the workflow orchestrator.
- **MLflow:** records candidate experiments and associated metrics/artifacts. Local MLflow history is not automatically shared with GitHub clones.
- **Python modules:** implement preprocessing, model training, drift detection, robustness evaluation, and promotion.
- **`scripts/run_pipeline.py`:** orchestrates tests, preprocessing, training, drift generation, monitoring, and robustness evaluation on any supported machine.
- **GitHub Actions:** runs CI separately and provides a manually dispatched orchestration workflow that provisions an Ubuntu runner, restores DVC artifacts, invokes the Python orchestrator, and uploads results. It does not promote models.

### Main project paths

```text
.github/workflows/ci.yml       Continuous-integration workflow
.github/workflows/orchestration_pipeline.yml  Manually dispatched pipeline workflow
scripts/run_pipeline.py       Shared local / GitHub Actions orchestrator
src/data/drift_generator.py   Synthetic drift scenario generator
scripts/run_preprocessing.py   Preprocessing entry point
src/data/preprocess.py         Loading, splitting, transformations
src/data/monitoring.py         Evidently drift detection
src/models/train.py            Model sweep and MLflow logging
src/models/promote.py          Explicit candidate validation/promotion
src/models/robustness.py       Evaluation under synthetic drift
tests/                         pytest test suite
data/raw/                      DVC-retrieved Adult data
data/processed/<variant>/      Generated training/validation/test partitions
data/drift/<scenario>/         Generated control/mild/moderate/severe datasets
models/preprocessors/          Fitted preprocessing artifacts
models/runs/<execution_id>/    Isolated experiment output
models/reference/              DVC-tracked promoted model and metadata
models/reference.dvc           Git-tracked pointer for reference directory
reports/drift/                 Generated drift reports
reports/robustness/            Generated robustness reports (ignored by Git)
mlflow.db                      Local MLflow SQLite tracking store
```

Generated paths may not exist in a fresh clone until their corresponding stage runs.

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

The last command ensures that the DVC installation supports S3. Dependency availability and exact behavior may differ across platforms.

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

The last recorded complete local run passed **451 tests with 11 third-party deprecation warnings**. This is a historical result, not a guarantee of the same count or runtime on another checkout.

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

The standard configuration uses `random_state=42`.

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
2. **Generate model candidates:** Construct predefined algorithm and hyperparameter combinations.
3. **Fit candidates:** Train each candidate using its training partition.
4. **Generate validation probabilities:** Estimate positive-class probabilities on validation observations.
5. **Optimize decision thresholds:** Find the threshold maximizing validation F1 for each candidate.
6. **Evaluate and rank candidates:** Calculate validation metrics and select the strongest configuration within each variant.
7. **Evaluate variant winners:** In full mode, evaluate each variant winner using its held-out test partition.
8. **Select the preferred candidate:** Compare variant winners using validation results, not test performance.
9. **Save artifacts:** Serialize the selected fitted pipeline, metadata, experiment summaries, and leaderboards.
10. **Track experiments:** Record candidate parameters, metrics, and related artifacts in MLflow.

Training and promotion are deliberately separate operations. A successful training execution does not automatically replace the reference model.

### 6.2 Model Families and Hyperparameter Search

Four classification algorithms were selected to provide methodological diversity:

| Algorithm | Main characteristics | Role in experiment |
|---|---|---|
| Logistic Regression | Linear, interpretable probabilistic classifier | Establishes a simpler baseline |
| Random Forest | Ensemble of independently constructed decision trees | Captures nonlinear relationships and interactions |
| HistGradientBoosting | Sequentially constructed boosted trees using histogram-based splits | Evaluates gradient boosting on structured tabular data |
| Extra Trees | Ensemble of highly randomized decision trees | Tests whether additional tree randomization improves generalization |

The experiment uses a predefined Cartesian-product hyperparameter grid implemented directly in Python.

#### Hyperparameter configurations

| Algorithm | Hyperparameters explored | Combinations per variant |
|---|---|---:|
| Logistic Regression | `C`: 0.1, 1.0, 10.0; `class_weight`: None, balanced | 6 |
| Random Forest | `max_depth`: None, 15; `min_samples_leaf`: 1, 5 | 4 |
| HistGradientBoosting | `learning_rate`: 0.05, 0.1; `max_leaf_nodes`: 15, 31 | 4 |
| Extra Trees | `max_depth`: None, 15; `min_samples_leaf`: 1, 5 | 4 |
| **Total** | | **18** |

Additional fixed settings include:

- Logistic Regression: `solver="lbfgs"`, `max_iter=5000`.
- Random Forest: `n_estimators=150`.
- HistGradientBoosting: `max_iter=150`.
- Extra Trees: `n_estimators=150`.
- Reproducibility seed: `42`.

Logistic Regression includes an additional `StandardScaler` fitted exclusively on the training partition.

The search is repeated across the three preprocessing variants:

| Variant | Candidates |
|---|---:|
| Baseline | 18 |
| Robust | 18 |
| Interactions | 18 |
| **Full experiment** | **54** |

This is a manually defined grid search, not an automated `GridSearchCV`, randomized search, or Bayesian optimization procedure.

The purpose is to compare distinct modeling approaches and validate experiment management rather than search the entire hyperparameter space.

### 6.3 Quick and Full Execution Modes

The training module supports two execution modes.

**Quick mode** is intended for development, testing, and orchestration verification:

```bash
python -m src.models.train --quick
```

It trains one configuration from each model family across all three preprocessing variants, producing 12 candidate evaluations.

Quick mode records validation results but does not evaluate held-out test data or produce a promotable model candidate.

**Full mode** executes the complete experiment:

```bash
python -m src.models.train
```

It evaluates 54 candidates, selects the best configuration within each preprocessing variant, evaluates the three winners on their corresponding test partitions, and saves the preferred candidate with its metadata.

The completed GitHub Actions full execution successfully evaluated **54 of 54 candidates**.

### 6.4 Validation, Model Ranking, and Decision Threshold Optimization

The project uses **positive-class F1** as its primary model-selection metric.

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

The secondary metrics serve as tie-breakers.

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

The completed full experiment selected `HistGradientBoostingClassifier` as the strongest model family within all three preprocessing variants.

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

MLflow provides experiment tracking for the training pipeline.

Each candidate evaluation is associated with an MLflow run containing information such as:

- Model family and preprocessing variant.
- Hyperparameter configuration.
- Validation metrics.
- Optimized decision threshold.
- Experiment artifacts and diagnostic information.
- Run identifiers and execution metadata.

The project uses a local SQLite tracking database:

```text
mlflow.db
```

Experiment artifacts are stored in a separately configured local artifact directory.

To obtain the configured MLflow UI command:

```bash
python -m src.models.train --show-mlflow-command
```

Execute the printed command and open the displayed local address, commonly:

```text
http://127.0.0.1:5000
```

MLflow makes it possible to inspect and compare candidate runs without relying exclusively on console output or manually maintained experiment tables.

**Reproducibility limitation:** The local MLflow database and all historical execution artifacts are not automatically distributed with a fresh GitHub clone. Reviewers can regenerate their own experiment history by running the training pipeline.

### 6.7 Model Artifacts, Isolation, and Reproducibility

Each training invocation creates an isolated execution directory identified by a unique 32-character hexadecimal `execution_id`.

The generated structure is:

```text
models/runs/<execution_id>/
    sweep_leaderboard.csv
    training_summary.json
    best_model_pipeline.joblib
    best_model_metadata.json
    baseline/
    robust/
    interactions/
```

The full-execution directory contains:

- **`sweep_leaderboard.csv`:** Candidate-level performance information for comparing configurations.
- **`training_summary.json`:** Execution status, candidate counts, selected winners, and evaluation results.
- **`best_model_pipeline.joblib`:** Serialized preferred model with its fitted preprocessing pipeline.
- **`best_model_metadata.json`:** Model configuration, decision threshold, performance, and provenance information.
- **Variant directories:** Additional variant-specific experiment outputs.

Quick executions instead produce `quick_training_summary.json` and do not save a promotable candidate.

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
- Optimized decision threshold.

These values support artifact traceability and integrity verification.

For the verified full GitHub Actions execution:

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

SHA-256 verification establishes file integrity relative to the recorded fingerprint; it does not independently guarantee model correctness or bit-for-bit reproducibility across different environments.

**Training isolation:** Newly generated models remain in `models/runs/<execution_id>/`. The training module does not overwrite the DVC-tracked reference model.

Candidate validation and explicit promotion are described in Section 7.

### 6.8 Model Optimization Scope and Future Improvements

**The primary objective of this project was to implement a complete, reproducible MLOps pipeline rather than maximize predictive performance.**

Model experimentation was deliberately limited to four algorithm families, three preprocessing variants, and 54 predefined candidates.

This was sufficient to demonstrate training automation, hyperparameter comparison, MLflow tracking, validation-based selection, threshold optimization, held-out evaluation, artifact serialization, and explicit model promotion.

The selected model therefore represents a functional experimental baseline rather than an exhaustively optimized classifier.

Potential improvements include:

1. **Targeted hyperparameter tuning:** Expand the search around HistGradientBoosting and the strongest preprocessing variants. Explore learning rate, tree complexity, regularization, minimum leaf size, and boosting iterations.

2. **Automated hyperparameter optimization:** Integrate Optuna, `RandomizedSearchCV`, or `GridSearchCV`. Optuna could explore larger, conditional parameter spaces using adaptive sampling and trial pruning. An exhaustive grid search would provide systematic coverage of a predefined parameter space at potentially greater computational cost.

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
- Record configurations, metrics, and artifacts in MLflow.
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

- **Git:** To track the DVC pointer file and associate model versions with repository revisions.
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
git add models/reference.dvc models/.gitignore
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
| Python source, tests, README, workflows | Git |
| Raw Adult data | DVC + S3 |
| Promoted `models/reference/` directory | DVC + S3; pointer in Git |
| Per-execution training candidates | Generated locally; not automatically promoted or uploaded |
| Processed datasets / fitted preprocessors | Generated by preprocessing; check individual DVC tracking before assuming persistence |
| Local MLflow SQLite database/artifacts | Local; not automatically shared |
| Drift and robustness reports | Generated locally; not automatically versioned |

**DVC does not orchestrate the stages.** The Python script `scripts/run_pipeline.py` defines stage order, and GitHub Actions provides remote execution infrastructure. There is no requirement to introduce `dvc.yaml` solely to use DVC for artifact versioning.

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

These values are from the inspected JSON and CSV robustness artifacts, each covering four scenarios of 9,764 observations. Severe drift reduced F1 by approximately **22.7% relative to control**. The robustness report references the **promoted DVC reference model** (SHA-256 prefix `d5254757`), not the new unpromoted full-run candidate (SHA-256 prefix `f646ea46`). This distinction is intentional: orchestration never automatically promotes a candidate. The synthetic scenarios demonstrate stress-test sensitivity, not production performance. Reports are written under `reports/robustness/` and are not currently committed to Git.

## 11. Testing, CI, and pipeline orchestration

### 11.1 Run tests locally

```bash
# Full suite
python -m pytest tests/ -q

# Individual modules
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_train.py -v
python -m pytest tests/test_promote.py -v
python -m pytest tests/test_monitoring.py -v
python -m pytest tests/test_robustness.py -v
```

Test coverage includes preprocessing behavior and duplicate preservation; model training and selection; candidate validation and promotion safeguards; drift monitoring; synthetic drift generation; and robustness evaluation.

The latest reported full run was:

```text
451 passed, 11 warnings in 123.17s
```

The warnings were deprecation notices from third-party dependencies (Evidently/NumPy, Litestar, and MLflow/SQLAlchemy). Passing tests do not prove absence of all data leakage, fairness issues, or deployment failures.

### 11.2 Continuous integration (CI)

`.github/workflows/ci.yml` handles automated code checks. A separate `retrain.yml` was intentionally removed. CI is distinct from end-to-end pipeline execution: it does not imply model deployment.

### 11.3 Shared pipeline orchestration

`scripts/run_pipeline.py` is the single source of truth for execution order. It runs these stages sequentially:

1. Optional explicit DVC pull (`--pull-dvc`).
2. `python -m pytest tests/ -q`.
3. `python -m scripts.run_preprocessing`.
4. `python -m src.models.train --quick` or `python -m src.models.train`.
5. `python -m src.data.drift_generator`.
6. `python -m src.data.monitoring`.
7. `python -m src.models.robustness`.

An unexpected stage failure stops execution. Monitoring exit code `1` is treated as an expected critical-drift alert for the deliberately severe synthetic scenario; other unexpected failures are not ignored. No stage promotes the reference model.

### 11.4 GitHub Actions orchestration

The manually dispatched `.github/workflows/orchestration_pipeline.yml` exposes a `training_mode` choice (`quick` or `full`). Its infrastructure responsibilities are repository checkout, Python 3.11 and dependency setup, explicit DVC restoration, reference artifact verification, and artifact upload. It invokes the shared orchestrator without `--pull-dvc` because the workflow has already restored DVC artifacts:

```bash
python -m scripts.run_pipeline --quick
# or
python -m scripts.run_pipeline --full
```

The workflow uploads `reports/drift/`, `reports/robustness/`, `data/drift/drift_metadata.json`, and `models/runs/` as GitHub Actions artifacts (14-day retention). The GitHub Actions run summary records training mode, job result, reference-promotion policy, and commit. Drift details are available in logs and reports rather than a dedicated workflow summary field.

**Verification status (November 2026):** The shared Python orchestrator completed a local Windows quick-mode run. The GitHub Actions orchestration workflow subsequently completed **both quick and full runs successfully** on Ubuntu. The full run trained **54/54 candidates**; its training summary and candidate metadata were inspected. Drift monitoring and reference-model robustness reports from the run were also inspected. This verifies execution of the workflow, not automatic model promotion or production deployment.

## 12. Reproducing the full workflow

### 12.1 One-command local orchestration

After cloning the repository, activating Python 3.11, and installing dependencies as described in Section 3, run from the project root:

```bash
# Complete quick-mode workflow, including DVC retrieval
python -m scripts.run_pipeline --quick --pull-dvc

# Full candidate search, including DVC retrieval
python -m scripts.run_pipeline --full --pull-dvc

# Skip DVC retrieval when required artifacts already exist locally
python -m scripts.run_pipeline --quick
```

`--quick` is the default training mode; `--full` runs the larger candidate search. `--pull-dvc` restores both the raw dataset and reference model. The orchestrator verifies that the reference model and metadata are present, executes tests and all ML stages, then prints a summary. A critical synthetic drift alert is expected and does not automatically fail the overall pipeline.

**Recorded local quick-mode result (Windows 11, November 17, 2026):** all seven stages completed; 451 tests passed with 11 dependency warnings; 12 training candidates were evaluated; the severe drift scenario produced a CRITICAL alert; robustness evaluation completed; reference promotion remained disabled. Total recorded stage duration was approximately 229 seconds.

### 12.2 GitHub Actions execution

In the repository's **Actions** tab, select **MLOps Pipeline Orchestration**, choose **Run workflow**, select `quick` or `full`, and start the run. The workflow uses an Ubuntu 24.04 runner, explicitly restores DVC artifacts, and invokes the same Python orchestrator used locally. Download generated reports from the run's uploaded artifacts.

Both **quick** and **full** GitHub Actions orchestration runs have completed successfully using the shared Python orchestrator. The full run evaluated all 54 candidates. The workflow uploads generated training, drift, and robustness artifacts for reviewer inspection; it does **not** update the DVC reference model or publish new S3 objects.

### 12.3 Optional candidate review and promotion

Full training generates an isolated candidate under `models/runs/<execution_id>/`. Promotion is deliberately **not** part of orchestration. After reviewing the results, validate the candidate:

```bash
python -m src.models.promote --run-id <FULL_EXECUTION_ID>
```

If promotion is explicitly approved, update the local reference model:

```bash
python -m src.models.promote --run-id <FULL_EXECUTION_ID> --confirm
```

**Optional — authorized repository owner only:**

```bash
dvc add models/reference
dvc push
```

The `dvc push` command requires authorized AWS write permissions and will fail for reviewers without those permissions when new objects must be uploaded. This restriction is intentional to protect the owner's S3 storage from unauthorized modifications and uncontrolled upload costs.

Local promotion does not require S3 write access. When publishing a new reference version, the updated `models/reference.dvc` pointer must also be committed and pushed to Git (see Section 7).

### 12.4 Reviewing execution evidence

From the repository's **Actions** tab, open a successful **MLOps Pipeline Orchestration** run. Check the selected `training_mode`, job logs, and uploaded artifacts at the bottom of the run page. For a full run, inspect `models/runs/<execution_id>/training_summary.json` and `best_model_metadata.json`; compare the reported candidate counts, validation selection, and model SHA-256. Inspect `reports/drift/` for Evidently output and `reports/robustness/` for labeled stress-test results. The latest inspected full-run execution ID is `ef47ebedf90145199cba3ead91314b90`.

**Artifact policy:** GitHub Actions outputs are evidence of a specific execution and expire according to retention settings. Git/DVC pointers and the S3-backed reference model are the persistent versioning mechanism. The unpromoted candidate and the promoted reference model are intentionally different artifacts.

### Reproducibility boundaries

- Fixed seeds and DVC pointers help reproduce inputs and processing but do not guarantee bit-for-bit identical models across platforms or library versions.
- MLflow run IDs and training execution IDs are generated anew on each run.
- Full training can be substantially more expensive than quick mode.
- Local MLflow history is not automatically shared by cloning the repository.
- The reference model is an explicitly promoted snapshot, not automatically the most recently trained candidate.
- The orchestrator is a sequential local/CI runner, not a production deployment or distributed scheduling service.

## 13. Command reference

| Task | Command |
|---|---|
| Download DVC-managed data and model | `dvc pull data/raw.dvc models/reference.dvc` |
| Run local orchestrator (quick, with DVC) | `python -m scripts.run_pipeline --quick --pull-dvc` |
| Run local orchestrator (full, with DVC) | `python -m scripts.run_pipeline --full --pull-dvc` |
| Run local orchestrator (existing DVC artifacts) | `python -m scripts.run_pipeline --quick` |
| Generate drift scenarios | `python -m src.data.drift_generator` |
| Check DVC status | `dvc status` |
| Run all tests | `python -m pytest tests/ -q` |
| Run preprocessing | `python -m scripts.run_preprocessing` |
| Run one preprocessing variant | `python -m scripts.run_preprocessing --config baseline` |
| Quick training | `python -m src.models.train --quick` |
| Full training | `python -m src.models.train` |
| Display MLflow launch command | `python -m src.models.train --show-mlflow-command` |
| Validate candidate | `python -m src.models.promote --run-id <ID>` |
| Promote candidate | `python -m src.models.promote --run-id <ID> --confirm` |
| Generate drift monitoring report | `python -m src.data.monitoring` |
| Evaluate robustness | See [Section 10](#10-model-robustness-under-drift) |
| Update reference DVC pointer | `dvc add models/reference` |
| Upload DVC objects (authorized writers) | `dvc push` |

## 14. Scope, limitations, and next steps

### Implemented engineering controls

- Separate training, validation, and test responsibilities.
- Explicit candidate selection by validation metrics.
- SHA-256 and provenance checks for model promotion.
- Manual confirmation before reference replacement.
- DVC versioning of the promoted model **together with its metadata**.
- Drift monitoring and labeled robustness stress tests.
- Automated pytest coverage and existing GitHub Actions CI.
- Shared Python pipeline orchestration, verified locally in quick mode.
- Successful local Windows quick-mode orchestration and successful GitHub Actions quick- and full-mode orchestration using the shared Python runner.

### Limitations

1. **No production serving or deployment.** The project is an educational MLOps implementation, not a live prediction service.
2. **Limited model search.** The 54-candidate sweep is deliberately constrained; it is not exhaustive optimization.
3. **Synthetic drift.** Stress scenarios are controlled perturbations, not representative evidence of production drift.
4. **Local experiment history.** MLflow tracking data and isolated run artifacts are not automatically distributed with the Git repository.
5. **Promotion replacement is not transactionally atomic across both files.** Exception rollback is implemented, but abrupt interruption requires manual verification.
6. **No demonstrated production fairness or external validity.** Adult Income includes demographic attributes, historical biases, and substantial domain limitations. The model must not be used for consequential decisions without separate fairness, governance, privacy, and validity assessments.
7. **Cross-platform reproducibility is not bit-for-bit guaranteed.** Quick mode was verified locally on Windows; quick and full modes were verified on Ubuntu in GitHub Actions. Full mode has not been independently verified on every supported operating system.
8. **GitHub Actions artifacts are temporary.** The configured retention period is 14 days; preserve any evidence needed beyond that window.
9. **Explicit DVC targets are used in the tested workflow.** The cause of an earlier untargeted `dvc pull` omission of the reference directory has not been conclusively identified.

### Remaining verification and maintenance

- Preserve the successful GitHub Actions run links, training summary, drift reports, and robustness reports as submission evidence.
- Optionally validate full-mode orchestration locally or on another operating system for additional portability evidence.
- Investigate why an earlier untargeted `dvc pull` did not restore the reference directory, while explicit targets did.
- Review action version deprecation notices and update individual GitHub Actions dependencies only when compatible.
- Preserve the boundary between **automated** testing, training, drift analysis, and reporting and **manual/authorized** reference promotion and DVC publishing.

No production model-serving API or automatic model deployment is included in this project.

---

**Dataset attribution:** UCI Machine Learning Repository, Adult (Census Income) dataset. Refer to the [dataset record](https://archive.ics.uci.edu/dataset/2/adult) for source details and citation guidance.
