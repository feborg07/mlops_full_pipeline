# Sprint 17 — End-to-End MLOps Pipeline

**Reproducible machine learning workflow for the UCI Adult Income classification dataset.** This educational project focuses on the *lifecycle* of a model—not merely fitting a classifier—including versioned data, reproducible preprocessing, experiment tracking, isolated training executions, explicit model promotion, drift detection, robustness evaluation, and automated tests.

> **Project status (November 2026):** Core modeling, promotion, monitoring, robustness evaluation, DVC storage, and a GitHub Actions CI workflow are implemented. **The multi-stage pipeline has passed both an earlier GitHub Actions quick-mode run and a local Windows quick-mode run using `scripts/run_pipeline.py`. The revised GitHub Actions workflow that invokes the shared Python orchestrator still requires a verification run.** There is no production prediction API or automatic deployment.

## Contents

1. [Project at a glance](#1-project-at-a-glance)
2. [Architecture and responsibilities](#2-architecture-and-responsibilities)
3. [Installation and reviewer quick start](#3-installation-and-reviewer-quick-start)
4. [Dataset and exploratory analysis](#4-dataset-and-exploratory-analysis)
5. [Preprocessing](#5-preprocessing)
6. [Training and MLflow](#6-training-and-mlflow)
7. [Candidate validation and explicit promotion](#7-candidate-validation-and-explicit-promotion)
8. [DVC and artifact versioning](#8-dvc-and-artifact-versioning)
9. [Data drift monitoring](#9-data-drift-monitoring)
10. [Model robustness under drift](#10-model-robustness-under-drift)
11. [Testing, CI, and pipeline orchestration](#11-testing-ci-and-pipeline-orchestration)
12. [Reproducing the full workflow](#12-reproducing-the-full-workflow)
13. [Command reference](#13-command-reference)
14. [Scope, limitations, and next steps](#14-scope-limitations-and-next-steps)

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
| GitHub Actions end-to-end orchestration (earlier stage-by-stage workflow) | Quick mode verified |
| Shared local Python orchestration (`scripts/run_pipeline.py`) | Quick mode verified on Windows |
| Updated GitHub Actions workflow calling the Python orchestrator | Configured; verification run pending |
| Full-mode orchestration across environments | Not yet verified |
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
GitHub Actions: orchestration_pipeline.yml -> run_pipeline.py
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
git clone <ACTUAL_REPOSITORY_URL>
cd mlops-final-project
```

**Maintainer action:** Replace `<ACTUAL_REPOSITORY_URL>` with the real repository URL before publishing this README. The previous README contained an unverified placeholder repository address.

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

## 4. Dataset and exploratory analysis

### 4.1 Source and target

The project combines UCI's original `adult.data` (32,561 rows) and `adult.test` (16,281 rows) into a single dataset of **48,842 rows**. The first line of `adult.test` is skipped, trailing periods are removed from its income labels, and missing categorical values marked `?` are interpreted as missing.

The original UCI train/test division is **not** reused. The project creates its own partitions for training, validation, and held-out evaluation.

The original predictors are `age`, `workclass`, `fnlwgt`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, and `native-country`.

### 4.2 Key EDA findings

- The target is imbalanced: approximately **76.1%** of records are `<=50K` and **23.9%** are `>50K`.
- Missingness is concentrated in `workclass` (~5.73%), `occupation` (~5.75%), and `native-country` (~1.75%).
- `capital-gain` and `capital-loss` are strongly zero-inflated, motivating logarithmic transformations and positive-value indicators.
- `education-num`, `hours-per-week`, and `age` showed meaningful relationships with the target in the initial exploratory analysis.
- Exact duplicate rows are **retained**. Identical demographic/financial records can represent distinct people; duplicate-aware partitioning is intended to reduce cross-partition leakage.

The dataset contains sensitive demographic variables and represents historical census data. Predictive associations are **not causal findings** and should not be interpreted as a basis for consequential decisions about individuals.

## 5. Preprocessing

### 5.1 Entry point

```bash
# Generate all supported variants
python -m scripts.run_preprocessing

# Generate a single variant
python -m scripts.run_preprocessing --config baseline
python -m scripts.run_preprocessing --config robust
python -m scripts.run_preprocessing --config interactions
```

The runner reads the two raw input files, validates and combines them, creates partitions, fits transformations on training data, and writes processed artifacts.

### 5.2 Variants

| Variant | Main behavior |
|---|---|
| `baseline` | Median numerical imputation, standard scaling, categorical encoding, `log1p` capital features and positive-capital indicators |
| `robust` | Robust scaling; retains original and logarithmic capital representations |
| `interactions` | Baseline transformations plus engineered interaction features |

Categorical missing values are represented as `Unknown`. The textual `education` column is excluded from the baseline model representation while `education-num` is retained. Categorical features use one-hot encoding with handling for infrequent/unseen categories.

### 5.3 Partitioning and leakage controls

The project creates **train, validation, and test** partitions with a fixed random seed and class stratification. Duplicate records are retained, with duplicate-aware splitting logic to limit leakage between partitions.

- **Training:** fit preprocessing parameters and classifiers.
- **Validation:** compare candidates and optimize decision thresholds.
- **Test:** evaluate selected configuration winners without using test metrics to choose the preferred configuration.

Learned transformations must be fitted on training data only. The training module consumes the saved partitions rather than silently recreating them.

### 5.4 Generated files

For each variant, the preprocessing runner produces:

```text
data/processed/<variant>/
    X_train.csv      y_train.csv
    X_val.csv        y_val.csv
    X_test.csv       y_test.csv
    preprocessing_config.json

models/preprocessors/<variant>.joblib
```

Processed datasets and fitted transformers are generated artifacts; do not assume that all of them are DVC-tracked or committed to Git. Re-run preprocessing when they are absent.

## 6. Training and MLflow

### 6.1 Candidate families and search

The training module evaluates four model families across three preprocessing variants:

| Family | Full candidates per variant |
|---|---:|
| Logistic Regression | 6 |
| Random Forest | 4 |
| HistGradientBoosting | 4 |
| Extra Trees | 4 |
| **Total per variant** | **18** |

A full execution evaluates **54 candidates** across all variants. Logistic Regression uses an additional model-specific scaler fitted on training data.

### 6.2 Quick versus full execution

```bash
# Development/smoke test: 12 candidates, validation only
python -m src.models.train --quick

# Full experiment: 54 candidates, test evaluation of winners,
# and a promotable candidate model
python -m src.models.train
```

**Quick mode** trains one configuration from each model family for each preprocessing variant. It produces validation results and a quick summary, but **does not produce a promotable reference candidate** and does not evaluate the held-out test partitions.

**Full mode** evaluates all 54 candidates, selects a winner for each variant using validation results, evaluates those winners on the held-out test sets, and saves an isolated candidate and its metadata. Training does **not** automatically replace the promoted reference model.

### 6.3 Selection policy

The primary selection criterion is **positive-class validation F1**, with average precision (PR-AUC) and ROC-AUC available as additional ranking information. The decision threshold is optimized on validation predictions, not the test set.

The preferred preprocessing configuration is selected using **validation results**, even if another variant has a slightly higher test F1. This avoids using the held-out test set for model selection.

### 6.4 Recorded results

The completed 54-candidate execution selected HistGradientBoosting for all three variants:

| Variant | Validation F1 | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|
| **Baseline (selected)** | **0.7337** | 0.7150 | **0.9247** |
| Robust | **0.7337** | 0.7150 | **0.9247** |
| Interactions | 0.7314 | **0.7192** | 0.9240 |

The selected baseline candidate uses a validation-optimized threshold of approximately **0.3713**. These are recorded results for a particular execution, not universal performance guarantees.

### 6.5 Isolated execution artifacts

Each invocation receives a distinct 32-character hexadecimal `execution_id`:

```text
models/runs/<execution_id>/
    sweep_leaderboard.csv
    training_summary.json              # full execution
    quick_training_summary.json        # quick execution
    best_model_pipeline.joblib         # full execution only
    best_model_metadata.json           # full execution only
    baseline/...
    robust/...
    interactions/...
```

The full-training metadata records the chosen model family, preprocessing variant, decision threshold, MLflow run ID, execution ID, input-dataset hashes, preprocessing hash, and model SHA-256. The training summary records the execution's candidate counts, completion status, winners, and evaluation results.

**Example full execution:** `ad7990374a3f453b9d879498349f714f`. This identifier documents the recorded experiment; a reviewer running training will generate a **different** ID.

### 6.6 MLflow

The project uses a local SQLite tracking database (`mlflow.db`) and local logged artifacts. To obtain the configured UI command:

```bash
python -m src.models.train --show-mlflow-command
```

Run the printed command, then open the address it displays (commonly `http://127.0.0.1:5000`). MLflow allows comparison of parameters, metrics, convergence diagnostics, and experiment runs.

**Important:** A fresh clone does not automatically include the maintainer's local MLflow database or all isolated training executions. Re-run training to regenerate local experiment history.

## 7. Candidate validation and explicit promotion

Training and promotion are intentionally separate operations. A successful training run is **not** automatically the reference model.

### 7.1 Dry-run validation

```bash
python -m src.models.promote --run-id <FULL_EXECUTION_ID>
```

The dry run checks the candidate without modifying `models/reference/`. The validator checks, among other things:

- Matching candidate and summary execution IDs.
- Existence of `training_summary.json` with `quick: false`.
- Positive candidate count and all candidates reported successful.
- Agreement between the preferred variant and candidate metadata.
- Matching winning MLflow run ID and `FINISHED` status.
- Candidate SHA-256 integrity.
- A finite decision threshold in the valid range.
- A loadable, trusted classifier pipeline with `predict_proba()` and positive class `1`.

Example result:

```json
{
  "execution_id": "<FULL_EXECUTION_ID>",
  "candidate_sha256": "<SHA256>",
  "validated": true,
  "promoted": false
}
```

### 7.2 Explicit promotion

Only after reviewing the validation result:

```bash
python -m src.models.promote --run-id <FULL_EXECUTION_ID> --confirm
```

Promotion writes the selected candidate to:

```text
models/reference/
    best_model_pipeline.joblib
    best_model_metadata.json
```

Reference metadata additionally records promotion time, promotion source, and explicit promotion type. Source candidate files remain in their isolated run directory.

The implementation backs up existing reference artifacts and attempts rollback if an exception occurs during replacement. **The model and metadata are two files, so the replacement is not a single atomic transaction; abrupt process termination may leave an inconsistent pair.** Treat promotion as an explicitly controlled operation, not a fully transactional deployment.

### 7.3 Persist the promoted reference with DVC

After promotion:

```bash
dvc add models/reference
dvc push
git add models/reference.dvc models/.gitignore
git commit -m "feat: update promoted reference model"
git push
```

`models/reference.dvc` tracks the **directory as one DVC object**, including the model and its metadata. Git tracks the pointer, while S3 stores the artifact contents. DVC may report `Everything is up to date` when all required objects already exist remotely.

**Do not run `git add -f` on the binary model or commit it directly to Git.** If Git reports that the directory is already tracked, inspect existing tracking before changing the index; the project has already migrated from individual `.dvc` pointers to a directory-level pointer.

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

The recorded overall status was **CRITICAL** because the severe scenario crossed the configured threshold. Reports are generated under `reports/drift/`.

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

These values are from a recorded run; results depend on the reference model and the generated drift datasets. The severe scenario showed substantial F1 degradation, consistent with the monitoring alert. Reports are written under `reports/robustness/` and are not currently committed to Git.

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

**Verification status:** An earlier GitHub Actions quick-mode workflow completed successfully. The shared Python orchestrator separately completed locally on Windows 11 in quick mode with DVC pull. The *revised* GitHub Actions YAML that calls `run_pipeline.py` has not yet been verified by a reported successful run. Full-mode orchestration is also not yet verified.

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

The revised workflow still requires a successful GitHub Actions verification run after it is committed. The earlier successful GitHub Actions run used separate workflow steps rather than the shared Python entry point.

### 12.3 Optional candidate review and promotion

Full training generates an isolated candidate under `models/runs/<execution_id>/`. Promotion is deliberately **not** part of orchestration. After reviewing the results, validate the candidate:

```bash
python -m src.models.promote --run-id <FULL_EXECUTION_ID>
```

If promotion is explicitly approved:

```bash
python -m src.models.promote --run-id <FULL_EXECUTION_ID> --confirm
dvc add models/reference
dvc push
```

Writing to the shared S3 DVC remote requires authorized AWS write access. Reviewers can reproduce training and evaluate local candidates without those credentials. The Git-tracked `models/reference.dvc` pointer must also be committed and pushed when publishing a new reference version (see Section 7).

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
- An earlier GitHub Actions end-to-end quick-mode run, with a revised shared-script workflow awaiting verification.

### Limitations

1. **No production serving or deployment.** The project is an educational MLOps implementation, not a live prediction service.
2. **Limited model search.** The 54-candidate sweep is deliberately constrained; it is not exhaustive optimization.
3. **Synthetic drift.** Stress scenarios are controlled perturbations, not representative evidence of production drift.
4. **Local experiment history.** MLflow tracking data and isolated run artifacts are not automatically distributed with the Git repository.
5. **Promotion replacement is not transactionally atomic across both files.** Exception rollback is implemented, but abrupt interruption requires manual verification.
6. **No demonstrated production fairness or external validity.** Adult Income includes demographic attributes, historical biases, and substantial domain limitations. The model must not be used for consequential decisions without separate fairness, governance, privacy, and validity assessments.
7. **Revised GitHub Actions integration not yet reverified.** The earlier stage-by-stage GitHub Actions quick-mode workflow passed, but the updated YAML invoking the shared Python orchestrator still requires a new run.
8. **Full-mode orchestration not yet verified.** Quick-mode successes do not demonstrate the entire 54-candidate execution in both environments.
9. **Explicit DVC targets are currently required in the tested GitHub workflow.** The cause of the bare `dvc pull` omission of the reference model has not been conclusively identified.

### Remaining verification and maintenance

- Run the revised `orchestration_pipeline.yml` in GitHub Actions to confirm it successfully calls `scripts/run_pipeline.py`.
- Optionally validate full-mode orchestration on an appropriately provisioned runner or local machine.
- Investigate why an earlier untargeted `dvc pull` did not restore the reference directory, while explicit targets did.
- Review action version deprecation notices and update individual GitHub Actions dependencies only when compatible.
- Preserve the boundary between **automated** testing, training, drift analysis, and reporting and **manual/authorized** reference promotion and DVC publishing.

No production model-serving API or automatic model deployment is included in this project.

---

**Dataset attribution:** UCI Machine Learning Repository, Adult (Census Income) dataset. Refer to the [dataset record](https://archive.ics.uci.edu/dataset/2/adult) for source details and citation guidance.
