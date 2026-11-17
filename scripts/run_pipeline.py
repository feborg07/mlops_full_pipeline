
"""
Sprint 17 - MLOps Pipeline Orchestrator

Runs the complete ML workflow locally or through GitHub Actions.

Stages:
    1. Restore DVC artifacts (optional)
    2. Run automated tests
    3. Preprocess datasets
    4. Train candidate models
    5. Generate synthetic drift datasets
    6. Monitor data drift
    7. Evaluate reference model robustness

Usage:
    python -m scripts.run_pipeline --quick
    python -m scripts.run_pipeline --full
    python -m scripts.run_pipeline --quick --pull-dvc

Notes:
    - Reference model promotion is never automatic.
    - Critical drift is expected during the severe stress test.
    - All stages run sequentially.
    - The pipeline stops when a required stage fails.
"""

import argparse
import logging
import subprocess
import sys
import time

from pathlib import Path
from datetime import datetime, timezone


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOGGER = logging.getLogger("mlops_pipeline")


# ============================================================
# Logging
# ============================================================

def configure_logging() -> None:
    """Configure readable pipeline logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


# ============================================================
# Stage execution
# ============================================================

def run_stage(
    name: str,
    command: list[str],
    allowed_exit_codes: tuple[int, ...] = (0,),
) -> dict:
    """
    Execute a pipeline stage as a subprocess.

    Parameters
    ----------
    name:
        Human-readable stage name.

    command:
        Command and arguments to execute.

    allowed_exit_codes:
        Exit codes considered successful for this stage.

    Returns
    -------
    dict:
        Stage execution metadata.

    Raises
    ------
    RuntimeError:
        When the stage returns an unexpected exit code.
    """

    LOGGER.info("=" * 60)
    LOGGER.info("STARTING: %s", name)
    LOGGER.info("COMMAND: %s", " ".join(command))
    LOGGER.info("=" * 60)

    start = time.perf_counter()

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
        )

    except OSError as exc:
        raise RuntimeError(
            f"Unable to execute stage '{name}': {exc}"
        ) from exc

    duration = time.perf_counter() - start

    if result.returncode not in allowed_exit_codes:
        raise RuntimeError(
            f"Stage '{name}' failed with exit code "
            f"{result.returncode} after {duration:.2f}s."
        )

    LOGGER.info(
        "COMPLETED: %s | Exit code: %s | Duration: %.2fs",
        name,
        result.returncode,
        duration,
    )

    return {
        "stage": name,
        "exit_code": result.returncode,
        "duration_seconds": round(duration, 2),
    }


# ============================================================
# Pipeline
# ============================================================

def run_pipeline(
    training_mode: str = "quick",
    pull_dvc: bool = False,
) -> list[dict]:
    """
    Execute the complete MLOps pipeline.

    Parameters
    ----------
    training_mode:
        'quick' or 'full'.

    pull_dvc:
        Whether to retrieve DVC artifacts before execution.
    """

    if training_mode not in {"quick", "full"}:
        raise ValueError(
            "training_mode must be 'quick' or 'full'."
        )

    python = sys.executable

    results = []

    started_at = datetime.now(timezone.utc)

    LOGGER.info("SPRINT 17 - MLOPS PIPELINE")
    LOGGER.info("Training mode: %s", training_mode)
    LOGGER.info("Project root: %s", PROJECT_ROOT)
    LOGGER.info("Started (UTC): %s", started_at.isoformat())

    # --------------------------------------------------------
    # 1. Restore DVC artifacts
    # --------------------------------------------------------

    if pull_dvc:
        results.append(
            run_stage(
                "Restore DVC artifacts",
                [
                    "dvc",
                    "pull",
                    "data/raw.dvc",
                    "models/reference.dvc",
                ],
            )
        )

    # Reference artifacts must exist before evaluation.
    reference_files = [
        PROJECT_ROOT
        / "models/reference/best_model_pipeline.joblib",

        PROJECT_ROOT
        / "models/reference/best_model_metadata.json",
    ]

    for path in reference_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required reference artifact not found: {path}\n"
                "Run the pipeline with --pull-dvc."
            )

    # --------------------------------------------------------
    # 2. Automated tests
    # --------------------------------------------------------

    results.append(
        run_stage(
            "Automated tests",
            [
                python,
                "-m",
                "pytest",
                "tests/",
                "-q",
            ],
        )
    )

    # --------------------------------------------------------
    # 3. Preprocessing
    # --------------------------------------------------------

    results.append(
        run_stage(
            "Data preprocessing",
            [
                python,
                "-m",
                "scripts.run_preprocessing",
            ],
        )
    )

    # --------------------------------------------------------
    # 4. Model training
    # --------------------------------------------------------

    training_command = [
        python,
        "-m",
        "src.models.train",
    ]

    if training_mode == "quick":
        training_command.append("--quick")

    results.append(
        run_stage(
            f"Model training ({training_mode})",
            training_command,
        )
    )

    # --------------------------------------------------------
    # 5. Generate drift scenarios
    # --------------------------------------------------------

    results.append(
        run_stage(
            "Generate drift datasets",
            [
                python,
                "-m",
                "src.data.drift_generator",
            ],
        )
    )

    # --------------------------------------------------------
    # 6. Drift monitoring
    # --------------------------------------------------------

    monitoring = run_stage(
        "Data drift monitoring",
        [
            python,
            "-m",
            "src.data.monitoring",
        ],
        allowed_exit_codes=(0, 1),
    )

    # Exit code 1 represents critical drift in our project.
    if monitoring["exit_code"] == 1:
        monitoring["drift_alert"] = "critical"

        LOGGER.warning(
            "Critical drift detected in synthetic stress test."
        )

    else:
        monitoring["drift_alert"] = "none"

    results.append(monitoring)

    # --------------------------------------------------------
    # 7. Reference model robustness
    # --------------------------------------------------------

    results.append(
        run_stage(
            "Model robustness evaluation",
            [
                python,
                "-m",
                "src.models.robustness",
            ],
        )
    )

    # --------------------------------------------------------
    # Execution summary
    # --------------------------------------------------------

    total_duration = sum(
        result["duration_seconds"]
        for result in results
    )

    LOGGER.info("=" * 60)
    LOGGER.info("SPRINT 17 - PIPELINE COMPLETED")
    LOGGER.info("=" * 60)

    for result in results:
        LOGGER.info(
            "%-32s | %8.2fs | Exit: %s",
            result["stage"],
            result["duration_seconds"],
            result["exit_code"],
        )

    LOGGER.info("-" * 60)
    LOGGER.info("Training mode: %s", training_mode)
    LOGGER.info("Drift alert: %s", monitoring["drift_alert"])
    LOGGER.info("Reference model promotion: DISABLED")
    LOGGER.info("Total stage duration: %.2fs", total_duration)
    LOGGER.info("Overall status: SUCCESS")

    return results


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """Parse CLI arguments and execute the pipeline."""

    parser = argparse.ArgumentParser(
        description="Sprint 17 MLOps pipeline orchestrator."
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--quick",
        action="store_true",
        help="Run quick model training (default).",
    )

    mode.add_argument(
        "--full",
        action="store_true",
        help="Run the complete model search.",
    )

    parser.add_argument(
        "--pull-dvc",
        action="store_true",
        help="Download DVC-tracked data and reference model.",
    )

    args = parser.parse_args()

    configure_logging()

    training_mode = "full" if args.full else "quick"

    try:
        run_pipeline(
            training_mode=training_mode,
            pull_dvc=args.pull_dvc,
        )

    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        LOGGER.error("PIPELINE FAILED: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
