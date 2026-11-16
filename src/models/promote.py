
"""
Sprint 17 - MLOps Final Project
Module: src/models/promote.py

Explicitly promote a validated training candidate to the
DVC-managed reference model.

Promotion is never triggered automatically by training.

Example:
    python -m src.models.promote \
        --run-id <execution_id> \
        --confirm

Dry run:
    python -m src.models.promote \
        --run-id <execution_id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

RUNS_DIR = ROOT / "models" / "runs"
REFERENCE_DIR = ROOT / "models" / "reference"

MODEL_FILENAME = "best_model_pipeline.joblib"
METADATA_FILENAME = "best_model_metadata.json"

LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 hash."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict:
    """Read and validate a JSON object."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return data


def validate_run_id(run_id: str) -> str:
    """Reject traversal and invalid execution identifiers."""

    if not isinstance(run_id, str):
        raise ValueError("Run ID must be a string.")

    if (
        not run_id
        or len(run_id) != 32
        or any(
            char not in "0123456789abcdef"
            for char in run_id
        )
    ):
        raise ValueError(
            "Run ID must be a 32-character lowercase "
            "hexadecimal execution identifier."
        )

    return run_id


def resolve_candidate(run_id: str) -> tuple[Path, Path]:
    """
    Resolve candidate artifacts inside a training run.

    The expected layout is:
        models/runs/<execution_id>/
            best_model_pipeline.joblib
            best_model_metadata.json
    """

    validate_run_id(run_id)

    run_dir = RUNS_DIR / run_id

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Training execution not found: {run_dir}"
        )

    model_path = run_dir / MODEL_FILENAME
    metadata_path = run_dir / METADATA_FILENAME

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Candidate model not found: {model_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Candidate metadata not found: {metadata_path}"
        )

    return model_path, metadata_path


def validate_candidate(
    run_id: str,
    model_path: Path,
    metadata_path: Path,
) -> dict:
    """
    Validate candidate provenance, integrity, and structure.

    This function never modifies reference artifacts.
    """

    metadata = load_json(metadata_path)

    if metadata.get("execution_id") != run_id:
        raise ValueError(
            "Candidate execution ID does not match "
            "the selected training run."
        )

    # Read the training summary from the same execution.
    summary_path = model_path.parent / "training_summary.json"

    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Full-training summary not found: {summary_path}"
        )

    summary = load_json(summary_path)

    # Verify that the summary belongs to this execution.
    if summary.get("execution_id") != run_id:
        raise ValueError(
            "Training summary execution ID mismatch."
        )

    # Quick-training executions cannot be promoted.
    if summary.get("quick") is not False:
        raise ValueError(
            "Only full-training candidates may be promoted."
        )

    # All candidates must have completed successfully.
    candidate_count = summary.get("candidate_count")
    successful_count = summary.get("successful_candidates")

    if (
        type(candidate_count) is not int
        or candidate_count <= 0
        or type(successful_count) is not int
        or successful_count != candidate_count
    ):
        raise ValueError(
            "Training execution did not complete successfully."
        )

    # Verify that the candidate matches the selected variant.
    preferred_variant = summary.get(
        "preferred_variant_by_validation"
    )

    if metadata.get("preprocessing_variant") != preferred_variant:
        raise ValueError(
            "Candidate variant does not match training selection."
        )

    # Verify the winning MLflow run.
    winners = summary.get("winners", {})

    winner = winners.get(preferred_variant, {})
    validation = winner.get("validation", {})

    if validation.get("status") != "FINISHED":
        raise ValueError(
            "Winning training run is not marked FINISHED."
        )

    if metadata.get("mlflow_run_id") != validation.get("run_id"):
        raise ValueError(
            "Candidate MLflow run ID does not match the winner."
        )

    expected_hash = metadata.get("model_sha256")

    if not isinstance(expected_hash, str):
        raise ValueError(
            "Candidate metadata is missing model_sha256."
        )

    actual_hash = sha256_file(model_path)

    if actual_hash != expected_hash:
        raise ValueError(
            "Candidate model SHA-256 mismatch."
        )

    threshold = metadata.get("decision_threshold")

    if threshold is None:
        raise ValueError(
            "Candidate metadata is missing decision_threshold."
        )

    threshold = float(threshold)

    if not np.isfinite(threshold) or not (
        0.0 <= threshold <= 1.0
    ):
        raise ValueError(
            f"Invalid decision threshold: {threshold}"
        )

    # Loading joblib files is only safe for trusted artifacts.
    model = joblib.load(model_path)

    if not callable(
        getattr(model, "predict_proba", None)
    ):
        raise TypeError(
            "Candidate model must support predict_proba()."
        )

    classes = getattr(model, "classes_", None)

    if classes is None or 1 not in np.asarray(classes):
        raise ValueError(
            "Candidate model must expose positive class 1."
        )

    return metadata


def build_reference_metadata(
    candidate_metadata: dict,
    run_id: str,
    model_hash: str,
) -> dict:
    """Construct metadata for the new reference model."""

    metadata = dict(candidate_metadata)

    metadata.update({
        "execution_id": run_id,
        "model_sha256": model_hash,
        "promoted_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "promotion_source": (
            f"models/runs/{run_id}"
        ),
        "promotion_type": "explicit",
    })

    return metadata


def atomic_write_json(
    destination: Path,
    data: dict,
) -> None:
    """Write JSON through a temporary file."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

            json.dump(
                data,
                handle,
                indent=2,
                allow_nan=False,
            )

        os.replace(temporary_path, destination)

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def promote_candidate(
    run_id: str,
    *,
    confirm: bool = False,
) -> dict:
    """
    Validate and optionally promote a training candidate.

    Without --confirm, this function performs validation
    only and leaves the reference model unchanged.
    """

    model_path, metadata_path = resolve_candidate(
        run_id
    )

    candidate_metadata = validate_candidate(
        run_id,
        model_path,
        metadata_path,
    )

    candidate_hash = sha256_file(model_path)

    result = {
        "execution_id": run_id,
        "candidate_sha256": candidate_hash,
        "validated": True,
        "promoted": False,
    }

    if not confirm:
        LOGGER.info(
            "Candidate validated. No files changed."
        )
        LOGGER.info(
            "Use --confirm to promote this candidate."
        )
        return result

    REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_model = (
        REFERENCE_DIR / MODEL_FILENAME
    )
    reference_metadata = (
        REFERENCE_DIR / METADATA_FILENAME
    )

    # Back up both reference artifacts before modification.
    with tempfile.TemporaryDirectory(
        prefix="promotion_backup_",
        dir=REFERENCE_DIR.parent,
    ) as backup_name:
        backup_dir = Path(backup_name)

        backups = {}

        for path in (
            reference_model,
            reference_metadata,
        ):
            if path.exists():
                backup = backup_dir / path.name

                shutil.copy2(path, backup)
                backups[path] = backup

        new_metadata = build_reference_metadata(
            candidate_metadata,
            run_id,
            candidate_hash,
        )

        temporary_model = (
            REFERENCE_DIR / ".candidate_model.tmp"
        )

        try:
            shutil.copy2(
                model_path,
                temporary_model,
            )

            if sha256_file(
                temporary_model
            ) != candidate_hash:
                raise ValueError(
                    "Candidate changed during copying."
                )

            # Each file is replaced atomically, but the
            # pair is not a single atomic transaction.
            os.replace(
                temporary_model,
                reference_model,
            )

            atomic_write_json(
                reference_metadata,
                new_metadata,
            )

        except Exception:
            LOGGER.exception(
                "Promotion failed. Restoring reference."
            )

            for destination in (
                reference_model,
                reference_metadata,
            ):
                backup = backups.get(destination)

                if backup is not None:
                    shutil.copy2(
                        backup,
                        destination,
                    )
                elif destination.exists():
                    destination.unlink()

            raise

        finally:
            if temporary_model.exists():
                temporary_model.unlink()

    result["promoted"] = True
    result["reference_model"] = str(
        reference_model
    )

    LOGGER.info(
        "Candidate promoted successfully: %s",
        run_id,
    )

    return result


def parse_args():
    """Parse promotion command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate or explicitly promote a "
            "full-training candidate."
        )
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Training execution identifier.",
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Replace the reference model.",
    )

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()

    result = promote_candidate(
        args.run_id,
        confirm=args.confirm,
    )

    print(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
