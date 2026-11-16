
"""
Sprint 17 - MLOps Final Project
Tests: tests/test_promote.py

Tests for explicit model promotion.

Run:
    python -m pytest tests/test_promote.py -v
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression

from src.models import promote


@pytest.fixture
def promotion_environment(
    tmp_path,
    monkeypatch,
):
    """Create isolated training and reference directories."""

    runs_dir = tmp_path / "models" / "runs"
    reference_dir = (
        tmp_path / "models" / "reference"
    )

    runs_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)

    monkeypatch.setattr(
        promote,
        "RUNS_DIR",
        runs_dir,
    )

    monkeypatch.setattr(
        promote,
        "REFERENCE_DIR",
        reference_dir,
    )

    run_id = "a" * 32
    run_dir = runs_dir / run_id
    run_dir.mkdir()

    X = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55],
        "hours": [20, 25, 30, 35, 40, 45, 50, 55],
    })

    y = np.array([
        0, 0, 0, 0, 1, 1, 1, 1
    ])

    model = LogisticRegression(
        max_iter=1000
    )
    model.fit(X, y)

    candidate_model = (
        run_dir / promote.MODEL_FILENAME
    )
    candidate_metadata = (
        run_dir / promote.METADATA_FILENAME
    )

    joblib.dump(
        model,
        candidate_model,
    )

    metadata = {
        "execution_id": run_id,
        "decision_threshold": 0.40,
        "model_family": "logistic_regression",
        "mlflow_run_id": "b" * 32,
        "preprocessing_variant": "baseline",
        "model_sha256": promote.sha256_file(
            candidate_model
        ),
    }

    candidate_metadata.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    # Mirror the summary written by a completed full training run.
    summary_path = run_dir / "training_summary.json"
    summary = {
        "execution_id": run_id,
        "quick": False,
        "candidate_count": 54,
        "successful_candidates": 54,
        "preferred_variant_by_validation": "baseline",
        "winners": {
            "baseline": {
                "validation": {
                    "run_id": metadata["mlflow_run_id"],
                    "status": "FINISHED",
                }
            }
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "runs_dir": runs_dir,
        "reference_dir": reference_dir,
        "candidate_model": candidate_model,
        "candidate_metadata": candidate_metadata,
        "metadata": metadata,
        "summary": summary,
        "summary_path": summary_path,
    }


def update_metadata(env, **changes):
    """Modify temporary candidate metadata."""

    metadata = dict(env["metadata"])
    metadata.update(changes)

    env["candidate_metadata"].write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def update_summary(env, **changes):
    """Modify the training summary without changing other artifacts."""
    summary = json.loads(env["summary_path"].read_text(encoding="utf-8"))
    summary.update(changes)
    env["summary_path"].write_text(json.dumps(summary), encoding="utf-8")


def update_winner(env, **changes):
    """Modify the winning MLflow validation record."""
    summary = json.loads(env["summary_path"].read_text(encoding="utf-8"))
    summary["winners"]["baseline"]["validation"].update(changes)
    env["summary_path"].write_text(json.dumps(summary), encoding="utf-8")


def test_validate_run_id():
    """Accept valid execution identifiers."""

    assert promote.validate_run_id(
        "a" * 32
    ) == "a" * 32


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "../reference",
        "invalid",
        "A" * 32,
        "z" * 32,
        "a" * 31,
    ],
)
def test_invalid_run_id(run_id):
    """Reject malformed or unsafe identifiers."""

    with pytest.raises(ValueError):
        promote.validate_run_id(run_id)


def test_resolve_candidate(
    promotion_environment,
):
    """Resolve candidate artifacts from a run directory."""

    env = promotion_environment

    model, metadata = promote.resolve_candidate(
        env["run_id"]
    )

    assert model == env["candidate_model"]
    assert metadata == env["candidate_metadata"]


def test_missing_run(
    promotion_environment,
):
    """Missing execution directories must fail."""

    with pytest.raises(FileNotFoundError):
        promote.resolve_candidate("b" * 32)


def test_missing_candidate_model(
    promotion_environment,
):
    """Candidate model must exist."""

    env = promotion_environment

    env["candidate_model"].unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Candidate model not found",
    ):
        promote.resolve_candidate(
            env["run_id"]
        )


def test_missing_candidate_metadata(
    promotion_environment,
):
    """Candidate metadata must exist."""

    env = promotion_environment

    env["candidate_metadata"].unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Candidate metadata not found",
    ):
        promote.resolve_candidate(
            env["run_id"]
        )


def test_valid_candidate(
    promotion_environment,
):
    """A valid full-training candidate passes checks."""

    env = promotion_environment

    metadata = promote.validate_candidate(
        env["run_id"],
        env["candidate_model"],
        env["candidate_metadata"],
    )

    assert metadata["preprocessing_variant"] == "baseline"


def test_quick_candidate_rejected(
    promotion_environment,
):
    """Quick-training candidates cannot be promoted."""

    env = promotion_environment

    update_summary(env, quick=True)

    with pytest.raises(
        ValueError,
        match="full-training",
    ):
        promote.validate_candidate(
            env["run_id"],
            env["candidate_model"],
            env["candidate_metadata"],
        )


def test_missing_full_training_summary(promotion_environment):
    env = promotion_environment
    env["summary_path"].unlink()
    with pytest.raises(FileNotFoundError, match="Full-training summary"):
        promote.promote_candidate(env["run_id"])


def test_summary_execution_id_mismatch(promotion_environment):
    env = promotion_environment
    update_summary(env, execution_id="c" * 32)
    with pytest.raises(ValueError, match="summary execution ID mismatch"):
        promote.promote_candidate(env["run_id"])


@pytest.mark.parametrize("successful", [0, 53, 55, True])
def test_incomplete_training_rejected(promotion_environment, successful):
    env = promotion_environment
    update_summary(env, successful_candidates=successful)
    with pytest.raises(ValueError, match="did not complete"):
        promote.promote_candidate(env["run_id"])


def test_variant_mismatch(promotion_environment):
    env = promotion_environment
    update_metadata(env, preprocessing_variant="robust")
    with pytest.raises(ValueError, match="variant"):
        promote.promote_candidate(env["run_id"])


def test_mlflow_run_mismatch(promotion_environment):
    env = promotion_environment
    update_metadata(env, mlflow_run_id="c" * 32)
    with pytest.raises(ValueError, match="MLflow run ID"):
        promote.promote_candidate(env["run_id"])


def test_unfinished_winner_rejected(promotion_environment):
    env = promotion_environment
    update_winner(env, status="FAILED")
    with pytest.raises(ValueError, match="FINISHED"):
        promote.promote_candidate(env["run_id"])


def test_execution_id_mismatch(
    promotion_environment,
):
    """Candidate must belong to the selected execution."""

    env = promotion_environment

    update_metadata(
        env,
        execution_id="b" * 32,
    )

    with pytest.raises(
        ValueError,
        match="execution ID",
    ):
        promote.validate_candidate(
            env["run_id"],
            env["candidate_model"],
            env["candidate_metadata"],
        )


def test_candidate_hash_mismatch(
    promotion_environment,
):
    """Modified candidate artifacts must be rejected."""

    env = promotion_environment

    update_metadata(
        env,
        model_sha256="0" * 64,
    )

    with pytest.raises(
        ValueError,
        match="SHA-256 mismatch",
    ):
        promote.validate_candidate(
            env["run_id"],
            env["candidate_model"],
            env["candidate_metadata"],
        )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_invalid_threshold(
    promotion_environment,
    threshold,
):
    """Reject thresholds outside the valid range."""

    env = promotion_environment

    update_metadata(
        env,
        decision_threshold=threshold,
    )

    with pytest.raises(
        ValueError,
        match="Invalid decision threshold",
    ):
        promote.validate_candidate(
            env["run_id"],
            env["candidate_model"],
            env["candidate_metadata"],
        )


def test_dry_run_does_not_create_reference(
    promotion_environment,
):
    """Validation alone must not promote the model."""

    env = promotion_environment

    result = promote.promote_candidate(
        env["run_id"],
        confirm=False,
    )

    assert result["validated"] is True
    assert result["promoted"] is False

    assert not (
        env["reference_dir"]
        / promote.MODEL_FILENAME
    ).exists()


def test_explicit_promotion(
    promotion_environment,
):
    """Confirmed promotion must create reference artifacts."""

    env = promotion_environment

    result = promote.promote_candidate(
        env["run_id"],
        confirm=True,
    )

    reference_model = (
        env["reference_dir"]
        / promote.MODEL_FILENAME
    )

    reference_metadata = (
        env["reference_dir"]
        / promote.METADATA_FILENAME
    )

    assert result["promoted"] is True

    assert reference_model.is_file()
    assert reference_metadata.is_file()

    assert promote.sha256_file(
        reference_model
    ) == promote.sha256_file(
        env["candidate_model"]
    )


def test_promoted_metadata(
    promotion_environment,
):
    """Reference metadata must record promotion provenance."""

    env = promotion_environment

    promote.promote_candidate(
        env["run_id"],
        confirm=True,
    )

    metadata_path = (
        env["reference_dir"]
        / promote.METADATA_FILENAME
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert metadata["execution_id"] == env["run_id"]
    assert metadata["promotion_type"] == "explicit"
    assert "promoted_at" in metadata
    assert "promotion_source" in metadata


def test_promotion_preserves_candidate(
    promotion_environment,
):
    """Promotion must not modify source candidate artifacts."""

    env = promotion_environment

    before_model = promote.sha256_file(
        env["candidate_model"]
    )

    before_metadata = promote.sha256_file(
        env["candidate_metadata"]
    )

    promote.promote_candidate(
        env["run_id"],
        confirm=True,
    )

    assert promote.sha256_file(
        env["candidate_model"]
    ) == before_model

    assert promote.sha256_file(
        env["candidate_metadata"]
    ) == before_metadata


def test_promotion_replaces_existing_reference(
    promotion_environment,
):
    """A confirmed promotion may replace an old reference."""

    env = promotion_environment

    reference_model = (
        env["reference_dir"]
        / promote.MODEL_FILENAME
    )

    reference_metadata = (
        env["reference_dir"]
        / promote.METADATA_FILENAME
    )

    reference_model.write_bytes(
        b"old reference"
    )

    reference_metadata.write_text(
        '{"old": true}',
        encoding="utf-8",
    )

    promote.promote_candidate(
        env["run_id"],
        confirm=True,
    )

    assert promote.sha256_file(
        reference_model
    ) == promote.sha256_file(
        env["candidate_model"]
    )


def test_promotion_rolls_back_on_failure(
    promotion_environment,
    monkeypatch,
):
    """A failed metadata write must restore the old reference."""

    env = promotion_environment

    reference_model = (
        env["reference_dir"]
        / promote.MODEL_FILENAME
    )

    reference_metadata = (
        env["reference_dir"]
        / promote.METADATA_FILENAME
    )

    reference_model.write_bytes(
        b"original reference"
    )

    reference_metadata.write_text(
        '{"original": true}',
        encoding="utf-8",
    )

    original_model_hash = promote.sha256_file(
        reference_model
    )

    original_metadata_hash = promote.sha256_file(
        reference_metadata
    )

    def fail_write(*args, **kwargs):
        raise OSError(
            "Simulated metadata write failure"
        )

    monkeypatch.setattr(
        promote,
        "atomic_write_json",
        fail_write,
    )

    with pytest.raises(
        OSError,
        match="Simulated metadata write failure",
    ):
        promote.promote_candidate(
            env["run_id"],
            confirm=True,
        )

    assert promote.sha256_file(
        reference_model
    ) == original_model_hash

    assert promote.sha256_file(
        reference_metadata
    ) == original_metadata_hash


def test_invalid_candidate_never_changes_reference(
    promotion_environment,
):
    """Validation failures must preserve existing references."""

    env = promotion_environment

    reference_model = (
        env["reference_dir"]
        / promote.MODEL_FILENAME
    )

    reference_model.write_bytes(
        b"protected reference"
    )

    before = promote.sha256_file(
        reference_model
    )

    update_summary(env, quick=True)

    with pytest.raises(ValueError):
        promote.promote_candidate(
            env["run_id"],
            confirm=True,
        )

    assert promote.sha256_file(
        reference_model
    ) == before
