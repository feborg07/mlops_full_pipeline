"""Unit tests for the MLflow experiment comparison helper."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import compare_experiments as comparison


def test_rank_rejects_failed_and_final_test_runs():
    runs = pd.DataFrame([
        {"run_id": "candidate", "status": "FINISHED", "tags.model_family": "random_forest", "tags.preprocessing_variant": "baseline", "tags.mode": "quick", "metrics.val_f1": 0.72},
        {"run_id": "failed", "status": "FAILED", "tags.model_family": "random_forest", "tags.preprocessing_variant": "baseline", "tags.mode": "quick", "metrics.val_f1": 0.99},
        {"run_id": "final-test", "status": "FINISHED", "tags.model_family": None, "tags.preprocessing_variant": "baseline", "tags.mode": None, "metrics.val_f1": 0.98},
    ])
    ranked = comparison.rank_candidates(runs)
    assert ranked["run_id"].tolist() == ["candidate"]


def test_rank_uses_validation_f1_and_tie_breakers():
    runs = pd.DataFrame([
        {"run_id": "a", "status": "FINISHED", "tags.model_family": "extra_trees", "tags.preprocessing_variant": "baseline", "metrics.val_f1": "0.71", "metrics.val_pr_auc": 0.8, "metrics.val_roc_auc": 0.9},
        {"run_id": "b", "status": "FINISHED", "tags.model_family": "random_forest", "tags.preprocessing_variant": "baseline", "metrics.val_f1": 0.72, "metrics.val_pr_auc": 0.75, "metrics.val_roc_auc": 0.9},
        {"run_id": "c", "status": "FINISHED", "tags.model_family": "logistic_regression", "tags.preprocessing_variant": "baseline", "metrics.val_f1": 0.72, "metrics.val_pr_auc": 0.82, "metrics.val_roc_auc": 0.9},
    ])
    assert comparison.rank_candidates(runs)["run_id"].tolist() == ["c", "b", "a"]


def test_rank_filters_training_mode_and_variant():
    runs = pd.DataFrame([
        {"run_id": "a", "status": "FINISHED", "tags.model_family": "extra_trees", "tags.preprocessing_variant": "baseline", "tags.mode": "quick", "metrics.val_f1": 0.6},
        {"run_id": "b", "status": "FINISHED", "tags.model_family": "random_forest", "tags.preprocessing_variant": "robust", "tags.mode": "full", "metrics.val_f1": 0.7},
    ])
    assert comparison.rank_candidates(runs, mode="full", variant="robust")["run_id"].tolist() == ["b"]


def test_rank_skips_nan_and_missing_metrics():
    runs = pd.DataFrame([
        {"run_id": "bad", "status": "FINISHED", "tags.model_family": "extra_trees", "tags.preprocessing_variant": "baseline", "metrics.val_f1": "not-a-number"},
        {"run_id": "good", "status": "FINISHED", "tags.model_family": "extra_trees", "tags.preprocessing_variant": "baseline", "metrics.val_f1": 0.65},
    ])
    assert comparison.rank_candidates(runs)["run_id"].tolist() == ["good"]


def test_load_settings_from_yaml(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text("variants:\n  - baseline\n  - robust\n  - interactions\nmlflow:\n  experiment: some-experiment\npaths:\n  mlflow_db: mlflow.db\n", encoding="utf-8")
    name, db, variants = comparison.load_settings(config)
    assert name == "some-experiment"
    assert db == (comparison.ROOT / "mlflow.db").resolve()
    assert variants == ["baseline", "robust", "interactions"]


def test_compare_queries_mlflow_and_writes_csv(tmp_path, monkeypatch):
    db = tmp_path / "mlflow.db"
    db.touch()
    config = tmp_path / "config.yml"
    config.write_text(f"variants:\n  - baseline\n  - robust\n  - interactions\nmlflow:\n  experiment: example\npaths:\n  mlflow_db: {db.as_posix()}\n", encoding="utf-8")
    calls = {}
    monkeypatch.setattr(comparison.mlflow, "set_tracking_uri", lambda uri: calls.setdefault("uri", uri))
    monkeypatch.setattr(comparison.mlflow, "get_experiment_by_name", lambda name: SimpleNamespace(experiment_id="42"))

    def fake_search_runs(**kwargs):
        calls["query"] = kwargs
        return pd.DataFrame([{
            "run_id": "winner", "status": "FINISHED", "tags.model_family": "hist_gradient_boosting",
            "tags.preprocessing_variant": "baseline", "tags.mode": "full",
            "metrics.val_f1": 0.74, "params.model_learning_rate": "0.1",
        }])
    monkeypatch.setattr(comparison.mlflow, "search_runs", fake_search_runs)
    exported = tmp_path / "leaderboard.csv"
    result = comparison.compare(config_path=config, output=exported)
    assert calls["query"]["experiment_ids"] == ["42"]
    assert result.iloc[0]["run_id"] == "winner"
    assert pd.read_csv(exported).iloc[0]["run_id"] == "winner"


def test_missing_mlflow_database_fails_clearly(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text(f"variants:\n  - baseline\n  - robust\n  - interactions\nmlflow:\n  experiment: example\npaths:\n  mlflow_db: {(tmp_path / 'missing.db').as_posix()}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Run training first"):
        comparison.compare(config_path=config)


def test_compare_defaults_to_full_runs(tmp_path, monkeypatch):
    """Quick smoke-check records must not compete with full experiments."""
    db = tmp_path / "mlflow.db"
    db.touch()
    config = tmp_path / "config.yml"
    config.write_text(
        f"variants:\n  - baseline\n  - robust\n  - interactions\nmlflow:\n  experiment: example\npaths:\n  mlflow_db: {db.as_posix()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(comparison.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        comparison.mlflow, "get_experiment_by_name",
        lambda name: SimpleNamespace(experiment_id="42"),
    )
    monkeypatch.setattr(comparison.mlflow, "search_runs", lambda **kwargs: pd.DataFrame([
        {"run_id": "quick-high", "status": "FINISHED", "tags.model_family": "random_forest",
         "tags.preprocessing_variant": "baseline", "tags.mode": "quick", "metrics.val_f1": 0.99},
        {"run_id": "full-good", "status": "FINISHED", "tags.model_family": "extra_trees",
         "tags.preprocessing_variant": "baseline", "tags.mode": "full", "metrics.val_f1": 0.73},
    ]))
    assert comparison.compare(config_path=config)["run_id"].tolist() == ["full-good"]
    assert comparison.compare(config_path=config, mode="all")["run_id"].tolist() == ["quick-high", "full-good"]


def test_variant_ties_follow_yaml_order():
    runs = pd.DataFrame([
        {"run_id": "robust-first", "status": "FINISHED",
         "tags.model_family": "hist_gradient_boosting", "tags.preprocessing_variant": "robust",
         "metrics.val_f1": 0.7337, "metrics.val_pr_auc": 0.8296, "metrics.val_roc_auc": 0.9304},
        {"run_id": "baseline-second", "status": "FINISHED",
         "tags.model_family": "hist_gradient_boosting", "tags.preprocessing_variant": "baseline",
         "metrics.val_f1": 0.7337, "metrics.val_pr_auc": 0.8296, "metrics.val_roc_auc": 0.9304},
    ])
    result = comparison.rank_candidates(
        runs, variant_order=["baseline", "robust", "interactions"]
    )
    assert result["run_id"].tolist() == ["baseline-second", "robust-first"]


def test_variant_order_is_from_yaml_not_hardcoded():
    runs = pd.DataFrame([
        {"run_id": "baseline", "status": "FINISHED",
         "tags.model_family": "random_forest", "tags.preprocessing_variant": "baseline",
         "metrics.val_f1": 0.7},
        {"run_id": "robust", "status": "FINISHED",
         "tags.model_family": "random_forest", "tags.preprocessing_variant": "robust",
         "metrics.val_f1": 0.7},
    ])
    result = comparison.rank_candidates(runs, variant_order=["robust", "baseline"])
    assert result["run_id"].tolist() == ["robust", "baseline"]


def test_execution_filter_keeps_only_requested_runs():
    chosen = "a" * 32
    runs = pd.DataFrame([
        {"run_id": "old", "status": "FINISHED", "tags.model_family": "random_forest",
         "tags.preprocessing_variant": "baseline", "tags.mode": "full",
         "tags.training_execution_id": "b" * 32, "metrics.val_f1": 0.99},
        {"run_id": "current", "status": "FINISHED", "tags.model_family": "random_forest",
         "tags.preprocessing_variant": "baseline", "tags.mode": "full",
         "tags.training_execution_id": chosen, "metrics.val_f1": 0.72},
    ])
    assert comparison.rank_candidates(
        runs, mode="full", execution_id=chosen
    )["run_id"].tolist() == ["current"]


def test_candidate_index_breaks_same_variant_ties():
    runs = pd.DataFrame([
        {"run_id": "later", "status": "FINISHED", "tags.model_family": "random_forest",
         "tags.preprocessing_variant": "baseline", "tags.candidate_index": "5",
         "metrics.val_f1": 0.7},
        {"run_id": "earlier", "status": "FINISHED", "tags.model_family": "random_forest",
         "tags.preprocessing_variant": "baseline", "tags.candidate_index": "2",
         "metrics.val_f1": 0.7},
    ])
    assert comparison.rank_candidates(runs)["run_id"].tolist() == ["earlier", "later"]


def test_compare_rejects_invalid_execution_id(tmp_path):
    config = tmp_path / "config.yml"
    with pytest.raises(ValueError, match="32-character"):
        comparison.compare(config_path=config, execution_id="not-an-id")
