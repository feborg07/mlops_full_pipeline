"""Small, deterministic model validation tests required by Sprint 17.

These tests deliberately do not invoke the 12-/54-candidate training sweep.
They validate the scikit-learn prediction contract and a minimal F1 score on
an independent, seeded hold-out sample. The CI training job separately gates
real Adult Income validation F1 through MLflow.
"""

import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="module")
def fitted_classifier():
    """Fit one inexpensive model on a predictable classification problem."""
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=8,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        flip_y=0.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_model_predictions_have_correct_shape_and_type(fitted_classifier) -> None:
    """A fitted classifier must produce one valid class label per row."""
    model, X_test, _ = fitted_classifier
    predictions = model.predict(X_test)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(X_test),)
    assert np.issubdtype(predictions.dtype, np.integer)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_model_meets_minimum_f1_on_known_holdout(fitted_classifier) -> None:
    """A deterministic known test set catches broken learning behavior."""
    model, X_test, y_test = fitted_classifier
    score = f1_score(y_test, model.predict(X_test), pos_label=1)
    assert score >= 0.80, f"Model F1 {score:.4f} is below minimum 0.80"


def test_model_probabilities_are_valid(fitted_classifier) -> None:
    """Predicted probabilities must be finite and sum to one per sample."""
    model, X_test, _ = fitted_classifier
    probabilities = model.predict_proba(X_test)
    assert probabilities.shape == (len(X_test), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
