"""Smoke test for models/tabular/sklearn.py to bump coverage."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from glassalpha.models.tabular import sklearn as gsk


def test_sklearn_adapter_smoke():
    """One fitted model round-trip on a 2×2 toy dataset."""
    X = np.array([[0, 0], [1, 1]], dtype=float)
    y = np.array([0, 1])
    # Train the model first, then wrap it
    lr = LogisticRegression()
    lr.fit(X, y)
    clf = gsk.SklearnGenericWrapper(lr)
    proba = clf.predict_proba(X)
    assert proba.shape == (2, 2)
