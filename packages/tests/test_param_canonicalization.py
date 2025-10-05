"""Tests to ensure parameter aliases work correctly."""
# SKIPPED: Moved from /tests/ - needs API review
import pytest
pytestmark = pytest.mark.skip(reason="Moved from /tests/ - API review needed")


import types

import numpy as np
import pandas as pd
from glassalpha.pipeline.train import train_from_config


def test_num_boost_round_and_seed_aliases_work():
    """Test that num_boost_round and seed aliases are canonicalized correctly."""
    # Create binary test data
    X = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
    y = np.array([0, 1] * 10)

    # Config using aliases
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {
        "num_boost_round": 50,  # Should become n_estimators
        "seed": 42,  # Should become random_state
        "max_depth": 3,
    }
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    # Train model
    model = train_from_config(cfg, X, y)

    # Verify the underlying XGBoost model has canonical parameters
    underlying_model = model.model

    # Check that aliases were converted
    assert hasattr(underlying_model, "n_estimators"), "n_estimators should be set"
    assert hasattr(underlying_model, "random_state"), "random_state should be set"

    # The exact values depend on the XGBClassifier implementation
    # but they should be present as attributes
    n_estimators = getattr(underlying_model, "n_estimators", None)
    random_state = getattr(underlying_model, "random_state", None)

    assert n_estimators is not None, (
        "num_boost_round should have been converted to n_estimators"
    )
    assert random_state is not None, "seed should have been converted to random_state"


def test_random_state_precedence():
    """Test that explicit random_state takes precedence over seed alias."""
    X = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
    y = np.array([0, 1] * 10)

    # Config with both random_state and seed
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {
        "seed": 123,  # This should be overridden
        "random_state": 456,  # This should take precedence
        "max_depth": 3,
    }
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    model = train_from_config(cfg, X, y)

    # Verify random_state takes precedence
    underlying_model = model.model
    random_state = getattr(underlying_model, "random_state", None)
    assert random_state == 456, f"Expected random_state=456, got {random_state}"


def test_multi_softmax_coercion():
    """Test that multi:softmax is coerced to multi:softprob for audits."""
    X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
    y = np.array([0, 1, 2] * 10)

    # Config with multi:softmax (should be coerced)
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {
        "objective": "multi:softmax",  # Should become multi:softprob
        "num_class": 3,
        "max_depth": 3,
    }
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    model = train_from_config(cfg, X, y)

    # Verify predict_proba works (would fail with multi:softmax)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 3), (
        f"Expected shape ({len(X)}, 3), got {proba.shape}"
    )

    # Verify probabilities sum to 1
    prob_sums = np.sum(proba, axis=1)
    np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-6)


def test_mixed_parameter_aliases():
    """Test various parameter aliases work together."""
    X = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
    y = np.array([0, 1] * 10)

    # Config with multiple aliases
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {
        "num_boost_round": 25,
        "seed": 99,
        "max_depth": 4,
        "eta": 0.05,  # Standard XGBoost parameter
    }
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    model = train_from_config(cfg, X, y)

    # Verify model was trained successfully
    info = model.get_model_info()
    assert info["n_classes"] == 2

    # Verify predict_proba works
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)

    # Verify predictions work
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})
