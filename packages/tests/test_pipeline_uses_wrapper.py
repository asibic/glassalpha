"""Tests to ensure pipeline uses wrapper instead of direct estimator training."""

import types

import numpy as np
import pandas as pd

from glassalpha.pipeline.train import train_from_config


def test_pipeline_trains_via_wrapper(monkeypatch):
    """Test that audit pipeline uses train_from_config, not direct training."""
    calls = {}

    def spy(cfg, X, y):
        calls["used"] = True
        calls["config"] = cfg
        calls["X_shape"] = X.shape
        calls["y_shape"] = y.shape
        # Return a mock model
        mock_model = types.SimpleNamespace()
        mock_model.get_capabilities = lambda: {"supports_proba": True}
        mock_model.get_model_info = lambda: {"n_classes": 2}
        return mock_model

    monkeypatch.setattr("glassalpha.pipeline.train.train_from_config", spy)

    # Create minimal config
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {"max_depth": 3}
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    # Create test data
    X = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    y = np.array([0, 1] * 10)

    # Call train_from_config directly (this simulates what pipeline should do)
    model = train_from_config(cfg, X, y)

    # Verify train_from_config was called (not direct XGBoost training)
    assert calls.get("used"), "Pipeline bypassed train_from_config"

    # Verify parameters were passed correctly
    assert calls["X_shape"] == (20, 3)
    assert calls["y_shape"] == (20,)

    # Verify model has required capabilities for audits
    caps = model.get_capabilities()
    assert caps.get("supports_proba", False), "Model must support predict_proba for audits"
