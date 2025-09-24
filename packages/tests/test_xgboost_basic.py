"""Basic XGBoost model wrapper tests.

Tests that XGBoost wrapper can be instantiated and basic functionality works.
These tests focus on covering xgboost.py for minimum viable coverage.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from glassalpha.core.registry import ModelRegistry
from glassalpha.models.tabular.xgboost import XGBoostWrapper


def test_xgboost_wrapper_registration():
    """Test that XGBoostWrapper is registered correctly."""
    # Check that xgboost is in the registry
    components = ModelRegistry.list_components()
    assert "xgboost" in components

    # Check that we can get the wrapper class
    wrapper_class = ModelRegistry.get_component("xgboost")
    assert wrapper_class == XGBoostWrapper


def test_xgboost_wrapper_capabilities():
    """Test XGBoostWrapper declares correct capabilities."""
    assert XGBoostWrapper.capabilities["supports_shap"] is True
    assert XGBoostWrapper.capabilities["supports_feature_importance"] is True
    assert XGBoostWrapper.capabilities["supports_proba"] is True
    assert XGBoostWrapper.capabilities["data_modality"] == "tabular"


def test_xgboost_wrapper_init_no_model():
    """Test XGBoostWrapper can be initialized without a model."""
    wrapper = XGBoostWrapper()
    assert wrapper._model is None
    assert wrapper.feature_names is None


def test_xgboost_wrapper_with_trained_model():
    """Test XGBoostWrapper with a simple trained model."""
    # Create simple training data
    X = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5], "feature_2": [2, 4, 6, 8, 10]})
    y = np.array([0, 0, 1, 1, 1])

    # Train a simple XGBoost model
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "binary:logistic", "eval_metric": "logloss"}
    model = xgb.train(params, dtrain, num_boost_round=10)

    # Create wrapper with the trained model
    wrapper = XGBoostWrapper(model=model)
    assert wrapper._model is not None
    assert wrapper.feature_names == ["feature_1", "feature_2"]


def test_xgboost_wrapper_predict():
    """Test XGBoostWrapper predict method."""
    # Create simple training data
    X_train = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5], "feature_2": [2, 4, 6, 8, 10]})
    y_train = np.array([0, 0, 1, 1, 1])

    # Train model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {"objective": "binary:logistic", "eval_metric": "logloss"}
    model = xgb.train(params, dtrain, num_boost_round=10)

    # Create wrapper and test prediction
    wrapper = XGBoostWrapper(model=model)

    # Test data
    X_test = pd.DataFrame({"feature_1": [2.5, 3.5], "feature_2": [5, 7]})

    predictions = wrapper.predict(X_test)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 2
    assert all(0 <= p <= 1 for p in predictions)  # Should be probabilities


def test_xgboost_wrapper_predict_proba():
    """Test XGBoostWrapper predict_proba method."""
    # Create simple training data
    X_train = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5], "feature_2": [2, 4, 6, 8, 10]})
    y_train = np.array([0, 0, 1, 1, 1])

    # Train model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {"objective": "binary:logistic", "eval_metric": "logloss"}
    model = xgb.train(params, dtrain, num_boost_round=10)

    # Create wrapper and test prediction
    wrapper = XGBoostWrapper(model=model)

    # Test data
    X_test = pd.DataFrame({"feature_1": [2.5, 3.5], "feature_2": [5, 7]})

    proba = wrapper.predict_proba(X_test)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (2, 2)  # 2 samples, 2 classes
    assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1


def test_xgboost_wrapper_feature_importance():
    """Test XGBoostWrapper get_feature_importance method."""
    # Create simple training data
    X_train = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5], "feature_2": [2, 4, 6, 8, 10]})
    y_train = np.array([0, 0, 1, 1, 1])

    # Train model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {"objective": "binary:logistic", "eval_metric": "logloss"}
    model = xgb.train(params, dtrain, num_boost_round=10)

    # Create wrapper and test feature importance
    wrapper = XGBoostWrapper(model=model)

    importance = wrapper.get_feature_importance()
    assert isinstance(importance, dict)
    assert "feature_1" in importance
    assert "feature_2" in importance
    assert all(isinstance(v, (int, float)) for v in importance.values())


def test_xgboost_wrapper_save_and_load():
    """Test XGBoostWrapper save and load functionality."""
    # Create simple training data
    X_train = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5], "feature_2": [2, 4, 6, 8, 10]})
    y_train = np.array([0, 0, 1, 1, 1])

    # Train model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {"objective": "binary:logistic", "eval_metric": "logloss"}
    model = xgb.train(params, dtrain, num_boost_round=10)

    # Create wrapper
    wrapper = XGBoostWrapper(model=model)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
        model_path = tmp_file.name

    try:
        wrapper.save(model_path)

        # Load from file
        new_wrapper = XGBoostWrapper(model_path=model_path)

        # Test that loaded model works
        X_test = pd.DataFrame({"feature_1": [2.5], "feature_2": [5]})

        original_pred = wrapper.predict(X_test)
        loaded_pred = new_wrapper.predict(X_test)

        assert np.allclose(original_pred, loaded_pred)

    finally:
        Path(model_path).unlink(missing_ok=True)


def test_xgboost_wrapper_without_model_raises_error():
    """Test that operations without a model raise appropriate errors."""
    wrapper = XGBoostWrapper()

    X_test = pd.DataFrame({"feature_1": [1], "feature_2": [2]})

    with pytest.raises(ValueError, match="No model loaded"):
        wrapper.predict(X_test)
