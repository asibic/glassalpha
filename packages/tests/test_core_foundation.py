"""Test core foundation components work together.

This tests the basic architecture patterns: interfaces, registries,
and NoOp implementations.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from glassalpha.core import (
    ExplainerRegistry,
    FeatureNotAvailable,
    MetricRegistry,
    ModelRegistry,
    NoOpExplainer,
    NoOpMetric,
    PassThroughModel,
    check_feature,
    is_enterprise,
    list_components,
    select_explainer,
)


def test_interfaces_are_protocols():
    """Verify interfaces use Protocol pattern."""
    from glassalpha.core.interfaces import ModelInterface

    assert hasattr(ModelInterface, "__subclasshook__")


def test_passthrough_model_works():
    """Test PassThrough model generates predictions."""
    model = PassThroughModel(default_value=0.7)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    predictions = model.predict(df)
    assert len(predictions) == 3
    assert all(p == 0.7 for p in predictions)

    proba = model.predict_proba(df)
    assert proba.shape == (3, 2)
    assert all(p[1] == 0.7 for p in proba)


def test_noop_explainer_works():
    """Test NoOp explainer returns valid structure."""
    explainer = NoOpExplainer()
    model = PassThroughModel()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    explanation = explainer.explain(model, df)

    assert explanation["status"] == "no_explanation"
    assert explanation["shap_values"].shape == (2, 2)
    assert explainer.supports_model(model)


def test_noop_metric_works():
    """Test NoOp metric returns placeholder values."""
    metric = NoOpMetric()
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])

    result = metric.compute(y_true, y_pred)

    assert "noop_metric" in result
    assert result["samples_processed"] == 4
    assert not metric.requires_sensitive_features()


def test_registry_registration():
    """Test components are registered."""
    # Check PassThrough model is registered
    model_cls = ModelRegistry.get("passthrough")
    assert model_cls is not None
    assert model_cls == PassThroughModel

    # Check NoOp explainer is registered
    explainer_cls = ExplainerRegistry.get("noop")
    assert explainer_cls is not None
    assert explainer_cls == NoOpExplainer

    # Check NoOp metric is registered
    metric_cls = MetricRegistry.get("noop")
    assert metric_cls is not None
    assert metric_cls == NoOpMetric


def test_deterministic_explainer_selection():
    """Test explainer selection is deterministic."""
    config1 = {"explainers": {"priority": ["noop", "nonexistent"]}}
    config2 = {"explainers": {"priority": ["noop", "nonexistent"]}}  # Same order

    selected1 = select_explainer("xgboost", config1)
    selected2 = select_explainer("xgboost", config2)

    assert selected1 == selected2 == "noop"


def test_list_components():
    """Test listing registered components."""
    components = list_components()

    assert "models" in components
    assert "passthrough" in components["models"]

    assert "explainers" in components
    assert "noop" in components["explainers"]

    assert "metrics" in components
    assert "noop" in components["metrics"]


def test_enterprise_feature_flag():
    """Test enterprise feature detection."""
    # Without license key
    assert not is_enterprise()

    # With license key
    with patch.dict("os.environ", {"GLASSALPHA_LICENSE_KEY": "test-key"}):
        assert is_enterprise()


def test_feature_gating_decorator():
    """Test feature gating works correctly."""

    @check_feature("test_feature")
    def enterprise_only_function():
        return "enterprise_result"

    # Should raise without license
    with pytest.raises(FeatureNotAvailable) as exc:
        enterprise_only_function()
    assert "test_feature" in str(exc.value)

    # Should work with license
    with patch.dict("os.environ", {"GLASSALPHA_LICENSE_KEY": "test-key"}):
        result = enterprise_only_function()
        assert result == "enterprise_result"


def test_registry_priority_selection():
    """Test priority-based selection with fallback."""

    # Register a higher priority explainer
    @ExplainerRegistry.register("test_explainer", priority=100)
    class TestExplainer:
        capabilities = {"supported_models": ["test_model"]}
        version = "1.0.0"
        priority = 100

    config = {"explainers": {"priority": ["test_explainer", "noop"]}}

    # Should select test_explainer for compatible model
    selected = select_explainer("test_model", config)
    assert selected == "test_explainer"

    # Should fall back to noop for incompatible model
    selected = select_explainer("unknown_model", config)
    assert selected == "noop"  # NoOp supports all models


def test_enterprise_component_filtering():
    """Test enterprise components are filtered correctly."""

    # Register an enterprise component
    @MetricRegistry.register("enterprise_metric", enterprise=True)
    class EnterpriseMetric:
        metric_type = "enterprise"
        version = "1.0.0"

    # Without enterprise flag, shouldn't see enterprise metric
    oss_metrics = MetricRegistry.get_all(include_enterprise=False)
    assert "enterprise_metric" not in oss_metrics
    assert "noop" in oss_metrics

    # With enterprise flag, should see all
    all_metrics = MetricRegistry.get_all(include_enterprise=True)
    assert "enterprise_metric" in all_metrics
    assert "noop" in all_metrics
