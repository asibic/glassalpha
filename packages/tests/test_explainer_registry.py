"""Explainer registry and compatibility tests.

Tests for explainer registration, discovery, and model compatibility checking.
"""

from glassalpha.core import ExplainerRegistry
from glassalpha.explain.shap.kernel import KernelSHAPExplainer
from glassalpha.explain.shap.tree import TreeSHAPExplainer


class TestExplainerRegistry:
    """Test explainer registration and discovery."""

    def test_shap_explainers_are_registered(self):
        """Test that SHAP explainers are properly registered."""
        components = ExplainerRegistry.get_all()

        # Should include SHAP explainers
        assert "treeshap" in components
        assert "kernelshap" in components

    def test_get_shap_explainer_classes(self):
        """Test that we can retrieve SHAP explainer classes."""
        tree_cls = ExplainerRegistry.get("treeshap")
        assert tree_cls == TreeSHAPExplainer

        kernel_cls = ExplainerRegistry.get("kernelshap")
        assert kernel_cls == KernelSHAPExplainer

    def test_explainer_priorities(self):
        """Test that explainers have expected priorities."""
        tree_cls = ExplainerRegistry.get("treeshap")
        kernel_cls = ExplainerRegistry.get("kernelshap")

        # TreeSHAP should have higher priority
        assert tree_cls.priority == 100
        assert kernel_cls.priority == 50
        assert tree_cls.priority > kernel_cls.priority


class TestExplainerCompatibility:
    """Test explainer compatibility with different model types."""

    def test_explainer_selection_by_priority(self):
        """Test that explainers are selected by priority."""
        # TreeSHAP has higher priority than KernelSHAP
        tree_cls = ExplainerRegistry.get("treeshap")
        kernel_cls = ExplainerRegistry.get("kernelshap")

        assert tree_cls.priority > kernel_cls.priority

    def test_explainer_compatibility_filtering(self):
        """Test filtering explainers by model compatibility."""
        tree_explainer = TreeSHAPExplainer()
        kernel_explainer = KernelSHAPExplainer()

        # For tree model, TreeSHAP should be compatible
        assert tree_explainer.is_compatible("xgboost") is True
        assert kernel_explainer.is_compatible("xgboost") is True

        # For linear model, only KernelSHAP should be compatible
        assert tree_explainer.is_compatible("logistic_regression") is False
        assert kernel_explainer.is_compatible("logistic_regression") is True
