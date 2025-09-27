"""High-risk explainer selection tests - critical path validation.

These tests target the actual risk areas in explainer selection:
- Specific model types must resolve to expected explainers
- Registry must handle mock objects properly
- Priority ordering must be deterministic
"""

import pytest


class TestExplainerSelectionRisk:
    """Test explainer selection critical paths that prevent customer issues."""

    def test_xgboost_selects_treeshap_by_default(self):
        """XGBoost must select TreeSHAP by default - customer expectation."""
        from glassalpha.explain.registry import ExplainerRegistry

        explainer_class = ExplainerRegistry.find_compatible("xgboost")
        assert explainer_class is not None, "XGBoost must have compatible explainer"
        assert explainer_class.__name__ == "TreeSHAPExplainer", "XGBoost should prefer TreeSHAP"

    def test_logistic_regression_selects_kernelshap(self):
        """LogisticRegression must select KernelSHAP - tree explainers don't work."""
        from glassalpha.explain.registry import ExplainerRegistry

        explainer_class = ExplainerRegistry.find_compatible("logistic_regression")
        assert explainer_class is not None, "LogisticRegression must have compatible explainer"
        assert explainer_class.__name__ == "KernelSHAPExplainer", "LogisticRegression needs KernelSHAP"

    def test_lightgbm_priority_selection(self):
        """LightGBM should prefer TreeSHAP when available."""
        from glassalpha.explain.registry import ExplainerRegistry

        explainer_class = ExplainerRegistry.find_compatible("lightgbm")
        assert explainer_class is not None, "LightGBM must have compatible explainer"
        # Should pick TreeSHAP due to priority ordering
        assert explainer_class.__name__ == "TreeSHAPExplainer", "LightGBM should prefer TreeSHAP"

    def test_unsupported_model_raises_runtime_error(self):
        """Unsupported models must raise RuntimeError with exact message."""
        from glassalpha.explain.registry import ExplainerRegistry

        with pytest.raises(RuntimeError, match="No compatible explainer found"):
            ExplainerRegistry.select_for_model("completely_unknown_model")

    def test_mock_model_object_compatibility(self):
        """Model objects with get_model_info should work correctly."""
        from glassalpha.explain.registry import ExplainerRegistry

        class MockModel:
            def get_model_info(self):
                return {"type": "xgboost"}

        mock = MockModel()
        explainer_class = ExplainerRegistry.find_compatible(mock)
        assert explainer_class is not None, "Mock XGBoost should be compatible"
        assert explainer_class.__name__ == "TreeSHAPExplainer"

    def test_none_model_info_returns_none(self):
        """Objects without valid model info should return None gracefully."""
        from glassalpha.explain.registry import ExplainerRegistry

        class BadMock:
            def get_model_info(self):
                return None

        bad_mock = BadMock()
        explainer_class = ExplainerRegistry.find_compatible(bad_mock)
        assert explainer_class is None, "Invalid model info should return None"

    def test_explainer_registry_deterministic_selection(self):
        """Same input must always select same explainer - reproducibility critical."""
        from glassalpha.explain.registry import ExplainerRegistry

        # Test multiple times to ensure deterministic
        selections = []
        for _ in range(5):
            explainer_class = ExplainerRegistry.find_compatible("xgboost")
            selections.append(explainer_class.__name__ if explainer_class else None)

        # All selections should be identical
        assert len(set(selections)) == 1, f"Selection not deterministic: {selections}"
        assert selections[0] == "TreeSHAPExplainer", "Should consistently select TreeSHAP"
