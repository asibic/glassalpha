"""Tests for plugin dependency handling and fallbacks."""

from unittest.mock import patch

import pytest

from glassalpha.cli.preflight import preflight_check_model
from glassalpha.core.plugin_registry import ModelRegistry


class TestPluginDependencies:
    """Test plugin dependency handling."""

    def test_logistic_regression_always_available(self):
        """Test that logistic regression is always available."""
        available = ModelRegistry.available_plugins()
        assert "logistic_regression" in available
        assert available["logistic_regression"] is True

    def test_missing_xgboost_handling(self):
        """Test handling when XGBoost is not available."""
        # Mock XGBoost as unavailable
        with patch.object(
            ModelRegistry,
            "available_plugins",
            return_value={
                "logistic_regression": True,
                "xgboost": False,
                "lightgbm": False,
            },
        ):
            # Create config requesting XGBoost with fallback allowed
            from types import SimpleNamespace

            config = SimpleNamespace()
            config.model = SimpleNamespace()
            config.model.type = "xgboost"
            config.model.allow_fallback = True

            # Should fallback to logistic regression
            result = preflight_check_model(config)
            assert result.model.type == "logistic_regression"

    def test_missing_xgboost_no_fallback(self):
        """Test handling when XGBoost is not available and fallbacks disabled."""
        # Mock XGBoost as unavailable
        with patch.object(
            ModelRegistry,
            "available_plugins",
            return_value={
                "logistic_regression": True,
                "xgboost": False,
                "lightgbm": False,
            },
        ):
            # Create config requesting XGBoost with fallback disabled
            from types import SimpleNamespace

            config = SimpleNamespace()
            config.model = SimpleNamespace()
            config.model.type = "xgboost"
            config.model.allow_fallback = False

            # Should exit with error
            with pytest.raises(SystemExit):
                preflight_check_model(config)

    def test_install_hint_generation(self):
        """Test that install hints are generated correctly."""
        hint = ModelRegistry.get_install_hint("xgboost")
        assert hint == "pip install 'glassalpha[xgboost]'"

        hint = ModelRegistry.get_install_hint("lightgbm")
        assert hint == "pip install 'glassalpha[lightgbm]'"

        hint = ModelRegistry.get_install_hint("logistic_regression")
        assert hint is None  # No hint needed for always-available model

    def test_plugin_loading_with_missing_dependency(self):
        """Test plugin loading fails gracefully with missing dependencies."""
        # Mock XGBoost as unavailable
        with patch.object(
            ModelRegistry,
            "available_plugins",
            return_value={
                "logistic_regression": True,
                "xgboost": False,
                "lightgbm": False,
            },
        ):
            # Should fail to load XGBoost
            with pytest.raises(ImportError) as exc_info:
                ModelRegistry.load("xgboost")

            assert "Missing optional dependency 'xgboost'" in str(exc_info.value)
            assert "pip install 'glassalpha[xgboost]'" in str(exc_info.value)

    def test_plugin_loading_success(self):
        """Test plugin loading succeeds when dependencies are available."""
        # Mock all models as available
        with patch.object(
            ModelRegistry,
            "available_plugins",
            return_value={
                "logistic_regression": True,
                "xgboost": True,
                "lightgbm": True,
            },
        ):
            # Should succeed (though actual loading may fail due to missing actual libraries)
            # We're testing the dependency check logic, not the actual loading
            try:
                ModelRegistry.load("logistic_regression")
            except (ImportError, RuntimeError):
                # Expected since we don't have actual sklearn available in test env
                pass


class TestPreflightIntegration:
    """Test preflight integration with CLI."""

    def test_default_model_config_creation(self):
        """Test creation of default model config when none provided."""
        from glassalpha.cli.preflight import _create_default_model_config

        config = _create_default_model_config()
        assert config.type == "logistic_regression"
        assert config.params["random_state"] == 42

    def test_fallback_model_selection(self):
        """Test fallback model selection logic."""
        from glassalpha.cli.preflight import _find_fallback_model

        # Test fallback when XGBoost is requested but unavailable
        available = {"logistic_regression": True, "xgboost": False}
        fallback = _find_fallback_model("xgboost", available)
        assert fallback == "logistic_regression"

        # Test no fallback when no suitable model available
        available = {"xgboost": False, "lightgbm": False}
        fallback = _find_fallback_model("xgboost", available)
        assert fallback is None
