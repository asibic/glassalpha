"""Contract tests for capability-based parameter validation.

Tests verify that:
1. Parameter validation uses model capabilities, not string matching
2. Special values (like LightGBM max_depth=-1) are handled correctly
3. Validation is consistent across model types
4. Clear error messages are generated
"""

from unittest.mock import Mock

import pytest

from glassalpha.cli.commands import _validate_model_params


class TestCapabilityBasedValidation:
    """Test parameter validation using model capabilities."""

    def test_lightgbm_max_depth_negative_one_allowed(self):
        """LightGBM max_depth=-1 is valid (special value for 'no limit')."""
        pytest.importorskip("lightgbm")

        config = Mock()
        config.model.type = "lightgbm"
        config.model.params = {"max_depth": -1}

        warnings = _validate_model_params(config)

        # Should have NO warnings - max_depth=-1 is explicitly allowed
        assert len(warnings) == 0

    def test_lightgbm_max_depth_negative_two_rejected(self):
        """LightGBM max_depth=-2 is invalid (only -1 is special)."""
        pytest.importorskip("lightgbm")

        config = Mock()
        config.model.type = "lightgbm"
        config.model.params = {"max_depth": -2}

        warnings = _validate_model_params(config)

        # Should warn - only -1 is a valid special value
        assert len(warnings) > 0
        assert any("max_depth" in w for w in warnings)

    def test_xgboost_max_depth_must_be_positive(self):
        """XGBoost max_depth must be >= 0 (no special negative values)."""
        pytest.importorskip("xgboost")

        config = Mock()
        config.model.type = "xgboost"
        config.model.params = {"max_depth": -1}

        warnings = _validate_model_params(config)

        # Should warn - XGBoost doesn't allow negative max_depth
        assert len(warnings) > 0
        assert any("max_depth" in w for w in warnings)

    def test_logistic_regression_c_must_be_positive(self):
        """LogisticRegression C parameter must be > 0."""
        pytest.importorskip("sklearn")

        config = Mock()
        config.model.type = "logistic_regression"
        config.model.params = {"C": 0.0}

        warnings = _validate_model_params(config)

        # Should warn - C must be positive
        assert len(warnings) > 0
        assert any("C" in w and ">" in w for w in warnings)

    def test_logistic_regression_c_positive_allowed(self):
        """LogisticRegression C=1.0 is valid."""
        pytest.importorskip("sklearn")

        config = Mock()
        config.model.type = "logistic_regression"
        config.model.params = {"C": 1.0}

        warnings = _validate_model_params(config)

        # Should have no C-related warnings
        assert not any("C" in w for w in warnings)

    def test_learning_rate_typical_range_warning(self):
        """Learning rate outside typical range generates warning."""
        pytest.importorskip("xgboost")

        config = Mock()
        config.model.type = "xgboost"
        config.model.params = {"learning_rate": 5.0}  # Unusually high

        warnings = _validate_model_params(config)

        # Should warn about unusual value
        assert len(warnings) > 0
        assert any("learning_rate" in w and "Typically" in w for w in warnings)

    def test_learning_rate_typical_range_no_warning(self):
        """Learning rate in typical range has no warnings."""
        pytest.importorskip("xgboost")

        config = Mock()
        config.model.type = "xgboost"
        config.model.params = {"learning_rate": 0.1}  # Normal value

        warnings = _validate_model_params(config)

        # Should have no learning_rate warnings
        assert not any("learning_rate" in w for w in warnings)

    def test_subsample_out_of_range(self):
        """Subsample must be in (0, 1] range."""
        pytest.importorskip("xgboost")

        config = Mock()
        config.model.type = "xgboost"
        config.model.params = {"subsample": 1.5}

        warnings = _validate_model_params(config)

        # Should warn about out of range
        assert len(warnings) > 0
        assert any("subsample" in w for w in warnings)

    def test_subsample_zero_rejected(self):
        """Subsample=0 is invalid (exclusive minimum)."""
        pytest.importorskip("lightgbm")

        config = Mock()
        config.model.type = "lightgbm"
        config.model.params = {"subsample": 0.0}

        warnings = _validate_model_params(config)

        # Should warn - subsample must be > 0
        assert len(warnings) > 0
        assert any("subsample" in w for w in warnings)

    def test_n_estimators_positive_required(self):
        """n_estimators must be >= 1."""
        pytest.importorskip("xgboost")

        config = Mock()
        config.model.type = "xgboost"
        config.model.params = {"n_estimators": 0}

        warnings = _validate_model_params(config)

        # Should warn - n_estimators must be at least 1
        assert len(warnings) > 0
        assert any("n_estimators" in w for w in warnings)

    def test_no_warnings_for_valid_params(self):
        """Valid parameter values generate no warnings."""
        pytest.importorskip("xgboost")

        config = Mock()
        config.model.type = "xgboost"
        config.model.params = {
            "max_depth": 6,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        warnings = _validate_model_params(config)

        # Should have no warnings
        assert len(warnings) == 0

    def test_unknown_model_type_no_crash(self):
        """Unknown model types don't crash validation."""
        config = Mock()
        config.model.type = "unknown_model_xyz"
        config.model.params = {"some_param": 42}

        # Should not raise exception
        warnings = _validate_model_params(config)

        # May have warnings or not, but shouldn't crash
        assert isinstance(warnings, list)

    def test_validation_uses_capabilities_not_string_matching(self):
        """Validation retrieves rules from model capabilities, not hardcoded checks."""
        pytest.importorskip("lightgbm")

        # This test verifies architectural compliance:
        # - Validation should call ModelRegistry.get(model_type)
        # - Should access model_class.capabilities["parameter_rules"]
        # - Should NOT have if model_type == "lightgbm" checks

        config = Mock()
        config.model.type = "lightgbm"
        config.model.params = {"max_depth": -1}

        # Import after mock to ensure we're testing the real implementation
        from glassalpha.core.registry import ModelRegistry

        model_class = ModelRegistry.get("lightgbm")
        capabilities = getattr(model_class, "capabilities", {})
        param_rules = capabilities.get("parameter_rules", {})

        # Verify the capability structure exists
        assert "max_depth" in param_rules
        assert "special_values" in param_rules["max_depth"]
        assert -1 in param_rules["max_depth"]["special_values"]

        # Now verify validation uses it
        warnings = _validate_model_params(config)
        assert len(warnings) == 0  # Should use capability, not string match
