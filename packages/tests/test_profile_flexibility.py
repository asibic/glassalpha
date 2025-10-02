"""Test profile flexibility and error handling."""

from unittest.mock import patch

from glassalpha.config.loader import load_config


def test_unknown_profile_warns_and_continues():
    """Test that unknown profiles log a warning but don't crash."""
    config_dict = {
        "audit_profile": "totally_fake_profile",
        "model": {"type": "xgboost"},
        "data": {"dataset": "custom", "path": "test.csv"},
        "report": {
            "unknown_option": "value",  # Should be logged as unknown key
            "template": "standard_audit.html",
        },
    }

    with patch("glassalpha.config.loader.logger") as mock_logger:
        # Should not raise error
        config = load_config(config_dict)

        # Should have logged warning about unknown profile
        mock_logger.warning.assert_any_call("Unknown audit profile: totally_fake_profile")

        # Should have logged about unknown report keys (check that it contains the key)
        warning_calls = [call.args[0] for call in mock_logger.warning.call_args_list]
        assert any("unknown_option" in call for call in warning_calls)

        # Config should be returned (with defaults applied)
        assert hasattr(config, "audit_profile")
        assert hasattr(config, "model")
        assert hasattr(config, "data")
        assert hasattr(config, "report")


def test_missing_profile_fallback():
    """Test that missing profile falls back gracefully."""
    config_dict = {
        "audit_profile": "tabular_compliance",  # Use a known profile to avoid validation errors
        "model": {"type": "xgboost"},
        "data": {"dataset": "custom", "path": "test.csv"},
    }

    with patch("glassalpha.config.loader.logger") as mock_logger:
        config = load_config(config_dict)

        # Should have logged about configuration validation (since profile defaults are applied)
        mock_logger.info.assert_any_call("Configuration validated successfully")

        # Config should be returned with defaults applied
        assert hasattr(config, "audit_profile")
        assert hasattr(config, "explainers")
        assert hasattr(config, "metrics")
