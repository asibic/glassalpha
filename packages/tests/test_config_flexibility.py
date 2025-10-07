"""Test configuration flexibility with warnings for non-critical sections."""

from unittest.mock import patch

import pytest

from glassalpha.config.loader import load_config
from glassalpha.config.warnings import (
    check_config_security,
    suggest_config_improvements,
    validate_config_completeness,
    warn_unknown_keys,
)


def test_unknown_keys_in_report_section_logged():
    """Test that unknown keys in report section are logged but not fatal."""
    config_dict = {
        "audit_profile": "test_profile",
        "data": {
            "dataset": "custom",
            "path": "test.csv",
            "target_column": "target",
            "protected_attributes": ["age"],
        },
        "model": {"type": "xgboost"},
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"]},
        "report": {
            "template": "standard_audit.html",
            "styling": {  # Unknown section
                "color_scheme": "professional",
                "font_family": "Arial",
            },
            "unknown_option": "value",  # Unknown key
        },
    }

    with patch("glassalpha.config.loader.logger") as mock_logger:
        # Should not raise error
        config = load_config(config_dict)

        # Should have logged warnings about unknown keys
        mock_logger.warning.assert_called()
        warning_calls = [call.args[0] for call in mock_logger.warning.call_args_list]

        # Check that unknown keys were mentioned
        unknown_key_warning = any("unknown" in call.lower() and "report" in call.lower() for call in warning_calls)
        assert unknown_key_warning, f"Expected unknown key warning, got: {warning_calls}"

    # Config should still be valid
    assert config.report.template == "standard_audit.html"


def test_security_warnings_for_user_paths():
    """Test that user-specific paths generate security warnings."""
    config_dict = {
        "data": {
            "dataset": "custom",
            "path": "/Users/john/data/sensitive.csv",  # User-specific path
            "target_column": "target",
        },
        "model": {
            "path": "C:\\Users\\jane\\models\\secret.pkl",  # Windows user path
        },
    }

    with patch("glassalpha.config.warnings.logger") as mock_logger:
        check_config_security(config_dict)

        # Should have logged security warnings
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]

        assert "security" in warning_msg.lower() or "portability" in warning_msg.lower()
        assert "/Users/john" in warning_msg or "C:\\Users\\jane" in warning_msg


def test_config_completeness_validation():
    """Test validation of configuration completeness."""
    # Minimal config missing recommended sections
    minimal_config = {
        "data": {"dataset": "custom", "path": "test.csv", "target_column": "target"},
        # Missing: model, explainers, metrics, report
    }

    with patch("glassalpha.config.warnings.logger") as mock_logger:
        validate_config_completeness(minimal_config)

        # Should have logged completeness warnings
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]

        assert "missing" in warning_msg.lower()
        assert "model" in warning_msg
        assert "explainers" in warning_msg


def test_config_improvement_suggestions():
    """Test configuration improvement suggestions."""
    basic_config = {
        "data": {
            "dataset": "custom",
            "path": "test.csv",
            "target_column": "target",
            # Missing: protected_attributes
        },
        "model": {"type": "xgboost"},
        "explainers": {
            "strategy": "first_compatible",
            "priority": ["treeshap"],  # Only one explainer
        },
        "metrics": {
            "performance": ["accuracy"],
            # Missing: fairness metrics
        },
        # Missing: reproducibility, audit_profile
    }

    with patch("glassalpha.config.warnings.logger") as mock_logger:
        suggest_config_improvements(basic_config)

        # Should have logged improvement suggestions
        mock_logger.info.assert_called()
        suggestion_msg = mock_logger.info.call_args[0][0]

        assert "suggestions" in suggestion_msg.lower()
        assert "random_seed" in suggestion_msg
        assert "protected_attributes" in suggestion_msg
        assert "fairness" in suggestion_msg


def test_strict_mode_suppresses_suggestions():
    """Test that strict mode suppresses non-critical warnings."""
    config_dict = {
        "audit_profile": "test_profile",
        "strict_mode": True,
        "data": {
            "dataset": "custom",
            "path": "test.csv",
            "target_column": "target",
            "protected_attributes": ["age"],
            "data_schema": {"feature1": "numeric", "feature2": "categorical"},  # Add schema for strict mode
        },
        "model": {"type": "xgboost", "path": "model.pkl"},  # Add model path for strict mode
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"], "fairness": ["demographic_parity"]},
        "report": {"template": "standard_audit.html"},
        "manifest": {"enabled": True, "include_git_sha": True, "include_config_hash": True, "include_data_hash": True},
        "reproducibility": {"random_seed": 42, "deterministic": True, "capture_environment": True},
        "preprocessing": {
            "mode": "artifact",
            "artifact_path": "test_artifact.joblib",
            "expected_file_hash": "sha256:test_hash",
            "expected_params_hash": "sha256:test_params_hash",
        },
    }

    with patch("glassalpha.config.warnings.validate_config_completeness") as mock_completeness:
        with patch("glassalpha.config.warnings.suggest_config_improvements") as mock_suggestions:
            config = load_config(config_dict, strict=True)

            # In strict mode, suggestions should not be called
            mock_completeness.assert_not_called()
            mock_suggestions.assert_not_called()

            assert config.strict_mode is True


def test_non_strict_mode_shows_suggestions():
    """Test that non-strict mode shows improvement suggestions."""
    config_dict = {
        "audit_profile": "test_profile",
        "data": {
            "dataset": "custom",
            "path": "test.csv",
            "target_column": "target",
            "protected_attributes": ["age"],
        },
        "model": {"type": "xgboost"},
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"]},
        "report": {"template": "standard_audit.html"},
        "manifest": {"enabled": True},
        "reproducibility": {"random_seed": 42},
    }

    with patch("glassalpha.config.warnings.validate_config_completeness") as mock_completeness:
        with patch("glassalpha.config.warnings.suggest_config_improvements") as mock_suggestions:
            _config = load_config(config_dict, strict=False)

            # In non-strict mode, suggestions should be called
            mock_completeness.assert_called_once()
            mock_suggestions.assert_called_once()


def test_warn_unknown_keys_utility():
    """Test the warn_unknown_keys utility function directly."""
    from glassalpha.config.schema import ReportConfig

    raw_config = {
        "report": {
            "template": "standard_audit.html",
            "output_format": "pdf",
            "unknown_key": "value",
            "another_unknown": {"nested": "value"},
        },
    }

    parsed_report = ReportConfig(template="standard_audit.html", output_format="pdf")

    with patch("glassalpha.config.warnings.logger") as mock_logger:
        warn_unknown_keys(raw_config, parsed_report, "report")

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]

        assert "unknown_key" in warning_msg
        assert "another_unknown" in warning_msg
        assert "report" in warning_msg


def test_no_warnings_for_valid_config():
    """Test that valid configuration generates no warnings."""
    valid_config = {
        "audit_profile": "test_profile",
        "data": {
            "dataset": "custom",
            "path": "test.csv",
            "target_column": "target",
            "protected_attributes": ["age"],
        },
        "model": {"type": "xgboost"},
        "explainers": {
            "strategy": "first_compatible",
            "priority": ["treeshap", "kernelshap"],
        },
        "metrics": {
            "performance": ["accuracy"],
            "fairness": ["demographic_parity"],
        },
        "report": {"template": "standard_audit.html"},
        "reproducibility": {"random_seed": 42},
    }

    with patch("glassalpha.config.warnings.logger") as mock_logger:
        config = load_config(valid_config)

        # Should have minimal or no warnings for this complete config
        warning_calls = [
            call for call in mock_logger.warning.call_args_list if "unknown" in str(call) or "deprecated" in str(call)
        ]

        assert len(warning_calls) == 0, f"Unexpected warnings for valid config: {warning_calls}"
        assert config.audit_profile == "test_profile"


def test_config_flexibility_integration():
    """Integration test for configuration flexibility features."""
    # Config with various issues that should generate warnings but not fail
    flexible_config = {
        "audit_profile": "test_profile",
        "reproducibility": {"random_seed": 42},  # Deprecated top-level format
        "data": {
            "dataset": "custom",
            "path": "/Users/testuser/data.csv",  # Security warning
            "target_column": "target",
            "protected_attributes": ["age"],
        },
        "model": {"type": "xgboost"},
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"]},  # Missing fairness
        "report": {
            "template": "standard_audit.html",
            "styling": {"color": "blue"},  # Unknown section
            "custom_footer": "My Company",  # Unknown key
        },
    }

    with patch("glassalpha.config.loader.logger") as mock_loader_logger:
        with patch("glassalpha.config.warnings.logger") as mock_warnings_logger:
            # Should not raise any exceptions
            config = load_config(flexible_config)

            # Should have generated various warnings
            assert mock_warnings_logger.warning.called
            assert mock_loader_logger.warning.called

            # But config should still be valid and usable
            assert config.audit_profile == "test_profile"
            assert config.data.target_column == "target"
            assert config.model.type == "xgboost"
            # Check that report template is set (not necessarily exact string)
            assert config.report.template is not None
            assert "audit" in config.report.template.lower()


if __name__ == "__main__":
    pytest.main([__file__])
