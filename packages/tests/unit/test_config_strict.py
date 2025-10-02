"""Unit tests for config strict mode validation."""

import pytest

from glassalpha.config.schema import AuditConfig, ReportConfig
from glassalpha.config.strict import StrictModeError, validate_strict_mode
from glassalpha.config.warnings import warn_unknown_keys


def test_strict_mode_missing_seed_raises():
    """Test that strict mode raises error when random seed is missing."""
    # Create minimal config with missing random seed
    config_dict = {
        "audit_profile": "tabular_compliance",
        "model": {"type": "logistic_regression", "path": "/tmp/model.pkl"},
        "data": {
            "dataset": "custom",
            "path": "/tmp/data.csv",
            "target_column": "target",
            "protected_attributes": ["gender"],
        },
        "reproducibility": {"random_seed": None},  # Missing seed
        "explainers": {"priority": ["treeshap"], "strategy": "first_compatible"},
        "manifest": {
            "enabled": True,
            "include_git_sha": True,
            "include_config_hash": True,
            "include_data_hash": True,
        },
        "metrics": {"performance": ["accuracy"], "fairness": ["demographic_parity"]},
    }

    config = AuditConfig(**config_dict)
    config.strict_mode = True

    with pytest.raises(StrictModeError, match="Explicit random seed is required"):
        validate_strict_mode(config)


def test_warn_unknown_keys_logs_warning(caplog):
    """Test that unknown keys generate warnings."""
    raw_config = {"report": {"unknown_key": 1, "template": "standard"}}
    report = ReportConfig(template="standard")

    warn_unknown_keys(raw_config, report, "report")

    assert any("unknown_key" in record.message for record in caplog.records)


def test_warn_unknown_keys_no_warning_for_known(caplog):
    """Test that known keys don't generate warnings."""
    raw_config = {"report": {"template": "standard"}}
    report = ReportConfig(template="standard")

    warn_unknown_keys(raw_config, report, "report")

    # Should not have warnings about 'template'
    assert not any("template" in record.message for record in caplog.records if "unknown" in record.message.lower())
