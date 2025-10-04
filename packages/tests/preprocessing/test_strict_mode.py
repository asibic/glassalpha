"""Test strict mode enforcement for preprocessing."""

from pathlib import Path

import pytest

# Import stubs
try:
    from glassalpha.config.schema import AuditConfig, PreprocessingConfig
    from glassalpha.config.strict import validate_strict_mode
    from glassalpha.preprocessing.validation import assert_runtime_versions
except ImportError:
    pytest.skip("preprocessing module not implemented yet", allow_module_level=True)


def test_strict_mode_fails_on_version_mismatch(mismatched_version_manifest: dict):
    """Patch manifest to simulate sklearn 1.3.2 vs runtime 1.5.0.
    In strict profile, expect RuntimeError. In non-strict, UserWarning.
    """
    # Strict mode: should raise
    with pytest.raises(RuntimeError) as exc_info:
        assert_runtime_versions(
            mismatched_version_manifest,
            strict=True,
            allow_minor=False,
        )

    error_msg = str(exc_info.value)
    assert "version" in error_msg.lower() or "mismatch" in error_msg.lower()
    assert "1.3.2" in error_msg or "1.5.0" in error_msg

    # Non-strict mode: should warn but not raise
    with pytest.warns(UserWarning):
        assert_runtime_versions(
            mismatched_version_manifest,
            strict=False,
            allow_minor=True,
        )


def test_strict_profile_blocks_auto_mode():
    """With profile=tabular_compliance, preprocessing.mode='auto' must error before running."""
    # Create config with auto mode
    config = AuditConfig(
        audit_profile="tabular_compliance",
        preprocessing=PreprocessingConfig(mode="auto"),
        model={"path": "dummy.pkl"},
        data={"path": "dummy.csv"},
    )

    # Strict validation should fail
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        validate_strict_mode(config, strict=True)

    error_msg = str(exc_info.value)
    assert "auto" in error_msg.lower() or "artifact" in error_msg.lower()
    assert "strict" in error_msg.lower() or "compliance" in error_msg.lower()


def test_strict_mode_requires_hashes():
    """Strict mode must require both file_hash and params_hash."""
    # Missing file_hash
    config = AuditConfig(
        audit_profile="tabular_compliance",
        preprocessing=PreprocessingConfig(
            mode="artifact",
            artifact_path=Path("dummy.pkl"),
            expected_params_hash="sha256:abc123",
            # expected_file_hash missing
        ),
        model={"path": "dummy.pkl"},
        data={"path": "dummy.csv"},
    )

    with pytest.raises((ValueError, RuntimeError)):
        validate_strict_mode(config, strict=True)

    # Missing params_hash
    config.preprocessing.expected_file_hash = "sha256:def456"
    config.preprocessing.expected_params_hash = None

    with pytest.raises((ValueError, RuntimeError)):
        validate_strict_mode(config, strict=True)
