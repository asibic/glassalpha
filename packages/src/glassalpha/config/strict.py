"""Strict mode validation for regulatory compliance.

This module enforces additional requirements when strict mode is enabled,
ensuring all necessary fields are present and determinism is guaranteed.
"""

import logging
import warnings
from typing import Any

from .schema import AuditConfig

logger = logging.getLogger(__name__)


class StrictModeError(ValueError):
    """Raised when strict mode validation fails."""

    pass


def validate_strict_mode(config: AuditConfig) -> None:
    """Validate configuration meets strict mode requirements.

    Strict mode enforces:
    - Explicit random seeds (no defaults)
    - Locked data schema (no inference)
    - Full manifest generation
    - Deterministic plugin selection
    - No configuration defaults
    - All optional fields become required
    - Warnings become errors

    Args:
        config: Configuration to validate

    Raises:
        StrictModeError: If validation fails

    """
    logger.info("Validating configuration for strict mode compliance")
    errors: list[str] = []

    # Check reproducibility settings
    if config.reproducibility.random_seed is None:
        errors.append("Explicit random seed is required in strict mode")

    if not config.reproducibility.deterministic:
        errors.append("Deterministic mode must be enabled in strict mode")

    if not config.reproducibility.capture_environment:
        errors.append("Environment capture must be enabled in strict mode")

    # Check data configuration
    if not config.data.path:
        errors.append("Data path must be specified in strict mode")

    if not config.data.schema_path and not config.data.schema:
        errors.append("Data schema must be specified (either path or inline) in strict mode")

    if not config.data.protected_attributes:
        errors.append("Protected attributes must be specified for fairness analysis in strict mode")

    if not config.data.target_column:
        errors.append("Target column must be explicitly specified in strict mode")

    # Check model configuration
    if not config.model.path:
        errors.append("Model path must be specified in strict mode")

    # Check explainer configuration
    if not config.explainers.priority:
        errors.append("Explainer priority list must be specified in strict mode")

    if config.explainers.strategy != "first_compatible":
        errors.append("Explainer strategy must be 'first_compatible' for determinism in strict mode")

    # Check manifest configuration
    if not config.manifest.enabled:
        errors.append("Manifest generation must be enabled in strict mode")

    if not config.manifest.include_git_sha:
        errors.append("Git SHA must be included in manifest in strict mode")

    if not config.manifest.include_config_hash:
        errors.append("Config hash must be included in manifest in strict mode")

    if not config.manifest.include_data_hash:
        errors.append("Data hash must be included in manifest in strict mode")

    # Check audit profile
    if not config.audit_profile:
        errors.append("Audit profile must be specified in strict mode")

    # Check metrics are specified
    if not config.metrics.performance:
        errors.append("Performance metrics must be specified in strict mode")

    if not config.metrics.fairness:
        errors.append("Fairness metrics must be specified in strict mode")

    # Check recourse if enabled
    if config.recourse.enabled and not config.recourse.immutable_features:
        errors.append("Immutable features must be specified when recourse is enabled in strict mode")

    # Convert warnings to errors
    warnings.simplefilter('error')

    # Report all errors
    if errors:
        error_msg = "Strict mode validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise StrictModeError(error_msg)

    logger.info("Strict mode validation passed")


def validate_deterministic_config(config: dict[str, Any]) -> bool:
    """Validate that configuration ensures deterministic behavior.

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is deterministic

    """
    deterministic = True

    # Check for random elements
    if config.get('reproducibility', {}).get('random_seed') is None:
        logger.warning("No random seed specified - results may vary")
        deterministic = False

    # Check for sorted operations
    explainer_priority = config.get('explainers', {}).get('priority', [])
    if not explainer_priority:
        logger.warning("No explainer priority specified - selection may vary")
        deterministic = False

    # Check for unordered collections
    metrics = config.get('metrics', {})
    for category in ['performance', 'fairness', 'drift']:
        if category in metrics and isinstance(metrics[category], set):
            logger.warning(f"Metrics in '{category}' use set - order may vary")
            deterministic = False

    return deterministic


def validate_reproducible_environment() -> bool:
    """Check if environment supports reproducible execution.

    Returns:
        True if environment is reproducible

    """
    import platform

    reproducible = True
    issues = []

    # Check Python version

    # Check for known non-deterministic libraries
    try:
        import tensorflow
        issues.append("TensorFlow detected - may introduce randomness")
        reproducible = False
    except ImportError:
        pass

    # Check platform
    if platform.system() == "Windows":
        logger.warning("Windows platform may have path handling differences")

    if issues:
        logger.warning("Environment issues for reproducibility: " + ", ".join(issues))

    return reproducible


def enforce_strict_defaults(config: AuditConfig) -> AuditConfig:
    """Apply strict mode defaults to configuration.

    This modifies the configuration to use the most conservative
    settings for regulatory compliance.

    Args:
        config: Configuration to modify

    Returns:
        Modified configuration

    """
    # Force deterministic settings
    config.reproducibility.deterministic = True
    config.reproducibility.capture_environment = True

    # Force manifest generation
    config.manifest.enabled = True
    config.manifest.include_git_sha = True
    config.manifest.include_config_hash = True
    config.manifest.include_data_hash = True
    config.manifest.include_model_hash = True

    # Force deterministic explainer selection
    config.explainers.strategy = "first_compatible"

    # Enable all auditing features
    config.report.include_sections = [
        "lineage",
        "data_schema",
        "global_explanations",
        "local_explanations",
        "fairness",
        "drift",
        "recourse",
        "assumptions",
    ]

    logger.info("Applied strict mode defaults")
    return config
