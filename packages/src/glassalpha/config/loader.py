"""Configuration loading and validation.

This module handles loading configuration from YAML files and
validating against the schema and audit profile requirements.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from ..core.registry import ProfileRegistry
from .schema import AuditConfig

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML content

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration (deep merge)

    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursive merge for nested dicts
            result[key] = merge_configs(result[key], value)
        else:
            # Direct override
            result[key] = value

    return result


def apply_profile_defaults(config: dict[str, Any], profile_name: str | None = None) -> dict[str, Any]:
    """Apply audit profile defaults to configuration.

    Args:
        config: Configuration dictionary
        profile_name: Override profile name (default: from config)

    Returns:
        Configuration with profile defaults applied

    """
    profile_name = profile_name or config.get("audit_profile")
    if not profile_name:
        logger.warning("No audit profile specified, using raw config")
        return config

    # Get profile from registry
    profile_cls = ProfileRegistry.get(profile_name)
    if not profile_cls:
        logger.warning(f"Unknown audit profile: {profile_name}")
        return config

    # Get default config from profile
    if hasattr(profile_cls, "get_default_config"):
        default_config = profile_cls.get_default_config()
        # Merge with user config (user config takes precedence)
        config = merge_configs(default_config, config)
        logger.info(f"Applied defaults from profile '{profile_name}'")

    # Validate against profile requirements
    if hasattr(profile_cls, "validate_config"):
        try:
            profile_cls.validate_config(config)
            logger.info(f"Configuration validated against profile '{profile_name}'")
        except ValueError as e:
            logger.error(f"Configuration invalid for profile '{profile_name}': {e}")
            raise

    return config


def validate_config(config: dict[str, Any] | AuditConfig) -> AuditConfig:
    """Validate configuration against schema.

    Args:
        config: Configuration dictionary or AuditConfig object

    Returns:
        Validated AuditConfig object

    Raises:
        ValueError: If configuration is invalid

    """
    if isinstance(config, AuditConfig):
        return config

    try:
        # Apply profile defaults first
        config = apply_profile_defaults(config, config.get("audit_profile"))

        # Create and validate config object
        audit_config = AuditConfig(**config)
        logger.info("Configuration validated successfully")
        return audit_config

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ValueError(f"Invalid configuration: {e}") from e


def load_config(config_dict: dict[str, Any], profile_name: str | None = None, strict: bool = False) -> AuditConfig:
    """Load configuration from dictionary.

    Args:
        config_dict: Configuration dictionary
        profile_name: Override audit profile
        strict: Enable strict mode validation

    Returns:
        Validated AuditConfig object

    """
    # Override profile if specified
    if profile_name:
        config_dict["audit_profile"] = profile_name

    # Override strict mode if specified
    if strict:
        config_dict["strict_mode"] = True

    # Import warning utilities
    from .warnings import (  # noqa: PLC0415
        check_config_security,
        suggest_config_improvements,
        validate_config_completeness,
        warn_deprecated_options,
        warn_unknown_keys,
    )

    # Check for deprecated options and security issues
    warn_deprecated_options(config_dict)
    check_config_security(config_dict)

    # Apply defaults and validate
    config_dict = apply_profile_defaults(config_dict, config_dict.get("audit_profile"))
    audit_config = validate_config(config_dict)

    # Warn about unknown keys in non-critical sections
    warn_unknown_keys(config_dict, audit_config.report, "report")
    if hasattr(audit_config, "recourse") and audit_config.recourse:
        warn_unknown_keys(config_dict, audit_config.recourse, "recourse")

    # Validate completeness and suggest improvements (unless in strict mode)
    if not (audit_config.strict_mode or strict):
        validate_config_completeness(config_dict)
        suggest_config_improvements(config_dict)

    # Apply strict mode validation if enabled
    if audit_config.strict_mode or strict:
        from .strict import validate_strict_mode  # noqa: PLC0415

        validate_strict_mode(audit_config)

    logger.info("Configuration validated successfully")
    return audit_config


def load_config_from_file(
    path: str | Path,
    override_path: str | Path | None = None,
    profile_name: str | None = None,
    strict: bool = False,
) -> AuditConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to main configuration file
        override_path: Optional path to override configuration
        profile_name: Override audit profile
        strict: Enable strict mode validation

    Returns:
        Validated AuditConfig object

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid

    """
    logger.info(f"Loading configuration from {path}")

    # Load main config
    config_dict = load_yaml(path)

    # Load and merge override config if provided
    if override_path:
        logger.info(f"Loading override configuration from {override_path}")
        override_dict = load_yaml(override_path)
        config_dict = merge_configs(config_dict, override_dict)

    # Load and validate
    return load_config(config_dict, profile_name, strict)


def save_config(config: AuditConfig, path: str | Path, include_defaults: bool = False) -> None:
    """Save audit configuration to YAML file for reproducible compliance audits.

    Serializes the complete audit configuration including model settings,
    data parameters, and compliance policies to enable audit trail documentation
    and configuration version control.

    Args:
        config: Complete audit configuration object with validated settings
        path: Target file path for configuration storage
        include_defaults: Include default values for complete audit trail documentation

    Side Effects:
        - Creates or overwrites YAML file at specified path
        - May create parent directories if they don't exist
        - File written with UTF-8 encoding for international compliance

    Raises:
        IOError: If path is not writable or insufficient disk space
        ValidationError: If configuration contains invalid settings

    Note:
        For regulatory compliance, always save with include_defaults=True to
        maintain complete audit trail of configuration decisions.

    """
    path = Path(path)

    # Convert to dictionary - use ternary for cleaner code
    config_dict = config.model_dump() if include_defaults else config.model_dump(exclude_defaults=True)

    # Write YAML
    with open(path, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved to {path}")
