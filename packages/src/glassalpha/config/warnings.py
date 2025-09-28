"""Configuration warning utilities for unknown keys and deprecated options."""

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def warn_unknown_keys(raw_config: dict[str, Any], parsed_model: BaseModel, section: str) -> None:
    """Warn about unknown configuration keys in non-critical sections.

    This function helps users identify typos or deprecated configuration options
    without failing the entire audit process.

    Args:
        raw_config: Raw configuration dictionary from YAML
        parsed_model: Parsed Pydantic model
        section: Configuration section name for logging

    """
    if section not in raw_config:
        return

    raw_section = raw_config[section]
    if not isinstance(raw_section, dict):
        return

    # Get known keys from the parsed model
    known_keys = set(parsed_model.model_dump().keys())
    raw_keys = set(raw_section.keys())

    # Find unknown keys
    unknown_keys = raw_keys - known_keys

    if unknown_keys:
        logger.warning(
            f"Ignoring unknown {section} configuration keys: {sorted(unknown_keys)}. Check for typos or deprecated options. Known keys: {sorted(known_keys)}",
        )


def warn_deprecated_options(config_dict: dict[str, Any]) -> None:
    """Warn about deprecated configuration options.

    Args:
        config_dict: Full configuration dictionary

    """
    deprecated_mappings = {
        # Old key -> (new key, section)
        "random_seed": ("reproducibility.random_seed", "reproducibility"),
        "target": ("data.target_column", "data"),
        "protected_attrs": ("data.protected_attributes", "data"),
        "model_type": ("model.type", "model"),
        "model_params": ("model.params", "model"),
        "explainer_type": ("explainers.priority", "explainers"),
        "metrics_config": ("metrics", "metrics"),
        "report_template": ("report.template", "report"),
    }

    deprecated_found = []

    def check_nested_keys(obj: Any, prefix: str = "") -> None:
        """Recursively check for deprecated keys."""
        if not isinstance(obj, dict):
            return

        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if key in deprecated_mappings and prefix == "":  # Only check top-level keys
                old_key, new_section = deprecated_mappings[key]
                deprecated_found.append((full_key, old_key))

            if isinstance(value, dict):
                check_nested_keys(value, full_key)

    check_nested_keys(config_dict)

    if deprecated_found:
        warning_msg = "Deprecated configuration options found:\n"
        for old_key, new_key in deprecated_found:
            warning_msg += f"  • '{old_key}' is deprecated, use '{new_key}' instead\n"
        warning_msg += "These options still work but may be removed in future versions."
        logger.warning(warning_msg.rstrip())


def validate_config_completeness(config_dict: dict[str, Any]) -> None:
    """Validate that configuration has all required sections for production use.

    Args:
        config_dict: Full configuration dictionary

    """
    required_sections = ["data", "model", "explainers", "metrics", "report"]
    missing_sections = [section for section in required_sections if section not in config_dict]

    if missing_sections:
        logger.warning(
            f"Configuration missing recommended sections: {missing_sections}. This may limit audit functionality.",
        )

    # Check for minimal required fields within sections
    minimal_requirements = {
        "data": ["path", "target_column"],
        "model": ["type"],
        "explainers": ["strategy"],
        "metrics": ["performance"],
        "report": ["template"],
    }

    incomplete_sections = []
    for section, required_fields in minimal_requirements.items():
        if section in config_dict:
            section_config = config_dict[section]
            if isinstance(section_config, dict):
                missing_fields = [field for field in required_fields if field not in section_config]
                if missing_fields:
                    incomplete_sections.append(f"{section}: missing {missing_fields}")

    if incomplete_sections:
        logger.warning(
            f"Configuration sections incomplete: {'; '.join(incomplete_sections)}. Consider adding these fields for full functionality.",
        )


def suggest_config_improvements(config_dict: dict[str, Any]) -> None:
    """Suggest configuration improvements for better audit quality.

    Args:
        config_dict: Full configuration dictionary

    """
    suggestions = []

    # Check for reproducibility settings
    if "reproducibility" not in config_dict or not config_dict.get("reproducibility", {}).get("random_seed"):
        suggestions.append("Add 'reproducibility.random_seed' for deterministic audits")

    # Check for protected attributes
    data_config = config_dict.get("data", {})
    if not data_config.get("protected_attributes"):
        suggestions.append("Add 'data.protected_attributes' for fairness analysis")

    # Check for multiple explainers
    explainers_config = config_dict.get("explainers", {})
    priority = explainers_config.get("priority", [])
    if isinstance(priority, list) and len(priority) < 2:
        suggestions.append("Add multiple explainers in 'explainers.priority' for robustness")

    # Check for fairness metrics
    metrics_config = config_dict.get("metrics", {})
    if not metrics_config.get("fairness"):
        suggestions.append("Add 'metrics.fairness' for bias detection")

    # Check for audit profile
    if not config_dict.get("audit_profile"):
        suggestions.append("Add 'audit_profile' to specify compliance framework")

    if suggestions:
        suggestion_msg = "Configuration suggestions for better audits:\n"
        for suggestion in suggestions:
            suggestion_msg += f"  • {suggestion}\n"
        suggestion_msg += "These are optional but recommended for production use."
        logger.info(suggestion_msg.rstrip())


def check_config_security(config_dict: dict[str, Any]) -> None:
    """Check configuration for potential security issues.

    Args:
        config_dict: Full configuration dictionary

    """
    security_warnings = []

    def check_nested_values(obj: Any, path: str = "") -> None:
        """Recursively check for security issues."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                check_nested_values(value, current_path)
        elif isinstance(obj, str):
            # Check for potential secrets
            lower_obj = obj.lower()
            if any(secret_word in lower_obj for secret_word in ["password", "secret", "key", "token"]):
                if len(obj) > 10:  # Likely an actual secret, not just the word
                    security_warnings.append(f"Potential secret in '{path}': {obj[:10]}...")

            # Check for absolute paths that might be user-specific
            if obj.startswith("/Users/") or obj.startswith("C:\\Users\\"):
                security_warnings.append(f"User-specific path in '{path}': {obj}")

    check_nested_values(config_dict)

    if security_warnings:
        warning_msg = "Security/portability warnings in configuration:\n"
        for warning in security_warnings:
            warning_msg += f"  • {warning}\n"
        warning_msg += "Consider using environment variables or relative paths."
        logger.warning(warning_msg.rstrip())
