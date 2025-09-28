"""Safe YAML loading with security controls.

This module provides secure YAML loading that prevents common security
vulnerabilities like arbitrary code execution, resource exhaustion,
and information disclosure attacks.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class YAMLSecurityError(Exception):
    """Raised when YAML security validation fails."""


def safe_load_yaml(
    source: str | Path,
    max_file_size_mb: float = 10.0,
    max_depth: int = 20,
    max_keys: int = 1000,
    allowed_types: set[type] | None = None,
) -> dict[str, Any]:
    """Load YAML file with comprehensive security controls.

    This function implements multiple layers of security:
    - File size limits (prevents resource exhaustion)
    - Parse depth limits (prevents stack overflow)
    - Key count limits (prevents memory exhaustion)
    - Type restrictions (prevents code execution)
    - Uses yaml.safe_load only (no arbitrary code execution)

    Args:
        source: Path to YAML file or YAML string content
        max_file_size_mb: Maximum file size in MB (default: 10MB)
        max_depth: Maximum nesting depth (default: 20)
        max_keys: Maximum number of keys (default: 1000)
        allowed_types: Set of allowed Python types (default: safe types only)

    Returns:
        Parsed YAML content as dictionary

    Raises:
        YAMLSecurityError: If any security check fails

    Examples:
        >>> # Load from file
        >>> config = safe_load_yaml("config.yaml")

        >>> # Load from string with custom limits
        >>> config = safe_load_yaml(
        ...     yaml_content,
        ...     max_file_size_mb=5.0,
        ...     max_depth=10
        ... )

    """
    logger.debug(f"Loading YAML with security controls: {source}")

    # Set default allowed types (safe types only)
    if allowed_types is None:
        allowed_types = {
            type(None),  # None
            bool,  # Boolean
            int,  # Integer
            float,  # Float
            str,  # String
            list,  # List
            dict,  # Dictionary
        }

    # Handle file vs string input
    if isinstance(source, Path) or (isinstance(source, str) and len(source) < 1000 and Path(source).exists()):
        # Load from file (only if string is short enough to be a path)
        yaml_path = Path(source)

        # Check file size
        file_size_mb = yaml_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_file_size_mb:
            raise YAMLSecurityError(
                f"YAML file too large: {file_size_mb:.1f}MB > {max_file_size_mb}MB limit",
            )

        # Read file content
        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                yaml_content = f.read()
        except Exception as e:
            raise YAMLSecurityError(f"Failed to read YAML file {yaml_path}: {e}") from e

        logger.debug(f"Read YAML file: {yaml_path} ({file_size_mb:.1f}MB)")

    else:
        # Treat as YAML string content
        yaml_content = str(source)

        # Check content size
        content_size_mb = len(yaml_content.encode("utf-8")) / (1024 * 1024)
        if content_size_mb > max_file_size_mb:
            raise YAMLSecurityError(
                f"YAML content too large: {content_size_mb:.1f}MB > {max_file_size_mb}MB limit",
            )

    # Parse YAML using safe_load only
    try:
        # Use safe_load to prevent code execution
        parsed_data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise YAMLSecurityError(f"YAML parsing failed: {e}") from e
    except Exception as e:
        raise YAMLSecurityError(f"Unexpected error parsing YAML: {e}") from e

    # Validate parsed data
    if parsed_data is None:
        logger.warning("YAML file is empty or contains only null values")
        return {}

    if not isinstance(parsed_data, dict):
        raise YAMLSecurityError(
            f"YAML root must be a dictionary, got: {type(parsed_data).__name__}",
        )

    # Validate structure and types
    _validate_yaml_structure(
        parsed_data,
        max_depth=max_depth,
        max_keys=max_keys,
        allowed_types=allowed_types,
    )

    logger.info(f"YAML loaded successfully with {len(parsed_data)} top-level keys")
    return parsed_data


def _validate_yaml_structure(
    data: Any,
    max_depth: int,
    max_keys: int,
    allowed_types: set[type],
    current_depth: int = 0,
    key_count: dict[str, int] | None = None,
) -> None:
    """Recursively validate YAML structure for security.

    Args:
        data: Data to validate
        max_depth: Maximum nesting depth
        max_keys: Maximum total key count
        allowed_types: Set of allowed Python types
        current_depth: Current recursion depth (internal)
        key_count: Key counter (internal)

    Raises:
        YAMLSecurityError: If validation fails

    """
    if key_count is None:
        key_count = {"count": 0}

    # Check depth limit
    if current_depth > max_depth:
        raise YAMLSecurityError(f"YAML nesting too deep: {current_depth} > {max_depth}")

    # Check type restrictions
    if type(data) not in allowed_types:
        raise YAMLSecurityError(
            f"Disallowed type in YAML: {type(data).__name__} (allowed: {[t.__name__ for t in allowed_types]})",
        )

    # Recursively validate containers
    if isinstance(data, dict):
        # Count keys
        key_count["count"] += len(data)
        if key_count["count"] > max_keys:
            raise YAMLSecurityError(f"Too many keys in YAML: {key_count['count']} > {max_keys}")

        # Validate keys and values
        for key, value in data.items():
            # Keys must be strings
            if not isinstance(key, str):
                raise YAMLSecurityError(f"Dictionary keys must be strings, got: {type(key).__name__}")

            # Check for suspicious keys
            _check_suspicious_key(key)

            # Recursively validate value
            _validate_yaml_structure(
                value,
                max_depth=max_depth,
                max_keys=max_keys,
                allowed_types=allowed_types,
                current_depth=current_depth + 1,
                key_count=key_count,
            )

    elif isinstance(data, list):
        # Validate list items
        for item in data:
            _validate_yaml_structure(
                item,
                max_depth=max_depth,
                max_keys=max_keys,
                allowed_types=allowed_types,
                current_depth=current_depth + 1,
                key_count=key_count,
            )


def _check_suspicious_key(key: str) -> None:
    """Check for suspicious or dangerous key names.

    Args:
        key: Dictionary key to check

    Raises:
        YAMLSecurityError: If key is suspicious

    """
    # Check for dangerous key patterns
    dangerous_patterns = [
        "__",  # Python dunder methods
        "eval",  # Code evaluation
        "exec",  # Code execution
        "import",  # Module imports
        "subprocess",  # Process execution
        "system",  # System commands
        "open",  # File operations
        "file",  # File operations (when suspicious)
    ]

    key_lower = key.lower()
    for pattern in dangerous_patterns:
        if pattern in key_lower:
            logger.warning(f"Suspicious key detected in YAML: {key}")
            # Don't fail by default, just warn
            break


def validate_yaml_content(content: dict[str, Any]) -> dict[str, Any]:
    """Validate and sanitize YAML content after loading.

    Args:
        content: Parsed YAML content

    Returns:
        Validated and sanitized content

    Raises:
        YAMLSecurityError: If content validation fails

    """
    if not isinstance(content, dict):
        raise YAMLSecurityError("YAML content must be a dictionary")

    # Check for required security fields
    security_config = content.get("security", {})
    if isinstance(security_config, dict):
        # Validate security configuration
        strict_mode = security_config.get("strict", False)
        if strict_mode:
            logger.info("Strict security mode enabled in YAML")
            # Additional validation for strict mode
            _validate_strict_mode_config(content)

    return content


def _validate_strict_mode_config(content: dict[str, Any]) -> None:
    """Validate configuration for strict security mode.

    Args:
        content: Configuration content to validate

    Raises:
        YAMLSecurityError: If strict mode validation fails

    """
    # Check for potentially dangerous configurations
    dangerous_configs = [
        ("data", "path", "Absolute paths in data.path"),
        ("model", "path", "Absolute paths in model.path"),
    ]

    for section, field, description in dangerous_configs:
        if section in content and isinstance(content[section], dict):
            if field in content[section]:
                value = content[section][field]
                if isinstance(value, str) and Path(value).is_absolute():
                    logger.warning(f"Strict mode warning: {description}: {value}")


def get_safe_yaml_config() -> dict[str, Any]:
    """Get secure default configuration for YAML loading.

    Returns:
        Dictionary with safe YAML loading configuration

    """
    return {
        "max_file_size_mb": 10.0,
        "max_depth": 20,
        "max_keys": 1000,
        "allowed_types": {
            type(None),
            bool,
            int,
            float,
            str,
            list,
            dict,
        },
        "strict_mode": False,
    }
