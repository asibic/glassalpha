"""Security utilities for GlassAlpha.

This module provides comprehensive security controls for production deployments,
including path validation, safe YAML loading, and log sanitization.

The security model follows "default secure, explicit opt-outs" principles:
- Model paths are restricted to allow-listed directories by default
- YAML loading is capped and uses safe_load only
- Logs are sanitized to prevent information leakage
- Remote resources require explicit configuration and checksums
"""

from .logs import sanitize_log_message, setup_secure_logging
from .paths import validate_local_model, validate_model_uri
from .yaml_loader import safe_load_yaml

__all__ = [
    "safe_load_yaml",
    "sanitize_log_message",
    "setup_secure_logging",
    "validate_local_model",
    "validate_model_uri",
]
