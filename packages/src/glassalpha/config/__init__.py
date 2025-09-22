"""Configuration system for Glass Alpha.

This module provides YAML-based configuration with Pydantic validation,
supporting audit profiles, plugin priorities, and strict mode.
"""

from .schema import (
    AuditConfig,
    ModelConfig,
    DataConfig,
    ExplainerConfig,
    MetricsConfig,
    RecourseConfig,
    ReportConfig,
    ReproducibilityConfig,
    ManifestConfig,
)

from .loader import (
    load_config,
    load_config_from_file,
    validate_config,
)

from .strict import (
    validate_strict_mode,
    StrictModeError,
)

__all__ = [
    # Config schemas
    "AuditConfig",
    "ModelConfig",
    "DataConfig", 
    "ExplainerConfig",
    "MetricsConfig",
    "RecourseConfig",
    "ReportConfig",
    "ReproducibilityConfig",
    "ManifestConfig",
    # Loaders
    "load_config",
    "load_config_from_file",
    "validate_config",
    # Strict mode
    "validate_strict_mode",
    "StrictModeError",
]
