"""Core architecture components for GlassAlpha.

This module provides the foundational interfaces, registries, and
feature management system that enable the plugin architecture.
"""

from .features import (
    FeatureNotAvailable,
    check_feature,
    is_enterprise,
)
from .interfaces import (
    AuditProfileInterface,
    DataInterface,
    ExplainerInterface,
    MetricInterface,
    ModelInterface,
)

# Import NoOp components to auto-register them
from .noop_components import (
    NoOpMetric,
    PassThroughModel,
)
from .registry import (
    DataRegistry,
    ExplainerRegistry,  # This will be the lazy proxy
    MetricRegistry,
    ModelRegistry,
    ProfileRegistry,
    instantiate_explainer,
    list_components,
    select_explainer,
)

__all__ = [
    # Interfaces
    "ModelInterface",
    "ExplainerInterface",
    "MetricInterface",
    "DataInterface",
    "AuditProfileInterface",
    # Registries
    "ModelRegistry",
    "ExplainerRegistry",
    "MetricRegistry",
    "ProfileRegistry",
    "DataRegistry",
    # Registry utilities
    "instantiate_explainer",
    "list_components",
    "select_explainer",
    # Feature management
    "is_enterprise",
    "check_feature",
    "FeatureNotAvailable",
    # NoOp implementations
    "PassThroughModel",
    "NoOpMetric",
]
