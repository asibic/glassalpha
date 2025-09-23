"""Core architecture components for Glass Alpha.

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
    NoOpExplainer,
    NoOpMetric,
    PassThroughModel,
)
from .registry import (
    DataRegistry,
    ExplainerRegistry,
    MetricRegistry,
    ModelRegistry,
    ProfileRegistry,
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
    "DataRegistry",
    "ProfileRegistry",
    # Registry utilities
    "select_explainer",
    "list_components",
    # Feature management
    "is_enterprise",
    "check_feature",
    "FeatureNotAvailable",
    # NoOp implementations
    "PassThroughModel",
    "NoOpExplainer",
    "NoOpMetric",
]
