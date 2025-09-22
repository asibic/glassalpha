"""Core architecture components for Glass Alpha.

This module provides the foundational interfaces, registries, and
feature management system that enable the plugin architecture.
"""

from .interfaces import (
    ModelInterface,
    ExplainerInterface,
    MetricInterface,
    DataInterface,
    AuditProfileInterface,
)

from .registry import (
    ModelRegistry,
    ExplainerRegistry,
    MetricRegistry,
    DataRegistry,
    ProfileRegistry,
    select_explainer,
    list_components,
)

from .features import (
    is_enterprise,
    check_feature,
    FeatureNotAvailable,
)

# Import NoOp components to auto-register them
from .noop_components import (
    PassThroughModel,
    NoOpExplainer,
    NoOpMetric,
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
