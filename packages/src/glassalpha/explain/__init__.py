"""Explainer modules for GlassAlpha."""

from .base import ExplainerBase
from .noop import NoOpExplainer
from .registry import ExplainerRegistry

# Conditionally import SHAP explainers if SHAP is available
# This ensures they are registered without eager imports
try:
    import shap  # noqa: F401

    # Import the modules to trigger registration via decorators
    from .shap import kernel, tree  # noqa: F401
except ImportError:
    # SHAP not available - explainers will be registered via entry points when needed
    pass

__all__ = ["ExplainerBase", "ExplainerRegistry", "NoOpExplainer"]
