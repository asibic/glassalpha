"""Explainer modules for GlassAlpha."""

# Import coefficients explainer (no dependencies)
from . import coefficients
from .base import ExplainerBase
from .noop import NoOpExplainer
from .permutation import PermutationExplainer
from .registry import ExplainerRegistry

# Register explainers manually since they don't use decorators
ExplainerRegistry.register("coefficients", coefficients.CoefficientsExplainer)
ExplainerRegistry.register("permutation", PermutationExplainer)

# Add aliases for backward compatibility and common usage patterns
ExplainerRegistry.alias("coef", "coefficients")
ExplainerRegistry.alias("coeff", "coefficients")
ExplainerRegistry.alias("permutation_importance", "permutation")
ExplainerRegistry.alias("perm", "permutation")

# Conditionally import SHAP explainers if SHAP is available
# This ensures they are registered without eager imports
try:
    import shap  # noqa: F401

    # Import the modules to trigger registration via decorators
    from .shap import kernel, tree  # noqa: F401
except ImportError:
    # SHAP not available - explainers will be registered via entry points when needed
    pass

__all__ = ["ExplainerBase", "ExplainerRegistry", "NoOpExplainer", "PermutationExplainer"]
