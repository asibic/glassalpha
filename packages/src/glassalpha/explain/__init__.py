"""Explainer modules for GlassAlpha."""

from .base import ExplainerBase
from .noop import NoOpExplainer
from .permutation import PermutationExplainer
from .registry import ExplainerRegistry

# Discover explainers from entry points
ExplainerRegistry.discover()

# Add aliases for backward compatibility and common usage patterns
ExplainerRegistry.alias("coef", "coefficients")
ExplainerRegistry.alias("coeff", "coefficients")
ExplainerRegistry.alias("permutation_importance", "permutation")
ExplainerRegistry.alias("perm", "permutation")

__all__ = ["ExplainerBase", "ExplainerRegistry", "NoOpExplainer", "PermutationExplainer"]
