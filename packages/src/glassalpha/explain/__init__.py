"""Explainer modules for GlassAlpha."""

from .base import ExplainerBase
from .noop import NoOpExplainer
from .permutation import PermutationExplainer
from .registry import ExplainerRegistry

# Discover explainers from entry points (aliases are registered in registry.py)
ExplainerRegistry.discover()

__all__ = ["ExplainerBase", "ExplainerRegistry", "NoOpExplainer", "PermutationExplainer"]
