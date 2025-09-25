"""Explainer modules for GlassAlpha."""

from .base import ExplainerBase
from .registry import ExplainerRegistry

# Import specific explainers
try:
    from .shap.kernel import KernelSHAPExplainer
    from .shap.tree import TreeSHAPExplainer
except ImportError:
    # Fallback when SHAP unavailable
    TreeSHAPExplainer = None  # type: ignore
    KernelSHAPExplainer = None  # type: ignore

__all__ = ["ExplainerBase", "ExplainerRegistry", "KernelSHAPExplainer", "TreeSHAPExplainer"]
