"""GlassAlpha: AI Compliance Toolkit for Regulated ML

Fast imports with lazy module loading (PEP 562).
"""

import importlib
from typing import Any

__version__ = "0.2.0"

# Public API (lazy-loaded modules)
__all__ = [
    "__version__",
    "audit",
    "datasets",
    "utils",
]

# Lazy module loading (PEP 562)
_LAZY_MODULES = {
    "audit": "glassalpha.api",  # Maps to glassalpha.api (contains from_model, etc.)
    "datasets": "glassalpha.datasets",
    "utils": "glassalpha.utils",
}


def __getattr__(name: str) -> Any:
    """Lazy-load modules on first access (PEP 562)

    This enables fast imports by deferring heavy dependencies
    (sklearn, xgboost, matplotlib) until actually needed.

    Example:
        >>> import glassalpha as ga  # <200ms, no heavy deps
        >>> result = ga.audit.from_model(...)  # Now loads deps

    """
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'glassalpha' has no attribute '{name}'")


def __dir__():
    """Enable tab-completion for lazy modules"""
    return __all__
