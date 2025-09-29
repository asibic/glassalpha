"""Centralized constants and contract strings for GlassAlpha.

This module contains all exact strings used in error messages, logging,
and contract assertions to prevent drift and ensure consistency.
"""

# Exact contract strings - primary names used in tests
NO_MODEL_MSG = "Model not loaded. Load a model first."
NO_EXPLAINER_MSG = "No compatible explainer found"
INIT_LOG_MESSAGE = "Initialized audit pipeline with profile: {profile}"

# Backward-compatible aliases (keep existing imports working)
ERR_NOT_LOADED = NO_MODEL_MSG
ERR_NOT_FITTED = "Model not fitted"
ERR_NO_EXPLAINER = NO_EXPLAINER_MSG
ERR_NO_MODEL = "No model loaded"
INIT_LOG_TEMPLATE = INIT_LOG_MESSAGE

# Status values
STATUS_CLEAN = "clean"
STATUS_DIRTY = "dirty"
STATUS_NO_GIT = "no_git"

# Binary classification constants
BINARY_CLASSES = 2
BINARY_THRESHOLD = 0.5

# Template names
STANDARD_AUDIT_TEMPLATE = "standard_audit.html"

# Package paths for resources
TEMPLATES_PACKAGE = "glassalpha.report.templates"

# Export the primary contract strings and backward-compatible aliases
__all__ = [
    "BINARY_CLASSES",
    "BINARY_THRESHOLD",
    "ERR_NOT_FITTED",
    # Backward-compatible aliases (existing imports)
    "ERR_NOT_LOADED",
    "ERR_NO_EXPLAINER",
    "ERR_NO_MODEL",
    "INIT_LOG_MESSAGE",
    "INIT_LOG_TEMPLATE",
    "NO_EXPLAINER_MSG",
    # Primary contract strings (use these in new code)
    "NO_MODEL_MSG",
    "STANDARD_AUDIT_TEMPLATE",
    # Other constants
    "STATUS_CLEAN",
    "STATUS_DIRTY",
    "STATUS_NO_GIT",
    "TEMPLATES_PACKAGE",
]
