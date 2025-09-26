"""Centralized constants and contract strings for Glass Alpha.

This module contains all exact strings used in error messages, logging,
and contract assertions to prevent drift and ensure consistency.
"""

# Logging messages
INIT_LOG_TEMPLATE = "Initialized audit pipeline with profile: {profile}"

# Error messages - exact strings used in tests
ERR_NOT_LOADED = "Model not loaded. Load a model first."
ERR_NOT_FITTED = "Model not fitted"
ERR_NO_EXPLAINER = "No compatible explainer found"
ERR_NO_MODEL = "No model loaded"

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
