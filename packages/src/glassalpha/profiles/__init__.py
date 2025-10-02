"""Audit profile system for GlassAlpha.

Profiles define valid component combinations for different audit types,
ensuring appropriate metrics, explainers, and report templates are used.
"""

from .registry import ProfileRegistry

__all__ = ["ProfileRegistry"]
