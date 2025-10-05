"""Model wrappers for GlassAlpha."""

from __future__ import annotations

from glassalpha.core.registry import ModelRegistry

# Discover models from entry points
ModelRegistry.discover()

__all__ = ["ModelRegistry"]
