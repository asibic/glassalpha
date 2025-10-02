from __future__ import annotations

from glassalpha.core.registry import ModelRegistry

# Ensure entry points are visible to the registry
ModelRegistry.discover()

# Register built-ins that should always exist for tests and quickstart flows
from .passthrough import PassThroughModel  # noqa: E402

ModelRegistry.register("passthrough", PassThroughModel)

__all__ = ["ModelRegistry", "PassThroughModel"]
