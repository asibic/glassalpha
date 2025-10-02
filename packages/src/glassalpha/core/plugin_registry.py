from __future__ import annotations

from importlib import import_module, metadata
from typing import Any


class PluginRegistry:
    """Simple entry-point aware registry with lazy loading."""

    def __init__(self, group: str):
        self.group = group
        self._entry_points: dict[str, str] = {}  # name -> "module:attr"
        self._objects: dict[str, Any] = {}  # name -> loaded object (class/callable)
        self._discovered = False

    # ----- public API expected by tests -----

    def register(self, name: str, obj: Any) -> None:
        """Register an object directly."""
        self._objects[name] = obj

    def get(self, name: str) -> Any:
        """Return the registered object (class/callable). Lazy-load from entry points if needed."""
        self._ensure_discovered()
        if name in self._objects:
            return self._objects[name]
        if name in self._entry_points:
            module, attr = self._entry_points[name].split(":")
            obj = getattr(import_module(module), attr)
            self._objects[name] = obj
            return obj
        raise KeyError(f"Unknown plugin '{name}' for group '{self.group}'")

    # ----- convenience, not required by the failing test -----

    def has(self, name: str) -> bool:
        self._ensure_discovered()
        return name in self._objects or name in self._entry_points

    def names(self) -> list[str]:
        self._ensure_discovered()
        return sorted(set(self._objects) | set(self._entry_points))

    def load(self, name: str, *args, **kwargs) -> Any:
        """Instantiate if callable is a class or factory. Otherwise return as-is."""
        obj = self.get(name)
        try:
            return obj(*args, **kwargs)
        except TypeError:
            return obj

    # ----- entry point discovery -----

    def discover(self) -> None:
        """Populate entry points but do not import heavy modules yet."""
        self._entry_points.clear()
        for ep in metadata.entry_points(group=self.group):
            self._entry_points[ep.name] = ep.value
        self._discovered = True

    # ----- internals -----

    def _ensure_discovered(self) -> None:
        if not self._discovered:
            self.discover()


# One shared instance for models
ModelRegistry = PluginRegistry(group="glassalpha.models")


def list_components(component_type: str = None, include_enterprise: bool = False) -> dict[str, list[str]]:
    """List all registered components.

    Args:
        component_type: Specific type to list, or None for all
        include_enterprise: Whether to include enterprise components

    Returns:
        Dictionary mapping component types to lists of names

    """
    if component_type == "models":
        return {"models": ModelRegistry.names()}

    # For now, only models are supported
    return {"models": ModelRegistry.names()}


def select_explainer(model_type: str, config: dict[str, Any]) -> str | None:
    """Select appropriate explainer based on model and config.

    This ensures deterministic selection based on configuration priorities.

    Args:
        model_type: Type of model being explained
        config: Configuration with explainer priorities

    Returns:
        Selected explainer name or None

    """
    # Placeholder implementation - will be expanded later
    explainer_config = config.get("explainers", {})
    priority_list = explainer_config.get("priority", [])

    if priority_list:
        return priority_list[0]

    return None
