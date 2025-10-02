from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import metadata
from importlib.util import find_spec
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PluginSpec:
    """Specification for a plugin with dependency information."""

    name: str
    entry_point: str  # "module.path:function_function"
    import_check: str | None  # Package name to check (e.g., "xgboost")
    extra_hint: str | None  # Optional dependency group name (e.g., "xgboost")
    description: str | None = None


class PluginRegistry:
    """Simple entry-point aware registry with lazy loading."""

    def __init__(self, group: str):
        self.group = group
        self._entry_points: dict[str, str] = {}  # name -> "module:attr"
        self._objects: dict[str, Any] = {}  # name -> loaded object (class/callable)
        self._specs: dict[str, PluginSpec] = {}  # name -> PluginSpec
        self._discovered = False

    # ----- public API expected by tests -----

    def register(self, name_or_spec, obj=None, priority: int = 0, **metadata) -> None:
        """Register a plugin.

        Args:
            name_or_spec: Either a plugin name string or PluginSpec object
            obj: The object to register (if name string provided)
            priority: Priority for selection (if name string provided)
            **metadata: Additional metadata

        """
        if isinstance(name_or_spec, PluginSpec):
            # Register a PluginSpec object
            self._specs[name_or_spec.name] = name_or_spec
            logger.debug(f"Registered plugin '{name_or_spec.name}' via PluginSpec")
        elif obj is not None:
            # Register object directly
            name = name_or_spec
            self._objects[name] = obj
            logger.debug(f"Registered plugin '{name}' with object")
        else:
            # Legacy method for backward compatibility
            name = name_or_spec
            logger.debug(f"Legacy register call for plugin '{name}' with priority {priority}")

    def get(self, name: str) -> Any:
        """Return the registered object (class/callable). Lazy-load from entry points if needed."""
        self._ensure_discovered()
        if name in self._objects:
            return self._objects[name]
        if name in self._entry_points:
            module, attr = self._entry_points[name].split(":")
            obj = getattr(__import__(module, fromlist=[attr]), attr)
            self._objects[name] = obj
            return obj
        if name in self._specs:
            # For backward compatibility, also check specs
            return self._specs[name]
        raise KeyError(f"Unknown plugin '{name}' for group '{self.group}'")

    # ----- convenience, not required by the failing test -----

    def has(self, name: str) -> bool:
        self._ensure_discovered()
        return name in self._objects or name in self._entry_points

    def names(self) -> list[str]:
        self._ensure_discovered()
        return sorted(set(self._objects) | set(self._entry_points))

    def available_plugins(self) -> dict[str, bool]:
        """Get availability status of all plugins.

        Returns:
            Dictionary mapping plugin names to availability status

        """
        availability = {}
        all_names = set(self.names()) | set(self._specs.keys())

        for name in all_names:
            # Check if plugin has dependency requirements
            if name in self._specs:
                spec = self._specs[name]
                if spec.import_check and find_spec(spec.import_check) is None:
                    availability[name] = False
                else:
                    availability[name] = True
            else:
                # For directly registered objects, assume available
                availability[name] = True

        return availability

    def get_install_hint(self, name: str) -> str | None:
        """Get installation hint for a plugin.

        Args:
            name: Plugin name

        Returns:
            Installation hint string or None if no hint available

        """
        # Simple implementation - could be enhanced
        if name in ["xgboost", "lightgbm"]:
            return f"pip install 'glassalpha[{name}]'"
        return None

    def get_all(self) -> dict[str, Any]:
        """Get all registered plugin specs.

        Returns:
            Dictionary mapping plugin names to specs

        """
        return {name: {"name": name} for name in self.names()}

    def load(self, name: str, *args, **kwargs) -> Any:
        """Load a plugin instance.

        Args:
            name: Plugin name
            **kwargs: Arguments to pass to plugin constructor

        Returns:
            Plugin instance

        Raises:
            KeyError: If plugin not found
            ImportError: If optional dependency missing with installation hint

        """
        # Check if plugin is available before loading
        available = self.available_plugins()
        if not available.get(name, False):
            hint = self.get_install_hint(name)
            if hint:
                raise ImportError(
                    f"Missing optional dependency for plugin '{name}'. {hint}",
                )
            raise ImportError(f"Plugin '{name}' is not available.")

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


# Global registries for different component types
ModelRegistry = PluginRegistry("glassalpha.models")
# ExplainerRegistry is now defined in the explain module to avoid circular imports
MetricRegistry = PluginRegistry("glassalpha.metrics")
ProfileRegistry = PluginRegistry("glassalpha.profiles")
DataRegistry = PluginRegistry("glassalpha.data_handlers")

# Discover plugins on import
ModelRegistry.discover()
# ExplainerRegistry.discover() is handled in the explain module
MetricRegistry.discover()
ProfileRegistry.discover()
DataRegistry.discover()


def _get_explainer_registry():
    """Lazy getter for ExplainerRegistry to avoid circular imports."""
    try:
        from ..explain.registry import ExplainerRegistry
        return ExplainerRegistry
    except ImportError:
        # Fallback to basic registry if explain module not available
        return PluginRegistry("glassalpha.explainers")


# For backward compatibility, provide ExplainerRegistry through lazy loading
class ExplainerRegistryProxy:
    """Proxy for ExplainerRegistry that loads it lazily."""

    @classmethod
    def __getattr__(cls, name):
        # Import the real ExplainerRegistry when first accessed
        real_registry = _get_explainer_registry()
        # Replace this proxy with the real registry in the module
        import sys
        current_module = sys.modules[__name__]
        setattr(current_module, 'ExplainerRegistry', real_registry)
        return getattr(real_registry, name)


# Initially set ExplainerRegistry to the proxy
ExplainerRegistry = ExplainerRegistryProxy()


def list_components(component_type: str = None, include_enterprise: bool = False) -> dict[str, list[str]]:
    """List all registered components.

    Args:
        component_type: Specific type to list, or None for all
        include_enterprise: Whether to include enterprise components

    Returns:
        Dictionary mapping component types to lists of names

    """
    registries = {
        "models": ModelRegistry,
        "explainers": ExplainerRegistry,
        "metrics": MetricRegistry,
        "data_handlers": DataRegistry,
        "profiles": ProfileRegistry,
    }

    if component_type:
        registry = registries.get(component_type)
        if registry:
            return {component_type: registry.names()}
        return {}

    result = {}
    for name, registry in registries.items():
        result[name] = registry.names()

    return result


def select_explainer(model_type: str, config: dict[str, Any]) -> str | None:
    """Select appropriate explainer based on model and config.

    This ensures deterministic selection based on configuration priorities.

    Args:
        model_type: Type of model being explained
        config: Configuration with explainer priorities

    Returns:
        Selected explainer name or None

    """
    # Get priority list from config
    explainer_config = config.get("explainers", {})
    priority_list = explainer_config.get("priority", [])

    if not priority_list:
        # Fall back to all registered explainers
        priority_list = ExplainerRegistry.names()

    # Return first in priority list
    for name in priority_list:
        if ExplainerRegistry.has(name):
            return name

    return None
