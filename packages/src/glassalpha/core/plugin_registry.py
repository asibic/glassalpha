"""Plugin registry system with lazy loading and dependency management.

This module provides a robust plugin system that handles optional dependencies
gracefully with clear error messages and installation hints.
"""

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
    """Registry for plugins with lazy loading and dependency checking."""

    def __init__(self, entry_point_group: str):
        """Initialize plugin registry.

        Args:
            entry_point_group: Entry point group name (e.g., "glassalpha.models")

        """
        self.entry_point_group = entry_point_group
        self._specs: dict[str, PluginSpec] = {}
        self._instances: dict[str, Any] = {}
        self._unavailable: dict[str, str] = {}

    def register(self, spec_or_name, priority: int = 0, **metadata) -> None:
        """Register a plugin class or spec.

        Args:
            spec_or_name: Either a PluginSpec object or a plugin name string
            priority: Priority for selection (if name string provided)
            **metadata: Additional metadata (if name string provided)

        """
        if isinstance(spec_or_name, PluginSpec):
            # Register a PluginSpec object
            self._specs[spec_or_name.name] = spec_or_name
            logger.debug(f"Registered plugin '{spec_or_name.name}' via PluginSpec")
        else:
            # Legacy method for backward compatibility
            name = spec_or_name
            logger.debug(f"Legacy register call for plugin '{name}' with priority {priority}")

    def __call__(self, name: str, priority: int = 0, **metadata):
        """Decorator to register a plugin class.

        Args:
            name: Plugin name
            priority: Priority for selection
            **metadata: Additional metadata

        Returns:
            Decorator function

        """

        def decorator(cls):
            # Store the class for later loading via entry points
            # For backward compatibility, we register it directly
            spec = PluginSpec(
                name=name,
                entry_point=f"{cls.__module__}:{cls.__name__}",
                import_check=None,  # Will be determined by the class itself
                extra_hint=None,
                description=getattr(cls, "__doc__", None),
            )
            self._specs[name] = spec
            logger.debug(f"Registered plugin '{name}' via decorator")
            return cls

        return decorator

    def discover(self) -> None:
        """Discover plugins from entry points."""
        try:
            for entry_point in metadata.entry_points(group=self.entry_point_group):
                try:
                    # Store the entry point for lazy loading
                    spec = PluginSpec(
                        name=entry_point.name,
                        entry_point=entry_point.value,
                        import_check=self._guess_import_check(entry_point.name),
                        extra_hint=self._guess_extra_hint(entry_point.name),
                        description=getattr(entry_point, "description", None),
                    )
                    self.register(spec)
                except Exception as e:
                    self._unavailable[entry_point.name] = f"Invalid entry point: {e}"
                    logger.debug(f"Failed to register plugin '{entry_point.name}': {e}")
        except Exception as e:
            logger.debug(f"Could not discover plugins: {e}")

    def load(self, name: str, **kwargs) -> Any:
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
        if name not in self._specs:
            raise KeyError(f"Unknown plugin '{name}'")

        spec = self._specs[name]

        # Check if already loaded
        if name in self._instances:
            return self._instances[name]

        # Check for missing optional dependency
        if spec.import_check and find_spec(spec.import_check) is None:
            hint = f"Try: pip install 'glassalpha[{spec.extra_hint}]'" if spec.extra_hint else ""
            raise ImportError(
                f"Missing optional dependency '{spec.import_check}' for plugin '{name}'. {hint}".strip(),
            )

        # Load the plugin
        try:
            module_path, func_name = spec.entry_point.split(":")
            module = __import__(module_path, fromlist=[func_name])
            constructor = getattr(module, func_name)
            instance = constructor(**kwargs)
            self._instances[name] = instance
            logger.debug(f"Loaded plugin '{name}'")
            return instance
        except Exception as e:
            error_msg = f"Failed to load plugin '{name}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def available_plugins(self) -> dict[str, bool]:
        """Get availability status of all plugins.

        Returns:
            Dictionary mapping plugin names to availability status

        """
        availability = {}
        for name, spec in self._specs.items():
            if name in self._unavailable:
                availability[name] = False
            else:
                available = not spec.import_check or find_spec(spec.import_check) is not None
                availability[name] = available
        return availability

    def get_install_hint(self, name: str) -> str | None:
        """Get installation hint for a plugin.

        Args:
            name: Plugin name

        Returns:
            Installation hint string or None if no hint available

        """
        if name not in self._specs:
            return None

        spec = self._specs[name]
        if spec.extra_hint:
            return f"pip install 'glassalpha[{spec.extra_hint}]'"
        return None

    def get_all(self) -> dict[str, Any]:
        """Get all registered plugin specs.

        Returns:
            Dictionary mapping plugin names to specs

        """
        return self._specs.copy()

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a plugin.

        Args:
            name: Plugin name

        Returns:
            Metadata dictionary

        """
        return {"priority": 0}  # Default priority for now

    def _guess_import_check(self, name: str) -> str | None:
        """Guess import check package name from plugin name."""
        # Map common plugin names to their package names
        mapping = {
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "logistic_regression": None,  # Always available
        }
        return mapping.get(name)

    def _guess_extra_hint(self, name: str) -> str | None:
        """Guess extra dependency group from plugin name."""
        # Map plugin names to optional dependency groups
        mapping = {
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "logistic_regression": None,  # Always available
        }
        return mapping.get(name)


# Global registries for different component types
ModelRegistry = PluginRegistry("glassalpha.models")
ExplainerRegistry = PluginRegistry("glassalpha.explainers")
MetricRegistry = PluginRegistry("glassalpha.metrics")
ProfileRegistry = PluginRegistry("glassalpha.profiles")
DataRegistry = PluginRegistry("glassalpha.data_handlers")

# Discover plugins on import
ModelRegistry.discover()
ExplainerRegistry.discover()
MetricRegistry.discover()
ProfileRegistry.discover()
DataRegistry.discover()


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
            return {component_type: list(registry.available_plugins().keys())}
        return {}

    result = {}
    for name, registry in registries.items():
        result[name] = list(registry.available_plugins().keys())

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
        all_explainers = ExplainerRegistry.get_all()
        priority_list = list(all_explainers.keys())

    # Filter function to check model compatibility
    def is_compatible(explainer_name):
        try:
            # For now, assume all explainers are compatible
            # In the future, we could check explainer capabilities
            return True
        except Exception:
            return False

    # Try to find a compatible explainer
    for name in priority_list:
        if name in ExplainerRegistry._specs and is_compatible(name):
            return name

    return None
