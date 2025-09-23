"""Component registry system for plugin architecture.

This module provides the registration and discovery mechanism for all
Glass Alpha components, ensuring deterministic selection and extensibility.
"""

import logging
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ComponentRegistry:
    """Base registry for managing component plugins."""

    def __init__(self, component_type: str):
        """Initialize registry for a component type.

        Args:
            component_type: Type of components (e.g., 'model', 'explainer')

        """
        self.component_type = component_type
        self._registry: dict[str, type] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(self, name: str, priority: int = 0, enterprise: bool = False, **metadata):
        """Decorator to register a component.

        Args:
            name: Unique name for the component
            priority: Priority for selection (higher = preferred)
            enterprise: Whether this is an enterprise-only feature
            **metadata: Additional metadata to store

        Returns:
            Decorator function

        """

        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                logger.warning(f"Overwriting existing {self.component_type} '{name}'")

            self._registry[name] = cls
            self._metadata[name] = {"priority": priority, "enterprise": enterprise, **metadata}

            logger.debug(f"Registered {self.component_type} '{name}'")
            return cls

        return decorator

    def get(self, name: str) -> type | None:
        """Get a component by name.

        Args:
            name: Component name

        Returns:
            Component class or None if not found

        """
        return self._registry.get(name)

    def get_all(self, include_enterprise: bool = False) -> dict[str, type]:
        """Get all registered components.

        Args:
            include_enterprise: Whether to include enterprise components

        Returns:
            Dictionary of name to component class

        """
        if include_enterprise:
            return self._registry.copy()

        return {
            name: cls
            for name, cls in self._registry.items()
            if not self._metadata[name].get("enterprise", False)
        }

    def select_by_priority(self, names: list[str], filter_fn: Any | None = None) -> str | None:
        """Select component by priority order.

        Args:
            names: Ordered list of component names to try
            filter_fn: Optional function to filter compatible components

        Returns:
            Selected component name or None

        """
        from ..core.features import is_enterprise

        for name in names:
            if name not in self._registry:
                logger.debug(f"{self.component_type} '{name}' not found")
                continue

            # Check enterprise feature
            if self._metadata[name].get("enterprise", False) and not is_enterprise():
                logger.debug(f"{self.component_type} '{name}' requires enterprise license")
                continue

            # Apply filter function if provided
            if filter_fn:
                cls = self._registry[name]
                if not filter_fn(cls):
                    logger.debug(f"{self.component_type} '{name}' filtered out")
                    continue

            logger.info(f"Selected {self.component_type}: '{name}'")
            return name

        return None

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a component.

        Args:
            name: Component name

        Returns:
            Metadata dictionary

        """
        return self._metadata.get(name, {})


# Global registries for each component type
ModelRegistry = ComponentRegistry("model")
ExplainerRegistry = ComponentRegistry("explainer")
MetricRegistry = ComponentRegistry("metric")
DataRegistry = ComponentRegistry("data_handler")
ProfileRegistry = ComponentRegistry("audit_profile")


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
        # Fall back to all registered explainers by priority
        all_explainers = ExplainerRegistry.get_all()
        priority_list = sorted(
            all_explainers.keys(),
            key=lambda x: ExplainerRegistry.get_metadata(x).get("priority", 0),
            reverse=True,
        )

    # Filter function to check model compatibility
    def is_compatible(explainer_cls):
        if hasattr(explainer_cls, "capabilities"):
            supported_models = explainer_cls.capabilities.get("supported_models", [])
            if "all" in supported_models or model_type in supported_models:
                return True
        return False

    selected = ExplainerRegistry.select_by_priority(priority_list, is_compatible)

    if not selected:
        logger.warning(f"No compatible explainer found for model type '{model_type}'")

    return selected


def list_components(
    component_type: str = None, include_enterprise: bool = False
) -> dict[str, list[str]]:
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
            return {component_type: list(registry.get_all(include_enterprise).keys())}
        return {}

    result = {}
    for name, registry in registries.items():
        result[name] = list(registry.get_all(include_enterprise).keys())

    return result
