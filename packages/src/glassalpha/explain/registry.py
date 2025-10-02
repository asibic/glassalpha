"""Registry for explainer classes and compatibility detection."""

from __future__ import annotations

import logging
from typing import Any

from glassalpha.constants import NO_EXPLAINER_MSG
from glassalpha.core.registry import PluginRegistry, PluginSpec

logger = logging.getLogger(__name__)


# Priority order for explainer selection (TreeSHAP preferred for tree models)
PRIORITY = ("treeshap", "kernelshap")

# Model type to compatible explainers mapping
TYPE_TO_EXPLAINERS = {
    "xgboost": ("treeshap", "kernelshap"),
    "lightgbm": ("treeshap", "kernelshap"),
    "random_forest": ("treeshap", "kernelshap"),
    "decision_tree": ("treeshap", "kernelshap"),
    "logistic_regression": ("kernelshap",),
    "linear_model": ("kernelshap",),
}


class ExplainerPluginRegistry(PluginRegistry):
    """Enhanced explainer registry with decorator support and dependency checking."""

    def register(self, name_or_obj, obj=None, **meta):
        """Register an explainer plugin.

        Supports multiple registration patterns:
        1. Direct call: register("name", Class, import_check="shap")
        2. Decorator without args: @ExplainerRegistry.register
        3. Decorator with args: @ExplainerRegistry.register("name", import_check="shap")

        Args:
            name_or_obj: Either plugin name (str) or object to register (class/callable)
            obj: The object to register (if name string provided)
            **meta: Metadata including import_check, extra_hint, etc.

        Returns:
            For decorator usage: Returns the decorator function or registered object
            For direct calls: Returns the registered object

        """
        # Direct call: register("name", Class, import_check=..., extra_hint=...)
        if obj is not None and callable(obj):
            # Create PluginSpec for optional dependency support
            import_check = meta.get("import_check")
            extra_hint = meta.get("extra_hint", import_check)

            spec = PluginSpec(
                name=name_or_obj,
                entry_point="",  # Not using entry points for direct registration
                import_check=import_check,
                extra_hint=extra_hint,
                description=meta.get("description"),
            )

            # Register both the spec and the object
            self._specs[name_or_obj] = spec
            super().register(name_or_obj, obj, **meta)
            return obj

        # Decorator without args: @ExplainerRegistry.register
        if callable(name_or_obj) and obj is None and not meta:
            cls = name_or_obj
            name = cls.__name__.lower()
            super().register(name, cls)
            return cls

        # Decorator with args: @ExplainerRegistry.register("name", import_check="shap", ...)
        # This is the case when called as @ExplainerRegistry.register("name", import_check="shap")
        if isinstance(name_or_obj, str) and obj is None:
            name = name_or_obj
            import_check = meta.get("import_check")
            extra_hint = meta.get("extra_hint", import_check)

            # Capture self for use in decorator
            registry = self

            def deco(cls):
                # Create PluginSpec for optional dependency support
                spec = PluginSpec(
                    name=name,
                    entry_point="",
                    import_check=import_check,
                    extra_hint=extra_hint,
                    description=meta.get("description"),
                )
                # Register both the spec and the object
                registry._specs[name] = spec
                PluginRegistry.register(registry, name, cls)
                return cls

            return deco

        # Fallback - should not happen in normal usage
        return None

    def get_install_hint(self, name: str) -> str | None:
        """Get installation hint for an explainer plugin.

        Args:
            name: Plugin name

        Returns:
            Installation hint string or None

        """
        # For SHAP-based explainers
        if name in ["kernelshap", "treeshap"]:
            return "pip install 'glassalpha[shap]'"
        return None


# Create the global explainer registry instance
_explainer_plugin_registry = ExplainerPluginRegistry(group="glassalpha.explainers")
_explainer_plugin_registry.discover()  # Safe discovery without imports


# Backward compatibility class that delegates to the plugin registry
class ExplainerRegistryCompat:
    """Backward-compatible explainer registry that delegates to plugin registry.

    Provides the same API as the old ExplainerRegistry for existing code.
    """

    @classmethod
    def register(cls, key_or_obj, obj=None, **meta) -> type:
        """Register an explainer class with a key."""
        # Simply delegate to the plugin registry
        return _explainer_plugin_registry.register(key_or_obj, obj, **meta)

    @classmethod
    def get(cls, key: str) -> type | None:
        """Get explainer class by key."""
        try:
            return _explainer_plugin_registry.get(key)
        except KeyError:
            return None

    @classmethod
    def get_all(cls) -> dict[str, type]:
        """Get all registered explainers."""
        # Return a dict mapping names to classes for backward compatibility
        result = {}
        for name in cls.names():
            try:
                result[name] = cls.get(name)
            except ImportError:
                # Skip unavailable plugins
                continue
        return result

    @classmethod
    def names(cls) -> list[str]:
        """Get list of registered explainer names."""
        return _explainer_plugin_registry.names()

    @classmethod
    def get_install_hint(cls, name: str) -> str | None:
        """Get installation hint for an explainer plugin."""
        return _explainer_plugin_registry.get_install_hint(name)

    @classmethod
    def available_plugins(cls) -> dict[str, bool]:
        """Get availability status of all plugins."""
        return _explainer_plugin_registry.available_plugins()

    @classmethod
    def find_compatible(cls, model_type_or_obj: Any) -> type | None:  # noqa: ANN401
        """Find a compatible explainer for the given model type or object.

        Args:
            model_type_or_obj: String model type or model object with get_model_info()

        Returns:
            Explainer class if compatible, None if no match found

        """
        # Extract model type string
        if isinstance(model_type_or_obj, str):
            model_type = model_type_or_obj.lower()
        # Try to get model type from wrapper object - prefer explicit methods
        elif hasattr(model_type_or_obj, "get_model_type"):
            try:
                model_type = model_type_or_obj.get_model_type().lower()
            except Exception:  # noqa: BLE001
                model_type = ""
        elif hasattr(model_type_or_obj, "model_type"):
            try:
                model_type = model_type_or_obj.model_type.lower()
            except Exception:  # noqa: BLE001
                model_type = ""
        # Fallback to get_model_info() for compatibility with existing objects
        elif hasattr(model_type_or_obj, "get_model_info"):
            try:
                model_info = model_type_or_obj.get_model_info()
                model_type = (model_info or {}).get("type", "").lower()
            except Exception:  # noqa: BLE001
                model_type = ""
        else:
            # Final fallback to class name
            model_type = type(model_type_or_obj).__name__.lower()

        if not model_type:
            logger.debug("Could not determine model type")
            return None

        # Check for compatible explainers in priority order
        compatible_explainers = TYPE_TO_EXPLAINERS.get(model_type)
        if not compatible_explainers:
            logger.debug(f"No explainer mapping for model type: {model_type}")
            raise RuntimeError(NO_EXPLAINER_MSG)

        # Find first available explainer in priority order
        for explainer_key in PRIORITY:
            if explainer_key in compatible_explainers:
                try:
                    explainer_class = _explainer_plugin_registry.get(explainer_key)
                    logger.debug(f"Selected {explainer_key} for model type: {model_type}")
                    return explainer_class
                except (KeyError, ImportError):
                    # Skip unavailable explainers
                    continue

        logger.debug(f"No compatible explainer found for model type: {model_type}")
        raise RuntimeError(NO_EXPLAINER_MSG)

    @classmethod
    def select_for_model(cls, model_type_or_obj: Any) -> type:  # noqa: ANN401
        """Select explainer for model, raising error if none compatible.

        Args:
            model_type_or_obj: String model type or model object

        Returns:
            Explainer class

        Raises:
            RuntimeError: If no compatible explainer found with exact contract message

        """
        # find_compatible now raises RuntimeError if no compatible explainer found
        return cls.find_compatible(model_type_or_obj)


# For backward compatibility, alias the compat class as ExplainerRegistry
ExplainerRegistry = ExplainerRegistryCompat
