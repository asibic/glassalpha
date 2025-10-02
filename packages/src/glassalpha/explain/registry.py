"""Registry for explainer classes and compatibility detection."""

from __future__ import annotations

import logging
from typing import Any

from glassalpha.constants import NO_EXPLAINER_MSG
from glassalpha.core.decor_registry import DecoratorFriendlyRegistry

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


class ExplainerRegistryClass(DecoratorFriendlyRegistry):
    """Enhanced explainer registry with compatibility-based selection."""

    def is_compatible(self, name: str, model_type: str, model=None) -> bool:
        """Check if explainer is compatible with model type.

        Args:
            name: Explainer name
            model_type: Model type string
            model: Optional model object for dynamic checks

        Returns:
            True if compatible, False otherwise

        """
        # Special case: noop explainer is compatible with everything when explicitly requested
        if name == "noop":
            return True

        # 1) Check explicit supports list at registration
        supports = (self._meta.get(name, {}) or {}).get("supports")
        if supports:
            if "*" in supports or model_type in supports:
                return True
            return False

        # 2) Check class-level override
        try:
            cls = self.get(name)
            if hasattr(cls, "is_compatible"):
                return bool(cls.is_compatible(model_type=model_type, model=model))
        except (KeyError, ImportError):
            pass

        # 3) Default: not compatible (prevents silent fallbacks)
        return False

    def find_compatible(self, model, config: dict | None = None) -> str:
        """Find compatible explainer for model.

        Args:
            model: Model object or model type string
            config: Optional configuration dict

        Returns:
            Explainer class

        Raises:
            RuntimeError: If no compatible explainer found

        """
        # Extract model type
        if isinstance(model, str):
            model_type = model.lower()
        else:
            info = getattr(model, "get_model_info", dict)() or {}
            model_type = info.get("type") or getattr(model, "type", None) or model.__class__.__name__.lower()

        # Get priority list from config or use default ordering
        prio = ((config or {}).get("explainers") or {}).get("priority")
        if prio:
            # Use explicit priority list from config
            names = prio
        else:
            # Use default ordering but exclude noop unless explicitly requested
            names = [name for name in getattr(self, "names_by_priority", self.names)() if name != "noop"]

        # Find first compatible explainer
        for name in names:
            # Skip if not available
            if not self.has(name):
                continue

            # Check compatibility
            try:
                if self.is_compatible(name, model_type, model=model):
                    return name
            except ImportError:
                # Skip unavailable explainers
                continue

        # No compatible explainer found
        raise RuntimeError(NO_EXPLAINER_MSG)


# Create the real explainer registry instance
ExplainerRegistry = ExplainerRegistryClass(group="glassalpha.explainers")
ExplainerRegistry.discover()  # Safe discovery without heavy imports


# Add get_install_hint method to the registry instance
def _get_install_hint(name: str) -> str | None:
    """Get installation hint for an explainer plugin."""
    if name in ["kernelshap", "treeshap"]:
        return "pip install 'glassalpha[shap]'"
    return None


ExplainerRegistry.get_install_hint = _get_install_hint

# Register the built-in noop explainer
from .noop import NoOpExplainer

ExplainerRegistry.register("noop", NoOpExplainer, priority=-100)

# Register SHAP explainers if SHAP is available
try:
    import shap  # noqa: F401

    from .shap.kernel import KernelSHAPExplainer
    from .shap.tree import TreeSHAPExplainer

    ExplainerRegistry.register(
        "treeshap",
        TreeSHAPExplainer,
        import_check="shap",
        extra_hint="shap",
        priority=100,
        supports=["xgboost", "lightgbm", "random_forest", "decision_tree", "gradient_boosting"],
    )
    ExplainerRegistry.register(
        "kernelshap",
        KernelSHAPExplainer,
        import_check="shap",
        extra_hint="shap",
        priority=50,
        supports=["xgboost", "lightgbm", "random_forest", "decision_tree", "logistic_regression", "linear_model"],
    )
except ImportError:
    # SHAP not available
    pass


# Legacy compatibility class that delegates to the instance registry
class ExplainerRegistryCompat:
    """Legacy compatibility wrapper for ExplainerRegistry.

    Provides the same API as the old ExplainerRegistry for existing code.
    Delegates all calls to the real ExplainerRegistry instance.
    """

    @classmethod
    def register(cls, key_or_obj, obj=None, **meta) -> type:
        """Register an explainer class with a key."""
        return ExplainerRegistry.register(key_or_obj, obj, **meta)

    @classmethod
    def get(cls, key: str) -> type | None:
        """Get explainer class by key."""
        try:
            return ExplainerRegistry.get(key)
        except KeyError:
            return None

    @classmethod
    def get_all(cls) -> dict[str, type]:
        """Get all registered explainers."""
        result = {}
        for name in cls.names():
            try:
                result[name] = cls.get(name)
            except ImportError:
                continue
        return result

    @classmethod
    def names(cls) -> list[str]:
        """Get list of registered explainer names."""
        return ExplainerRegistry.names()

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if explainer is available."""
        return ExplainerRegistry.has(name)

    @classmethod
    def get_install_hint(cls, name: str) -> str | None:
        """Get installation hint for an explainer plugin."""
        return ExplainerRegistry.get_install_hint(name)

    @classmethod
    def available_plugins(cls) -> dict[str, bool]:
        """Get availability status of all plugins."""
        return ExplainerRegistry.available_plugins()

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
                    explainer_class = ExplainerRegistry.get(explainer_key)
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


# Export both the real registry and the compat class for backward compatibility
__all__ = ["ExplainerRegistry", "ExplainerRegistryCompat"]
