"""Registry for explainer classes and compatibility detection."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from glassalpha.constants import NO_EXPLAINER_MSG

logger = logging.getLogger(__name__)

# Import explainer classes
try:
    from .shap.kernel import KernelSHAPExplainer
    from .shap.tree import TreeSHAPExplainer

    EXPLAINERS_AVAILABLE = True
except ImportError:
    TreeSHAPExplainer = None  # type: ignore[assignment]
    KernelSHAPExplainer = None  # type: ignore[assignment]
    EXPLAINERS_AVAILABLE = False


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


class ExplainerRegistry:
    """Registry for explainer classes with priority-based selection.

    Provides deterministic explainer selection based on model types
    with configurable priority ordering.
    """

    _by_key: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, key: str, explainer_class: type) -> None:
        """Register an explainer class with a key."""
        cls._by_key[key] = explainer_class
        logger.debug("Registered explainer: {key} -> %s", explainer_class)

    @classmethod
    def get(cls, key: str) -> type | None:
        """Get explainer class by key."""
        if key not in cls._by_key:
            msg = f"Explainer '{key}' not found in registry"
            raise KeyError(msg)
        return cls._by_key[key]

    @classmethod
    def get_all(cls) -> dict[str, type]:
        """Get all registered explainers."""
        return cls._by_key.copy()

    @classmethod
    def find_compatible(cls, model_type_or_obj: Any) -> type | None:  # noqa: ANN401, C901
        """Find a compatible explainer for the given model type or object.

        Args:
            model_type_or_obj: String model type or model object with get_model_info()

        Returns:
            Explainer class if compatible, None if no match found

        """
        if not EXPLAINERS_AVAILABLE:
            logger.debug("No explainers available - SHAP library may not be installed")
            return None

        # Extract model type string
        if isinstance(model_type_or_obj, str):
            model_type = model_type_or_obj.lower()
        # Try to get model info from object
        elif hasattr(model_type_or_obj, "get_model_info"):
            try:
                model_info = model_type_or_obj.get_model_info()
                model_type = model_info.get("type", "").lower()
            except Exception:  # noqa: BLE001
                model_type = ""
        else:
            # Fallback to class name
            model_type = type(model_type_or_obj).__name__.lower()

        if not model_type:
            logger.debug("Could not determine model type")
            return None

        # Check for compatible explainers in priority order
        compatible_explainers = TYPE_TO_EXPLAINERS.get(model_type)
        if not compatible_explainers:
            logger.debug("No explainer mapping for model type: %s", model_type)
            return None

        # Find first available explainer in priority order
        for explainer_key in PRIORITY:
            if explainer_key in compatible_explainers and explainer_key in cls._by_key:
                explainer_class = cls._by_key[explainer_key]
                # Double-check compatibility
                try:
                    explainer = explainer_class()
                    if hasattr(explainer, "is_compatible") and explainer.is_compatible(model_type_or_obj):
                        logger.debug("Selected {explainer_key} for model type: %s", model_type)
                        return explainer_class
                except Exception:  # noqa: BLE001, S112
                    continue

        logger.debug("No compatible explainer found for model type: %s", model_type)
        return None

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
        explainer_class = cls.find_compatible(model_type_or_obj)
        if explainer_class is None:
            raise RuntimeError(NO_EXPLAINER_MSG)
        return explainer_class


# Initialize registry with available explainers
if EXPLAINERS_AVAILABLE:
    if TreeSHAPExplainer:
        ExplainerRegistry.register("treeshap", TreeSHAPExplainer)
    if KernelSHAPExplainer:
        ExplainerRegistry.register("kernelshap", KernelSHAPExplainer)
