"""Registry for explainer classes and compatibility detection."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Import explainer classes
try:
    from .shap.kernel import KernelSHAPExplainer
    from .shap.tree import TreeSHAPExplainer

    EXPLAINERS_AVAILABLE = True
except ImportError:
    TreeSHAPExplainer = None  # type: ignore
    KernelSHAPExplainer = None  # type: ignore
    EXPLAINERS_AVAILABLE = False


class ExplainerRegistry:
    """Registry for explainer classes with compatibility detection.

    This registry maps explainer keys to classes and provides logic
    for finding compatible explainers based on model characteristics.
    """

    _by_key: dict[str, type] = {}

    @classmethod
    def register(cls, key: str, explainer_class: type) -> None:
        """Register an explainer class with a key."""
        cls._by_key[key] = explainer_class
        logger.debug(f"Registered explainer: {key} -> {explainer_class}")

    @classmethod
    def get(cls, key: str):
        """Get explainer class by key."""
        if key not in cls._by_key:
            raise KeyError(f"Explainer '{key}' not found in registry")
        return cls._by_key[key]

    @classmethod
    def get_all(cls) -> dict[str, type]:
        """Get all registered explainers."""
        return cls._by_key.copy()

    @classmethod
    def find_compatible(cls, model: Any) -> type | None:
        """Find a compatible explainer for the given model.

        Uses conservative heuristics to detect tree-based models that
        work with TreeSHAP. For all other models, returns None to force
        explicit explainer selection via configuration.

        Args:
            model: Model object to check compatibility for

        Returns:
            Explainer class if compatible model detected, None otherwise

        """
        if not EXPLAINERS_AVAILABLE:
            logger.warning("No explainers available - SHAP library may not be installed")
            return None

        # Extremely conservative: only return TreeSHAP for obvious tree libs
        model_module = getattr(model, "__module__", "") or ""
        model_name = type(model).__name__.lower()

        # Check for XGBoost/LightGBM models
        is_tree_model = (
            "xgboost" in model_module or "xgb" in model_name or "lightgbm" in model_module or "lgb" in model_name
        )

        if is_tree_model:
            logger.debug(f"Detected tree model: {model_name} (module: {model_module})")
            return TreeSHAPExplainer

        logger.debug(f"No compatible explainer found for model: {model_name} (module: {model_module})")
        # Could add more heuristics here; otherwise return None to force Kernel via config
        return None


# Initialize registry with available explainers
if EXPLAINERS_AVAILABLE:
    if TreeSHAPExplainer:
        ExplainerRegistry.register("treeshap", TreeSHAPExplainer)
    if KernelSHAPExplainer:
        ExplainerRegistry.register("kernelshap", KernelSHAPExplainer)
