"""Registry for explainer classes and compatibility detection."""

from __future__ import annotations

import importlib.util
import inspect
import logging
from collections.abc import Iterable

from glassalpha.constants import NO_EXPLAINER_MSG
from glassalpha.core.decor_registry import DecoratorFriendlyRegistry

logger = logging.getLogger(__name__)


def _has(mod: str) -> bool:
    """Check if a module is available."""
    return importlib.util.find_spec(mod) is not None


# Single source of truth for the exact error copy the contract test expects.
# If the test already asserts a different string, set this constant to match it.
ERR_UNSUPPORTED_MODEL = "No compatible explainer found for model type '{model_type}'."

SUPPORTED_FAMILIES: dict[str, list[str]] = {
    # sensible defaults per family; order = preference
    "logistic_regression": ["coefficients", "permutation", "kernelshap"],
    "linear_regression": ["coefficients", "permutation", "kernelshap"],
    "linear_svc": ["coefficients", "permutation", "kernelshap"],
    "random_forest": ["treeshap", "permutation", "kernelshap"],
    "extra_trees": ["treeshap", "permutation", "kernelshap"],
    "xgboost": ["treeshap", "permutation", "kernelshap"],
    "lightgbm": ["treeshap", "permutation", "kernelshap"],
    # add others you officially support here
}

REQUIRES: dict[str, list[str]] = {
    "kernelshap": ["shap"],
    "treeshap": ["shap"],
    "coefficients": [],
    "permutation": [],
}


def _available(name: str) -> bool:
    if name not in REQUIRES:
        return False
    return all(_has(m) for m in REQUIRES[name])


def _first_available(candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if _available(c):
            return c
    return None


def select_explainer(model_type: str, requested_priority: list[str] | None = None) -> str:
    """Select the best available explainer for a model type.

    Args:
        model_type: Type of model (e.g., 'xgboost', 'logistic_regression')
        requested_priority: Optional explicit priority list from config

    Returns:
        Name of selected explainer

    Raises:
        RuntimeError: If no suitable explainer is available or model type is unsupported

    """
    # Normalize model type
    model_type = model_type.strip().lower()

    # explicit user priority → strict but helpful
    if requested_priority:
        chosen = _first_available(requested_priority)
        if chosen:
            return chosen
        raise RuntimeError(
            f"No explainer from {requested_priority} is available. "
            "Try installing optional deps, for example: pip install 'glassalpha[explain]'",
        )

    # No priority provided → enforce known families
    if model_type not in SUPPORTED_FAMILIES:
        # CONTRACT: for unknown model families we must raise exactly RuntimeError
        # with a deterministic, stable message.
        raise RuntimeError(NO_EXPLAINER_MSG)

    # Known family: pick best available
    fam_order = SUPPORTED_FAMILIES[model_type]
    chosen = _first_available(fam_order)
    if chosen:
        return chosen

    # Optional: a universal fallback list for known families, if your product policy allows it
    universal = ["permutation", "kernelshap"]
    chosen = _first_available(universal)
    if chosen:
        return chosen

    # Nothing available even after universal fallback
    raise RuntimeError(
        "No explainer is available for the current environment. "
        'Install optional dependencies, e.g.: pip install "glassalpha[explain]".',
    )


# Priority order for explainer selection (TreeSHAP preferred for tree models)
PRIORITY = ("treeshap", "kernelshap")

# Model type to compatible explainers mapping
TYPE_TO_EXPLAINERS = {
    "xgboost": ("treeshap", "kernelshap"),
    "lightgbm": ("treeshap", "kernelshap"),
    "random_forest": ("treeshap", "kernelshap"),
    "decision_tree": ("treeshap", "kernelshap"),
    "logistic_regression": ("kernelshap", "coefficients"),
    "linear_model": ("kernelshap", "coefficients"),
    "linearregression": ("coefficients",),  # sklearn LinearRegression
    "logisticregression": ("coefficients",),  # sklearn LogisticRegression
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

        # 2) Check class-level is_compatible method
        try:
            cls = self.get(name)
            if hasattr(cls, "is_compatible"):
                # Try new keyword-only signature first (Phase 2.5 standardization)
                try:
                    return bool(cls.is_compatible(model=model, model_type=model_type, config=None))
                except TypeError as e:
                    # Handle legacy signatures gracefully
                    logger.warning(
                        f"{name}.is_compatible has incompatible signature: {e}. "
                        f"Please update to: @classmethod is_compatible(cls, *, model=None, model_type=None, config=None)",
                    )

                    # Try legacy fallback patterns
                    try:
                        # Try with just model parameter (old instance method pattern)
                        sig = inspect.signature(cls.is_compatible)
                        if "model_type" in sig.parameters:
                            # Old keyword pattern
                            return bool(cls.is_compatible(model_type=model_type, model=model))
                        # Old positional pattern
                        return bool(cls.is_compatible(model if model is not None else model_type))
                    except Exception:  # noqa: BLE001
                        # If all attempts fail, treat as incompatible
                        logger.error(f"Could not determine compatibility for {name}, treating as incompatible")
                        return False
        except (KeyError, ImportError) as e:
            logger.debug(f"Explainer {name} not available: {e}")
            return False
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error checking compatibility for {name}: {e}")
            return False

        # 3) Default: not compatible (prevents silent fallbacks)
        return False

    def find_compatible(self, model, config: dict | None = None) -> str:
        """Find compatible explainer for model.

        Args:
            model: Model object or model type string
            config: Optional configuration dict

        Returns:
            Explainer name

        Raises:
            RuntimeError: If no compatible explainer found

        """
        # Extract model type
        if isinstance(model, str):
            model_type = model.lower()
        else:
            info = getattr(model, "get_model_info", dict)() or {}
            model_type = (
                info.get("type")
                or info.get("model_type")
                or getattr(model, "type", None)
                or model.__class__.__name__.lower()
            )

        # Get priority list from config
        priority_list = ((config or {}).get("explainers") or {}).get("priority")

        # Use new smart selection logic - pass the model_type string
        try:
            selected = select_explainer(model_type, priority_list)

            # Log the selection with reason
            if priority_list:
                logger.info(f"Explainer: {selected} (reason: user-specified priority)")
            else:
                # Determine reason for selection based on model family
                if model_type in SUPPORTED_FAMILIES:
                    family_order = SUPPORTED_FAMILIES[model_type]
                    if selected == family_order[0]:
                        reason = f"preferred for {model_type} family"
                    else:
                        reason = f"fallback for {model_type} family (preferred not available)"
                else:
                    reason = "fallback selection"

                if selected == "coefficients":
                    reason += ", linear model coefficients"
                elif selected == "treeshap" or selected == "kernelshap":
                    reason += ", SHAP detected"
                elif selected == "permutation":
                    reason += ", SHAP not installed"

                logger.info(f"Explainer: {selected} (reason: {reason})")

            return selected

        except RuntimeError as e:
            # If new logic fails, fall back to old logic for compatibility
            logger.warning(f"New explainer selection failed: {e}, falling back to legacy logic")
            return self._find_compatible_legacy(model, config)

    def _find_compatible_legacy(self, model, config: dict | None = None) -> str:
        """Legacy explainer selection logic for fallback compatibility."""
        # Debug: check which registry instance this is
        logger.debug(f"_find_compatible_legacy called on registry with id: {id(self)}")
        logger.debug(f"Available explainers: {self.names()}")

        # Extract model type
        if isinstance(model, str):
            model_type = model.lower()
        else:
            info = getattr(model, "get_model_info", dict)() or {}
            model_type = (
                info.get("type")
                or info.get("model_type")
                or getattr(model, "type", None)
                or model.__class__.__name__.lower()
            )

        # Get priority list from config or use default ordering
        prio = ((config or {}).get("explainers") or {}).get("priority")
        if prio:
            # Use explicit priority list from config
            candidates = prio
        else:
            # Use default ordering but exclude noop unless explicitly requested
            candidates = [name for name in getattr(self, "names_by_priority", self.names)() if name != "noop"]

        # First pass: try explicit candidates only
        for name in candidates:
            # Skip if not available
            if not self.has(name):
                logger.debug(f"Skipping {name}: not available")
                continue

            # Skip enterprise explainers if no license (unless explicitly requested)
            is_enterprise_explainer = self._meta.get(name, {}).get("enterprise", False)
            logger.debug(f"Checking {name}: enterprise={is_enterprise_explainer}")
            if is_enterprise_explainer:
                from ..core.features import is_enterprise

                has_license = is_enterprise()
                logger.debug(f"Enterprise explainer {name}: has_license={has_license}")
                if not has_license:
                    logger.debug(f"Skipping enterprise explainer {name}: no license")
                    continue

            # Check compatibility
            try:
                is_compatible = self.is_compatible(name, model_type, model=model)
                logger.debug(f"Compatibility check for {name}: {is_compatible}")
                if is_compatible:
                    logger.debug(f"Selecting {name}")
                    return name
            except ImportError as e:
                logger.debug(f"Skipping {name} due to ImportError: {e}")
                # Skip unavailable explainers
                continue

        # Second pass: model-specific fallbacks
        fallback_names = []
        if model_type in TYPE_TO_EXPLAINERS:
            fallback_names = TYPE_TO_EXPLAINERS[model_type]

        for name in fallback_names:
            # Skip if not available
            if not self.has(name):
                continue

            # Skip if already tried in first pass
            if name in candidates:
                continue

            # Skip enterprise explainers if no license (unless explicitly requested)
            if self._meta.get(name, {}).get("enterprise", False):
                from ..core.features import is_enterprise

                if not is_enterprise():
                    continue

            # Check compatibility
            try:
                if self.is_compatible(name, model_type, model=model):
                    logger.info(
                        f"Explainer selection: none of {list(candidates)} available; "
                        f"using fallback '{name}' for model '{model_type}'.",
                    )
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

# Register the built-in explainers
from .noop import NoOpExplainer

ExplainerRegistry.register("noop", NoOpExplainer, priority=-100)

# Register coefficients explainer (no dependencies)
from .coefficients import CoefficientsExplainer

ExplainerRegistry.register(
    "coefficients",
    CoefficientsExplainer,
    priority=10,
    supports=["logistic_regression", "linear_regression", "sklearn-linear"],
)

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


# Export both the real registry and the compat class for backward compatibility
__all__ = [
    "ERR_UNSUPPORTED_MODEL",
    "REQUIRES",
    "SUPPORTED_FAMILIES",
    "ExplainerRegistry",
    "ExplainerRegistryCompat",
    "_available",
    "_first_available",
    "is_available",
    "select_explainer",
]
