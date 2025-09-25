"""TreeSHAP explainer for tree-based models.

TreeSHAP is an exact, efficient algorithm for computing SHAP values for tree-based
models. It provides local explanations and can aggregate to global feature importance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

# Conditional shap import with graceful fallback for CI compatibility
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    # Fallback when shap unavailable (CI environment issues)
    SHAP_AVAILABLE = False
    shap = None

# Optional backends; don't hard-fail if not installed
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None  # type: ignore

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore

from ...core.interfaces import ModelInterface
from ...core.registry import ExplainerRegistry

logger = logging.getLogger(__name__)


# Only register if shap is available
if SHAP_AVAILABLE:

    @ExplainerRegistry.register("treeshap", priority=100)
    class TreeSHAPExplainer:
        """Tree-based SHAP explainer with a clean, stable API."""

        # Class attributes expected by tests
        priority = 100  # Higher than KernelSHAP (which should be 50)
        capabilities = {
            "supported_models": ["xgboost", "lightgbm", "random_forest", "decision_tree"],
            "explanation_type": "shap_values",
            "supports_local": True,
            "supports_global": True,
            "data_modality": "tabular",
        }
        version = "1.0.0"

        def __init__(self, check_additivity: bool = False, **kwargs: Any) -> None:
            """Initialize TreeSHAP explainer.

            Args:
                check_additivity: Whether to check SHAP value additivity
                **kwargs: Additional parameters for explainer

            """
            self.check_additivity = check_additivity
            self.options = kwargs
            self.wrapper = None
            self.explainer = None
            self.background_ = None
            self.feature_names_ = None
            logger.info("TreeSHAPExplainer initialized")

        def fit(self, background_X, feature_names=None):
            """Fit the explainer with background data.

            Args:
                background_X: Background data for SHAP
                feature_names: Optional feature names

            """
            self.background_ = background_X
            self.feature_names_ = self._extract_feature_names(background_X, feature_names)
            logger.debug(f"TreeSHAPExplainer fitted with {len(background_X)} background samples")
            return self

        @staticmethod
        def is_compatible(wrapper) -> bool:
            """Check if this explainer is compatible with the given model."""
            # Get the underlying model
            model = getattr(wrapper, "model", wrapper)
            model_type = getattr(wrapper, "get_model_type", lambda: str(type(model).__name__.lower()))()

            # Check compatible model types
            compatible_types = ["xgboost", "lightgbm", "random_forest", "decision_tree", "gradient_boosting"]

            # Also check by class type for sklearn models
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                GradientBoostingRegressor,
                RandomForestClassifier,
                RandomForestRegressor,
            )
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

            sklearn_tree_types = (
                DecisionTreeClassifier,
                DecisionTreeRegressor,
                RandomForestClassifier,
                RandomForestRegressor,
                GradientBoostingClassifier,
                GradientBoostingRegressor,
            )

            # Check by model type string or class instance
            is_compatible = (
                model_type in compatible_types
                or isinstance(model, sklearn_tree_types)
                or (xgb is not None and model.__class__.__module__.startswith("xgboost"))
                or (lgb is not None and model.__class__.__module__.startswith("lightgbm"))
            )

            logger.debug(f"TreeSHAP compatibility check for {model_type}: {is_compatible}")
            return is_compatible

        def supports_model(self, model: ModelInterface) -> bool:
            """Check if this explainer supports the given model."""
            return self.is_compatible(model)

        def explain(self, model: ModelInterface, X: pd.DataFrame, y=None) -> dict[str, Any]:
            """Generate SHAP explanations for the model."""
            try:
                if not self.supports_model(model):
                    return {
                        "status": "error",
                        "reason": f"Model type '{model.get_model_type()}' not supported by TreeSHAP",
                        "explainer_type": "treeshap",
                    }

                # Get the underlying model object
                underlying_model = getattr(model, "model", model)

                # Create TreeSHAP explainer
                if self.explainer is None:
                    self.explainer = shap.TreeExplainer(underlying_model)

                # Calculate SHAP values
                shap_values = self.explainer.shap_values(X)

                # Handle multi-class output
                if isinstance(shap_values, list):
                    # For binary classification, use positive class
                    if len(shap_values) == 2:
                        shap_values_array = shap_values[1]
                    else:
                        # Multi-class: average across classes for global importance
                        shap_values_array = np.mean([np.abs(v) for v in shap_values], axis=0)
                else:
                    shap_values_array = shap_values

                # Extract feature names
                feature_names = self._extract_feature_names(X)

                # Calculate global feature importance
                feature_importance = self._aggregate_to_global(shap_values_array)
                feature_importance_dict = dict(zip(feature_names, feature_importance, strict=False))

                return {
                    "status": "success",
                    "shap_values": shap_values_array,
                    "base_value": getattr(self.explainer, "expected_value", 0.0),
                    "feature_importance": feature_importance_dict,
                    "feature_names": feature_names,
                    "explainer_type": "treeshap",
                    "n_samples_explained": len(X),
                    "n_features": len(feature_names),
                }

            except Exception as e:
                logger.exception("Error in TreeSHAP explanation")
                return {"status": "error", "reason": str(e), "explainer_type": "treeshap"}

        def explain_local(self, model, X, **kwargs):
            """Generate local explanations (per-sample SHAP values)."""
            result = self.explain(model, X)
            if result["status"] == "success":
                return result["shap_values"]
            return None

        def _extract_feature_names(self, X, feature_names=None):
            """Extract feature names from data."""
            if feature_names is not None:
                return list(feature_names)
            if hasattr(X, "columns"):
                return list(X.columns)
            return [f"feature_{i}" for i in range(X.shape[1])]

        def _aggregate_to_global(self, shap_values):
            """Aggregate local SHAP values to global feature importance."""
            return np.mean(np.abs(shap_values), axis=0)

        def get_explanation_type(self) -> str:
            """Return the type of explanation provided."""
            return "shap_values"

        def __repr__(self) -> str:
            """String representation of the explainer."""
            return f"TreeSHAPExplainer(priority={self.priority}, version={self.version})"

else:
    # Stub class when shap unavailable
    class TreeSHAPExplainer:
        """Stub class when SHAP library is unavailable."""

        def __init__(self, *args, **kwargs):
            """Initialize stub - raises ImportError."""
            raise ImportError("shap not available - install shap library or fix CI environment")
