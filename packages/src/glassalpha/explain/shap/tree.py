"""TreeSHAP explainer for tree-based models.

TreeSHAP is an exact, efficient algorithm for computing SHAP values for tree-based
models. It provides local explanations and can aggregate to global feature importance.
"""

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

from ...core.interfaces import ModelInterface
from ...core.registry import ExplainerRegistry

logger = logging.getLogger(__name__)


# Only register if shap is available
if SHAP_AVAILABLE:

    @ExplainerRegistry.register("treeshap", priority=100)
    class TreeSHAPExplainer:
        """TreeSHAP explainer for tree-based models.

        This explainer uses the TreeSHAP algorithm to compute exact SHAP values
        for tree-based models like XGBoost, LightGBM, and Random Forest. It has
        the highest priority for these model types as it's both exact and efficient.
        """

        # Required class attributes for ExplainerInterface
        capabilities = {
            "supported_models": ["xgboost", "lightgbm", "random_forest", "decision_tree"],
            "explanation_type": "shap_values",
            "supports_local": True,
            "supports_global": True,
            "data_modality": "tabular",
        }
        version = "1.0.0"
        priority = 100  # Highest priority for tree models

        def __init__(self, check_additivity: bool = False, **kwargs):
            """Initialize TreeSHAP explainer.

            Args:
                check_additivity: Whether to check SHAP value additivity
                **kwargs: Additional parameters for explainer

            """
            self.check_additivity = check_additivity
            self.explainer = None
            self.base_value = None
            self.background_ = None
            self.feature_names_ = None
            logger.info("TreeSHAPExplainer initialized")

    def explain(self, model: ModelInterface, X: pd.DataFrame, y: np.ndarray | None = None) -> dict[str, Any]:
        """Generate SHAP explanations for the model.

        Args:
            model: Model to explain (must be tree-based)
            X: Input data to explain
            y: Optional target values (not used by TreeSHAP)

        Returns:
            Dictionary containing:
                - status: Success or error status
                - shap_values: SHAP values for each sample and feature
                - base_value: Expected value (baseline) for predictions
                - feature_importance: Global feature importance (mean absolute SHAP)
                - explainer_type: Type of explainer used
                - feature_names: Names of features

        """
        try:
            # Check if model is supported
            if not self.supports_model(model):
                return {
                    "status": "error",
                    "reason": f"Model type '{model.get_model_type()}' not supported by TreeSHAP",
                    "explainer_type": "treeshap",
                }

            # Get the underlying model object
            model_type = model.get_model_type()

            if model_type == "xgboost":
                # For XGBoost, use the Booster object directly
                if hasattr(model, "model") and model.model is not None:
                    underlying_model = model.model
                else:
                    raise ValueError("XGBoost model not properly loaded")
            elif model_type == "lightgbm":
                # For LightGBM, use the Booster object
                if hasattr(model, "model") and model.model is not None:
                    underlying_model = model.model
                else:
                    raise ValueError("LightGBM model not properly loaded")
            else:
                # For sklearn models, use the model directly
                underlying_model = model

            # Create TreeSHAP explainer
            logger.info(f"Creating TreeSHAP explainer for {model_type} model")
            self.explainer = shap.TreeExplainer(
                underlying_model,
                feature_perturbation="tree_path_dependent",  # More accurate for tree models
            )

            # Store base value
            if hasattr(self.explainer, "expected_value"):
                self.base_value = self.explainer.expected_value
                # Handle multi-class case
                if isinstance(self.base_value, np.ndarray) and len(self.base_value) > 1:
                    # For binary classification, often we focus on the positive class
                    # For multi-class, we might need all values
                    logger.debug(f"Multi-class model with {len(self.base_value)} classes")
            else:
                self.base_value = 0.0

            # Calculate SHAP values
            logger.info(f"Computing SHAP values for {len(X)} samples")
            shap_values = self.explainer.shap_values(X)

            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class output - take the positive class for binary or all for multi
                if len(shap_values) == 2:
                    # Binary classification - use positive class
                    shap_values_array = shap_values[1]
                    base_value_scalar = (
                        self.base_value[1] if isinstance(self.base_value, np.ndarray) else self.base_value
                    )
                else:
                    # Multi-class - would need special handling
                    shap_values_array = shap_values
                    base_value_scalar = self.base_value
            else:
                # Single output (regression or binary with single output)
                shap_values_array = shap_values
                if isinstance(self.base_value, np.ndarray):
                    if len(self.base_value) == 2:
                        base_value_scalar = float(self.base_value[1])
                    else:
                        base_value_scalar = float(self.base_value[0])
                else:
                    base_value_scalar = float(self.base_value)

            # Calculate global feature importance (mean absolute SHAP values)
            if isinstance(shap_values_array, list):
                # Multi-class case - average across classes
                feature_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values_array], axis=0)
            else:
                feature_importance = np.abs(shap_values_array).mean(axis=0)

            # Create feature importance dictionary
            feature_names = list(X.columns)
            feature_importance_dict = {
                name: float(importance) for name, importance in zip(feature_names, feature_importance, strict=False)
            }

            # Sort by importance
            feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))

            logger.info("SHAP explanation completed successfully")

            return {
                "status": "success",
                "shap_values": shap_values_array,
                "base_value": base_value_scalar,
                "feature_importance": feature_importance,
                "feature_importance_dict": feature_importance_dict,
                "feature_names": feature_names,
                "explainer_type": "treeshap",
                "n_samples_explained": len(X),
                "n_features": len(feature_names),
            }

        except Exception as e:
            logger.error(f"Error in TreeSHAP explanation: {e!s}", exc_info=True)
            return {"status": "error", "reason": str(e), "explainer_type": "treeshap"}

        def fit(self, background_X, feature_names=None):
            """Fit the explainer with background data.

            Args:
                background_X: Background data for SHAP
                feature_names: Optional feature names

            """
            self.background_ = background_X
            self.feature_names_ = self._extract_feature_names(background_X, feature_names)
            logger.debug(f"TreeSHAPExplainer fitted with {len(background_X)} background samples")

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

        def supports_model(self, model: ModelInterface) -> bool:
            """Check if this explainer supports the given model.

            Args:
                model: Model to check compatibility

            Returns:
                True if model is a supported tree-based model

            """
            model_type = model.get_model_type()
            supported = model_type in self.capabilities["supported_models"]

            if supported:
                logger.debug(f"TreeSHAP supports model type: {model_type}")
            else:
                logger.debug(f"TreeSHAP does not support model type: {model_type}")

            return supported

    def get_explanation_type(self) -> str:
        """Return the type of explanation provided.

        Returns:
            String identifier for SHAP value explanations

        """
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
