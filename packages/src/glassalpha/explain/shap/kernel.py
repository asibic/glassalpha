"""KernelSHAP explainer for model-agnostic SHAP explanations.

KernelSHAP provides model-agnostic SHAP explanations by sampling coalitions of features
and fitting a linear model to approximate the original model locally.
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

from ...core.interfaces import ModelInterface
from ...core.registry import ExplainerRegistry

logger = logging.getLogger(__name__)


# Only register if shap is available
if SHAP_AVAILABLE:

    @ExplainerRegistry.register("kernelshap", priority=50)
    class KernelSHAPExplainer:
        """Model-agnostic SHAP explainer using KernelSHAP algorithm."""

        # Class attributes expected by tests
        priority = 50  # Lower than TreeSHAP
        capabilities = {
            "supported_models": ["all"],  # Works with any model
            "explanation_type": "shap_values",
            "supports_local": True,
            "supports_global": True,
            "data_modality": "tabular",
        }
        version = "1.0.0"

        def __init__(self, n_samples: int = 100, background_size: int = 100, link: str = "identity", **kwargs):
            """Initialize KernelSHAP explainer.

            Args:
                n_samples: Number of samples to use for SHAP value estimation
                background_size: Size of background dataset for baseline
                link: Link function for the explainer ('identity' or 'logit')
                **kwargs: Additional parameters for explainer

            """
            self.n_samples = n_samples
            self.background_size = background_size
            self.link = link
            self.explainer = None
            self.background_data = None
            self.base_value = None
            self.background_ = None
            self.feature_names_ = None
            logger.info(f"KernelSHAPExplainer initialized (n_samples={n_samples}, bg_size={background_size})")

        def fit(self, background_X, feature_names=None):
            """Fit the explainer with background data.

            Args:
                background_X: Background data for SHAP
                feature_names: Optional feature names

            """
            self.background_ = background_X
            self.feature_names_ = self._extract_feature_names(background_X, feature_names)
            # Also set the existing background_data for compatibility
            self.background_data = background_X
            logger.debug(f"KernelSHAPExplainer fitted with {len(background_X)} background samples")
            return self

        @staticmethod
        def is_compatible(wrapper) -> bool:
            """Check if this explainer is compatible with the given model.

            KernelSHAP is model-agnostic and works with any model that has predict methods.
            """
            # KernelSHAP supports any model that has predict method
            return hasattr(wrapper, "predict")

        def supports_model(self, model: ModelInterface) -> bool:
            """Check if this explainer supports the given model."""
            return self.is_compatible(model)

        def explain(self, model: ModelInterface, X: pd.DataFrame, y=None) -> dict[str, Any]:
            """Generate SHAP explanations for the model."""
            try:
                # Sample background data if needed
                background_data = self.background_data
                if background_data is None:
                    # Use a sample of X as background if no background provided
                    sample_size = min(self.background_size, len(X))
                    background_data = X.sample(n=sample_size, random_state=42)

                # Create prediction function
                def predict_fn(data):
                    """Wrapper function for model predictions."""
                    try:
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(pd.DataFrame(data, columns=X.columns))
                            # For binary classification, return probability of positive class
                            if proba.shape[1] == 2:
                                return proba[:, 1]
                            # Multi-class: return all probabilities
                            return proba
                    except Exception:
                        # Fall back to predict if predict_proba fails
                        pass

                    return model.predict(pd.DataFrame(data, columns=X.columns))

                # Create KernelSHAP explainer
                if self.explainer is None:
                    self.explainer = shap.KernelExplainer(predict_fn, background_data.values)

                # Limit explanation to reasonable number of samples
                n_explain = min(len(X), self.n_samples)
                X_sample = X.iloc[:n_explain]

                # Calculate SHAP values
                shap_values = self.explainer.shap_values(X_sample.values)

                # Extract feature names
                feature_names = self._extract_feature_names(X)

                # Calculate global feature importance
                feature_importance = self._aggregate_to_global(shap_values)
                feature_importance_dict = dict(zip(feature_names, feature_importance, strict=False))

                return {
                    "status": "success",
                    "shap_values": shap_values,
                    "base_value": getattr(self.explainer, "expected_value", 0.0),
                    "feature_importance": feature_importance_dict,
                    "feature_names": feature_names,
                    "explainer_type": "kernelshap",
                    "n_samples_explained": len(X_sample),
                    "n_features": len(feature_names),
                    "n_background_samples": len(background_data),
                    "kernel_samples_used": self.n_samples,
                    "actually_computed_samples": n_explain,
                }

            except Exception as e:
                logger.exception("Error in KernelSHAP explanation")
                return {"status": "error", "reason": str(e), "explainer_type": "kernelshap"}

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
            return f"KernelSHAPExplainer(priority={self.priority}, version={self.version})"

else:
    # Stub class when shap unavailable
    class KernelSHAPExplainer:
        """Stub class when SHAP library is unavailable."""

        def __init__(self, *args, **kwargs):
            """Initialize stub - raises ImportError."""
            raise ImportError("shap not available - install shap library or fix CI environment")
