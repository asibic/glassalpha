"""KernelSHAP explainer for model-agnostic explanations.

KernelSHAP is a model-agnostic SHAP explainer that can work with any model
that provides predictions. It uses sampling and linear regression to estimate
SHAP values, making it slower but more general than TreeSHAP.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

from ...core.interfaces import ModelInterface
from ...core.registry import ExplainerRegistry

logger = logging.getLogger(__name__)


@ExplainerRegistry.register("kernelshap", priority=50)
class KernelSHAPExplainer:
    """KernelSHAP explainer for model-agnostic SHAP explanations.

    This explainer uses the KernelSHAP algorithm to compute approximate SHAP values
    for any model that provides predictions. It has lower priority than TreeSHAP
    but works with any model type including linear models, SVMs, and other sklearn models.
    """

    # Required class attributes for ExplainerInterface
    capabilities = {
        "supported_models": ["all"],  # Works with any model
        "explanation_type": "shap_values",
        "supports_local": True,
        "supports_global": True,
        "data_modality": "tabular",
    }
    version = "1.0.0"
    priority = 50  # Lower priority than TreeSHAP

    def __init__(self, n_samples: int = 100, background_size: int = 100, link: str = "identity"):
        """Initialize KernelSHAP explainer.

        Args:
            n_samples: Number of samples to use for SHAP value estimation
            background_size: Size of background dataset for baseline
            link: Link function for the explainer ('identity' or 'logit')

        """
        self.n_samples = n_samples
        self.background_size = background_size
        self.link = link
        self.explainer = None
        self.background_data = None
        self.base_value = None
        logger.info(f"KernelSHAPExplainer initialized (n_samples={n_samples}, bg_size={background_size})")

    def explain(self, model: ModelInterface, X: pd.DataFrame, y: np.ndarray | None = None) -> dict[str, Any]:
        """Generate SHAP explanations for the model.

        Args:
            model: Model to explain (any model with predict/predict_proba)
            X: Input data to explain
            y: Optional target values (not used by KernelSHAP but can help with background)

        Returns:
            Dictionary containing:
                - status: Success or error status
                - shap_values: SHAP values for each sample and feature
                - base_value: Expected value (baseline) for predictions
                - feature_importance: Global feature importance (mean absolute SHAP)
                - explainer_type: Type of explainer used
                - feature_names: Names of features
                - n_samples: Number of samples used for estimation

        """
        try:
            # Check if model is supported (KernelSHAP supports all models)
            if not self.supports_model(model):
                return {
                    "status": "error",
                    "reason": "Model not supported by KernelSHAP (this shouldn't happen)",
                    "explainer_type": "kernelshap",
                }

            # Create background dataset for KernelSHAP
            # Use a sample of the input data as background
            n_background = min(self.background_size, len(X))
            if n_background < len(X):
                # Sample background data
                background_idx = np.random.choice(len(X), n_background, replace=False)
                self.background_data = X.iloc[background_idx]
            else:
                # Use all data as background
                self.background_data = X

            logger.info(f"Using background dataset of size {len(self.background_data)}")

            # Create prediction function for KernelSHAP
            def predict_fn(data):
                """Prediction function for KernelSHAP."""
                if isinstance(data, np.ndarray):
                    data = pd.DataFrame(data, columns=X.columns)

                # Try predict_proba first (for classification), then predict
                if hasattr(model, "predict_proba") and model.get_capabilities().get("supports_proba", False):
                    try:
                        proba = model.predict_proba(data)
                        # For binary classification, return positive class probability
                        if proba.shape[1] == 2:
                            return proba[:, 1]
                        else:
                            # Multi-class: return all probabilities
                            return proba
                    except Exception:
                        # Fall back to predict if predict_proba fails
                        return model.predict(data)
                else:
                    return model.predict(data)

            # Create KernelSHAP explainer
            logger.info(f"Creating KernelSHAP explainer for {model.get_model_type()} model")
            self.explainer = shap.KernelExplainer(predict_fn, self.background_data, link=self.link)

            # Store base value (expected value from explainer)
            self.base_value = self.explainer.expected_value
            if isinstance(self.base_value, np.ndarray):
                # For multi-class, take the first class or mean
                if len(self.base_value) == 1:
                    self.base_value = float(self.base_value[0])
                else:
                    # Multi-class case - could take mean or handle differently
                    self.base_value = float(self.base_value.mean())
            else:
                self.base_value = float(self.base_value)

            # Calculate SHAP values
            # Use a smaller subset for speed if dataset is large
            n_explain = min(len(X), 200)  # Limit for reasonable computation time
            if n_explain < len(X):
                explain_idx = np.random.choice(len(X), n_explain, replace=False)
                X_explain = X.iloc[explain_idx]
                logger.info(f"Computing SHAP values for {n_explain} samples (subset of {len(X)})")
            else:
                X_explain = X
                logger.info(f"Computing SHAP values for {len(X_explain)} samples")

            shap_values = self.explainer.shap_values(
                X_explain,
                nsamples=self.n_samples,
                silent=True,  # Suppress progress bar
            )

            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class output - take the positive class for binary or first class
                # Binary vs multi-class classification
                shap_values_array = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            else:
                # Single output
                shap_values_array = shap_values

            # If we used a subset, we need to handle the shape difference
            if n_explain < len(X):
                # Extend SHAP values to match original data size
                # For non-explained samples, set SHAP values to 0
                full_shap_values = np.zeros((len(X), shap_values_array.shape[1]))
                full_shap_values[explain_idx] = shap_values_array
                shap_values_array = full_shap_values
                logger.info(f"Extended SHAP values to full dataset size: {shap_values_array.shape}")

            # Calculate global feature importance (mean absolute SHAP values)
            # Only use the actually computed SHAP values for importance
            importance_shap = shap_values_array[explain_idx] if n_explain < len(X) else shap_values_array

            feature_importance = np.abs(importance_shap).mean(axis=0)

            # Create feature importance dictionary
            feature_names = list(X.columns)
            feature_importance_dict = {
                name: float(importance) for name, importance in zip(feature_names, feature_importance, strict=False)
            }

            # Sort by importance
            feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))

            logger.info("KernelSHAP explanation completed successfully")

            return {
                "status": "success",
                "shap_values": shap_values_array,
                "base_value": self.base_value,
                "feature_importance": feature_importance,
                "feature_importance_dict": feature_importance_dict,
                "feature_names": feature_names,
                "explainer_type": "kernelshap",
                "n_samples_explained": len(X),
                "n_features": len(feature_names),
                "n_background_samples": len(self.background_data),
                "kernel_samples_used": self.n_samples,
                "actually_computed_samples": n_explain,
            }

        except Exception as e:
            logger.error(f"Error in KernelSHAP explanation: {str(e)}", exc_info=True)
            return {"status": "error", "reason": str(e), "explainer_type": "kernelshap"}

    def supports_model(self, model: ModelInterface) -> bool:
        """Check if this explainer supports the given model.

        KernelSHAP is model-agnostic and supports any model with predict method.

        Args:
            model: Model to check compatibility

        Returns:
            True (KernelSHAP supports all models)

        """
        # KernelSHAP supports any model that has predict method
        supported = hasattr(model, "predict")

        if supported:
            logger.debug(f"KernelSHAP supports model type: {model.get_model_type()}")
        else:
            logger.debug(f"KernelSHAP does not support model (no predict method): {model.get_model_type()}")

        return supported

    def get_explanation_type(self) -> str:
        """Return the type of explanation provided.

        Returns:
            String identifier for SHAP value explanations

        """
        return "shap_values"

    def __repr__(self) -> str:
        """String representation of the explainer."""
        return f"KernelSHAPExplainer(priority={self.priority}, n_samples={self.n_samples}, version={self.version})"
