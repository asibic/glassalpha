"""KernelSHAP explainer for model-agnostic SHAP explanations.

KernelSHAP provides model-agnostic SHAP explanations by sampling coalitions of features
and fitting a linear model to approximate the original model locally.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

# Conditional shap import with graceful fallback for CI compatibility
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    # Fallback when shap unavailable (CI environment issues)
    SHAP_AVAILABLE = False
    shap = None

from ...core.registry import ExplainerRegistry
from ..base import ExplainerBase

logger = logging.getLogger(__name__)


# Only register if shap is available
if SHAP_AVAILABLE:

    @ExplainerRegistry.register("kernelshap", priority=50)
    class KernelSHAPExplainer(ExplainerBase):
        """Model-agnostic SHAP explainer using KernelSHAP algorithm."""

        # Class attributes expected by tests
        priority = 50  # Lower than TreeSHAP
        version = "1.0.0"

        def __init__(self) -> None:
            """Initialize KernelSHAP explainer.

            Tests expect 'explainer' attribute to exist and be None before fit().
            """
            # Tests expect this attribute to exist and be None before fit
            self.explainer = None
            self.feature_names: Sequence[str] | None = None
            logger.info("KernelSHAPExplainer initialized")

        def fit(self, wrapper: Any, background_X, feature_names: Sequence[str] | None = None):
            """Fit the explainer with a model wrapper and background data.

            IMPORTANT: Tests call fit(wrapper, background_df) - wrapper is first parameter.

            Args:
                wrapper: Model wrapper with predict/predict_proba methods
                background_X: Background data for explainer baseline
                feature_names: Optional feature names for interpretation

            Returns:
                self: Returns self for chaining

            """
            if background_X is None or getattr(background_X, "shape", (0, 0))[0] == 0:
                raise ValueError("KernelSHAPExplainer: background data is empty")

            # Build a callable f(X) -> proba or prediction
            if hasattr(wrapper, "predict_proba"):
                f = lambda X: wrapper.predict_proba(X)
                logger.debug("Using predict_proba for KernelSHAP")
            elif hasattr(wrapper, "predict"):
                f = lambda X: wrapper.predict(X)
                logger.debug("Using predict for KernelSHAP")
            else:
                raise ValueError("KernelSHAPExplainer: wrapper lacks predict/predict_proba")

            # Create KernelSHAP explainer
            self.explainer = shap.KernelExplainer(f, background_X)

            # Extract and store feature names
            if feature_names is not None:
                self.feature_names = list(feature_names)
            elif hasattr(background_X, "columns"):
                self.feature_names = list(background_X.columns)
            else:
                self.feature_names = None

            logger.debug(f"KernelSHAPExplainer fitted with {len(background_X)} background samples")
            return self

        def explain(self, X, nsamples: int = 100):
            """Generate SHAP explanations for input data.

            Args:
                X: Input data to explain
                nsamples: Number of samples for KernelSHAP approximation

            Returns:
                Dictionary containing SHAP values and feature names

            """
            if self.explainer is None:
                raise RuntimeError("KernelSHAPExplainer: call fit() before explain()")

            logger.debug(f"Generating KernelSHAP explanations for {len(X)} samples with {nsamples} samples")

            # Calculate SHAP values
            shap_values = np.array(self.explainer.shap_values(X, nsamples=nsamples))

            return {
                "shap_values": shap_values,
                "feature_names": self.feature_names,
                "explainer_type": "kernelshap",
                "n_samples_explained": len(X),
                "kernel_samples_used": nsamples,
            }

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
