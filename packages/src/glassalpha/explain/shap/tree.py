"""TreeSHAP explainer for tree-based models.

TreeSHAP is an exact, efficient algorithm for computing SHAP values for tree-based
models. It provides local explanations and can aggregate to global feature importance.
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

    @ExplainerRegistry.register("treeshap", priority=100)
    class TreeSHAPExplainer(ExplainerBase):
        """Tree-based SHAP explainer with expected API contract."""

        # Class attributes expected by tests
        priority = 100  # Higher than KernelSHAP
        version = "1.0.0"

        def __init__(self) -> None:
            """Initialize TreeSHAP explainer.

            Tests expect 'explainer' attribute to exist and be None before fit().
            """
            # Tests expect this attribute to exist and be None before fit
            self.explainer = None
            self.feature_names: Sequence[str] | None = None
            logger.info("TreeSHAPExplainer initialized")

        def fit(self, wrapper: Any, background_X, feature_names: Sequence[str] | None = None):
            """Fit the explainer with a model wrapper and background data.

            Args:
                wrapper: Model wrapper with predict/predict_proba methods
                background_X: Background data for explainer baseline
                feature_names: Optional feature names for interpretation

            Returns:
                self: Returns self for chaining

            """
            if background_X is None or getattr(background_X, "shape", (0, 0))[0] == 0:
                raise ValueError("TreeSHAPExplainer: background data is empty")

            # Get the underlying model from wrapper
            model = getattr(wrapper, "model", None) or wrapper

            # Create TreeSHAP explainer - prefer generic API, fall back to TreeExplainer
            try:
                self.explainer = shap.Explainer(model, background_X)
                logger.debug("Created generic SHAP Explainer")
            except Exception as e:
                logger.debug(f"Generic Explainer failed: {e}, falling back to TreeExplainer")
                self.explainer = shap.TreeExplainer(model)

            # Extract and store feature names
            if feature_names is not None:
                self.feature_names = list(feature_names)
            elif hasattr(background_X, "columns"):
                self.feature_names = list(background_X.columns)
            else:
                self.feature_names = None

            logger.debug(f"TreeSHAPExplainer fitted with {len(background_X)} background samples")
            return self

        def explain(self, X):
            """Generate SHAP explanations for input data.

            Args:
                X: Input data to explain

            Returns:
                Dictionary containing SHAP values, base values, and feature names

            """
            if self.explainer is None:
                raise RuntimeError("TreeSHAPExplainer: call fit() before explain()")

            logger.debug(f"Generating SHAP explanations for {len(X)} samples")

            # Get SHAP explanations
            res = self.explainer(X)  # shap.Explanation in modern shap

            try:
                # Modern SHAP returns Explanation object
                shap_values = np.array(res.values)
                base_values = np.array(res.base_values)
            except Exception:
                # Older fallback if a raw array is returned
                shap_values = np.array(res)
                base_values = np.zeros(len(X))

            return {
                "shap_values": shap_values,
                "base_values": base_values,
                "feature_names": self.feature_names,
                "explainer_type": "treeshap",
                "n_samples_explained": len(X),
            }

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
