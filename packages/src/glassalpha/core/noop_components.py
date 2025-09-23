"""NoOp implementations for testing and partial pipelines.

These minimal implementations allow the pipeline to run even when
some components are not yet available or need to be disabled.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .interfaces import (
    ModelInterface,
)
from .registry import ExplainerRegistry, MetricRegistry, ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("passthrough", priority=-100)
class PassThroughModel:
    """Minimal model that returns constant predictions."""

    capabilities = {
        "supports_shap": True,  # Works with any explainer
        "supports_proba": True,
        "data_modality": "any",
    }
    version = "1.0.0"

    def __init__(self, default_value: float = 0.5):
        """Initialize with default prediction value.

        Args:
            default_value: Value to return for all predictions

        """
        self.default_value = default_value
        logger.info("PassThroughModel initialized")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return constant predictions.

        Args:
            X: Input features (shape determines output size)

        Returns:
            Array of default values

        """
        n_samples = len(X)
        logger.debug(f"PassThroughModel predicting for {n_samples} samples")
        return np.full(n_samples, self.default_value)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return constant probabilities.

        Args:
            X: Input features

        Returns:
            Array of probabilities (binary classification assumed)

        """
        n_samples = len(X)
        proba = np.zeros((n_samples, 2))
        proba[:, 0] = 1 - self.default_value
        proba[:, 1] = self.default_value
        return proba

    def get_model_type(self) -> str:
        """Return model type."""
        return "passthrough"

    def get_capabilities(self) -> dict[str, Any]:
        """Return model capabilities."""
        return self.capabilities


@ExplainerRegistry.register("noop", priority=-100)
class NoOpExplainer:
    """Minimal explainer that returns empty explanations."""

    capabilities = {
        "supported_models": ["all"],  # Works with any model
        "explanation_type": "none",
        "data_modality": "any",
    }
    version = "1.0.0"
    priority = -100  # Lowest priority fallback

    def __init__(self):
        """Initialize NoOp explainer."""
        logger.info("NoOpExplainer initialized")

    def explain(self, model: ModelInterface, X: pd.DataFrame, y: np.ndarray | None = None) -> dict[str, Any]:
        """Return minimal explanation structure.

        Args:
            model: Model to explain (ignored)
            X: Input data (used for shape)
            y: Target values (ignored)

        Returns:
            Minimal explanation dictionary

        """
        n_samples, n_features = X.shape
        logger.debug(f"NoOpExplainer generating empty explanation for {n_samples} samples")

        return {
            "status": "no_explanation",
            "reason": "NoOp explainer - no actual explanation generated",
            "shap_values": np.zeros((n_samples, n_features)),
            "feature_importance": np.zeros(n_features),
            "base_value": 0.0,
            "explainer_type": "noop",
        }

    def supports_model(self, model: ModelInterface) -> bool:
        """Support all models.

        Args:
            model: Any model

        Returns:
            Always True

        """
        return True

    def get_explanation_type(self) -> str:
        """Return explanation type."""
        return "noop"


@MetricRegistry.register("noop", priority=-100)
class NoOpMetric:
    """Minimal metric that returns placeholder values."""

    metric_type = "noop"
    version = "1.0.0"

    def __init__(self):
        """Initialize NoOp metric."""
        logger.info("NoOpMetric initialized")

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: pd.DataFrame | None = None
    ) -> dict[str, float]:
        """Return placeholder metrics.

        Args:
            y_true: Ground truth (used for basic stats)
            y_pred: Predictions (used for basic stats)
            sensitive_features: Ignored

        Returns:
            Dictionary with placeholder metrics

        """
        logger.debug("NoOpMetric computing placeholder metrics")

        return {
            "noop_metric": 0.0,
            "samples_processed": len(y_true),
            "unique_predictions": len(np.unique(y_pred)),
            "status": "placeholder",
        }

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return ["noop_metric", "samples_processed", "unique_predictions", "status"]

    def requires_sensitive_features(self) -> bool:
        """No sensitive features required."""
        return False


# Auto-register on import
logger.debug("NoOp components registered")
