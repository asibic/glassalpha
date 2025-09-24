"""XGBoost model wrapper for GlassAlpha.

This wrapper implements the ModelInterface protocol and enables XGBoost models
to work within the GlassAlpha pipeline. It supports SHAP explanations and
provides feature importance capabilities.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from ...core.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("xgboost", priority=100)
class XGBoostWrapper:
    """Wrapper for XGBoost models implementing ModelInterface protocol.

    This class wraps XGBoost models to make them compatible with the GlassAlpha
    audit pipeline. It supports loading pre-trained models, predictions, and
    capability declaration for plugin selection.
    """

    # Required class attributes for ModelInterface
    capabilities = {
        "supports_shap": True,
        "supports_feature_importance": True,
        "supports_proba": True,
        "data_modality": "tabular",
    }
    version = "1.0.0"

    def __init__(self, model_path: str | Path | None = None, model: xgb.Booster | None = None):
        """Initialize XGBoost wrapper.

        Args:
            model_path: Path to pre-trained XGBoost model file
            model: Pre-loaded XGBoost Booster object

        """
        self.model: xgb.Booster | None = model
        self.feature_names: list | None = None
        self.n_classes: int = 2  # Default to binary classification

        if model_path:
            self.load(model_path)
        elif model:
            self.model = model
            self._extract_model_info()

        if self.model:
            logger.info("XGBoostWrapper initialized with model")
        else:
            logger.info("XGBoostWrapper initialized without model")

    def load(self, path: str | Path):
        """Load XGBoost model from file.

        Args:
            path: Path to XGBoost model file (JSON or binary format)

        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        logger.info(f"Loading XGBoost model from {path}")
        self.model = xgb.Booster()
        self.model.load_model(str(path))
        self._extract_model_info()

    def _extract_model_info(self):
        """Extract metadata from loaded model."""
        if self.model:
            # Try to get feature names
            try:
                self.feature_names = self.model.feature_names
            except Exception:
                # Graceful fallback - some models don't have feature names
                logger.debug("Could not extract feature names from model")

            # Try to determine number of classes
            try:
                # For binary classification, XGBoost often outputs single value
                # For multiclass, it outputs multiple values
                # This is a heuristic - may need refinement based on actual usage
                config = self.model.save_config()
                if '"num_class"' in config:
                    import json

                    config_dict = json.loads(config)
                    learner = config_dict.get("learner", {})
                    learner_params = learner.get("learner_model_param", {})
                    num_class = learner_params.get("num_class", "0")
                    self.n_classes = max(2, int(num_class))
                else:
                    self.n_classes = 2  # Default to binary
            except Exception:
                # Graceful fallback - config parsing can fail for various reasons
                logger.debug("Could not determine number of classes, defaulting to binary")
                self.n_classes = 2

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for input data.

        Args:
            X: Input features as DataFrame

        Returns:
            Array of predictions

        """
        if self.model is None:
            raise ValueError("Model not loaded. Load a model first.")

        # Convert to DMatrix for XGBoost
        if self.feature_names and list(X.columns) != self.feature_names:
            logger.warning("Input feature names don't match model's expected features")

        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))

        # Get raw predictions
        predictions = self.model.predict(dmatrix)

        # For binary classification with probability output, convert to class predictions
        if self.n_classes == 2 and predictions.ndim == 1:
            # If predictions are probabilities, convert to binary classes
            if np.all((predictions >= 0) & (predictions <= 1)):
                predictions = (predictions > 0.5).astype(int)
        elif predictions.ndim == 2:
            # Multiclass - take argmax
            predictions = np.argmax(predictions, axis=1)

        logger.debug(f"Generated predictions for {len(X)} samples")
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions for input data.

        Args:
            X: Input features as DataFrame

        Returns:
            Array of prediction probabilities (n_samples, n_classes)

        """
        if self.model is None:
            raise ValueError("Model not loaded. Load a model first.")

        # Convert to DMatrix
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))

        # Get raw predictions (probabilities)
        predictions = self.model.predict(dmatrix)

        # Handle binary vs multiclass
        if self.n_classes == 2 and predictions.ndim == 1:
            # Binary classification - create 2-column probability matrix
            proba = np.column_stack([1 - predictions, predictions])
        else:
            # Multiclass or already in correct format
            proba = predictions

        logger.debug(f"Generated probability predictions for {len(X)} samples")
        return proba

    def get_model_type(self) -> str:
        """Return the model type identifier.

        Returns:
            String identifier for XGBoost models

        """
        return "xgboost"

    def get_capabilities(self) -> dict[str, Any]:
        """Return model capabilities for plugin selection.

        Returns:
            Dictionary of capability flags

        """
        return self.capabilities

    def get_feature_importance(self, importance_type: str = "weight") -> dict[str, float]:
        """Get feature importance scores from the model.

        Args:
            importance_type: Type of importance to return
                - "weight": Number of times feature is used to split
                - "gain": Average gain when feature is used
                - "cover": Average coverage when feature is used

        Returns:
            Dictionary mapping feature names to importance scores

        """
        if self.model is None:
            raise ValueError("Model not loaded. Load a model first.")

        # Get importance scores
        importance = self.model.get_score(importance_type=importance_type)

        logger.debug(f"Extracted {importance_type} feature importance for {len(importance)} features")
        return importance

    def save(self, path: str | Path):
        """Save the model to file.

        Args:
            path: Path to save the model (will save in JSON format)

        """
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save in JSON format for better compatibility
        self.model.save_model(str(path))
        logger.info(f"Saved XGBoost model to {path}")

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        status = "loaded" if self.model else "not loaded"
        return f"XGBoostWrapper(status={status}, n_classes={self.n_classes}, version={self.version})"
