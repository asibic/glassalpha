"""LightGBM model wrapper for GlassAlpha.

This wrapper implements the ModelInterface protocol and enables LightGBM models
to work within the GlassAlpha pipeline. It supports SHAP explanations and
provides feature importance capabilities.
"""

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from ...core.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("lightgbm", priority=90)
class LightGBMWrapper:
    """Wrapper for LightGBM models implementing ModelInterface protocol.

    This class wraps LightGBM models to make them compatible with the GlassAlpha
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

    def __init__(self, model_path: str | Path | None = None, model: lgb.Booster | None = None):
        """Initialize LightGBM wrapper.

        Args:
            model_path: Path to pre-trained LightGBM model file
            model: Pre-loaded LightGBM Booster object

        """
        self.model: lgb.Booster | None = model
        self.feature_names: list | None = None
        self.n_classes: int = 2  # Default to binary classification

        if model_path:
            self.load(model_path)
        elif model:
            self.model = model
            self._extract_model_info()

        if self.model:
            logger.info("LightGBMWrapper initialized with model")
        else:
            logger.info("LightGBMWrapper initialized without model")

    def load(self, path: str | Path):
        """Load trained LightGBM model from saved file for inference or analysis.

        Loads a previously saved LightGBM model from disk, supporting both text
        and JSON formats. The model structure, learned parameters, and feature
        mappings are restored to enable predictions and explainability analysis.

        Args:
            path: Path to LightGBM model file (supports .txt, .json, .model extensions)

        Raises:
            FileNotFoundError: If the specified model file does not exist
            LightGBMError: If the model file is corrupted or incompatible format
            OSError: If file system permissions prevent reading the model file
            ValueError: If the model file format is unrecognized or invalid
            ImportError: If LightGBM library is not properly installed

        Note:
            Model loading preserves all training metadata including feature names,
            categorical feature handling, and objective function configuration.
            Loaded models are immediately ready for prediction and explanation.

        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        logger.info(f"Loading LightGBM model from {path}")
        self.model = lgb.Booster(model_file=str(path))
        self._extract_model_info()

    def _extract_model_info(self):
        """Extract metadata from loaded model."""
        if self.model:
            # Try to get feature names
            try:
                self.feature_names = self.model.feature_name()
            except Exception:
                # Graceful fallback - some models don't have feature names
                logger.debug("Could not extract feature names from model")

            # Try to determine number of classes
            try:
                # LightGBM stores the number of classes in the model
                num_class = self.model.num_model_per_iteration()
                if num_class > 1:
                    # Multi-class model
                    self.n_classes = num_class
                else:
                    # Binary classification typically has 1 model per iteration
                    self.n_classes = 2
            except Exception:
                # Graceful fallback - determine from prediction shape later
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

        # Check feature compatibility
        if self.feature_names and list(X.columns) != self.feature_names:
            logger.warning("Input feature names don't match model's expected features")

        # Get raw predictions
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)

        # Handle different prediction formats
        if predictions.ndim == 1:
            # Binary classification - convert probabilities to class predictions
            if np.all((predictions >= 0) & (predictions <= 1)):
                predictions = (predictions > 0.5).astype(int)
        elif predictions.ndim == 2:
            # Multi-class - take argmax
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

        # Get raw predictions (probabilities)
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)

        # Handle binary vs multiclass
        proba = np.column_stack([1 - predictions, predictions]) if predictions.ndim == 1 else predictions

        logger.debug(f"Generated probability predictions for {len(X)} samples")
        return proba

    def get_model_type(self) -> str:
        """Return the model type identifier.

        Returns:
            String identifier for LightGBM models

        """
        return "lightgbm"

    def get_capabilities(self) -> dict[str, Any]:
        """Return model capabilities for plugin selection.

        Returns:
            Dictionary of capability flags

        """
        return self.capabilities

    def get_feature_importance(self, importance_type: str = "split") -> dict[str, float]:
        """Get feature importance scores from the model.

        Args:
            importance_type: Type of importance to return
                - "split": Number of times feature is used to split
                - "gain": Total gain when feature is used to split

        Returns:
            Dictionary mapping feature names to importance scores

        """
        if self.model is None:
            raise ValueError("Model not loaded. Load a model first.")

        # Get importance scores
        importance = self.model.feature_importance(importance_type=importance_type)

        # Get feature names
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]

        # Create dictionary
        importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance, strict=False)}

        logger.debug(f"Extracted {importance_type} feature importance for {len(importance_dict)} features")
        return importance_dict

    def save(self, path: str | Path):
        """Save trained LightGBM model to disk in human-readable text format.

        Persists the complete model structure including learned parameters,
        tree structure, and feature mappings in LightGBM's native text format.
        This format enables model inspection and cross-platform compatibility.

        Args:
            path: Target file path for model storage (recommended: .txt extension)

        Side Effects:
            - Creates or overwrites model file at specified path
            - Saves in text format (~1-50MB depending on model complexity)
            - File contains feature names and model hyperparameters
            - Preserves exact model state for reproducible predictions

        Raises:
            ValueError: If no trained model exists to save
            IOError: If path is not writable or insufficient disk space
            LightGBMError: If model serialization fails due to corrupted state

        Note:
            Text format enables model audit and regulatory inspection but is
            larger than binary format. For compliance audits, this transparency
            is preferred over compact binary serialization.

        """
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save in text format for better compatibility
        self.model.save_model(str(path))
        logger.info(f"Saved LightGBM model to {path}")

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        status = "loaded" if self.model else "not loaded"
        return f"LightGBMWrapper(status={status}, n_classes={self.n_classes}, version={self.version})"
