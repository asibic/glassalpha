"""XGBoost model wrapper for GlassAlpha.

This wrapper implements the ModelInterface protocol and enables XGBoost models
to work within the GlassAlpha pipeline. It supports SHAP explanations and
provides feature importance capabilities.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import xgboost as xgb

from glassalpha.core.registry import ModelRegistry

from .base import BaseTabularWrapper

logger = logging.getLogger(__name__)


@ModelRegistry.register("xgboost", priority=100)
class XGBoostWrapper(BaseTabularWrapper):
    """Wrapper for XGBoost models implementing ModelInterface protocol.

    This class wraps XGBoost models to make them compatible with the GlassAlpha
    audit pipeline. It supports loading pre-trained models, predictions, and
    capability declaration for plugin selection.
    """

    # Required class attributes for ModelInterface
    capabilities: ClassVar[dict[str, Any]] = {
        "supports_shap": True,
        "supports_feature_importance": True,
        "supports_proba": True,
        "data_modality": "tabular",
    }
    version = "1.0.0"

    # Contract compliance: Binary classification constants
    BINARY_CLASSES = 2
    BINARY_THRESHOLD = 0.5

    def __init__(self, model_path: str | Path | None = None, model: xgb.Booster | None = None) -> None:
        """Initialize XGBoost wrapper.

        Args:
            model_path: Path to pre-trained XGBoost model file
            model: Pre-loaded XGBoost Booster object

        """
        super().__init__()
        self.model: xgb.Booster | None = model
        self.feature_names: list | None = None
        self.feature_names_: list | None = None  # For sklearn compatibility
        self.n_classes: int = 2  # Default to binary classification

        if model_path:
            self.load(model_path)
        elif model:
            self.model = model
            self._is_fitted = True
            self._extract_model_info()

        if self.model:
            logger.info("XGBoostWrapper initialized with model")
        else:
            logger.info("XGBoostWrapper initialized without model")

    def load(self, path: str | Path) -> "XGBoostWrapper":
        """Load trained XGBoost model from saved file for inference and analysis.

        Loads a previously saved XGBoost booster model from disk, supporting both
        JSON and binary formats. The complete model structure including boosted
        trees, learned parameters, and training configuration is restored for
        immediate use in predictions and explainability analysis.

        Args:
            path: Path to XGBoost model file (supports .json, .model, .ubj extensions)

        Raises:
            FileNotFoundError: If the specified model file does not exist
            XGBoostError: If the model file is corrupted, incompatible, or malformed
            OSError: If file system permissions prevent reading the model file
            ValueError: If the model file format is unrecognized by XGBoost
            ImportError: If XGBoost library is not properly installed or configured
            JSONDecodeError: If JSON model file contains malformed JSON data

        Note:
            Model loading restores full training context including objective function,
            evaluation metrics, and feature importance calculations. Loaded models
            maintain compatibility with SHAP explainers and audit requirements.

        """
        path = Path(path)
        if not path.exists():
            msg = f"Model file not found: {path}"
            raise FileNotFoundError(msg)

        logger.info("Loading XGBoost model from %s", path)

        # Contract compliance: Load JSON with model/feature_names_/n_classes
        import json  # noqa: PLC0415

        try:
            with Path(path).open(encoding="utf-8") as f:
                data = json.load(f)

            # Reconstruct from saved structure
            self.model = xgb.Booster()
            self.model.load_model_from_string(data["model"])
            self.feature_names_ = data.get("feature_names_")
            self.n_classes = data.get("n_classes", 2)
            self._is_fitted = True

        except (json.JSONDecodeError, KeyError):
            # Fallback to direct XGBoost model loading (old format)
            self.model = xgb.Booster()
            self.model.load_model(str(path))
            self._is_fitted = True
            self._extract_model_info()

        return self

    def _extract_model_info(self) -> None:
        """Extract metadata from loaded model."""
        if self.model:
            # Try to get feature names
            try:
                self.feature_names = self.model.feature_names
            except Exception:  # noqa: BLE001
                # Graceful fallback - some models don't have feature names
                logger.debug("Could not extract feature names from model")

            # Try to determine number of classes
            try:
                # For binary classification, XGBoost often outputs single value
                # For multiclass, it outputs multiple values
                # This is a heuristic - may need refinement based on actual usage
                config = self.model.save_config()
                if '"num_class"' in config:
                    import json  # noqa: PLC0415

                    config_dict = json.loads(config)
                    learner = config_dict.get("learner", {})
                    learner_params = learner.get("learner_model_param", {})
                    num_class = learner_params.get("num_class", "0")
                    self.n_classes = max(2, int(num_class))
                else:
                    self.n_classes = 2  # Default to binary
            except Exception:  # noqa: BLE001
                # Graceful fallback - config parsing can fail for various reasons
                logger.debug("Could not determine number of classes, defaulting to binary")
                self.n_classes = 2

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Generate predictions for input data.

        Args:
            X: Input features as DataFrame

        Returns:
            Array of predictions

        """
        self._ensure_fitted()

        # Use centralized feature handling
        X_processed = self._prepare_x(X)  # noqa: N806

        # Convert to DMatrix for XGBoost
        feature_names = list(X.columns) if hasattr(X, "columns") else None
        if hasattr(X_processed, "columns"):
            feature_names = list(X_processed.columns)

        dmatrix = xgb.DMatrix(X_processed, feature_names=feature_names)

        # Get raw predictions
        preds = self.model.predict(dmatrix)

        # Contract compliance: Handle binary/multiclass shape consistently
        if self.n_classes == self.BINARY_CLASSES and preds.ndim == 1:
            # Convert logits/prob of class 1 into two-column proba
            proba1 = preds
            proba0 = 1.0 - proba1
            probs = np.column_stack([proba0, proba1])
        else:
            probs = preds

        # Use argmax for predict to get class predictions
        predictions = np.argmax(probs, axis=1) if probs.ndim > 1 else (probs > self.BINARY_THRESHOLD).astype(int)

        logger.debug("Generated predictions for %s samples", len(X))
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Generate probability predictions for input data.

        Args:
            X: Input features as DataFrame

        Returns:
            Array of prediction probabilities (n_samples, n_classes)

        """
        self._ensure_fitted()

        # Use centralized feature handling
        X_processed = self._prepare_x(X)  # noqa: N806

        # Convert to DMatrix
        feature_names = list(X.columns) if hasattr(X, "columns") else None
        if hasattr(X_processed, "columns"):
            feature_names = list(X_processed.columns)

        dmatrix = xgb.DMatrix(X_processed, feature_names=feature_names)

        # Get raw predictions (probabilities)
        preds = self.model.predict(dmatrix)

        # Contract compliance: Handle binary/multiclass shape consistently
        if self.n_classes == self.BINARY_CLASSES and preds.ndim == 1:
            # Convert logits/prob of class 1 into two-column proba
            proba1 = preds
            proba0 = 1.0 - proba1
            probs = np.column_stack([proba0, proba1])
        else:
            probs = preds

        logger.debug("Generated probability predictions for %s samples", len(X))
        return probs

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

    def get_model_info(self) -> dict[str, Any]:
        """Get model information including n_classes.

        Returns:
            Dictionary with model information

        """
        return {
            "model_type": "xgboost",
            "n_classes": self.n_classes,
            "status": "loaded" if self.model else "not_loaded",
            "n_features": len(self.feature_names_) if self.feature_names_ else None,
        }

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
        self._ensure_fitted()

        # Get importance scores
        importance = self.model.get_score(importance_type=importance_type)

        logger.debug("Extracted %s feature importance for %s features", importance_type, len(importance))
        return importance

    def save(self, path: str | Path) -> None:
        """Save trained XGBoost model to disk in structured JSON format.

        Persists the complete model structure including boosted trees, learned
        parameters, and feature mappings in XGBoost's native JSON format.
        This format provides cross-platform compatibility and enables detailed
        model inspection for compliance verification.

        Args:
            path: Target file path for model storage (recommended: .json extension)

        Side Effects:
            - Creates or overwrites model file at specified path
            - Creates parent directories if they don't exist
            - Saves in JSON format (~500KB-20MB depending on model complexity)
            - File contains complete tree structure and training metadata

        Raises:
            ValueError: If no trained model exists to save
            IOError: If path is not writable or insufficient disk space
            XGBoostError: If model serialization fails due to corrupted state

        Note:
            JSON format enables detailed model audit for regulatory compliance.
            Contains full tree structure, split conditions, and leaf values
            required for explainability and bias analysis verification.

        """
        self._ensure_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Contract compliance: Save JSON with model/feature_names_/n_classes
        import json  # noqa: PLC0415

        data = {
            "model": self.model.save_model_to_string(),
            "feature_names_": self.feature_names_,
            "n_classes": self.n_classes,
        }

        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved XGBoost model to %s", path)

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        status = "loaded" if self.model else "not loaded"
        return f"XGBoostWrapper(status={status}, n_classes={self.n_classes}, version={self.version})"
