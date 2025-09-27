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
        self.n_classes_: int = 2  # Default to binary classification
        self.classes_: np.ndarray | None = None  # Original class labels

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

    def _canonicalize_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize XGBoost parameters to handle aliases and ensure consistency.

        Args:
            kwargs: Raw parameters from configuration

        Returns:
            Canonicalized parameters

        """
        params = dict(kwargs)

        # Handle seed aliases
        if "random_state" not in params and "seed" in params:
            params["random_state"] = params["seed"]

        # Handle boosting rounds aliases
        if "n_estimators" not in params and "num_boost_round" in params:
            params["n_estimators"] = params["num_boost_round"]

        # Keep num_class - do not filter it out, it's needed for validation
        # XGBClassifier will handle it appropriately

        return params

    def fit(self, X: Any, y: Any, require_proba: bool = True, **kwargs: Any) -> "XGBoostWrapper":  # noqa: ANN401, N803
        """Fit XGBoost model with training data.

        Args:
            X: Training features (DataFrame or array)
            y: Target values
            require_proba: If True, ensure model can produce probabilities for audits
            **kwargs: Additional parameters including objective, num_class, etc.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If objective and num_class are inconsistent with data

        """
        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415

        # Convert inputs to proper format
        X = pd.DataFrame(X)
        y = np.asarray(y)

        # Normalize labels to 0..k-1 but keep original labels for mapping
        classes, y_enc = np.unique(y, return_inverse=True)
        self.classes_ = classes
        self.n_classes_ = classes.size

        # Store feature names
        self.feature_names_ = list(X.columns)

        # Set defaults that can be overridden by kwargs
        defaults = {"max_depth": 6, "eta": 0.1}

        # Auto-infer objective if not provided
        obj = kwargs.get("objective")
        if obj is None:
            obj = "multi:softprob" if self.n_classes_ > 2 else "binary:logistic"
            kwargs["objective"] = obj

        # Handle require_proba: coerce multi:softmax to multi:softprob for audit compatibility
        if require_proba and obj == "multi:softmax":
            logger.warning("Coercing multi:softmax to multi:softprob for audit compatibility (predict_proba required)")
            kwargs["objective"] = "multi:softprob"
            obj = "multi:softprob"

        # Strict validations before training
        if "binary" in obj and self.n_classes_ > 2:
            msg = f"Binary objective '{obj}' incompatible with {self.n_classes_} classes"
            raise ValueError(msg)

        if obj.startswith("multi:"):
            num_class_cfg = kwargs.get("num_class", self.n_classes_)
            if num_class_cfg != self.n_classes_:
                msg = f"num_class={num_class_cfg} does not match observed classes={self.n_classes_}"
                raise ValueError(msg)

        # Canonicalize parameters to handle aliases
        params = self._canonicalize_params({**defaults, **kwargs})

        # Create and fit XGBoost model
        from xgboost import XGBClassifier  # noqa: PLC0415

        logger.info(f"Creating XGBClassifier with params: {params}")
        logger.info(f"Target classes: {self.classes_}, n_classes: {self.n_classes_}")

        sklearn_model = XGBClassifier(**params)
        sklearn_model.fit(X, y_enc)

        # Keep XGBClassifier for full functionality (predict_proba, etc.)
        self.model = sklearn_model

        # Mark as fitted
        self._is_fitted = True

        logger.info(f"Fitted XGBoost model: {self.n_classes_} classes, objective={params.get('objective', 'unknown')}")

        return self

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

        logger.info(f"Loading XGBoost model from {path}")

        # Contract compliance: Load JSON with model/feature_names_/n_classes
        import json  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        try:
            with Path(path).open(encoding="utf-8") as f:
                data = json.load(f)

            # Reconstruct from saved structure using temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                tmp.write(data["model"].encode("utf-8"))
                tmp.flush()

                self.model = xgb.Booster()
                self.model.load_model(tmp.name)

            # Clean up temp file
            Path(tmp.name).unlink(missing_ok=True)

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
                config = self.model.save_config()
                if '"num_class"' in config:
                    import json  # noqa: PLC0415

                    config_dict = json.loads(config)
                    learner = config_dict.get("learner", {})
                    learner_params = learner.get("learner_model_param", {})
                    num_class = learner_params.get("num_class", "0")
                    self.n_classes_ = max(2, int(num_class))
                else:
                    self.n_classes_ = 2  # Default to binary
            except Exception:  # noqa: BLE001
                # Graceful fallback - config parsing can fail for various reasons
                logger.debug("Could not determine number of classes, defaulting to binary")
                self.n_classes_ = 2

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Generate predictions for input data.

        Args:
            X: Input features as DataFrame

        Returns:
            Array of predictions (original class labels)

        """
        self._ensure_fitted()

        # Convert to DataFrame if needed
        X = pd.DataFrame(X)

        # Align features using utility function
        if self.feature_names_:
            from ...utils.features import align_features  # noqa: PLC0415

            X = align_features(X, self.feature_names_)

        # Get encoded predictions (0..k-1)
        encoded_preds = self.model.predict(X)

        # Convert to class indices
        if self.n_classes_ > 2:
            # Multi-class: argmax returns class indices
            class_indices = np.argmax(encoded_preds, axis=1) if encoded_preds.ndim > 1 else encoded_preds
        else:
            # Binary: threshold predictions
            class_indices = (encoded_preds > self.BINARY_THRESHOLD).astype(int)

        # Map indices back to original labels
        predictions = self.classes_[class_indices.astype(int)]

        logger.debug(f"Generated predictions for {len(X)} samples")
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Generate probability predictions for input data.

        Args:
            X: Input features as DataFrame

        Returns:
            Array of prediction probabilities (n_samples, n_classes)

        """
        self._ensure_fitted()

        # Convert to DataFrame if needed
        X = pd.DataFrame(X)

        # Align features using utility function
        if self.feature_names_:
            from ...utils.features import align_features  # noqa: PLC0415

            X = align_features(X, self.feature_names_)

        # Get probabilities from XGBoost
        probs = self.model.predict_proba(X)

        # Ensure shape (n, k) even for binary
        if probs.ndim == 1:
            # Binary case - XGBoost sometimes returns 1D array
            proba1 = probs
            proba0 = 1.0 - proba1
            probs = np.column_stack([proba0, proba1])

        logger.debug(f"Generated probability predictions for {len(X)} samples")
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
            "n_classes": self.n_classes_,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
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

        logger.debug(f"Extracted {importance_type} feature importance for {len(importance)} features")
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
        import tempfile  # noqa: PLC0415

        # Save model to temp file then read as string
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            self.model.save_model(tmp.name)
            model_str = Path(tmp.name).read_text(encoding="utf-8")

        # Clean up temp file
        Path(tmp.name).unlink(missing_ok=True)

        data = {
            "model": model_str,
            "feature_names_": self.feature_names_,
            "n_classes": self.n_classes,
        }

        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved XGBoost model to {path}")

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        status = "loaded" if self.model else "not loaded"
        return f"XGBoostWrapper(status={status}, n_classes={self.n_classes}, version={self.version})"
