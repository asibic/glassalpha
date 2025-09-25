"""Scikit-learn model wrappers for GlassAlpha.

This module provides wrappers for scikit-learn models to make them compatible
with the GlassAlpha audit pipeline. It supports SHAP explanations and
provides model-specific feature importance capabilities.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Conditional sklearn import with graceful fallback for CI compatibility
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback stubs when sklearn unavailable (CI environment issues)
    SKLEARN_AVAILABLE = False
    BaseEstimator = object  # Minimal fallback base class
    ClassifierMixin = object
    LogisticRegression = None

from ...core.registry import ModelRegistry

logger = logging.getLogger(__name__)


# Only register if sklearn is available
if SKLEARN_AVAILABLE:

    @ModelRegistry.register("logistic_regression", priority=80)
    class LogisticRegressionWrapper(BaseEstimator, ClassifierMixin):
        """Wrapper for scikit-learn LogisticRegression models implementing ModelInterface protocol.

        This class wraps sklearn LogisticRegression models to make them compatible with the
        GlassAlpha audit pipeline. It supports loading pre-trained models, predictions, and
        capability declaration for plugin selection.
        """
        
        # Required class attributes for ModelInterface
        capabilities = {
            "supports_shap": True,  # Will use KernelSHAP, not TreeSHAP
            "supports_feature_importance": True,
            "supports_proba": True,
            "data_modality": "tabular",
        }
        version = "1.0.0"

        def __init__(self, model_path: str | Path | None = None, model: LogisticRegression | None = None, **kwargs):
            """Initialize LogisticRegression wrapper.

            Args:
                model_path: Path to pre-trained sklearn model file (pickle/joblib)
                model: Pre-trained LogisticRegression model object
                **kwargs: Additional parameters passed to LogisticRegression constructor if no model provided

            """
            self.model: LogisticRegression | None = None
            self.feature_names: list | None = None
            self.n_classes: int = 2  # Default to binary classification
            self._sklearn_params = kwargs  # Store sklearn parameters for get_params/set_params

            if model_path:
                self.load(model_path)
            elif model is not None:
                self.model = model
                self._extract_model_info()
            elif kwargs:
                # Create new LogisticRegression with provided parameters
                self.model = LogisticRegression(**kwargs)

            if self.model:
                logger.info("LogisticRegressionWrapper initialized with model")
            else:
                logger.info("LogisticRegressionWrapper initialized without model")

        def fit(self, X, y):
            """Fit the LogisticRegression model.
            
            Args:
                X: Training data features
                y: Training data targets
                
            Returns:
                self: Returns self for chaining
            """
            if self.model is None:
                self.model = LogisticRegression(**self._sklearn_params)
            
            self.model.fit(X, y)
            self._extract_model_info()
            return self

        def get_params(self, deep=True):
            """Get parameters for this estimator."""
            if self.model is not None:
                return self.model.get_params(deep=deep)
            return self._sklearn_params.copy()

        def set_params(self, **params):
            """Set parameters for this estimator."""
            self._sklearn_params.update(params)
            if self.model is not None:
                self.model.set_params(**params)
            return self

        @property
        def classes_(self):
            """Access to classes_ attribute for sklearn compatibility."""
            if self.model is not None and hasattr(self.model, 'classes_'):
                return self.model.classes_
            return None

        def load(self, path: str | Path):
            """Load trained scikit-learn model from saved file for inference and analysis."""
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            logger.info(f"Loading sklearn model from {path}")

            # Try joblib first (recommended for sklearn), then pickle
            try:
                self.model = joblib.load(path)
            except Exception as joblib_error:
                try:
                    with open(path, "rb") as f:
                        self.model = pickle.load(f)
                except Exception as pickle_error:
                    raise ValueError(
                        f"Could not load model with joblib ({joblib_error}) or pickle ({pickle_error})"
                    ) from pickle_error

            # Verify it's a LogisticRegression model
            if not isinstance(self.model, LogisticRegression):
                raise ValueError(f"Expected LogisticRegression, got {type(self.model)}")

            self._extract_model_info()

        def _extract_model_info(self):
            """Extract metadata from loaded model."""
            if self.model:
                # Try to get feature names (if available from recent sklearn versions)
                try:
                    if hasattr(self.model, "feature_names_in_"):
                        self.feature_names = list(self.model.feature_names_in_)
                except Exception:
                    # Graceful fallback - feature names not available
                    logger.debug("Could not extract feature names from model")

                # Determine number of classes
                try:
                    if hasattr(self.model, "classes_"):
                        self.n_classes = len(self.model.classes_)
                    else:
                        self.n_classes = 2  # Default assumption
                except Exception:
                    # Graceful fallback
                    logger.debug("Could not determine number of classes, defaulting to binary")
                    self.n_classes = 2

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            """Generate predictions for input data."""
            if self.model is None:
                raise ValueError("Model not loaded. Load a model first.")

            # Check feature compatibility
            if self.feature_names and list(X.columns) != self.feature_names:
                logger.warning("Input feature names don't match model's expected features")

            # Get predictions
            predictions = self.model.predict(X)

            logger.debug(f"Generated predictions for {len(X)} samples")
            return predictions

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            """Generate probability predictions for input data."""
            if self.model is None:
                raise ValueError("Model not loaded. Load a model first.")

            # Get probability predictions
            probabilities = self.model.predict_proba(X)

            logger.debug(f"Generated probability predictions for {len(X)} samples")
            return probabilities

        def get_model_type(self) -> str:
            """Return the model type identifier."""
            return "logistic_regression"

        def get_capabilities(self) -> dict[str, Any]:
            """Return model capabilities for plugin selection."""
            return self.capabilities

        def get_feature_importance(self, importance_type: str = "coef") -> dict[str, float]:
            """Get feature importance scores from the model."""
            if self.model is None:
                raise ValueError("Model not loaded. Load a model first.")

            # Get coefficients
            if hasattr(self.model, "coef_"):
                coef = self.model.coef_

                # Handle multi-class case (coef is 2D)
                if coef.ndim == 2:
                    # Binary vs multi-class classification
                    coef = coef.flatten() if coef.shape[0] == 1 else np.abs(coef).mean(axis=0)

                # Apply importance type
                if importance_type == "coef":
                    importance_values = np.abs(coef)
                elif importance_type == "coef_raw":
                    importance_values = coef
                else:
                    raise ValueError(f"Unknown importance_type: {importance_type}")
            else:
                raise ValueError("Model has no coefficients (coef_) attribute")

            # Create feature names if not available
            n_features = len(importance_values)
            if self.feature_names and len(self.feature_names) == n_features:
                feature_names = self.feature_names
            else:
                feature_names = [f"feature_{i}" for i in range(n_features)]

            # Create dictionary
            importance_dict = {
                name: float(importance) for name, importance in zip(feature_names, importance_values, strict=False)
            }

            logger.debug(f"Extracted {importance_type} feature importance for {len(importance_dict)} features")
            return importance_dict

        def save(self, path: str | Path, use_joblib: bool = True):
            """Save trained scikit-learn model to disk with optimized serialization."""
            if self.model is None:
                raise ValueError("No model to save")

            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            if use_joblib:
                joblib.dump(self.model, path)
                logger.info(f"Saved sklearn model to {path} using joblib")
            else:
                with open(path, "wb") as f:
                    pickle.dump(self.model, f)
                logger.info(f"Saved sklearn model to {path} using pickle")

        def get_model_info(self) -> dict[str, Any]:
            """Get additional model information specific to LogisticRegression."""
            if self.model is None:
                return {"status": "not_loaded"}

            info = {
                "status": "loaded",
                "n_classes": self.n_classes,
                "n_features": self.model.n_features_in_ if hasattr(self.model, "n_features_in_") else None,
                "solver": self.model.solver,
                "penalty": self.model.penalty,
                "C": self.model.C,
                "max_iter": self.model.max_iter,
            }

            if hasattr(self.model, "n_iter_"):
                info["n_iter"] = (
                    self.model.n_iter_.tolist() if hasattr(self.model.n_iter_, "tolist") else self.model.n_iter_
                )

            return info

        def __repr__(self) -> str:
            """String representation of the wrapper."""
            status = "loaded" if self.model else "not loaded"
            return f"LogisticRegressionWrapper(status={status}, n_classes={self.n_classes}, version={self.version})"

else:
    # Stub class when sklearn unavailable
    class LogisticRegressionWrapper:
        """Stub class when sklearn is unavailable."""

        def __init__(self, *args, **kwargs):
            """Initialize stub - raises ImportError."""
            raise ImportError("sklearn not available - install scikit-learn or fix CI environment")


# Generic sklearn wrapper (keeping existing implementation but simplified)
if SKLEARN_AVAILABLE:

    @ModelRegistry.register("sklearn_generic", priority=50)
    class SklearnGenericWrapper(BaseEstimator):
        """Generic wrapper for any scikit-learn estimator implementing ModelInterface protocol."""
        
        capabilities = {
            "supports_shap": True,  # Will use KernelSHAP
            "supports_feature_importance": False,  # Not all models have feature importance
            "supports_proba": False,  # Not all models have predict_proba
            "data_modality": "tabular",
        }
        version = "1.0.0"

        def __init__(self, model_path: str | Path | None = None, model: BaseEstimator | None = None):
            """Initialize generic sklearn wrapper."""
            self.model: BaseEstimator | None = model
            self.feature_names: list | None = None
            self.n_classes: int = 2  # Default assumption

            if model_path:
                self.load(model_path)
            elif model:
                self.model = model
                self._extract_model_info()
                self._update_capabilities()

        def load(self, path: str | Path):
            """Load trained scikit-learn model from saved file."""
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            try:
                self.model = joblib.load(path)
            except Exception:
                with open(path, "rb") as f:
                    self.model = pickle.load(f)

            if not isinstance(self.model, BaseEstimator):
                raise ValueError(f"Expected sklearn BaseEstimator, got {type(self.model)}")

            self._extract_model_info()
            self._update_capabilities()

        def _extract_model_info(self):
            """Extract metadata from loaded model."""
            if self.model and hasattr(self.model, "feature_names_in_"):
                self.feature_names = list(self.model.feature_names_in_)

        def _update_capabilities(self):
            """Update capabilities based on the loaded model."""
            if self.model:
                self.capabilities["supports_proba"] = hasattr(self.model, "predict_proba")
                has_importance = any([
                    hasattr(self.model, "coef_"),
                    hasattr(self.model, "feature_importances_"),
                ])
                self.capabilities["supports_feature_importance"] = has_importance

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            """Generate predictions for input data."""
            if self.model is None:
                raise ValueError("Model not loaded. Load a model first.")
            return self.model.predict(X)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            """Generate probability predictions for input data."""
            if self.model is None:
                raise ValueError("Model not loaded. Load a model first.")
            if not hasattr(self.model, "predict_proba"):
                raise ValueError(f"Model {type(self.model).__name__} does not support predict_proba")
            return self.model.predict_proba(X)

        def get_model_type(self) -> str:
            """Return the model type identifier."""
            return "sklearn_generic"

        def get_capabilities(self) -> dict[str, Any]:
            """Return model capabilities for plugin selection."""
            return self.capabilities

else:
    # Stub class when sklearn unavailable
    class SklearnGenericWrapper:
        """Stub class when sklearn is unavailable."""

        def __init__(self, *args, **kwargs):
            """Initialize stub - raises ImportError."""
            raise ImportError("sklearn not available - install scikit-learn or fix CI environment")
