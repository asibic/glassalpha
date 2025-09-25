"""Scikit-learn model wrappers for Glass Alpha audit pipeline.

This module provides wrappers for scikit-learn models, making them compatible
with the Glass Alpha audit interface. It includes LogisticRegression and
generic scikit-learn model wrappers.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence
import logging
import numpy as np
import joblib

logger = logging.getLogger(__name__)

# Conditional imports for sklearn
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback when sklearn unavailable
    BaseEstimator = object
    ClassifierMixin = object
    LogisticRegression = None
    SKLEARN_AVAILABLE = False

from ...core.registry import ModelRegistry


class SklearnGenericWrapper:
    def __init__(self, model: Any, feature_names: Optional[Sequence[str]] = None):
        self.model = model
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.capabilities = {
            "predict": hasattr(model, "predict"),
            "predict_proba": hasattr(model, "predict_proba"),
            "feature_importance": hasattr(model, "feature_importances_") or hasattr(model, "coef_"),
        }

    def predict(self, X):
        if not hasattr(self.model, "predict"):
            raise AttributeError("Underlying model has no predict")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Underlying model has no predict_proba")
        return self.model.predict_proba(X)

    def feature_importance(self):
        if hasattr(self.model, "feature_importances_"):
            vals = np.asarray(self.model.feature_importances_)
        elif hasattr(self.model, "coef_"):
            vals = np.asarray(self.model.coef_)
            if vals.ndim > 1:
                vals = np.mean(np.abs(vals), axis=0)
            else:
                vals = np.abs(vals)
        else:
            raise AttributeError("Model exposes no feature importances")
        names = self.feature_names or [f"f{i}" for i in range(len(vals))]
        return dict(zip(names, vals.tolist()))

    def save(self, path: str):
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)

    @classmethod
    def load(cls, path: str) -> "SklearnGenericWrapper":
        data = joblib.load(path)
        return cls(model=data["model"], feature_names=data.get("feature_names"))

    # Some tests call instance.load(); keep a passthrough
    def load_instance(self, path: str) -> "SklearnGenericWrapper":
        return self.__class__.load(path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={type(self.model).__name__})"


class LogisticRegressionWrapper(SklearnGenericWrapper):
    pass


# Only register if sklearn is available
if SKLEARN_AVAILABLE:
    
    @ModelRegistry.register("logistic_regression", priority=80)
    class LogisticRegressionWrapper(BaseEstimator, ClassifierMixin):
        """Wrapper for scikit-learn LogisticRegression with Glass Alpha compatibility."""

        # Required class attributes for ModelInterface
        capabilities = {
            "supports_shap": True,
            "supports_feature_importance": True,
            "supports_proba": True,
            "data_modality": "tabular",
            "feature_names": True,
        }
        version = "1.0.0"
        model_type = "logistic_regression"

        def __init__(self, model=None, feature_names=None, **kwargs):
            """Initialize LogisticRegression wrapper.

            Args:
                model: Pre-fitted LogisticRegression model or None to create new one
                feature_names: List of feature names
                **kwargs: Parameters passed to LogisticRegression constructor

            """
            if model is not None:
                # Use provided model
                self.model = model
                self.feature_names = list(feature_names) if feature_names else None
            else:
                # Create new model with parameters
                if kwargs:
                    self.model = LogisticRegression(**kwargs)
                else:
                    self.model = LogisticRegression(random_state=42, max_iter=1000)
                self.feature_names = list(feature_names) if feature_names else None

            # Initialize status
            self._is_fitted = hasattr(self.model, "coef_") and self.model.coef_ is not None

            logger.info("LogisticRegressionWrapper initialized")

        def fit(self, X, y, **kwargs):
            """Fit the logistic regression model."""
            # Store feature names if X is DataFrame
            if hasattr(X, "columns") and self.feature_names is None:
                self.feature_names = list(X.columns)

            # Fit the model
            self.model.fit(X, y, **kwargs)
            self._is_fitted = True

            # Store classes for compatibility
            self.classes_ = self.model.classes_

            return self

        def predict(self, X):
            """Make predictions."""
            if not self._is_fitted:
                raise RuntimeError("Model not fitted")

            # Validate and reorder features if needed
            X_processed = self._validate_and_reorder_features(X)
            predictions = self.model.predict(X_processed)
            
            # Ensure 1D numpy array output
            return np.array(predictions).flatten()

        def predict_proba(self, X):
            """Get prediction probabilities."""
            if not self._is_fitted:
                raise RuntimeError("Model not fitted")

            # Validate and reorder features if needed
            X_processed = self._validate_and_reorder_features(X)
            return self.model.predict_proba(X_processed)

        def _validate_and_reorder_features(self, X):
            """Validate and reorder features to match training data."""
            # If no stored feature names, return as-is
            if self.feature_names is None:
                return X

            # If X has columns, ensure they match expected features
            if hasattr(X, "columns"):
                # Reorder columns to match training order
                try:
                    return X[self.feature_names]
                except KeyError as e:
                    missing_features = set(self.feature_names) - set(X.columns)
                    raise ValueError(f"Missing features: {missing_features}") from e

            # For arrays, just return as-is (assume correct order)
            return X

        def get_params(self, deep=True):
            """Get model parameters."""
            return self.model.get_params(deep=deep)

        def set_params(self, **params):
            """Set model parameters."""
            return self.model.set_params(**params)

        @property
        def classes_(self):
            """Get fitted classes."""
            if hasattr(self.model, "classes_"):
                return self.model.classes_
            return None

        @classes_.setter
        def classes_(self, value):
            """Set classes (for compatibility)."""
            if hasattr(self.model, "classes_"):
                self.model.classes_ = value

        def get_feature_importance(self, importance_type="coef"):
            """Get feature importance from coefficients."""
            if not self._is_fitted:
                raise RuntimeError("Model not fitted")

            if importance_type == "coef":
                # Use raw coefficients
                importance = np.abs(self.model.coef_[0])  # Take first class for binary
            else:
                raise ValueError(f"Unknown importance type: {importance_type}")

            # Create importance dict
            feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]
            return dict(zip(feature_names, importance.tolist()))

        def get_model_info(self):
            """Get model information."""
            return {
                "status": "fitted" if self._is_fitted else "not_fitted",
                "n_features": len(self.feature_names) if self.feature_names else None,
                **self.get_params(),
            }

        def __repr__(self):
            """String representation."""
            status = "fitted" if self._is_fitted else "not_fitted"
            n_classes = len(self.classes_) if hasattr(self, "classes_") and self.classes_ is not None else "unknown"
            return f"LogisticRegressionWrapper(status={status}, n_classes={n_classes}, version={self.version})"

    @ModelRegistry.register("sklearn_generic", priority=70)  
    class SklearnGenericWrapper(BaseEstimator):
        """Generic wrapper for any scikit-learn estimator."""

        # Required class attributes  
        capabilities = {
            "supports_feature_importance": True,
            "supports_proba": False,  # Will be updated based on model
            "data_modality": "tabular",
        }
        version = "1.0.0"
        model_type = "sklearn_generic"

        def __init__(self, model=None, feature_names=None, **kwargs):
            """Initialize generic sklearn wrapper.

            Args:
                model: Pre-fitted sklearn model
                feature_names: List of feature names
                **kwargs: If provided without model, raises error

            """
            if model is None and kwargs:
                raise ValueError("SklearnGenericWrapper requires a fitted model when kwargs provided")

            self.model = model
            self.feature_names = list(feature_names) if feature_names else None

            # Update capabilities based on model
            if model:
                self.capabilities["supports_proba"] = hasattr(model, "predict_proba")

            logger.info("SklearnGenericWrapper initialized")

        def predict(self, X):
            """Make predictions."""
            if self.model is None:
                raise RuntimeError("No model loaded")
            return self.model.predict(X)

        def predict_proba(self, X):
            """Get prediction probabilities if supported."""
            if self.model is None:
                raise RuntimeError("No model loaded")
            if not hasattr(self.model, "predict_proba"):
                raise AttributeError("Model does not support predict_proba")
            return self.model.predict_proba(X)

        def get_feature_importance(self, importance_type="auto"):
            """Get feature importance."""
            if self.model is None:
                raise RuntimeError("No model loaded")

            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                coef = self.model.coef_
                if coef.ndim > 1:
                    # Multi-class: take mean absolute coefficients
                    importance = np.mean(np.abs(coef), axis=0)
                else:
                    importance = np.abs(coef)
            else:
                # Fallback: uniform importance
                n_features = len(self.feature_names) if self.feature_names else 1
                importance = np.ones(n_features) / n_features

            feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]
            return dict(zip(feature_names, importance.tolist()))

        def get_model_type(self):
            """Get model type."""
            return self.model_type

        def __repr__(self):
            """String representation."""
            model_name = type(self.model).__name__ if self.model else "None"
            return f"SklearnGenericWrapper(model={model_name}, version={self.version})"

else:
    # Stub classes when sklearn unavailable
    class LogisticRegressionWrapper:
        """Stub class when scikit-learn is unavailable."""

        def __init__(self, *args, **kwargs):
            """Initialize stub - raises ImportError."""
            raise ImportError("scikit-learn not available - install sklearn or fix CI environment")

    class SklearnGenericWrapper:
        """Stub class when scikit-learn is unavailable."""

        def __init__(self, *args, **kwargs):
            """Initialize stub - raises ImportError."""  
            raise ImportError("scikit-learn not available - install sklearn or fix CI environment")