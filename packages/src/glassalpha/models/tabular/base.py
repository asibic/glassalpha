# SPDX-License-Identifier: Apache-2.0
"""Base wrapper functionality for tabular model wrappers.

This module provides common functionality that prevents regression of
the chronic save/load and column handling issues.
"""

import logging
from pathlib import Path
from typing import Any

from glassalpha.constants import NO_MODEL_MSG

# Conditional imports
try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseTabularWrapper:
    """Base class for tabular model wrappers with robust save/load and column handling."""

    def __init__(self) -> None:
        """Initialize base wrapper."""
        self.model = None
        self.feature_names_ = None
        self.n_classes = None
        self._is_fitted = False

    def _ensure_fitted(self) -> None:
        """Ensure model is fitted, raise if not.

        Raises:
            ValueError: If model is not loaded/fitted

        """
        if self.model is None:
            raise ValueError(NO_MODEL_MSG)

    def _prepare_x(self, X: Any) -> Any:  # noqa: ANN401, N803
        """Robust DataFrame column handling - prevents feature name mismatch errors.

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Prepared features (handles column renames gracefully)

        """
        if not PANDAS_AVAILABLE:
            return X

        if not isinstance(X, pd.DataFrame):
            return X

        fitted_names = getattr(self, "feature_names_", None)
        if fitted_names is None:
            return X

        # If same width but different names, assume same order
        if len(fitted_names) == X.shape[1] and list(X.columns) != list(fitted_names):
            logger.debug("DataFrame columns renamed but same width - using array conversion")
            return X.to_numpy()

        # Try to reindex to fitted column order
        if set(fitted_names).issubset(set(X.columns)):
            return X.reindex(columns=fitted_names, fill_value=0)

        return X

    def save(self, path: Path) -> None:
        """Save model state to file.

        Args:
            path: File path to save to

        Raises:
            ValueError: If no model to save
            ImportError: If joblib not available

        """
        if not JOBLIB_AVAILABLE:
            msg = "joblib is required for saving models"
            raise ImportError(msg)

        self._ensure_fitted()

        save_dict = {
            "model": self.model,
            "feature_names_": self.feature_names_,
            "n_classes": self.n_classes,
            "_is_fitted": self._is_fitted,
        }

        joblib.dump(save_dict, path)
        logger.info("Saved model state to %s", path)

    def load(self, path: Path) -> "BaseTabularWrapper":
        """Load model state from file.

        Args:
            path: File path to load from

        Returns:
            Self (for method chaining)

        Raises:
            ImportError: If joblib not available

        """
        if not JOBLIB_AVAILABLE:
            msg = "joblib is required for loading models"
            raise ImportError(msg)

        save_dict = joblib.load(path)

        self.model = save_dict["model"]
        self.feature_names_ = save_dict.get("feature_names_")
        self.n_classes = save_dict.get("n_classes")
        self._is_fitted = save_dict.get("_is_fitted", True)

        logger.info("Loaded model state from %s", path)
        return self
