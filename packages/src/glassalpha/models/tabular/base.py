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


def _ensure_fitted(obj: Any, attr: str = "_fitted_", message: str = NO_MODEL_MSG) -> None:  # noqa: ANN401
    """Ensure object is fitted, raise if not.

    Args:
        obj: Object to check
        attr: Attribute name to check (not used for tabular wrappers)
        message: Error message to raise

    Raises:
        ValueError: If object is not fitted

    """
    # For tabular wrappers, check if model attribute exists
    if (hasattr(obj, "model") and obj.model is None) or (hasattr(obj, "_is_fitted") and not obj._is_fitted):
        raise ValueError(message)


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
        """Robust DataFrame column handling - uses centralized feature alignment.

        Contract compliance: Uses shared align_features function for consistent
        feature drift handling across all wrappers.

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Prepared features with proper alignment

        """
        return self._align_features(X)

    def _align_features(self, X: Any) -> Any:  # noqa: ANN401, N803
        """Shared feature alignment helper - contract compliance.

        Uses centralized align_features function for consistency across all wrappers.

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Aligned features with proper column handling

        Raises:
            ValueError: If features cannot be aligned and strict validation is needed

        """
        from glassalpha.models._features import align_features  # noqa: PLC0415

        feature_names = getattr(self, "feature_names_", None)

        # For strict validation, check for mismatched features before alignment
        if PANDAS_AVAILABLE and hasattr(X, "columns") and feature_names:
            expected = list(feature_names)
            actual = list(X.columns)

            # If it's not a simple rename (same width), check for strict mismatch
            if len(actual) != len(expected) or set(actual) != set(expected):
                missing = [c for c in expected if c not in actual]
                extra = [c for c in actual if c not in expected]

                # If there are both missing and extra features, this might be a true mismatch
                # rather than just a column order issue
                if missing and extra and len(missing) == len(expected):
                    # Complete mismatch - no overlap in features
                    raise ValueError(
                        f"Feature mismatch: expected {expected}; missing {missing}; extra {extra}",
                    )

        return align_features(X, feature_names)

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

        # Create parent directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(save_dict, path)
        logger.info(f"Saved model state to {path}")

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

        logger.info(f"Loaded model state from {path}")
        return self
