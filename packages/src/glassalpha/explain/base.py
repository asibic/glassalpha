"""Base classes and interfaces for explainers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class ExplainerBase:
    """Base class for all explainers with expected interface contract.

    This defines the interface that tests expect all explainers to implement.
    Tests specifically check for 'priority' as a class attribute and 'explainer'
    instance attribute that starts as None before fit() is called.
    """

    # Tests expect these as class attributes
    priority: int = 0
    version: str = "1.0.0"

    def fit(self, wrapper: Any, background_X, feature_names: Sequence[str] | None = None):
        """Fit the explainer with a model wrapper and background data.

        Args:
            wrapper: Model wrapper with predict/predict_proba methods
            background_X: Background data for explainer baseline
            feature_names: Optional feature names for interpretation

        Returns:
            self: Returns self for chaining

        """
        raise NotImplementedError

    def explain(self, X, **kwargs):
        """Generate explanations for input data.

        Args:
            X: Input data to explain
            **kwargs: Additional parameters for explanation generation

        Returns:
            Dictionary containing explanation results

        """
        raise NotImplementedError
