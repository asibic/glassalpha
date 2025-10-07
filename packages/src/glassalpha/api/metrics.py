"""Metric section wrappers with dict + attribute access.

Phase 2: ReadonlyMetrics base class with deep immutability.
"""

from __future__ import annotations

import types
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np


def _freeze_nested(obj: Any) -> Any:
    """Recursively freeze mutable containers.

    Converts:
    - dict → MappingProxyType (nested frozen dicts recursively)
    - list → tuple (recursively)
    - np.ndarray → read-only C-contiguous array
    - Other types → pass through

    Args:
        obj: Object to freeze

    Returns:
        Immutable version of object

    """
    if isinstance(obj, dict):
        # Recursively freeze nested dicts, then wrap in MappingProxyType
        frozen_dict = {k: _freeze_nested(v) for k, v in obj.items()}
        return types.MappingProxyType(frozen_dict)
    if isinstance(obj, list):
        return tuple(_freeze_nested(x) for x in obj)
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        arr.setflags(write=False)
        return arr
    return obj


class ReadonlyMetrics:
    """Base class for immutable metric sections with Mapping + attribute access.

    Provides both dict-style and attribute-style access:
    - Dict-style: result.performance["accuracy"] (raises KeyError if missing)
    - Attribute-style: result.performance.accuracy (raises GlassAlphaError if missing)

    All nested data is recursively frozen to prevent mutation.
    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        """Initialize with frozen data.

        Args:
            data: Metric dictionary to wrap

        """
        # Freeze nested dicts recursively (returns MappingProxyType if dict)
        if isinstance(data, types.MappingProxyType):
            # Already frozen
            frozen = data
        else:
            frozen = _freeze_nested(dict(data))
        object.__setattr__(self, "_data", frozen)

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        """Dict-style access: result.performance['accuracy'].

        Raises:
            KeyError: If metric not in result

        """
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over metric names."""
        return iter(self._data)

    def __len__(self) -> int:
        """Number of metrics."""
        return len(self._data)

    def keys(self):
        """View of metric names."""
        return self._data.keys()

    def values(self):
        """View of metric values."""
        return self._data.values()

    def items(self):
        """View of (name, value) pairs."""
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get metric with default.

        Args:
            key: Metric name
            default: Default value if not found

        Returns:
            Metric value or default

        """
        return self._data.get(key, default)

    # Attribute access
    def __getattr__(self, name: str) -> Any:
        """Attribute-style access: result.performance.accuracy.

        Raises GlassAlphaError (not AttributeError) for unknown metrics.
        This provides better error messages with docs links.

        Phase 3 will integrate with metric registry for helpful errors.
        """
        # Phase 3: Will use metric registry for better errors
        # For now, just raise AttributeError with helpful message
        try:
            return self._data[name]
        except KeyError:
            msg = (
                f"Metric '{name}' not available in this result. "
                f"Check {self.__class__.__name__.lower().replace('metrics', '')}.keys() "
                "for available metrics."
            )
            raise AttributeError(msg) from None

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({dict(self._data)})"

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent attribute mutation."""
        msg = f"{self.__class__.__name__} is immutable"
        raise AttributeError(msg)


class PerformanceMetrics(ReadonlyMetrics):
    """Performance metrics with optional plotting.

    Metrics requiring y_proba:
    - roc_auc, pr_auc, brier_score, log_loss

    Accessing these without probabilities will raise a helpful error
    in Phase 5 (when GlassAlphaError is implemented).
    """

    # Metrics requiring y_proba
    _PROBA_REQUIRED = {"roc_auc", "pr_auc", "brier_score", "log_loss"}

    def plot_confusion_matrix(self, ax=None, **kwargs):
        """Plot confusion matrix (requires matplotlib).

        Phase 3: Will implement with matplotlib integration.
        """
        msg = "Plotting will be implemented in Phase 3"
        raise NotImplementedError(msg)

    def plot_roc_curve(self, ax=None, **kwargs):
        """Plot ROC curve (requires y_proba).

        Phase 3: Will implement with matplotlib integration.
        """
        msg = "Plotting will be implemented in Phase 3"
        raise NotImplementedError(msg)


class FairnessMetrics(ReadonlyMetrics):
    """Fairness metrics with group-level details."""

    def plot_group_metrics(self, metric: str = "tpr", ax=None, **kwargs):
        """Plot metric by protected group.

        Phase 3: Will implement with matplotlib integration.
        """
        msg = "Plotting will be implemented in Phase 3"
        raise NotImplementedError(msg)


class CalibrationMetrics(ReadonlyMetrics):
    """Calibration metrics."""

    def plot(self, ax=None, **kwargs):
        """Plot calibration curve.

        Phase 3: Will implement with matplotlib integration.
        """
        msg = "Plotting will be implemented in Phase 3"
        raise NotImplementedError(msg)


class StabilityMetrics(ReadonlyMetrics):
    """Stability test results."""


class ExplanationSummary(ReadonlyMetrics):
    """SHAP explanation summary."""


class RecourseSummary(ReadonlyMetrics):
    """Recourse generation summary."""
