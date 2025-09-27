"""Fairness metrics runner for GlassAlpha.

This module provides utilities for running fairness metrics with proper
handling of numpy arrays and pandas DataFrames.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _as_series(v: Any, name: str) -> pd.Series:
    """Convert input to pandas Series for consistent handling."""
    if isinstance(v, pd.Series):
        return v
    return pd.Series(v, name=name)


def run_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray | pd.DataFrame,
    metrics: list[Any],
) -> dict[str, Any]:
    """Run fairness metrics with proper input handling.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        sensitive: Sensitive features (numpy array or DataFrame)
        metrics: List of metric classes to run

    Returns:
        Dictionary with results from all metrics

    """
    # Convert inputs to consistent format
    y_true = _as_series(y_true, "y_true")
    y_pred = _as_series(y_pred, "y_pred")

    # Handle sensitive features
    if isinstance(sensitive, pd.DataFrame):
        sensitive_df = sensitive
    else:
        # Single sensitive feature as numpy array
        sensitive_df = pd.DataFrame({"sensitive": np.asarray(sensitive)})

    # Validate inputs
    if len(y_true) != len(y_pred):
        msg = f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
        raise ValueError(msg)

    if len(y_true) != len(sensitive_df):
        msg = f"Length mismatch: y_true ({len(y_true)}) vs sensitive ({len(sensitive_df)})"
        raise ValueError(msg)

    results = {}
    for metric_class in metrics:
        try:
            metric = metric_class()
            metric_name = metric_class.__name__.lower().replace("metric", "")

            # Run metric with appropriate sensitive features format
            if isinstance(sensitive, pd.DataFrame):
                # Multiple sensitive features
                metric_result = metric.compute(y_true.values, y_pred.values, sensitive)
            else:
                # Single sensitive feature
                metric_result = metric.compute(y_true.values, y_pred.values, np.asarray(sensitive))

            results[metric_name] = metric_result
            logger.debug(f"Computed {metric_name}: {list(metric_result.keys())}")

        except Exception as e:
            logger.warning(f"Failed to compute {metric_class.__name__}: {e}")
            results[metric_class.__name__.lower().replace("metric", "")] = {"error": str(e)}

    return results
