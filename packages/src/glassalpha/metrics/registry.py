"""Metric registry for GlassAlpha.

This module defines the MetricRegistry using the decorator-friendly registry
and provides utilities for metric selection and computation.
"""

import logging
from typing import Any

from ..core.decor_registry import DecoratorFriendlyRegistry

logger = logging.getLogger(__name__)

# Create the metric registry using the decorator-friendly registry
MetricRegistry = DecoratorFriendlyRegistry(group="glassalpha.metrics")
MetricRegistry.discover()  # Safe discovery without heavy imports


def get_metrics_by_type(metric_type: str) -> list[str]:
    """Get all registered metrics of a specific type.

    Args:
        metric_type: Type of metrics to retrieve (performance, fairness, drift)

    Returns:
        List of metric names of the specified type

    """
    matching_metrics = []

    all_metrics = MetricRegistry.get_all()
    for name, metric_cls in all_metrics.items():
        try:
            if hasattr(metric_cls, "metric_type") and metric_cls.metric_type == metric_type:
                matching_metrics.append(name)
        except Exception as e:
            logger.warning(f"Error checking metric type for {name}: {e}")

    return matching_metrics


def get_required_metrics_for_profile(profile_name: str) -> dict[str, list[str]]:
    """Get required metrics for a specific audit profile.

    Args:
        profile_name: Name of the audit profile

    Returns:
        Dictionary mapping metric types to lists of required metric names

    """
    # This will be enhanced when audit profiles are fully implemented
    if profile_name == "tabular_compliance":
        return {
            "performance": ["accuracy", "precision", "recall", "f1", "auc_roc"],
            "fairness": ["demographic_parity", "equal_opportunity"],
            "drift": ["psi", "ks_test"],
        }
    # Default minimal set
    return {"performance": ["accuracy"], "fairness": [], "drift": []}


def select_appropriate_metrics(
    model_type: str,
    task_type: str = "classification",
    has_sensitive_features: bool = False,
    audit_profile: str = "standard",
) -> dict[str, list[str]]:
    """Select appropriate metrics based on model and data characteristics.

    Args:
        model_type: Type of model (xgboost, lightgbm, logistic_regression, etc.)
        task_type: Type of ML task (classification, regression)
        has_sensitive_features: Whether sensitive features are available for fairness metrics
        audit_profile: Name of audit profile to use

    Returns:
        Dictionary mapping metric types to lists of selected metric names

    """
    selected = {"performance": [], "fairness": [], "drift": []}

    # Get profile requirements
    profile_requirements = get_required_metrics_for_profile(audit_profile)

    # Performance metrics (always needed)
    performance_metrics = get_metrics_by_type("performance")
    if task_type == "classification":
        # Prefer classification metrics
        preferred = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        selected["performance"] = [m for m in preferred if m in performance_metrics]
    else:
        # Regression metrics (when implemented)
        preferred = ["mse", "rmse", "mae", "r2"]
        selected["performance"] = [m for m in preferred if m in performance_metrics]

    # Fairness metrics (only if sensitive features available)
    if has_sensitive_features:
        fairness_metrics = get_metrics_by_type("fairness")
        preferred = profile_requirements.get("fairness", ["demographic_parity"])
        selected["fairness"] = [m for m in preferred if m in fairness_metrics]

    # Drift metrics (useful for monitoring)
    drift_metrics = get_metrics_by_type("drift")
    preferred = profile_requirements.get("drift", ["psi"])
    selected["drift"] = [m for m in preferred if m in drift_metrics]

    # Remove empty categories
    selected = {k: v for k, v in selected.items() if v}

    logger.info(f"Selected metrics for {model_type} {task_type}: {selected}")
    return selected


def compute_all_metrics(
    metric_names: list[str],
    y_true: Any,
    y_pred: Any,
    sensitive_features: Any = None,
) -> dict[str, dict[str, float]]:
    """Compute multiple metrics at once.

    Args:
        metric_names: List of metric names to compute
        y_true: Ground truth values
        y_pred: Predicted values
        sensitive_features: Optional sensitive features for fairness metrics

    Returns:
        Dictionary mapping metric names to their computed values

    """
    results = {}

    for metric_name in metric_names:
        try:
            metric_cls = MetricRegistry.get(metric_name)
            metric = metric_cls()

            # Check if metric requires sensitive features
            if metric.requires_sensitive_features() and sensitive_features is None:
                logger.warning(f"Metric {metric_name} requires sensitive features but none provided")
                continue

            computed = metric.compute(y_true, y_pred, sensitive_features)
            results[metric_name] = computed

        except Exception as e:
            logger.error(f"Error computing metric {metric_name}: {e}")
            results[metric_name] = {"error": str(e)}

    return results


def get_metric_summary(results: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Create a summary of computed metrics.

    Args:
        results: Dictionary of computed metric results

    Returns:
        Summary dictionary with aggregated information

    """
    summary = {
        "total_metrics": len(results),
        "successful_metrics": len([r for r in results.values() if "error" not in r]),
        "failed_metrics": len([r for r in results.values() if "error" in r]),
        "metric_types": {},
        "key_metrics": {},
    }

    # Group by metric type and extract key values
    for metric_name, metric_results in results.items():
        if "error" not in metric_results:
            try:
                metric_cls = MetricRegistry.get(metric_name)
                metric_type = getattr(metric_cls, "metric_type", "unknown")

                if metric_type not in summary["metric_types"]:
                    summary["metric_types"][metric_type] = []
                summary["metric_types"][metric_type].append(metric_name)

                # Extract primary metric value (usually the first or most important)
                if len(metric_results) > 0:
                    primary_key = list(metric_results.keys())[0]
                    summary["key_metrics"][metric_name] = metric_results[primary_key]

            except Exception as e:
                logger.warning(f"Error summarizing metric {metric_name}: {e}")

    return summary
