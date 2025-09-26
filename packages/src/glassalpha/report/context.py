"""Context normalization utilities for audit report rendering.

This module provides utilities to normalize audit results for template
rendering, ensuring consistent data structures that prevent Jinja2 errors.
"""

from typing import Any


def normalize_metrics(metrics: Any) -> dict[str, Any]:  # noqa: ANN401
    """Normalize metrics for template compatibility.

    Contract compliance: Handle metrics that may be numbers or dicts.
    Template expects metrics[key] to have accessible attributes/values,
    but sometimes metrics[key] is just a float (e.g. accuracy: 0.868).

    This prevents "TypeError: 'float' object has no attribute 'accuracy'"
    errors in Jinja2 templates.

    Args:
        metrics: Raw metrics from audit results (dict or None)

    Returns:
        Normalized metrics where values are always accessible as objects

    Example:
        Input: {"accuracy": 0.868, "f1": {"value": 0.75, "details": {...}}}
        Output: {"accuracy": {"value": 0.868}, "f1": {"value": 0.75, "details": {...}}}

    """
    if not metrics:
        return {}

    normalized = dict(metrics) if isinstance(metrics, dict) else {}

    # Normalize each metric - if it's a number, wrap it for template access
    for key, value in list(normalized.items()):
        if isinstance(value, (int, float)):
            normalized[key] = {"value": float(value)}

    return normalized


def normalize_audit_context(audit_results: Any) -> dict[str, Any]:  # noqa: ANN401
    """Normalize complete audit results for template rendering.

    Args:
        audit_results: Audit results object with various attributes

    Returns:
        Normalized context dict safe for template rendering

    """
    context = {}

    # Safely extract and normalize metrics
    if hasattr(audit_results, "model_performance"):
        context["model_performance"] = normalize_metrics(audit_results.model_performance)

    # Safely extract other results with defaults
    context.update(
        {
            "fairness_analysis": getattr(audit_results, "fairness_analysis", {}),
            "drift_analysis": getattr(audit_results, "drift_analysis", {}),
            "explanations": getattr(audit_results, "explanations", {}),
            "data_summary": getattr(audit_results, "data_summary", {}),
            "schema_info": getattr(audit_results, "schema_info", {}),
            "model_info": getattr(audit_results, "model_info", {}),
            "selected_components": getattr(audit_results, "selected_components", {}),
            "manifest": getattr(audit_results, "manifest", {}),
            "success": getattr(audit_results, "success", False),
            "error_message": getattr(audit_results, "error_message", None),
        },
    )

    return context


def safe_get_nested(data: Any, *keys: str, default: Any = None) -> Any:  # noqa: ANN401
    """Safely get nested dictionary/object values for templates.

    Args:
        data: Source data (dict, object, or None)
        *keys: Sequence of keys/attributes to traverse
        default: Default value if path doesn't exist

    Returns:
        Value at nested path or default

    Example:
        safe_get_nested(results, "execution_info", "audit_id", default="unknown")

    """
    current = data

    for key in keys:
        if current is None:
            return default

        current = current.get(key) if isinstance(current, dict) else getattr(current, key, None)

        if current is None:
            return default

    return current if current is not None else default
