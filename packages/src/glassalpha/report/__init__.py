"""Report generation modules for GlassAlpha audit outputs.

This package provides comprehensive report generation capabilities including
plotting, templates, and PDF rendering for professional audit documentation.
"""

from .plots import (
    AuditPlotter,
    create_fairness_plots,
    create_performance_plots,
    create_shap_plots,
    plot_drift_analysis,
)
from .renderer import (
    AuditReportRenderer,
    render_audit_report,
)

__all__ = [
    "AuditPlotter",
    "create_shap_plots",
    "create_performance_plots",
    "create_fairness_plots",
    "plot_drift_analysis",
    "AuditReportRenderer",
    "render_audit_report",
]
