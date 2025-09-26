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
from .renderers import (
    AuditPDFRenderer,
    render_audit_pdf,
)
from .renderers.pdf import PDFConfig

__all__ = [
    "AuditPDFRenderer",
    "AuditPlotter",
    "AuditReportRenderer",
    "PDFConfig",
    "create_fairness_plots",
    "create_performance_plots",
    "create_shap_plots",
    "plot_drift_analysis",
    "render_audit_pdf",
    "render_audit_report",
]

# Optional packaging validation - enable in CI with GLASSALPHA_ASSERT_PACKAGING=1
import os

if os.getenv("GLASSALPHA_ASSERT_PACKAGING") == "1":
    from importlib.resources import files

    from glassalpha.constants import STANDARD_AUDIT_TEMPLATE

    template_path = files("glassalpha.report.templates").joinpath(STANDARD_AUDIT_TEMPLATE)
    assert template_path.is_file(), f"Template not packaged correctly: {template_path}"  # noqa: S101
