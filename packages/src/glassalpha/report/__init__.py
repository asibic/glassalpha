"""Report generation package for GlassAlpha.

Provides functionality for generating audit reports including:
- HTML report rendering
- PDF report generation (optional dependency)
- Plotting and visualization (optional dependency)
"""

from __future__ import annotations

from .renderer import AuditReportRenderer, render_audit_report

try:
    from .plots import (
        AuditPlotter,
        create_fairness_plots,
        create_performance_plots,
        create_shap_plots,
        plot_drift_analysis,
    )

    _PLOTS_AVAILABLE = True
except ImportError:
    _PLOTS_AVAILABLE = False

try:
    from .renderers import (
        AuditPDFRenderer,
        render_audit_pdf,
    )
    from .renderers.pdf import PDFConfig

    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

# Public API
__all__ = [
    "AuditReportRenderer",
    "render_audit_report",
]

if _PLOTS_AVAILABLE:
    __all__ += [
        "AuditPlotter",
        "create_fairness_plots",
        "create_performance_plots",
        "create_shap_plots",
        "plot_drift_analysis",
    ]

if _PDF_AVAILABLE:
    __all__ += [
        "AuditPDFRenderer",
        "PDFConfig",
        "render_audit_pdf",
    ]
