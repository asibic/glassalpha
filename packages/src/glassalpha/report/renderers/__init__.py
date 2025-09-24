"""Report renderers for GlassAlpha audit outputs.

This package provides various output format renderers including
PDF generation from HTML templates.
"""

from .pdf import AuditPDFRenderer, render_audit_pdf

__all__ = [
    "AuditPDFRenderer",
    "render_audit_pdf",
]
