"""Template rendering engine for audit reports.

This module provides the core template rendering functionality for generating
professional audit reports from audit results. It handles data preparation,
template processing, and asset management.
"""

import base64
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import jinja2
from matplotlib.figure import Figure

# Friend's spec: Use importlib.resources for template loading from installed package
try:
    from importlib.resources import files

    IMPORTLIB_RESOURCES_AVAILABLE = True
except ImportError:
    # Fallback for older Python versions
    IMPORTLIB_RESOURCES_AVAILABLE = False
    files = None

from glassalpha.pipeline.audit import AuditResults

from .plots import AuditPlotter, create_fairness_plots, create_performance_plots, create_shap_plots

logger = logging.getLogger(__name__)


class AuditReportRenderer:
    """Professional template renderer for audit reports."""

    def __init__(self, template_dir: Path | None = None) -> None:
        """Initialize the audit report renderer.

        Args:
            template_dir: Directory containing Jinja2 templates

        """
        # Friend's spec: Use package resources for template loading from installed package
        if template_dir is None:
            # Try to use PackageLoader which is designed for package resources
            try:
                loader = jinja2.PackageLoader("glassalpha.report", "templates")
                self.template_dir = Path("glassalpha.report.templates")  # Virtual path for logging
                logger.debug("Using PackageLoader for template resources")
            except (ImportError, AttributeError, ValueError):
                # Fall back to filesystem loader
                self.template_dir = Path(__file__).parent / "templates"
                loader = jinja2.FileSystemLoader(str(self.template_dir))
                logger.debug("Falling back to FileSystemLoader: %s", self.template_dir)
        else:
            # Use provided template_dir with filesystem loader
            self.template_dir = template_dir
            loader = jinja2.FileSystemLoader(str(self.template_dir))
            logger.debug("Using provided template directory: %s", template_dir)

        # Configure Jinja2 environment
        self.env = jinja2.Environment(
            loader=loader,
            autoescape=jinja2.select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add custom filters
        self.env.filters["format_number"] = self._format_number
        self.env.filters["format_percentage"] = self._format_percentage
        self.env.filters["format_datetime"] = self._format_datetime
        self.env.filters["unique"] = lambda x: list(set(x)) if x else []
        self.env.filters["safe_get"] = lambda d, k, default=None: d.get(k, default) if isinstance(d, dict) else default

        # Initialize plotter
        self.plotter = AuditPlotter(style="professional")

        logger.info("Initialized AuditReportRenderer with template directory: %s", self.template_dir)

    def render_audit_report(
        self,
        audit_results: AuditResults,
        template_name: str = "standard_audit.html",
        output_path: Path | None = None,
        embed_plots: bool = True,
        **template_vars,
    ) -> str:
        """Render complete audit report from audit results.

        Args:
            audit_results: Complete audit results from pipeline
            template_name: Name of the template file to use
            output_path: Optional path to save rendered HTML
            embed_plots: Whether to embed plots as base64 images
            **template_vars: Additional variables to pass to template

        Returns:
            Rendered HTML content as string

        """
        logger.info("Rendering audit report using template: %s", template_name)

        # Prepare template context
        context = self._prepare_template_context(audit_results, embed_plots)
        context.update(template_vars)

        # Load and render template
        template = self.env.get_template(template_name)
        rendered_html = template.render(**context)

        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rendered_html)

            logger.info("Saved rendered report to: %s", output_path)

        return rendered_html

    def _prepare_template_context(self, audit_results: AuditResults, embed_plots: bool = True) -> dict[str, Any]:
        """Prepare comprehensive template context from audit results.

        Args:
            audit_results: Audit results to process
            embed_plots: Whether to create and embed plots

        Returns:
            Dictionary with template context variables

        """
        logger.debug("Preparing template context from audit results")

        # Basic audit information
        context = {
            "report_title": "Machine Learning Model Audit Report",
            "generation_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "version": "1.0.0",
            "contact_email": "audit-support@example.com",
            "success": audit_results.success,
            "error_message": audit_results.error_message,
        }

        # Extract audit metadata
        if audit_results.execution_info:
            context.update(
                {
                    "audit_id": audit_results.execution_info.get("audit_id"),
                    "audit_profile": audit_results.execution_info.get("audit_profile"),
                    "strict_mode": audit_results.execution_info.get("strict_mode", False),
                },
            )

        # Add core audit results
        context.update(
            {
                "model_performance": audit_results.model_performance,
                "fairness_analysis": audit_results.fairness_analysis,
                "drift_analysis": audit_results.drift_analysis,
                "explanations": audit_results.explanations,
                "data_summary": audit_results.data_summary,
                "schema_info": audit_results.schema_info,
                "model_info": audit_results.model_info,
                "selected_components": audit_results.selected_components,
                "manifest": audit_results.manifest,
            },
        )

        # Add metric descriptions for better understanding
        context["metric_descriptions"] = self._get_metric_descriptions()
        context["plot_descriptions"] = self._get_plot_descriptions()
        context["shap_descriptions"] = self._get_shap_descriptions()

        # Generate and embed plots if requested
        if embed_plots:
            plots = self._generate_plots(audit_results)
            context.update(
                {
                    "shap_plots": plots.get("shap", {}),
                    "performance_plots": plots.get("performance", {}),
                    "fairness_plots": plots.get("fairness", {}),
                },
            )

        logger.debug("Template context prepared with %s variables", len(context))
        return context

    def _generate_plots(self, audit_results: AuditResults) -> dict[str, dict[str, str]]:
        """Generate plots and return as base64-encoded data URLs.

        Args:
            audit_results: Audit results containing plot data

        Returns:
            Dictionary with plot categories and base64 data URLs

        """
        logger.debug("Generating plots for template embedding")
        plots = {"shap": {}, "performance": {}, "fairness": {}}

        try:
            # Generate SHAP plots
            if audit_results.explanations:
                feature_names = list(audit_results.schema_info.get("features", [])) if audit_results.schema_info else []
                shap_figures = create_shap_plots(audit_results.explanations, feature_names)

                for plot_name, figure in shap_figures.items():
                    plots["shap"][plot_name] = self._figure_to_base64(figure)

                logger.debug("Generated %s SHAP plots", len(shap_figures))

            # Generate performance plots
            if audit_results.model_performance:
                perf_figures = create_performance_plots(audit_results.model_performance)

                for plot_name, figure in perf_figures.items():
                    plots["performance"][plot_name] = self._figure_to_base64(figure)

                logger.debug("Generated %s performance plots", len(perf_figures))

            # Generate fairness plots
            if audit_results.fairness_analysis:
                fairness_figures = create_fairness_plots(audit_results.fairness_analysis)

                for plot_name, figure in fairness_figures.items():
                    plots["fairness"][plot_name] = self._figure_to_base64(figure)

                logger.debug("Generated %s fairness plots", len(fairness_figures))

        except Exception as e:
            logger.warning("Failed to generate some plots: %s", e)

        return plots

    def _figure_to_base64(self, figure: Figure) -> str:
        """Convert matplotlib figure to base64 data URL.

        Args:
            figure: Matplotlib figure object

        Returns:
            Base64 data URL string for embedding in HTML

        """
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        buffer.seek(0)

        # Encode as base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()

        # Create data URL
        return f"data:image/png;base64,{image_base64}"

    def _get_metric_descriptions(self) -> dict[str, str]:
        """Get descriptions for performance metrics."""
        return {
            "accuracy": "Overall classification accuracy across all classes",
            "precision": "Proportion of positive predictions that were correct",
            "recall": "Proportion of actual positives that were correctly identified",
            "f1": "Harmonic mean of precision and recall",
            "auc_roc": "Area under the ROC curve - discrimination ability",
            "classification_report": "Comprehensive performance summary",
            "demographic_parity": "Equal positive prediction rates across groups",
            "equal_opportunity": "Equal true positive rates across groups",
            "equalized_odds": "Equal TPR and FPR across groups",
            "predictive_parity": "Equal precision across groups",
        }

    def _get_plot_descriptions(self) -> dict[str, str]:
        """Get descriptions for different plot types."""
        return {
            "summary": "Overall performance metrics dashboard",
            "confusion_matrix": "True vs predicted labels breakdown",
            "roc_curve": "True positive rate vs false positive rate",
            "feature_importance": "Most influential features for predictions",
        }

    def _get_shap_descriptions(self) -> dict[str, str]:
        """Get descriptions for SHAP plot types."""
        return {
            "global_importance": "Average impact of each feature across all predictions",
            "summary": "Distribution of feature impacts across all samples",
            "waterfall": "Step-by-step explanation of individual prediction",
        }

    def _format_number(self, value: Any, decimals: int = 3) -> str:
        """Format numeric values for display."""
        if value is None:
            return "N/A"

        if isinstance(value, (int, float)):
            if decimals == 0:
                return f"{value:,.0f}"
            return f"{value:.{decimals}f}"

        return str(value)

    def _format_percentage(self, value: Any, decimals: int = 1) -> str:
        """Format values as percentages."""
        if value is None:
            return "N/A"

        if isinstance(value, (int, float)):
            return f"{value * 100:.{decimals}f}%"

        return str(value)

    def _format_datetime(self, value: Any, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime values."""
        if value is None:
            return "N/A"

        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                return dt.strftime(format_string)
            except ValueError:
                return value

        if hasattr(value, "strftime"):
            return value.strftime(format_string)

        return str(value)


def render_audit_report(
    audit_results: AuditResults,
    output_path: Path | None = None,
    template_name: str = "standard_audit.html",
    **template_vars,
) -> str:
    """Convenience function to render audit report.

    Args:
        audit_results: Complete audit results from pipeline
        output_path: Optional path to save HTML file
        template_name: Template file to use
        **template_vars: Additional template variables

    Returns:
        Rendered HTML content

    """
    renderer = AuditReportRenderer()
    return renderer.render_audit_report(
        audit_results=audit_results,
        output_path=output_path,
        template_name=template_name,
        **template_vars,
    )
