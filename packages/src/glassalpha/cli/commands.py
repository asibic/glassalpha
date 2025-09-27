"""CLI commands for GlassAlpha.

This module implements the main commands available in the CLI,
including the core audit command with strict mode support.

ARCHITECTURE NOTE: Exception handling in CLI commands intentionally
suppresses Python tracebacks (using 'from None') to provide clean
user-facing error messages. This is the correct pattern for CLI tools.
Do not "fix" this by adding 'from e' - users don't need stack traces.
"""

import logging
import sys
from pathlib import Path

import typer

logger = logging.getLogger(__name__)


def _ensure_components_loaded() -> None:
    """Ensure all required components are imported and registered."""
    try:
        # Import model modules to trigger registration
        from ..explain.shap import kernel, tree  # noqa: F401
        from ..metrics.fairness import bias_detection  # noqa: F401
        from ..metrics.performance import classification  # noqa: F401
        from ..models.tabular import lightgbm, sklearn, xgboost  # noqa: F401

        logger.debug("All component modules imported and registered")
    except ImportError as e:
        logger.warning(f"Some components could not be imported: {e}")


def _run_audit_pipeline(config, output_path: Path) -> None:
    """Execute the complete audit pipeline and generate PDF report.

    Args:
        config: Validated audit configuration
        output_path: Path where PDF report should be saved

    """
    import time
    from datetime import datetime

    # Import here to avoid circular imports and startup overhead
    from ..pipeline.audit import run_audit_pipeline
    from ..report import PDFConfig, render_audit_pdf

    start_time = time.time()

    try:
        # Step 1: Run audit pipeline
        typer.echo("  Loading data and initializing components...")

        # Run the actual audit pipeline
        audit_results = run_audit_pipeline(config)

        if not audit_results.success:
            typer.secho(f"❌ Audit pipeline failed: {audit_results.error_message}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)

        pipeline_time = time.time() - start_time
        typer.secho(f"✓ Audit pipeline completed in {pipeline_time:.2f}s", fg=typer.colors.GREEN)

        # Show audit summary
        _display_audit_summary(audit_results)

        # Step 2: Generate PDF report
        typer.echo(f"\nGenerating PDF report: {output_path}")

        # Create PDF configuration
        pdf_config = PDFConfig(
            page_size="A4",
            title="ML Model Audit Report",
            author="GlassAlpha",
            subject="Machine Learning Model Compliance Assessment",
            optimize_size=True,
        )

        # Generate PDF
        pdf_start = time.time()
        pdf_path = render_audit_pdf(
            audit_results=audit_results,
            output_path=output_path,
            config=pdf_config,
            report_title=f"ML Model Audit Report - {datetime.now().strftime('%Y-%m-%d')}",
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        pdf_time = time.time() - pdf_start
        file_size = pdf_path.stat().st_size

        # Generate manifest sidecar if provenance manifest is available
        manifest_path = None
        if hasattr(audit_results, "execution_info") and "provenance_manifest" in audit_results.execution_info:
            from ..provenance import write_manifest_sidecar  # noqa: PLC0415

            try:
                manifest_path = write_manifest_sidecar(
                    audit_results.execution_info["provenance_manifest"],
                    output_path,
                )
                typer.echo(f"📋 Manifest: {manifest_path}")
            except Exception as e:
                logger.warning("Failed to write manifest sidecar: %s", e)

        # Success message
        total_time = time.time() - start_time
        typer.echo("\n🎉 Audit Report Generated Successfully!")
        typer.echo(f"{'=' * 50}")
        typer.secho(f"📁 Output: {pdf_path}", fg=typer.colors.GREEN)
        typer.echo(f"📊 Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
        typer.echo(f"⏱️  Total time: {total_time:.2f}s")
        typer.echo(f"   • Pipeline: {pipeline_time:.2f}s")
        typer.echo(f"   • PDF generation: {pdf_time:.2f}s")

        # Regulatory compliance message
        if config.strict_mode:
            typer.secho("\n🛡️  Strict mode: Report meets regulatory compliance requirements", fg=typer.colors.YELLOW)

        typer.echo("\nThe audit report is ready for review and regulatory submission.")

    except Exception as e:
        typer.secho(f"\n❌ Audit failed: {e!s}", fg=typer.colors.RED, err=True)

        # Show more details in verbose mode
        if "--verbose" in sys.argv or "-v" in sys.argv:
            logger.exception("Detailed audit failure information:")

        raise typer.Exit(1)


def _display_audit_summary(audit_results) -> None:
    """Display a summary of audit results."""
    typer.echo("\n📊 Audit Summary:")

    # Model performance
    if audit_results.model_performance:
        perf_count = len(
            [m for m in audit_results.model_performance.values() if isinstance(m, dict) and "error" not in m],
        )
        typer.echo(f"  ✅ Performance metrics: {perf_count} computed")

        # Show key metrics
        for name, result in audit_results.model_performance.items():
            if isinstance(result, dict) and "accuracy" in result:
                accuracy = result["accuracy"]
                status = "✅" if accuracy > 0.8 else "⚠️" if accuracy > 0.6 else "❌"
                typer.echo(f"     {status} {name}: {accuracy:.1%}")
                break

    # Fairness analysis
    if audit_results.fairness_analysis:
        bias_detected = []
        total_metrics = 0
        failed_metrics = 0

        for attr, metrics in audit_results.fairness_analysis.items():
            for metric, result in metrics.items():
                total_metrics += 1
                if isinstance(result, dict):
                    if "error" in result:
                        failed_metrics += 1
                    elif result.get("is_fair") is False:
                        bias_detected.append(f"{attr}.{metric}")

        computed_metrics = total_metrics - failed_metrics
        typer.echo(f"  ⚖️  Fairness metrics: {computed_metrics}/{total_metrics} computed")

        if bias_detected:
            typer.secho(f"     ⚠️  Bias detected in: {', '.join(bias_detected[:2])}", fg=typer.colors.YELLOW)
        elif computed_metrics > 0:
            typer.secho("     ✅ No bias detected", fg=typer.colors.GREEN)

    # SHAP explanations
    if audit_results.explanations:
        has_importance = "global_importance" in audit_results.explanations

        if has_importance:
            typer.echo("  🔍 Explanations: ✅ Global feature importance")

            # Show top feature
            importance = audit_results.explanations.get("global_importance", {})
            if importance:
                top_feature = max(importance.items(), key=lambda x: abs(x[1]))
                typer.echo(f"     Most important: {top_feature[0]} ({top_feature[1]:+.3f})")
        else:
            typer.echo("  🔍 Explanations: ❌ Not available")

    # Data summary
    if audit_results.data_summary and "shape" in audit_results.data_summary:
        rows, cols = audit_results.data_summary["shape"]
        typer.echo(f"  📋 Dataset: {rows:,} samples, {cols} features")

    # Selected components
    if audit_results.selected_components:
        typer.echo(f"  🔧 Components: {len(audit_results.selected_components)} selected")

        # Show model type
        for _comp_name, comp_info in audit_results.selected_components.items():
            if comp_info.get("type") == "model":
                typer.echo(f"     Model: {comp_info.get('name', 'unknown')}")
                break


def audit(
    # Typer requires function calls in defaults - this is the documented pattern
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to audit configuration YAML file",
        # Remove exists=True to handle file checking manually for better error messages
        file_okay=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path for output PDF report",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Enable strict mode for regulatory compliance",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        "-p",
        help="Override audit profile",
    ),
    override_config: Path | None = typer.Option(
        None,
        "--override",
        help="Additional config file to override settings",
        file_okay=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate configuration without generating report",
    ),
):
    """Generate a compliance audit PDF report.

    This is the main command for GlassAlpha. It loads a configuration file,
    runs the audit pipeline, and generates a deterministic PDF report.

    Examples:
        # Basic audit
        glassalpha audit --config audit.yaml --output report.pdf

        # Strict mode for regulatory compliance
        glassalpha audit --config audit.yaml --output report.pdf --strict

        # Override specific settings
        glassalpha audit -c base.yaml --override custom.yaml -o report.pdf

    """
    try:
        # Check file existence early with specific error message
        if not config.exists():
            typer.echo(f"File '{config}' does not exist.", err=True)
            raise typer.Exit(1)

        # Check override config if provided
        if override_config and not override_config.exists():
            typer.echo(f"Override file '{override_config}' does not exist.", err=True)
            raise typer.Exit(1)

        # Import here to avoid circular imports
        from ..config import load_config_from_file
        from ..core import list_components

        # Import all component modules to trigger registration
        _ensure_components_loaded()

        typer.echo("GlassAlpha Audit Generation")
        typer.echo(f"{'=' * 40}")

        # Load configuration
        typer.echo(f"Loading configuration from: {config}")
        if override_config:
            typer.echo(f"Applying overrides from: {override_config}")

        audit_config = load_config_from_file(config, override_path=override_config, profile_name=profile, strict=strict)

        # Report configuration
        typer.echo(f"Audit profile: {audit_config.audit_profile}")
        typer.echo(f"Strict mode: {'ENABLED' if audit_config.strict_mode else 'disabled'}")

        if audit_config.strict_mode:
            typer.secho("⚠️  Strict mode enabled - enforcing regulatory compliance", fg=typer.colors.YELLOW)

        # Validate components exist
        available = list_components()
        model_type = audit_config.model.type

        if model_type not in available.get("models", []) and model_type != "passthrough":
            typer.secho(f"Warning: Model type '{model_type}' not found in registry", fg=typer.colors.YELLOW)

        if dry_run:
            typer.secho("✓ Configuration valid (dry run - no report generated)", fg=typer.colors.GREEN)
            return

        # Run audit pipeline
        typer.echo("\nRunning audit pipeline...")
        _run_audit_pipeline(audit_config, output)

    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        # Use 'from None' to suppress Python traceback for clean CLI UX
        # Users should see "File not found", not internal stack traces
        raise typer.Exit(1) from None
    except ValueError as e:
        typer.secho(f"Configuration error: {e}", fg=typer.colors.RED, err=True)
        # Intentional: Clean error message for end users
        raise typer.Exit(1) from None
    except Exception as e:
        if "--verbose" in sys.argv or "-v" in sys.argv:
            logger.exception("Audit failed")
        typer.secho(f"Audit failed: {e}", fg=typer.colors.RED, err=True)
        # CLI design: Hide Python internals from users (verbose mode shows full details)
        raise typer.Exit(1) from None


def validate(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration file to validate",
        exists=True,
        file_okay=True,
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        "-p",
        help="Validate against specific profile",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate for strict mode compliance",
    ),
):
    """Validate a configuration file.

    This command checks if a configuration file is valid without
    running the audit pipeline.

    Examples:
        # Basic validation
        glassalpha validate --config audit.yaml

        # Validate for specific profile
        glassalpha validate -c audit.yaml --profile tabular_compliance

        # Check strict mode compliance
        glassalpha validate -c audit.yaml --strict

    """
    try:
        from ..config import load_config_from_file

        typer.echo(f"Validating configuration: {config}")

        # Load and validate
        audit_config = load_config_from_file(config, profile_name=profile, strict=strict)

        typer.echo(f"Profile: {audit_config.audit_profile}")
        typer.echo(f"Model type: {audit_config.model.type}")
        typer.echo(f"Strict mode: {'valid' if strict else 'not checked'}")

        # Report validation results
        typer.secho("\n✓ Configuration is valid", fg=typer.colors.GREEN)

        # Show warnings if any
        if not audit_config.reproducibility.random_seed:
            typer.secho("Warning: No random seed specified - results may vary", fg=typer.colors.YELLOW)

        if not audit_config.data.protected_attributes:
            typer.secho(
                "Warning: No protected attributes - fairness analysis limited",
                fg=typer.colors.YELLOW,
            )

    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        # CLI UX: Clean error messages, no Python tracebacks for users
        raise typer.Exit(1) from None
    except ValueError as e:
        typer.secho(f"Validation failed: {e}", fg=typer.colors.RED, err=True)
        # Intentional: User-friendly validation errors
        raise typer.Exit(1) from None
    except Exception as e:
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        # Design choice: Hide implementation details from end users
        raise typer.Exit(1) from None


def list_components_cmd(
    component_type: str | None = typer.Argument(
        None,
        help="Component type to list (models, explainers, metrics, profiles)",
    ),
    include_enterprise: bool = typer.Option(
        False,
        "--include-enterprise",
        "-e",
        help="Include enterprise components",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show component details",
    ),
):
    """List available components.

    Shows registered models, explainers, metrics, and audit profiles.

    Examples:
        # List all components
        glassalpha list

        # List specific type
        glassalpha list models

        # Include enterprise components
        glassalpha list --include-enterprise

    """
    from ..core import list_components

    components = list_components(component_type=component_type, include_enterprise=include_enterprise)

    if not components:
        typer.echo(f"No components found for type: {component_type}")
        return

    typer.echo("Available Components")
    typer.echo("=" * 40)

    for comp_type, items in components.items():
        typer.echo(f"\n{comp_type.upper()}:")

        if not items:
            typer.echo("  (none registered)")
        else:
            for item in sorted(items):
                if verbose:
                    # TODO: Show more details about each component
                    typer.echo(f"  - {item}")
                else:
                    typer.echo(f"  - {item}")

    if include_enterprise:
        typer.echo("\n" + "=" * 40)
        typer.secho("Note: Enterprise components require a valid license key", fg=typer.colors.YELLOW)
