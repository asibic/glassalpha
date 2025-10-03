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


def print_banner(title: str = "GlassAlpha Audit Generation") -> None:
    """Print a standardized banner for CLI commands."""
    typer.echo(title)
    typer.echo("=" * 40)


def _ensure_docs_if_pdf(output_path: str) -> None:
    """Check if PDF output is requested and ensure docs dependencies are available.

    Args:
        output_path: Path to the output file

    Raises:
        SystemExit: If PDF is requested but jinja2 is not available

    """
    from pathlib import Path

    if Path(output_path).suffix.lower() == ".pdf":
        try:
            import weasyprint  # noqa: F401
        except ImportError:
            try:
                import reportlab  # noqa: F401
            except ImportError:
                raise SystemExit(
                    "PDF requested but no PDF backend found.\n"
                    'Install: pip install "glassalpha[docs]"\n'
                    "Or use: --output audit.html",
                )


def _ascii(s: str) -> str:
    """Convert Unicode characters to ASCII equivalents for CLI compatibility."""
    return (
        s.replace("✓", "OK")
        .replace("•", "*")
        .replace("—", "-")
        .replace("–", "-")
        .replace("…", "...")
        .replace(""", '"').replace(""", '"')
        .replace("'", "'")
        .replace("❌", "X")
        .replace("⚠️", "!")
        .replace("🎉", "")
        .replace("📊", "")
        .replace("📁", "")
        .replace("⏱️", "")
        .replace("🛡️", "")
        .replace("⚖️", "")
        .replace("🔍", "")
        .replace("📋", "")
        .replace("🔧", "")
    )


def _bootstrap_components() -> None:
    """Bootstrap basic built-in components for CLI operation.

    This imports the core built-ins that should always be available,
    ensuring the registry has basic models and explainers before
    preflight checks run.
    """
    logger.debug("Bootstrapping basic built-in components")

    # Import basic models that should always be available
    try:
        from ..models import passthrough  # noqa: F401 - registers PassThroughModel

        logger.debug("PassThroughModel imported")
    except ImportError as e:
        logger.error(f"Failed to import PassThroughModel: {e}")
        raise typer.Exit(2) from e

    # Import sklearn models if available (they're optional)
    try:
        from ..models.tabular import sklearn  # noqa: F401 - registers LogisticRegression, etc.

        logger.debug("sklearn models imported")
    except ImportError as e:
        logger.warning(f"sklearn models not available: {e}. Will use passthrough model only.")

    # Import basic explainers
    try:
        from ..explain import (
            coefficients,  # noqa: F401 - registers CoefficientsExplainer
            noop,  # noqa: F401 - registers NoOpExplainer
        )

        logger.debug("Basic explainers imported")
    except ImportError as e:
        logger.warning(f"Failed to import basic explainers: {e}")

    # Import basic metrics
    try:
        from ..metrics.performance import classification  # noqa: F401 - registers accuracy, etc.

        logger.debug("Basic metrics imported")
    except ImportError as e:
        logger.warning(f"Failed to import basic metrics: {e}")

    # Call discover() to ensure entry points are also registered
    from ..core.registry import ExplainerRegistry, MetricRegistry, ModelRegistry, ProfileRegistry

    ModelRegistry.discover()
    ExplainerRegistry.discover()
    MetricRegistry.discover()
    ProfileRegistry.discover()

    logger.debug("Component bootstrap completed")


def _run_audit_pipeline(config, output_path: Path, selected_explainer: str | None = None) -> None:
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
        audit_results = run_audit_pipeline(config, selected_explainer=selected_explainer)

        if not audit_results.success:
            typer.secho(
                _ascii(f"❌ Audit pipeline failed: {audit_results.error_message}"),
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

        pipeline_time = time.time() - start_time
        typer.secho(_ascii(f"✓ Audit pipeline completed in {pipeline_time:.2f}s"), fg=typer.colors.GREEN)

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
                typer.echo(_ascii(f"📋 Manifest: {manifest_path}"))
            except Exception as e:
                logger.warning(f"Failed to write manifest sidecar: {e}")

        # Success message
        total_time = time.time() - start_time
        typer.echo(_ascii("\n🎉 Audit Report Generated Successfully!"))
        typer.echo("=" * 50)
        typer.secho(_ascii(f"📁 Output: {pdf_path}"), fg=typer.colors.GREEN)
        typer.echo(_ascii(f"📊 Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)"))
        typer.echo(_ascii(f"⏱️  Total time: {total_time:.2f}s"))
        typer.echo(_ascii(f"   • Pipeline: {pipeline_time:.2f}s"))
        typer.echo(_ascii(f"   • PDF generation: {pdf_time:.2f}s"))

        # Regulatory compliance message
        if config.strict_mode:
            typer.secho(
                _ascii("\n🛡️  Strict mode: Report meets regulatory compliance requirements"),
                fg=typer.colors.YELLOW,
            )

        typer.echo("\nThe audit report is ready for review and regulatory submission.")

    except Exception as e:
        typer.secho(_ascii(f"\n❌ Audit failed: {e!s}"), fg=typer.colors.RED, err=True)

        # Show more details in verbose mode
        if "--verbose" in sys.argv or "-v" in sys.argv:
            logger.exception("Detailed audit failure information:")

        raise typer.Exit(1)


def _display_audit_summary(audit_results) -> None:
    """Display a summary of audit results."""
    typer.echo(_ascii("\n📊 Audit Summary:"))

    # Model performance
    if audit_results.model_performance:
        perf_count = len(
            [m for m in audit_results.model_performance.values() if isinstance(m, dict) and "error" not in m],
        )
        typer.echo(_ascii(f"  ✅ Performance metrics: {perf_count} computed"))

        # Show key metrics
        for name, result in audit_results.model_performance.items():
            if isinstance(result, dict) and "accuracy" in result:
                accuracy = result["accuracy"]
                status = "✅" if accuracy > 0.8 else "⚠️" if accuracy > 0.6 else "❌"
                typer.echo(_ascii(f"     {status} {name}: {accuracy:.1%}"))
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
        typer.echo(_ascii(f"  ⚖️  Fairness metrics: {computed_metrics}/{total_metrics} computed"))

        if bias_detected:
            typer.secho(_ascii(f"     ⚠️  Bias detected in: {', '.join(bias_detected[:2])}"), fg=typer.colors.YELLOW)
        elif computed_metrics > 0:
            typer.secho(_ascii("     ✅ No bias detected"), fg=typer.colors.GREEN)

    # SHAP explanations
    if audit_results.explanations:
        has_importance = "global_importance" in audit_results.explanations

        if has_importance:
            typer.echo(_ascii("  🔍 Explanations: ✅ Global feature importance"))

            # Show top feature
            importance = audit_results.explanations.get("global_importance", {})
            if importance:
                top_feature = max(importance.items(), key=lambda x: abs(x[1]))
                typer.echo(f"     Most important: {top_feature[0]} ({top_feature[1]:+.3f})")
        else:
            typer.echo(_ascii("  🔍 Explanations: ❌ Not available"))

    # Data summary
    if audit_results.data_summary and "shape" in audit_results.data_summary:
        rows, cols = audit_results.data_summary["shape"]
        typer.echo(_ascii(f"  📋 Dataset: {rows:,} samples, {cols} features"))

    # Selected components
    if audit_results.selected_components:
        typer.echo(_ascii(f"  🔧 Components: {len(audit_results.selected_components)} selected"))

        # Show model type
        for _comp_name, comp_info in audit_results.selected_components.items():
            if comp_info.get("type") == "model":
                typer.echo(f"     Model: {comp_info.get('name', 'unknown')}")
                break


def audit(  # pragma: no cover
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
    repro: bool = typer.Option(
        False,
        "--repro",
        help="Enable deterministic reproduction mode for byte-identical results",
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
        from .preflight import preflight_check_dependencies, preflight_check_model

        # Bootstrap basic components before any preflight checks
        _bootstrap_components()

        print_banner()

        # Preflight checks - ensure dependencies are available
        if not preflight_check_dependencies():
            raise typer.Exit(1)

        # Load configuration - this doesn't need heavy ML libraries
        typer.echo(f"Loading configuration from: {config}")
        if override_config:
            typer.echo(f"Applying overrides from: {override_config}")

        audit_config = load_config_from_file(config, override_path=override_config, profile_name=profile, strict=strict)

        # Validate model availability and apply fallbacks
        audit_config = preflight_check_model(audit_config)

        # Determine explainer selection early for consistent display
        from ..explain.registry import ExplainerRegistry

        selected_explainer = ExplainerRegistry.find_compatible(audit_config.model.type, audit_config.model_dump())
        typer.echo(f"Explainer: {selected_explainer}")

        # Apply repro mode if requested
        if repro:
            from ..runtime import set_repro  # noqa: PLC0415

            # Use config seed if available, otherwise default
            seed = (
                getattr(audit_config.reproducibility, "random_seed", 42)
                if hasattr(audit_config, "reproducibility")
                else 42
            )

            typer.echo("🔒 Enabling deterministic reproduction mode...")
            repro_status = set_repro(
                seed=seed,
                strict=True,  # Always use strict mode with --repro flag
                thread_control=True,  # Control threads for determinism
                warn_on_failure=True,
            )

            successful = sum(1 for control in repro_status["controls"].values() if control.get("success", False))
            total = len(repro_status["controls"])
            typer.echo(f"   Configured {successful}/{total} determinism controls")

            if successful < total:
                typer.secho(
                    "⚠️  Some determinism controls failed - results may not be fully reproducible",
                    fg=typer.colors.YELLOW,
                )

        # Report configuration
        typer.echo(f"Audit profile: {audit_config.audit_profile}")
        typer.echo(f"Strict mode: {'ENABLED' if audit_config.strict_mode else 'disabled'}")
        typer.echo(f"Repro mode: {'ENABLED' if repro else 'disabled'}")

        if audit_config.strict_mode:
            typer.secho("⚠️  Strict mode enabled - enforcing regulatory compliance", fg=typer.colors.YELLOW)

        if repro:
            typer.secho("🔒 Repro mode enabled - results will be deterministic", fg=typer.colors.BLUE)

        # Validate components exist
        available = list_components()
        model_type = audit_config.model.type

        if model_type not in available.get("models", []) and model_type != "passthrough":
            typer.secho(f"Warning: Model type '{model_type}' not found in registry", fg=typer.colors.YELLOW)

        if dry_run:
            typer.secho(_ascii("✓ Configuration valid (dry run - no report generated)"), fg=typer.colors.GREEN)
            return

        # Check PDF dependencies if PDF output requested
        _ensure_docs_if_pdf(str(output))

        # Run audit pipeline
        typer.echo("\nRunning audit pipeline...")
        _run_audit_pipeline(audit_config, output, selected_explainer)

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


def doctor():  # pragma: no cover
    """Check environment and optional features.

    This command diagnoses the current environment and shows what optional
    features are available and how to enable them.

    Examples:
        # Basic environment check
        glassalpha doctor

        # Verbose output
        glassalpha doctor --verbose

    """
    import importlib.util
    import platform

    typer.echo("GlassAlpha Environment Check")
    typer.echo("=" * 40)

    # Basic environment info
    typer.echo("Environment")
    typer.echo(f"  Python: {sys.version}")
    typer.echo(f"  OS: {platform.system()} {platform.machine()}")
    typer.echo()

    # Core features - always available
    typer.echo("Core Features (always available)")
    typer.echo("-" * 20)
    typer.echo("  ✅ LogisticRegression (scikit-learn)")
    typer.echo("  ✅ NoOp explainers (baseline)")
    typer.echo("  ✅ HTML reports (jinja2)")
    typer.echo("  ✅ Basic metrics (performance, fairness)")
    typer.echo()

    # Optional features check
    typer.echo("Optional Features")
    typer.echo("-" * 20)

    # Check all components
    has_shap = importlib.util.find_spec("shap") is not None
    has_xgboost = importlib.util.find_spec("xgboost") is not None
    has_lightgbm = importlib.util.find_spec("lightgbm") is not None
    has_matplotlib = importlib.util.find_spec("matplotlib") is not None
    
    # PDF backend check
    has_pdf_backend = False
    pdf_backend_name = None
    try:
        import weasyprint  # noqa: F401
        has_pdf_backend = True
        pdf_backend_name = "weasyprint"
    except ImportError:
        try:
            import reportlab  # noqa: F401
            has_pdf_backend = True
            pdf_backend_name = "reportlab"
        except ImportError:
            pass

    # Group: SHAP + Tree models (they come together in [explain] extra)
    has_all_explain = has_shap and has_xgboost and has_lightgbm
    if has_all_explain:
        typer.echo("  SHAP + tree models: ✅ installed")
        typer.echo("    (includes SHAP, XGBoost, LightGBM)")
    else:
        typer.echo("  SHAP + tree models: ❌ not installed")
        # Show what's partially there if any
        installed_parts = []
        if has_shap: installed_parts.append("SHAP")
        if has_xgboost: installed_parts.append("XGBoost")
        if has_lightgbm: installed_parts.append("LightGBM")
        if installed_parts:
            typer.echo(f"    (partially installed: {', '.join(installed_parts)})")

    # Templating (always available)
    typer.echo("  Templating: ✅ installed (jinja2)")

    # PDF backend
    if has_pdf_backend:
        typer.echo(f"  PDF generation: ✅ installed ({pdf_backend_name})")
    else:
        typer.echo("  PDF generation: ❌ not installed")

    # Visualization
    if has_matplotlib:
        typer.echo("  Visualization: ✅ installed (matplotlib)")
    else:
        typer.echo("  Visualization: ❌ not installed")

    typer.echo()

    # Status summary and next steps
    typer.echo("Status & Next Steps")
    typer.echo("-" * 20)

    missing_features = []
    
    # Check what's missing
    if not has_all_explain:
        missing_features.append("SHAP + tree models")
    if not has_pdf_backend:
        missing_features.append("PDF generation")
    if not has_matplotlib:
        missing_features.append("visualization")

    # Show appropriate message
    if not missing_features:
        typer.echo("  ✅ All optional features installed!")
        typer.echo()
    else:
        typer.echo("  Missing features:")
        typer.echo()
        
        # Show specific install commands for what's missing
        if not has_all_explain:
            typer.echo("  📦 For SHAP + tree models (XGBoost, LightGBM):")
            typer.echo("     pip install 'glassalpha[explain]'")
            typer.echo()
        
        if not has_pdf_backend:
            typer.echo("  📄 For PDF reports:")
            typer.echo("     pip install 'glassalpha[docs]'")
            typer.echo()
        
        if not has_matplotlib:
            typer.echo("  📊 For enhanced plots:")
            typer.echo("     pip install 'glassalpha[viz]'")
            typer.echo()
        
        # Show quick install if multiple things missing
        if len(missing_features) > 1:
            typer.echo("  💡 Or install everything at once:")
            typer.echo("     pip install 'glassalpha[all]'")
            typer.echo()

    # Smart recommendation based on what's installed
    if has_pdf_backend:
        suggested_command = "glassalpha audit --config configs/quickstart.yaml --output quickstart.pdf"
    else:
        suggested_command = "glassalpha audit --config configs/quickstart.yaml --output quickstart.html"

    typer.echo(f"Ready to run: {suggested_command}")
    typer.echo()


def validate(  # pragma: no cover
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
        typer.secho(_ascii("\n✓ Configuration is valid"), fg=typer.colors.GREEN)

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


def list_components_cmd(  # pragma: no cover
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
