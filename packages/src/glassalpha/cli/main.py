"""Main CLI application using Typer.

This module sets up the command groups and structure for the GlassAlpha CLI,
enabling future expansion without breaking changes.

ARCHITECTURE NOTE: Uses Typer function-call defaults (B008 lint rule)
which is the documented Typer pattern. Also uses clean CLI exception
handling with 'from None' to hide Python internals from end users.
"""

import logging
import sys
from pathlib import Path

import typer
from platformdirs import user_data_dir

from .. import __version__

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Main CLI app
app = typer.Typer(
    name="glassalpha",
    help="GlassAlpha - AI Compliance Toolkit",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=True,
    epilog="""Installation Options:
  Minimal install: pip install glassalpha                   # LogisticRegression + basic explainers
  SHAP + trees:    pip install 'glassalpha[explain]'        # SHAP + XGBoost + LightGBM
  Visualization:   pip install 'glassalpha[viz]'            # Matplotlib + Seaborn
  PDF reports:     pip install 'glassalpha[docs]'           # PDF generation
  Everything:      pip install 'glassalpha[all]'            # All optional features

For more information, visit: https://glassalpha.com""",
)

# Command groups
# Core functionality (OSS)
datasets_app = typer.Typer(
    help="Dataset management operations",
    no_args_is_help=True,
)

# Future command groups (for Phase 2+)
# These are stubbed now to establish the structure
dashboard_app = typer.Typer(
    help="Dashboard operations (Enterprise only)",
    no_args_is_help=True,
)

monitor_app = typer.Typer(
    help="Monitoring operations (Enterprise only)",
    no_args_is_help=True,
)


# First-run detection helper
def _show_first_run_tip():
    """Show helpful tip on first run."""
    # Skip for --help, --version, and doctor command
    if "--help" in sys.argv or "-h" in sys.argv or "--version" in sys.argv or "-V" in sys.argv or "doctor" in sys.argv:
        return

    # Check for state file
    state_dir = Path(user_data_dir("glassalpha", "glassalpha"))
    state_file = state_dir / ".first_run_complete"

    if not state_file.exists():
        # Create state directory and file
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file.touch()

        # Show tip
        typer.echo()
        typer.secho("👋 Welcome to GlassAlpha!", fg=typer.colors.BRIGHT_BLUE, bold=True)
        typer.echo("   Run 'glassalpha doctor' to check your environment and see what features are available.")
        typer.echo()


# Version callback
def version_callback(value: bool):
    """Print version and exit."""
    if value:
        typer.echo(f"GlassAlpha version {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-error output",
    ),
):
    """GlassAlpha - Transparent, auditable, regulator-ready ML audits.

    Use 'glassalpha COMMAND --help' for more information on a command.
    """
    # Set logging level based on flags
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # First-run detection - show helpful tip once
    _show_first_run_tip()


# Add command groups to main app
# Core functionality (OSS)
app.add_typer(datasets_app, name="datasets")

# Enterprise features - these will check for license
app.add_typer(dashboard_app, name="dashboard")
app.add_typer(monitor_app, name="monitor")

# Import and register commands
from .commands import audit, doctor, list_components_cmd, validate

# Register main commands
app.command()(audit)
app.command("validate")(validate)
app.command("list", help="List available components")(list_components_cmd)
app.command("doctor", help="Check environment and optional features")(doctor)

# Register datasets commands with lazy loading (Phase 2 performance optimization)
# These import the datasets module only when the command is actually invoked,
# not during --help rendering, saving ~500ms of import time


@datasets_app.command("list")
def list_datasets_lazy():
    """List all available datasets in the registry."""
    from .datasets import list_datasets

    return list_datasets()


@datasets_app.command("info")
def dataset_info_lazy(dataset: str = typer.Argument(..., help="Dataset key to inspect")):
    """Show detailed information about a specific dataset."""
    from .datasets import dataset_info

    return dataset_info(dataset)


@datasets_app.command("cache-dir")
def show_cache_dir_lazy():
    """Show the directory where datasets are cached."""
    from .datasets import show_cache_dir

    return show_cache_dir()


@datasets_app.command("fetch")
def fetch_dataset_lazy(
    dataset: str = typer.Argument(..., help="Dataset key to fetch"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download even if file exists"),
    dest: Path = typer.Option(None, "--dest", help="Custom destination path"),
):
    """Fetch and cache a dataset from the registry."""
    from .datasets import fetch_dataset

    return fetch_dataset(dataset, force, dest)


# Dashboard commands (enterprise stubs)
@dashboard_app.command("serve")  # pragma: no cover
def dashboard_serve(
    port: int = typer.Option(8080, "--port", "-p", help="Port to serve on"),
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to"),
):
    """Start the monitoring dashboard (Enterprise only)."""
    from ..core.features import check_feature

    @check_feature("dashboard")
    def _serve():
        typer.echo(f"Starting dashboard on {host}:{port}")
        # Future implementation
        typer.echo("Dashboard feature coming in Phase 2")

    try:
        _serve()
    except Exception as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        # CLI design: Clean error messages for enterprise feature failures
        raise typer.Exit(1) from None


@monitor_app.command("drift")  # pragma: no cover
def monitor_drift(
    config: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    baseline: Path = typer.Option(..., "--baseline", "-b", help="Baseline manifest"),
):
    """Monitor model drift (Enterprise only)."""
    from ..core.features import check_feature

    @check_feature("monitoring")
    def _monitor():
        typer.echo(f"Monitoring drift from {baseline}")
        # Future implementation
        typer.echo("Monitoring feature coming in Phase 2")

    try:
        _monitor()
    except Exception as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        # Intentional: Hide Python internals from CLI users
        raise typer.Exit(1) from None


# Add models command to show available models and installation options
@app.command()
def models():
    """Show available models and installation options."""
    # Import models to trigger registration

    from ..core import ModelRegistry

    typer.echo("Available Models:")
    typer.echo("=" * 50)

    available_models = ModelRegistry.available_plugins()

    if not available_models:
        typer.secho("No models available. Install with: pip install 'glassalpha[tabular]'", fg=typer.colors.YELLOW)
        return

    for model, available in available_models.items():
        status = "✅" if available else "❌"
        extra_info = ""

        if model in ["xgboost", "lightgbm"] and not available:
            extra_info = " (install with: pip install 'glassalpha[explain]')"
        elif model in ["logistic_regression", "sklearn_generic"] and available:
            extra_info = " (included in base install)"

        typer.echo(f"  {status} {model}{extra_info}")

        typer.echo()
        typer.echo("Installation Options:")
        typer.echo("  Minimal install: pip install glassalpha                  # LogisticRegression + basic explainers")
        typer.echo("  SHAP + trees:    pip install 'glassalpha[explain]'       # SHAP + XGBoost + LightGBM")
        typer.echo("  Visualization:   pip install 'glassalpha[viz]'           # Matplotlib + Seaborn")
        typer.echo("  PDF reports:     pip install 'glassalpha[docs]'          # PDF generation")
        typer.echo("  Everything:      pip install 'glassalpha[all]'           # All optional features")


if __name__ == "__main__":  # pragma: no cover
    app()
