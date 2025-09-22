"""Main CLI application using Typer.

This module sets up the command groups and structure for the Glass Alpha CLI,
enabling future expansion without breaking changes.
"""

import typer
from typing import Optional
from pathlib import Path
import logging

from ..core import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Main CLI app
app = typer.Typer(
    name="glassalpha",
    help="Glass Alpha - AI Compliance Toolkit",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=True,
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

# Version callback
def version_callback(value: bool):
    """Print version and exit."""
    if value:
        typer.echo(f"Glass Alpha version {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Optional[bool] = typer.Option(
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
    """Glass Alpha - Transparent, auditable, regulator-ready ML audits.
    
    Use 'glassalpha COMMAND --help' for more information on a command.
    """
    # Set logging level based on flags
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)


# Add command groups to main app
# Enterprise features - these will check for license
app.add_typer(dashboard_app, name="dashboard")
app.add_typer(monitor_app, name="monitor")

# Import and register commands
from .commands import audit, validate, list_components_cmd

# Register main commands
app.command()(audit)
app.command("validate")(validate)
app.command("list", help="List available components")(list_components_cmd)

# Dashboard commands (enterprise stubs)
@dashboard_app.command("serve")
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
        raise typer.Exit(1)


@monitor_app.command("drift")
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
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
