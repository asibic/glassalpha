"""CLI commands for Glass Alpha.

This module implements the main commands available in the CLI,
including the core audit command with strict mode support.
"""

import typer
from pathlib import Path
from typing import Optional
import logging
import sys

logger = logging.getLogger(__name__)


def audit(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to audit configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
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
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Override audit profile",
    ),
    override_config: Optional[Path] = typer.Option(
        None,
        "--override",
        help="Additional config file to override settings",
        exists=True,
        file_okay=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate configuration without generating report",
    ),
):
    """Generate a compliance audit PDF report.
    
    This is the main command for Glass Alpha. It loads a configuration file,
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
        # Import here to avoid circular imports
        from ..config import load_config_from_file
        from ..core import list_components
        
        typer.echo(f"Glass Alpha Audit Generation")
        typer.echo(f"{'=' * 40}")
        
        # Load configuration
        typer.echo(f"Loading configuration from: {config}")
        if override_config:
            typer.echo(f"Applying overrides from: {override_config}")
        
        audit_config = load_config_from_file(
            config,
            override_path=override_config,
            profile_name=profile,
            strict=strict
        )
        
        # Report configuration
        typer.echo(f"Audit profile: {audit_config.audit_profile}")
        typer.echo(f"Strict mode: {'ENABLED' if audit_config.strict_mode else 'disabled'}")
        
        if audit_config.strict_mode:
            typer.secho(
                "⚠️  Strict mode enabled - enforcing regulatory compliance",
                fg=typer.colors.YELLOW
            )
        
        # Validate components exist
        available = list_components()
        model_type = audit_config.model.type
        
        if model_type not in available.get('models', []) and model_type != 'passthrough':
            typer.secho(
                f"Warning: Model type '{model_type}' not found in registry",
                fg=typer.colors.YELLOW
            )
        
        if dry_run:
            typer.secho(
                "✓ Configuration valid (dry run - no report generated)",
                fg=typer.colors.GREEN
            )
            return
        
        # TODO: Implement actual audit pipeline
        typer.echo(f"\nGenerating audit report...")
        typer.echo(f"Output: {output}")
        
        # Placeholder for actual implementation
        typer.secho(
            "\n⚠️  Note: Audit pipeline implementation pending (Phase 1)",
            fg=typer.colors.YELLOW
        )
        
        # Simulate success
        typer.secho(
            f"\n✓ Audit report would be generated at: {output}",
            fg=typer.colors.GREEN
        )
        
    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.secho(f"Configuration error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except Exception as e:
        if "--verbose" in sys.argv or "-v" in sys.argv:
            logger.exception("Audit failed")
        typer.secho(f"Audit failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def validate(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration file to validate",
        exists=True,
        file_okay=True,
    ),
    profile: Optional[str] = typer.Option(
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
        audit_config = load_config_from_file(
            config,
            profile_name=profile,
            strict=strict
        )
        
        typer.echo(f"Profile: {audit_config.audit_profile}")
        typer.echo(f"Model type: {audit_config.model.type}")
        typer.echo(f"Strict mode: {'valid' if strict else 'not checked'}")
        
        # Report validation results
        typer.secho(
            f"\n✓ Configuration is valid",
            fg=typer.colors.GREEN
        )
        
        # Show warnings if any
        if not audit_config.reproducibility.random_seed:
            typer.secho(
                "Warning: No random seed specified - results may vary",
                fg=typer.colors.YELLOW
            )
        
        if not audit_config.data.protected_attributes:
            typer.secho(
                "Warning: No protected attributes - fairness analysis limited",
                fg=typer.colors.YELLOW
            )
        
    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.secho(f"Validation failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def list_components_cmd(
    component_type: Optional[str] = typer.Argument(
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
    
    components = list_components(
        component_type=component_type,
        include_enterprise=include_enterprise
    )
    
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
        typer.secho(
            "Note: Enterprise components require a valid license key",
            fg=typer.colors.YELLOW
        )
