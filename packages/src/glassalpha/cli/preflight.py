"""Preflight checks for CLI commands with dependency validation and fallbacks.

This module handles dependency checking, model availability validation,
and provides graceful fallbacks when optional components are missing.
"""

import logging
from typing import Any

import typer

from ..core.registry import ModelRegistry, ProfileRegistry
from ..explain.registry import ExplainerRegistry
from ..metrics.registry import MetricRegistry

logger = logging.getLogger(__name__)


def preflight_check_model(config: Any) -> Any:
    """Validate model availability and provide fallbacks.

    Args:
        config: Audit configuration object

    Returns:
        Modified config with fallback model if needed

    Raises:
        typer.Exit: If no suitable model is available and fallbacks disabled

    """
    if not hasattr(config, "model") or not config.model:
        # Use default model if no model specified
        config.model = _create_default_model_config()
        return config

    model_type = config.model.type

    # Check if requested model is available
    available_models = ModelRegistry.available_plugins()

    if not available_models.get(model_type, False):
        # Model not available - check for fallbacks
        if getattr(config.model, "allow_fallback", True):
            fallback_model = _find_fallback_model(model_type, available_models)
            if fallback_model:
                typer.echo(
                    f"Model '{model_type}' not available. Falling back to '{fallback_model}'. "
                    f"To enable '{model_type}', run: pip install 'glassalpha[{model_type}]'",
                )
                config.model.type = fallback_model
                return config
            # No fallback available
            _show_installation_error(model_type)
            raise typer.Exit(1)
        # Fallbacks disabled
        _show_installation_error(model_type)
        raise typer.Exit(1)

    return config


def _create_default_model_config():
    """Create default model configuration for logistic regression."""
    from types import SimpleNamespace

    model_config = SimpleNamespace()
    model_config.type = "logistic_regression"
    model_config.params = {"random_state": 42}
    return model_config


def _find_fallback_model(requested_model: str, available_models: dict[str, bool]) -> str | None:
    """Find a suitable fallback model when requested model is unavailable.

    Args:
        requested_model: The originally requested model type
        available_models: Dictionary of available models

    Returns:
        Fallback model name or None if no suitable fallback

    """
    # Priority order for fallbacks
    fallback_priority = [
        "logistic_regression",  # Always available baseline
    ]

    # Remove requested model from fallback list if it's in there
    fallback_priority = [m for m in fallback_priority if m != requested_model]

    # Return first available fallback
    for model in fallback_priority:
        if available_models.get(model, False):
            return model

    return None


def _show_installation_error(model_type: str):
    """Show helpful installation error message.

    Args:
        model_type: The model type that failed

    """
    hint = ModelRegistry.get_install_hint(model_type)
    if hint:
        typer.secho(
            f"Model '{model_type}' requires an optional dependency. {hint}",
            fg=typer.colors.RED,
            err=True,
        )
    else:
        typer.secho(
            f"Model '{model_type}' is not available. Please check your installation.",
            fg=typer.colors.RED,
            err=True,
        )


def preflight_check_dependencies():
    """Check core dependencies and show warnings for missing optional ones."""
    typer.echo("GlassAlpha Audit Generation")
    typer.echo("=" * 40)

    # Check model availability
    available_models = ModelRegistry.available_plugins()

    if not available_models.get("logistic_regression", False):
        typer.secho(
            "Warning: No models available. Install scikit-learn for basic functionality.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return False

    # Check explainer compatibility
    try:
        # Try to find a compatible explainer for a basic model type
        ExplainerRegistry.find_compatible("logistic_regression", config)
        typer.secho("✓ Explainers available and compatible", fg=typer.colors.GREEN)
    except RuntimeError as e:
        # Get install hint for the first available explainer that needs installation
        for name in ExplainerRegistry.names():
            hint = ExplainerRegistry.get_install_hint(name)
            if hint:
                typer.secho(
                    f"Note: {e!s}. {hint}",
                    fg=typer.colors.BLUE,
                    err=True,
                )
                break
        else:
            typer.secho(
                f"Warning: {e!s}",
                fg=typer.colors.YELLOW,
                err=True,
            )

    # Check metric availability
    available_metrics = MetricRegistry.available_plugins()

    # At least one metric should be available for meaningful audits
    metric_available = any(available_metrics.values())
    if not metric_available:
        typer.secho(
            "Warning: No metrics available. Core functionality may be limited.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    # Check profile availability
    profile_name = config.get("audit_profile")
    if profile_name:
        try:
            profile_cls = ProfileRegistry.get(profile_name)
            # Check if profile requires extras
            if hasattr(profile_cls, "required_extras"):
                required_extras = profile_cls.required_extras(config)
                if required_extras:
                    typer.secho(
                        f"Profile '{profile_name}' requires extras: {', '.join(required_extras)}",
                        fg=typer.colors.BLUE,
                    )
        except KeyError:
            typer.secho(
                f"Warning: Unknown audit profile '{profile_name}'. Using generic defaults.",
                fg=typer.colors.YELLOW,
                err=True,
            )

    return True


def show_available_models():
    """Display available models and their installation status."""
    available_models = ModelRegistry.available_plugins()

    typer.echo("Available models:")
    for model, available in available_models.items():
        status = "✓" if available else "✗"
        hint = ModelRegistry.get_install_hint(model)
        install_info = f" (install with: {hint})" if hint and not available else ""

        typer.echo(f"  {status} {model}{install_info}")

    if not any(available_models.values()):
        typer.secho(
            "No models available. Install models with: pip install 'glassalpha[tabular]'",
            fg=typer.colors.YELLOW,
        )
