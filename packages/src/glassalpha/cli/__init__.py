"""Command-line interface for GlassAlpha.

This module provides the Typer-based CLI with command groups
for extensibility and clean organization.
"""

from .commands import audit, list_components_cmd, validate
from .main import app

__all__ = [
    "app",
    "audit",
    "validate",
    "list_components_cmd",
]


# Entry point for the CLI
def main():
    """Main CLI entry point."""
    app()
