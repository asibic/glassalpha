"""Command-line interface for GlassAlpha.

This module provides the Typer-based CLI with command groups
for extensibility and clean organization.
"""

from .commands import audit, docs, list_components_cmd, reasons, validate
from .main import app

__all__ = [
    "app",
    "audit",
    "docs",
    "list_components_cmd",
    "reasons",
    "validate",
]


# Entry point for the CLI
def main():
    """Main CLI entry point."""
    app()
