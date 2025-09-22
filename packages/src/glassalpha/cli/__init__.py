"""Command-line interface for Glass Alpha.

This module provides the Typer-based CLI with command groups
for extensibility and clean organization.
"""

from .main import app
from .commands import audit, validate, list_components_cmd

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
