"""CLI commands for dataset management.

This module provides CLI commands for listing, inspecting, and fetching
datasets from the GlassAlpha dataset registry.
"""

from pathlib import Path

import typer

from ..datasets.registry import REGISTRY
from ..utils.cache_dirs import resolve_data_root

app = typer.Typer(help="Manage built-in datasets")


@app.command("list")
def list_datasets():
    """List all available datasets in the registry."""
    if not REGISTRY:
        typer.echo("No datasets registered.")
        return

    typer.echo("Available datasets:")
    typer.echo("KEY              SCHEMA    DEFAULT_FILE")
    typer.echo("-" * 45)

    for key, spec in REGISTRY.items():
        typer.echo(f"{key: <15} {spec.schema_version: <9} {spec.default_relpath}")


@app.command("info")
def dataset_info(dataset: str = typer.Argument(..., help="Dataset key to inspect")):
    """Show detailed information about a specific dataset."""
    spec = REGISTRY.get(dataset)
    if not spec:
        typer.echo(f"Dataset '{dataset}' not found in registry.")
        typer.echo(f"Available datasets: {', '.join(REGISTRY.keys())}")
        raise typer.Exit(code=1)

    cache_root = resolve_data_root()
    expected_path = cache_root / spec.default_relpath

    # Log cache directory info for user visibility
    import os

    raw_env = os.getenv("GLASSALPHA_DATA_DIR")
    if raw_env:
        typer.echo(f"Cache dir: {raw_env} → {cache_root}")
    else:
        typer.echo(f"Cache dir: {cache_root}")

    typer.echo(f"Dataset: {spec.key}")
    typer.echo(f"Schema version: {spec.schema_version}")
    typer.echo(f"Default file: {spec.default_relpath}")
    typer.echo(f"Expected location: {expected_path}")
    typer.echo(f"Currently exists: {expected_path.exists()}")

    if spec.checksum:
        typer.echo(f"Checksum: {spec.checksum}")


@app.command("cache-dir")
def show_cache_dir():
    """Show the directory where datasets are cached."""
    import os

    cache_root = resolve_data_root()

    raw_env = os.getenv("GLASSALPHA_DATA_DIR")
    if raw_env:
        typer.echo(f"Cache dir: {raw_env} → {cache_root}")
    else:
        typer.echo(f"Cache dir: {cache_root}")


@app.command("fetch")
def fetch_dataset(
    dataset: str = typer.Argument(..., help="Dataset key to fetch"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download even if file exists"),
    dest: Path = typer.Option(None, "--dest", help="Custom destination path (overrides default cache location)"),
):
    """Fetch and cache a dataset from the registry."""
    spec = REGISTRY.get(dataset)
    if not spec:
        typer.echo(f"Dataset '{dataset}' not found in registry.")
        typer.echo(f"Available datasets: {', '.join(REGISTRY.keys())}")
        raise typer.Exit(code=1)

    # Import here to avoid circular imports
    from ..pipeline.audit import _ensure_dataset_availability

    # Create a minimal config object for the fetch operation
    class ConfigData:
        def __init__(self):
            self.dataset = dataset
            self.path = str(dest) if dest else None
            self.fetch = "always" if force else "if_missing"
            self.offline = False

    config = type("Config", (), {"data": ConfigData()})()

    try:
        # Use the same logic as the audit pipeline
        requested = _resolve_requested_path(config)
        final_path = _ensure_dataset_availability(config, requested)
        typer.echo(f"✅ Dataset '{dataset}' fetched successfully")
        typer.echo(f"📁 Location: {final_path}")

    except Exception as e:
        typer.echo(f"❌ Failed to fetch dataset '{dataset}': {e}")
        raise typer.Exit(code=1)


# Helper function (will be moved to audit.py)
def _resolve_requested_path(cfg) -> Path:
    """Resolve the requested data path from config."""
    from ..utils.cache_dirs import resolve_data_root

    if cfg.data.path:
        return Path(str(cfg.data.path)).expanduser().resolve()

    if cfg.data.dataset and cfg.data.dataset in REGISTRY:
        cache_root = resolve_data_root()
        return (cache_root / REGISTRY[cfg.data.dataset].default_relpath).resolve()

    raise FileNotFoundError("No data.path or data.dataset provided")
