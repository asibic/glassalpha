"""Dataset loaders for common benchmark datasets.

This package provides loaders for canonical machine learning datasets
commonly used for fairness and compliance auditing.

Performance note: This module uses lazy imports via __getattr__ to avoid
loading heavy dependencies (pandas, numpy) during CLI --help rendering.
Datasets are imported only when actually accessed.
"""

from typing import Any

# Import registry - this is lightweight
from .registry import REGISTRY, DatasetSpec

__all__ = [
    "REGISTRY",
    "DatasetSpec",
    "GermanCreditDataset",
    "get_german_credit_schema",
    "load_german_credit",
]

# Track if we've registered datasets yet
_datasets_registered = False


def __getattr__(name: str) -> Any:
    """Lazy import dataset loaders to avoid heavy imports during CLI --help.

    This function is called when an attribute is not found in the module,
    allowing us to defer expensive imports until they're actually needed.
    """
    global _datasets_registered

    # Lazy import and register datasets on first access
    if not _datasets_registered:
        from .register_builtin import register_builtin_datasets

        register_builtin_datasets()
        _datasets_registered = True

    # Import german_credit components on demand
    if name == "GermanCreditDataset":
        from .german_credit import GermanCreditDataset

        return GermanCreditDataset
    if name == "get_german_credit_schema":
        from .german_credit import get_german_credit_schema

        return get_german_credit_schema
    if name == "load_german_credit":
        from .german_credit import load_german_credit

        return load_german_credit

    # If not found, raise AttributeError as expected
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
