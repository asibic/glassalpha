"""Dataset loaders for common benchmark datasets.

This package provides loaders for canonical machine learning datasets
commonly used for fairness and compliance auditing.
"""

from .german_credit import (
    GermanCreditDataset,
    get_german_credit_schema,
    load_german_credit,
)
from .register_builtin import register_builtin_datasets

# Import registry and register built-in datasets
from .registry import REGISTRY, DatasetSpec

# Register built-in datasets during package import
register_builtin_datasets()

__all__ = [
    "REGISTRY",
    "DatasetSpec",
    "GermanCreditDataset",
    "get_german_credit_schema",
    "load_german_credit",
]
