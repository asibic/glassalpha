"""Dataset loaders for common benchmark datasets.

This package provides loaders for canonical machine learning datasets
commonly used for fairness and compliance auditing.
"""

from .german_credit import (
    GermanCreditDataset,
    get_german_credit_schema,
    load_german_credit,
)

__all__ = [
    "GermanCreditDataset",
    "load_german_credit",
    "get_german_credit_schema",
]
