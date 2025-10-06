"""Fairness metrics for bias detection and evaluation.

This module contains metrics for evaluating model fairness across different
demographic groups, including demographic parity, equalized odds, and
equal opportunity metrics.

Includes E11: Individual fairness metrics for detecting disparate treatment.
"""

from .individual import (
    IndividualFairnessMetrics,
    compute_consistency_score,
    counterfactual_flip_test,
    find_matched_pairs,
)

__all__ = [
    "IndividualFairnessMetrics",
    "compute_consistency_score",
    "counterfactual_flip_test",
    "find_matched_pairs",
]
