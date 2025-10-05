"""Metrics modules for GlassAlpha.

This module provides various metric implementations for model evaluation,
including performance metrics, fairness metrics, and drift metrics.
"""

from .registry import MetricRegistry

# Discover metrics from entry points
MetricRegistry.discover()

__all__ = ["MetricRegistry"]
