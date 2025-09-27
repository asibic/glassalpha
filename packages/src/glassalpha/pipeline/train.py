"""Training utilities for GlassAlpha pipeline.

This module provides utilities for training models from configuration,
ensuring that model parameters from YAML configuration are properly
passed to the underlying estimators.
"""

import logging
from typing import Any

import pandas as pd

from ..core.registry import ModelRegistry

logger = logging.getLogger(__name__)


def train_from_config(cfg: Any, X: pd.DataFrame, y: Any) -> Any:
    """Train a model from configuration parameters.

    Args:
        cfg: Configuration object with model.type and model.params
        X: Training features
        y: Target values

    Returns:
        Trained model instance

    Raises:
        ValueError: If model type is not found in registry
        RuntimeError: If model doesn't support fit method

    """
    # Get model type from config
    model_type = cfg.model.type

    # Ensure model modules are imported for registration
    if (
        model_type == "xgboost"
        or model_type == "lightgbm"
        or model_type == "logistic_regression"
        or model_type == "sklearn_generic"
    ):
        pass

    # Get model class from registry
    model_class = ModelRegistry.get(model_type)
    if not model_class:
        msg = f"Unknown model type: {model_type}"
        raise ValueError(msg)

    logger.info(f"Training {model_type} model from configuration")

    # Create model instance
    model = model_class()

    # Extract parameters from config
    params = dict(getattr(cfg.model, "params", {}))

    # Add random_state if specified in reproducibility config
    if hasattr(cfg, "reproducibility") and hasattr(cfg.reproducibility, "random_seed"):
        params["random_state"] = cfg.reproducibility.random_seed

    # Fit model with parameters
    if not hasattr(model, "fit"):
        msg = f"Model type {model_type} does not support fit method"
        raise RuntimeError(msg)

    logger.info(f"Fitting model with parameters: {list(params.keys())}")
    logger.info(f"Parameter values: {params}")
    model.fit(X, y, **params)

    logger.info(f"Model training completed for {model_type}")
    return model
