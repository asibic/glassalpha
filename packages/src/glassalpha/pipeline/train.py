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
    """Train a model from configuration parameters with optional calibration.

    Args:
        cfg: Configuration object with model.type, model.params, and optional model.calibration
        X: Training features
        y: Target values

    Returns:
        Trained model instance (potentially calibrated)

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

    # Check if we need probability predictions for calibration
    calibration_config = getattr(cfg.model, "calibration", None)
    require_proba = bool(calibration_config and calibration_config.method)

    model.fit(X, y, require_proba=require_proba, **params)

    logger.info(f"Model training completed for {model_type}")

    # Apply calibration if requested
    if calibration_config and calibration_config.method:
        from ..models.calibration import maybe_calibrate  # noqa: PLC0415

        logger.info("Applying probability calibration")

        # Get the underlying sklearn-compatible estimator
        base_estimator = getattr(model, "model", model)

        # Apply calibration
        calibrated_estimator = maybe_calibrate(
            base_estimator,
            method=calibration_config.method,
            cv=calibration_config.cv,
            ensemble=calibration_config.ensemble,
        )

        # Update the model's estimator
        if hasattr(model, "model"):
            model.model = calibrated_estimator
            logger.info("Updated wrapper with calibrated estimator")
        else:
            # For models that don't have a .model attribute, we need to handle differently
            logger.warning("Model wrapper doesn't have .model attribute, calibration may not work properly")

        # Re-fit the calibrated estimator (CalibratedClassifierCV needs this)
        if hasattr(calibrated_estimator, "fit"):
            # Encode labels if the wrapper has label encoding
            y_for_calibration = y
            if hasattr(model, "_encode_labels"):
                y_for_calibration = model._encode_labels(y)
            elif hasattr(model, "classes_") and hasattr(model, "_label_encoder"):
                # Use existing label encoding if available
                y_for_calibration = model._label_encoder.transform(y)

            calibrated_estimator.fit(X, y_for_calibration)
            logger.info("Calibrated estimator fitted successfully")

    return model
