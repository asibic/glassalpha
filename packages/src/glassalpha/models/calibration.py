"""Probability calibration utilities for GlassAlpha models.

This module provides optional probability calibration using sklearn's
CalibratedClassifierCV to improve probability estimates for compliance
use cases that depend on calibrated risk scores.
"""

import logging
from typing import Any

from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


def maybe_calibrate(
    estimator: Any,
    method: str | None = None,
    cv: int | None = None,
    ensemble: bool = True,
) -> Any:
    """Optionally calibrate a trained estimator's probability predictions.

    This function wraps an estimator with CalibratedClassifierCV if calibration
    is requested. It's designed to be called after model training to improve
    probability estimates for compliance and risk assessment use cases.

    Args:
        estimator: Trained sklearn-compatible estimator
        method: Calibration method ('isotonic', 'sigmoid', or None for no calibration)
        cv: Number of cross-validation folds (default: 5)
        ensemble: Whether to use ensemble=True (recommended for better calibration)

    Returns:
        Original estimator if method is None, otherwise CalibratedClassifierCV wrapper

    Raises:
        ValueError: If method is not recognized

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> rf = RandomForestClassifier().fit(X, y)
        >>> calibrated_rf = maybe_calibrate(rf, method='isotonic', cv=5)
        >>> # Now calibrated_rf.predict_proba() gives better calibrated probabilities

    """
    if not method:
        logger.debug("No calibration method specified, returning original estimator")
        return estimator

    method = method.lower()
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError(f"Unknown calibration method: '{method}'. Must be 'isotonic' or 'sigmoid'")

    cv = cv or 5  # Default to 5-fold CV

    logger.info("Applying %s calibration with %d-fold cross-validation", method, cv)

    try:
        calibrated_estimator = CalibratedClassifierCV(
            estimator,
            method=method,
            cv=cv,
            ensemble=ensemble,
        )

        logger.info("Successfully created calibrated estimator wrapper")
        return calibrated_estimator

    except Exception as e:
        logger.error("Failed to create calibrated estimator: %s", e)
        logger.warning("Falling back to uncalibrated estimator")
        return estimator


def validate_calibration_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize calibration configuration.

    Args:
        config: Raw calibration configuration dictionary

    Returns:
        Validated and normalized configuration

    Raises:
        ValueError: If configuration is invalid

    """
    if not config:
        return {}

    # Normalize method
    method = config.get("method")
    if method:
        method = method.lower()
        if method not in {"isotonic", "sigmoid"}:
            raise ValueError(f"Invalid calibration method: '{method}'. Must be 'isotonic' or 'sigmoid'")

    # Validate CV folds
    cv = config.get("cv", 5)
    if not isinstance(cv, int) or cv < 2:
        raise ValueError(f"CV folds must be an integer >= 2, got: {cv}")

    # Validate ensemble flag
    ensemble = config.get("ensemble", True)
    if not isinstance(ensemble, bool):
        raise ValueError(f"Ensemble must be a boolean, got: {ensemble}")

    validated_config = {
        "method": method,
        "cv": cv,
        "ensemble": ensemble,
    }

    logger.debug("Validated calibration config: %s", validated_config)
    return validated_config


def get_calibration_info(estimator: Any) -> dict[str, Any]:
    """Get information about calibration status of an estimator.

    Args:
        estimator: Estimator to inspect

    Returns:
        Dictionary with calibration information

    """
    info = {
        "is_calibrated": False,
        "calibration_method": None,
        "cv_folds": None,
        "ensemble": None,
        "base_estimator_type": type(estimator).__name__,
    }

    if isinstance(estimator, CalibratedClassifierCV):
        info.update(
            {
                "is_calibrated": True,
                "calibration_method": estimator.method,
                "cv_folds": estimator.cv,
                "ensemble": estimator.ensemble,
                "base_estimator_type": type(estimator.estimator).__name__,
            }
        )

    return info


def assess_calibration_quality(y_true, y_proba, n_bins: int = 10) -> dict[str, float]:
    """Assess the quality of probability calibration using reliability metrics.

    This function computes calibration metrics to evaluate how well-calibrated
    the probability predictions are. Well-calibrated probabilities are crucial
    for compliance and risk assessment applications.

    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for positive class
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration quality metrics

    """
    import numpy as np  # noqa: PLC0415
    from sklearn.calibration import calibration_curve  # noqa: PLC0415
    from sklearn.metrics import brier_score_loss  # noqa: PLC0415

    try:
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true,
            y_proba,
            n_bins=n_bins,
            strategy="uniform",
        )

        # Compute Brier score (lower is better)
        brier_score = brier_score_loss(y_true, y_proba)

        # Compute Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            # Find predictions in this bin
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = y_true[in_bin].mean()
                # Average confidence in this bin
                avg_confidence_in_bin = y_proba[in_bin].mean()
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        # Compute Maximum Calibration Error (MCE)
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

        return {
            "brier_score": float(brier_score),
            "expected_calibration_error": float(ece),
            "maximum_calibration_error": float(mce),
            "n_bins": n_bins,
        }

    except Exception as e:
        logger.error("Failed to assess calibration quality: %s", e)
        return {
            "brier_score": None,
            "expected_calibration_error": None,
            "maximum_calibration_error": None,
            "error": str(e),
        }


def recommend_calibration_method(estimator_type: str, dataset_size: int) -> str:
    """Recommend calibration method based on estimator type and dataset size.

    Args:
        estimator_type: Type of base estimator (e.g., 'XGBClassifier', 'LogisticRegression')
        dataset_size: Number of training samples

    Returns:
        Recommended calibration method ('isotonic' or 'sigmoid')

    """
    # General recommendations based on sklearn documentation and best practices

    # For small datasets, sigmoid is more stable
    if dataset_size < 1000:
        return "sigmoid"

    # For tree-based models, isotonic often works better
    tree_based = {"XGBClassifier", "LGBMClassifier", "RandomForestClassifier", "GradientBoostingClassifier"}
    if any(tree_type in estimator_type for tree_type in tree_based):
        return "isotonic"

    # For linear models, sigmoid is often sufficient
    linear_models = {"LogisticRegression", "LinearSVC", "SGDClassifier"}
    if any(linear_type in estimator_type for linear_type in linear_models):
        return "sigmoid"

    # Default to isotonic for larger datasets (more flexible)
    return "isotonic"
