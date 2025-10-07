"""Audit entry points: from_model, from_predictions, from_config.

Phase 3: Main API surface for generating audit results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from glassalpha.api.result import AuditResult


def from_model(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    protected_attributes: Mapping[str, pd.Series | np.ndarray] | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
    random_seed: int = 42,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    explain: bool = True,
    recourse: bool = False,
    calibration: bool = True,
    stability: bool = False,
) -> AuditResult:
    """Generate audit from fitted model.
    
    Primary entry point for auditing ML models. Automatically extracts
    predictions and probabilities from the model, computes fairness metrics,
    and generates a byte-identical reproducible result.
    
    Args:
        model: Fitted sklearn-compatible model with predict() method
        X: Feature matrix (n_samples, n_features)
        y: True labels (n_samples,)
        protected_attributes: Dict mapping attribute names to arrays.
            Example: {"gender": gender_array, "race": race_array}
            Missing values (NaN) are mapped to "Unknown" category.
        sample_weight: Sample weights for weighted metrics (optional)
        random_seed: Random seed for deterministic SHAP sampling
        feature_names: Feature names (inferred from X if DataFrame)
        class_names: Class names for classification (e.g., ["Denied", "Approved"])
        explain: Generate SHAP explanations (default: True)
        recourse: Generate recourse recommendations (default: False)
        calibration: Compute calibration metrics (default: True, requires predict_proba)
        stability: Run stability tests (default: False, slower)
        
    Returns:
        AuditResult with performance, fairness, calibration, explanations
        
    Raises:
        GlassAlphaError (GAE1001): Invalid protected_attributes format
        GlassAlphaError (GAE1003): Length mismatch between X, y, protected_attributes
        GlassAlphaError (GAE1004): Non-binary classification (not yet supported)
        GlassAlphaError (GAE1012): MultiIndex not supported
        
    Examples:
        Basic audit (binary classification):
        >>> result = ga.audit.from_model(model, X_test, y_test)
        >>> result.performance.accuracy
        0.847
        
        With protected attributes:
        >>> result = ga.audit.from_model(
        ...     model, X_test, y_test,
        ...     protected_attributes={"gender": gender, "race": race}
        ... )
        >>> result.fairness.demographic_parity_max_diff
        0.023
        
        Missing values in protected attributes:
        >>> gender_with_nan = pd.Series([0, 1, np.nan, 1, 0])  # NaN → "Unknown"
        >>> result = ga.audit.from_model(model, X, y, protected_attributes={"gender": gender_with_nan})
        
        Without explanations (faster):
        >>> result = ga.audit.from_model(model, X, y, explain=False)
    """
    # Phase 3: Will implement with full validation and pipeline integration
    msg = "from_model() will be implemented in Phase 3"
    raise NotImplementedError(msg)


def from_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.DataFrame | np.ndarray | None = None,
    *,
    protected_attributes: Mapping[str, pd.Series | np.ndarray] | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
    random_seed: int = 42,
    class_names: Sequence[str] | None = None,
    model_fingerprint: str | None = None,
    calibration: bool = True,
) -> AuditResult:
    """Generate audit from predictions (no model required).
    
    Use this when you have predictions but not the model itself
    (e.g., external model, model deleted, compliance verification).
    
    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        y_proba: Predicted probabilities (n_samples, n_classes) or (n_samples,) for binary.
            Required for AUC, Brier score, calibration metrics.
        protected_attributes: Dict mapping attribute names to arrays
        sample_weight: Sample weights for weighted metrics (optional)
        random_seed: Random seed for deterministic operations
        class_names: Class names (e.g., ["Denied", "Approved"])
        model_fingerprint: Optional model hash for tracking (default: "unknown")
        calibration: Compute calibration metrics (default: True, requires y_proba)
        
    Returns:
        AuditResult with performance and fairness metrics.
        No explanations or recourse (requires model).
        
    Raises:
        GlassAlphaError (GAE1001): Invalid protected_attributes format
        GlassAlphaError (GAE1003): Length mismatch
        GlassAlphaError (GAE1004): Non-binary classification
        
    Examples:
        Binary classification with probabilities:
        >>> result = ga.audit.from_predictions(
        ...     y_true=y_test,
        ...     y_pred=predictions,
        ...     y_proba=probabilities[:, 1],  # Positive class proba
        ...     protected_attributes={"gender": gender}
        ... )
        >>> result.performance.roc_auc
        0.891
        
        Without probabilities (no AUC/calibration):
        >>> result = ga.audit.from_predictions(
        ...     y_true=y_test,
        ...     y_pred=predictions,
        ...     protected_attributes={"gender": gender},
        ...     calibration=False
        ... )
        >>> result.performance.accuracy
        0.847
        >>> result.performance.get("roc_auc")  # None (no proba)
        None
    """
    # Phase 3: Will implement with validation
    msg = "from_predictions() will be implemented in Phase 3"
    raise NotImplementedError(msg)


def from_config(config_path: str | Path) -> AuditResult:
    """Generate audit from YAML config file.
    
    Loads config, dataset, model from paths specified in YAML.
    Used for reproducible audits in CI/CD pipelines.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        AuditResult matching the config specification
        
    Raises:
        GlassAlphaError (GAE2002): Result ID mismatch (if expected_result_id provided)
        GlassAlphaError (GAE2003): Data hash mismatch
        FileNotFoundError: Config or referenced files not found
        
    Config schema:
        model:
          path: "models/xgboost.pkl"  # Pickled model
          type: "xgboost.XGBClassifier"  # For verification
          
        data:
          X_path: "data/X_test.parquet"
          y_path: "data/y_test.parquet"
          protected_attributes:
            gender: "data/gender.parquet"
            race: "data/race.parquet"
          expected_hashes:
            X: "sha256:abc123..."
            y: "sha256:def456..."
            
        audit:
          random_seed: 42
          explain: true
          recourse: false
          calibration: true
          
        validation:
          expected_result_id: "abc123..."  # Optional: fail if mismatch
          
    Examples:
        Basic usage:
        >>> result = ga.audit.from_config("audit.yaml")
        >>> result.to_pdf("report.pdf")
        
        Verify reproducibility:
        >>> result1 = ga.audit.from_config("audit.yaml")
        >>> result2 = ga.audit.from_config("audit.yaml")
        >>> assert result1.id == result2.id  # Byte-identical
    """
    # Phase 3: Will implement with config loading
    msg = "from_config() will be implemented in Phase 3"
    raise NotImplementedError(msg)


# Internal helpers for validation and normalization


def _normalize_protected_attributes(
    protected_attributes: Mapping[str, pd.Series | np.ndarray] | None,
    n_samples: int,
) -> dict[str, np.ndarray] | None:
    """Normalize protected attributes to dict of arrays with "Unknown" for NaN.
    
    Validates:
    - All arrays have length n_samples
    - Converts pandas Series to numpy arrays
    - Maps NaN → "Unknown" category (string)
    - Validates attribute names are strings
    
    Args:
        protected_attributes: Dict of attribute arrays or None
        n_samples: Expected number of samples
        
    Returns:
        Dict of numpy arrays with NaN → "Unknown", or None
        
    Raises:
        GlassAlphaError (GAE1001): Invalid format or names
        GlassAlphaError (GAE1003): Length mismatch
    """
    if protected_attributes is None:
        return None

    # Phase 3: Implement validation
    msg = "_normalize_protected_attributes will be implemented in Phase 3"
    raise NotImplementedError(msg)


def _get_probabilities(
    model: Any,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray | None:
    """Extract probabilities from model (predict_proba or decision_function).
    
    Tries in order:
    1. model.predict_proba(X) → returns probabilities
    2. model.decision_function(X) → returns scores
    3. Returns None if neither available
    
    For binary classification, returns (n_samples,) array of positive class proba.
    
    Args:
        model: Fitted model
        X: Feature matrix
        
    Returns:
        Array of probabilities/scores (n_samples,) or None
        
    Raises:
        GlassAlphaError (GAE1008): Only decision_function available (not probabilities)
    """
    # Phase 3: Implement probability extraction
    msg = "_get_probabilities will be implemented in Phase 3"
    raise NotImplementedError(msg)


def _validate_binary_classification(
    y: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray | None = None,
) -> None:
    """Validate that labels are binary (0/1 or two unique values).
    
    Args:
        y: Label array
        y_pred: Predicted labels (optional, for validation)
        
    Raises:
        GlassAlphaError (GAE1004): Non-binary classification
    """
    # Phase 3: Implement validation
    msg = "_validate_binary_classification will be implemented in Phase 3"
    raise NotImplementedError(msg)


def _validate_no_multiindex(df: pd.DataFrame) -> None:
    """Validate DataFrame does not have MultiIndex.
    
    Args:
        df: DataFrame to check
        
    Raises:
        GlassAlphaError (GAE1012): MultiIndex detected
    """
    # Phase 3: Implement validation
    msg = "_validate_no_multiindex will be implemented in Phase 3"
    raise NotImplementedError(msg)


def _extract_feature_names(X: pd.DataFrame | np.ndarray) -> list[str]:
    """Extract feature names from DataFrame or generate default names.
    
    Args:
        X: Feature matrix
        
    Returns:
        List of feature names (e.g., ["feature_0", "feature_1", ...] if ndarray)
    """
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    return [f"feature_{i}" for i in range(X.shape[1])]


def _to_numpy(arr: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert pandas Series/DataFrame to numpy array.
    
    Args:
        arr: Input array
        
    Returns:
        Numpy array
    """
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        return arr.values
    return arr

