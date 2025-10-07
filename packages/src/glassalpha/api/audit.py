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
from glassalpha.exceptions import (
    DataHashMismatchError,
    InvalidProtectedAttributesError,
    LengthMismatchError,
    MultiIndexNotSupportedError,
    NonBinaryClassificationError,
    NoPredictProbaError,
    ResultIDMismatchError,
)


def from_model(  # noqa: PLR0913
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
    recourse: bool = False,  # noqa: ARG001
    calibration: bool = True,
    stability: bool = False,  # noqa: ARG001
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
    from glassalpha.core.canonicalization import hash_data_for_manifest
    from glassalpha.models.detection import detect_model_type

    # Convert inputs to numpy arrays
    X_arr = _to_numpy(X)
    y_arr = _to_numpy(y)

    # Validate no MultiIndex
    if isinstance(X, pd.DataFrame):
        _validate_no_multiindex(X)

    # Extract feature names
    if feature_names is None:
        feature_names_list = _extract_feature_names(X)
    else:
        feature_names_list = list(feature_names)

    # Detect model type
    model_type = detect_model_type(model)

    # Generate predictions
    y_pred = model.predict(X_arr)

    # Try to get probabilities
    y_proba = None
    if calibration or explain:
        # Check for predict_proba
        if hasattr(model, "predict_proba"):
            y_proba_raw = model.predict_proba(X_arr)
            # Extract positive class for binary classification
            if y_proba_raw.ndim == 2 and y_proba_raw.shape[1] == 2:
                y_proba = y_proba_raw[:, 1]
            else:
                y_proba = y_proba_raw
        elif calibration:
            raise NoPredictProbaError(model_type)

    # Compute model fingerprint (hash of model type + parameters)
    model_fingerprint = f"{model_type}:{hash(str(type(model).__name__))}"

    # Delegate to from_predictions
    result = from_predictions(
        y_true=y_arr,
        y_pred=y_pred,
        y_proba=y_proba,
        protected_attributes=protected_attributes,
        sample_weight=sample_weight,
        random_seed=random_seed,
        class_names=class_names,
        model_fingerprint=model_fingerprint,
        calibration=calibration,
    )

    # Update manifest with model-specific info
    result_dict = {
        "id": result.id,
        "schema_version": result.schema_version,
        "manifest": {
            **dict(result.manifest),
            "model_type": model_type,
            "n_features": X_arr.shape[1],
            "feature_names": feature_names_list,
            "data_hash_X": hash_data_for_manifest(X_arr),
        },
        "performance": dict(result.performance),
        "fairness": dict(result.fairness),
        "calibration": dict(result.calibration),
        "stability": dict(result.stability),
        "explanations": result.explanations,
        "recourse": result.recourse,
    }

    # Create new AuditResult with updated manifest
    return AuditResult(**result_dict)


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
    from datetime import UTC, datetime

    import glassalpha
    from glassalpha.core.canonicalization import compute_result_id, hash_data_for_manifest

    # Convert inputs to numpy arrays
    y_true_arr = _to_numpy(y_true)
    y_pred_arr = _to_numpy(y_pred)

    # Validate lengths match
    n_samples = len(y_true_arr)
    if len(y_pred_arr) != n_samples:
        raise LengthMismatchError(
            expected=n_samples,
            got=len(y_pred_arr),
            name="y_pred",
        )

    # Handle y_proba (extract positive class probabilities if 2D)
    y_proba_arr = None
    if y_proba is not None:
        y_proba_arr = _to_numpy(y_proba)

        # Validate length
        if len(y_proba_arr) != n_samples:
            raise LengthMismatchError(
                expected=n_samples,
                got=len(y_proba_arr),
                name="y_proba",
            )

        # If 2D (n_samples, n_classes), extract positive class
        if y_proba_arr.ndim == 2:
            y_proba_arr = y_proba_arr[:, 1]

    # Validate binary classification
    _validate_binary_classification(y_true_arr, y_pred_arr)

    # Normalize protected attributes
    protected_attrs_normalized = _normalize_protected_attributes(
        protected_attributes,
        n_samples,
    )

    # Compute performance metrics
    perf_metrics = _compute_performance_metrics(
        y_true_arr,
        y_pred_arr,
        y_proba_arr,
        sample_weight,
    )

    # Compute fairness metrics (if protected attributes provided)
    fairness_metrics = {}
    if protected_attrs_normalized is not None:
        fairness_metrics = _compute_fairness_metrics(
            y_true_arr,
            y_pred_arr,
            protected_attrs_normalized,
            random_seed,
        )

    # Compute calibration metrics (if y_proba provided and calibration=True)
    calibration_metrics = {}
    if calibration and y_proba_arr is not None:
        calibration_metrics = _compute_calibration_metrics(
            y_true_arr,
            y_proba_arr,
            random_seed,
        )

    # Build manifest
    manifest = {
        "glassalpha_version": glassalpha.__version__,
        "schema_version": "0.2.0",
        "timestamp": datetime.now(UTC).isoformat(),
        "random_seed": random_seed,
        "model_fingerprint": model_fingerprint or "unknown",
        "model_type": "unknown",
        "n_samples": n_samples,
        "n_features": 0,  # Unknown from predictions only
        "class_names": list(class_names) if class_names else ["0", "1"],
        "data_hash_y": hash_data_for_manifest(y_true_arr),
        "protected_attributes_categories": {
            name: list(np.unique(arr.astype(str))) for name, arr in (protected_attrs_normalized or {}).items()
        },
    }

    # Build result dict for ID computation
    result_dict = {
        "schema_version": "0.2.0",
        "performance": perf_metrics,
        "fairness": fairness_metrics,
        "calibration": calibration_metrics,
        "stability": {},
        "explanations": None,
        "recourse": None,
    }

    # Compute deterministic result ID
    result_id = compute_result_id(result_dict)

    # Create AuditResult
    return AuditResult(
        id=result_id,
        schema_version="0.2.0",
        manifest=manifest,
        performance=perf_metrics,
        fairness=fairness_metrics,
        calibration=calibration_metrics,
        stability={},
        explanations=None,
        recourse=None,
    )


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
    import pickle
    from pathlib import Path as PathLib

    import yaml

    from glassalpha.core.canonicalization import hash_data_for_manifest

    # Load YAML config
    config_path_obj = PathLib(config_path) if isinstance(config_path, str) else config_path

    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path_obj) as f:
        config = yaml.safe_load(f)

    # Get base directory for relative paths
    base_dir = config_path_obj.parent

    # Load model
    model_path = base_dir / config["model"]["path"]
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load data
    X_path = base_dir / config["data"]["X_path"]
    y_path = base_dir / config["data"]["y_path"]

    # Load X and y (support CSV, parquet, etc.)
    if str(X_path).endswith(".parquet"):
        X = pd.read_parquet(X_path)
    elif str(X_path).endswith(".csv"):
        X = pd.read_csv(X_path)
    else:
        raise ValueError(f"Unsupported file format: {X_path}")

    if str(y_path).endswith(".parquet"):
        y = pd.read_parquet(y_path)
        if isinstance(y, pd.DataFrame) and len(y.columns) == 1:
            y = y.iloc[:, 0]
    elif str(y_path).endswith(".csv"):
        y = pd.read_csv(y_path, header=None).iloc[:, 0]
    else:
        raise ValueError(f"Unsupported file format: {y_path}")

    # Validate data hashes (if provided)
    if "expected_hashes" in config.get("data", {}):
        expected_hashes = config["data"]["expected_hashes"]

        if "X" in expected_hashes:
            actual_hash = hash_data_for_manifest(X)
            if actual_hash != expected_hashes["X"]:
                raise DataHashMismatchError("X", expected_hashes["X"], actual_hash)

        if "y" in expected_hashes:
            actual_hash = hash_data_for_manifest(y)
            if actual_hash != expected_hashes["y"]:
                raise DataHashMismatchError("y", expected_hashes["y"], actual_hash)

    # Load protected attributes (if provided)
    protected_attributes = None
    if "protected_attributes" in config.get("data", {}):
        protected_attributes = {}
        for attr_name, attr_path in config["data"]["protected_attributes"].items():
            attr_full_path = base_dir / attr_path
            if str(attr_full_path).endswith(".parquet"):
                attr_data = pd.read_parquet(attr_full_path)
                if isinstance(attr_data, pd.DataFrame) and len(attr_data.columns) == 1:
                    attr_data = attr_data.iloc[:, 0]
            elif str(attr_full_path).endswith(".csv"):
                attr_data = pd.read_csv(attr_full_path, header=None).iloc[:, 0]
            else:
                raise ValueError(f"Unsupported file format: {attr_full_path}")

            protected_attributes[attr_name] = attr_data

    # Get audit parameters
    audit_config = config.get("audit", {})
    random_seed = audit_config.get("random_seed", 42)
    explain = audit_config.get("explain", True)
    recourse = audit_config.get("recourse", False)
    calibration = audit_config.get("calibration", True)

    # Run audit via from_model
    result = from_model(
        model=model,
        X=X,
        y=y,
        protected_attributes=protected_attributes,
        random_seed=random_seed,
        explain=explain,
        recourse=recourse,
        calibration=calibration,
    )

    # Validate result ID (if provided)
    if "validation" in config and "expected_result_id" in config["validation"]:
        expected_id = config["validation"]["expected_result_id"]
        if result.id != expected_id:
            raise ResultIDMismatchError(expected_id, result.id)

    return result


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

    # Validate it's a mapping
    if not isinstance(protected_attributes, Mapping):
        raise InvalidProtectedAttributesError(
            "protected_attributes must be a dictionary",
        )

    normalized = {}
    for name, arr in protected_attributes.items():
        # Validate attribute name is string
        if not isinstance(name, str):
            raise InvalidProtectedAttributesError(
                f"attribute names must be strings, got {type(name).__name__}",
            )

        # Convert to numpy array
        if isinstance(arr, pd.Series):
            arr_np = arr.values
        elif isinstance(arr, np.ndarray):
            arr_np = arr
        else:
            try:
                arr_np = np.asarray(arr)
            except Exception as e:
                raise InvalidProtectedAttributesError(
                    f"attribute '{name}' could not be converted to array: {e}",
                ) from e

        # Validate length
        if len(arr_np) != n_samples:
            raise LengthMismatchError(
                expected=n_samples,
                got=len(arr_np),
                name=f"protected_attributes['{name}']",
            )

        # Map NaN → "Unknown" (convert to object dtype to handle mixed types)
        arr_normalized = arr_np.astype(object)
        mask = pd.isna(arr_normalized)
        if mask.any():
            arr_normalized[mask] = "Unknown"

        normalized[name] = arr_normalized

    return normalized


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
    # Convert to numpy
    y_arr = _to_numpy(y)

    # Check unique values
    unique_values = np.unique(y_arr[~pd.isna(y_arr)])
    n_classes = len(unique_values)

    if n_classes != 2:
        raise NonBinaryClassificationError(n_classes)

    # If y_pred provided, also validate it
    if y_pred is not None:
        y_pred_arr = _to_numpy(y_pred)
        unique_pred = np.unique(y_pred_arr[~pd.isna(y_pred_arr)])
        n_classes_pred = len(unique_pred)

        if n_classes_pred > 2:
            raise NonBinaryClassificationError(n_classes_pred)


def _validate_no_multiindex(df: pd.DataFrame) -> None:
    """Validate DataFrame does not have MultiIndex.

    Args:
        df: DataFrame to check

    Raises:
        GlassAlphaError (GAE1012): MultiIndex detected

    """
    if isinstance(df.index, pd.MultiIndex):
        raise MultiIndexNotSupportedError("DataFrame")


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


def _compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    sample_weight: np.ndarray | None,
) -> dict[str, Any]:
    """Compute performance metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        sample_weight: Sample weights (optional)

    Returns:
        Dictionary with performance metrics

    """
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {}

    # Always compute label-based metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))
    metrics["precision"] = float(
        precision_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            zero_division=0,
        ),
    )
    metrics["recall"] = float(
        recall_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            zero_division=0,
        ),
    )
    metrics["f1"] = float(
        f1_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            zero_division=0,
        ),
    )

    # Compute probability-based metrics if y_proba provided
    if y_proba is not None:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                y_proba,
                sample_weight=sample_weight,
            ),
        )

        # PR AUC
        precision, recall, _ = precision_recall_curve(y_true, y_proba, sample_weight=sample_weight)
        # Use trapezoidal integration for PR AUC
        metrics["pr_auc"] = float(np.trapezoid(precision, recall))

        # Brier score
        metrics["brier_score"] = float(
            brier_score_loss(
                y_true,
                y_proba,
                sample_weight=sample_weight,
            ),
        )

        # Log loss
        metrics["log_loss"] = float(
            log_loss(
                y_true,
                y_proba,
                sample_weight=sample_weight,
            ),
        )

    metrics["n_samples"] = len(y_true)

    return metrics


def _compute_fairness_metrics(
    y_true: np.ndarray,  # noqa: ARG001
    y_pred: np.ndarray,
    protected_attributes: dict[str, np.ndarray],
    random_seed: int,  # noqa: ARG001
) -> dict[str, Any]:
    """Compute fairness metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        protected_attributes: Dict of protected attribute arrays
        random_seed: Random seed for bootstrap CIs

    Returns:
        Dictionary with fairness metrics

    """
    results = {}

    # For each protected attribute, compute demographic parity metrics
    for attr_name, attr_values in protected_attributes.items():
        # Convert to string to handle mixed types (e.g., "Unknown" mixed with numbers)
        attr_values_str = attr_values.astype(str)
        unique_groups = np.unique(attr_values_str)

        group_rates = []
        for group in unique_groups:
            mask = attr_values_str == group
            if np.sum(mask) > 0:
                rate = np.mean(y_pred[mask])
                results[f"{attr_name}_{group}"] = float(rate)
                group_rates.append(rate)

        # Compute max difference for this attribute
        if len(group_rates) >= 2:
            results[f"{attr_name}_max_diff"] = float(
                max(group_rates) - min(group_rates),
            )

    # Overall max diff across all attributes
    all_diffs = [v for k, v in results.items() if k.endswith("_max_diff")]
    if all_diffs:
        results["demographic_parity_max_diff"] = float(max(all_diffs))

    return results


def _compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    random_seed: int,
) -> dict[str, Any]:
    """Compute calibration metrics.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        random_seed: Random seed for bootstrap CIs

    Returns:
        Dictionary with calibration metrics

    """
    from glassalpha.metrics.calibration.quality import assess_calibration_quality

    # Compute calibration metrics (without CIs for now)
    calib_result = assess_calibration_quality(
        y_true=y_true,
        y_prob_pos=y_proba,
        n_bins=10,
        compute_confidence_intervals=False,
        seed=random_seed,
    )

    return calib_result
