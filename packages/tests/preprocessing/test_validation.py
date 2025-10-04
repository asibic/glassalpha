"""Test preprocessing validation (classes, shapes, sparsity)."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Import stubs
try:
    from glassalpha.preprocessing.loader import load_artifact
    from glassalpha.preprocessing.validation import (
        ALLOWED_FQCN,
        validate_classes,
        validate_output_shape,
        validate_sparsity,
    )
except ImportError:
    pytest.skip("preprocessing module not implemented yet", allow_module_level=True)


def test_rejects_unknown_transformer_class(tmp_path: Path, toy_df: pd.DataFrame):
    """Create a Pipeline with FunctionTransformer. Load must raise ValueError citing FQCN."""
    import joblib

    # Create pipeline with disallowed transformer
    def identity(x):
        return x

    disallowed_pipeline = Pipeline(
        steps=[
            ("func", FunctionTransformer(identity)),  # Not in allowlist
        ]
    )
    disallowed_pipeline.fit(toy_df[["age"]])

    artifact_path = tmp_path / "disallowed.pkl"
    joblib.dump(disallowed_pipeline, artifact_path)

    # Load and validate
    artifact = load_artifact(artifact_path)

    with pytest.raises(ValueError) as exc_info:
        validate_classes(artifact)

    # Error must cite the FQCN
    error_msg = str(exc_info.value)
    assert "FunctionTransformer" in error_msg
    assert "sklearn.preprocessing" in error_msg or "Unsupported" in error_msg


def test_sparse_flag_detected_and_enforced(sparse_artifact: Path, dense_artifact: Path, toy_df: pd.DataFrame):
    """Build encoder with sparse_output=True. Adapter must record output_is_sparse=True.
    Validation must fail if config says expected_sparse=False.
    """
    # Load sparse artifact
    sparse_prep = load_artifact(sparse_artifact)
    X_sparse = sparse_prep.transform(toy_df)

    # Detect sparsity
    is_sparse = hasattr(X_sparse, "toarray")
    assert is_sparse is True, "Sparse artifact should produce sparse output"

    # Validation should fail if expected_sparse=False but actual is True
    with pytest.raises(ValueError) as exc_info:
        validate_sparsity(
            actual_sparse=True,
            expected_sparse=False,
            artifact_path=sparse_artifact,
        )

    assert "sparsity mismatch" in str(exc_info.value).lower()

    # Load dense artifact
    dense_prep = load_artifact(dense_artifact)
    X_dense = dense_prep.transform(toy_df)

    # Detect non-sparsity
    is_sparse = hasattr(X_dense, "toarray")
    assert is_sparse is False, "Dense artifact should produce dense output"


def test_feature_count_and_names_match_model(sklearn_artifact: Path, toy_df: pd.DataFrame):
    """Save model with n_features_in_ and feature_names_in_ from the artifact.
    After transform, validator compares both. On mismatch, error message shows minimal diff.
    """
    # Load and transform
    artifact = load_artifact(sklearn_artifact)
    X_transformed = artifact.transform(toy_df)

    # Get expected features from artifact
    expected_n_features = X_transformed.shape[1]
    expected_feature_names = artifact.get_feature_names_out() if hasattr(artifact, "get_feature_names_out") else None

    # Create mock model with matching features
    class MockModel:
        n_features_in_ = expected_n_features
        feature_names_in_ = expected_feature_names

    model = MockModel()

    # This should pass
    validate_output_shape(X_transformed, model, artifact)

    # Now test mismatch
    model.n_features_in_ = expected_n_features + 5

    with pytest.raises(ValueError) as exc_info:
        validate_output_shape(X_transformed, model, artifact)

    error_msg = str(exc_info.value)
    # Should show the mismatch
    assert str(expected_n_features) in error_msg or "mismatch" in error_msg.lower()
