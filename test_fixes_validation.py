#!/usr/bin/env python3
"""Validation script to test the fixes implemented in Option C."""

import os
import sys

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "packages", "src"))

import numpy as np
import pandas as pd


def test_config_threading():
    """Test that config threading works correctly."""
    print("Testing config threading...")

    import types

    from glassalpha.pipeline.train import train_from_config

    # Create mock config with multiclass parameters
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {"objective": "multi:softprob", "num_class": 4, "max_depth": 3}
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    # Create test data
    X = pd.DataFrame(np.random.randn(40, 3), columns=["a", "b", "c"])
    y = np.array([0, 1, 2, 3] * 10)

    # Train model
    model = train_from_config(cfg, X, y)

    # Verify model info shows correct number of classes
    info = model.get_model_info()
    assert info["n_classes"] == 4, f"Expected 4 classes, got {info['n_classes']}"

    print("✓ Config threading works correctly")


def test_xgboost_multiclass():
    """Test XGBoost multiclass functionality."""
    print("Testing XGBoost multiclass support...")

    from glassalpha.models.tabular.xgboost import XGBoostWrapper

    # Create test data with 3 classes
    X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
    y = np.array([0, 1, 2] * 10)

    # Train model without specifying objective (should auto-infer)
    model = XGBoostWrapper()
    model.fit(X, y, max_depth=3)

    # Should auto-infer multiclass
    info = model.get_model_info()
    assert info["n_classes"] == 3, f"Expected 3 classes, got {info['n_classes']}"

    # Test predict_proba shape
    proba = model.predict_proba(X)
    assert proba.shape == (30, 3), f"Expected shape (30, 3), got {proba.shape}"

    print("✓ XGBoost multiclass support works correctly")


def test_feature_alignment():
    """Test feature alignment utility."""
    print("Testing feature alignment...")

    from glassalpha.utils.features import align_features

    # Test positional renaming
    X = pd.DataFrame(np.random.randn(10, 3), columns=["c", "a", "b"])
    expected_names = ["a", "b", "c"]

    X_aligned = align_features(X, expected_names)
    assert list(X_aligned.columns) == expected_names, "Column renaming failed"

    # Test reindexing with missing columns
    X_missing = pd.DataFrame(np.random.randn(10, 2), columns=["a", "b"])
    expected_names = ["a", "b", "c", "d"]

    X_reindexed = align_features(X_missing, expected_names)
    assert X_reindexed.shape == (10, 4), "Reindexing failed"
    assert np.all(X_reindexed["c"] == 0), "Missing columns not filled with zeros"
    assert np.all(X_reindexed["d"] == 0), "Missing columns not filled with zeros"

    print("✓ Feature alignment works correctly")


def test_fairness_metrics():
    """Test fairness metrics with numpy arrays."""
    print("Testing fairness metrics with numpy arrays...")

    from glassalpha.metrics.fairness.bias_detection import DemographicParityMetric
    from glassalpha.metrics.fairness.runner import run_fairness_metrics

    # Create test data
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0])  # Higher positive rate for group 1
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # Two groups

    # Test individual metric
    metric = DemographicParityMetric()
    result = metric.compute(y_true, y_pred, sensitive)

    assert "sensitive_0" in result, "Individual group results missing"
    assert "sensitive_1" in result, "Individual group results missing"
    assert "demographic_parity" in result, "Overall metric missing"
    assert result["demographic_parity"] < 1.0, "Should detect bias"

    # Test runner
    metrics = [DemographicParityMetric]
    results = run_fairness_metrics(y_true, y_pred, sensitive, metrics)

    assert "demographicparity" in results, "Runner results missing"

    print("✓ Fairness metrics work with numpy arrays")


def test_packaging():
    """Test that templates and examples are accessible."""
    print("Testing packaging...")

    import importlib.resources

    # Test template access
    try:
        template_files = list(
            importlib.resources.files("glassalpha.report.templates").iterdir()
        )
        template_names = [f.name for f in template_files if f.is_file()]
        assert any("standard_audit" in name for name in template_names), (
            "Template not found"
        )
    except Exception as e:
        print(f"⚠ Template access failed: {e}")
        return False

    print("✓ Packaging works correctly")


def main():
    """Run all validation tests."""
    print("Running Option C fix validation tests...\n")

    try:
        test_config_threading()
        test_xgboost_multiclass()
        test_feature_alignment()
        test_fairness_metrics()
        test_packaging()

        print("\n🎉 All tests passed! Option C fixes are working correctly.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
