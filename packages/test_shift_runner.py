#!/usr/bin/env python3
"""Manual validation of shift metrics runner."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from glassalpha.metrics.shift.runner import run_shift_analysis


def create_synthetic_data(n=1000, seed=42):
    """Create synthetic binary classification data."""
    np.random.seed(seed)

    # Protected attribute: 40% gender=1
    gender = np.random.binomial(1, 0.4, n)

    # Features correlated with gender
    feature1 = gender * 0.5 + np.random.normal(0, 1, n)
    feature2 = np.random.normal(0, 1, n)

    # Target: biased toward gender=1
    logits = feature1 * 0.8 + feature2 * 0.3 + gender * 0.5
    y_true = (logits > 0).astype(int)

    # Predictions: slightly different from true
    noise = np.random.normal(0, 0.3, n)
    pred_logits = logits + noise
    y_pred = (pred_logits > 0).astype(int)
    y_proba = 1 / (1 + np.exp(-pred_logits))  # Sigmoid

    return y_true, y_pred, y_proba, gender


def test_basic_shift_analysis():
    """Test basic shift analysis."""
    print("\n1. Testing basic shift analysis...")

    # Create data
    y_true, y_pred, y_proba, gender = create_synthetic_data()
    sensitive_df = pd.DataFrame({"gender": gender})

    print(f"   - Data: N={len(y_true)}, gender=1: {gender.mean():.2%}")
    print(f"   - Accuracy: {(y_true == y_pred).mean():.3f}")

    # Run shift analysis
    result = run_shift_analysis(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_df,
        attribute="gender",
        shift=0.1,  # +10pp
        y_proba=y_proba,
        threshold=None,  # No gating
    )

    print("\n   Shift Specification:")
    print(f"   - Original: {result.shift_spec.original_proportion:.3f}")
    print(f"   - Shifted: {result.shift_spec.shifted_proportion:.3f}")
    print(f"   - Shift: {result.shift_spec.shift:+.3f}")

    print("\n   Baseline Metrics:")
    if "fairness" in result.baseline_metrics:
        for k, v in result.baseline_metrics["fairness"].items():
            print(f"     {k}: {v:.4f}")

    print("\n   Shifted Metrics:")
    if "fairness" in result.shifted_metrics:
        for k, v in result.shifted_metrics["fairness"].items():
            print(f"     {k}: {v:.4f}")

    print("\n   Degradation:")
    for k, v in result.degradation.items():
        print(f"     {k}: {v:+.4f}")

    print(f"\n   Gate Status: {result.gate_status}")

    assert result.gate_status == "PASS", "Expected PASS with no threshold"
    assert len(result.degradation) > 0, "Should have computed degradation"

    print("\n   ✓ Basic shift analysis passed")


def test_degradation_detection():
    """Test degradation gate detection."""
    print("\n2. Testing degradation detection...")

    # Create data
    y_true, y_pred, y_proba, gender = create_synthetic_data()
    sensitive_df = pd.DataFrame({"gender": gender})

    # Test with strict threshold (should fail)
    result_strict = run_shift_analysis(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_df,
        attribute="gender",
        shift=0.2,  # Larger shift
        y_proba=y_proba,
        threshold=0.01,  # Very strict
    )

    print("   Strict threshold (0.01):")
    print(f"   - Gate Status: {result_strict.gate_status}")
    print(f"   - Violations: {len(result_strict.violations)}")
    if result_strict.violations:
        for v in result_strict.violations[:3]:  # Show first 3
            print(f"     • {v}")

    # Test with loose threshold (should pass)
    result_loose = run_shift_analysis(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_df,
        attribute="gender",
        shift=0.05,  # Smaller shift
        y_proba=y_proba,
        threshold=0.5,  # Very loose
    )

    print("\n   Loose threshold (0.5):")
    print(f"   - Gate Status: {result_loose.gate_status}")
    print(f"   - Violations: {len(result_loose.violations)}")

    assert result_loose.gate_status == "PASS", "Should pass with loose threshold"

    print("\n   ✓ Degradation detection passed")


def test_calibration_metrics():
    """Test calibration metrics computation."""
    print("\n3. Testing calibration metrics...")

    # Create data
    y_true, y_pred, y_proba, gender = create_synthetic_data()
    sensitive_df = pd.DataFrame({"gender": gender})

    # Run analysis
    result = run_shift_analysis(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_df,
        attribute="gender",
        shift=0.1,
        y_proba=y_proba,
    )

    print("   Baseline Calibration:")
    if "calibration" in result.baseline_metrics:
        for k, v in result.baseline_metrics["calibration"].items():
            print(f"     {k}: {v:.4f}")

    print("\n   Shifted Calibration:")
    if "calibration" in result.shifted_metrics:
        for k, v in result.shifted_metrics["calibration"].items():
            print(f"     {k}: {v:.4f}")

    assert "calibration" in result.baseline_metrics
    assert "ece" in result.baseline_metrics["calibration"]
    assert "brier_score" in result.baseline_metrics["calibration"]

    print("\n   ✓ Calibration metrics passed")


def test_to_dict_serialization():
    """Test JSON serialization."""
    print("\n4. Testing JSON serialization...")

    # Create data
    y_true, y_pred, y_proba, gender = create_synthetic_data(n=100)
    sensitive_df = pd.DataFrame({"gender": gender})

    # Run analysis
    result = run_shift_analysis(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_df,
        attribute="gender",
        shift=0.1,
        y_proba=y_proba,
        threshold=0.05,
    )

    # Convert to dict
    d = result.to_dict()

    print("   Serialized keys:")
    for key in d.keys():
        print(f"     - {key}")

    assert "shift_specification" in d
    assert "baseline_metrics" in d
    assert "shifted_metrics" in d
    assert "degradation" in d
    assert "gate_status" in d

    # Check nested structure
    assert "attribute" in d["shift_specification"]
    assert "fairness" in d["baseline_metrics"]

    print("\n   ✓ JSON serialization passed")


def main():
    """Run all tests."""
    print("=" * 80)
    print("SHIFT METRICS RUNNER - MANUAL VALIDATION")
    print("=" * 80)

    try:
        test_basic_shift_analysis()
        test_degradation_detection()
        test_calibration_metrics()
        test_to_dict_serialization()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print("\nMetrics runner validated successfully!")
        print("Ready to proceed with CLI integration.")
        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ TESTS FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
