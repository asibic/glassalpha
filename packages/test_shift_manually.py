#!/usr/bin/env python3
"""Manual validation of shift reweighting logic."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from glassalpha.metrics.shift.reweighting import (
    ShiftSpecification,
    compute_shifted_weights,
    parse_shift_spec,
    validate_shift_feasibility,
)


def test_parse():
    """Test parsing."""
    print("\n1. Testing shift specification parsing...")

    # Test positive shift
    attr, shift = parse_shift_spec("gender:+0.1")
    assert attr == "gender" and shift == 0.1, "Failed: positive shift with +"
    print("   ✓ Positive shift with +: gender:+0.1 → ('gender', 0.1)")

    # Test negative shift
    attr, shift = parse_shift_spec("age:-0.05")
    assert attr == "age" and shift == -0.05, "Failed: negative shift"
    print("   ✓ Negative shift: age:-0.05 → ('age', -0.05)")

    # Test without +
    attr, shift = parse_shift_spec("race:0.1")
    assert attr == "race" and shift == 0.1, "Failed: positive without +"
    print("   ✓ Positive without +: race:0.1 → ('race', 0.1)")

    # Test error cases
    try:
        parse_shift_spec("gender+0.1")  # Missing colon
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   ✓ Invalid format raises: {str(e)[:50]}...")

    print("   ✓ Parse tests passed")


def test_validation():
    """Test validation."""
    print("\n2. Testing shift feasibility validation...")

    # Valid shifts
    validate_shift_feasibility(0.3, 0.1)  # 0.3 → 0.4 OK
    print("   ✓ Valid positive shift: 0.3 + 0.1 = 0.4 (OK)")

    validate_shift_feasibility(0.3, -0.1)  # 0.3 → 0.2 OK
    print("   ✓ Valid negative shift: 0.3 - 0.1 = 0.2 (OK)")

    # Boundary shifts
    validate_shift_feasibility(0.02, -0.01)  # → 0.01 OK
    validate_shift_feasibility(0.98, 0.01)  # → 0.99 OK
    print("   ✓ Boundary shifts OK: →0.01 and →0.99")

    # Invalid shifts
    try:
        validate_shift_feasibility(0.05, -0.05)  # → 0.00 FAIL
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   ✓ Below minimum raises: {str(e)[:50]}...")

    try:
        validate_shift_feasibility(0.95, 0.1)  # → 1.05 FAIL
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   ✓ Above maximum raises: {str(e)[:50]}...")

    print("   ✓ Validation tests passed")


def test_weight_computation():
    """Test weight computation."""
    print("\n3. Testing weight computation...")

    # Balanced data
    data = pd.DataFrame(
        {
            "gender": [1, 0, 1, 0, 1, 0, 1, 0],
        },
    )

    weights, spec = compute_shifted_weights(data, "gender", 0.1)

    # Original: 50% gender=1
    # Shifted: 60% gender=1
    # Weight for gender=1: 0.6 / 0.5 = 1.2
    # Weight for gender=0: 0.4 / 0.5 = 0.8

    assert spec.original_proportion == 0.5, f"Expected 0.5, got {spec.original_proportion}"
    assert spec.shifted_proportion == 0.6, f"Expected 0.6, got {spec.shifted_proportion}"

    gender_1_weights = weights[data["gender"] == 1]
    gender_0_weights = weights[data["gender"] == 0]

    assert np.allclose(gender_1_weights, 1.2), f"Expected 1.2, got {gender_1_weights[0]}"
    assert np.allclose(gender_0_weights, 0.8), f"Expected 0.8, got {gender_0_weights[0]}"

    print(f"   ✓ Original proportion: {spec.original_proportion:.2f}")
    print(f"   ✓ Shifted proportion: {spec.shifted_proportion:.2f}")
    print(f"   ✓ Weight for gender=1: {gender_1_weights[0]:.2f} (expected: 1.20)")
    print(f"   ✓ Weight for gender=0: {gender_0_weights[0]:.2f} (expected: 0.80)")

    # Verify weights sum to N
    assert np.allclose(weights.sum(), len(data)), "Weights should sum to N"
    print(f"   ✓ Weights sum to N: {weights.sum():.2f} = {len(data)}")

    # Imbalanced data
    data2 = pd.DataFrame(
        {
            "gender": [1, 1, 1, 0],  # 75% gender=1
        },
    )

    weights2, spec2 = compute_shifted_weights(data2, "gender", -0.1)

    # Original: 75% gender=1
    # Shifted: 65% gender=1
    # Weight for gender=1: 0.65 / 0.75 ≈ 0.867
    # Weight for gender=0: 0.35 / 0.25 = 1.4

    assert spec2.original_proportion == 0.75
    assert spec2.shifted_proportion == 0.65

    print("\n   ✓ Imbalanced data: 75% → 65%")
    print(f"     Weight for gender=1: {weights2[0]:.3f} (expected: ~0.867)")
    print(f"     Weight for gender=0: {weights2[3]:.3f} (expected: 1.400)")

    print("   ✓ Weight computation tests passed")


def test_specification_dataclass():
    """Test ShiftSpecification."""
    print("\n4. Testing ShiftSpecification...")

    spec = ShiftSpecification(
        attribute="gender",
        shift=0.1,
        original_proportion=0.35,
        shifted_proportion=0.45,
    )

    assert spec.attribute == "gender"
    assert spec.shift == 0.1

    # Test to_dict
    d = spec.to_dict()
    assert d["attribute"] == "gender"
    assert d["shift_value"] == 0.1
    assert d["original_proportion"] == 0.35
    assert d["shifted_proportion"] == 0.45

    print("   ✓ ShiftSpecification creation works")
    print("   ✓ to_dict() serialization works")

    # Test immutability
    try:
        spec.shift = 0.2  # type: ignore
        assert False, "Should be immutable"
    except AttributeError:
        print("   ✓ Specification is immutable (frozen)")

    # Test validation
    try:
        ShiftSpecification("gender", 0.7, 0.35, 1.05)  # Invalid: > 0.99
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   ✓ Invalid shifted proportion raises: {str(e)[:50]}...")

    print("   ✓ ShiftSpecification tests passed")


def main():
    """Run all manual tests."""
    print("=" * 80)
    print("SHIFT REWEIGHTING - MANUAL VALIDATION")
    print("=" * 80)

    try:
        test_parse()
        test_validation()
        test_weight_computation()
        test_specification_dataclass()

        print("\n" + "=" * 80)
        print("✓ ALL MANUAL TESTS PASSED")
        print("=" * 80)
        print("\nCore reweighting logic validated successfully!")
        print("Ready to proceed with metrics runner implementation.")
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
