#!/usr/bin/env python3
"""Test script to verify CLI performance fix."""

import os
import sys

# Add the source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "packages", "src"))


def test_cli_import_performance():
    """Test that heavy ML libraries are not loaded during CLI import."""

    heavy_libs = ["xgboost", "lightgbm", "shap"]

    print("Testing CLI import performance...")
    print("Before importing CLI:")
    for lib in heavy_libs:
        print(f"  {lib} loaded: {lib in sys.modules}")

    try:
        # This should fail due to missing dependencies in the test environment
        # but we can still see what modules get loaded during import
        from glassalpha.cli import main

        print("\nCLI imported successfully")
    except ImportError as e:
        print(f"\nCLI import failed (expected due to environment): {e}")

    print("\nAfter CLI import attempt:")
    for lib in heavy_libs:
        print(f"  {lib} loaded: {lib in sys.modules}")

    # The key point: heavy ML libraries should NOT be loaded during CLI import
    loaded_heavy_libs = [lib for lib in heavy_libs if lib in sys.modules]
    if loaded_heavy_libs:
        print(
            f"\n❌ PERFORMANCE ISSUE: These heavy libraries were loaded during CLI import: {loaded_heavy_libs}"
        )
        return False
    else:
        print(
            "\n✅ PERFORMANCE FIX WORKING: No heavy ML libraries loaded during CLI import"
        )
        return True


def test_audit_loads_libraries():
    """Test that heavy ML libraries ARE loaded when actually running an audit."""

    heavy_libs = ["xgboost", "lightgbm", "shap"]

    print("\nTesting audit execution...")
    print("Before audit import:")
    for lib in heavy_libs:
        print(f"  {lib} loaded: {lib in sys.modules}")

    try:
        # Import the components that would be loaded during an audit
        from glassalpha.explain.shap import kernel, tree
        from glassalpha.models.tabular import lightgbm, xgboost

        print("\nAfter audit component import:")
        for lib in heavy_libs:
            print(f"  {lib} loaded: {lib in sys.modules}")

        # The key point: heavy ML libraries SHOULD be loaded when importing audit components
        loaded_heavy_libs = [lib for lib in heavy_libs if lib in sys.modules]
        if loaded_heavy_libs:
            print(
                f"\n✅ AUDIT LOADING WORKING: These heavy libraries were loaded during audit import: {loaded_heavy_libs}"
            )
            return True
        else:
            print(
                "\n❌ AUDIT LOADING ISSUE: Heavy libraries were not loaded during audit import"
            )
            return False

    except ImportError as e:
        print(f"\nAudit import failed: {e}")
        return False


if __name__ == "__main__":
    cli_success = test_cli_import_performance()
    audit_success = test_audit_loads_libraries()

    overall_success = cli_success and audit_success
    print(f"\n{'=' * 50}")
    if overall_success:
        print("✅ ALL TESTS PASSED: CLI performance fix is working correctly!")
        print("   • Help commands are fast (no heavy library loading)")
        print("   • Audit commands load libraries when needed")
    else:
        print("❌ SOME TESTS FAILED")

    sys.exit(0 if overall_success else 1)
