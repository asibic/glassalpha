#!/usr/bin/env python3
"""Diagnostic script to check wheel vs source build status.

Run this script to identify if packages were built from source vs wheels.
This helps debug CI issues where source builds cause failures.
"""

import importlib.metadata as im
import sys


def check_package_build_type(name: str) -> str:
    """Check if package was installed from wheel or built from source."""
    try:
        dist = im.distribution(name)
        if dist and dist.files:
            # Check if this looks like a wheel installation
            # Wheels typically have more files and specific structure
            files = dist.files
            if len(files) > 10:  # Wheels usually have many files
                return "wheel"
            return "source"
        return "unknown"
    except Exception as e:
        return f"error: {e}"


def main():
    """Main diagnostic function."""
    print(f"Python: {sys.platform} {sys.version}")

    packages_to_check = [
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "shap",
        "xgboost",
        "lightgbm",
    ]

    print("\nPackage build analysis:")
    print("=" * 50)

    for name in packages_to_check:
        try:
            dist = im.distribution(name)
            build_type = check_package_build_type(name)
            print(f"{name:<15} {dist.version if dist else 'MISSING':<10} {build_type}")
        except Exception as e:
            print(f"{name:<15} ERROR: {e}")


if __name__ == "__main__":
    main()
