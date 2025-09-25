#!/usr/bin/env python3
"""CI Environment Diagnosis Script.

This script diagnoses CI environment issues by testing package imports
and providing detailed error information.
"""

import os
import sys


def check_python_environment():
    """Check Python environment details."""
    print("🐍 PYTHON ENVIRONMENT")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print(f"Working Directory: {os.getcwd()}")
    print()


def check_package_installation():
    """Check if glassalpha package is properly installed."""
    print("📦 PACKAGE INSTALLATION")
    print("=" * 50)

    try:
        import glassalpha

        print("✓ glassalpha imported successfully")
        print(f"  Version: {getattr(glassalpha, '__version__', 'unknown')}")
        print(f"  Location: {glassalpha.__file__}")
    except ImportError as e:
        print(f"✗ glassalpha import failed: {e}")
        return False

    # Check submodules
    submodules = ["data", "models", "core", "config", "metrics"]
    for module in submodules:
        try:
            __import__(f"glassalpha.{module}", fromlist=[module])
            print(f"✓ glassalpha.{module} available")
        except ImportError as e:
            print(f"✗ glassalpha.{module} failed: {e}")

    print()
    return True


def check_numpy_ecosystem():
    """Check NumPy ecosystem package imports."""
    print("🔢 NUMPY ECOSYSTEM")
    print("=" * 50)

    packages = {"numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "sklearn": "sklearn"}

    for name, import_name in packages.items():
        try:
            package = __import__(import_name)
            version = getattr(package, "__version__", "unknown")
            print(f"✓ {name}: {version}")
        except ImportError as e:
            print(f"✗ {name}: Import failed - {e}")

            # Detailed error analysis for numpy
            if name == "numpy":
                print("  🚨 NumPy import failure details:")
                try:
                    import importlib

                    spec = importlib.util.find_spec("numpy")
                    if spec:
                        print(f"    - NumPy found at: {spec.origin}")
                    else:
                        print("    - NumPy spec not found")
                except Exception as spec_error:
                    print(f"    - Spec check failed: {spec_error}")

    print()


def check_specific_imports():
    """Test specific imports that are failing in CI."""
    print("🎯 SPECIFIC IMPORT TESTS")
    print("=" * 50)

    failing_imports = [
        ("numpy.__version__", "from numpy import __version__"),
        ("scipy.sparse", "from scipy.sparse import csr_matrix"),
        ("sklearn.datasets", "from sklearn.datasets import make_classification"),
        ("glassalpha.data", "from glassalpha.data import TabularDataLoader"),
        ("glassalpha.pipeline", "from glassalpha.pipeline.audit import AuditPipeline"),
    ]

    for description, import_statement in failing_imports:
        try:
            exec(import_statement)
            print(f"✓ {description}: Success")
        except ImportError as e:
            print(f"✗ {description}: {e}")
        except Exception as e:
            print(f"✗ {description}: Unexpected error - {e}")

    print()


def check_pip_environment():
    """Check pip list and installation status."""
    print("📋 PIP ENVIRONMENT")
    print("=" * 50)

    try:
        import subprocess

        result = subprocess.run(["pip", "list"], capture_output=True, text=True)

        if result.returncode == 0:
            # Filter for relevant packages
            lines = result.stdout.split("\n")
            relevant_packages = ["numpy", "scipy", "pandas", "scikit-learn", "sklearn", "xgboost", "glassalpha"]

            print("Relevant installed packages:")
            for line in lines:
                for pkg in relevant_packages:
                    if line.lower().startswith(pkg.lower()):
                        print(f"  {line}")
                        break
        else:
            print(f"pip list failed: {result.stderr}")

    except Exception as e:
        print(f"Could not run pip list: {e}")

    print()


def main():
    """Main diagnosis function."""
    print("🏥 CI ENVIRONMENT DIAGNOSIS")
    print("=" * 60)
    print()

    check_python_environment()

    # Check if package is installed first
    package_ok = check_package_installation()

    check_numpy_ecosystem()
    check_specific_imports()
    check_pip_environment()

    print("🏁 DIAGNOSIS COMPLETE")
    print("=" * 60)

    if not package_ok:
        print("❌ CRITICAL: glassalpha package not properly installed")
        sys.exit(1)
    else:
        print("✅ glassalpha package installation appears OK")


if __name__ == "__main__":
    main()
