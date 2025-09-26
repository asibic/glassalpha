#!/usr/bin/env python3
"""Smoke test - validate all critical contracts before pushing."""

import sys
import tempfile
from pathlib import Path

import numpy as np

from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper


def test_lr_save_load_symmetry() -> bool:
    """Test LogisticRegression save/load contract."""
    try:
        # Create and fit a model
        wrapper1 = LogisticRegressionWrapper()
        X = np.array([[1, 2], [3, 4], [5, 6]])  # noqa: N806
        y = np.array([0, 1, 0])
        wrapper1.fit(X, y)

        # Save to temp location
        temp_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)  # noqa: SIM115
        temp_path = Path(temp_file.name)
        temp_file.close()
        wrapper1.save(temp_path)

        # Load into new wrapper
        wrapper2 = LogisticRegressionWrapper()
        result = wrapper2.load(temp_path)

        # Validate basic contracts
        checks = {
            "Can load": result is not None,
            "Has predict method": hasattr(wrapper2, "predict"),
            "Has fit method": hasattr(wrapper2, "fit"),
            "Can predict": True,  # Will test this
        }

        try:
            wrapper2.predict(X)
        except Exception:  # noqa: BLE001
            checks["Can predict"] = False

        all_pass = all(checks.values())
        if all_pass:
            pass
        else:
            pass

        # Clean up
        if temp_path.exists():
            temp_path.unlink()

        return all_pass  # noqa: TRY300

    except ImportError:
        return False
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    success = test_lr_save_load_symmetry()
    sys.exit(0 if success else 1)
