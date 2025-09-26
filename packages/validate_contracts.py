#!/usr/bin/env python3
"""Validate core contracts without full dependency installation."""

import sys
import zipfile
from pathlib import Path


def validate_contracts() -> int:  # noqa: C901, PLR0912
    """Validate the 4 critical contracts."""
    results = {}

    # 1. Logger Format Check
    audit_file = Path("src/glassalpha/pipeline/audit.py")
    content = audit_file.read_text()
    if 'logger.info(f"Initialized audit pipeline with profile: {config.audit_profile}")' in content:
        results["Logger"] = True
    else:
        for line in content.split("\n"):
            if "Initialized audit pipeline with profile" in line:
                pass
        results["Logger"] = False

    # 2. Template Packaging Check
    wheels = list(Path("dist").glob("glassalpha-*.whl"))
    if wheels:
        wheel = wheels[0]
        with zipfile.ZipFile(wheel, "r") as zf:
            files = zf.namelist()
            template = "glassalpha/report/templates/standard_audit.html"
            if template in files:
                results["Templates"] = True
            else:
                for f in files:
                    if "report" in f:
                        pass
                results["Templates"] = False
    else:
        results["Templates"] = False

    # 3. Model Training Logic Check
    if 'if getattr(self.model, "model", None) is None:' in content:
        if "self.model.fit(X_processed, y_true" in content:
            results["Training"] = True
        else:
            results["Training"] = False
    else:
        results["Training"] = False

    # 4. LR Save/Load Check
    sklearn_file = Path("src/glassalpha/models/tabular/sklearn.py")
    sklearn_content = sklearn_file.read_text()

    checks = {
        "Load returns self": "return self" in sklearn_content,
        "Load sets _is_fitted": "self._is_fitted = True" in sklearn_content,
        "Save includes n_classes": '"n_classes": len(getattr(self.model, "classes_"' in sklearn_content,
    }

    all_pass = all(checks.values())
    for _passed in checks.values():
        pass
    results["Save/Load"] = all_pass

    # Summary

    all_good = all(results.values())
    for _passed in results.values():
        pass

    if all_good:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(validate_contracts())
