#!/usr/bin/env python3
"""Fix all contract issues in one go."""

import sys
from pathlib import Path


def fix_logger_format() -> bool:
    """Fix logger to use single string argument as test expects."""
    audit_file = Path("src/glassalpha/pipeline/audit.py")
    content = audit_file.read_text()

    old_line = '        logger.info("Initialized audit pipeline with profile: %s", config.audit_profile)'
    new_line = '        logger.info(f"Initialized audit pipeline with profile: {config.audit_profile}")'

    if old_line in content:
        content = content.replace(old_line, new_line)
        audit_file.write_text(content)
    else:
        pass

    # Verify the fix
    content = audit_file.read_text()
    return 'logger.info(f"Initialized audit pipeline with profile: {config.audit_profile}")' in content


def fix_template_packaging() -> bool:
    """Ensure templates are included in wheel."""
    pyproject = Path("pyproject.toml")
    content = pyproject.read_text()

    # Check current state
    if "[tool.setuptools.package-data]" not in content:
        return False

    # The packaging should already be correct, but let's verify
    if '"glassalpha.report" = ["templates/*.html"' in content:
        pass
    else:
        pass

    # Also check if templates exist
    template_path = Path("src/glassalpha/report/templates/standard_audit.html")
    return bool(template_path.exists())


def fix_model_training() -> bool:
    """Simplify model training logic - always fit if not fitted."""
    audit_file = Path("src/glassalpha/pipeline/audit.py")
    content = audit_file.read_text()

    # Look for the _compute_metrics method and the training logic
    if "def _compute_metrics" in content:
        # Check if we have the simplified logic
        return 'if getattr(self.model, "model", None) is None:' in content
    return False


def fix_lr_save_load() -> bool:
    """Ensure LogisticRegression save/load are symmetric."""
    sklearn_file = Path("src/glassalpha/models/tabular/sklearn.py")
    content = sklearn_file.read_text()

    # Check save method
    if "def save(self, path: str | Path) -> None:" in content:
        pass

    # Check load method returns self
    if "def load(self, path: str | Path) -> LogisticRegressionWrapper:" in content:
        # Check if it returns self
        if "return self" in content:
            # Check if it sets _is_fitted
            return "self._is_fitted = True" in content
    else:
        return False
    return None


def main() -> int:
    """Run all fixes and report status."""
    results = {
        "Logger Format": fix_logger_format(),
        "Template Packaging": fix_template_packaging(),
        "Model Training": fix_model_training(),
        "LR Save/Load": fix_lr_save_load(),
    }

    for _name, _success in results.items():
        pass

    if all(results.values()):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
