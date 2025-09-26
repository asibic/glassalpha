"""Contract tests for constants and resource packaging.

These tests ensure that exact contract strings never drift and that
wheel packaging correctly includes template resources.
"""

import pytest


def test_exact_contract_strings():
    """Ensure contract strings never drift from expected values."""
    from glassalpha.constants import NO_EXPLAINER_MSG, NO_MODEL_MSG

    # Exact strings required by tests and error handling
    assert NO_MODEL_MSG == "Model not loaded. Load a model first."
    assert NO_EXPLAINER_MSG == "No compatible explainer found"


def test_wheel_template_resources_available():
    """Ensure wheel packaging includes template resources."""
    try:
        from importlib.resources import files
    except ImportError:
        pytest.skip("importlib.resources not available")

    # Check that standard audit template is packaged
    template_files = files("glassalpha.report.templates")
    standard_template = template_files.joinpath("standard_audit.html")

    assert standard_template.is_file(), (
        "standard_audit.html template not found in wheel. Check pyproject.toml package-data and MANIFEST.in"
    )


def test_constants_module_exports():
    """Ensure all required constants are properly exported."""
    from glassalpha.constants import BINARY_CLASSES, BINARY_THRESHOLD, INIT_LOG_MESSAGE

    # Check that key constants exist and have expected types
    assert isinstance(INIT_LOG_MESSAGE, str)
    assert BINARY_CLASSES == 2
    assert BINARY_THRESHOLD == 0.5


def test_backward_compatible_aliases():
    """Ensure backward-compatible aliases are maintained."""
    from glassalpha.constants import ERR_NO_EXPLAINER, ERR_NOT_LOADED

    # These aliases should still work for existing code
    assert isinstance(ERR_NOT_LOADED, str)
    assert isinstance(ERR_NO_EXPLAINER, str)
