"""Basic package tests."""

import glassalpha


def test_version():
    """Ensure version is accessible."""
    assert hasattr(glassalpha, "__version__")
    assert glassalpha.__version__ == "0.1.0"
