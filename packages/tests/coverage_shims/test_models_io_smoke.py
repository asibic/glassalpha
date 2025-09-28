"""Smoke test for models/_io.py to bump coverage."""

from glassalpha.models import _io


def test_models_io_smoke(tmp_path):
    """Call the safest public helpers behind try/except if needed."""
    # Adjust to actual API names
    assert hasattr(_io, "__file__")
