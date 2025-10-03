# packages/tests/conftest.py
import os
import sys

import pytest

# Force non-interactive backend for *all* tests before anything imports pyplot
os.environ.setdefault("MPLBACKEND", "Agg")

# Optional: silence tqdm monitor threads in tests (prevents noisy shutdowns)
os.environ.setdefault("TQDM_DISABLE", "1")


# Pytest markers for platform-specific tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "linux_only: marks tests that should only run on Linux (e.g., PDF rendering with WeasyPrint)",
    )


# Skip decorator for Linux-only tests
linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="PDF rendering via WeasyPrint is verified on Linux in CI.",
)
