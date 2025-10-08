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


@pytest.fixture(autouse=True)
def isolate_tests(request):
    """Ensure test isolation by cleaning up between tests."""
    import gc
    import tempfile

    # Skip isolation for tests that explicitly test module loading
    if "lazy_load" in request.node.name or "import" in request.node.name.lower():
        yield
        return

    # Save original temp directory
    original_tempdir = tempfile.gettempdir()

    yield  # Run the test

    # Force garbage collection to clean up any cached objects
    gc.collect()

    # Restore temp directory
    tempfile.tempdir = original_tempdir

    # Clear any test-specific environment variables
    test_env_vars = [k for k in os.environ if k.startswith("TEST_") or k.startswith("GLASSALPHA_TEST_")]
    for var in test_env_vars:
        os.environ.pop(var, None)
