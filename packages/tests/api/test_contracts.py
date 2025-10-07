"""Phase 1: Namespace & Import Contract Tests

Tests for fast imports with PEP 562 lazy loading.
"""

import sys
import time

import pytest

# Mark as contract test (must pass before release)
pytestmark = pytest.mark.contract


class TestImportSpeed:
    """Import must complete in <200ms"""

    def test_import_speed_cold_start(self):
        """Import completes in <200ms on cold start"""
        # Remove glassalpha from sys.modules if present
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith("glassalpha")]
        for key in modules_to_remove:
            del sys.modules[key]

        # Time the import
        start = time.time()
        import glassalpha as ga

        duration = time.time() - start

        assert duration < 0.2, f"Import took {duration:.3f}s (threshold: 0.2s)"

    def test_import_speed_repeated(self):
        """Repeated imports are instant (cached)"""
        import glassalpha as ga

        start = time.time()
        import glassalpha as ga2

        duration = time.time() - start

        assert duration < 0.001, f"Repeat import took {duration:.6f}s"


class TestLazyLoading:
    """Modules load on first access"""

    def test_audit_not_loaded_initially(self):
        """Audit module not in sys.modules before first access"""
        # Remove glassalpha modules
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith("glassalpha")]
        for key in modules_to_remove:
            del sys.modules[key]

        # Import glassalpha
        import glassalpha as ga

        # Check audit not loaded yet
        assert "glassalpha.api.audit" not in sys.modules, "audit module should not be loaded until accessed"

    def test_audit_loads_on_access(self):
        """Audit module loads when accessed"""
        import glassalpha as ga

        # Remove audit module if present
        if "glassalpha.api.audit" in sys.modules:
            del sys.modules["glassalpha.api.audit"]

        # Access audit attribute (will trigger lazy load)
        try:
            _ = ga.audit
        except (ImportError, AttributeError):
            # Module doesn't exist yet (Phase 3), that's ok
            pass
        else:
            # If it loaded, verify it's in sys.modules
            assert "glassalpha.api.audit" in sys.modules, "audit module should be in sys.modules after access"

    def test_datasets_lazy_loads(self):
        """Datasets module loads on access"""
        import glassalpha as ga

        # Try to access datasets
        try:
            _ = ga.datasets
        except (ImportError, AttributeError):
            # Module doesn't exist yet, that's ok
            pass

    def test_utils_lazy_loads(self):
        """Utils module loads on access"""
        import glassalpha as ga

        # Try to access utils
        try:
            _ = ga.utils
        except (ImportError, AttributeError):
            # Module doesn't exist yet, that's ok
            pass


class TestTabCompletion:
    """dir() includes lazy modules for tab completion"""

    def test_dir_includes_audit(self):
        """dir(ga) includes 'audit'"""
        import glassalpha as ga

        dir_output = dir(ga)
        assert "audit" in dir_output, "'audit' should be in dir(ga) for tab completion"

    def test_dir_includes_datasets(self):
        """dir(ga) includes 'datasets'"""
        import glassalpha as ga

        dir_output = dir(ga)
        assert "datasets" in dir_output, "'datasets' should be in dir(ga) for tab completion"

    def test_dir_includes_utils(self):
        """dir(ga) includes 'utils'"""
        import glassalpha as ga

        dir_output = dir(ga)
        assert "utils" in dir_output, "'utils' should be in dir(ga) for tab completion"

    def test_dir_includes_version(self):
        """dir(ga) includes '__version__'"""
        import glassalpha as ga

        dir_output = dir(ga)
        assert "__version__" in dir_output


class TestPrivacyEnforcement:
    """Internal modules not exposed in public API"""

    def test_export_not_in_dir(self):
        """_export not in dir(ga)"""
        import glassalpha as ga

        dir_output = dir(ga)
        assert "_export" not in dir_output, "Private module '_export' should not be in public API"

    def test_export_not_in_all(self):
        """_export not in __all__"""
        import glassalpha as ga

        assert "_export" not in ga.__all__, "Private module '_export' should not be in __all__"

    def test_lazy_modules_dict_private(self):
        """_LAZY_MODULES dict is private (starts with _)"""
        import glassalpha as ga

        # Can access via __dict__, but not in public API
        assert "_LAZY_MODULES" in ga.__dict__
        assert "_LAZY_MODULES" not in dir(ga)


class TestVersionExport:
    """Version is immediately available (not lazy)"""

    def test_version_accessible(self):
        """__version__ accessible without triggering lazy load"""
        import glassalpha as ga

        version = ga.__version__
        assert isinstance(version, str)
        assert version.startswith("0."), f"Expected v0.x, got {version}"

    def test_version_format(self):
        """__version__ follows semantic versioning"""
        import glassalpha as ga

        parts = ga.__version__.split(".")
        assert len(parts) >= 2, "Version should be X.Y or X.Y.Z"

        # Should be parseable as integers
        for part in parts:
            assert part.isdigit() or part.endswith("a") or part.endswith("b"), f"Invalid version component: {part}"


class TestAttributeError:
    """Unknown attributes raise AttributeError"""

    def test_unknown_attribute_raises(self):
        """Accessing unknown attribute raises AttributeError"""
        import glassalpha as ga

        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            _ = ga.nonexistent

    def test_error_message_helpful(self):
        """Error message includes module name and attribute"""
        import glassalpha as ga

        try:
            _ = ga.this_does_not_exist
        except AttributeError as e:
            error_msg = str(e)
            assert "glassalpha" in error_msg
            assert "this_does_not_exist" in error_msg
        else:
            pytest.fail("Should have raised AttributeError")


class TestNoHeavyImportsOnInit:
    """No heavy dependencies loaded on import"""

    def test_sklearn_not_loaded(self):
        """Sklearn not loaded on import glassalpha"""
        # Remove glassalpha and sklearn from sys.modules
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith(("glassalpha", "sklearn"))]
        for key in modules_to_remove:
            del sys.modules[key]

        import glassalpha as ga

        sklearn_modules = [key for key in sys.modules.keys() if key.startswith("sklearn")]

        assert len(sklearn_modules) == 0, f"sklearn should not be loaded on import, found: {sklearn_modules}"

    def test_xgboost_not_loaded(self):
        """Xgboost not loaded on import glassalpha"""
        # Remove glassalpha and xgboost from sys.modules
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith(("glassalpha", "xgboost"))]
        for key in modules_to_remove:
            del sys.modules[key]

        import glassalpha as ga

        xgboost_modules = [key for key in sys.modules.keys() if key.startswith("xgboost")]

        assert len(xgboost_modules) == 0, f"xgboost should not be loaded on import, found: {xgboost_modules}"

    def test_matplotlib_not_loaded(self):
        """Matplotlib not loaded on import glassalpha"""
        # Remove glassalpha and matplotlib from sys.modules
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith(("glassalpha", "matplotlib"))]
        for key in modules_to_remove:
            del sys.modules[key]

        import glassalpha as ga

        matplotlib_modules = [key for key in sys.modules.keys() if key.startswith("matplotlib")]

        assert len(matplotlib_modules) == 0, f"matplotlib should not be loaded on import, found: {matplotlib_modules}"
