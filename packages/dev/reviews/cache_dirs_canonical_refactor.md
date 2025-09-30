# Cache Directory Canonical Path Refactor

**Date:** September 30, 2025
**Status:** ✅ Complete
**Type:** Architecture Improvement

## Summary

Refactored the cache directory resolution system to use canonical paths throughout, eliminating edge cases around symlinks and OS-specific path mappings (e.g., `/tmp` → `/private/tmp` on macOS).

## Problem Statement

The original implementation had inconsistent path handling:

- Some paths were canonicalized, others were not
- Tests failed on macOS due to `/tmp` vs `/private/tmp` differences
- Users couldn't see path transformations, causing confusion
- Mixed pure and side-effect functions made testing difficult

## Solution

**Canonical Path Everywhere:**

1. `resolve_data_root()` always returns canonical absolute paths via `.resolve()`
2. No filesystem I/O in path resolution (pure function)
3. All paths (env overrides, platformdirs, fallbacks) canonicalized consistently
4. Logging shows both requested and effective paths for transparency

**Separation of Concerns:**

- `resolve_data_root()`: Pure path resolution, no side effects
- `ensure_dir_writable()`: Side effects (mkdir, chmod, write test)

## Changes Made

### 1. Core Module: `src/glassalpha/utils/cache_dirs.py`

**Before:**

```python
def resolve_data_root() -> Path:
    env_override = os.environ.get("GLASSALPHA_DATA_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    # ... rest of logic
```

**After:**

```python
def resolve_data_root() -> Path:
    """
    Return a canonical absolute path for the cache root.
    No filesystem writes here.
    """
    raw_env = os.getenv("GLASSALPHA_DATA_DIR")
    if raw_env:
        # Expand and canonicalize even if the final directory doesn't exist yet.
        return Path(raw_env).expanduser().resolve()
    # ... rest of logic (all paths .resolve()'d)
```

### 2. Logging: `src/glassalpha/pipeline/audit.py` & `src/glassalpha/cli/datasets.py`

Added transparent logging:

```python
raw_env = os.getenv("GLASSALPHA_DATA_DIR")
effective = resolve_data_root()

if raw_env:
    logger.info(f"Cache dir requested via GLASSALPHA_DATA_DIR: {raw_env} | effective: {effective}")
else:
    logger.info(f"Cache dir (default): {effective}")
```

### 3. Tests: `tests/test_cache_dirs.py`

**Updated all assertions to compare canonical paths:**

```python
# Before
assert cache_root == Path("/tmp/test")

# After
assert cache_root == Path("/tmp/test").resolve()
```

**Fixed platform-specific tests:**

```python
# Mock platformdirs to test fallback paths
with patch("platform.system", return_value="Linux"):
    with patch("glassalpha.utils.cache_dirs.user_data_dir", None):
        cache_root = resolve_data_root()
        expected = Path("/custom/xdg/glassalpha/data")
        assert cache_root == expected.resolve()
```

**Fixed permission error tests:**

```python
# Test subdirectory creation failure in readonly parent
readonly_dir = tmp_path / "readonly"
readonly_dir.mkdir()
readonly_dir.chmod(0o555)  # Read-only, no write

target_dir = readonly_dir / "subdir"  # This should fail
with pytest.raises(RuntimeError, match="Cannot create or write to cache directory"):
    ensure_dir_writable(target_dir)
```

### 4. Documentation

**Module docstring:**

```python
"""
Cache Path Resolution
---------------------
GlassAlpha canonicalizes all cache directory paths to absolute real paths
(symlinks resolved). This applies to:
- Environment variable overrides (GLASSALPHA_DATA_DIR)
- Platform-specific defaults (via platformdirs when available)
- Manual OS-specific fallbacks

Example:
    >>> os.environ['GLASSALPHA_DATA_DIR'] = '/tmp/my-cache'
    >>> resolve_data_root()
    PosixPath('/private/tmp/my-cache')  # On macOS
"""
```

**README.md section:**

```markdown
**Cache Directory Resolution:**

- Default: OS-specific user data directory
- Override: Set `GLASSALPHA_DATA_DIR` environment variable
- All paths are canonicalized (symlinks resolved) for consistency
- The system logs both requested and effective paths for transparency
```

## Results

### Test Results

```
============================= test session starts ==============================
tests/test_cache_dirs.py::TestGetDataRoot::test_env_override_uses_realpath PASSED
tests/test_cache_dirs.py::TestGetDataRoot::test_windows_defaults_use_realpath PASSED
tests/test_cache_dirs.py::TestGetDataRoot::test_linux_defaults_use_realpath PASSED
tests/test_cache_dirs.py::TestGetDataRoot::test_env_override_canonicalize PASSED
tests/test_cache_dirs.py::TestGetDataRoot::test_cache_directory_creation_canonical PASSED
tests/test_cache_dirs.py::TestGetDataRoot::test_cache_directory_permission_error_canonical PASSED
tests/test_cache_dirs.py::TestGetCachePath::test_cache_path_construction_canonical PASSED
tests/test_cache_dirs.py::TestGetCachePath::test_cache_path_different_dataset_canonical PASSED
tests/test_cache_dirs.py::TestEnsureDirWritable::test_ensure_dir_writable_creates_directory PASSED
tests/test_cache_dirs.py::TestEnsureDirWritable::test_ensure_dir_writable_with_existing_directory PASSED
tests/test_cache_dirs.py::TestEnsureDirWritable::test_ensure_dir_writable_creates_parent_directories PASSED
tests/test_cache_dirs.py::TestEnsureDirWritable::test_ensure_dir_writable_tests_writability PASSED
tests/test_cache_dirs.py::TestEnsureDirWritable::test_ensure_dir_writable_permission_error PASSED

============================== 13 passed in 0.21s ==============================
```

### End-to-End Verification

```
✅ Environment /tmp/test-final → /private/tmp/test-final
   Canonical: True

✅ Dataset fetched successfully: True
   Location: /private/tmp/test-final/german_credit_processed.csv

🎯 Summary:
  ✓ Paths are canonicalized (symlinks resolved)
  ✓ Tests compare realpaths
  ✓ Logging shows requested → effective
  ✓ No side-effects in resolve_data_root()
```

## Benefits

| Aspect              | Before                       | After                          |
| ------------------- | ---------------------------- | ------------------------------ |
| **Simplicity**      | Mixed literal/resolved paths | One rule: always canonicalize  |
| **Predictability**  | Path forms varied            | Consistent canonical form      |
| **Test friction**   | macOS /tmp issues            | Compare realpaths, no issues   |
| **User experience** | Only saw effective path      | See both requested & effective |
| **Testability**     | Side effects in resolution   | Pure function, easy to test    |

## Backwards Compatibility

The deprecated `get_data_root()` function remains available with a deprecation warning:

```python
def get_data_root() -> Path:
    """DEPRECATED: Use resolve_data_root() directly."""
    import warnings
    warnings.warn(
        "get_data_root() is deprecated. Use resolve_data_root() and handle directory creation at call sites",
        DeprecationWarning,
        stacklevel=2
    )
    root = resolve_data_root()
    return ensure_dir_writable(root)
```

## Files Modified

- `src/glassalpha/utils/cache_dirs.py` - Core refactor
- `src/glassalpha/pipeline/audit.py` - Added logging
- `src/glassalpha/cli/datasets.py` - Added logging
- `tests/test_cache_dirs.py` - Updated all tests
- `README.md` - Added dataset management documentation

## Migration Guide

**For code using the old API:**

```python
# Old
from glassalpha.utils.cache_dirs import get_data_root
cache = get_data_root()  # Creates directory

# New
from glassalpha.utils.cache_dirs import resolve_data_root, ensure_dir_writable
cache = ensure_dir_writable(resolve_data_root())  # Explicit side effects
```

**For tests:**

```python
# Old
assert path == Path("/tmp/cache")

# New
assert path == Path("/tmp/cache").resolve()  # Compare canonical paths
```

## Next Steps

1. Monitor deprecation warnings in logs
2. Consider removing `get_data_root()` in next major version
3. Apply same pattern to other path-handling code if needed

## References

- [PEP 519 - Adding a file system path protocol](https://www.python.org/dev/peps/pep-0519/)
- [pathlib.Path.resolve() documentation](https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve)
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)
