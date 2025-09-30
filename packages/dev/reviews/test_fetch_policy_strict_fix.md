# Test Fetch Policy Fix: Strict Dataset Policy Compliance

**Date:** September 30, 2025
**Status:** ✅ Complete
**Type:** Test Fix

## Problem

The `test_fetch_policy.py` file was written before the strict dataset policy was enforced. It used `dataset="test_dataset"` (a registry dataset) with `path=...` which violates the new strict rule:

> **Registry datasets cannot specify custom paths. Only `dataset="custom"` can use `path`.**

**Error:**

```
ValidationError: data.path must be omitted for registry dataset 'test_dataset'.
Use data.dataset='custom' to provide a custom path.
```

## Root Cause

The tests were conceptually testing custom file behavior (user-provided paths with different fetch policies) but using registry dataset syntax. This created ambiguity:

- Is this testing registry dataset fetch logic?
- Is this testing custom file fetch logic?
- Why does it need both `dataset` and `path`?

## Solution

Rewrote the entire test file to use `dataset="custom"` which is the correct pattern for testing user-provided files with fetch policies.

### Before (Ambiguous)

```python
config = DataConfig(
    dataset="test_dataset",  # Registry dataset
    path=str(existing_file),  # But custom path?
    fetch="never",
    offline=False,
)
```

**Problems:**

- Violates strict policy (registry + path)
- Unclear what's being tested
- Doesn't match real user patterns

### After (Clear)

```python
config = DataConfig(
    dataset="custom",         # Explicit: custom file
    path=str(existing_file),  # User's file path
    fetch="never",
    offline=False,
)
```

**Benefits:**

- Follows strict policy
- Clear intent: testing custom files
- Matches real user patterns
- Tests what users will actually do

## Changes Made

### File: `tests/test_fetch_policy.py`

**1. Updated Module Docstring**

```python
"""Tests for fetch policy behavior with custom datasets.

Note: These tests use dataset='custom' because fetch policies primarily
apply to custom user-provided files. Registry datasets have their own
fetch logic tested in test_concurrency_fetch.py.
"""
```

**2. Removed Registry Setup**

**Before:**

```python
def setup_method(self):
    """Set up test environment."""
    REGISTRY.clear()
    REGISTRY["test_dataset"] = DatasetSpec(...)

def teardown_method(self):
    REGISTRY.clear()
```

**After:**

```python
# No setup/teardown needed for custom dataset tests
```

**3. Updated All Test Cases**

Changed from `dataset="test_dataset"` to `dataset="custom"` in:

- `test_custom_fetch_never_with_existing_file`
- `test_custom_fetch_never_with_missing_file`
- `test_custom_fetch_if_missing_with_existing_file`
- `test_custom_offline_with_missing_file`
- `test_fetch_policy_validation`
- `test_fetch_policy_defaults`

**4. Added Policy Enforcement Test**

```python
def test_registry_dataset_forbids_path(self):
    """Test that registry datasets cannot specify custom paths."""
    with pytest.raises(ValueError, match="data.path must be omitted"):
        DataConfig(
            dataset="german_credit",  # Registry dataset
            path="/tmp/my_custom_path.csv",  # Not allowed
            fetch="if_missing",
        )
```

**5. Fixed Validation Test**

Changed from `ValueError` to `ValidationError` for Pydantic v2 compatibility:

```python
from pydantic import ValidationError

with pytest.raises(ValidationError, match="String should match pattern"):
    DataConfig(
        dataset="custom",
        path="/tmp/test.csv",
        fetch="invalid_policy",  # Invalid
        offline=False,
    )
```

## Test Coverage

### What These Tests Now Cover

| Test                                              | What It Verifies                                       |
| ------------------------------------------------- | ------------------------------------------------------ |
| `test_custom_fetch_never_with_existing_file`      | Custom file + fetch=never returns file without network |
| `test_custom_fetch_never_with_missing_file`       | Custom file + fetch=never + missing file raises error  |
| `test_custom_fetch_if_missing_with_existing_file` | Custom file + fetch=if_missing returns existing file   |
| `test_custom_offline_with_missing_file`           | Custom file + offline=true raises clear error          |
| `test_fetch_policy_validation`                    | Invalid fetch values are rejected                      |
| `test_fetch_policy_defaults`                      | fetch defaults to "if_missing"                         |
| `test_registry_dataset_forbids_path`              | Strict policy: registry + path fails                   |

### What's Tested Elsewhere

- **Registry dataset fetch logic** → `test_concurrency_fetch.py`
- **Cache directory resolution** → `test_cache_dirs.py`
- **File locking** → `test_concurrency_fetch.py`

## Test Results

```
============================= test session starts ==============================
platform darwin -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0
tests/test_fetch_policy.py::TestFetchPolicyCustomDatasets::test_custom_fetch_never_with_existing_file PASSED
tests/test_fetch_policy.py::TestFetchPolicyCustomDatasets::test_custom_fetch_never_with_missing_file PASSED
tests/test_fetch_policy.py::TestFetchPolicyCustomDatasets::test_custom_fetch_if_missing_with_existing_file PASSED
tests/test_fetch_policy.py::TestFetchPolicyCustomDatasets::test_custom_offline_with_missing_file PASSED
tests/test_fetch_policy.py::TestFetchPolicyCustomDatasets::test_fetch_policy_validation PASSED
tests/test_fetch_policy.py::TestFetchPolicyCustomDatasets::test_fetch_policy_defaults PASSED
tests/test_fetch_policy.py::TestFetchPolicyCustomDatasets::test_registry_dataset_forbids_path PASSED

============================== 7 passed in 0.59s ================================
```

## Design Benefits

### 1. Clear Separation of Concerns

**Custom Datasets (this file):**

- User-provided files
- Explicit paths
- Fetch policies apply to existence checks

**Registry Datasets (other files):**

- Built-in datasets
- Cache management
- Fetch policies apply to download/cache logic

### 2. Matches User Patterns

Tests now match what users will actually do:

```yaml
# User config for custom file
data:
  dataset: custom
  path: /my/data/file.csv
  fetch: if_missing
```

### 3. Enforces Strict Policy

New test explicitly verifies the strict rule:

```python
# This should fail
DataConfig(
    dataset="german_credit",  # Registry
    path="/custom/path.csv",  # Not allowed!
)
```

## Learning from This Fix

### Why the Original Tests Were Wrong

1. **Pre-dated strict policy** - Written when path-only or mixed patterns were allowed
2. **Conceptual confusion** - Testing custom file behavior with registry syntax
3. **Didn't match reality** - Users don't provide paths for registry datasets

### Why the New Tests Are Right

1. **Follow strict policy** - `dataset="custom"` + `path` is explicit
2. **Clear intent** - Obviously testing custom file behavior
3. **Match user patterns** - Tests what users will actually configure

### Preventing Future Issues

**Rule of Thumb:**

- Testing custom files? → Use `dataset="custom"` + `path`
- Testing registry datasets? → Use `dataset="german_credit"` (no path)
- Testing policy enforcement? → Test both patterns and verify errors

## Conclusion

This fix completes the strict dataset policy enforcement by ensuring all tests follow the new rules. The test file now:

1. **Tests the right thing** - Custom file fetch policies
2. **Uses the right pattern** - `dataset="custom"` + `path`
3. **Matches user reality** - What users will actually configure
4. **Enforces the policy** - Includes test for policy violation

The codebase now has complete consistency: config model, pipeline logic, documentation, and tests all follow the same strict, clear rule set.
