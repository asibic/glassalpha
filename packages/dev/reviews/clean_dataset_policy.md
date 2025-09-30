# Clean Dataset Policy Implementation

**Date:** September 30, 2025
**Status:** ✅ Complete
**Type:** Architecture Simplification

## Summary

Implemented a clean, strict dataset policy that eliminates backward compatibility complexity and establishes clear, simple rules for data configuration.

## The Clean Policy

### Three Simple Rules

1. **`data.dataset` is required for any real run**
2. **Use `data.dataset="custom"` with `data.path` for external files**
3. **`data_schema` alone is allowed for schema-only utilities/tests**

### Valid Configurations

| Pattern              | Usage               | Example                                           |
| -------------------- | ------------------- | ------------------------------------------------- |
| **Registry dataset** | Built-in datasets   | `dataset: "german_credit"`                        |
| **Custom dataset**   | User-provided files | `dataset: "custom"<br/>path: "/path/to/data.csv"` |
| **Schema-only**      | Tests/utilities     | `data_schema: {...}`                              |

### Invalid Configurations

| Pattern                 | Error                                         |
| ----------------------- | --------------------------------------------- |
| Path without dataset    | `data.dataset is required`                    |
| Registry dataset + path | `data.path must be omitted`                   |
| Custom without path     | `data.path is required when dataset='custom'` |
| Empty config            | `data.dataset is required`                    |

## Implementation

### A. Config Model (`src/glassalpha/config/schema.py`)

**Validation Logic:**

```python
@model_validator(mode="after")
def enforce_dataset_policy(self) -> "DataConfig":
    """Enforce clean dataset policy.

    Rules:
    1. Schema-only configs are allowed (no dataset/path needed)
    2. For actual data use, dataset is required
    3. dataset="custom" requires path; other datasets forbid path
    """
    # Schema-only configs are allowed (no dataset/path needed)
    if self.data_schema is not None and not (self.dataset or self.path):
        return self

    # For any actual data use, dataset is required
    if not self.dataset:
        raise ValueError(
            "data.dataset is required. "
            "Use data.dataset='custom' with data.path for external files."
        )

    # Custom dataset requires path
    if self.dataset == "custom":
        if not self.path:
            raise ValueError("data.path is required when data.dataset='custom'.")
    else:
        # Registry datasets forbid path
        if self.path:
            raise ValueError(
                f"data.path must be omitted for registry dataset '{self.dataset}'. "
                "Use data.dataset='custom' to provide a custom path."
            )

    return self
```

**Field Definitions:**

```python
class DataConfig(BaseModel):
    # Dataset specification (required for runs; "custom" enables path)
    dataset: str | None = Field(None, description="Dataset key from registry or 'custom' for external files")

    # Path specification (only valid when dataset == "custom")
    path: str | None = Field(None, description="Path to data file (only when dataset='custom')")

    # Fetch policy for automatic dataset downloading
    fetch: str = Field("if_missing", ...)

    # Offline mode (disables network operations)
    offline: bool = Field(False, ...)

    # Schema specification (for validation utilities and tests)
    data_schema: dict[str, Any] | None = Field(None, ...)
```

### B. Pipeline (`src/glassalpha/pipeline/audit.py`)

**Simplified Path Resolution:**

```python
def _resolve_requested_path(self) -> Path:
    """Resolve the requested data path from configuration."""
    from ..datasets.registry import REGISTRY
    from ..utils.cache_dirs import resolve_data_root

    cfg = self.config.data

    # Custom dataset: user provides explicit path
    if cfg.dataset == "custom":
        return Path(cfg.path).expanduser().resolve()

    # Built-in dataset: resolve to canonical cache file
    spec = REGISTRY.get(cfg.dataset)
    if not spec:
        raise ValueError(f"Unknown dataset key: {cfg.dataset}")

    cache_root = resolve_data_root()
    return (cache_root / spec.default_relpath).resolve()
```

**Simplified Dataset Availability:**

```python
def _ensure_dataset_availability(self, requested_path: Path) -> Path:
    """Ensure dataset is available, fetching if necessary."""
    cfg = self.config.data
    ds_key = cfg.dataset

    # Custom dataset: user path, just ensure parent exists if needed
    if ds_key == "custom":
        if not requested_path.exists():
            if cfg.offline:
                raise FileNotFoundError("Data file not found and offline is true.")
            raise FileNotFoundError("Custom data file not found. Please provide the file.")
        return requested_path

    # Built-in dataset: ensure cache exists, then mirror if needed
    spec = REGISTRY[ds_key]
    cache_root = ensure_dir_writable(resolve_data_root())
    final_cache_path = (cache_root / spec.default_relpath).resolve()

    # ... fetch and mirror logic ...
```

### C. Updated Tests

**Test File Renamed:**
`tests/test_deprecation_path_only.py` → Tests clean policy enforcement

**New Test Cases:**

```python
class TestDatasetPolicy:
    """Test clean dataset policy: dataset required, custom enables path."""

    def test_path_only_config_validation(self):
        """Path without dataset should fail."""
        with pytest.raises(ValueError, match="data.dataset is required"):
            DataConfig(path="/tmp/test.csv")

    def test_path_with_registry_dataset_fails(self):
        """Registry dataset + path should fail."""
        with pytest.raises(ValueError, match="data.path must be omitted"):
            DataConfig(dataset="german_credit", path="/tmp/test.csv")

    def test_custom_dataset_with_path_works(self):
        """Custom dataset + path should work."""
        config = DataConfig(dataset="custom", path="/tmp/test.csv")
        assert config.dataset == "custom"
        assert config.path == "/tmp/test.csv"

    def test_custom_dataset_without_path_fails(self):
        """Custom dataset without path should fail."""
        with pytest.raises(ValueError, match="data.path is required"):
            DataConfig(dataset="custom")
```

## Benefits

### Before vs After

| Aspect                   | Before (Permissive)                      | After (Clean)                                 |
| ------------------------ | ---------------------------------------- | --------------------------------------------- |
| **Mental model**         | Complex: dataset OR path OR schema       | Simple: dataset required, custom enables path |
| **Code branches**        | Many: handle path-only, both, neither    | Few: handle custom vs registry                |
| **Deprecation warnings** | Multiple paths, warnings, legacy support | One path, clear errors                        |
| **Error messages**       | Vague: "provide dataset or path"         | Specific: "use dataset='custom'"              |
| **User confusion**       | "Do I use dataset or path?"              | "Use dataset; custom if external file"        |
| **Test complexity**      | Mock both patterns, handle warnings      | Test clean rules                              |

### Code Complexity Reduction

**Lines Removed:**

- Deprecation warning logic
- Path-only handling branches
- Legacy compatibility code
- Duplicate validation checks

**Lines Simplified:**

- Path resolution: 30 → 15 lines
- Dataset availability: 150 → 90 lines
- Validation: Multiple checks → Single clear policy

## Migration Guide (for future reference)

If users had old configs (not applicable now, but documented for posterity):

### Old Config Pattern

```yaml
data:
  path: /path/to/data.csv
  target_column: outcome
```

### New Config Pattern

```yaml
data:
  dataset: custom
  path: /path/to/data.csv
  target_column: outcome
```

**Error Message Guides Migration:**

```
ValueError: data.dataset is required.
Use data.dataset='custom' with data.path for external files.
```

## Test Results

```
34 tests passed ✅

Policy validation tests:
✅ Registry dataset (german_credit)
✅ Custom dataset with path
✅ Schema-only config
✅ Path without dataset fails correctly
✅ Registry dataset + path fails correctly
✅ Custom without path fails correctly
✅ Empty config fails correctly
```

## Files Modified

1. **`src/glassalpha/config/schema.py`**

   - Implemented clean dataset policy in `enforce_dataset_policy()`
   - Updated field descriptions for clarity
   - Added `expand_user_path()` validator

2. **`src/glassalpha/pipeline/audit.py`**

   - Simplified `_resolve_requested_path()` (30 → 15 lines)
   - Simplified `_ensure_dataset_availability()` (150 → 90 lines)
   - Removed redundant validation checks

3. **`tests/test_deprecation_path_only.py`** (renamed from deprecation focus)

   - Updated to test clean policy enforcement
   - Added test cases for all policy rules
   - Removed deprecation warning tests

4. **`tests/test_config_schema_deprecation.py`**

   - Updated `test_strict_mode_validation_works` to use `dataset`

5. **`tests/test_concurrency_fetch.py`**
   - Updated `test_concurrent_fetch_same_dataset` to use clean policy

## Design Rationale

### Why This is Better

**1. No Backward Compatibility Burden**

- We control the repository
- No external users yet
- Can enforce clean patterns from day one

**2. Clear Mental Model**

- One way to do things
- Easy to explain: "Use dataset; custom for external files"
- No confusion about dataset vs path

**3. Fewer Edge Cases**

- No path-only configs to handle
- No warnings to maintain
- No deprecation timelines

**4. Better Error Messages**

- Specific: tells you exactly what to do
- Actionable: shows the solution
- Consistent: same pattern everywhere

**5. Simpler Implementation**

- Less branching in code
- Fewer validation steps
- Easier to test and maintain

### Alternative Considered (and Rejected)

**Keep permissive model with warnings:**

```python
# Allow both, warn on path-only
if self.path and not self.dataset:
    warnings.warn("path-only is deprecated")
```

**Why Rejected:**

- Adds complexity for no benefit
- Confusing for users (two ways to do same thing)
- More code to test and maintain
- No backward compatibility burden to justify it

## Documentation Updates Needed

1. **User Guide:**

   - Document the three valid patterns
   - Show examples of each
   - Explain when to use `custom`

2. **Migration Guide** (for posterity):

   - How to convert path-only configs
   - Error message explanations

3. **API Reference:**
   - Update `DataConfig` docstring
   - Document validation rules
   - Show valid/invalid examples

## Success Criteria

✅ All tests pass
✅ Policy is consistently enforced
✅ Error messages are clear and actionable
✅ Code is simpler than before
✅ No backward compatibility complexity

## Conclusion

The clean dataset policy eliminates unnecessary complexity by establishing simple, clear rules:

> **"Use `dataset` for data. Use `dataset='custom'` for external files. That's it."**

This simplicity makes the system easier to understand, implement, test, and maintain - with no downside since we control compatibility.
