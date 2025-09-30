# DataConfig Validation Relaxation

**Date:** September 30, 2025
**Status:** ✅ Complete
**Type:** Architecture Improvement

## Summary

Relaxed `DataConfig` model validation to allow schema-only configurations for tests and utilities, while maintaining strict data source requirements at the pipeline boundary where data is actually needed.

## Problem

The original `DataConfig` validation was too strict at the model level:

```python
@model_validator(mode="after")
def validate_dataset_or_path(self):
    """Ensure either dataset or path is specified."""
    if not self.dataset and not self.path:
        raise ValueError("Either 'dataset' or 'path' must be specified")
    return self
```

**Issues:**

1. Blocked schema-only configurations used in tests and validation utilities
2. Required test hacks (mocking paths) just to satisfy constructor
3. Enforced strictness too early - before knowing if data would actually be accessed
4. Made `DataConfig` less flexible for tooling and utilities

**Failing Test:**

```python
def test_data_schema_field_works():
    """Test that the data_schema field works correctly."""
    schema_data = {"type": "object", "properties": {"name": {"type": "string"}}}

    config = DataConfig(data_schema=schema_data)  # ❌ ValidationError!
```

## Solution

### A. Relaxed Model Validation

**File:** `src/glassalpha/config/schema.py`

Changed to permit three valid scenarios:

1. `dataset` specified (built-in dataset)
2. `path` specified (custom file)
3. `data_schema` specified (schema-only config)

```python
@model_validator(mode="after")
def validate_source_or_schema(self) -> "DataConfig":
    """Ensure a data source or schema is specified.

    Permits schema-only configs for validation utilities and tests.
    The pipeline will enforce dataset/path when it actually needs data.
    """
    if not (self.dataset or self.path or self.data_schema):
        raise ValueError(
            "Either 'dataset' or 'path' must be specified, "
            "or provide 'data_schema' for schema-only configs"
        )
    return self
```

**Key Changes:**

- Added `or self.data_schema` to validation condition
- Updated error message to document schema-only option
- Renamed method to reflect new semantics
- Added docstring explaining the design decision

### B. Strict Pipeline Boundary Guard

**File:** `src/glassalpha/pipeline/audit.py`

Added runtime enforcement where data is actually needed:

```python
def _load_data(self) -> tuple[pd.DataFrame, TabularDataSchema]:
    """Load and validate dataset.

    Returns:
        Tuple of (data, schema)

    Raises:
        ValueError: If no data source is provided

    """
    logger.info("Loading and validating dataset")

    # Enforce source presence at runtime unless this is a schema-only flow
    if not (self.config.data.dataset or self.config.data.path):
        raise ValueError(
            "No data source provided. Set data.dataset or data.path "
            "(data_schema alone is only for schema-only operations/tests)."
        )

    # Resolve and ensure dataset availability
    data_path = self._resolve_requested_path()
    data_path = self._ensure_dataset_availability(data_path)
    ...
```

**Why This Works:**

- Model validation is permissive (accepts schema-only configs)
- Pipeline validation is strict (requires actual data source)
- Clear separation: model = "what's allowed", pipeline = "what's required"
- Error message guides users on proper usage

### C. Updated Test

**File:** `tests/test_config_schema_deprecation.py`

Fixed test that assumed empty config was valid:

```python
def test_data_schema_field_none_by_default():
    """Test that data_schema is None by default when other fields are provided."""
    config = DataConfig(dataset="test_dataset")  # Provide required field

    assert config.data_schema is None
```

## Valid Configuration Patterns

### 1. Dataset-Based (Preferred)

```python
config = DataConfig(
    dataset="german_credit",
    fetch="if_missing",
    offline=False
)
```

### 2. Path-Based

```python
config = DataConfig(
    path="/path/to/data.csv",
    target_column="outcome"
)
```

### 3. Schema-Only (Tests/Utilities)

```python
config = DataConfig(
    data_schema={
        "type": "object",
        "properties": {
            "age": {"type": "integer"},
            "income": {"type": "number"}
        }
    }
)
```

### 4. Invalid (Empty)

```python
config = DataConfig()  # ❌ ValidationError: must provide dataset, path, or data_schema
```

## Validation Boundaries

| Boundary               | When              | What's Checked                              | Why                                          |
| ---------------------- | ----------------- | ------------------------------------------- | -------------------------------------------- |
| **Model (Pydantic)**   | At construction   | At least one of: dataset, path, data_schema | Ensures config object isn't completely empty |
| **Pipeline (Runtime)** | At `_load_data()` | Must have: dataset OR path                  | Ensures actual data source when loading data |

## Test Results

### All Tests Pass

```
tests/test_config_schema_deprecation.py - 10 passed ✅
tests/test_cache_dirs.py - 13 passed ✅
tests/test_concurrency_fetch.py - 5 passed ✅

Total: 28 passed in 3.07s
```

### Validation Behavior Verified

```
✅ Schema-only config works at model level
✅ Dataset config works at model level
✅ Path config works at model level
✅ Empty config fails at model level (correct)
✅ Schema-only config fails at pipeline level (correct)
```

## Benefits

| Aspect                | Before                      | After                         |
| --------------------- | --------------------------- | ----------------------------- |
| **Test flexibility**  | Required mock paths         | Can use schema-only configs   |
| **Validation timing** | Too early (at construction) | Right time (when data needed) |
| **Error clarity**     | Generic "path required"     | Context-specific messages     |
| **Tooling support**   | Blocked schema-only utils   | Enables validation utilities  |
| **Production safety** | Strict at model level       | Strict at pipeline level      |

## Use Cases Enabled

### 1. Schema Validation Utilities

```python
def validate_schema(schema_dict):
    """Validate a data schema without needing actual data."""
    config = DataConfig(data_schema=schema_dict)
    # Perform schema validation...
    return validation_results
```

### 2. Test Fixtures

```python
@pytest.fixture
def schema_only_config():
    """Config for testing schema validation logic."""
    return DataConfig(data_schema=TEST_SCHEMA)
```

### 3. Configuration Templates

```python
def create_config_template(schema):
    """Create a configuration template from a schema."""
    return DataConfig(
        data_schema=schema,
        # Other fields can be filled in later
    )
```

## Files Modified

1. **`src/glassalpha/config/schema.py`**

   - Relaxed `DataConfig` validation to accept schema-only configs
   - Updated method name and documentation

2. **`src/glassalpha/pipeline/audit.py`**

   - Added strict validation in `_load_data()` method
   - Clear error message for missing data sources

3. **`tests/test_config_schema_deprecation.py`**
   - Fixed test to provide required fields
   - Updated test docstring

## Design Rationale

### Why Relax at Model Level?

**Pros:**

- Enables legitimate use cases (tests, utilities, tooling)
- Follows "permissive construction, strict usage" pattern
- Doesn't compromise safety (enforced at pipeline)
- Makes the model more composable

**Cons:**

- Could allow invalid configs to be created
- Mitigated by: pipeline validation catches issues before data access

### Why Enforce at Pipeline Level?

**Pros:**

- Validation happens where data is actually needed
- Clear error messages with context
- Allows schema-only configs for non-data operations
- Follows "fail fast at the right time" principle

**Cons:**

- Validation split across two places
- Mitigated by: clear documentation and error messages

### Alternative Considered (and Rejected)

**Make everything optional, no validation:**

```python
dataset: Optional[str] = None
path: Optional[str] = None
data_schema: Optional[dict] = None
# No model_validator at all
```

**Why Rejected:**

- Too permissive - allows completely empty configs
- Pushes all validation burden to pipeline
- Harder to catch mistakes early
- Worse error messages (less context)

## Documentation Notes

The relaxed validation is documented in:

1. **Model validator docstring** - Explains the three valid patterns
2. **Pipeline error message** - Guides users when they're missing data
3. **This review document** - Full context and rationale

Users should understand:

- `data_schema` alone is valid for tests/utilities
- Production audits require `dataset` or `path`
- Pipeline will reject schema-only configs when loading data

## Future Considerations

1. **Schema-only audit mode** - Could add a `--schema-only` CLI flag that:

   - Validates schema structure
   - Checks compatibility with model
   - Doesn't require actual data
   - Uses the schema-only config pattern

2. **Dry-run mode** - Could use schema-only configs for:

   - Configuration validation
   - Pipeline dry-runs
   - Compatibility checks

3. **Multi-source configs** - Could support:
   - Multiple datasets
   - Dataset unions/joins
   - Schema validation across sources

## Conclusion

The validation relaxation follows the principle:

> **"Be permissive in what you accept, strict in what you do."**

- Model validation: Permissive (accepts schema-only configs)
- Pipeline validation: Strict (requires data when loading)

This provides flexibility for tests and utilities while maintaining safety for production use.
