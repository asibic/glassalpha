# Strict Dataset Policy Lockdown

**Date:** September 30, 2025
**Status:** ✅ Complete
**Type:** API Finalization

## Summary

Locked in the explicit, strict dataset policy by systematically updating all tests and removing any auto-normalization logic. The API is now clean, clear, and ready for OSS launch with no future breaking changes.

## The Final Policy

### Three Clear Rules

1. **`data.dataset` is required for any run**
2. **`data.dataset="custom"` enables `data.path` for external files**
3. **`data_schema` alone is allowed for schema-only utilities**

### Why This is Better Than Auto-Normalization

**We Rejected:**

```python
# Auto-normalize path-only to dataset="custom"
if config.path and not config.dataset:
    config.dataset = "custom"  # ❌ Hidden magic
```

**Why:**

- Creates ambiguity: Is it explicit or implicit?
- Makes future changes harder (what if we want different behavior?)
- Users don't learn the right pattern
- Debugging is harder ("why did it add that?")
- Future breaking changes are inevitable

**We Chose:**

```python
# Explicit, strict validation
if not self.dataset:
    raise ValueError(
        "data.dataset is required. "
        "Use data.dataset='custom' with data.path for external files."
    )
```

**Why:**

- Crystal clear: one way to do things
- Error message teaches the right pattern
- No hidden behavior to document or maintain
- Future-proof: no breaking changes needed
- Easier to understand and debug

## Implementation Summary

### A. Updated Test Files

**Batch Updated (23 occurrences across 4 files):**

- `tests/test_provenance_written.py` - 3 occurrences
- `tests/test_utils_comprehensive.py` - 1 occurrence
- `tests/test_config_flexibility.py` - 9 occurrences
- `tests/test_end_to_end.py` - 10 occurrences

**Manually Fixed:**

- `tests/test_ci_regression_guards.py` - Fixed provenance test
- `tests/test_config_loading.py` - Fixed 5 test cases

**Pattern Applied:**

```yaml
# Before
data:
  path: test.csv

# After
data:
  dataset: custom
  path: test.csv
```

### B. Removed Auto-Normalization

**File:** `src/glassalpha/config/loader.py`

**Removed:**

```python
# Check for deprecated data.path usage without data.dataset
if audit_config.data.path and not audit_config.data.dataset and "path" in config_dict.get("data", {}):
    logger.warning(
        "Using data.path without data.dataset is deprecated. "
        "Consider using data.dataset for better functionality and future compatibility.",
    )
```

**Why:** Path-only configs now fail at model validation with a clear, actionable error message.

### C. Updated Documentation

**File:** `packages/README.md`

**Added Section:** "Data Sources" with three clear patterns:

1. **Built-in Datasets**

   ```yaml
   data:
     dataset: german_credit
     fetch: if_missing
   ```

2. **Custom Files**

   ```yaml
   data:
     dataset: custom
     path: /path/to/data.csv
   ```

3. **Schema-Only**
   ```yaml
   data:
     data_schema: { ... }
   ```

## Test Results

### All Tests Pass

```
✅ 95 passed in test_config_loading, test_provenance_written, test_utils_comprehensive, test_config_flexibility
✅ 15 passed in test_end_to_end
✅ 50 passed, 1 skipped in test_ci_regression_guards, test_config_schema_deprecation, test_deprecation_path_only, test_cache_dirs, test_concurrency_fetch

Total: 160+ tests passing ✅
```

### Validation Scenarios Tested

| Scenario             | Result  | Error Message                                            |
| -------------------- | ------- | -------------------------------------------------------- |
| Registry dataset     | ✅ Pass | N/A                                                      |
| Custom + path        | ✅ Pass | N/A                                                      |
| Schema-only          | ✅ Pass | N/A                                                      |
| Path without dataset | ❌ Fail | `data.dataset is required. Use data.dataset='custom'...` |
| Registry + path      | ❌ Fail | `data.path must be omitted...`                           |
| Custom without path  | ❌ Fail | `data.path is required when data.dataset='custom'`       |
| Empty config         | ❌ Fail | `data.dataset is required...`                            |

## Error Messages

### Error Message Quality

All error messages now:

1. **State the problem** - "data.dataset is required"
2. **Provide the solution** - "Use data.dataset='custom' with data.path"
3. **Are actionable** - User knows exactly what to change
4. **Are consistent** - Same pattern across all validation errors

**Example:**

```
ValidationError: data.dataset is required.
Use data.dataset='custom' with data.path for external files.
```

**User knows:**

- What's missing: `data.dataset`
- How to fix it: Add `dataset: custom`
- Why: To use external files with path

## Files Modified

1. **`src/glassalpha/config/loader.py`**

   - Removed deprecation warning for path-only configs
   - Clean validation now handled at model level

2. **`packages/README.md`**

   - Added comprehensive "Data Sources" section
   - Three clear patterns with examples
   - Clear distinction between use cases

3. **Test Files (11 files updated)**
   - `test_ci_regression_guards.py`
   - `test_config_loading.py`
   - `test_provenance_written.py`
   - `test_utils_comprehensive.py`
   - `test_config_flexibility.py`
   - `test_end_to_end.py`
   - All path-only configs updated to `dataset: custom`

## Benefits for OSS Launch

### 1. Clear Mental Model

**Before (Permissive):**

- "Do I use dataset or path?"
- "What's the difference?"
- "Why did my path-only config work but show a warning?"

**After (Strict):**

- "Always use dataset"
- "Use dataset='custom' for my files"
- "Error tells me exactly what to do"

### 2. No Future Breaking Changes

**Avoiding:**

```
# Version 1.0: path-only with warning
data:
  path: data.csv  # Warning: deprecated

# Version 2.0: breaks existing configs
# ValidationError: data.dataset is required
```

**Instead:**

```
# Version 1.0: strict from day one
data:
  dataset: custom
  path: data.csv

# Version 2.0+: same config works forever
```

### 3. Easier Support

**User:** "My config isn't working!"

**Before:**

```
Support: "What does your config look like?"
User: "I have data.path set..."
Support: "Ah, you need to also set data.dataset, or wait,
         do you have the warning suppressed? Or is this an old config?"
```

**After:**

```
Support: "What error do you see?"
User: "data.dataset is required. Use data.dataset='custom'..."
Support: "Perfect! The error message tells you the fix.
         Just add dataset: custom"
```

### 4. Better First Impressions

**New User Experience:**

**Before (Permissive):**

1. User reads docs, sees path example
2. Uses path-only, gets warning
3. Confused: "Is this wrong? Should I fix it?"
4. Searches docs, finds dataset option
5. Still confused about when to use what

**After (Strict):**

1. User reads docs, sees three clear patterns
2. Chooses pattern based on use case
3. Config works or fails with clear error
4. Error message guides them to fix
5. Learns the right pattern immediately

## Comparison: Permissive vs Strict

| Aspect              | Permissive (Auto-Normalize)  | Strict (Explicit)           |
| ------------------- | ---------------------------- | --------------------------- |
| **First Use**       | Seems easier (path works)    | Requires learning pattern   |
| **Understanding**   | Hidden magic, confusing      | Crystal clear               |
| **Error Messages**  | Vague warnings               | Actionable errors           |
| **Future Changes**  | Breaking changes likely      | Future-proof                |
| **Support Burden**  | High (explain auto-behavior) | Low (error explains itself) |
| **Code Complexity** | Auto-normalization logic     | Simple validation           |
| **User Confidence** | Unsure if doing it right     | Knows exact pattern         |
| **OSS Adoption**    | Users hit warnings, unsure   | Clear path to success       |

## Production Readiness Checklist

✅ **API is explicit** - No hidden auto-normalization
✅ **Validation is strict** - Fails fast with clear errors
✅ **Error messages are actionable** - User knows how to fix
✅ **Documentation is comprehensive** - Three clear patterns
✅ **All tests pass** - 160+ tests covering all scenarios
✅ **No deprecation warnings** - Clean codebase
✅ **Future-proof** - No breaking changes needed

## Design Rationale

### Why Strict > Permissive for OSS

**Permissive Approach:**

- ❌ Requires maintaining auto-normalization code
- ❌ Creates two ways to do the same thing
- ❌ Harder to explain in docs ("it auto-converts but...")
- ❌ Users don't learn the right pattern
- ❌ Future changes require deprecation cycles

**Strict Approach:**

- ✅ Simple: one way to do things
- ✅ Clear: error tells you what to do
- ✅ Teachable: learn by doing (error-driven)
- ✅ Future-proof: no breaking changes needed
- ✅ Less code: validation only, no normalization

### Real-World Analogy

**Permissive = Auto-correct:**

- Tries to guess what you mean
- Sometimes wrong, confusing
- You don't learn proper spelling
- Inconsistent behavior

**Strict = Spell checker:**

- Points out mistakes
- Shows correct spelling
- You learn the right way
- Consistent, predictable

## Success Metrics

| Metric                 | Target              | Actual                           |
| ---------------------- | ------------------- | -------------------------------- |
| Tests passing          | 100%                | ✅ 160+ tests                    |
| Error message clarity  | Actionable          | ✅ All errors show solution      |
| Documentation complete | 3 patterns          | ✅ Built-in, custom, schema-only |
| Code complexity        | Minimal             | ✅ Simple validation only        |
| Future-proof           | No breaking changes | ✅ API locked in                 |

## Conclusion

The strict dataset policy is now locked in and ready for OSS launch. By rejecting auto-normalization and requiring explicit configuration, we've created a system that is:

1. **Easy to understand** - One clear way to do things
2. **Easy to support** - Error messages are self-documenting
3. **Easy to maintain** - No hidden behavior to maintain
4. **Future-proof** - No breaking changes needed

**The API contract:**

> **`data.dataset` is required. Use `dataset='custom'` for external files. That's it.**

This simplicity will serve users well and prevent future pain.
