# Preprocessing TDD Status - Initial Baseline

## ✅ Test Suite Created (Red Phase Established)

Created **10 contract tests** covering catastrophic failure modes and boundaries. Tests written **before** implementation to define acceptance criteria.

### Test Results: 5 passed, 4 failed, 1 skipped

```
PASSED  test_loader.py::test_hashes_present_and_stable                   [√]
PASSED  test_loader.py::test_missing_artifact_is_actionable_error        [√]
PASSED  test_loader.py::test_corrupted_artifact_is_actionable_error      [√]
PASSED  test_manifest.py::test_manifest_canonicalization_matches_golden  [√]
PASSED  test_validation.py::test_feature_count_and_names_match_model     [√]

FAILED  test_manifest.py::test_encoder_settings_exposed                  [×]
FAILED  test_manifest.py::test_unknown_category_rates_reported           [×]
FAILED  test_validation.py::test_rejects_unknown_transformer_class       [×]
FAILED  test_validation.py::test_sparse_flag_detected_and_enforced       [×]

SKIPPED test_strict_mode.py::test_strict_mode_fails_on_version_mismatch [−]
SKIPPED test_strict_mode.py::test_strict_profile_blocks_auto_mode       [−]
SKIPPED test_strict_mode.py::test_strict_mode_requires_hashes           [−]
```

## Files Created

### Test Files (packages/tests/preprocessing/)

- ✅ `conftest.py` - 8 fixtures (toy_df, sklearn_artifact, corrupted_artifact, etc.)
- ✅ `test_loader.py` - 3 tests (hashing, error handling)
- ✅ `test_validation.py` - 3 tests (class allowlist, sparsity, shape)
- ✅ `test_manifest.py` - 3 tests (encoder params, unknown rates, golden)
- ✅ `test_strict_mode.py` - 3 tests (version policy, auto block, hashes) [SKIPPED - needs config]
- ✅ `README.md` - Test suite documentation

### Stub Implementations (packages/src/glassalpha/preprocessing/)

- ✅ `__init__.py` - Module exports
- ✅ `loader.py` - load_artifact, compute_file_hash, compute_params_hash
- ✅ `manifest.py` - MANIFEST_SCHEMA_VERSION, params_hash
- ✅ `validation.py` - ALLOWED_FQCN, validate_classes, validate_sparsity, validate_output_shape, assert_runtime_versions
- ✅ `introspection.py` - extract_sklearn_manifest, compute_unknown_rates, \_extract_component, \_find_encoders

## What's Working (Green Tests)

### ✅ Dual Hash System

- File hash computed correctly (SHA256/BLAKE2b)
- Params hash stable across repickling ✨
- Both hashes present in output

### ✅ Error Handling

- Missing artifact → clear FileNotFoundError with path
- Corrupted artifact → clean ValueError, not stack dump

### ✅ Shape Validation

- Feature count checking works
- Feature name validation works
- Minimal diff on mismatch

### ✅ Manifest Canonicalization

- Params hash is deterministic
- Schema version included
- JSON serialization stable

## What Needs Fixing (Red Tests)

### ❌ test_encoder_settings_exposed

**Issue:** OneHotEncoder component not found in manifest

**Fix needed:**

- `_extract_component()` not detecting nested transformers in ColumnTransformer
- Need to extract encoder from the Pipeline inside ColumnTransformer

**Code location:** `introspection.py:_extract_component()`

---

### ❌ test_unknown_category_rates_reported

**Issue:** `compute_unknown_rates()` returns empty dict (no unknowns detected)

**Fix needed:**

- `_find_encoders()` not finding encoders in nested ColumnTransformer structure
- Need to properly extract column mappings

**Code location:** `introspection.py:_find_encoders()`

---

### ❌ test_rejects_unknown_transformer_class

**Issue:** Can't pickle lambda function in FunctionTransformer

**Fix needed:**

- Change test to use a module-level function (not nested lambda)
- Alternatively, test with a different disallowed class

**Code location:** `test_validation.py:27-35`

---

### ❌ test_sparse_flag_detected_and_enforced

**Issue:** Not detecting sparse output correctly

**Fix needed:**

- Improve sparsity detection logic
- Need to handle scipy.sparse matrix types properly

**Code location:** `test_validation.py:52-60`

## Skipped Tests (Needs Config Integration)

### ⏸️ test_strict_mode_fails_on_version_mismatch

**Reason:** Needs `PreprocessingConfig` in `config/schema.py`

**Blocks:** Phase 2 (strict mode integration)

---

### ⏸️ test_strict_profile_blocks_auto_mode

**Reason:** Needs `validate_strict_mode()` in `config/strict.py`

**Blocks:** Phase 2 (strict mode integration)

---

### ⏸️ test_strict_mode_requires_hashes

**Reason:** Needs `PreprocessingConfig` validation

**Blocks:** Phase 2 (strict mode integration)

## Next Steps (TDD Green Phase)

### Immediate (Fix Red Tests)

1. **Fix test_encoder_settings_exposed** (~15 min)

   - Flatten ColumnTransformer components properly
   - Extract encoders from nested pipelines

2. **Fix test_unknown_category_rates_reported** (~20 min)

   - Fix `_find_encoders()` to traverse nested structures
   - Match encoders to their column mappings

3. **Fix test_rejects_unknown_transformer_class** (~10 min)

   - Change test to use module-level function
   - OR test with non-picklable class (e.g., custom class)

4. **Fix test_sparse_flag_detected_and_enforced** (~15 min)
   - Check for `scipy.sparse.issparse()` instead of `.toarray` attribute
   - Improve detection robustness

**Estimated time to green:** 1 hour

### After Green (Config Integration)

5. **Create PreprocessingConfig schema** (~30 min)

   - Add to `config/schema.py`
   - Include all fields: mode, artifact_path, hashes, version_policy, thresholds

6. **Un-skip strict mode tests** (~30 min)

   - Implement `validate_strict_mode()` preprocessing checks
   - Update `config/strict.py`

7. **Verify all 10 tests pass** (~15 min)
   - Full green suite
   - Ready for implementation phases

**Total estimated time to full green:** 2-2.5 hours

## Command to Run Tests

```bash
cd packages
source venv/bin/activate

# Run all preprocessing tests
python -m pytest tests/preprocessing/ -v

# Run specific test
python -m pytest tests/preprocessing/test_manifest.py::test_encoder_settings_exposed -v

# Watch mode (run on file change) - if pytest-watch installed
ptw tests/preprocessing/
```

## TDD Philosophy

> "The test is the specification. If it passes, the spec is met."

- Tests define **what** we need (acceptance criteria)
- Implementation defines **how** we achieve it
- Red → Green → Refactor cycle
- No gold-plating: implement only what tests demand

## Success Metric

✅ **All 10 core tests passing** = Core boundaries validated, catastrophic paths locked
→ Then add cosmetic features (CLI pretty-print, report templates, doctor messages)

---

**Status:** TDD spine established. Ready to iterate red→green on failing tests.
**Next:** Fix 4 red tests, then integrate config schema for strict mode tests.
