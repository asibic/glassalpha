# API Redesign Phase Checklist

**Track progress**: Check boxes as tasks complete

---

## Phase 1: Namespace & Import Convenience (30k tokens)

### Implementation

- [ ] Create `src/glassalpha/__init__.py` with PEP 562 lazy loading
- [ ] Define `_LAZY_MODULES` dict for audit, datasets, utils
- [ ] Implement `__getattr__()` with importlib.import_module()
- [ ] Implement `__dir__()` for tab completion
- [ ] Add `__all__` with public API surface
- [ ] Verify `_export` NOT in `__all__` (privacy enforcement)

### Testing

- [ ] `test_import_speed()` - <200ms cold start
- [ ] `test_lazy_loading()` - modules load on first access
- [ ] `test_tab_completion()` - dir(ga) includes lazy modules
- [ ] `test_private_module_not_exposed()` - \_export not in dir()

### Validation

- [ ] `python -c "import glassalpha; print(dir(glassalpha))"` works
- [ ] `import glassalpha as ga; ga.audit` triggers load
- [ ] mypy --strict passes

**Exit Criteria**: Import <200ms, lazy loading verified

---

## Phase 2: Result Object Wrapper (80k tokens)

### Implementation

- [ ] Create `src/glassalpha/api/result.py`
- [ ] `AuditResult` dataclass with frozen=True
- [ ] Add fields: id, schema_version, manifest, metrics
- [ ] `__eq__()` via result.id comparison
- [ ] `__hash__()` via int(self.id[:16], 16)
- [ ] `equals()` with tolerance (rtol, atol)
- [ ] `__repr__()` compact format
- [ ] `_repr_html_()` for Jupyter (O(1) complexity)
- [ ] `summary()` method for logging
- [ ] `to_json()`, `to_pdf()`, `to_config()`, `save()` methods

### Metric Wrappers

- [ ] Create `src/glassalpha/api/metrics.py`
- [ ] `ReadonlyMetrics` base class with Mapping protocol
- [ ] `_freeze_nested()` recursively freezes dicts/lists/arrays
- [ ] `__getitem__()` for dict-style access
- [ ] `__getattr__()` for attribute-style access (raises GlassAlphaError)
- [ ] `PerformanceMetrics` with plot methods
- [ ] `FairnessMetrics` with group plotting
- [ ] `CalibrationMetrics` with calibration curve
- [ ] `StabilityMetrics`, `ExplanationSummary`, `RecourseSummary`

### Immutability

- [ ] `_freeze_array()` - make C-contiguous + setflags(write=False)
- [ ] Dataclass frozen=True prevents attribute mutation
- [ ] Manifest wrapped in types.MappingProxyType
- [ ] Nested dicts recursively frozen

### Testing

- [ ] `test_deep_immutability()` - dataclass, dicts, arrays
- [ ] `test_result_eq_via_id()` - strict equality
- [ ] `test_result_hash_stability()` - stable after pickle
- [ ] `test_equals_with_tolerance()` - rtol/atol work
- [ ] `test_equals_handles_none()` - graceful None handling
- [ ] `test_repr_html_performance()` - <10ms
- [ ] `test_readonly_metrics_key_error()` - dict access
- [ ] `test_readonly_metrics_attr_error()` - attribute access
- [ ] `test_plot_no_rcparams_mutation()` - global state preserved

**Exit Criteria**: Deep immutability verified, both access patterns work

---

## Phase 3: Entry Point Signatures (60k tokens)

### from_model() Implementation

- [ ] Create `src/glassalpha/api/audit.py`
- [ ] `from_model()` signature with all parameters
- [ ] Validate problem type (binary classification only)
- [ ] Validate missing_policy (only "include_unknown" in v0.2)
- [ ] Deep copy inputs (X, y) to prevent mutation
- [ ] `_normalize_protected_attributes()` helper
  - [ ] Handle dict, DataFrame, list[str] formats
  - [ ] Convert to categorical dtype
  - [ ] Map NaN to "Unknown" (not "nan")
  - [ ] Stable category order (sorted + Unknown last)
  - [ ] Reject MultiIndex (GAE1012)
- [ ] `_get_probabilities()` helper
  - [ ] Use y_proba if provided
  - [ ] Try model.predict_proba() if available
  - [ ] Raise GAE1008 if only decision_function
- [ ] Fingerprint model with hash_model()
- [ ] Hash data with hash_data_for_manifest()
- [ ] Build result with minimal dict for ID computation
- [ ] Compute result_id via compute_result_id()
- [ ] Build full manifest with provenance
- [ ] Return AuditResult

### from_predictions() Implementation

- [ ] `from_predictions()` signature
- [ ] Validate inputs (convert to numpy)
- [ ] Make y_proba immutable
- [ ] Compute metrics (omit AUC if no y_proba)
- [ ] Build result (no explanations/recourse)

### from_config() Implementation

- [ ] `from_config()` signature
- [ ] Load config with load_config()
- [ ] Load model by fingerprint
- [ ] Load data by hash
- [ ] Verify data hashes (raise GAE2003 on mismatch)
- [ ] Reconstruct protected_attributes with exact categories
- [ ] Call from_model()
- [ ] Verify result.id matches expected (raise GAE2002 on mismatch)

### Testing

- [ ] `test_inputs_are_copied()` - no mutation
- [ ] `test_protected_attrs_length_validation()` - GAE1003
- [ ] `test_protected_attrs_nan_to_unknown()` - "Unknown" not "nan"
- [ ] `test_protected_attrs_category_order()` - sorted + Unknown
- [ ] `test_multiindex_rejection()` - GAE1012
- [ ] `test_decision_function_model_error()` - GAE1008
- [ ] `test_auc_omitted_without_proba()` - keys check
- [ ] `test_auc_access_error_without_proba()` - GAE1009
- [ ] `test_missing_policy_validation()` - GAE1005
- [ ] `test_multiclass_rejection()` - GAE1004

**Exit Criteria**: All entry points work, validation catches errors

---

## Phase 4: Determinism & Hashing (70k tokens) **CRITICAL**

### Canonicalization

- [ ] Create `src/glassalpha/core/canonicalization.py`
- [ ] `canonicalize()` function with recursive handling
- [ ] Dict: sort keys, recurse values
- [ ] List/tuple: recurse elements
- [ ] np.ndarray: wrap in `{"__ndarray__": {dtype, shape, data}}`
- [ ] Float: normalize -0.0 to 0.0
- [ ] NaN: convert to None (null in JSON)
- [ ] Infinity: sentinel `{"__float__": "±Infinity"}`
- [ ] Int/bool: convert to Python types
- [ ] String: pass through
- [ ] Bytes: base64 encode
- [ ] Datetime: UTC ISO 8601 (naive treated as UTC)
- [ ] None: pass through

### Result ID Computation

- [ ] Create `src/glassalpha/core/hashing.py`
- [ ] `compute_result_id()` function
- [ ] Canonicalize input dict
- [ ] Serialize to JSON (sort_keys=True, ensure_ascii=True, allow_nan=False)
- [ ] SHA-256 hash of UTF-8 bytes
- [ ] Return hex digest (no prefix here)

### Data Hashing

- [ ] `hash_data_for_manifest()` function with streaming
- [ ] Return format: `"sha256:{hex_digest}"`
- [ ] DataFrame: hash columns, index, values
- [ ] Series: hash dtype, index, values
- [ ] ndarray: hash dtype, shape, bytes
- [ ] Categorical: hash categories + codes separately
- [ ] String/object: normalize to StringDtype, sentinel for NA
- [ ] Boolean: convert to uint8
- [ ] Datetime: normalize to UTC, convert to int64 ns
- [ ] Timedelta: convert to int64 ns
- [ ] Numeric: hash bytes directly
- [ ] RangeIndex: hash parameters only (no materialization)
- [ ] MultiIndex: reject with GAE1012

### Atomic Writes

- [ ] `_atomic_write()` function
- [ ] Write to .tmp file
- [ ] os.replace() (atomic on POSIX)
- [ ] Cleanup .tmp on failure
- [ ] Support mode="w" and mode="wb"

### Testing

- [ ] `test_result_id_determinism()` - same inputs = same ID
- [ ] `test_canonical_hash_nan()` - NaN → null
- [ ] `test_canonical_hash_neg_zero()` - -0.0 → 0.0
- [ ] `test_canonical_hash_infinity()` - Inf sentinel
- [ ] `test_canonical_hash_ndarray()` - dtype/shape preserved
- [ ] `test_canonical_hash_bytes()` - base64 encoding
- [ ] `test_data_hash_prefix()` - "sha256:" prefix
- [ ] `test_data_hash_dtype_sensitivity()` - int32 vs int64
- [ ] `test_data_hash_column_order()` - order matters
- [ ] `test_data_hash_range_index()` - fast path (3x speedup)
- [ ] `test_data_hash_categorical()` - categories matter
- [ ] `test_data_hash_timezone()` - UTC normalization
- [ ] `test_atomic_write_cleanup()` - .tmp removed on failure
- [ ] `test_atomic_write_binary_mode()` - mode="wb" works
- [ ] `test_to_json_uses_strict_canonicalization()` - no NaN/Inf

**Exit Criteria**: Byte-identical hashes, all edge cases handled

---

## Phase 5: Error Handling (40k tokens)

### Implementation

- [ ] Create `src/glassalpha/exceptions.py`
- [ ] `GlassAlphaError` dataclass with frozen=True
- [ ] Fields: code, message, fix, docs
- [ ] `__str__()` method with formatted output
- [ ] Create `src/glassalpha/error_codes.py` with constants

### Error Codes

- [ ] GAE1001_INVALID_PROTECTED_ATTRS - format validation
- [ ] GAE1002_METRIC_ACCESS - unknown metric
- [ ] GAE1003_LENGTH_MISMATCH - data length mismatch
- [ ] GAE1004_UNSUPPORTED_PROBLEM_TYPE - non-binary classification
- [ ] GAE1005_MISSING_POLICY_UNSUPPORTED - unsupported missing policy
- [ ] GAE1008_NO_PROBA_AVAILABLE - decision_function only
- [ ] GAE1009_PROBA_REQUIRED_FOR_AUC - AUC without probabilities
- [ ] GAE1012_UNSUPPORTED_MULTIINDEX - MultiIndex rejection
- [ ] GAE2002_RESULT_ID_MISMATCH - reproduction mismatch
- [ ] GAE2003_DATA_HASH_MISMATCH - data changed
- [ ] GAE4001_PATH_EXISTS - file overwrite protection

### Integration

- [ ] Update `_normalize_protected_attributes()` to raise GlassAlphaError
- [ ] Update `_get_probabilities()` to raise GlassAlphaError
- [ ] Update `from_model()` validation to raise GlassAlphaError
- [ ] Update `from_config()` verification to raise GlassAlphaError
- [ ] Update `ReadonlyMetrics.__getattr__()` to raise GlassAlphaError
- [ ] Add `PerformanceMetrics.__getattr__()` override for GAE1009

### Testing

- [ ] Test all error codes raised in correct contexts
- [ ] Test error messages include fix and docs
- [ ] `test_performance_metrics_gae1009_override()` - specific error
- [ ] `test_unknown_metric_error()` - GAE1002 with docs

**Exit Criteria**: All errors helpful, 100% have fix/docs

---

## Phase 6: Testing (80k tokens)

### Test Organization

- [ ] Create `tests/api/test_contracts.py` - core contracts
- [ ] Create `tests/api/test_determinism.py` - reproducibility
- [ ] Create `tests/api/test_immutability.py` - frozen checks
- [ ] Create `tests/api/test_errors.py` - error quality

### Contract Tests (All from spec)

- [ ] Import speed <200ms
- [ ] Lazy loading verification
- [ ] Result ID determinism (10 runs)
- [ ] Deep immutability
- [ ] Inputs not mutated
- [ ] All canonical hash edge cases
- [ ] All data hash edge cases
- [ ] Protected attributes validation
- [ ] Probability handling
- [ ] Error code coverage
- [ ] Atomic write safety
- [ ] Hash stability across pickle
- [ ] Tolerance-based equality
- [ ] HTML repr performance
- [ ] Metric access patterns
- [ ] Plot method safety

### Performance Tests

- [ ] Import speed benchmark
- [ ] RangeIndex vs materialized (3x threshold)
- [ ] _repr_html_() <10ms

### Integration Tests

- [ ] Full audit workflow with from_model()
- [ ] Reproduction with from_config()
- [ ] Model without probabilities
- [ ] Data with missing values
- [ ] Cross-platform tolerance (if CI available)

**Exit Criteria**: 95%+ coverage, all edge cases tested

---

## Phase 7: Documentation (50k tokens)

### API Reference

- [ ] Create/update `site/docs/reference/api/audit.md`
- [ ] Document `from_model()` with all parameters
- [ ] Document `from_predictions()` with limitations
- [ ] Document `from_config()` with verification
- [ ] Document AuditResult class and methods
- [ ] Document metric access patterns (dict vs attribute)
- [ ] Add examples for all entry points
- [ ] Document error codes with fixes

### Guides

- [ ] Create `site/docs/guides/probability-requirements.md`
  - [ ] Models with predict_proba()
  - [ ] Models with decision_function only
  - [ ] CalibratedClassifierCV example
  - [ ] from_predictions() without y_proba
- [ ] Create `site/docs/guides/missing-data.md`
  - [ ] v0.2: include_unknown policy
  - [ ] NaN → "Unknown" mapping
  - [ ] v0.3: exclude_unknown (deferred)
- [ ] Update `site/docs/guides/reproducibility.md`
  - [ ] Timezone handling section
  - [ ] Data hash format explanation
  - [ ] Reproduction validation
- [ ] Update `site/docs/reference/api/data-requirements.md`
  - [ ] Problem type restrictions
  - [ ] Index requirements
  - [ ] MultiIndex rejection
  - [ ] Timezone policy

### Notebooks

- [ ] Update `examples/notebooks/quickstart_from_model.ipynb`
  - [ ] Dict and attribute access
  - [ ] Plot methods
  - [ ] summary() for logging
  - [ ] to_config() for reproduction
  - [ ] Error handling demo
- [ ] Update `examples/notebooks/german_credit_walkthrough.ipynb`
  - [ ] Reproducibility section
  - [ ] Hash format explanation
  - [ ] Config generation

**Exit Criteria**: All public APIs documented, notebooks work

---

## Phase 8: Quality Gates (20k tokens)

### Metric Registry

- [ ] Update `src/glassalpha/metrics/registry.py`
- [ ] `MetricSpec` dataclass with metadata
- [ ] Fields: name, display_name, description
- [ ] Fields: higher_is_better, compute_requirements
- [ ] Fields: tolerance, aggregation
- [ ] Fields: fairness_definition, reference_group
- [ ] Register all performance metrics
- [ ] Register all fairness metrics
- [ ] Register all calibration metrics

### Documentation

- [ ] Add metric registry docs to API reference
- [ ] Show higher_is_better usage
- [ ] Show fairness_definition for dashboards

**Exit Criteria**: Registry complete, ready for enterprise features

---

## Phase 9: Stability Index (10k tokens)

### Documentation

- [ ] Create `site/docs/reference/stability-index.md`
- [ ] Document maturity levels (Stable/Beta/Experimental)
- [ ] List v0.2 stable APIs
- [ ] List v0.2 beta APIs
- [ ] List v0.2 experimental APIs
- [ ] Document breaking change policy
- [ ] Create `site/docs/reference/tolerance-policy.md`
- [ ] Document default tolerances by metric type
- [ ] Show strict vs tolerance equality examples
- [ ] Explain cross-platform differences

**Exit Criteria**: API stability guarantees published

---

## Final Exit Checklist

**Blockers (Must Fix)**:

- [ ] Blocker #1: GAE1009 override in PerformanceMetrics
- [ ] Blocker #2: NaN → "Unknown" (not "nan")
- [ ] Blocker #3: Array canonicalization preserves dtype/shape
- [ ] Blocker #4: Bytes canonicalize to base64
- [ ] Blocker #5: Data hash returns "sha256:..." prefix

**Should-Fix (Quality)**:

- [ ] equals() handles None gracefully
- [ ] Atomic write supports binary mode
- [ ] to_json uses strict canonicalization
- [ ] model_type in manifest includes module + qualname
- [ ] Plot methods don't mutate rcParams

**Quality Gates**:

- [ ] Import speed <200ms (cold start)
- [ ] Determinism 100% (10 runs, same platform)
- [ ] All errors have code/fix/docs
- [ ] Deep immutability verified
- [ ] mypy --strict passes
- [ ] Test coverage >95% for api/
- [ ] All public functions have docstrings + examples
- [ ] Notebooks updated
- [ ] API reference complete
- [ ] Stability index published
- [ ] Tolerance policy published

**Deliverables**:

- [ ] API_DESIGN_ANALYSIS.md generated
- [ ] CHANGELOG.md updated
- [ ] Zero lint errors
- [ ] All contract tests pass on Linux + macOS

**Cleanup**:

- [ ] Delete API_REDESIGN_MASTER_PLAN.md
- [ ] Delete API_REDESIGN_PHASE_CHECKLIST.md
- [ ] Delete API_DESIGN_DECISIONS.md (if created)
- [ ] Move final analysis to permanent location
