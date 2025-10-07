# API Test Plan

**Purpose**: Comprehensive testing strategy for v0.2 API redesign.
**Coverage Target**: 95%+ for api/ module
**Quality Gates**: All contract tests must pass before PyPI publication.

---

## Test Organization

### Directory Structure

```
tests/api/
├── test_contracts.py         # Core API contracts
├── test_determinism.py        # Reproducibility tests
├── test_immutability.py       # Frozen checks
├── test_errors.py             # Error quality tests
├── test_hashing.py            # Canonicalization & data hashing
├── test_entry_points.py       # from_model, from_predictions, from_config
├── test_metrics_access.py     # Dict + attribute access
└── test_performance.py        # Benchmarks
```

### Test Markers

```python
@pytest.mark.contract      # Must pass before release
@pytest.mark.determinism   # Reproducibility critical
@pytest.mark.performance   # Benchmark tests
@pytest.mark.integration   # Full workflow tests
```

---

## Phase 1: Namespace & Import Tests

### test_import_speed()

**Goal**: Import completes in <200ms
**Method**: Time `import glassalpha` on cold start
**Assertion**: `duration < 0.2`

### test_lazy_loading()

**Goal**: Modules load on first access
**Method**: Check `sys.modules` before/after attribute access
**Assertions**:

- `"glassalpha.audit"` not in sys.modules before `ga.audit`
- `"glassalpha.audit"` in sys.modules after `ga.audit`

### test_tab_completion()

**Goal**: dir() includes lazy modules
**Method**: Call `dir(glassalpha)`
**Assertions**:

- `"audit"` in dir(ga)
- `"datasets"` in dir(ga)
- `"utils"` in dir(ga)

### test_private_module_not_exposed()

**Goal**: Internal modules not in **all**
**Method**: Check `"_export"` not in dir(ga)
**Assertion**: Privacy enforcement

---

## Phase 2: Result Object Tests

### test_deep_immutability()

**Goal**: Result objects fully frozen
**Method**: Try to mutate various parts
**Assertions**:

- `result.id = "new"` raises FrozenInstanceError
- `result.performance._data["accuracy"] = 0.99` raises TypeError
- `result.performance["confusion_matrix"][0, 0] = 999` raises ValueError (read-only)

### test_result_eq_via_id()

**Goal**: Equality uses result.id
**Method**: Create two results with same ID
**Assertion**: `result1 == result2` iff `result1.id == result2.id`

### test_result_hash_stability()

**Goal**: Hash stable across pickle
**Method**: Pickle + unpickle, compare hashes
**Assertion**: `hash(result) == hash(pickle.loads(pickle.dumps(result)))`

### test_equals_with_tolerance()

**Goal**: Tolerance-based comparison works
**Method**: Create results with tiny differences
**Assertions**:

- Strict: `result1 != result2`
- Tolerant: `result1.equals(result2, rtol=1e-5)`

### test_equals_handles_none()

**Goal**: Graceful None handling
**Method**: Create result with None metric value
**Assertion**: `equals()` doesn't crash, returns False if keys differ

### test_repr_html_performance()

**Goal**: Jupyter display <10ms
**Method**: Time `result._repr_html_()`
**Assertion**: `duration < 0.01`

### test_summary_performance()

**Goal**: summary() is O(1)
**Method**: Time `result.summary()` on large result
**Assertion**: `duration < 0.001` (should be instant dict access)

### test_readonly_metrics_key_error()

**Goal**: Dict access raises KeyError for missing keys
**Method**: `result.performance["nonexistent"]`
**Assertion**: KeyError raised

### test_readonly_metrics_attr_error()

**Goal**: Attribute access raises GlassAlphaError
**Method**: `result.performance.nonexistent`
**Assertion**: GAE1002 raised with docs link

### test_plot_no_rcparams_mutation()

**Goal**: Plot methods don't mutate global state
**Method**: Save rcParams, call plot, compare
**Assertion**: `plt.rcParams == original_params`

---

## Phase 3: Entry Point Tests

### test_inputs_are_copied()

**Goal**: from_model() doesn't mutate inputs
**Method**: Deep copy inputs, call from_model(), compare
**Assertions**:

- `X.equals(X_orig)`
- `y.equals(y_orig)`

### test_protected_attrs_length_validation()

**Goal**: Length mismatch raises GAE1003
**Method**: Pass protected_attributes with wrong length
**Assertion**: GAE1003 raised with fix message

### test_protected_attrs_nan_to_unknown()

**Goal**: NaN mapped to "Unknown" (not "nan")
**Method**: Create protected attr with NaN, check manifest
**Assertions**:

- `"Unknown"` in categories
- `categories[-1] == "Unknown"` (last in order)
- `"nan"` not in categories

### test_protected_attrs_category_order()

**Goal**: Categories sorted with Unknown last
**Method**: Create categorical with unsorted categories
**Assertion**: manifest shows sorted + Unknown

### test_multiindex_rejection()

**Goal**: MultiIndex raises GAE1012
**Method**: Pass DataFrame with MultiIndex
**Assertion**: GAE1012 raised with flatten fix

### test_decision_function_model_error()

**Goal**: SVM without predict_proba raises GAE1008
**Method**: Train LinearSVC, call from_model()
**Assertion**: GAE1008 raised with CalibratedClassifierCV fix

### test_auc_omitted_without_proba()

**Goal**: AUC metrics excluded if no y_proba
**Method**: Call from_predictions() without y_proba
**Assertion**: `"roc_auc"` not in result.performance.keys()

### test_auc_access_error_without_proba()

**Goal**: Accessing AUC raises GAE1009
**Method**: Get result without y_proba, access .roc_auc
**Assertion**: GAE1009 raised with y_proba fix

### test_performance_metrics_gae1009_override()

**Goal**: PerformanceMetrics raises GAE1009 for proba metrics
**Method**: Access each of ["roc_auc", "pr_auc", "brier_score", "log_loss"]
**Assertions**:

- GAE1009 raised (not GAE1002)
- Message mentions "probability predictions"
- Docs link present

### test_missing_policy_validation()

**Goal**: Unsupported policy raises GAE1005
**Method**: Pass `missing_policy="exclude_unknown"`
**Assertion**: GAE1005 raised with v0.3 message

### test_multiclass_rejection()

**Goal**: Non-binary classification raises GAE1004
**Method**: Pass y with 3 classes
**Assertion**: GAE1004 raised with binary filter fix

### test_from_config_reproduction()

**Goal**: from_config() reproduces result
**Method**: Generate result, save config, reload
**Assertion**: `result.id == result_reproduced.id`

### test_from_config_data_hash_mismatch()

**Goal**: Changed data raises GAE2003
**Method**: Modify X after saving config, reload
**Assertion**: GAE2003 raised with hash comparison

### test_from_config_result_id_mismatch()

**Goal**: Changed config raises GAE2002
**Method**: Manually edit expected_result_id in config
**Assertion**: GAE2002 raised with troubleshooting link

---

## Phase 4: Hashing Tests

### Canonicalization Tests

#### test_canonical_hash_nan()

**Goal**: NaN converts to null
**Method**: Canonicalize `{"a": np.nan, "b": [1.0, np.nan]}`
**Assertions**:

- `canonical["a"] is None`
- `canonical["b"] == [1.0, None, 3.0]`
- `json.dumps(canonical, allow_nan=False)` succeeds

#### test_canonical_hash_neg_zero()

**Goal**: -0.0 normalizes to 0.0
**Method**: Canonicalize -0.0 and np.float64(-0.0)
**Assertion**: Both return `0.0`

#### test_canonical_hash_infinity()

**Goal**: Infinity uses sentinel
**Method**: Canonicalize np.inf and -np.inf
**Assertions**:

- `canonicalize(np.inf) == {"__float__": "Infinity"}`
- `canonicalize(-np.inf) == {"__float__": "-Infinity"}`
- `json.dumps(canonical, allow_nan=False)` succeeds

#### test_canonical_hash_ndarray()

**Goal**: Arrays preserve dtype/shape
**Method**: Canonicalize int32 and int64 arrays
**Assertions**:

- `canonical["__ndarray__"]["dtype"] == "int32"`
- `canonical["__ndarray__"]["shape"] == [2, 2]`
- Different dtypes produce different canonicals

#### test_canonical_hash_bytes()

**Goal**: Bytes encode to base64
**Method**: Canonicalize `b"hello world"`
**Assertion**: Result is base64-encoded string

#### test_canonical_dict_sorted_keys()

**Goal**: Dict keys sorted
**Method**: Canonicalize `{"z": 1, "a": 2}`
**Assertion**: JSON serialization has "a" before "z"

### Data Hashing Tests

#### test_data_hash_prefix()

**Goal**: Hash includes "sha256:" prefix
**Method**: Hash simple DataFrame
**Assertions**:

- `hash.startswith("sha256:")`
- `len(hash) == 71` (7 + 64 hex chars)

#### test_data_hash_dtype_sensitivity()

**Goal**: dtype changes hash
**Method**: Hash int32 vs int64 array with same values
**Assertion**: `hash(int32) != hash(int64)`

#### test_data_hash_column_order()

**Goal**: Column order changes hash
**Method**: Hash `{"a": [1], "b": [2]}` vs `{"b": [2], "a": [1]}`
**Assertion**: Different hashes

#### test_data_hash_range_index()

**Goal**: RangeIndex fast path (3x speedup)
**Method**: Hash 1M-row DataFrame with RangeIndex vs materialized
**Assertions**:

- Both produce hashes
- RangeIndex is >3x faster

#### test_data_hash_categorical()

**Goal**: Categories change hash
**Method**: Hash categorical with different category sets
**Assertion**: Different categories → different hash

#### test_data_hash_timezone()

**Goal**: Timezone normalization to UTC
**Method**: Hash UTC vs EST timestamps (same instant)
**Assertions**:

- UTC and EST produce same hash
- Naive treated as UTC (same hash as explicit UTC)

#### test_data_hash_string_na()

**Goal**: NA in strings handled consistently
**Method**: Hash string column with pd.NA, np.nan, None
**Assertion**: All map to same sentinel "\x00NA"

#### test_data_hash_boolean()

**Goal**: Boolean hashed as uint8
**Method**: Hash boolean Series
**Assertion**: Deterministic hash

#### test_data_hash_timedelta()

**Goal**: Timedelta as int64 ns
**Method**: Hash timedelta Series
**Assertion**: Deterministic hash

### Result ID Tests

#### test_result_id_determinism()

**Goal**: Same inputs produce same ID
**Method**: Generate audit 10 times with same seed
**Assertion**: All 10 IDs identical

#### test_result_id_sensitivity()

**Goal**: Changes propagate to ID
**Method**: Change one metric value by 1e-6
**Assertion**: Different ID

---

## Phase 5: Error Tests

### test_all_errors_have_code()

**Goal**: Every GlassAlphaError has code
**Method**: Grep codebase for `raise GlassAlphaError`
**Assertion**: All have `code="GAExxxx"`

### test_all_errors_have_fix()

**Goal**: Every error has actionable fix
**Method**: Trigger each error code
**Assertion**: `exc.fix` is non-empty string

### test_all_errors_have_docs()

**Goal**: Every error has docs link
**Method**: Trigger each error code
**Assertion**: `exc.docs` starts with "https://glassalpha.com"

### test_error_str_formatting()

**Goal**: Error string is readable
**Method**: Trigger GAE1003
**Assertions**:

- Header: "GlassAlpha Error [GAE1003]"
- Message present
- "How to fix:" section
- "Documentation:" section

---

## Phase 6: Atomic Write Tests

### test_atomic_write_cleanup()

**Goal**: .tmp file removed on failure
**Method**: Write with failing function
**Assertion**: .tmp file doesn't exist after exception

### test_atomic_write_success()

**Goal**: Final file exists after success
**Method**: Write with successful function
**Assertions**:

- Final file exists
- .tmp file doesn't exist
- Content correct

### test_atomic_write_binary_mode()

**Goal**: Binary mode works
**Method**: Write binary data with mode="wb"
**Assertion**: Data read back correctly

---

## Phase 7: Integration Tests

### test_full_audit_workflow()

**Goal**: End-to-end from_model() to PDF
**Method**: Train model, generate audit, export all formats
**Assertions**:

- result.id present
- PDF generated
- JSON generated
- Config generated
- Manifest generated

### test_reproduction_workflow()

**Goal**: Config → reproduction → verification
**Method**: Generate audit, save config, reproduce
**Assertions**:

- Reproduced ID matches original
- Metrics within tolerance

### test_german_credit_smoke()

**Goal**: Real dataset works
**Method**: Load German Credit, train XGBoost, audit
**Assertion**: No errors, all sections present

### test_adult_income_smoke()

**Goal**: Second real dataset works
**Method**: Load Adult Income, train model, audit
**Assertion**: No errors, all sections present

---

## Phase 8: Performance Tests

### test_import_speed_benchmark()

**Goal**: Track import speed over time
**Method**: Time import 100 times, report p50/p95
**Assertion**: p95 < 250ms (buffer for CI variance)

### test_audit_speed_german_credit()

**Goal**: Audit generation <5 seconds
**Method**: Time full audit on German Credit
**Assertion**: `duration < 5.0`

### test_hash_speed_large_dataframe()

**Goal**: Hashing 1M rows <1 second
**Method**: Time hash_data_for_manifest() on 1M x 10 DataFrame
**Assertion**: `duration < 1.0`

---

## Phase 9: Cross-Platform Tests (CI)

### test_determinism_linux_macos()

**Goal**: Same ID on Linux + macOS
**Method**: Generate audit on both, compare IDs
**Assertion**: IDs match (if exact), or `equals()` passes (if tolerance)

### test_float_tolerance_cross_platform()

**Goal**: Tolerance policy works
**Method**: Generate audit on different platforms
**Assertion**: `result1.equals(result2, rtol=1e-5)`

---

## Coverage Requirements

### Critical Paths (100% coverage required)

- `compute_result_id()`
- `canonicalize()`
- `hash_data_for_manifest()`
- `_normalize_protected_attributes()`
- `_get_probabilities()`
- `GlassAlphaError.__str__()`

### High Priority (95%+ coverage)

- `from_model()`
- `from_predictions()`
- `from_config()`
- `AuditResult` methods
- `ReadonlyMetrics` access patterns

### Medium Priority (90%+ coverage)

- Plot methods
- Export methods (to_json, to_pdf)
- Atomic write helpers

---

## Test Execution Strategy

### Local Development

```bash
# Run all API tests
pytest tests/api/ -v

# Run contracts only
pytest tests/api/ -v -m contract

# Run determinism only
pytest tests/api/ -v -m determinism

# Run with coverage
pytest tests/api/ --cov=src/glassalpha/api --cov-report=html
```

### CI Pipeline

```yaml
- name: API Contract Tests
  run: pytest tests/api/test_contracts.py -v --tb=short

- name: Determinism Tests (10 runs)
  run: |
    for i in {1..10}; do
      pytest tests/api/test_determinism.py -v
    done

- name: Coverage Check
  run: |
    pytest tests/api/ --cov=src/glassalpha/api --cov-fail-under=95
```

---

## Quality Gates

Before tagging v0.2:

1. **All contract tests pass** (0 failures)
2. **Determinism verified** (10 runs, 0 collisions)
3. **Coverage >95%** for api/ module
4. **Import speed <200ms** (p95)
5. **All errors tested** (every GAExxxx code)
6. **Cross-platform CI green** (Linux + macOS)

---

## Test Maintenance

### Adding New Tests

1. Place in appropriate file (contracts vs determinism vs errors)
2. Add marker if critical path
3. Update coverage report
4. Document edge case in test docstring

### Fixing Flaky Tests

1. Identify root cause (race condition, platform difference)
2. Add tolerance if floating point
3. Add retry if external dependency
4. Document workaround in test

### Deprecating Tests

1. Mark with `@pytest.mark.skip(reason="Superseded by X")`
2. Remove after 1 release cycle
3. Update coverage baselines

---

## Success Metrics

Track in CI dashboard:

- **Test count**: 60+ tests in tests/api/
- **Coverage**: 95%+ for api/ module
- **Speed**: All tests complete in <60 seconds
- **Flakiness**: <1% failure rate over 100 runs
- **Determinism**: 0 ID collisions in 1000 runs

**Final validation**: Run full test suite 10 times on CI. All must pass.
