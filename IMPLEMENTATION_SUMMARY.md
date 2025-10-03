# CLI Performance & Explainer Fix - Implementation Summary

## Status: Phase 2.5 Core Complete (60%), Ready for Testing

## ✅ Completed (6/43 tasks)

### Phase 2.5: Critical Explainer API Fix

1. **✅ Base Class Standardization** (`explain/base.py`)

   - Added `@classmethod is_compatible(cls, *, model=None, model_type=None, config=None)`
   - All args keyword-only to prevent signature drift
   - Clear interface contract with documentation

2. **✅ TreeSHAPExplainer Fixed** (`explain/shap/tree.py`)

   - Converted to `@classmethod`
   - Supports both `model` and `model_type` parameters
   - Handles tree model detection (XGBoost, LightGBM, RandomForest, etc.)

3. **✅ KernelSHAPExplainer Fixed** (`explain/shap/kernel.py`)

   - Converted to `@classmethod`
   - Returns `True` for unknown models (model-agnostic fallback)
   - Checks SUPPORTED_MODELS set

4. **✅ NoOpExplainer Fixed** (`explain/noop.py`)

   - Added missing `@classmethod is_compatible`
   - Always returns `True` (universal fallback)

5. **✅ CoefficientsExplainer Fixed** (`explain/coefficients.py`)

   - Added missing `@classmethod is_compatible`
   - Checks for linear models (LogisticRegression, etc.)
   - Fallback checks `coef_` attribute

6. **✅ Registry Updated** (`explain/registry.py`)
   - Calls `is_compatible` with keywords: `model=..., model_type=..., config=None`
   - Graceful TypeError handling with legacy fallbacks
   - Detailed logging for debugging signature issues

## 🔄 Next Critical Tasks (Required for Working System)

### Phase 2.5 Remaining (30 min)

**Task perf-2.5.4.1**: Create Contract Test

```python
# File: packages/tests/test_explainer_registry_contract.py
# Purpose: Validate all explainers have correct is_compatible signature
# Test: Call each explainer's is_compatible with keywords, verify no TypeError
```

**Task perf-2.5.5.1**: Create Smoke Test

```python
# File: packages/tests/test_audit_smoke.py
# Purpose: End-to-end test that audit completes without explainer errors
# Test: Run audit on german_credit_simple.yaml, verify PDF generates
```

**Tasks perf-2.5.4.2, 2.5.5.2-3**: Run Tests

- Execute contract test - fix any failing explainers
- Execute smoke test - verify no TypeError in logs
- Validate explainer selection works

### Phase 1: Performance Quick Win (10 min)

**Task perf-1.1**: Move sklearn Import

```python
# File: packages/src/glassalpha/data/tabular.py
# Change: Move `from sklearn.model_selection import train_test_split`
#         from line 19 (module level) to inside split_data() method
# Impact: Reduces --help time by ~365ms (58%)
```

**Tasks perf-1.2-1.3**: Test Performance

- Measure: `time glassalpha --help` (should be ~270ms, down from 635ms)
- Run test suite to verify no regressions

### Phase 2: Lazy Loading (1 hour)

**Tasks perf-2.2.1-2.2.3**: Lazy Dataset Commands

- Remove eager import of dataset functions in `cli/main.py` line 109
- Create lazy wrapper functions that import inside command
- Test all dataset commands work

**Tasks perf-2.3.1-2.3.3**: `__getattr__` Pattern

- Refactor `datasets/__init__.py` to use `__getattr__` for lazy imports
- Preserve backward compatibility (`from glassalpha.datasets import load_german_credit`)
- Test imports work

### Phase 5: Performance Tests (30 min)

**Task perf-5.1.1**: Create Performance Test Suite

```python
# File: packages/tests/test_cli_performance.py
# Tests:
# - test_help_is_fast() - verify --help < 300ms (600ms on CI)
# - test_version_is_instant() - verify --version < 100ms
```

**Task perf-5.1.3**: Add Smoke Test to Suite

- Reuse test_audit_smoke.py in performance suite
- Ensures performance optimizations don't break functionality

**Task perf-5.2.1**: CI Integration

- Add performance tests to CI workflow
- Set appropriate thresholds for CI environment

## 📊 Expected Results

| Metric           | Before   | After Phase 1 | After Phase 2 | Target    |
| ---------------- | -------- | ------------- | ------------- | --------- |
| Explainer API    | BROKEN   | FIXED ✅      | FIXED ✅      | FIXED ✅  |
| `--help` latency | 635ms    | ~270ms        | <200ms        | <300ms ✅ |
| Audit works      | ❌ CRASH | ✅ Works      | ✅ Works      | ✅ Works  |
| Contract test    | N/A      | ✅ PASS       | ✅ PASS       | ✅ PASS   |

## 📝 Files Modified

### Completed Changes:

1. `packages/src/glassalpha/explain/base.py` - Interface standardization
2. `packages/src/glassalpha/explain/shap/tree.py` - TreeSHAP fix
3. `packages/src/glassalpha/explain/shap/kernel.py` - KernelSHAP fix
4. `packages/src/glassalpha/explain/noop.py` - NoOp fix
5. `packages/src/glassalpha/explain/coefficients.py` - Coefficients fix
6. `packages/src/glassalpha/explain/registry.py` - Registry update

### Files to Create:

7. `packages/tests/test_explainer_registry_contract.py` - NEW
8. `packages/tests/test_audit_smoke.py` - NEW
9. `packages/tests/test_cli_performance.py` - NEW

### Files to Modify Next:

10. `packages/src/glassalpha/data/tabular.py` - sklearn import
11. `packages/src/glassalpha/cli/main.py` - lazy commands
12. `packages/src/glassalpha/datasets/__init__.py` - **getattr** pattern

## ⚡ Recommended Next Steps

### Option A: Complete & Test Phase 2.5 (30 min)

**Priority**: CRITICAL - Verify explainer fixes work

```bash
cd /Users/gabe/Sites/glassalpha/packages
source venv/bin/activate

# 1. Create contract test
# 2. Create smoke test
# 3. Run both tests
pytest tests/test_explainer_registry_contract.py -v
pytest tests/test_audit_smoke.py -v -s

# 4. If tests pass, Phase 2.5 is DONE ✅
```

### Option B: Continue with Quick Wins (10 min)

**Priority**: HIGH - Get measurable performance improvement

```bash
# 1. Move sklearn import in data/tabular.py
# 2. Test performance
time glassalpha --help  # Should be ~270ms
```

### Option C: Full Implementation (2-3 hours)

**Priority**: COMPLETE - Finish all critical tasks

- Complete Phase 2.5 testing
- Implement Phase 1 (sklearn)
- Implement Phase 2 (lazy loading)
- Implement Phase 5 (performance tests)
- Run full validation

## 🐛 Known Issues / Risks

1. **Contract test might find additional explainers** with wrong signature

   - **Mitigation**: Registry has fallback handling, will log warnings

2. **Smoke test might reveal other audit issues** beyond explainer

   - **Mitigation**: Fix as discovered, but explainer was main blocker

3. **Performance improvements might break functionality**

   - **Mitigation**: Run full test suite after each phase

4. **Lazy loading might affect help text rendering**
   - **Mitigation**: Test all commands after implementing lazy loading

## 📚 Documentation Needed

- Update CHANGELOG.md with explainer API fix
- Update CHANGELOG.md with performance improvements
- Add performance guidelines to CONTRIBUTING.md
- Document `is_compatible` signature requirements

## 🎯 Success Criteria

- [ ] All explainer contract tests pass
- [ ] Audit smoke test completes without errors
- [ ] `--help` latency < 300ms
- [ ] Full test suite passes
- [ ] No TypeError in audit logs
- [ ] Performance tests in CI

## 💡 Key Learnings

1. **Signature drift is dangerous** - Standardizing early prevents crashes
2. **Performance matters for CLI** - 635ms feels broken, <300ms feels snappy
3. **Lazy loading requires patterns** - `__getattr__` preserves backward compat
4. **Testing prevents regressions** - Contract + smoke + performance tests essential
5. **Graceful degradation** - Registry fallbacks prevent hard failures

---

**Generated**: 2025-01-02
**Phase**: Implementation
**Status**: Ready for Testing & Continuation
