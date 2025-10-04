# Verified Fix Checklist - User Experience Issues

**Status:** Ready for review and approval
**Scope:** Only verified, critical issues that actually exist
**Estimated Total Time:** 2-3 hours for all critical fixes

---

## 🔴 Critical Fixes (Must Complete)

### [ ] Task 1: Fix CLI `models` Command

**Priority:** 🔴 CRITICAL
**Estimated Time:** 15 minutes
**Status:** Ready to fix

**Problem:**

- CLI imports from `plugin_registry.py` which lacks `available_plugins()` method
- Method exists in `registry.py` but CLI uses wrong module
- Error: `AttributeError: 'PluginRegistry' object has no attribute 'available_plugins'`

**Files to Change:**

- [ ] `packages/src/glassalpha/cli/main.py` (line 237)

**Fix Steps:**

1. [ ] Check where `ModelRegistry` is properly exported in `core/__init__.py`
2. [ ] Update import in `main.py:237` from:
   ```python
   from ..core.plugin_registry import ModelRegistry
   ```
   To use the correct import (likely from `..core import ModelRegistry`)
3. [ ] Test: `glassalpha models` should list available models
4. [ ] Verify output shows proper installation status for each model

**Acceptance Criteria:**

- [ ] Command runs without AttributeError
- [ ] Lists logistic_regression, xgboost, lightgbm with correct status
- [ ] Shows installation instructions for unavailable models

**Test Command:**

```bash
glassalpha models
```

---

### [ ] Task 2: Fix Dataset Fetch Command

**Priority:** 🔴 CRITICAL
**Estimated Time:** 30-45 minutes
**Status:** Needs investigation

**Problem:**

- Command crashes: `ImportError: cannot import name '_ensure_dataset_availability'`
- Function doesn't exist in `glassalpha.pipeline.audit` or is private

**Files to Investigate:**

- [ ] `packages/src/glassalpha/cli/datasets.py` (line 91)
- [ ] `packages/src/glassalpha/pipeline/audit.py` (search for dataset functions)
- [ ] `packages/src/glassalpha/data/` (may have dataset utilities)

**Fix Steps:**

1. [ ] Search codebase for `_ensure_dataset_availability` or similar
   ```bash
   grep -r "_ensure_dataset_availability" packages/src/
   grep -r "ensure_dataset" packages/src/
   grep -r "dataset.*avail" packages/src/
   ```
2. [ ] If function exists elsewhere:
   - [ ] Update import path in `datasets.py`
3. [ ] If function doesn't exist:
   - [ ] Check what `glassalpha audit` uses for dataset loading
   - [ ] Use that same logic or refactor to shared function
   - [ ] Or implement minimal fetch function
4. [ ] Test: `glassalpha datasets fetch german_credit`
5. [ ] Verify: Dataset downloads to `~/.glassalpha/data/`
6. [ ] Test with `--force` flag for re-download

**Acceptance Criteria:**

- [ ] Command runs without ImportError
- [ ] Dataset downloads successfully
- [ ] Shows progress or completion message
- [ ] Cached dataset is reused on second run (unless `--force`)

**Test Commands:**

```bash
# First fetch
glassalpha datasets fetch german_credit

# Should use cache
glassalpha datasets fetch german_credit

# Force re-download
glassalpha datasets fetch german_credit --force
```

---

### [ ] Task 3: Fix LightGBM Model Detection

**Priority:** 🔴 CRITICAL
**Estimated Time:** 1-2 hours
**Status:** Needs debugging

**Problem:**

- LightGBM installed but not detected
- Falls back to LogisticRegression incorrectly
- Message: `Model 'lightgbm' not available. Falling back to 'logistic_regression'`

**Files to Investigate:**

- [ ] `packages/src/glassalpha/models/tabular/lightgbm.py`
- [ ] `packages/src/glassalpha/core/registry.py` or `plugin_registry.py`
- [ ] `packages/src/glassalpha/pipeline/train.py`
- [ ] Model registration decorators

**Debug Steps:**

1. [ ] Check if LightGBMWrapper has registration decorator
   ```python
   # Look for: @ModelRegistry.register("lightgbm")
   ```
2. [ ] Test registration directly:
   ```python
   from glassalpha.core import ModelRegistry
   print("Registered models:", ModelRegistry.names())
   print("Has lightgbm:", ModelRegistry.has("lightgbm"))
   ```
3. [ ] Check import guards in `lightgbm.py`:
   ```python
   # Should only register if lightgbm is installed
   try:
       import lightgbm
   except ImportError:
       # Should skip registration
   ```
4. [ ] Check model selection logic in `train.py`:
   - [ ] How does it detect available models?
   - [ ] Why does it fall back when lightgbm is available?
5. [ ] Add debug logging to see where detection fails
6. [ ] Fix the root cause (registration or detection)

**Acceptance Criteria:**

- [ ] With `pip install lightgbm`, model is detected
- [ ] Config with `model.type: lightgbm` uses LightGBM
- [ ] No fallback warning when lightgbm is available
- [ ] `glassalpha models` shows lightgbm as available
- [ ] Works with both included and custom data

**Test Commands:**

```bash
# Ensure lightgbm is installed
pip install lightgbm

# Verify detection
python -c "from glassalpha.core import ModelRegistry; print(ModelRegistry.names())"

# Test with config
glassalpha audit --config test_lightgbm_config.yaml --output test.html

# Verify it used LightGBM (not LogisticRegression)
grep -i "logistic" test.html  # Should not appear
grep -i "lightgbm" test.html  # Should appear
```

---

## 🟡 Optional Improvements (Only If Time Permits)

### [ ] Optional 1: Verify SHAP Dependency Handling

**Priority:** 🟡 MEDIUM (if time)
**Time:** 15 minutes to verify, 30 minutes to fix if needed

**Questions to Answer:**

1. [ ] Is SHAP intentionally optional or should it be core?
2. [ ] Check `pyproject.toml` - where is shap listed?
3. [ ] Are error messages clear when SHAP is missing?
4. [ ] Should `pip install glassalpha` include SHAP by default?

**Decision Points:**

- **If SHAP is intentional optional:** Error messages are fine, no fix needed
- **If SHAP should be core:** Move from `[explain]` to core dependencies

---

### [ ] Optional 2: Add CLI Argument Aliases

**Priority:** 🟢 LOW (nice to have)
**Time:** 15 minutes

**Question:** Should `--out` be supported as alias for `--output`?

**Check:**

1. [ ] Do official docs use `--out` or `--output`?
2. [ ] Do example configs use `--out` or `--output`?
3. [ ] What do users expect?

**If fixing:**

```python
# In audit command
output: Path = typer.Option(..., "--output", "--out", help="Output file path")
```

---

## ✅ Do NOT Fix (Already Works)

- ✅ `glassalpha doctor` - Already exists and works perfectly
- ✅ `glassalpha audit` - Core functionality works
- ✅ `glassalpha datasets list` - Works correctly
- ✅ LogisticRegression - No issues
- ✅ XGBoost - No issues
- ✅ Report generation - No issues
- ✅ Custom data loading - No issues

---

## 🧪 Testing Checklist

After each fix, run these tests:

### Smoke Tests

```bash
# Basic functionality
glassalpha --version
glassalpha --help
glassalpha doctor

# Fixed commands
glassalpha models
glassalpha datasets list
glassalpha datasets fetch german_credit

# End-to-end
glassalpha audit --config packages/configs/quickstart.yaml --output test.html
```

### Regression Tests

```bash
cd packages
pytest tests/
ruff check src/
mypy --strict src/
```

### Integration Tests

1. [ ] Test LogisticRegression with included data
2. [ ] Test XGBoost with included data
3. [ ] Test LightGBM with included data (after fix)
4. [ ] Test each with custom CSV data

---

## 📊 Success Metrics

Fix is complete when:

- [ ] All 3 critical tasks completed
- [ ] All smoke tests pass
- [ ] All regression tests pass
- [ ] Manual integration tests pass
- [ ] No new issues introduced

---

## 🎯 Definition of Done

Each task is done when:

1. [ ] Code fix implemented
2. [ ] Manual test passes (listed in task)
3. [ ] No new linter errors introduced
4. [ ] Regression tests still pass
5. [ ] Committed with clear message

---

## 📝 Notes for Developer

**What I learned:**

- Always verify existing functionality before recommending additions
- Test each issue claim before including in checklist
- Check actual module imports (error messages show which module)
- Distinguish between bugs and missing features

**Files that definitely need changes:**

1. `packages/src/glassalpha/cli/main.py:237` - Wrong import
2. `packages/src/glassalpha/cli/datasets.py:91` - Wrong/missing import
3. TBD for LightGBM (needs investigation)

**Total estimated time:** 2-3 hours for all critical fixes

---

**Created:** 2025-10-03
**Status:** ✅ READY FOR REVIEW AND APPROVAL
**Previous version:** Discarded (contained unverified recommendations)
