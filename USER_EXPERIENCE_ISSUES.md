# User Experience Testing - Issues & Improvement Checklist

**Test Date:** October 3, 2025
**Test Scope:** Clean install, CLI usage, included data, custom data across all models
**Test Environment:** macOS with clean Python 3.13 virtual environment

---

## 🔴 Critical Issues (Must Fix Before Release)

### Issue #1: CLI `models` Command Broken

**Status:** 🔴 CRITICAL
**Error:** `AttributeError: 'PluginRegistry' object has no attribute 'available_plugins'`
**Location:** `packages/src/glassalpha/cli/main.py:242`
**Impact:** Users cannot see available models via CLI
**Reproducible:** Yes - 100% failure rate

**Steps to Reproduce:**

```bash
glassalpha models
```

**Expected Behavior:** List available models (logistic_regression, xgboost, lightgbm, etc.)
**Actual Behavior:** Crashes with AttributeError

**Root Cause:** CLI code calls `ModelRegistry.available_plugins()` but the method doesn't exist
**Fix Required:** Either add the method or use correct registry API

---

### Issue #2: Dataset Fetch Command Broken

**Status:** 🔴 CRITICAL
**Error:** `ImportError: cannot import name '_ensure_dataset_availability' from 'glassalpha.pipeline.audit'`
**Location:** `packages/src/glassalpha/cli/datasets.py:91`
**Impact:** Users cannot fetch built-in datasets via CLI
**Reproducible:** Yes - 100% failure rate

**Steps to Reproduce:**

```bash
glassalpha datasets fetch german_credit
```

**Expected Behavior:** Downloads and caches german_credit dataset
**Actual Behavior:** Crashes with ImportError

**Root Cause:** Function `_ensure_dataset_availability` doesn't exist or is not exported
**Fix Required:** Fix import or refactor dataset fetching logic

---

### Issue #3: LightGBM Model Not Detected

**Status:** 🔴 CRITICAL
**Error:** `Model 'lightgbm' not available. Falling back to 'logistic_regression'`
**Location:** Model registry/detection logic
**Impact:** LightGBM doesn't work even when installed - falls back to logistic regression
**Reproducible:** Yes - 100% failure rate

**Steps to Reproduce:**

```bash
pip install lightgbm
glassalpha audit --config test_lightgbm_config.yaml --output test.html
```

**Expected Behavior:** Uses LightGBM model
**Actual Behavior:** Falls back to LogisticRegression despite lightgbm being installed

**Root Cause:** Model registry not detecting lightgbm package or wrapper not registering
**Fix Required:** Fix model registration/detection logic

---

## 🟡 High Priority Issues (Should Fix Soon)

### Issue #4: Missing SHAP Dependency Handling

**Status:** 🟡 HIGH
**Error:** `ImportError: SHAP library required for KernelSHAP explainer`
**Location:** `packages/src/glassalpha/explain/shap/kernel.py:138`
**Impact:** Explainers fail without helpful guidance
**Reproducible:** Yes - when SHAP not installed

**Current Behavior:** Error message is clear but requires manual pip install
**Better Behavior:** Either:

1. Auto-install SHAP when needed (pip install on demand)
2. Make SHAP a core dependency
3. Provide clearer installation instructions in error message

**Recommendation:** Make SHAP a core dependency since it's needed for most explainers

---

### Issue #5: CLI Argument Inconsistency

**Status:** 🟡 HIGH
**Error:** `No such option: --out Did you mean --output?`
**Location:** CLI argument parsing
**Impact:** User confusion, poor UX
**Reproducible:** Yes

**Problem:** Documentation and examples use `--out` but actual flag is `--output`
**Fix Required:**

- Either support both `--out` and `--output` as aliases
- Or standardize all docs to use `--output`
- Add better error suggestions

---

## 🟢 Medium Priority Improvements

### Improvement #1: Better Model Fallback Messages

**Status:** 🟢 MEDIUM
**Current:** `Model 'lightgbm' not available. Falling back to 'logistic_regression'. To enable 'lightgbm', run: pip install 'glassalpha[lightgbm]'`

**Issues:**

- Message appears even when lightgbm IS installed (Issue #3 root cause)
- Suggested command doesn't work with clean install
- No verification that fallback makes sense

**Improvement:**

- Check if package is actually installed before suggesting install
- Verify fallback is compatible with use case
- Warn if fallback changes expected behavior significantly

---

### Improvement #2: Enhanced Error Messages

**Status:** 🟢 MEDIUM

**Areas Needing Better Errors:**

1. When explainers fail - provide installation commands
2. When datasets are missing - show fetch command
3. When configs are invalid - show examples
4. When dependencies are missing - list specific packages needed

**Example Good Error:**

```
❌ SHAP library required for TreeSHAP explainer

To fix this, install SHAP:
  pip install shap

Or install all explainer dependencies:
  pip install 'glassalpha[explain]'
```

---

### Improvement #3: Configuration Validation

**Status:** 🟢 MEDIUM

**Current Issues:**

- Warnings about protected attributes come late in process
- Small datasets (n<20) not flagged early
- Invalid model parameters only fail during training

**Improvements:**

- Early validation of configuration file
- Check dataset size requirements upfront
- Validate model parameters before starting pipeline
- Suggest better configurations for edge cases

---

## 🔵 Low Priority Enhancements

### Enhancement #1: Progress Indicators

**Status:** 🔵 LOW

**Current:** Minimal feedback during long operations
**Enhancement:** Add progress bars for:

- Dataset downloading
- Model training
- SHAP computation
- Report generation

---

### Enhancement #2: CLI Enhancements

**Status:** 🔵 LOW

**Ideas:**

1. `glassalpha quickstart` - Interactive wizard for first audit
2. `glassalpha validate-config` - Check config before running
3. `glassalpha examples` - Show example configs for common scenarios
4. Better tab completion support

---

### Enhancement #3: Documentation Improvements

**Status:** 🔵 LOW

**Areas:**

1. Add "Common Issues" troubleshooting guide
2. Document all CLI commands with examples
3. Show example workflow for each model type
4. Add dependency matrix (what installs what)

---

## ✅ What Works Well (Keep As Is)

1. **Core Audit Pipeline** - Works flawlessly with LogisticRegression and XGBoost
2. **Custom Data Support** - Seamless CSV loading and processing
3. **Report Generation** - HTML reports are professional and complete
4. **Configuration System** - YAML configs are intuitive and well-structured
5. **Clean Installation** - `pip install -e packages/` works perfectly
6. **Included Data** - German Credit dataset loads correctly
7. **CLI Interface** - Professional appearance and good help text
8. **Manifest Generation** - Comprehensive provenance tracking

---

## 📊 Test Results Summary

| Component          | Included Data | Custom Data | Status                  |
| ------------------ | ------------- | ----------- | ----------------------- |
| LogisticRegression | ✅ Perfect    | ✅ Perfect  | 🟢 **Production Ready** |
| XGBoost            | ✅ Perfect    | ✅ Perfect  | 🟢 **Production Ready** |
| LightGBM           | ❌ Broken     | ❌ Broken   | 🔴 **Needs Fix**        |
| CLI Models         | ❌ Broken     | N/A         | 🔴 **Needs Fix**        |
| CLI Dataset Fetch  | ❌ Broken     | N/A         | 🔴 **Needs Fix**        |
| CLI Audit          | ✅ Works      | ✅ Works    | 🟢 **Production Ready** |
| Report Generation  | ✅ Perfect    | ✅ Perfect  | 🟢 **Production Ready** |

---

## 🎯 Recommended Fix Priority

### Phase 1: Critical Bugs (Block Release)

- [ ] Fix Issue #1: CLI models command
- [ ] Fix Issue #2: Dataset fetch command
- [ ] Fix Issue #3: LightGBM detection

### Phase 2: High Priority (Before v1.0)

- [ ] Fix Issue #4: SHAP dependency
- [ ] Fix Issue #5: CLI argument consistency
- [ ] Improvement #2: Enhanced error messages

### Phase 3: Polish (Post v1.0)

- [ ] Improvement #1: Better fallback messages
- [ ] Improvement #3: Configuration validation
- [ ] Enhancement #1: Progress indicators
- [ ] Enhancement #2: CLI enhancements
- [ ] Enhancement #3: Documentation

---

## 📝 Notes for Developers

**Testing Methodology Used:**

1. Fresh Python 3.13 virtual environment
2. Clean install: `pip install -e packages/`
3. Sequential dependency installation (base → xgboost → lightgbm → shap)
4. Tested CLI commands, included data, and custom data
5. Each model tested with identical workflows

**Test Coverage:**

- ✅ Clean installation
- ✅ Basic CLI commands
- ✅ Included dataset (german_credit)
- ✅ Custom CSV data
- ✅ All three model types
- ✅ Report generation
- ✅ Manifest creation

**Not Tested (Out of Scope):**

- PDF output (tested HTML only)
- Multiclass classification
- Enterprise features
- Dashboard commands
- Monitoring commands
- Multiple datasets simultaneously

---

**Generated:** 2025-10-03
**Tester:** Automated user experience testing
**Environment:** macOS, Python 3.13.7, Clean virtual environment
