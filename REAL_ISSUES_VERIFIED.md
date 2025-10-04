# GlassAlpha - Verified Issues & Fixes Needed

**Verification Date:** October 3, 2025
**Method:** Actual testing + code inspection
**Previous Error:** Made unfounded recommendations without verification

---

## ✅ What I Got WRONG in Previous Analysis

### ❌ WRONG: "Add glassalpha doctor command"

**Reality:** `glassalpha doctor` already EXISTS and WORKS PERFECTLY
**Verified:**

```bash
$ glassalpha doctor
GlassAlpha Environment Check
========================================
Environment
  Python: 3.13.7
  OS: Darwin arm64

Core Features (always available)
--------------------
  ✅ LogisticRegression (scikit-learn)
  ✅ NoOp explainers (baseline)
  ✅ HTML reports (jinja2)
  ✅ Basic metrics (performance, fairness)

Optional Features
--------------------
  SHAP + tree models: ✅ installed
    (includes SHAP, XGBoost, LightGBM)
  ...
```

**Lesson:** Should have checked existing CLI commands before recommending additions

---

## 🔴 VERIFIED Critical Issues (Confirmed via Testing)

### Issue #1: CLI `models` Command Crashes

**Status:** ✅ CONFIRMED BROKEN
**Root Cause:** CLI imports `ModelRegistry` from `plugin_registry.py` which lacks `available_plugins()` method

**Evidence:**

```python
# Error shows it's using wrong module:
# <glassalpha.core.plugin_registry.PluginRegistry object at 0x109495a90>

# The method EXISTS in registry.py but NOT in plugin_registry.py
```

**Files Involved:**

- `packages/src/glassalpha/cli/main.py:237` - imports from plugin_registry
- `packages/src/glassalpha/core/plugin_registry.py` - missing method
- `packages/src/glassalpha/core/registry.py:151` - HAS the method

**Fix:** Change import in CLI from `plugin_registry` to correct registry module

---

### Issue #2: Dataset Fetch Command Crashes

**Status:** ✅ CONFIRMED BROKEN
**Error:** `ImportError: cannot import name '_ensure_dataset_availability'`
**Location:** `packages/src/glassalpha/cli/datasets.py:91`

**Test Result:**

```bash
$ glassalpha datasets fetch german_credit
ImportError: cannot import name '_ensure_dataset_availability' from 'glassalpha.pipeline.audit'
```

**Root Cause:** Function doesn't exist or wrong import path

---

### Issue #3: LightGBM Detection Failure

**Status:** ✅ CONFIRMED BROKEN
**Evidence:**

```bash
$ pip list | grep lightgbm
lightgbm                 4.6.0

$ glassalpha audit --config test_lightgbm_config.yaml --output test.html
Model 'lightgbm' not available. Falling back to 'logistic_regression'.
```

**Test:** With lightgbm installed, config specifying `model.type: lightgbm` still uses LogisticRegression

**Root Cause:** TBD - needs debugging of model registration/detection logic

---

## ✅ VERIFIED Working Features (Don't Fix What Isn't Broken)

1. ✅ `glassalpha doctor` - Works perfectly
2. ✅ `glassalpha audit` - Core functionality works
3. ✅ `glassalpha datasets list` - Lists available datasets
4. ✅ `glassalpha --help` - Shows all commands
5. ✅ `glassalpha --version` - Shows version
6. ✅ LogisticRegression - Works with included & custom data
7. ✅ XGBoost - Works with included & custom data
8. ✅ Report generation (HTML) - Creates beautiful reports
9. ✅ Manifest generation - Complete provenance tracking
10. ✅ Custom data loading - CSV files work seamlessly

---

## 🟡 POTENTIAL Issues (Need Verification)

### Potential Issue #1: SHAP Dependency

**Claim:** SHAP not auto-installed, causes explainer failures
**Status:** ⚠️ NEEDS VERIFICATION
**Question:** Is SHAP supposed to be a core dependency or optional?

**To Verify:**

- Check `pyproject.toml` - where is SHAP declared?
- Is this intentional (optional) or a bug?
- What's the intended user experience?

---

### Potential Issue #2: CLI Argument Naming

**Claim:** `--out` vs `--output` inconsistency
**Status:** ⚠️ NEEDS VERIFICATION
**Test Result:** `--out` fails, `--output` works
**Question:** Is this a bug or are docs using wrong flag?

**To Verify:**

- Check if `--out` should be supported as alias
- Or are docs/examples using wrong flag?

---

## 📋 ACTIONABLE FIX CHECKLIST

### 🔴 Critical Fix #1: CLI Models Command

**Files:**

- `packages/src/glassalpha/cli/main.py:237`

**Options:**
A. Change import from `plugin_registry` to correct module
B. Add `available_plugins()` method to `plugin_registry.py`
C. Use different method that exists in both

**Recommended:** Option A - use the correct module

**Steps:**

1. Find where `ModelRegistry` is properly exported from
2. Update CLI import to use that
3. Test: `glassalpha models` should list models
4. Verify output shows installation status correctly

---

### 🔴 Critical Fix #2: Dataset Fetch

**Files:**

- `packages/src/glassalpha/cli/datasets.py:91`
- Possibly `packages/src/glassalpha/pipeline/audit.py`

**Steps:**

1. Search codebase for `_ensure_dataset_availability` or similar function
2. If exists, fix import path
3. If doesn't exist, implement or use alternative
4. Test: `glassalpha datasets fetch german_credit`
5. Verify dataset downloads to correct location

---

### 🔴 Critical Fix #3: LightGBM Detection

**Files:** TBD - needs investigation

**Debug Steps:**

1. Check if LightGBMWrapper is registered correctly
2. Verify import guards work when lightgbm is installed
3. Check model selection logic in train pipeline
4. Add logging to see why fallback is triggered
5. Fix registration or detection logic

**Test:**

```bash
pip install lightgbm
glassalpha audit --config test_lightgbm_config.yaml --output test.html
# Should use LightGBM, not fall back to LogisticRegression
```

---

## 🟢 Nice-to-Have Improvements (ONLY IF TIME PERMITS)

### Optional #1: Better Error Messages

**Status:** Low priority - current errors are functional
**Example:** When explainer fails, show install command

### Optional #2: Progress Indicators

**Status:** Low priority - feature request, not a bug
**Example:** Show progress during long operations

### Optional #3: Add `--out` Alias

**Status:** Low priority - `--output` works fine
**Only if:** Docs consistently use `--out` and changing docs is harder

---

## 🧪 Testing Protocol for Each Fix

Before marking a fix complete:

1. **Unit Test:** Add/update test case
2. **Integration Test:** Full workflow test
3. **Regression Test:** Ensure nothing broke
4. **User Test:** Try it as a new user would
5. **Documentation:** Update if CLI behavior changes

---

## 📊 Priority Matrix

| Issue              | Impact | Effort | Priority           |
| ------------------ | ------ | ------ | ------------------ |
| CLI models command | High   | Low    | 🔴 DO FIRST        |
| Dataset fetch      | High   | Medium | 🔴 DO SECOND       |
| LightGBM detection | High   | High   | 🔴 DO THIRD        |
| SHAP dependency    | Medium | Low    | 🟡 If time permits |
| CLI arg alias      | Low    | Low    | 🟢 Nice to have    |
| Error messages     | Low    | Medium | 🟢 Nice to have    |
| Progress bars      | Low    | High   | 🔵 Future          |

---

## ✅ Lessons Learned

1. **Verify before recommending** - Check if feature exists first
2. **Test each claim** - Don't assume, actually test it
3. **Check actual imports** - Error messages show which module is used
4. **Distinguish bugs from features** - `doctor` wasn't missing, I didn't check

---

**Next Steps:**

1. Get approval on this verified list
2. Fix critical issues in order
3. Test each fix thoroughly
4. Don't add features that already exist

**Estimated Time:**

- Fix #1 (models command): 15 minutes
- Fix #2 (dataset fetch): 30-45 minutes
- Fix #3 (LightGBM detection): 1-2 hours
- **Total Critical Fixes:** 2-3 hours
