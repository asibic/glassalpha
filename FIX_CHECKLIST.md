# GlassAlpha - User Experience Fixes Checklist

**Priority Level Guide:**

- 🔴 **CRITICAL** - Blocks release, must fix immediately
- 🟡 **HIGH** - Should fix before v1.0
- 🟢 **MEDIUM** - Quality improvement, fix when possible
- 🔵 **LOW** - Nice to have, can defer

---

## 🔴 Phase 1: Critical Bugs (MUST FIX - Blocks Release)

### ❌ Task 1.1: Fix CLI `models` Command

**Priority:** 🔴 CRITICAL
**Estimated Time:** 30 minutes
**Files to Edit:**

- `packages/src/glassalpha/cli/main.py` (line 242)
- `packages/src/glassalpha/core/registry.py` (if adding method)

**Subtasks:**

- [ ] Identify correct method name in PluginRegistry
- [ ] Either:
  - [ ] Option A: Add `available_plugins()` method to PluginRegistry
  - [ ] Option B: Update CLI to use correct existing method
- [ ] Test: `glassalpha models` should list available models
- [ ] Verify output format is user-friendly
- [ ] Add test case for models command

**Acceptance Criteria:**

- Command runs without error
- Lists at least: logistic_regression, xgboost (if installed), lightgbm (if installed)
- Shows installation instructions for unavailable models

---

### ❌ Task 1.2: Fix Dataset Fetch Command

**Priority:** 🔴 CRITICAL
**Estimated Time:** 45 minutes
**Files to Edit:**

- `packages/src/glassalpha/cli/datasets.py` (line 91)
- `packages/src/glassalpha/pipeline/audit.py` (if function exists but not exported)

**Subtasks:**

- [ ] Locate the `_ensure_dataset_availability` function or equivalent
- [ ] Either:
  - [ ] Option A: Fix import path
  - [ ] Option B: Refactor to use public API
  - [ ] Option C: Move function to appropriate module
- [ ] Test: `glassalpha datasets fetch german_credit` should work
- [ ] Verify dataset is cached correctly
- [ ] Test with forced re-download: `glassalpha datasets fetch german_credit --force`
- [ ] Add test case for dataset fetching

**Acceptance Criteria:**

- Command runs without error
- Dataset downloads to correct cache location
- Shows progress or confirmation message
- Handles network errors gracefully

---

### ❌ Task 1.3: Fix LightGBM Model Detection

**Priority:** 🔴 CRITICAL
**Estimated Time:** 1-2 hours
**Files to Investigate:**

- `packages/src/glassalpha/models/tabular/lightgbm.py`
- `packages/src/glassalpha/core/registry.py`
- `packages/src/glassalpha/pipeline/train.py`

**Subtasks:**

- [ ] Check if LightGBMWrapper is registered with correct decorator
- [ ] Verify import guards are working correctly
- [ ] Check registry lookup logic
- [ ] Test model availability detection
- [ ] Debug why installed lightgbm package isn't detected
- [ ] Fix registration or detection logic
- [ ] Test: Create config with `model.type: lightgbm` and verify it uses LightGBM
- [ ] Add integration test for LightGBM model selection

**Acceptance Criteria:**

- With lightgbm installed, `model.type: lightgbm` uses LightGBM not LogisticRegression
- Model shows correctly in `glassalpha models` (after Task 1.1)
- No fallback warnings when lightgbm is available
- Works with both included and custom data

**Debug Steps:**

```python
# Test script to debug registration
from glassalpha.core import ModelRegistry
print("Registered models:", ModelRegistry.get_all())

# Try importing wrapper directly
from glassalpha.models.tabular.lightgbm import LightGBMWrapper
print("LightGBMWrapper:", LightGBMWrapper)

# Check if lightgbm is installed
import importlib.util
spec = importlib.util.find_spec("lightgbm")
print("lightgbm installed:", spec is not None)
```

---

## 🟡 Phase 2: High Priority (Fix Before v1.0)

### ❌ Task 2.1: Make SHAP a Core Dependency

**Priority:** 🟡 HIGH
**Estimated Time:** 30 minutes
**Files to Edit:**

- `packages/pyproject.toml`
- Documentation (installation guides)

**Subtasks:**

- [ ] Move `shap` from optional `[explain]` dependencies to core dependencies
- [ ] Update installation docs to reflect change
- [ ] Remove conditional SHAP imports from explainers (make them required)
- [ ] Update error messages (no longer need "install shap" messages)
- [ ] Regenerate requirements/constraints files if needed
- [ ] Test clean install includes SHAP

**Acceptance Criteria:**

- `pip install glassalpha` installs SHAP automatically
- TreeSHAP and KernelSHAP work immediately after install
- No ImportError for SHAP in normal usage

**Alternative (If keeping SHAP optional):**

- [ ] Improve error message with clear install instructions
- [ ] Auto-detect SHAP and suggest better explainer alternatives
- [ ] Make error messages more actionable

---

### ❌ Task 2.2: Fix CLI Argument Consistency

**Priority:** 🟡 HIGH
**Estimated Time:** 20 minutes
**Files to Edit:**

- `packages/src/glassalpha/cli/main.py` (audit command)
- Documentation examples

**Subtasks:**

- [ ] Add `--out` as alias for `--output` in audit command
- [ ] Update help text to show both options
- [ ] Grep docs for `--out` usage and update or keep both
- [ ] Test both `--out` and `--output` work identically
- [ ] Consider adding aliases for other common abbreviations

**Acceptance Criteria:**

- Both `--out` and `--output` work for audit command
- Help text shows: `--output, --out`
- No confusing error messages

---

### ❌ Task 2.3: Enhanced Error Messages

**Priority:** 🟡 HIGH
**Estimated Time:** 1-2 hours
**Files to Edit:**

- Multiple explainer files
- Model loaders
- Dataset loaders
- Configuration validators

**Subtasks:**

- [ ] Create helper function for actionable error messages
- [ ] Update explainer errors to show install commands
- [ ] Update model errors to show availability and install steps
- [ ] Update dataset errors to show fetch commands
- [ ] Add examples to config validation errors
- [ ] Test error messages are helpful and accurate

**Example Template:**

```python
def actionable_error(error_type, component, fix_command=None, docs_link=None):
    """Generate user-friendly error with fix instructions."""
    message = f"❌ {error_type}: {component}\n\n"
    if fix_command:
        message += f"To fix this:\n  {fix_command}\n\n"
    if docs_link:
        message += f"Documentation: {docs_link}\n"
    return message
```

**Acceptance Criteria:**

- Error messages include fix instructions
- Commands are copy-pasteable
- Links to docs when appropriate
- User can resolve error without searching

---

## 🟢 Phase 3: Medium Priority (Quality Improvements)

### ❌ Task 3.1: Better Model Fallback Logic

**Priority:** 🟢 MEDIUM
**Estimated Time:** 1 hour

**Subtasks:**

- [ ] Verify package installation before suggesting fallback
- [ ] Check if fallback is semantically compatible
- [ ] Warn when fallback significantly changes behavior
- [ ] Log detailed fallback reason for debugging
- [ ] Add option to fail instead of fallback (`strict_model: true`)

**Acceptance Criteria:**

- Only suggests install when package truly missing
- Warns if fallback changes results significantly
- Provides debug information in verbose mode

---

### ❌ Task 3.2: Early Configuration Validation

**Priority:** 🟢 MEDIUM
**Estimated Time:** 2 hours

**Subtasks:**

- [ ] Add config pre-flight check function
- [ ] Validate dataset size requirements
- [ ] Check model parameters before training
- [ ] Warn about missing protected attributes early
- [ ] Suggest better configs for edge cases
- [ ] Add `--validate-only` flag to audit command

**Acceptance Criteria:**

- Validation happens before expensive operations
- Clear warnings for suboptimal configs
- Suggestions are actionable

---

### ❌ Task 3.3: Add `glassalpha doctor` Command

**Priority:** 🟢 MEDIUM
**Estimated Time:** 2 hours

**Subtasks:**

- [ ] Create doctor command that checks:
  - [ ] Python version
  - [ ] Core dependencies installed
  - [ ] Optional dependencies status
  - [ ] Model availability
  - [ ] Explainer availability
  - [ ] Cache directory writable
  - [ ] Network connectivity (for dataset fetch)
- [ ] Show version info
- [ ] Provide fix suggestions
- [ ] Add to CLI help

**Example Output:**

```
GlassAlpha Environment Check
✅ Python 3.13.7
✅ Core dependencies installed
✅ LogisticRegression available
✅ XGBoost available (v3.0.5)
❌ LightGBM not available - Run: pip install lightgbm
✅ SHAP available (v0.48.0)
✅ Cache directory writable: ~/.glassalpha/
⚠️  No network connection - dataset fetch may fail
```

---

## 🔵 Phase 4: Low Priority (Nice to Have)

### ❌ Task 4.1: Progress Indicators

**Priority:** 🔵 LOW
**Estimated Time:** 2-3 hours

**Subtasks:**

- [ ] Add tqdm progress bars for:
  - [ ] Dataset downloading
  - [ ] Model training (epochs/iterations)
  - [ ] SHAP computation (samples)
  - [ ] Report generation (sections)
- [ ] Make progress bars conditional (disable in CI/non-TTY)
- [ ] Add `--quiet` flag to suppress progress

---

### ❌ Task 4.2: Interactive Quickstart

**Priority:** 🔵 LOW
**Estimated Time:** 3-4 hours

**Subtasks:**

- [ ] Create `glassalpha quickstart` command
- [ ] Interactive prompts for:
  - [ ] Model type selection
  - [ ] Dataset (built-in or custom)
  - [ ] Output format
- [ ] Generate config file
- [ ] Run audit automatically
- [ ] Show next steps

---

### ❌ Task 4.3: Documentation Improvements

**Priority:** 🔵 LOW
**Estimated Time:** 4-6 hours

**Subtasks:**

- [ ] Create "Common Issues" troubleshooting guide
- [ ] Document all CLI commands with examples
- [ ] Add workflow examples for each model type
- [ ] Create dependency matrix table
- [ ] Add FAQ section
- [ ] Create video tutorials (optional)

---

## 🧪 Testing Checklist

After each fix, verify:

### Smoke Tests

- [ ] Clean install: `pip install -e packages/`
- [ ] Basic audit: `glassalpha audit --config quickstart.yaml`
- [ ] CLI help: `glassalpha --help`
- [ ] All models available: `glassalpha models`

### Integration Tests

- [ ] LogisticRegression with included data
- [ ] LogisticRegression with custom data
- [ ] XGBoost with included data
- [ ] XGBoost with custom data
- [ ] LightGBM with included data
- [ ] LightGBM with custom data

### Regression Tests

- [ ] Run existing test suite: `pytest`
- [ ] No new linter errors: `ruff check`
- [ ] Type checking passes: `mypy --strict`

---

## 📋 Definition of Done

A task is complete when:

- [ ] Code changes implemented
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Smoke tests pass
- [ ] Code review approved (if applicable)
- [ ] Merged to main branch

---

## 🎯 Success Metrics

After all critical fixes:

- ✅ All CLI commands work without errors
- ✅ All three model types work with included data
- ✅ All three model types work with custom data
- ✅ Clean install requires no manual fixes
- ✅ Error messages are actionable
- ✅ Zero regression in existing functionality

---

**Created:** 2025-10-03
**Status:** Ready for Review
**Estimated Total Time:** Phase 1 (Critical): 3-4 hours, Phase 2 (High): 3-4 hours
