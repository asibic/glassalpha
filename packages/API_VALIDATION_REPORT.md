# API Documentation Validation Report

**Status**: ⚠️ **MAJOR DISCREPANCIES FOUND**

**Date**: Phase 2 Pre-Launch Validation

---

## Executive Summary

The API documentation created for `site/docs/reference/api/` does NOT accurately reflect the current codebase. **Critical updates required before launch.**

### Impact Level: **HIGH**

- 3 notebooks reference non-existent API
- All API docs reference incorrect signatures
- Users will encounter `AttributeError` and `ImportError` on documented methods

---

## Detailed Discrepancies

### 1. Module-Level API (DOES NOT EXIST)

**Documented**:

```python
import glassalpha as ga
result = ga.audit.from_model(model, X_test, y_test, ...)
```

**Actual Reality**:

```python
# __init__.py exports NOTHING for audit
__all__ = ["__version__"]  # audit, explain, recourse NOT exported
```

**Issue**: Users cannot `import glassalpha as ga` and call `ga.audit.from_model()`.

**Actual working import**:

```python
from glassalpha.pipeline.audit import AuditPipeline
result = AuditPipeline.from_model(model, X_test, y_test, ...)
```

---

### 2. `from_model()` Signature MISMATCH

**Documented in api-audit.md**:

```python
def from_model(
    model,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    protected_attributes: dict[str, pd.Series | np.ndarray] | None = None,  # ❌ WRONG
    feature_names: list[str] | None = None,
    target_name: str = "target",
    threshold: float = 0.5,  # ❌ WRONG parameter name
    random_seed: int | None = None,
    explainer_samples: int = 1000,  # ❌ DOESN'T EXIST
    include_calibration: bool = True,  # ❌ DOESN'T EXIST
    include_fairness: bool = True,  # ❌ DOESN'T EXIST
    include_explanations: bool = True  # ❌ DOESN'T EXIST
) -> AuditResult  # ❌ WRONG - should be AuditResults
```

**Actual signature (line 149-164 in audit.py)**:

```python
@classmethod
def from_model(
    cls,
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    protected_attributes: list[str],  # ✅ list of column names, NOT dict
    *,
    random_seed: int = 42,  # ✅ different default
    audit_profile: str = "tabular_compliance",  # ✅ exists but not documented
    explainer: str | None = None,  # ✅ exists but not documented
    fairness_threshold: float | None = None,  # ✅ different name
    recourse_config: dict | None = None,  # ✅ exists but not documented
    feature_names: list[str] | None = None,  # ✅ correct
    target_name: str | None = None,  # ✅ correct
    **config_overrides: Any,  # ✅ exists but not documented
) -> AuditResults:  # ✅ correct (with 's')
```

**Key differences**:

1. `protected_attributes` is `list[str]` (column names), NOT `dict[str, array]`
2. Parameters `include_*` DON'T EXIST
3. `threshold` should be `fairness_threshold`
4. `explainer_samples` DOESN'T EXIST (maps to config, not direct param)
5. Return type is `AuditResults` (with 's'), not `AuditResult`

---

### 3. `AuditResult` vs `AuditResults` Class Name

**Documented**: `AuditResult` (singular)

**Actual**: `AuditResults` (plural) - line 32 in audit.py

**Impact**: All docs reference wrong class name.

---

### 4. Result Object Methods (NOT VALIDATED)

**Documented methods**:

- `result.to_pdf(filepath)` - ❓ NOT FOUND
- `result.to_json(filepath)` - ❓ NOT FOUND
- `result.to_config(filepath)` - ❓ NOT FOUND
- `result.summary()` - ❓ NOT FOUND
- `result.display()` - ❓ NOT FOUND (but `_repr_html_()` exists)

**Actual methods on `AuditResults`** (line 32-100):

- ✅ `_repr_html_()` - inline display (DOES EXIST)
- ❌ NO public methods for PDF, JSON, config export found in class definition

**Critical issue**: Documented export methods DON'T EXIST on `AuditResults` class.

---

### 5. Plot Methods (NOT VALIDATED)

**Documented**:

```python
result.performance.plot_confusion_matrix()
result.fairness.plot_group_metrics()
result.calibration.plot()
result.explanations.plot_importance()
```

**Actual**:

- `AuditResults` is a `@dataclass` with dict attributes (line 32-56)
- Attributes are `dict[str, Any]`, NOT objects with methods
- NO plot methods found

**Example from actual code**:

```python
self.results.model_performance = {}  # Plain dict
self.results.fairness_analysis = {}  # Plain dict
self.results.explanations = {}  # Plain dict
```

**Impact**: Users cannot call `.plot()` methods as documented.

---

### 6. Attribute Access Pattern MISMATCH

**Documented**:

```python
result.performance.accuracy  # Object with attributes
result.fairness.demographic_parity_difference  # Object with attributes
result.calibration.expected_calibration_error  # Object with attributes
```

**Actual**:

```python
result.model_performance["accuracy"]  # Dict access
result.fairness_analysis["..."]  # Dict access
# NO nested objects, everything is dict[str, Any]
```

---

## Required Actions

### Priority 0 (BLOCKING LAUNCH)

1. **Update all API docs** to match actual signature:

   - Change `ga.audit.from_model()` to `AuditPipeline.from_model()`
   - Fix `protected_attributes` parameter (list, not dict)
   - Remove non-existent parameters (`include_*`, `explainer_samples`)
   - Fix return type (`AuditResults` with 's')

2. **Update all notebooks** (3 files):

   - `quickstart_colab.ipynb`
   - `german_credit_walkthrough.ipynb`
   - `custom_data_template.ipynb`
   - Change import to `from glassalpha.pipeline.audit import AuditPipeline`
   - Change calls to `AuditPipeline.from_model(...)`
   - Fix `protected_attributes` usage

3. **Remove or mark aspirational**:
   - All plot method documentation
   - All `to_pdf()`, `to_json()`, `to_config()` documentation
   - All nested attribute access (`.performance.accuracy`)

### Priority 1 (BEFORE DISTRIBUTION)

4. **Decide on actual API strategy**:

   - Option A: Document current reality (`AuditPipeline.from_model()`, dict access)
   - Option B: Implement the documented API (create wrappers, export `ga.audit`)
   - Option C: Hybrid (implement some, defer others)

5. **Add API contract tests** to prevent future drift

### Priority 2 (NICE TO HAVE)

6. **Implement missing features** (if desired):
   - Export convenience API in `__init__.py`
   - Add `to_pdf()`, `to_json()` methods to `AuditResults`
   - Create result wrapper classes with `.plot()` methods
   - Add nested attribute access via properties

---

## Recommendations

### Immediate Fix (2-4 hours)

**Option A: Document Reality** (fastest, safest)

1. Update API docs to match actual code:

   - Import: `from glassalpha.pipeline.audit import AuditPipeline`
   - Call: `AuditPipeline.from_model(...)`
   - Access: `result.model_performance["accuracy"]` (dict access)
   - Remove: All plot methods, export methods

2. Update notebooks to match

3. Add warning: "Convenience API coming in v0.2"

**Pros**: No code changes, accurate docs, unblocked for launch
**Cons**: Less polished API, requires longer imports

---

### Future Implementation (10-20 hours)

**Option B: Implement Documented API**

Create thin wrapper in `__init__.py`:

```python
# glassalpha/__init__.py
from .pipeline.audit import AuditPipeline

class AuditAPI:
    @staticmethod
    def from_model(model, X_test, y_test, protected_attributes=None, **kwargs):
        # Convert dict to list if needed for backwards compat
        if isinstance(protected_attributes, dict):
            # Protected attrs passed as dict - extract from dataframe
            pass
        return AuditPipeline.from_model(model, X_test, y_test, protected_attributes, **kwargs)

audit = AuditAPI()
__all__ = ["audit", "__version__"]
```

Add methods to `AuditResults`:

```python
def to_pdf(self, filepath: str):
    from glassalpha.report.renderer import AuditReportRenderer
    renderer = AuditReportRenderer()
    renderer.render_pdf(self, filepath)
```

**Pros**: Matches documented API, better UX
**Cons**: More work, requires testing, delays launch

---

## Validation Checklist

Before updating docs, verify:

- [ ] Actual import path for `from_model()`
- [ ] Exact signature and parameter types
- [ ] Return type and class name
- [ ] Available methods on result object
- [ ] Attribute access pattern (dict vs object)
- [ ] Existence of plot methods
- [ ] Existence of export methods
- [ ] What's in `__init__.py` exports

---

## Impact on Existing Work

### Notebooks Created

- ❌ All 3 notebooks use non-working API
- ⚠️ Will fail immediately on `import glassalpha as ga`
- ⚠️ Will fail on `ga.audit.from_model()`
- ⚠️ Will fail on `protected_attributes` dict usage

### API Documentation Created

- ❌ api-audit.md: 80% incorrect
- ❌ api-config.md: Not validated (likely issues)
- ❌ api-pipeline.md: Not validated (likely issues)

### Configuration Patterns

- ⚠️ May reference incorrect API usage

---

## Decision Required

**Which option should we pursue?**

1. ✅ **Document reality** (fast, launch-ready)
2. ⏳ **Implement documented API** (better UX, delays launch)
3. 🔀 **Hybrid** (implement some, defer others)

**Recommendation**: **Option 1** for immediate launch, plan Option 2 for v0.2.

---

## Next Steps

1. **STOP** - Do not proceed with remaining tasks until API is clarified
2. **DECIDE** - Choose option (1, 2, or 3)
3. **UPDATE** - Fix docs and notebooks to match reality OR implement missing API
4. **VALIDATE** - Test notebooks end-to-end with actual code
5. **RESUME** - Continue with remaining validation tasks

---

**Validator**: AI Agent (Sonnet 4.5)
**Date**: Pre-launch validation
**Files Examined**:

- `packages/src/glassalpha/__init__.py`
- `packages/src/glassalpha/pipeline/audit.py` (lines 1-2383)
- `site/docs/reference/api/api-audit.md`
- `examples/notebooks/*.ipynb` (3 files)
