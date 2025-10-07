# Pre-Launch Codebase Review

**Last Chance for Breaking Changes**
**Date:** 2025-10-07
**Purpose:** Systematic review before backwards compatibility lock-in

---

## Executive Summary

**Scope:** 12-phase review covering architecture, APIs, config, CLI, abstractions, testing, dependencies, performance, naming, dead code, documentation, and prioritization.

**Key Findings:**

- **✅ Strong foundation:** Plugin architecture, registry patterns, and core abstractions are solid
- **⚠️ API surface needs cleanup:** Duplicate interfaces, inconsistent naming, unused code
- **⚠️ Configuration has technical debt:** Legacy compat shims, deprecated options
- **⚠️ Documentation cleanup needed:** Temp files, outdated content
- **✅ Dependencies well-managed:** Minimal core, clear optional extras
- **✅ Security practices sound:** No PII leaks, proper path validation

**Overall Assessment:** **READY FOR LAUNCH** with **targeted cleanup** (estimated ~150-200k tokens of work)

---

## Phase 1: Architecture & Extensibility Review

### ✅ Strengths

1. **Plugin Registry Pattern:** Well-implemented across all component types

   - `PluginRegistry` base class with discovery
   - Lazy loading prevents heavy imports
   - Entry points correctly configured in `pyproject.toml`

2. **Protocol-Based Interfaces:** Using `Protocol` classes (PEP 544) correctly

   - `ModelInterface`, `ExplainerInterface`, `MetricInterface`, `DataInterface`, `AuditProfileInterface`
   - Runtime checkable with `@runtime_checkable`

3. **Capability Detection:** Registry supports `supports` metadata for compatibility

   - Explainers declare compatible model types
   - Selection is deterministic via priority ordering

4. **Configuration-Driven:** Component selection driven by YAML config
   - Explainer priority list
   - Metric categories
   - Audit profiles

### ⚠️ Issues Found

#### **I1.1: Duplicate DataInterface Definitions** ⚠️ High Priority

**Location:** `core/interfaces.py` (line 128) vs `data/base.py` (line 24)

**Problem:** Two different `DataInterface` definitions with incompatible signatures:

- `core/interfaces.py`: Uses `Protocol` with `load(path: str) -> Any`
- `data/base.py`: Uses `ABC` with `load(path: Path, schema: DataSchema | None) -> pd.DataFrame`

**Impact:** Confusing for implementers, violates single source of truth

**Recommendation:**

```python
# DELETE: core/interfaces.py lines 128-190 (DataInterface Protocol)
# KEEP: data/base.py (ABC with concrete interface)
# REASON: The ABC version has more detailed requirements and is actually used
```

**Tradeoff:** None - this is pure tech debt removal. The ABC version is superior.

---

#### **I1.2: Legacy ExplainerRegistryCompat Wrapper** ⚠️ Medium Priority

**Location:** `explain/registry.py` lines 527-578

**Problem:** Unused compatibility wrapper for old API

- Delegates to real `ExplainerRegistry`
- Not exported in `__all__`
- No external users found in codebase

**Recommendation:** **DELETE** entire `ExplainerRegistryCompat` class

**Tradeoff:** Risk if external code depends on this (low risk, not in exports)

**Before doing this:** Search PyPI for any published packages importing this

---

#### **I1.3: Inconsistent Model Type Naming** ⚠️ Low Priority

**Location:** Multiple files

**Problem:** Model types use different naming conventions:

- `xgboost` (lowercase, underscores)
- `logistic_regression` (lowercase, underscores)
- `LogisticRegressionWrapper` (PascalCase class name)
- `sklearn_generic` (lowercase, underscores)

**Current state:** Mostly consistent (lowercase with underscores)

**Recommendation:** **ACCEPT AS-IS**

- External-facing keys are consistent (lowercase_underscore)
- Internal class names follow Python conventions (PascalCase)
- No change needed

---

### ✅ Architecture Validation

**Framework extensibility:** ✅ Pass

- New model types can be added without core changes
- New explainers register via entry points or decorators
- New metrics follow same pattern

**LLM/Vision readiness:** ✅ Pass

- `ModelInterface` is modality-agnostic (uses generic `predict` signature)
- `DataInterface` abstractions exist (though need cleanup per I1.1)
- Report templates are separate from logic

**Enterprise boundaries:** ✅ Pass

- `check_feature()` decorator ready for licensing
- Registry supports `enterprise` metadata flag
- Feature gating infrastructure in place

---

## Phase 2: Public API Surface Review

### Current API Surface

**Python API (exported from `glassalpha/__init__.py`):**

```python
__all__ = [
    "__version__",
    # Coming: "audit", "explain", "recourse"
]
```

**Observation:** Intentionally minimal exports - good for v0.1.0

**CLI Commands:**

```bash
glassalpha audit          # Core
glassalpha validate       # Config validation
glassalpha list           # List components
glassalpha doctor         # Environment check
glassalpha docs           # Open docs
glassalpha init           # Init config
glassalpha quickstart     # Template project
glassalpha reasons        # Reason codes (E2)
glassalpha recourse       # Recourse (E2.5)
glassalpha datasets       # Dataset operations
glassalpha prep           # Preprocessing artifacts
glassalpha models         # Show available models
glassalpha dashboard      # Enterprise stub
glassalpha monitor        # Enterprise stub
```

### ⚠️ Issues Found

#### **I2.1: Minimal Python API Exports** ⚠️ Decision Needed

**Location:** `glassalpha/__init__.py`

**Current state:** Only exports `__version__`, comments say `audit`, `explain`, `recourse` coming

**Question:** What should the v1.0 Python API look like?

**Options:**

**Option A: CLI-First (current approach)**

```python
# Minimal exports, encourage CLI usage
__all__ = ["__version__"]
```

- Pro: Forces users through validated CLI paths
- Pro: Simpler to maintain backwards compatibility
- Con: Less flexible for programmatic use

**Option B: Rich Python API**

```python
# Full programmatic access
from glassalpha import audit, explain, recourse, AuditResult
result = audit(config="audit.yaml", output="report.pdf")
```

- Pro: Better for notebooks, scripts, automation
- Pro: More Pythonic
- Con: Larger API surface to maintain

**Option C: Hybrid (middle ground)**

```python
# Expose pipeline but not internals
from glassalpha import AuditPipeline, from_model
pipeline = AuditPipeline.from_config("audit.yaml")
result = pipeline.run()
```

- Pro: Balances flexibility and control
- Pro: Natural for notebook usage (already have `from_model()`)
- Con: Still requires maintenance of public API

**Current phase2_priorities.mdc says:** Notebooks use `from_model()` API (Option C)

**Recommendation:** **Go with Option C (Hybrid)**

- Export: `AuditPipeline`, `from_model`, `AuditResult`, `__version__`
- Keep internal registries/interfaces private
- CLI remains primary interface for production
- Python API enables notebook exploration

**Tradeoff:** Larger API surface but aligns with phase 2 notebook strategy

---

#### **I2.2: CLI Command Organization** ✅ Good Shape

**Location:** `cli/main.py`

**Current structure:**

- Main commands at top level (`audit`, `validate`, `doctor`, etc.)
- Command groups for related ops (`datasets`, `prep`, `dashboard`, `monitor`)
- Enterprise stubs in place

**Assessment:** Well-organized, scales cleanly

**Recommendation:** **NO CHANGES NEEDED**

---

### API Stability Assessment

**Breaking change risk areas:**

1. **Config schema** (covered in Phase 3)
2. **CLI command names/flags** - stable, no changes needed
3. **Python API** - not locked in yet (see I2.1)
4. **Report format** - PDF structure is stable via templates

---

## Phase 3: Configuration & Schema Review

### Current Config Structure

```yaml
audit_profile: "tabular_compliance"
model:
  type: "xgboost"
  path: "model.joblib"
data:
  dataset: "german_credit"
  protected_attributes: ["age", "sex"]
explainers:
  strategy: "first_compatible"
  priority: ["treeshap", "kernelshap"]
metrics:
  performance: ["accuracy", "precision", "recall"]
  fairness: ["demographic_parity"]
report:
  template: "standard_audit"
reproducibility:
  random_seed: 42
```

### ✅ Strengths

1. **Pydantic Validation:** All configs use Pydantic v2 with `extra="forbid"`
2. **Nested Structure:** Logical grouping (model, data, explainers, etc.)
3. **Field Validators:** Comprehensive validation logic
4. **Extensible:** Easy to add new sections without breaking existing configs

### ⚠️ Issues Found

#### **I3.1: Deprecated Config Options Still Supported** ⚠️ Medium Priority

**Location:** `config/warnings.py` lines 43-86

**Problem:** Legacy flat keys still accepted and warned about:

```python
deprecated_mappings = {
    "random_seed": ("reproducibility.random_seed", "reproducibility"),
    "target": ("data.target_column", "data"),
    "protected_attrs": ("data.protected_attributes", "data"),
    "model_type": ("model.type", "model"),
    # ... more
}
```

**Impact:** Technical debt, confusing for new users

**Recommendation:** **REMOVE** deprecated key support before v1.0

**Migration path:**

1. Document deprecated keys in CHANGELOG under "Breaking Changes"
2. Provide migration script: `glassalpha config migrate old.yaml > new.yaml`
3. Delete `warn_deprecated_options()` and `_migrate_deprecated()` functions

**Tradeoff:**

- **Pro:** Cleaner config schema, less code to maintain
- **Con:** Breaks any configs using old keys
- **Mitigation:** No published v0.x versions, so no external users yet

**Decision:** **REMOVE NOW** (pre-launch is the right time)

---

#### **I3.2: AuditConfig.**init** Has Commented Logic** ⚠️ Low Priority

**Location:** `config/schema.py` lines 531-535

```python
def __init__(self, **data):
    # Profile defaults are applied in the loader, not here
    # Schema should only validate, not mutate inputs
    super().__init__(**data)
```

**Problem:** Comment suggests profile defaults should be in loader, not schema init

**Recommendation:** **VERIFY** that profile defaults are indeed in loader

- If yes: Delete the `__init__` override (unnecessary)
- If no: Move logic to loader and then delete override

**Tradeoff:** None - this is just removing unnecessary code

---

#### **I3.3: Config Schema Supports Future Needs** ✅ Pass

**LLM/Vision extensibility check:**

```yaml
# Hypothetical future config
model:
  type: "gpt4" # ✅ Works - just needs new model wrapper
  params:
    temperature: 0.7 # ✅ Works - params is dict[str, Any]
data:
  dataset: "custom"
  path: "prompts.jsonl" # ✅ Works - custom path supported
  modality: "text" # ❓ Need to add modality field
explainers:
  priority: ["attention", "gradient"] # ✅ Works - just needs new explainers
```

**Missing piece:** `data.modality` field for explicit modality declaration

**Recommendation:** **ADD OPTIONAL FIELD** `data.modality` to `DataConfig`

```python
modality: str | None = Field(
    None,
    description="Data modality: 'tabular', 'text', 'image', 'multimodal'",
    pattern="^(tabular|text|image|multimodal)$"
)
```

**Tradeoff:**

- **Pro:** Makes future LLM/vision support explicit
- **Pro:** No breaking change (optional field)
- **Con:** Not needed for v1.0 tabular focus
- **Decision:** **DEFER to Phase 3** (when LLM support added)

---

## Phase 4: CLI Structure & UX Review

### Command Organization ✅ Good

Already covered in Phase 2 - no issues found

### Error Messages

#### **I4.1: Error Message Quality Audit** ✅ Pass with Minor Issues

**Sample errors checked:**

**Good example (from `cli/error_formatter.py`):**

```python
def format_missing_dependency_error(feature: str, package: str) -> str:
    return (
        f"Missing optional dependency for '{feature}' feature.\n"
        f"Install with: pip install 'glassalpha[{package}]'"
    )
```

✅ States what failed, why, how to fix

**Bad example found:**

```python
# In some test files
raise ValueError("Invalid input")  # Too vague
```

**Recommendation:** Audit all `ValueError`, `RuntimeError` raises for specificity

- Search: `grep -r "raise ValueError\|raise RuntimeError" packages/src/`
- Ensure each includes: what failed, why, how to fix

**ETM:** ~20k tokens to audit and improve error messages

---

### Exit Codes ✅ Standardized

**Location:** `cli/exit_codes.py`

```python
class ExitCode:
    SUCCESS = 0
    USER_ERROR = 1          # Config errors, missing files
    RUNTIME_ERROR = 2        # Unexpected failures
    VALIDATION_ERROR = 3     # Validation failures
```

**Assessment:** Clean, follows conventions

**Recommendation:** NO CHANGES

---

## Phase 5: Data & Model Abstractions Review

### ⚠️ Critical Finding

#### **I5.1: DataInterface Abstraction Leaks Tabular Assumptions** ⚠️ High Priority

**Location:** `data/base.py`

**Problem:** Methods assume pandas DataFrames everywhere:

```python
def load(self, path: Path, schema: DataSchema | None = None) -> pd.DataFrame:
    """Returns DataFrame - what about images/text?"""

def extract_features_target(
    self,
    data: pd.DataFrame,  # ❌ Tabular assumption
    schema: DataSchema,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame | None]:
    """All return types are tabular"""
```

**Impact:** Will require breaking changes when adding LLM/vision support

**Recommendation:** **Fix before v1.0 if possible, otherwise document limitation**

**Option A: Generic data container (breaking change)**

```python
from typing import TypeVar
T = TypeVar('T')  # Could be DataFrame, list[str], list[Image], etc.

class DataInterface(ABC, Generic[T]):
    def load(self, path: Path, schema: DataSchema | None = None) -> T:
        """Load data in modality-appropriate format"""
```

**Option B: Accept tabular-only for v1.0, plan breaking change for v2.0**

- Document: "v1.0 supports tabular data only"
- Plan modality-specific interfaces: `TabularDataInterface`, `TextDataInterface`, etc.

**My recommendation:** **Option B**

- Reason: Tabular-first focus (per `long_term.mdc`)
- Reason: Generic type causes complexity without immediate benefit
- Plan: v2.0 will have `DataInterface` as Protocol with modality-specific ABC subclasses

**Tradeoff:**

- **Pro:** Honest about current scope
- **Pro:** Simpler implementation
- **Con:** Will require API change later (but that's acceptable for v1 → v2)

---

### Model Wrappers ✅ Mostly Good

**BaseTabularWrapper** (in `models/tabular/base.py`):

- ✅ Handles save/load correctly
- ✅ Feature alignment logic is robust
- ✅ Clear error messages

**No issues found**

---

## Phase 6: Testing Infrastructure Review

### Coverage Status

**Current coverage:** Gate 1 (critical) = 100%, Gate 2 (overall) = 63%

**Critical modules:** All covered ✅

**Gaps in non-critical modules:**

- `metrics/drift/detection.py` - omitted (placeholder)
- `profiles/*` - omitted (not shipped yet)
- `report/renderers/html.py` - omitted (PDF used in CI)

### Test Organization ✅ Good

**Structure:**

```
tests/
  test_core_foundation.py    # Registry, interfaces
  test_config_*.py           # Config validation
  test_audit_*.py            # End-to-end audits
  preprocessing/             # Preprocessing tests
  fixtures/                  # Golden fixtures
```

### ⚠️ Issues Found

#### **I6.1: Missing Contract Tests for Interfaces** ⚠️ Medium Priority

**Problem:** No systematic tests that all implementations satisfy their Protocol

**Recommendation:** Add contract test helper:

```python
# tests/test_interface_contracts.py
from glassalpha.core.interfaces import ModelInterface, ExplainerInterface
from glassalpha.core import ModelRegistry, ExplainerRegistry

def test_all_models_satisfy_protocol():
    """Verify all registered models satisfy ModelInterface"""
    for name in ModelRegistry.names():
        model_cls = ModelRegistry.get(name)
        assert isinstance(model_cls, ModelInterface)
        # Check required methods exist
        assert hasattr(model_cls, 'predict')
        assert hasattr(model_cls, 'predict_proba')
        assert hasattr(model_cls, 'get_model_type')
        assert hasattr(model_cls, 'get_capabilities')

def test_all_explainers_satisfy_protocol():
    """Verify all registered explainers satisfy ExplainerInterface"""
    for name in ExplainerRegistry.names():
        explainer_cls = ExplainerRegistry.get(name)
        assert isinstance(explainer_cls, ExplainerInterface)
        # Check required methods exist
        assert hasattr(explainer_cls, 'explain')
        assert hasattr(explainer_cls, 'supports_model')
        assert hasattr(explainer_cls, 'get_explanation_type')
```

**ETM:** ~30k tokens

---

#### **I6.2: Golden Fixtures Need Expansion** ⚠️ Low Priority

**Current:** German Credit dataset with XGBoost

**Needed for phase 2:** Adult Income, COMPAS (per `phase2_priorities.mdc`)

**Recommendation:** DEFER to phase 2 enhancement work

---

## Phase 7: Dependencies & Security Review

### Dependency Analysis ✅ Excellent

**Core dependencies (required):**

```toml
typer>=0.9.0          # CLI
pydantic>=2.5.0       # Config validation
pyyaml>=6.0           # Config parsing
numpy>=2.1            # Arrays
pandas>=2.2.3         # DataFrames
scipy>=1.15           # Stats
scikit-learn>=1.5     # ML utilities
matplotlib>=3.9       # Plotting
orjson>=3.9.0         # Fast JSON
platformdirs>=4.0.0   # User dirs
jinja2>=3.1,<4        # Templates
tqdm>=4.66            # Progress bars
```

**Assessment:** Minimal, well-justified, no bloat ✅

**Optional dependencies:** Clean separation for SHAP, XGBoost, LightGBM, viz, PDF

### Security Practices ✅ Pass

**Validated:**

- ✅ No hardcoded secrets in code
- ✅ Path validation in `security/paths.py`
- ✅ YAML safe loading in `security/yaml_loader.py`
- ✅ PII sanitization in `security/logs.py`
- ✅ No network calls in core lib (offline-first)

**Recommendation:** NO CHANGES NEEDED

---

## Phase 8: Performance & Bottlenecks Review

### CLI Startup Time

**Measured:** `glassalpha --help` should be <1 second

**Current implementation:** Lazy imports in CLI commands ✅

**Validation needed:**

```bash
time glassalpha --help
# Should be <1s
```

**Recommendation:** Verify performance on Linux/macOS/Windows before launch

---

### Audit Generation Performance

**No obvious bottlenecks found in code review**

**Validation needed:** Benchmark German Credit audit end-to-end

```bash
time glassalpha audit --config quickstart.yaml --out report.pdf
# Target: <60 seconds (per phase2_priorities.mdc)
```

**Recommendation:** Add performance benchmarks to CI

---

## Phase 9: Naming & Terminology Consistency Review

### Module Naming ✅ Consistent

**Checked:**

- `ModelInterface`, `ExplainerInterface`, `MetricInterface` → Consistent `-Interface` suffix
- `ModelRegistry`, `ExplainerRegistry`, `MetricRegistry` → Consistent `-Registry` suffix
- `BaseTabularWrapper`, `BaseAuditProfile` → Consistent `Base-` prefix

**No issues found**

### Terminology Consistency ✅ Good

**Checked:**

- "Audit" (not "report") for full compliance package
- "Explainer" (not "interpreter")
- "Protected attributes" (not "sensitive features" in configs, though code uses both)

**Minor inconsistency found:**

#### **I9.1: "Protected Attributes" vs "Sensitive Features"** ⚠️ Low Priority

**Config uses:** `protected_attributes`
**Code uses:** Both `protected_attributes` and `sensitive_features`

**Recommendation:** Standardize on `protected_attributes` everywhere

- Rationale: Clearer regulatory meaning
- Search/replace: `sensitive_features` → `protected_attributes` in code
- Update function signatures, docstrings

**ETM:** ~40k tokens

**Tradeoff:**

- **Pro:** More precise terminology
- **Con:** Larger change across codebase
- **Decision:** **DEFER** - not critical for launch, can do in v1.1

---

## Phase 10: Dead Code & Cruft Removal

### Findings

#### **I10.1: E2_5_RECOURSE_INTEGRATION.md File** ⚠️ Low Priority

**Location:** `explain/E2_5_RECOURSE_INTEGRATION.md`

**Status:** Integration is complete, file is outdated

**Recommendation:** **DELETE** per `dev_docs.mdc` rules (temp files before merge)

---

#### **I10.2: ExplainerRegistryCompat** ⚠️ Medium Priority

**Already covered in I1.2** - legacy wrapper

**Recommendation:** DELETE

---

#### **I10.3: Deprecated Config Migration Code** ⚠️ Medium Priority

**Already covered in I3.1**

**Files affected:**

- `config/warnings.py`: `warn_deprecated_options()`
- `config/loader.py`: `_migrate_deprecated()`

**Recommendation:** DELETE along with deprecated key support

---

#### **I10.4: get_data_root() Legacy Function** ⚠️ Low Priority

**Location:** `utils/cache_dirs.py` lines 138-159

**Status:** Function has deprecation warning, replacement exists

**Recommendation:** **DELETE** `get_data_root()` function

- No internal usage found
- Deprecation warning already in place
- Replacement (`resolve_data_root()` + `ensure_dir_writable()`) is better

---

#### **I10.5: Unused Imports and TODOs**

**Found TODOs/FIXMEs in:**

- `pipeline/audit.py`
- `cli/main.py`
- `cli/commands.py`
- `utils/determinism.py`

**Recommendation:** Review each TODO:

- If done: Remove comment
- If not done: Convert to GitHub issue or defer to phase 2

**ETM:** ~10k tokens to audit

---

### Coverage Omissions Analysis

**From `pyproject.toml` coverage config:**

```toml
omit = [
  "*/metrics/drift/detection.py",   # placeholder
  "*/metrics/registry.py",          # legacy shim, unused
  "*/models/_io.py",                # unused shim
  "*/profiles/*",                   # not shipped yet
  "*/report/renderers/html.py"      # PDF path used
]
```

**Question:** Are these truly dead code or just untested?

**Analysis:**

- `metrics/drift/detection.py` → Placeholder for future drift detection
- `metrics/registry.py` → Exports `MetricRegistry`, not a shim (mislabeled)
- `models/_io.py` → Check if truly unused
- `profiles/*` → Actively used (audit profiles), should NOT be omitted
- `report/renderers/html.py` → Alternative to PDF, should be tested

**Recommendation:**

#### **I10.6: Fix Coverage Omissions Config** ⚠️ High Priority

**Actions:**

1. Remove `metrics/registry.py` from omit list (it's used)
2. Remove `profiles/*` from omit list (it's used)
3. Add tests for `report/renderers/html.py` or remove it
4. Investigate `models/_io.py` - delete if truly unused

**ETM:** ~50k tokens to fix coverage and tests

---

## Phase 11: Documentation Alignment Review

### Temporary Files Check

**Found (from glob_file_search):**

- `explain/E2_5_RECOURSE_INTEGRATION.md` → **DELETE** (covered in I10.1)
- No `*_STATUS.md`, `*_CHECKLIST.md`, `*_SUMMARY.md` files ✅

**README files (legitimate):**

- `packages/README.md` ✅
- `templates/README.md` ✅
- `dev/README.md` ✅
- `tests/preprocessing/README.md` ✅
- `tests/README_REGRESSION_TESTS.md` ✅

**Assessment:** Clean ✅ (one file to delete)

---

### Documentation Consistency

#### **I11.1: Site Docs vs Reality Check** ⚠️ Medium Priority

**Locations:**

- `site/docs/getting-started/`
- `site/docs/guides/`
- `site/docs/reference/`
- `site/docs/examples/`

**Action needed:** Validate each doc page:

1. Quickstart actually works
2. Installation instructions are current
3. API reference matches code
4. Examples run without errors

**ETM:** ~80k tokens (comprehensive doc validation)

**Priority:** High (broken docs destroy trust)

---

### CHANGELOG.md Status ✅ Good

**Checked:** Present and structured correctly

**Current format:**

```markdown
### Added

- Feature descriptions

### Changed

- Breaking changes

### Fixed

- Bug fixes
```

**Recommendation:** NO CHANGES (already follows keep-a-changelog format)

---

## Phase 12: Consolidate Findings & Prioritize

### Summary of Issues

**Total issues found:** 20

**Breakdown by priority:**

- **High Priority (Must fix before launch):** 4 issues
- **Medium Priority (Should fix before launch):** 7 issues
- **Low Priority (Can defer to v1.1):** 9 issues

---

### High Priority Issues (Must Fix)

| ID    | Issue                                   | Impact                          | ETM |
| ----- | --------------------------------------- | ------------------------------- | --- |
| I1.1  | Duplicate DataInterface definitions     | Confusing API, tech debt        | 20k |
| I5.1  | DataInterface leaks tabular assumptions | Future LLM/vision extensibility | 40k |
| I10.6 | Fix coverage omissions config           | Inaccurate coverage reporting   | 50k |
| I11.1 | Validate documentation against reality  | User trust, broken examples     | 80k |

**Subtotal: 190k tokens**

---

### Medium Priority Issues (Should Fix)

| ID    | Issue                                          | Impact                      | ETM |
| ----- | ---------------------------------------------- | --------------------------- | --- |
| I1.2  | Legacy ExplainerRegistryCompat wrapper         | Tech debt, unused code      | 5k  |
| I3.1  | Deprecated config options still supported      | Confusing schema, tech debt | 40k |
| I3.2  | AuditConfig.**init** unnecessary override      | Code clarity                | 5k  |
| I6.1  | Missing contract tests for interfaces          | Test coverage gaps          | 30k |
| I10.1 | E2_5_RECOURSE_INTEGRATION.md temp file         | Violates dev_docs.mdc rules | 2k  |
| I10.3 | Deprecated config migration code (dup of I3.1) | -                           | -   |
| I10.4 | get_data_root() legacy function                | Dead code                   | 5k  |

**Subtotal: 87k tokens** (excluding duplicate I10.3)

---

### Low Priority Issues (Can Defer)

| ID    | Issue                                          | Impact                          | ETM |
| ----- | ---------------------------------------------- | ------------------------------- | --- |
| I1.3  | Inconsistent model type naming                 | ACCEPTED - no change needed     | 0k  |
| I2.1  | Minimal Python API exports                     | Decision needed on API strategy | 30k |
| I3.3  | Add data.modality field for future             | Future-proofing                 | 10k |
| I4.1  | Error message quality audit                    | UX improvement                  | 20k |
| I6.2  | Golden fixtures expansion                      | Phase 2 work                    | 0k  |
| I9.1  | "Protected attributes" vs "sensitive features" | Terminology consistency         | 40k |
| I10.2 | ExplainerRegistryCompat (dup of I1.2)          | -                               | -   |
| I10.5 | Review TODOs in code                           | Cleanup                         | 10k |

**Subtotal: 110k tokens** (excluding duplicates and I1.3)

---

### Effort Estimation

**Must fix (High):** 190k tokens (~Band M)
**Should fix (Medium):** 87k tokens (~Band S)
**Can defer (Low):** 110k tokens (~Band M)

**Total if we fix everything:** 387k tokens (~Band L)

---

### Recommended Launch Plan

#### **Option 1: Fix All High + Medium (277k tokens, ~Band M)**

**Scope:**

- I1.1: Fix duplicate DataInterface
- I5.1: Document tabular-only limitation (don't fix, just document)
- I10.6: Fix coverage config
- I11.1: Validate documentation
- I1.2: Delete legacy wrapper
- I3.1: Remove deprecated config support
- I3.2: Delete unnecessary **init**
- I6.1: Add contract tests
- I10.1: Delete temp file
- I10.4: Delete legacy function

**Timeline:** ~3-4 AI sessions

**Outcome:** Clean, maintainable codebase ready for v1.0

---

#### **Option 2: Fix Only High Priority (190k tokens, ~Band S)**

**Scope:**

- I1.1: Fix duplicate DataInterface
- I5.1: Document tabular-only limitation
- I10.6: Fix coverage config
- I11.1: Validate documentation

**Timeline:** ~2 AI sessions

**Outcome:** Launch-ready, defer tech debt cleanup to v1.1

---

#### **Option 3: Minimal Launch (110k tokens, ~Band M)**

**Scope:**

- I11.1: Validate documentation (must do)
- I10.6: Fix coverage config (must do)
- Defer everything else

**Timeline:** ~1-2 AI sessions

**Outcome:** Fast launch, accumulate tech debt

---

### My Recommendation

**Option 1: Fix All High + Medium**

**Rationale:**

- This is your "last chance" for breaking changes
- 277k tokens is manageable (~3-4 sessions with AI)
- Removing deprecated config support prevents future confusion
- Contract tests prevent regressions
- Clean codebase makes phase 2 work easier

**What we defer:**

- I2.1: Python API strategy (decide in phase 2 based on usage)
- I9.1: Terminology consistency (nice-to-have, not critical)
- I3.3: data.modality field (add when LLM support is implemented)
- I10.5: TODO audit (can do during phase 2)

---

### Risk Assessment

**If we launch without fixes:**

| Issue | Risk if Not Fixed                                     | Severity                     |
| ----- | ----------------------------------------------------- | ---------------------------- |
| I1.1  | Confusion over DataInterface, harder to extend        | Medium                       |
| I3.1  | Users learn deprecated syntax, harder migration later | Medium                       |
| I5.1  | Breaking change required for LLM support              | Low (v2.0 change acceptable) |
| I10.6 | Inaccurate coverage metrics, false confidence         | Medium                       |
| I11.1 | Broken docs, user frustration, loss of trust          | **High**                     |

**Overall risk of launching without fixes:** **Medium to High**

**Recommended action:** Fix High + Medium priorities before launch

---

## Appendix A: Detailed Issue Descriptions

### I1.1: Duplicate DataInterface Definitions

**Full analysis:**

**File 1: `core/interfaces.py` lines 128-190**

```python
@runtime_checkable
class DataInterface(Protocol):
    """Protocol for data handling across modalities."""
    modality: str
    version: str

    def load(self, path: str) -> Any: ...
    def validate_schema(self, data: Any, schema: dict[str, Any]) -> bool: ...
    def compute_hash(self, data: Any) -> str: ...
```

**File 2: `data/base.py` lines 24-102**

```python
class DataInterface(ABC):
    """Protocol for data loaders across different modalities."""

    @abstractmethod
    def load(self, path: Path, schema: DataSchema | None = None) -> pd.DataFrame: ...

    @abstractmethod
    def validate_schema(self, data: pd.DataFrame, schema: DataSchema) -> None: ...

    @abstractmethod
    def extract_features_target(...) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame | None]: ...

    @abstractmethod
    def hash_data(self, data: pd.DataFrame) -> str: ...

    @abstractmethod
    def split_data(...) -> tuple[pd.DataFrame, pd.DataFrame]: ...
```

**Key differences:**

1. Protocol vs ABC
2. Signature incompatibilities (`path: str` vs `path: Path`)
3. Return type differences (`Any` vs `pd.DataFrame`)
4. ABC version has more methods (`extract_features_target`, `split_data`)

**Which is used?**

```python
# data/tabular.py line 13
class TabularDataHandler(DataInterface):  # Uses ABC from data/base.py
    def load(self, path: Path, schema: DataSchema | None = None) -> pd.DataFrame:
        ...
```

**Conclusion:** The ABC version in `data/base.py` is the real interface

**Action:** Delete Protocol version from `core/interfaces.py`

---

### I3.1: Deprecated Config Options

**Full list of deprecated keys:**

```python
{
    "random_seed": "reproducibility.random_seed",
    "target": "data.target_column",
    "protected_attrs": "data.protected_attributes",
    "model_type": "model.type",
    "model_params": "model.params",
    "explainer_type": "explainers.priority",
    "metrics_config": "metrics",
    "report_template": "report.template",
}
```

**Current behavior:**

1. Old keys are detected by `warn_deprecated_options()`
2. Warning is logged
3. Keys are migrated by `_migrate_deprecated()`
4. Config works with old keys

**Proposed behavior:**

1. Old keys cause validation error
2. Clear error message with migration guide
3. No automatic migration

**Migration tool:**

```bash
glassalpha config migrate old.yaml
# Output:
# ⚠️  Migrating deprecated configuration options:
# • 'random_seed' → 'reproducibility.random_seed'
# • 'target' → 'data.target_column'
#
# New config written to: old_migrated.yaml
```

---

## Appendix B: Commands to Run

### Quick Wins (delete files)

```bash
# Delete temp markdown file
rm packages/src/glassalpha/explain/E2_5_RECOURSE_INTEGRATION.md

# Find all TODOs
grep -r "TODO\|FIXME\|XXX\|HACK\|BUG" packages/src/ --include="*.py"
```

### Coverage Analysis

```bash
# Current coverage
cd packages
pytest --cov=src/glassalpha --cov-report=term-missing

# Identify untested critical paths
pytest --cov=src/glassalpha --cov-report=html
open htmlcov/index.html
```

### Performance Benchmarks

```bash
# CLI startup
time glassalpha --help

# German Credit audit
time glassalpha audit --config packages/configs/german_credit_simple.yaml --out /tmp/test_audit.pdf

# With profiling
python -m cProfile -o audit.prof packages/src/glassalpha/cli/main.py audit --config packages/configs/german_credit_simple.yaml --out /tmp/test_audit.pdf
```

### Documentation Validation

```bash
# Test all example configs
for config in packages/configs/*.yaml; do
    echo "Testing $config..."
    glassalpha validate --config "$config" || echo "FAILED: $config"
done

# Test quickstart
cd examples/notebooks
jupyter nbconvert --to notebook --execute quickstart_from_model.ipynb --output /tmp/test_quickstart.ipynb
```

---

## Appendix C: Effort Breakdown by Task

### I1.1: Fix Duplicate DataInterface (20k tokens)

**Steps:**

1. Review both definitions (5 minutes)
2. Confirm which is used (grep for imports)
3. Delete Protocol version from `core/interfaces.py`
4. Update any imports (unlikely to find any)
5. Run tests: `pytest tests/test_core_foundation.py`
6. Update documentation if needed

### I3.1: Remove Deprecated Config Support (40k tokens)

**Steps:**

1. Create migration tool CLI command (15k)
2. Test migration on example configs (5k)
3. Delete `warn_deprecated_options()` (2k)
4. Delete `_migrate_deprecated()` (2k)
5. Update tests to use new schema (10k)
6. Update documentation (5k)
7. Add breaking change note to CHANGELOG (1k)

### I11.1: Validate Documentation (80k tokens)

**Steps:**

1. Test quickstart guide (10k)
2. Test all getting-started docs (15k)
3. Test all guides (20k)
4. Validate API reference (15k)
5. Test all examples (15k)
6. Fix any broken examples/docs (5k)

**Prioritized checklist:**

- [ ] Quickstart works end-to-end
- [ ] Installation instructions current
- [ ] German Credit example works
- [ ] Configuration guide accurate
- [ ] API reference matches code
- [ ] CLI reference current

---

## Conclusion

**Current status:** Codebase is in **good shape** with targeted issues to address

**Recommended path:** Fix High + Medium priority issues (~277k tokens, 3-4 sessions)

**Launch readiness after fixes:** **READY**

**Biggest risks if we skip fixes:**

1. Broken documentation (I11.1) - **must fix**
2. Coverage config wrong (I10.6) - **must fix**
3. Deprecated config support (I3.1) - **should fix now**
4. Duplicate interfaces (I1.1) - **should fix now**

**Next step:** Review this document, decide on Option 1/2/3, and proceed with fixes
