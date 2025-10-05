# Registry Consolidation Plan

**Date**: October 5, 2025
**Status**: In Progress
**Priority**: URGENT (Pre-Phase 2 blocker)
**Estimated Effort**: 2-3 days

---

## Problem Statement

GlassAlpha has registry system complexity that will hinder Phase 2 development:

1. **Dead code**: `core/plugin_registry.py` (111 lines) exists but is unused
2. **Proxy complexity**: `core/registry.py` has 3 proxy classes to work around circular imports
3. **Confusion**: Unclear which registry implementation to use for new features
4. **Maintenance burden**: Multiple implementations mean duplicate bug fixes

**Why urgent**: Phase 2 will add 5-7 new features interacting with registries. Clean foundation now = 3-5x less effort than mid-Phase 2.

---

## Current State Analysis

### Registry Files

| File                      | Lines | Usage                                     | Status                           |
| ------------------------- | ----- | ----------------------------------------- | -------------------------------- |
| `core/registry.py`        | 447   | 10 direct imports                         | **Keep** (main implementation)   |
| `core/decor_registry.py`  | 138   | 3 imports (explainers, metrics, profiles) | **Keep** (needed for decorators) |
| `core/plugin_registry.py` | 111   | 0 imports                                 | **DELETE** (dead code)           |

### Registry Instances

**Currently**:

```python
# In core/registry.py
ModelRegistry = PluginRegistry("glassalpha.models")
ProfileRegistry = PluginRegistry("glassalpha.profiles")  # But also defined in profiles/registry.py!
DataRegistry = PluginRegistry("glassalpha.data_handlers")

# Proxies for circular import workaround
ExplainerRegistry = ExplainerRegistryProxy()  # → explain/registry.py
MetricRegistry = MetricRegistryProxy()        # → metrics/registry.py
ProfileRegistry = ProfileRegistryProxy()      # → profiles/registry.py (duplicate!)
```

**Also defined elsewhere**:

```python
# In explain/registry.py
ExplainerRegistry = ExplainerRegistryClass(...)  # DecoratorFriendlyRegistry subclass

# In metrics/registry.py
MetricRegistry = DecoratorFriendlyRegistry("glassalpha.metrics")

# In profiles/registry.py
ProfileRegistry = DecoratorFriendlyRegistry("glassalpha.profiles")
```

### Import Patterns

```bash
# Direct imports from core/registry.py: 10 files
- cli/commands.py (4 registries)
- models/__init__.py (ModelRegistry)
- cli/preflight.py (ModelRegistry)
- models/tabular/xgboost.py (ModelRegistry)
- pipeline/audit.py (MetricRegistry, ModelRegistry)
- pipeline/train.py (ModelRegistry)
- models/tabular/lightgbm.py (ModelRegistry)
- models/tabular/sklearn.py (ModelRegistry)

# Imports from decor_registry.py: 3 files
- explain/registry.py
- profiles/registry.py
- metrics/registry.py

# Imports from plugin_registry.py: 0 files (DEAD CODE)
```

---

## Consolidation Strategy

### Phase 1: Remove Dead Code (1 hour)

**Actions**:

1. Delete `core/plugin_registry.py` (unused)
2. Remove any references in tests

**Testing**:

```bash
# Verify no imports
rg "from.*plugin_registry import" packages/
rg "import.*plugin_registry" packages/

# Run full test suite
pytest packages/tests/ -v
```

**Success Criteria**: All tests pass, no references to `plugin_registry.py`

---

### Phase 2: Eliminate Proxy Classes (4-6 hours)

**Problem**: Proxy classes in `core/registry.py` work around circular imports but add complexity.

**Current Flow** (convoluted):

```
core/registry.py defines ExplainerRegistryProxy
  ↓
User imports: from glassalpha.core import ExplainerRegistry
  ↓
First access triggers __getattr__
  ↓
Imports explain/registry.py
  ↓
Replaces proxy with real ExplainerRegistryClass instance
```

**Target Flow** (clean):

```
User imports: from glassalpha.explain.registry import ExplainerRegistry
  (or)
User imports: from glassalpha.core import ExplainerRegistry (re-export only)
```

**Actions**:

1. **Update `core/__init__.py`** to re-export registries from their canonical locations:

   ```python
   # OLD (proxies)
   from .registry import ExplainerRegistry, MetricRegistry, ProfileRegistry

   # NEW (direct imports from canonical locations)
   from ..explain.registry import ExplainerRegistry
   from ..metrics.registry import MetricRegistry
   from ..profiles.registry import ProfileRegistry
   from .registry import ModelRegistry, DataRegistry
   ```

2. **Remove proxy classes from `core/registry.py`**:

   - Delete `ExplainerRegistryProxy` class (lines ~325-337)
   - Delete `MetricRegistryProxy` class (lines ~340-352)
   - Delete `ProfileRegistryProxy` class (lines ~355-367)
   - Delete proxy instantiation (lines ~370-373)
   - Delete lazy getter functions `_get_explainer_registry()`, etc.

3. **Simplify `core/registry.py`**:

   - Keep only `ModelRegistry` and `DataRegistry` (defined directly in this file)
   - Remove duplicate `ProfileRegistry` definition (it's in `profiles/registry.py`)
   - Update `list_components()` to import registries properly

4. **Update imports in 10 affected files**:

   **Pattern 1**: Files importing `ExplainerRegistry` from `core/registry`:

   ```python
   # OLD
   from ..core.registry import ExplainerRegistry

   # NEW
   from ..explain.registry import ExplainerRegistry
   ```

   **Pattern 2**: Files importing `MetricRegistry` from `core/registry`:

   ```python
   # OLD
   from ..core.registry import MetricRegistry

   # NEW
   from ..metrics.registry import MetricRegistry
   ```

   **Pattern 3**: Files importing multiple registries:

   ```python
   # OLD
   from ..core.registry import ExplainerRegistry, MetricRegistry, ModelRegistry, ProfileRegistry

   # NEW
   from ..core.registry import ModelRegistry
   from ..explain.registry import ExplainerRegistry
   from ..metrics.registry import MetricRegistry
   from ..profiles.registry import ProfileRegistry
   ```

5. **Files to update** (based on grep results):
   - `cli/commands.py` (imports 4 registries - needs updating)
   - `pipeline/audit.py` (imports MetricRegistry, ModelRegistry)
   - Other files already use ModelRegistry only (no changes needed)

**Testing**:

```bash
# Verify import paths work
python -c "from glassalpha.core import ExplainerRegistry; print(ExplainerRegistry)"
python -c "from glassalpha.core import MetricRegistry; print(MetricRegistry)"
python -c "from glassalpha.core import ProfileRegistry; print(ProfileRegistry)"
python -c "from glassalpha.core import ModelRegistry; print(ModelRegistry)"

# Run full test suite
pytest packages/tests/ -v

# Run specific registry tests
pytest packages/tests/ -k registry -v
pytest packages/tests/ -k explainer -v
pytest packages/tests/ -k metric -v
```

**Success Criteria**:

- No proxy classes remain
- All imports work correctly
- All tests pass
- Code is more readable

---

### Phase 3: Comprehensive Testing (4-6 hours)

#### Unit Tests

**Test Coverage Required**:

1. **Registry Basic Operations** (`tests/test_registry_consolidation.py`):

   ```python
   def test_all_registries_importable():
       """Verify all registries can be imported."""
       from glassalpha.core import (
           ModelRegistry,
           ExplainerRegistry,
           MetricRegistry,
           ProfileRegistry,
           DataRegistry,
       )
       assert ModelRegistry is not None
       assert ExplainerRegistry is not None
       assert MetricRegistry is not None
       assert ProfileRegistry is not None
       assert DataRegistry is not None

   def test_registry_operations():
       """Test basic registry operations."""
       from glassalpha.core import ModelRegistry

       # Test registration
       ModelRegistry.register("test_model", lambda: "test")
       assert ModelRegistry.has("test_model")

       # Test retrieval
       model = ModelRegistry.get("test_model")
       assert callable(model)

       # Test listing
       assert "test_model" in ModelRegistry.names()

   def test_no_circular_imports():
       """Verify no circular import issues."""
       import sys
       # Clear module cache
       modules_to_clear = [k for k in sys.modules if k.startswith('glassalpha')]
       for mod in modules_to_clear:
           del sys.modules[mod]

       # Import in various orders - should all work
       from glassalpha.core import ModelRegistry
       from glassalpha.explain.registry import ExplainerRegistry
       from glassalpha.metrics.registry import MetricRegistry
       from glassalpha.profiles.registry import ProfileRegistry

   def test_decorator_registration():
       """Test decorator-based registration still works."""
       from glassalpha.metrics.registry import MetricRegistry

       @MetricRegistry.register("test_metric")
       class TestMetric:
           pass

       assert MetricRegistry.has("test_metric")
       assert MetricRegistry.get("test_metric") == TestMetric
   ```

2. **Model Registry Tests** (`tests/test_model_registry.py`):

   ```python
   def test_model_registration():
       """Test model can be registered and retrieved."""
       from glassalpha.core.registry import ModelRegistry

       class TestModel:
           def predict(self, X):
               return X

       ModelRegistry.register("test_model", TestModel)
       assert ModelRegistry.has("test_model")
       assert ModelRegistry.get("test_model") == TestModel

   def test_model_discovery():
       """Test entry point discovery works."""
       from glassalpha.core.registry import ModelRegistry

       # Should discover xgboost, lightgbm, logistic_regression
       ModelRegistry.discover()
       names = ModelRegistry.names()

       # At minimum, passthrough should be available
       assert "passthrough" in names or len(names) > 0
   ```

3. **Explainer Registry Tests** (`tests/test_explainer_registry.py`):

   ```python
   def test_explainer_compatibility():
       """Test explainer compatibility checking."""
       from glassalpha.explain.registry import ExplainerRegistry

       # Test noop explainer (always compatible)
       assert ExplainerRegistry.is_compatible("noop", "xgboost")
       assert ExplainerRegistry.is_compatible("noop", "unknown_model")

   def test_explainer_selection():
       """Test explainer selection logic."""
       from glassalpha.explain.registry import select_explainer

       # Should select based on model type
       result = select_explainer("xgboost", requested_priority=["treeshap", "kernelshap"])
       assert result in ["treeshap", "kernelshap", "noop"]
   ```

4. **Metric Registry Tests** (`tests/test_metric_registry.py`):

   ```python
   def test_metric_registration():
       """Test metric registration works."""
       from glassalpha.metrics.registry import MetricRegistry

       assert MetricRegistry.has("accuracy")
       accuracy_cls = MetricRegistry.get("accuracy")
       assert accuracy_cls is not None

   def test_metric_computation():
       """Test metrics can be computed."""
       from glassalpha.metrics.registry import compute_all_metrics
       import numpy as np

       y_true = np.array([0, 1, 0, 1])
       y_pred = np.array([0, 1, 0, 1])

       results = compute_all_metrics(["accuracy"], y_true, y_pred)
       assert "accuracy" in results
       assert results["accuracy"]["accuracy"] == 1.0
   ```

#### Integration Tests

**Test Coverage Required**:

1. **Full Audit Pipeline** (`tests/integration/test_audit_with_registries.py`):

   ```python
   def test_audit_uses_correct_registries(tmp_path):
       """Test full audit pipeline with registry system."""
       from glassalpha.config.loader import load_config_from_file
       from glassalpha.pipeline.audit import AuditPipeline

       # Use german_credit_simple config
       config = load_config_from_file("configs/german_credit_simple.yaml")

       # Run audit
       pipeline = AuditPipeline(config)
       results = pipeline.run()

       # Verify components were loaded from registries
       assert results.selected_components["model"]["name"] in ["xgboost", "passthrough"]
       assert "explainer" in results.selected_components
       assert len(results.metrics) > 0
   ```

2. **CLI Commands** (`tests/integration/test_cli_with_registries.py`):

   ```python
   def test_cli_audit_command(tmp_path):
       """Test CLI audit command uses registries correctly."""
       import subprocess

       output = tmp_path / "test_audit.pdf"
       result = subprocess.run(
           ["glassalpha", "audit",
            "--config", "configs/german_credit_simple.yaml",
            "--out", str(output)],
           capture_output=True
       )

       assert result.returncode == 0
       assert output.exists()

   def test_cli_list_command():
       """Test CLI list command shows all components."""
       import subprocess

       result = subprocess.run(
           ["glassalpha", "list", "--type", "models"],
           capture_output=True,
           text=True
       )

       assert result.returncode == 0
       assert "passthrough" in result.stdout or len(result.stdout) > 0
   ```

3. **Cross-Module Integration** (`tests/integration/test_cross_module.py`):

   ```python
   def test_pipeline_uses_all_registries():
       """Test pipeline correctly uses all registry types."""
       from glassalpha.pipeline.audit import AuditPipeline
       from glassalpha.core.registry import ModelRegistry
       from glassalpha.explain.registry import ExplainerRegistry
       from glassalpha.metrics.registry import MetricRegistry

       # Verify registries are populated
       assert len(ModelRegistry.names()) > 0
       assert len(ExplainerRegistry.names()) > 0
       assert len(MetricRegistry.names()) > 0
   ```

#### Regression Tests

**Run existing test suite**:

```bash
# Full test suite
pytest packages/tests/ -v

# Critical paths
pytest packages/tests/test_audit_pipeline.py -v
pytest packages/tests/test_explainer_selection.py -v
pytest packages/tests/test_metrics_*.py -v
pytest packages/tests/test_cli_*.py -v

# Contract tests (ensure no API breaks)
pytest packages/tests/ -k contract -v
```

#### Manual Testing Checklist

- [ ] Import all registries in Python REPL
- [ ] Run `glassalpha audit` with german_credit_simple.yaml
- [ ] Run `glassalpha list --type models`
- [ ] Run `glassalpha list --type explainers`
- [ ] Run `glassalpha list --type metrics`
- [ ] Verify PDF generation works
- [ ] Check manifest includes correct component selections
- [ ] Test with missing optional dependencies (uninstall shap, verify fallback)

---

### Phase 4: Documentation Update (1 hour)

**Update locations**:

1. **Architecture docs** (`packages/README.md`, lines 398-470):

   - Update registry architecture description
   - Remove references to proxy pattern
   - Show clean import paths

2. **Developer docs** (`site/docs/reference/trust-deployment.md`):

   - Update component registration examples
   - Show correct import patterns

3. **Code comments**:
   - Update comments in `core/registry.py`
   - Update comments in `core/__init__.py`
   - Remove outdated "circular import workaround" comments

---

## Rollback Plan

If consolidation causes issues:

1. **Immediate rollback** (< 5 minutes):

   ```bash
   git checkout main -- packages/src/glassalpha/core/
   git checkout main -- packages/src/glassalpha/cli/commands.py
   git checkout main -- packages/src/glassalpha/pipeline/audit.py
   ```

2. **Verification**:

   ```bash
   pytest packages/tests/ -v
   ```

3. **If partial rollback needed**:
   - Keep proxy removal if imports work
   - Only rollback specific file changes
   - Document which parts succeeded

---

## Success Criteria

### Functional Requirements

- [ ] All tests pass (100% of existing test suite)
- [ ] All CLI commands work correctly
- [ ] Full audit pipeline produces identical output
- [ ] Component registration works (models, explainers, metrics, profiles)
- [ ] No circular import errors

### Code Quality Requirements

- [ ] `core/plugin_registry.py` deleted
- [ ] No proxy classes in `core/registry.py`
- [ ] Clean import paths (direct, not through proxies)
- [ ] Reduced lines of code in `core/registry.py` (~200 lines vs. 447)
- [ ] Clear separation: each registry in its canonical location

### Documentation Requirements

- [ ] Architecture docs updated
- [ ] Import examples corrected
- [ ] No references to old proxy pattern

### Performance Requirements

- [ ] No performance regression (< 5% difference in audit time)
- [ ] No memory regression (< 5% difference in memory usage)

---

## Timeline

| Phase                      | Duration     | Deliverable                              |
| -------------------------- | ------------ | ---------------------------------------- |
| Phase 1: Remove dead code  | 1 hour       | `plugin_registry.py` deleted, tests pass |
| Phase 2: Eliminate proxies | 4-6 hours    | Clean imports, no proxies, tests pass    |
| Phase 3: Testing           | 4-6 hours    | New tests written, full suite passes     |
| Phase 4: Documentation     | 1 hour       | Docs updated, examples corrected         |
| **Total**                  | **2-3 days** | Clean registry system ready for Phase 2  |

---

## Risk Assessment

| Risk                          | Probability | Impact | Mitigation                                         |
| ----------------------------- | ----------- | ------ | -------------------------------------------------- |
| Circular imports resurface    | Medium      | High   | Add imports at function level if needed            |
| Tests fail after changes      | Medium      | Medium | Fix incrementally, rollback if blocked             |
| Performance regression        | Low         | Low    | Profile before/after, optimize if needed           |
| API breaks for external users | Low         | High   | Check public API carefully, version bump if needed |

---

## Next Steps

1. **Get approval** for this plan
2. **Create feature branch**: `refactor/registry-consolidation`
3. **Execute Phase 1** (1 hour): Remove dead code
4. **Execute Phase 2** (4-6 hours): Eliminate proxies
5. **Execute Phase 3** (4-6 hours): Comprehensive testing
6. **Execute Phase 4** (1 hour): Documentation
7. **PR review** and merge to `main`
8. **Start Phase 2 features** with clean foundation

---

## Notes for Future

**Why this matters for Phase 2**:

- Calibration metrics: Need to register with `MetricRegistry` - which one?
- GitHub Action: Will query registries for available components
- Gates/policies: Will interact with registry system
- Evidence pack: Will serialize registry selections

**Clean registries now = easier Phase 2 development**

---

**Status**: Ready to execute
**Owner**: AI Agent
**Reviewer**: User (Gabe)
