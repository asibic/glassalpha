# API Implementation Roadmap

**Purpose**: Detailed file-by-file implementation plan for v0.2 API redesign.
**Use**: Reference during implementation to track which files need changes.

---

## File Creation Plan

### New Files to Create

```
src/glassalpha/
├── api/
│   ├── __init__.py              # Public API surface
│   ├── audit.py                 # Entry points (from_model, from_predictions, from_config)
│   ├── result.py                # AuditResult dataclass
│   ├── metrics.py               # ReadonlyMetrics and subclasses
│   └── plots.py                 # Plot functions (lazy import)
├── core/
│   ├── canonicalization.py      # canonicalize() function
│   └── hashing.py               # compute_result_id(), hash_data_for_manifest()
├── exceptions.py                # GlassAlphaError base class
└── error_codes.py               # Error code constants

tests/api/
├── __init__.py
├── test_contracts.py            # Core API contracts
├── test_determinism.py          # Reproducibility tests
├── test_immutability.py         # Frozen checks
├── test_errors.py               # Error quality tests
├── test_hashing.py              # Canonicalization & data hashing
├── test_entry_points.py         # Entry point tests
├── test_metrics_access.py       # Dict + attribute access
└── test_performance.py          # Benchmarks

site/docs/
├── reference/api/
│   ├── audit.md                 # Entry point docs
│   ├── result.md                # AuditResult docs
│   └── errors.md                # Error code reference
├── guides/
│   ├── probability-requirements.md
│   ├── missing-data.md
│   └── data-requirements.md     # Update
└── reference/
    ├── stability-index.md       # New
    └── tolerance-policy.md      # New

examples/notebooks/
├── quickstart_from_model.ipynb  # Update
└── german_credit_walkthrough.ipynb  # Update
```

---

## File Modification Plan

### Files to Update

```
src/glassalpha/
├── __init__.py                  # Replace with PEP 562 lazy loading
├── metrics/
│   └── registry.py              # Add MetricSpec dataclass
└── utils/
    └── hashing.py               # May need to extract/refactor for new core/hashing.py

tests/
└── conftest.py                  # May need fixtures for testing

site/docs/
├── guides/reproducibility.md    # Add timezone section
└── reference/api/
    └── data-requirements.md     # Add MultiIndex, timezone policies
```

---

## Implementation Order with Dependencies

### Phase 1: Namespace (No Dependencies)

**Files to Create/Modify**:

1. `src/glassalpha/__init__.py` - PEP 562 implementation

**Tests**:

1. `tests/api/test_contracts.py` - import speed, lazy loading

**Validation**:

- `python -c "import glassalpha"` < 200ms
- Tab completion works

---

### Phase 2: Result & Metrics (Depends on Phase 1)

**Files to Create**:

1. `src/glassalpha/api/__init__.py`
2. `src/glassalpha/api/result.py` - AuditResult class
3. `src/glassalpha/api/metrics.py` - ReadonlyMetrics + subclasses

**Tests**:

1. `tests/api/test_immutability.py` - deep freeze checks
2. `tests/api/test_metrics_access.py` - dict + attr access

**Validation**:

- Frozen dataclass prevents mutation
- Arrays are read-only
- Both access patterns work

---

### Phase 3: Entry Points (Depends on Phase 2)

**Files to Create**:

1. `src/glassalpha/api/audit.py` - Entry point functions

**Tests**:

1. `tests/api/test_entry_points.py` - all entry points

**Validation**:

- `from_model()` works with all input formats
- Input validation catches errors
- Inputs not mutated

---

### Phase 4: Hashing (Depends on Phase 3) **CRITICAL**

**Files to Create**:

1. `src/glassalpha/core/canonicalization.py`
2. `src/glassalpha/core/hashing.py`

**Files to Update**:

1. `src/glassalpha/api/audit.py` - integrate hashing
2. `src/glassalpha/api/result.py` - integrate result.id computation

**Tests**:

1. `tests/api/test_hashing.py` - all edge cases
2. `tests/api/test_determinism.py` - 10-run tests

**Validation**:

- result.id deterministic
- All edge cases handled (NaN, Inf, -0.0, etc.)
- Data hash includes "sha256:" prefix

---

### Phase 5: Error Handling (Depends on Phase 3)

**Files to Create**:

1. `src/glassalpha/exceptions.py`
2. `src/glassalpha/error_codes.py`

**Files to Update**:

1. `src/glassalpha/api/audit.py` - raise GlassAlphaError
2. `src/glassalpha/api/metrics.py` - raise GlassAlphaError
3. All validation functions

**Tests**:

1. `tests/api/test_errors.py` - all error codes

**Validation**:

- All errors have code/fix/docs
- Error messages helpful

---

### Phase 6: Testing (Depends on Phases 4-5)

**Files to Create**:

1. All remaining test files from test plan

**Validation**:

- 95%+ coverage
- All contract tests pass
- Determinism verified

---

### Phase 7: Documentation (Depends on Phase 6)

**Files to Create**:

1. `site/docs/reference/api/audit.md`
2. `site/docs/reference/api/result.md`
3. `site/docs/reference/api/errors.md`
4. `site/docs/guides/probability-requirements.md`
5. `site/docs/guides/missing-data.md`

**Files to Update**:

1. `site/docs/guides/reproducibility.md`
2. `site/docs/reference/api/data-requirements.md`
3. `examples/notebooks/quickstart_from_model.ipynb`
4. `examples/notebooks/german_credit_walkthrough.ipynb`

**Validation**:

- All examples run
- Notebooks work end-to-end

---

### Phase 8: Quality Gates (Depends on Phase 7)

**Files to Update**:

1. `src/glassalpha/metrics/registry.py` - Add MetricSpec

**Validation**:

- Registry complete
- Metadata available

---

### Phase 9: Stability Index (Depends on Phase 8)

**Files to Create**:

1. `site/docs/reference/stability-index.md`
2. `site/docs/reference/tolerance-policy.md`

**Validation**:

- Stability guarantees published
- Tolerance policy documented

---

## Code Structure Details

### src/glassalpha/**init**.py

**Purpose**: Fast imports with lazy loading

**Structure**:

```python
"""GlassAlpha: AI Compliance Toolkit"""

__version__ = "0.2.0"
__all__ = ["__version__", "audit", "datasets", "utils"]

_LAZY_MODULES = {
    "audit": "glassalpha.api.audit",
    "datasets": "glassalpha.datasets",
    "utils": "glassalpha.utils",
}

def __getattr__(name: str):
    """Lazy-load modules on first access (PEP 562)"""
    if name in _LAZY_MODULES:
        import importlib
        module = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'glassalpha' has no attribute '{name}'")

def __dir__():
    """Enable tab-completion for lazy modules"""
    return __all__
```

**Key Points**:

- No heavy imports in module scope
- PEP 562 for lazy loading
- Private modules (\_export) not exposed

---

### src/glassalpha/api/result.py

**Purpose**: Immutable result object

**Key Classes**:

- `AuditResult` - frozen dataclass
- Helper: `_freeze_array()` for numpy arrays

**Key Methods**:

- `__eq__()` - via result.id
- `__hash__()` - via int(result.id[:16], 16)
- `equals()` - tolerance-based
- `to_json()`, `to_pdf()`, `to_config()`, `save()`
- `_repr_html_()` - Jupyter display

**Immutability Strategy**:

- Dataclass frozen=True
- Manifest wrapped in MappingProxyType
- Arrays made C-contiguous + read-only
- Nested dicts recursively frozen

---

### src/glassalpha/api/metrics.py

**Purpose**: Metric section wrappers

**Key Classes**:

- `ReadonlyMetrics` - base with Mapping protocol
- `PerformanceMetrics` - with plot methods + GAE1009 override
- `FairnessMetrics` - with group plotting
- `CalibrationMetrics` - with calibration curve
- `StabilityMetrics`, `ExplanationSummary`, `RecourseSummary`

**Key Methods**:

- `__getitem__()` - dict-style (raises KeyError)
- `__getattr__()` - attribute-style (raises GlassAlphaError)
- `_freeze_nested()` - recursive freezing

---

### src/glassalpha/api/audit.py

**Purpose**: Entry points for audit generation

**Key Functions**:

- `from_model()` - primary entry point
- `from_predictions()` - model-less audits
- `from_config()` - reproduction
- `_normalize_protected_attributes()` - helper
- `_get_probabilities()` - helper

**Validation Logic**:

- Problem type (binary only)
- Missing policy ("include_unknown" only)
- Length matching
- MultiIndex rejection

---

### src/glassalpha/core/canonicalization.py

**Purpose**: Deterministic object canonicalization

**Key Function**:

- `canonicalize()` - recursive, handles all edge cases

**Edge Cases**:

- NaN → None (null in JSON)
- Infinity → sentinel `{"__float__": "±Infinity"}`
- -0.0 → 0.0
- Arrays → `{"__ndarray__": {dtype, shape, data}}`
- Bytes → base64
- Datetime → UTC ISO 8601

---

### src/glassalpha/core/hashing.py

**Purpose**: Result ID and data hashing

**Key Functions**:

- `compute_result_id()` - canonical JSON → SHA-256
- `hash_data_for_manifest()` - streaming SHA-256
- `_atomic_write()` - safe file writes

**Data Hash Features**:

- Streaming (memory efficient)
- Dtype-aware (int32 vs int64)
- RangeIndex fast path
- Returns "sha256:{hex}" format

---

### src/glassalpha/exceptions.py

**Purpose**: Error handling

**Key Class**:

- `GlassAlphaError` - frozen dataclass with code/message/fix/docs

**Usage**:

```python
raise GlassAlphaError(
    code="GAE1003_LENGTH_MISMATCH",
    message="X has 100 samples, y has 99",
    fix="Ensure X and y have same length",
    docs="https://glassalpha.com/v0.2/reference/errors#gae1003"
)
```

---

## Integration Points

### Existing Code to Integrate With

**Current Pipeline**:

- `src/glassalpha/pipeline/` - May need to adapt for new entry points
- `src/glassalpha/metrics/` - Metrics computation unchanged
- `src/glassalpha/explain/` - SHAP computation unchanged
- `src/glassalpha/report/` - PDF generation may need AuditResult adapter

**Strategy**: Create adapters in api/ layer, don't modify existing pipeline.

---

## Migration Strategy

### Breaking Changes from Current API

1. **Import paths change**:

   - Old: `from glassalpha.pipeline import run_audit`
   - New: `import glassalpha as ga; ga.audit.from_model()`

2. **Result object structure changes**:

   - Old: Dict with nested structure
   - New: AuditResult dataclass with typed attributes

3. **Error types change**:
   - Old: ValueError, TypeError
   - New: GlassAlphaError with codes

**No migration needed** - v0.2 is clean break (no backwards compatibility).

---

## Rollout Checklist

### Pre-Implementation

- [x] Master plan created
- [x] Phase checklist created
- [x] Design decisions documented
- [x] Test plan created
- [x] Implementation roadmap created

### Implementation (This Document)

- [ ] Phase 1 complete (Namespace)
- [ ] Phase 2 complete (Result)
- [ ] Phase 3 complete (Entry points)
- [ ] Phase 4 complete (Hashing) **CRITICAL**
- [ ] Phase 5 complete (Errors)
- [ ] Phase 6 complete (Testing)
- [ ] Phase 7 complete (Documentation)
- [ ] Phase 8 complete (Quality gates)
- [ ] Phase 9 complete (Stability index)

### Post-Implementation

- [ ] API_DESIGN_ANALYSIS.md generated
- [ ] CHANGELOG.md updated
- [ ] All temporary docs deleted
- [ ] Notebooks verified
- [ ] CI green on Linux + macOS

---

## Risk Mitigation

### Risk: Import speed regression

**Monitor**: `test_import_speed()` in CI
**Threshold**: <200ms (fail if >250ms)
**Mitigation**: Profile with `python -X importtime`

### Risk: Cross-platform determinism

**Monitor**: `test_determinism_linux_macos()` in CI
**Threshold**: 100% ID match or equals() passes
**Mitigation**: Tolerance policy + explicit platform tests

### Risk: Coverage regression

**Monitor**: pytest --cov in CI
**Threshold**: 95%+ for api/
**Mitigation**: Fail CI if below threshold

### Risk: Breaking changes to existing code

**Monitor**: Run existing tests against new API
**Threshold**: 0 existing tests should pass (clean break)
**Mitigation**: Document breaking changes in CHANGELOG

---

## Success Criteria

✅ **Phase complete** when:

1. All files created/updated per plan
2. All tests pass
3. Coverage meets threshold
4. Documentation complete
5. Next phase not blocked

✅ **Full implementation complete** when:

1. All 9 phases complete
2. Exit checklist 100% done
3. CI green on all platforms
4. Notebooks work end-to-end
5. API_DESIGN_ANALYSIS.md finalized

---

## Next Steps

1. **Review with user**: Get approval on plan
2. **Start Phase 1**: Namespace implementation
3. **Iterate**: Complete phases in order
4. **Track progress**: Update checklists as work completes
5. **Final review**: Validate exit criteria before v0.2 tag

**Estimated Timeline**: 5-7 days of focused agent work (~400k tokens)

**Ready to begin!**
