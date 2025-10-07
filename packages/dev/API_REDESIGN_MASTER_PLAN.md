# API Redesign Master Plan

**Status**: Pre-Launch Blocker for v0.2
**ETM**: ~400k tokens | Band: L | Risk: Medium
**Started**: 2025-10-07
**Target Completion**: Before PyPI publication

---

## Executive Summary

**Goal**: Design and implement a best-in-class convenience API that exceeds pandas/scikit-learn/requests in ergonomics while maintaining byte-identical reproducibility for compliance audits.

**Why Now**: Current API surface misaligned with documentation (see `API_VALIDATION_REPORT.md`). No backwards compatibility required - perfect time for breaking changes.

**Success Criteria**:

- Import speed <200ms (lazy loading)
- Deterministic `result.id` across platforms
- All errors use GlassAlphaError with code/fix/docs
- Deep immutability (arrays read-only, dicts frozen)
- 95%+ test coverage, 100% type coverage
- Complete documentation with examples

---

## Implementation Phases

### Phase 1: Namespace & Import Convenience (~30k tokens)

**Goal**: Fast imports with lazy loading
**Files**: `src/glassalpha/__init__.py`
**Deliverables**:

- PEP 562 lazy module loading
- Import speed <200ms
- Tab completion support

### Phase 2: Result Object Wrapper (~80k tokens)

**Goal**: Immutable result with rich API
**Files**: `src/glassalpha/api/result.py`, `src/glassalpha/api/metrics.py`
**Deliverables**:

- `AuditResult` dataclass with frozen=True
- Metric section wrappers (ReadonlyMetrics base class)
- Deep immutability (arrays, dicts, nested)
- Dict + attribute access patterns

### Phase 3: Entry Point Signatures (~60k tokens)

**Goal**: Ergonomic entry points
**Files**: `src/glassalpha/api/audit.py`
**Deliverables**:

- `from_model()` with flexible protected_attributes
- `from_predictions()` for model-less audits
- `from_config()` for reproduction
- Input validation and error handling

### Phase 4: Determinism & Hashing (~70k tokens) **CRITICAL PATH**

**Goal**: Byte-identical reproducibility
**Files**: `src/glassalpha/core/hashing.py`, `src/glassalpha/core/canonicalization.py`
**Deliverables**:

- `compute_result_id()` with canonical JSON
- `hash_data_for_manifest()` with streaming SHA-256
- Handle NaN, Inf, -0.0, timezones, dtypes
- Self-documenting hash format (`"sha256:..."`)

### Phase 5: Error Handling (~40k tokens)

**Goal**: Helpful, actionable errors
**Files**: `src/glassalpha/exceptions.py`, `src/glassalpha/error_codes.py`
**Deliverables**:

- `GlassAlphaError` base class
- 15+ error codes (GAE1001-GAE4001)
- Every error has: code, message, fix, docs URL

### Phase 6: Testing (~80k tokens)

**Goal**: Comprehensive contract tests
**Files**: `tests/api/test_contracts.py`, `tests/api/test_determinism.py`
**Deliverables**:

- Import speed tests
- Determinism tests (10+ runs)
- Immutability tests (frozen checks)
- Error quality tests
- Hashing edge cases (NaN, Inf, dtypes, timezones)

### Phase 7: Documentation (~50k tokens)

**Goal**: Complete user-facing docs
**Files**: `site/docs/reference/api/*.md`, `site/docs/guides/*.md`
**Deliverables**:

- API reference with examples
- Probability requirements guide
- Missing data guide
- Reproducibility guide (updated)
- Timezone handling docs

### Phase 8: Quality Gates (~20k tokens)

**Goal**: Metric registry for enterprise
**Files**: `src/glassalpha/metrics/registry.py`
**Deliverables**:

- MetricSpec with metadata
- higher_is_better flags
- Tolerance specifications

### Phase 9: Stability Index (~10k tokens)

**Goal**: API stability guarantees
**Files**: `site/docs/reference/stability-index.md`, `site/docs/reference/tolerance-policy.md`
**Deliverables**:

- Maturity levels (Stable/Beta/Experimental)
- Breaking change policy
- Tolerance policy for cross-platform tests

---

## Critical Blockers (Must Fix Before v0.2)

### Blocker #1: GAE1009 Override in PerformanceMetrics

**Issue**: Accessing AUC without probabilities should raise GAE1009, not generic GAE1002
**Fix**: Override `__getattr__` in PerformanceMetrics class
**Test**: `test_performance_metrics_gae1009_override()`

### Blocker #2: NaN → "Unknown" (not "nan")

**Issue**: Missing values should map to `"Unknown"` string, not `"nan"`
**Fix**: In `_normalize_protected_attributes()`, explicitly check for NaN before string conversion
**Test**: `test_protected_attrs_nan_to_unknown()`

### Blocker #3: Array Canonicalization Preserves dtype/shape

**Issue**: Need to distinguish int32 vs int64 for determinism
**Fix**: Wrap arrays in `{"__ndarray__": {dtype, shape, data}}`
**Test**: `test_canonical_hash_ndarray()`, `test_data_hash_dtype_sensitivity()`

### Blocker #4: Bytes Canonicalize to base64

**Issue**: Bytes not JSON-serializable
**Fix**: Use `base64.b64encode(obj).decode("ascii")`
**Test**: `test_canonical_hash_bytes()`

### Blocker #5: Data Hash Returns "sha256:..." Prefix

**Issue**: Self-documenting format for audit trails
**Fix**: Return `f"sha256:{hasher.hexdigest()}"`
**Test**: `test_data_hash_prefix()`

---

## Should-Fix (Quality Improvements)

- `equals()` handles None gracefully
- Atomic write supports binary mode
- `to_json()` uses strict canonicalization (same as result.id)
- `model_type` in manifest includes module + qualname
- Plot methods don't mutate global rcParams

---

## Nits (Minor Polish)

- RangeIndex test uses relaxed threshold (3x instead of 10x for CI variance)
- Error codes consistent in docs
- Timezone warning in docs emphasized with ⚠️

---

## Dependencies

**Phase Order**:

1. Phase 1 (Namespace) - no dependencies
2. Phase 2 (Result) - needs Phase 1
3. Phase 3 (Entry points) - needs Phase 2
4. Phase 4 (Hashing) - **CRITICAL** - needs Phase 3
5. Phase 5 (Errors) - needs Phase 3
6. Phase 6 (Tests) - needs Phases 4-5
7. Phase 7 (Docs) - needs Phase 6 (to validate examples)
8. Phase 8-9 (Polish) - needs Phase 7

**Blocking Relationships**:

- PyPI publication blocked until Phase 9 complete
- Notebook updates blocked until Phase 7 complete
- GitHub Action blocked until PyPI published

---

## Success Metrics

Track in `API_DESIGN_ANALYSIS.md`:

| Metric            | Target | Measurement                          |
| ----------------- | ------ | ------------------------------------ |
| Import speed      | <200ms | `time python -c "import glassalpha"` |
| Determinism       | 100%   | 1000 runs, 0 collisions              |
| Error quality     | 100%   | All errors have fix field            |
| Type coverage     | 100%   | mypy --strict passes                 |
| Test coverage     | 95%+   | pytest --cov for api/ module         |
| Docs completeness | 100%   | All public functions have examples   |

---

## Exit Checklist

Before tagging v0.2 and publishing to PyPI:

- [ ] All 5 critical blockers fixed
- [ ] All should-fix items addressed
- [ ] Import speed <200ms verified on cold start
- [ ] Determinism verified across 10 runs (same platform)
- [ ] All errors use GlassAlphaError with code/fix/docs
- [ ] Deep immutability verified (arrays read-only, dicts frozen)
- [ ] mypy --strict passes on src/glassalpha/
- [ ] Test coverage >95% for api/ module
- [ ] All public functions have docstrings with examples
- [ ] Notebooks updated with new API
- [ ] API reference docs complete
- [ ] Stability index published
- [ ] Tolerance policy documented
- [ ] API_DESIGN_ANALYSIS.md generated
- [ ] Zero lint errors
- [ ] All contract tests pass on Linux + macOS

---

## Risk Mitigation

**Risk: Determinism across platforms**
Mitigation: Extensive testing on Linux + macOS matrix, tolerance policy for cross-platform

**Risk: Breaking existing code**
Mitigation: No backwards compatibility required for v0.2, clean break

**Risk: Import speed regression**
Mitigation: Lazy loading via PEP 562, benchmark in CI

**Risk: Scope creep**
Mitigation: Strict phase boundaries, exit checklist

---

## Communication

**To User**: Provide progress updates after each phase completion
**Commit Strategy**: One commit per phase with detailed message
**Documentation**: Update CHANGELOG.md at end with consolidated entry

---

## Timeline Estimate

- Phase 1: ~4 hours (30k tokens)
- Phase 2: ~10 hours (80k tokens)
- Phase 3: ~8 hours (60k tokens)
- Phase 4: ~9 hours (70k tokens) - CRITICAL PATH
- Phase 5: ~5 hours (40k tokens)
- Phase 6: ~10 hours (80k tokens)
- Phase 7: ~6 hours (50k tokens)
- Phase 8: ~2 hours (20k tokens)
- Phase 9: ~1 hour (10k tokens)

**Total**: ~55 agent hours (~400k tokens)

---

## Next Steps

1. Review this plan with user
2. Create phase-specific checklists
3. Begin Phase 1 implementation
4. Track progress in this document
