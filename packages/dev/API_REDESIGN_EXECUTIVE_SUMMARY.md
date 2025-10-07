# API Redesign - Executive Summary

**Status**: Pre-Launch Blocker for v0.2 PyPI Publication
**Created**: 2025-10-07
**Effort**: ~400k tokens | Band: L | Risk: Medium

---

## What's Happening

Redesigning the GlassAlpha API to be best-in-class (exceeding pandas/scikit-learn/requests ergonomics) while maintaining byte-identical reproducibility for compliance audits.

**Why Now**: Current API misaligned with documentation. No backwards compatibility required - perfect time for breaking changes before PyPI publication.

---

## What You Get

### User Experience

**Before** (Current):

```python
from glassalpha.pipeline import run_audit
result = run_audit(config_path="audit.yaml")
# Result is nested dict, hard to explore
# Errors are generic ValueError/TypeError
```

**After** (v0.2):

```python
import glassalpha as ga

# Fast import (<200ms)
result = ga.audit.from_model(
    model=xgb_model,
    X=X_test,
    y=y_test,
    protected_attributes={"gender": gender, "race": race},
    random_seed=42
)

# Rich API
result.performance.accuracy              # 0.847
result.fairness.demographic_parity_diff  # 0.023
result.calibration.plot()                # Inline visualization

# Multiple access patterns
result.performance["accuracy"]           # Dict-style
result.performance.accuracy              # Attribute-style

# Helpful errors
try:
    result.performance.roc_auc  # Model has no probabilities
except ga.GlassAlphaError as e:
    print(e.code)  # GAE1009_PROBA_REQUIRED_FOR_AUC
    print(e.fix)   # "Wrap model in CalibratedClassifierCV"
    print(e.docs)  # "https://glassalpha.com/..."

# Immutable (compliance-safe)
result.id                                # SHA-256 hash (deterministic)
result.to_pdf("audit.pdf")               # Export
result.to_config()                       # Reproduction config
```

### Developer Experience

**Determinism**:

- Same inputs → same `result.id` (byte-identical)
- Cross-platform reproducibility with tolerance policy
- Self-documenting hashes (`"sha256:abc123..."`)

**Type Safety**:

- 100% type hints, mypy --strict passes
- IDE autocomplete for all methods
- Frozen dataclasses prevent mutation

**Error Quality**:

- All errors have: code, message, fix, docs URL
- Machine-readable codes (GAE1001, GAE1009, etc.)
- Actionable fixes reduce support burden

**Testing**:

- 95%+ coverage for api/ module
- Comprehensive edge case testing
- Cross-platform CI validation

---

## Documents Created

I've created **5 planning documents** to guide implementation:

1. **[API_REDESIGN_MASTER_PLAN.md](API_REDESIGN_MASTER_PLAN.md)**

   - 9 implementation phases
   - 5 critical blockers
   - Exit checklist
   - Risk mitigation

2. **[API_REDESIGN_PHASE_CHECKLIST.md](API_REDESIGN_PHASE_CHECKLIST.md)**

   - Detailed checklist for each phase
   - Test requirements
   - Validation criteria

3. **[API_DESIGN_DECISIONS.md](API_DESIGN_DECISIONS.md)**

   - 20 key design decisions with rationales
   - Alternatives considered
   - Trade-offs documented

4. **[API_TEST_PLAN.md](API_TEST_PLAN.md)**

   - 60+ contract tests
   - Coverage requirements
   - Performance benchmarks

5. **[API_IMPLEMENTATION_ROADMAP.md](API_IMPLEMENTATION_ROADMAP.md)**
   - File-by-file changes
   - Dependency graph
   - Integration strategy

---

## The 9 Phases

### Phase 1: Namespace (~30k tokens)

Fast imports with PEP 562 lazy loading
**Exit**: Import <200ms

### Phase 2: Result Object (~80k tokens)

Immutable AuditResult with rich API
**Exit**: Deep immutability verified

### Phase 3: Entry Points (~60k tokens)

Ergonomic from_model(), from_predictions(), from_config()
**Exit**: All entry points work

### Phase 4: Hashing (~70k tokens) **CRITICAL PATH**

Byte-identical reproducibility
**Exit**: Determinism verified across 10 runs

### Phase 5: Errors (~40k tokens)

Helpful GlassAlphaError with codes
**Exit**: All errors have code/fix/docs

### Phase 6: Testing (~80k tokens)

Comprehensive contract tests
**Exit**: 95%+ coverage, all tests pass

### Phase 7: Documentation (~50k tokens)

Complete user-facing docs + notebooks
**Exit**: All examples work

### Phase 8: Quality Gates (~20k tokens)

Metric registry for enterprise
**Exit**: Registry complete

### Phase 9: Stability Index (~10k tokens)

API stability guarantees
**Exit**: Stability policy published

---

## Critical Blockers (Must Fix)

These **5 issues** are pre-launch blockers:

1. ✅ **GAE1009 Override**: PerformanceMetrics raises specific error for missing probabilities
2. ✅ **NaN → "Unknown"**: Missing values map to `"Unknown"` category (not `"nan"`)
3. ✅ **Array dtype preservation**: Canonicalization distinguishes int32 vs int64
4. ✅ **Bytes handling**: Canonicalize to base64 for JSON
5. ✅ **Hash prefix**: Data hashes use `"sha256:..."` format

---

## Success Metrics

| Metric            | Target | How Measured                         |
| ----------------- | ------ | ------------------------------------ |
| Import speed      | <200ms | `time python -c "import glassalpha"` |
| Determinism       | 100%   | 1000 runs, 0 ID collisions           |
| Error quality     | 100%   | All errors have fix field            |
| Type coverage     | 100%   | mypy --strict passes                 |
| Test coverage     | 95%+   | pytest --cov for api/                |
| Docs completeness | 100%   | All public functions have examples   |

---

## Timeline

**Estimated**: 5-7 days of focused agent work (~400k tokens)

**Breakdown**:

- Phase 1-3: Foundation (30% of work)
- Phase 4: Hashing - Critical Path (20% of work)
- Phase 5-6: Quality (30% of work)
- Phase 7-9: Documentation & Polish (20% of work)

**Blocking**: PyPI publication blocked until Phase 9 complete

---

## What Happens After

### v0.2 Launch

1. Tag v0.2.0
2. Publish to PyPI
3. Update README with new API
4. Announce on GitHub

### Post-Launch Work Unblocked

- GitHub Action (needs PyPI package)
- Interactive notebooks (needs published package)
- Pre-commit hooks
- Template repositories

### Future Versions

**v0.3**: Multiclass, exclude_unknown policy, MultiIndex support
**v0.4**: Regression support
**Enterprise**: Advanced features after OSS adoption

---

## Risks & Mitigation

**Risk**: Determinism across platforms
**Mitigation**: Tolerance policy + extensive CI testing

**Risk**: Import speed regression
**Mitigation**: Lazy loading + import time benchmarks in CI

**Risk**: Scope creep
**Mitigation**: Strict phase boundaries, exit checklists

**Risk**: Breaking existing code
**Mitigation**: Clean break (no backwards compatibility), clear migration guide

---

## Decision Points

### ✅ Approved Design Choices

- PEP 562 lazy loading
- Frozen dataclass for AuditResult
- Dict + attribute access for metrics
- GlassAlphaError with structured fields
- Canonical JSON for result.id
- Self-documenting hash format
- NaN → "Unknown" mapping
- Timezone-naive as UTC

### 🔒 One-Way Doors (Can't Change After v0.2)

- result.id computation algorithm
- Data hash format (`"sha256:..."`)
- Error code assignments (GAE1001, etc.)
- Protected attributes NaN handling
- Timezone-naive as UTC policy

---

## Next Steps

### For You (User)

1. **Review** this summary and the 5 planning documents
2. **Approve** or request changes to the plan
3. **Monitor** progress as I work through phases

### For Me (Agent)

1. **Begin Phase 1** after your approval
2. **Update checklists** as work completes
3. **Commit** after each phase with detailed message
4. **Report** progress after major milestones
5. **Delete** temporary docs before final delivery

---

## Questions to Consider

Before we proceed, consider:

1. **Timeline**: Is 5-7 days acceptable for this blocker?
2. **Risk**: Are you comfortable with the "Medium" risk rating?
3. **Scope**: Any features you want added/removed?
4. **Priorities**: Should any phases be reordered?

**My Recommendation**: Proceed as planned. This is a pre-launch blocker, and the investment (400k tokens) is justified by the quality improvement and enterprise-readiness it delivers.

---

## Approval

**Ready to proceed?** Reply with:

- ✅ "Proceed" - Start Phase 1 immediately
- 🔄 "Adjust X" - Request changes to plan
- ❓ "Question about Y" - Clarify specific aspects

**Once approved**, I'll:

1. Start Phase 1 (Namespace & Import)
2. Track progress in checklists
3. Report back after each major phase
4. Deliver completed API redesign with full test coverage and docs

---

**This is the foundation for v0.2 PyPI publication and all Phase 2 priorities. Let's make it exceptional.**
