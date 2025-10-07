# API Redesign Quick Reference

**ONE PAGE SUMMARY** - Pin this during implementation

---

## Status

- **Band**: L (~400k tokens)
- **Risk**: Medium
- **Blocker**: Pre-launch for v0.2 PyPI
- **Timeline**: 5-7 days

---

## The 9 Phases

| Phase              | Tokens | Status     | Exit Criteria            |
| ------------------ | ------ | ---------- | ------------------------ |
| 1. Namespace       | 30k    | 🔲 Pending | Import <200ms            |
| 2. Result Object   | 80k    | 🔲 Pending | Immutability verified    |
| 3. Entry Points    | 60k    | 🔲 Pending | All entry points work    |
| 4. Hashing ⚠️      | 70k    | 🔲 Pending | Determinism 100%         |
| 5. Errors          | 40k    | 🔲 Pending | All errors have fix/docs |
| 6. Testing         | 80k    | 🔲 Pending | 95%+ coverage            |
| 7. Documentation   | 50k    | 🔲 Pending | All examples work        |
| 8. Quality Gates   | 20k    | 🔲 Pending | Registry complete        |
| 9. Stability Index | 10k    | 🔲 Pending | Policy published         |

⚠️ = Critical Path

---

## 5 Critical Blockers

- [ ] **Blocker #1**: GAE1009 override in PerformanceMetrics
- [ ] **Blocker #2**: NaN → "Unknown" (not "nan")
- [ ] **Blocker #3**: Array canonicalization preserves dtype/shape
- [ ] **Blocker #4**: Bytes canonicalize to base64
- [ ] **Blocker #5**: Data hash returns "sha256:..." prefix

---

## Key Files to Create

**Core API**:

- `src/glassalpha/__init__.py` (PEP 562)
- `src/glassalpha/api/result.py` (AuditResult)
- `src/glassalpha/api/metrics.py` (ReadonlyMetrics)
- `src/glassalpha/api/audit.py` (entry points)
- `src/glassalpha/core/canonicalization.py`
- `src/glassalpha/core/hashing.py`
- `src/glassalpha/exceptions.py`

**Tests**:

- `tests/api/test_contracts.py`
- `tests/api/test_determinism.py`
- `tests/api/test_hashing.py`

**Docs**:

- `site/docs/reference/api/audit.md`
- `site/docs/guides/probability-requirements.md`
- `site/docs/guides/missing-data.md`

---

## Success Metrics

| Metric        | Target              |
| ------------- | ------------------- |
| Import speed  | <200ms              |
| Determinism   | 100% (10 runs)      |
| Error quality | 100% (all have fix) |
| Type coverage | 100% (mypy strict)  |
| Test coverage | 95%+ (api/)         |
| Docs          | 100% (examples)     |

---

## Key Design Decisions

1. **Lazy Loading**: PEP 562 for fast imports
2. **Immutability**: Frozen dataclass + read-only arrays
3. **Access Patterns**: Dict + attribute styles
4. **Errors**: GlassAlphaError(code, message, fix, docs)
5. **Hashing**: Canonical JSON → SHA-256
6. **Data Hashes**: `"sha256:..."` format
7. **NaN Handling**: Map to "Unknown" category
8. **Timezones**: Naive treated as UTC

---

## Error Codes Quick Reference

- **GAE1001**: Invalid protected_attributes format
- **GAE1002**: Unknown metric access
- **GAE1003**: Length mismatch
- **GAE1004**: Non-binary classification
- **GAE1005**: Unsupported missing_policy
- **GAE1008**: No predict_proba (only decision_function)
- **GAE1009**: AUC access without probabilities
- **GAE1012**: MultiIndex not supported
- **GAE2002**: Result ID mismatch
- **GAE2003**: Data hash mismatch
- **GAE4001**: File already exists

---

## Testing Checklist

**Must Pass Before v0.2**:

- [ ] Import speed <200ms
- [ ] Determinism (10 runs, 0 collisions)
- [ ] Deep immutability verified
- [ ] All error codes tested
- [ ] All canonical edge cases (NaN, Inf, -0.0, etc.)
- [ ] Data hash edge cases (dtype, timezone, categorical)
- [ ] Cross-platform CI (Linux + macOS)
- [ ] 95%+ coverage for api/
- [ ] All notebooks work

---

## Documentation Checklist

**Must Complete Before v0.2**:

- [ ] API reference (audit.md, result.md, errors.md)
- [ ] Probability requirements guide
- [ ] Missing data guide
- [ ] Reproducibility guide (timezone section)
- [ ] Data requirements (MultiIndex policy)
- [ ] Stability index
- [ ] Tolerance policy
- [ ] Update quickstart notebook
- [ ] Update German Credit notebook

---

## Final Exit Checklist

**Before tagging v0.2**:

- [ ] All 9 phases complete
- [ ] All 5 blockers fixed
- [ ] All success metrics met
- [ ] All tests pass (contract, determinism, coverage)
- [ ] All docs complete
- [ ] Notebooks verified
- [ ] CI green (Linux + macOS)
- [ ] CHANGELOG.md updated
- [ ] API_DESIGN_ANALYSIS.md generated
- [ ] Temporary docs deleted

---

## Command Reference

```bash
# Test import speed
time python -c "import glassalpha"

# Run API tests
pytest tests/api/ -v

# Run contracts only
pytest tests/api/ -v -m contract

# Run determinism (10 times)
for i in {1..10}; do pytest tests/api/test_determinism.py -v; done

# Check coverage
pytest tests/api/ --cov=src/glassalpha/api --cov-report=html

# Type check
mypy --strict src/glassalpha/

# Lint
ruff check src/glassalpha/

# Format
black src/glassalpha/
```

---

## Progress Tracking

Update this section as phases complete:

- Phase 1: ✅ Complete (Import: <200ms, 20/20 tests pass)
- Phase 2: ✅ Complete (Immutability verified, 50/50 tests pass, 1 xfail for pickle)
- Phase 3: ✅ Complete (Entry points defined, 26/26 tests pass)
- Phase 4: ✅ Complete (Determinism verified, 56/56 tests pass, 0 warnings) ⚠️ CRITICAL
- Phase 5: ⏳ Not Started
- Phase 6: ⏳ Not Started
- Phase 7: ⏳ Not Started
- Phase 8: ⏳ Not Started
- Phase 9: ⏳ Not Started

**Overall**: 4/9 phases complete (44%)

---

## Related Documents

- **Master Plan**: API_REDESIGN_MASTER_PLAN.md (detailed phases)
- **Checklist**: API_REDESIGN_PHASE_CHECKLIST.md (task lists)
- **Decisions**: API_DESIGN_DECISIONS.md (20 design rationales)
- **Testing**: API_TEST_PLAN.md (60+ tests)
- **Roadmap**: API_IMPLEMENTATION_ROADMAP.md (file-by-file)
- **Summary**: API_REDESIGN_EXECUTIVE_SUMMARY.md (high-level overview)

---

## Quick Start After Approval

```bash
# 1. Start Phase 1
cd /Users/gabe/Sites/glassalpha/packages

# 2. Create api/ directory
mkdir -p src/glassalpha/api
touch src/glassalpha/api/__init__.py

# 3. Update __init__.py with PEP 562
# (see API_IMPLEMENTATION_ROADMAP.md for code)

# 4. Test
pytest tests/api/test_contracts.py::test_import_speed -v

# 5. Move to Phase 2
# (see API_REDESIGN_PHASE_CHECKLIST.md)
```

---

**🚀 READY TO LAUNCH - Awaiting approval to proceed**
