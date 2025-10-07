# API Design Decisions Log

**Purpose**: Document key design decisions and rationales for v0.2 API redesign.
**Audience**: Future maintainers, enterprise customers, regulatory auditors.

---

## D1: Lazy Module Loading (PEP 562)

**Decision**: Use PEP 562 `__getattr__` for lazy module loading.

**Rationale**:

- Import speed <200ms critical for CLI responsiveness
- Heavy dependencies (sklearn, xgboost, matplotlib) shouldn't load on `import glassalpha`
- Enables tab completion without eager loading

**Alternatives Considered**:

1. Eager loading - rejected (slow imports)
2. Separate packages - rejected (deployment complexity)

**Trade-offs**:

- Pro: Fast imports, good UX
- Con: Slightly more complex **init**.py

**Validation**: `test_import_speed()` must pass <200ms

---

## D2: Frozen Dataclass for AuditResult

**Decision**: Use `@dataclass(frozen=True)` for AuditResult.

**Rationale**:

- Immutability critical for compliance (results can't be modified post-generation)
- Frozen dataclass prevents attribute mutation
- Enables `__hash__()` for collections
- Pythonic (matches stdlib patterns)

**Alternatives Considered**:

1. NamedTuple - rejected (no methods, less readable)
2. Plain class with **setattr** guard - rejected (more boilerplate)

**Trade-offs**:

- Pro: Pythonic, enforced immutability, hashable
- Con: No dynamic attributes (feature, not bug)

**Validation**: `test_deep_immutability()` must pass

---

## D3: Dict + Attribute Access for Metrics

**Decision**: Support both `result.performance["accuracy"]` and `result.performance.accuracy`.

**Rationale**:

- Dict-style: familiar to pandas users, safe for unknown keys
- Attribute-style: ergonomic for notebooks, IDE autocomplete
- Both patterns common in data science (pandas, sklearn)

**Alternatives Considered**:

1. Dict-only - rejected (less ergonomic)
2. Attribute-only - rejected (no safe access pattern)

**Trade-offs**:

- Pro: Best of both worlds, user choice
- Con: Two code paths to maintain

**Validation**: `test_readonly_metrics_key_error()`, `test_readonly_metrics_attr_error()`

---

## D4: GlassAlphaError with Structured Fields

**Decision**: All errors use `GlassAlphaError(code, message, fix, docs)`.

**Rationale**:

- Machine-readable codes enable programmatic handling
- Actionable fixes reduce support burden
- Docs links guide users to solutions
- Regulatory compliance requires error traceability

**Alternatives Considered**:

1. Plain ValueError/TypeError - rejected (not actionable)
2. Error codes only - rejected (poor UX)

**Trade-offs**:

- Pro: Excellent UX, debuggability, compliance-ready
- Con: More verbose error definitions

**Validation**: `test_unknown_metric_error()` checks docs field

---

## D5: Canonical JSON for result.id

**Decision**: Use strict JSON canonicalization (sorted keys, NaN→null, Inf→sentinel).

**Rationale**:

- Determinism requires stable serialization
- JSON: human-readable, auditable, language-neutral
- Strict mode: no implicit NaN/Infinity handling
- Sorted keys: eliminates ordering ambiguity

**Alternatives Considered**:

1. Pickle - rejected (not auditable, Python-only)
2. MessagePack - rejected (not human-readable)
3. CBOR - rejected (less common, tooling sparse)

**Trade-offs**:

- Pro: Auditable, deterministic, portable
- Con: Infinity requires sentinel (JSON limitation)

**Validation**: `test_canonical_hash_nan()`, `test_canonical_hash_infinity()`

---

## D6: Self-Documenting Hash Format ("sha256:...")

**Decision**: Data hashes use `"sha256:{hex_digest}"` format.

**Rationale**:

- Audit trails should be self-documenting
- Enables future algorithm upgrades (SHA-512, BLAKE3)
- Clear in logs and configs
- Industry standard (Docker, Git LFS use similar)

**Alternatives Considered**:

1. Bare hex digest - rejected (ambiguous algorithm)
2. Dict `{algorithm: "sha256", digest: "..."}` - rejected (verbose)

**Trade-offs**:

- Pro: Self-documenting, future-proof, clear
- Con: Slightly longer strings (marginal)

**Validation**: `test_data_hash_prefix()`

---

## D7: NaN → "Unknown" in Protected Attributes

**Decision**: Map NaN to `"Unknown"` string category, not `"nan"`.

**Rationale**:

- Explicit: "Unknown" is clear to auditors
- Avoids confusion with string "nan" from str(np.nan)
- Deterministic category order (sorted + Unknown last)
- Fairness analysis: Unknown is a separate group

**Alternatives Considered**:

1. Drop rows with NaN - rejected (data loss, v0.3 feature)
2. String "nan" - rejected (ambiguous, unprofessional)
3. Sentinel value -999 - rejected (type confusion)

**Trade-offs**:

- Pro: Clear, auditable, deterministic
- Con: None significant

**Validation**: `test_protected_attrs_nan_to_unknown()`

---

## D8: Timezone-Naive as UTC

**Decision**: Treat timezone-naive datetimes as UTC for hashing.

**Rationale**:

- Determinism requires absolute timestamps
- Naive timestamps have no timezone info
- Assuming local time → platform-dependent hashes
- UTC assumption: documented, deterministic, industry standard

**Alternatives Considered**:

1. Reject naive timestamps - rejected (too strict, breaks existing data)
2. Assume local time - rejected (non-deterministic)
3. Sentinel for naive - rejected (breaks comparisons)

**Trade-offs**:

- Pro: Deterministic, documented, standard
- Con: Requires user awareness (mitigated with docs)

**Validation**: `test_data_hash_timezone()`

---

## D9: Array Canonicalization with dtype/shape

**Decision**: Wrap arrays in `{"__ndarray__": {dtype, shape, data}}` for canonical form.

**Rationale**:

- Determinism: int32 vs int64 must produce different hashes
- Auditable: dtype visible in JSON
- Nested arrays: recursive canonicalization handles NaN/Inf

**Alternatives Considered**:

1. Raw tolist() - rejected (loses dtype info)
2. Base64 bytes - rejected (not human-readable)

**Trade-offs**:

- Pro: Deterministic, auditable, preserves structure
- Con: More verbose JSON (acceptable for compliance)

**Validation**: `test_canonical_hash_ndarray()`, `test_data_hash_dtype_sensitivity()`

---

## D10: GAE1009 Override in PerformanceMetrics

**Decision**: Override `__getattr__` in PerformanceMetrics to raise GAE1009 for AUC metrics without probabilities.

**Rationale**:

- Specific error: tells user exactly what's missing
- Better UX than generic GAE1002 "unknown metric"
- Fix field: shows how to provide y_proba
- Docs link: guides to CalibratedClassifierCV

**Alternatives Considered**:

1. Generic GAE1002 - rejected (not actionable)
2. Omit metrics silently - rejected (confusing)

**Trade-offs**:

- Pro: Excellent UX, actionable
- Con: Extra complexity in PerformanceMetrics

**Validation**: `test_performance_metrics_gae1009_override()`

---

## D11: RangeIndex Fast Path

**Decision**: Hash RangeIndex parameters without materializing values.

**Rationale**:

- Performance: RangeIndex materialization can be expensive (millions of rows)
- Determinism: Parameters uniquely identify the index
- Common case: Most DataFrames use RangeIndex

**Alternatives Considered**:

1. Materialize all indexes - rejected (slow, memory intensive)
2. Hash index type only - rejected (loses information)

**Trade-offs**:

- Pro: Fast, deterministic, memory efficient
- Con: Special case logic (justified by perf)

**Validation**: `test_data_hash_range_index()` (3x speedup threshold)

---

## D12: Atomic Writes with .tmp Files

**Decision**: Use `os.replace()` with .tmp files for atomic writes.

**Rationale**:

- Prevents partial writes on crashes
- POSIX guarantees: os.replace() is atomic
- Audit integrity: PDF/JSON either complete or absent
- Cleanup: .tmp removed even on failure

**Alternatives Considered**:

1. Direct writes - rejected (corruption on crash)
2. fsync() only - rejected (not atomic)

**Trade-offs**:

- Pro: Data integrity, crash-safe
- Con: Slightly more complex (worth it)

**Validation**: `test_atomic_write_cleanup()`

---

## D13: equals() with Tolerance vs **eq**

**Decision**: `__eq__` uses result.id (strict), `equals()` uses tolerance.

**Rationale**:

- Strict equality: compliance requires byte-identical verification
- Tolerance equality: cross-platform testing needs floating point tolerance
- Explicit: users choose which semantics they need

**Alternatives Considered**:

1. Always use tolerance - rejected (weakens compliance guarantees)
2. Only strict equality - rejected (breaks cross-platform CI)

**Trade-offs**:

- Pro: Both strict and practical equality available
- Con: Two methods (justified by different use cases)

**Validation**: `test_result_eq_via_id()`, `test_equals_with_tolerance()`

---

## D14: Stability Index (Stable/Beta/Experimental)

**Decision**: Publish API stability guarantees with maturity levels.

**Rationale**:

- Enterprise adoption: requires stability guarantees
- Breaking change policy: clear expectations for upgrades
- Innovation: Beta/Experimental allow iteration

**Alternatives Considered**:

1. Everything stable - rejected (limits iteration)
2. No guarantees - rejected (enterprise blocker)

**Trade-offs**:

- Pro: Clear expectations, enables enterprise + innovation
- Con: Requires discipline in API evolution

**Validation**: Published in docs, tracked in CHANGELOG

---

## D15: Tolerance Policy by Metric Type

**Decision**: Different default tolerances for performance (1e-5), fairness (1e-4), calibration (1e-4).

**Rationale**:

- Performance: high precision expected (sklearn-level)
- Fairness: group statistics introduce rounding
- Calibration: binning introduces discretization
- Documented: users understand why tolerances differ

**Alternatives Considered**:

1. Single global tolerance - rejected (too crude)
2. Per-metric tolerances - rejected (too granular)

**Trade-offs**:

- Pro: Matches statistical properties of metrics
- Con: More documentation (justified)

**Validation**: Published in tolerance policy docs

---

## D16: Metric Registry for Enterprise

**Decision**: `MetricSpec` with metadata (higher_is_better, fairness_definition, tolerance).

**Rationale**:

- Enterprise dashboards: need metric metadata for visualization
- Fairness definitions: auditors need mathematical statements
- Tolerance: enables programmatic comparison

**Alternatives Considered**:

1. Metrics as simple dicts - rejected (no metadata)
2. Metadata in docs only - rejected (not programmatic)

**Trade-offs**:

- Pro: Enables enterprise features, self-documenting
- Con: More upfront work (justified by enterprise value)

**Validation**: Registry complete, ready for Phase 3 dashboards

---

## D17: Binary Classification Only (v0.2)

**Decision**: Support only binary classification in v0.2.

**Rationale**:

- Focus: tabular credit scoring (binary) is 80% of compliance use cases
- Complexity: multiclass fairness definitions are contested
- Quality: better to ship solid binary than buggy multiclass

**Alternatives Considered**:

1. Multiclass support - deferred to v0.3
2. Regression support - deferred to v0.4

**Trade-offs**:

- Pro: High-quality, tested, focused
- Con: Limited scope (acceptable for v0.2)

**Validation**: `test_multiclass_rejection()` raises GAE1004

---

## D18: missing_policy="include_unknown" Only (v0.2)

**Decision**: Support only `"include_unknown"` in v0.2, defer `"exclude_unknown"`.

**Rationale**:

- Simplicity: one policy, well-tested
- Common case: most audits include Unknown as separate group
- Quality: avoid bugs from complex exclusion logic

**Alternatives Considered**:

1. Both policies - deferred to v0.3

**Trade-offs**:

- Pro: Simple, robust, well-documented
- Con: Limited flexibility (acceptable for v0.2)

**Validation**: `test_missing_policy_validation()` raises GAE1005

---

## D19: MultiIndex Rejection (v0.2)

**Decision**: Reject MultiIndex for both index and columns in v0.2.

**Rationale**:

- Complexity: MultiIndex hashing is error-prone
- Rare: most ML datasets use simple indexes
- Quality: avoid bugs in edge cases

**Alternatives Considered**:

1. MultiIndex support - deferred to v0.3

**Trade-offs**:

- Pro: Simple, robust, clear error message
- Con: Limited (rare use case)

**Validation**: `test_multiindex_rejection()` raises GAE1012

---

## D20: Deep Copy Inputs in from_model()

**Decision**: Deep copy X, y at entry to from_model().

**Rationale**:

- Safety: prevents accidental mutation by pipeline
- Compliance: input data must be pristine for hash computation
- Pythonic: functions shouldn't mutate arguments

**Alternatives Considered**:

1. No copy (trust pipeline) - rejected (unsafe)
2. Copy-on-write - rejected (complexity)

**Trade-offs**:

- Pro: Safe, predictable, compliant
- Con: Memory overhead (acceptable for audit workloads)

**Validation**: `test_inputs_are_copied()`

---

## Summary of Key Principles

1. **Determinism First**: All design decisions prioritize byte-identical reproducibility
2. **Compliance-Ready**: Errors, hashes, manifests designed for regulatory scrutiny
3. **Ergonomic**: API matches pandas/sklearn patterns for familiarity
4. **Auditable**: JSON, explicit hashes, self-documenting formats
5. **Safe**: Immutability, input copying, atomic writes
6. **Enterprise-Ready**: Stability guarantees, metric metadata, tolerance policies
7. **Focused**: Binary classification only for v0.2 quality

**Next Version (v0.3) Will Add**:

- Multiclass classification
- exclude_unknown policy
- MultiIndex support
- Additional metric types

**Review**: These decisions locked for v0.2. Breaking changes require v1.0 per stability index.
