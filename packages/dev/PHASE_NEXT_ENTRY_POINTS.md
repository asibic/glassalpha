# Entry Point Implementation: from_model, from_predictions, from_config

**Context**: API redesign (Phases 1-9) complete. Entry point stubs exist but need implementation.

**Goal**: Wire up the new high-level API (`ga.audit.from_model()`, etc.) to the existing pipeline infrastructure.

**Estimated Effort**: ~100-150k tokens | Band: M | Risk: Medium

---

## Current State

### What's Complete ✅

**API Infrastructure (Phases 1-9)**:
- `AuditResult` dataclass (immutable, pickable)
- `ReadonlyMetrics` base class (dict + attribute access)
- Entry point signatures defined in `src/glassalpha/api/audit.py`
- Deterministic hashing: `compute_result_id()`, `hash_data_for_manifest()`
- Error classes: `GlassAlphaError` + 10 subclasses (GAE1001-GAE4001)
- Metric registry: `MetricSpec` for 13 metrics
- 213 tests passing (86% coverage)

**Existing Pipeline**:
- `glassalpha.pipeline.AuditPipeline` (YAML-based config)
- Model wrappers: `XGBoostWrapper`, `LightGBMWrapper`, etc.
- Metrics: performance, fairness, calibration, stability
- SHAP explainers
- PDF report generation

### What's Missing ⏳

**Entry point implementations** (currently `NotImplementedError` stubs):
1. `ga.audit.from_model()` - Connect model + data to pipeline
2. `ga.audit.from_predictions()` - Compute metrics from predictions
3. `ga.audit.from_config()` - Load YAML config, run pipeline

**Integration work**:
- Map new API parameters to pipeline config
- Extract results from pipeline into `AuditResult`
- Handle protected attributes (NaN → "Unknown")
- Validate inputs (MultiIndex rejection, length checks)
- Error handling with new error codes

---

## Implementation Plan

### Task 1: from_predictions() (~40k tokens)

**Why start here**: Simplest - no model or explainers, just metric computation.

**Signature**:
```python
def from_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_proba: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    protected_attributes: Optional[dict[str, Union[pd.Series, np.ndarray]]] = None,
    missing_policy: Literal["error", "impute", "ignore"] = "error",
    probability_threshold: float = 0.5,
    fairness_metrics: Optional[list[str]] = None,
    performance_metrics: Optional[list[str]] = None,
    calibration_metrics: Optional[list[str]] = None,
    **kwargs: Any,
) -> AuditResult:
```

**Implementation steps**:
1. **Input validation**:
   - Check y_true, y_pred lengths match (raise GAE1003)
   - Check for MultiIndex (raise GAE1012)
   - Check binary classification (raise GAE1004 if >2 classes)
   - Validate protected_attributes format (raise GAE1001)
   - Check y_proba provided if AUC requested (raise GAE1009)

2. **Handle protected attributes**:
   - Apply missing_policy (error/impute/ignore)
   - NaN → "Unknown" if impute
   - Raise GAE1005 if invalid policy

3. **Compute metrics**:
   - Performance: accuracy, precision, recall, f1, roc_auc, pr_auc, brier, log_loss
   - Fairness: demographic_parity, equalized_odds, equal_opportunity (if protected_attributes provided)
   - Calibration: ECE, MCE (if y_proba provided)

4. **Build AuditResult**:
   - Create `PerformanceMetrics` from computed metrics
   - Create `FairnessMetrics` (if applicable)
   - Create `CalibrationMetrics` (if applicable)
   - Generate manifest with data hashes
   - Compute result_id

5. **Test coverage**:
   - Happy path: binary classification with all metrics
   - Edge cases: no y_proba (skip AUC), no protected_attributes (skip fairness)
   - Error cases: length mismatch, MultiIndex, non-binary, missing y_proba for AUC

**Files to create/modify**:
- `src/glassalpha/api/audit.py` - Implement `from_predictions()`
- `src/glassalpha/api/builders.py` - Helper to build `AuditResult` from metrics
- `tests/api/test_from_predictions.py` - Integration tests

---

### Task 2: from_model() (~50k tokens)

**Signature**:
```python
def from_model(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    protected_attributes: Optional[dict[str, Union[pd.Series, np.ndarray]]] = None,
    missing_policy: Literal["error", "impute", "ignore"] = "error",
    probability_threshold: float = 0.5,
    random_seed: Optional[int] = None,
    explainer_type: Literal["shap", "lime"] = "shap",
    fairness_metrics: Optional[list[str]] = None,
    performance_metrics: Optional[list[str]] = None,
    calibration_metrics: Optional[list[str]] = None,
    stability_metrics: Optional[list[str]] = None,
    **kwargs: Any,
) -> AuditResult:
```

**Implementation steps**:
1. **Wrap model**:
   - Detect model type (XGBoost, LightGBM, sklearn, etc.)
   - Use `glassalpha.models.registry.get_model_wrapper()`
   - Check for `predict_proba()` (raise GAE1008 if missing for probability-based metrics)

2. **Generate predictions**:
   - `y_pred = model.predict(X)`
   - `y_proba = model.predict_proba(X)` (if available)
   - Apply probability_threshold

3. **Compute explanations** (if random_seed provided):
   - Use `glassalpha.explain.shap` or `lime` explainer
   - Generate feature importance
   - Compute top features per prediction
   - Seed explainer for reproducibility

4. **Delegate to from_predictions()**:
   - Call `from_predictions(y_true=y, y_pred=y_pred, y_proba=y_proba, ...)`
   - Add explanation results to AuditResult

5. **Test coverage**:
   - XGBoost model with SHAP
   - sklearn model with SHAP
   - Model without predict_proba (raise GAE1008)
   - Random seed reproducibility (same seed → same result_id)

**Files to create/modify**:
- `src/glassalpha/api/audit.py` - Implement `from_model()`
- `src/glassalpha/api/model_utils.py` - Model wrapping helpers
- `tests/api/test_from_model.py` - Integration tests

---

### Task 3: from_config() (~30k tokens)

**Signature**:
```python
def from_config(config_path: Union[str, Path], **kwargs: Any) -> AuditResult:
```

**Implementation steps**:
1. **Load config**:
   - Parse YAML using existing `glassalpha.config.loader`
   - Validate schema
   - Check file exists (raise GAE4001)

2. **Load data + model**:
   - Load X, y from paths in config
   - Load protected_attributes from paths
   - Load model from pickle/joblib
   - Validate data hashes (raise GAE2003 if mismatch)

3. **Delegate to from_model()**:
   - Extract parameters from config
   - Call `from_model(model=model, X=X, y=y, ...)`

4. **Validate result_id** (if in config):
   - Compare computed result_id to expected
   - Raise GAE2002 if mismatch

5. **Test coverage**:
   - Valid config → successful audit
   - Missing file (raise GAE4001)
   - Hash mismatch (raise GAE2003)
   - Result ID mismatch (raise GAE2002)

**Files to create/modify**:
- `src/glassalpha/api/audit.py` - Implement `from_config()`
- `tests/api/test_from_config.py` - Integration tests

---

### Task 4: Update Notebooks (~20k tokens)

**Goal**: Migrate examples to new API.

**Files to update**:
1. `examples/notebooks/quickstart_colab.ipynb`
2. `examples/notebooks/german_credit_walkthrough.ipynb`

**Before (old API)**:
```python
from glassalpha.pipeline import AuditPipeline

pipeline = AuditPipeline(config="audit.yaml")
result = pipeline.run()
```

**After (new API)**:
```python
import glassalpha as ga

result = ga.audit.from_model(
    model=model,
    X=X_test,
    y=y_test,
    protected_attributes={"gender": gender, "age": age},
    random_seed=42
)

# Rich display in notebook
display(result)  # Uses _repr_html_()

# Export
result.to_pdf("audit.pdf")
result.save("result.json")
```

**Changes**:
- Add smoke cell (from `docs.mdc`)
- Use `from_model()` instead of `AuditPipeline`
- Show attribute access: `result.performance.accuracy`
- Show dict access: `result.performance["accuracy"]`
- Demonstrate `equals()` for reproducibility

**Test notebooks**:
```bash
cd examples/notebooks
jupyter nbconvert --to notebook --execute quickstart_colab.ipynb --output quickstart_colab_tested.ipynb
```

---

### Task 5: Integration Tests (~10k tokens)

**Goal**: End-to-end tests with real models + data.

**Test scenarios**:
1. **German Credit with XGBoost**:
   - `from_model()` with full config
   - Verify all metrics computed
   - Check result_id stability (same seed → same ID)

2. **Adult Income with sklearn**:
   - `from_predictions()` with pre-computed predictions
   - Fairness metrics with protected attributes

3. **Config-based audit**:
   - `from_config()` with YAML
   - Verify hash validation
   - Verify result_id validation

**Files to create**:
- `tests/integration/test_api_e2e.py`

---

## Design Decisions

### 1. Error Handling Strategy

**Decision**: Validate early, fail fast.

**Rationale**: Compliance audits require clear failures. Better to raise `GAE1003_LENGTH_MISMATCH` immediately than get cryptic sklearn errors later.

**Implementation**:
```python
def from_predictions(y_true, y_pred, ...):
    # Validate immediately
    if len(y_true) != len(y_pred):
        raise LengthMismatchError(
            expected_length=len(y_true),
            actual_length=len(y_pred),
            arrays=["y_true", "y_pred"]
        )
    
    # Then compute
    ...
```

---

### 2. Protected Attributes Handling

**Decision**: NaN → "Unknown" category by default.

**Rationale**: Fairness analysis requires non-null groups. "Unknown" is more informative than "nan".

**Implementation**:
```python
def _handle_protected_attrs(attrs, missing_policy):
    if missing_policy == "error":
        if any(pd.isna(attr).any() for attr in attrs.values()):
            raise InvalidProtectedAttributesError("NaN values found")
    elif missing_policy == "impute":
        return {k: v.fillna("Unknown") for k, v in attrs.items()}
    elif missing_policy == "ignore":
        mask = ~pd.concat(attrs.values(), axis=1).isna().any(axis=1)
        return {k: v[mask] for k, v in attrs.items()}
```

---

### 3. Metric Selection Logic

**Decision**: Compute all metrics by default, skip if requirements not met.

**Rationale**: Users expect comprehensive audits. Better to skip AUC (with warning) than fail.

**Implementation**:
```python
def _compute_performance_metrics(y_true, y_pred, y_proba, metrics=None):
    results = {}
    
    # Always compute label-based metrics
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["precision"] = precision_score(y_true, y_pred)
    ...
    
    # Skip probability metrics if y_proba is None
    if y_proba is not None:
        results["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        logger.warning("Skipping ROC AUC: y_proba not provided")
    
    return results
```

---

### 4. Manifest Generation

**Decision**: Include data hashes in manifest for reproducibility.

**Rationale**: Enables `from_config()` to validate data hasn't changed.

**Implementation**:
```python
manifest = {
    "glassalpha_version": glassalpha.__version__,
    "timestamp": datetime.now(UTC).isoformat(),
    "random_seed": random_seed,
    "data_hashes": {
        "X": hash_data_for_manifest(X),
        "y": hash_data_for_manifest(y),
    },
    "model_type": type(model).__name__,
}
```

---

### 5. Result ID Computation

**Decision**: Hash canonical JSON of result (excluding result_id itself).

**Rationale**: Enables byte-identical comparison without circular dependency.

**Implementation**:
```python
def compute_result_id(result_dict):
    # Exclude result_id from hash
    canonical = {k: v for k, v in result_dict.items() if k != "result_id"}
    canonical_json = canonicalize(canonical)
    return hashlib.sha256(canonical_json.encode()).hexdigest()[:16]
```

---

## Testing Strategy

### Unit Tests (src/glassalpha/api/)

**Coverage target**: 95%+

**Focus areas**:
- Input validation (all error codes)
- Protected attribute handling (error/impute/ignore)
- Metric computation (with/without y_proba)
- Result ID stability

### Integration Tests (tests/integration/)

**Focus areas**:
- End-to-end with real models (XGBoost, sklearn)
- Reproducibility (same seed → same result_id)
- Config-based workflow (YAML → AuditResult)

### Notebook Tests (examples/notebooks/)

**Strategy**:
- Use `nbconvert --execute` to run notebooks
- Check for errors
- Verify expected outputs exist

---

## Acceptance Criteria

Before marking complete:

- [ ] All 3 entry points implemented (no `NotImplementedError`)
- [ ] All 10 error codes raised in appropriate situations
- [ ] Protected attribute handling (error/impute/ignore) works
- [ ] Result ID is stable (same inputs → same ID)
- [ ] Manifest includes data hashes
- [ ] 2 notebooks migrated to new API
- [ ] Notebooks run without errors
- [ ] 95%+ test coverage for api/ module
- [ ] Integration tests pass with real models
- [ ] Documentation examples work

---

## Risk Assessment

### High Risk ⚠️

**Model wrapping complexity**: Existing model wrappers may need refactoring.

**Mitigation**: Start with `PassThroughModel` for predictions-only testing. Add model wrapping incrementally.

### Medium Risk ⚠️

**Config schema changes**: Existing YAML configs may not map cleanly to new API.

**Mitigation**: Support both old and new config formats initially. Provide migration guide.

### Low Risk ✅

**Metric computation**: Existing metrics infrastructure is stable.

**Mitigation**: Reuse existing `glassalpha.metrics` modules directly.

---

## Next Session Prompt

Use this prompt to start the next session:

---

## 🚀 Prompt for Next Session

**Goal**: Implement entry points for GlassAlpha v0.2 API

**Context**: API redesign complete (Phases 1-9). Infrastructure ready:
- `AuditResult` dataclass (immutable)
- Entry point stubs in `src/glassalpha/api/audit.py`
- 10 error classes (GAE1001-GAE4001)
- Metric registry (13 metrics)
- 213 tests passing

**Task**: Wire up new API to existing pipeline:

1. **from_predictions()** (~40k tokens)
   - Compute metrics from y_true, y_pred, y_proba
   - Handle protected attributes (NaN → "Unknown")
   - Build AuditResult with manifest + result_id
   - Test: happy path + 10 error codes

2. **from_model()** (~50k tokens)
   - Wrap model, generate predictions
   - Compute SHAP explanations (seeded)
   - Delegate to from_predictions()
   - Test: XGBoost + sklearn, reproducibility

3. **from_config()** (~30k tokens)
   - Load YAML config
   - Validate data hashes
   - Delegate to from_model()
   - Test: hash validation, result_id validation

4. **Update notebooks** (~20k tokens)
   - Migrate quickstart_colab.ipynb
   - Migrate german_credit_walkthrough.ipynb
   - Add smoke cells, demonstrate new API

5. **Integration tests** (~10k tokens)
   - E2E with German Credit + XGBoost
   - Config-based workflow

**Files**:
- Read: `packages/dev/PHASE_NEXT_ENTRY_POINTS.md` (this file)
- Implement: `src/glassalpha/api/audit.py`
- Test: `tests/api/test_from_*.py`

**Acceptance criteria**:
- All 3 entry points work (no NotImplementedError)
- 95%+ test coverage
- 2 notebooks run without errors
- Result ID stable (same seed → same ID)

**Start with**: Task 1 (from_predictions) - simplest, no model wrapping.

---

## Reference: Existing Infrastructure

### Model Wrappers
- `glassalpha.models.registry.get_model_wrapper(model)` - Auto-detect model type
- `glassalpha.models.tabular.xgboost.XGBoostWrapper`
- `glassalpha.models.tabular.lightgbm.LightGBMWrapper`
- `glassalpha.models.base.PassThroughModel` - For predictions-only

### Metrics
- `glassalpha.metrics.performance` - accuracy, precision, recall, f1, auc, brier
- `glassalpha.metrics.fairness` - demographic_parity, equalized_odds, equal_opportunity
- `glassalpha.metrics.calibration` - ECE, MCE

### Explainers
- `glassalpha.explain.shap.tree.TreeSHAP` - For XGBoost/LightGBM
- `glassalpha.explain.shap.kernel.KernelSHAP` - For any model

### Config Loading
- `glassalpha.config.loader.load_config(path)` - Parse YAML
- `glassalpha.config.validator.validate_config(config)` - Schema validation

### Data Hashing
- `glassalpha.core.canonicalization.hash_data_for_manifest(data)` - Already implemented
- `glassalpha.core.canonicalization.compute_result_id(result_dict)` - Already implemented

---

## Appendix: Error Code Quick Reference

| Code | Error | When to Raise |
|------|-------|---------------|
| GAE1001 | InvalidProtectedAttributesError | protected_attributes not dict or wrong format |
| GAE1003 | LengthMismatchError | X, y, protected_attributes length mismatch |
| GAE1004 | NonBinaryClassificationError | >2 classes in y |
| GAE1005 | UnsupportedMissingPolicyError | missing_policy not in ["error", "impute", "ignore"] |
| GAE1008 | NoPredictProbaError | Model has no predict_proba() but y_proba needed |
| GAE1009 | AUCWithoutProbaError | Accessing result.performance.auc but y_proba=None |
| GAE1012 | MultiIndexNotSupportedError | X or y has MultiIndex |
| GAE2002 | ResultIDMismatchError | Computed result_id ≠ expected in config |
| GAE2003 | DataHashMismatchError | Data hash ≠ expected in config |
| GAE4001 | FileExistsError | File exists and overwrite=False |

---

**Ready to implement! Start with from_predictions() for quickest validation.**

