# Explainer Selection: Compatibility Checking Gap

## Issue

**Status**: Documented for future improvement
**Priority**: Medium (UX enhancement)
**Impact**: Late failures when users request incompatible explainers

## Current Behavior

When users explicitly specify explainer priority in config:

```yaml
explainers:
  priority: ["treeshap"] # User wants TreeSHAP
```

The selection logic (`select_explainer()` → `_first_available()`) only checks:

1. ✅ Is explainer registered?
2. ✅ Are dependencies installed? (e.g., `shap` package)
3. ❌ **Is it compatible with the model type?** ← **NOT CHECKED**

This means:

- User requests TreeSHAP for logistic regression
- Selection "succeeds" (TreeSHAP is available)
- Pipeline continues
- **Fails later** during explanation generation

## Why It Matters

For a compliance/audit tool, **failing late is problematic**:

- Wastes user time (fails after data loading, model init, etc.)
- Unclear error messages (deep in SHAP internals)
- Poor UX for non-expert users

## Desired Behavior

Selection should validate compatibility early:

```python
# User config
explainers:
  priority: ["treeshap"]

# Model type
model_type: "logistic_regression"

# Desired outcome
RuntimeError: "TreeSHAP is not compatible with logistic_regression.
Compatible explainers: coefficients, kernelshap.
Try: pip install 'glassalpha[explain]'"
```

## Technical Details

### Where the gap exists

`packages/src/glassalpha/explain/registry.py`:

```python
def _first_available(candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if _available(c):  # Only checks dependencies
            return c      # ← Should also check is_compatible()
    return None
```

### Why it's not trivial

1. **Need model_type context**: `_first_available()` doesn't receive model_type parameter
2. **Need ExplainerRegistry reference**: To call `is_compatible(name, model_type)`
3. **Performance**: Adding compatibility check for every candidate
4. **Backwards compatibility**: Some configs might depend on current behavior

### Proposed Solution

```python
def _first_available_and_compatible(
    candidates: Iterable[str],
    model_type: str,
    registry: ExplainerRegistryClass
) -> str | None:
    """Find first explainer that is both available AND compatible."""
    for c in candidates:
        if _available(c) and registry.is_compatible(c, model_type):
            return c
    return None
```

Then update `select_explainer()`:

```python
def select_explainer(model_type: str, requested_priority: list[str] | None = None) -> str:
    if requested_priority:
        chosen = _first_available_and_compatible(
            requested_priority,
            model_type,
            ExplainerRegistry
        )
        if chosen:
            return chosen

        # Better error message
        available = [c for c in requested_priority if _available(c)]
        if available:
            raise RuntimeError(
                f"Explainers {available} are installed but not compatible with '{model_type}'. "
                f"Compatible explainers for {model_type}: {SUPPORTED_FAMILIES.get(model_type, [])}. "
            )
        else:
            raise RuntimeError(
                f"No explainer from {requested_priority} is available. "
                f"Try: pip install 'glassalpha[explain]'"
            )
```

## Workaround for Users

Until fixed, users should:

1. Check model type compatibility before setting priority
2. Rely on automatic selection (don't set explicit priority)
3. Use broad priority lists: `["treeshap", "kernelshap", "coefficients"]`

## Test Impact

Current test suite works around this by:

- `test_pipeline_basic.py::test_select_explainer_basic`: Tests with _non-existent_ explainer, not _incompatible_ one
- Tests assume compatibility checking happens elsewhere (it does, but later)

## Related Files

- `packages/src/glassalpha/explain/registry.py`: Selection logic
- `packages/src/glassalpha/explain/registry.py`: `ExplainerRegistryClass.is_compatible()` method
- `packages/tests/test_pipeline_basic.py`: Test adjusted to match current behavior

## Decision

**Not fixing now because:**

1. Requires careful design for error messages
2. May need backwards compatibility handling
3. Test suite already adjusted to work with current behavior
4. Users can workaround by using automatic selection

**Should fix when:**

- Multiple users report confusion about late failures
- Adding audit config validation tooling
- Major refactor of explainer selection logic

## References

- Discussion: [Test failure analysis - 2025-01-03]
- Related: Phase 2.5 explainer API standardization
