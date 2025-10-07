# Tolerance Policy

This document defines default tolerances for metric comparisons in GlassAlpha, explaining why exact equality is unreliable and how to validate reproducibility correctly.

---

## The Problem with Exact Equality

**Why `result1 == result2` fails**:

```python
# Same model, same data, same seed
result1 = ga.audit.from_model(..., random_seed=42)
result2 = ga.audit.from_model(..., random_seed=42)

# ❌ FAILS due to floating-point differences
assert result1.performance.accuracy == result2.performance.accuracy

# Platform differences:
# Linux:  0.8500000000000001
# macOS:  0.8499999999999999
# Windows: 0.8500000000000000
```

**Root causes**:
1. **Floating-point arithmetic**: Different CPU architectures, compiler optimizations
2. **Library implementations**: NumPy/sklearn may use different BLAS/LAPACK backends
3. **Summation order**: Parallel operations may sum in different orders

---

## Solution: Relative + Absolute Tolerance

GlassAlpha uses **`numpy.allclose()`** logic:

```python
abs(a - b) <= (atol + rtol * abs(b))
```

**Terminology**:
- **`rtol`** (relative tolerance): Tolerance as a fraction of the value
- **`atol`** (absolute tolerance): Absolute tolerance (for values near zero)

**Example**:
```python
# rtol=1e-5, atol=1e-8
result1.equals(result2, rtol=1e-5, atol=1e-8)

# Accuracy: 0.85 vs 0.8500000001
# abs(0.85 - 0.8500000001) = 1e-10
# 1e-10 <= (1e-8 + 1e-5 * 0.85) = 1e-8 + 8.5e-6 ≈ 8.5e-6 ✅

# Fairness: 0.02 vs 0.020001
# abs(0.02 - 0.020001) = 1e-6
# 1e-6 <= (1e-8 + 1e-5 * 0.02) = 1e-8 + 2e-7 ≈ 2e-7 ❌
```

---

## Default Tolerances by Metric Type

### Performance Metrics (Standard)

**Default**: `rtol=1e-5`, `atol=1e-8`

| Metric | rtol | atol | Rationale |
|--------|------|------|-----------|
| Accuracy | 1e-5 | 1e-8 | ~0.001% relative error |
| Precision | 1e-5 | 1e-8 | ~0.001% relative error |
| Recall | 1e-5 | 1e-8 | ~0.001% relative error |
| F1 | 1e-5 | 1e-8 | ~0.001% relative error |
| ROC AUC | 1e-5 | 1e-8 | AUC is stable |
| PR AUC | 1e-5 | 1e-8 | AUC is stable |
| Brier Score | 1e-5 | 1e-8 | MSE-based, stable |
| Log Loss | 1e-5 | 1e-8 | Log-based, needs small tolerance |

**Why 1e-5?**
- Allows ~0.001% relative error
- Example: Accuracy 0.85 ± 0.0000085
- Catches real differences while allowing floating-point noise

---

### Calibration Metrics (Looser)

**Default**: `rtol=1e-4`, `atol=1e-6`

| Metric | rtol | atol | Rationale |
|--------|------|------|-----------|
| ECE (Expected Calibration Error) | 1e-4 | 1e-6 | Binning introduces variance |
| MCE (Maximum Calibration Error) | 1e-4 | 1e-6 | Max operator amplifies noise |

**Why looser?**
- Calibration metrics use binning (bin edges can shift)
- Max operators amplify small differences
- Still catches real calibration drift

**Example**:
```python
# ECE: 0.0500 vs 0.0501
# abs(0.0500 - 0.0501) = 0.0001
# 0.0001 <= (1e-6 + 1e-4 * 0.0500) = 1e-6 + 5e-6 ≈ 6e-6 ❌
# Real difference! ECE changed by 2%

# ECE: 0.0500 vs 0.05000001
# abs(0.0500 - 0.05000001) = 1e-8
# 1e-8 <= 6e-6 ✅ Floating-point noise
```

---

### Fairness Metrics (Standard)

**Default**: `rtol=1e-5`, `atol=1e-8`

| Metric | rtol | atol | Rationale |
|--------|------|------|-----------|
| Demographic Parity Diff | 1e-5 | 1e-8 | Rate differences are stable |
| Equalized Odds Diff | 1e-5 | 1e-8 | TPR/FPR differences are stable |
| Equal Opportunity Diff | 1e-5 | 1e-8 | TPR differences are stable |

**Why standard?**
- Fairness metrics are rate differences (stable)
- Example: DP diff 0.05 ± 0.0000005
- Must catch real fairness drift

---

### Count Metrics (Exact)

**Default**: `rtol=0.0`, `atol=0.0`

| Metric | rtol | atol | Rationale |
|--------|------|------|-----------|
| Monotonicity Violations | 0.0 | 0.0 | Integer counts must match exactly |
| Support (sample sizes) | 0.0 | 0.0 | Integer counts must match exactly |

**Why exact?**
- Counts are integers (no floating-point error)
- Any difference is a real difference

---

## Usage Examples

### Default Tolerance

```python
import glassalpha as ga

result1 = ga.audit.from_model(..., random_seed=42)
result2 = ga.audit.from_model(..., random_seed=42)

# Uses default tolerances (rtol=1e-5, atol=1e-8)
assert result1.equals(result2)
```

### Custom Tolerance

```python
# Stricter tolerance for critical audits
assert result1.equals(result2, rtol=1e-6, atol=1e-9)

# Looser tolerance for exploratory analysis
assert result1.equals(result2, rtol=1e-4, atol=1e-6)
```

### Per-Metric Tolerance

```python
# Check only performance metrics with custom tolerance
perf1 = result1.performance
perf2 = result2.performance

import numpy as np
assert np.allclose(perf1.accuracy, perf2.accuracy, rtol=1e-6, atol=1e-9)
assert np.allclose(perf1.precision, perf2.precision, rtol=1e-6, atol=1e-9)
```

---

## Reproducibility Validation

### CI/CD: Byte-Identical Config

For **maximum reproducibility** in CI/CD:

```yaml
# .github/workflows/audit.yml
- name: Validate reproducibility
  run: |
    # Run audit twice
    glassalpha audit --config audit.yaml --out audit1.json
    glassalpha audit --config audit.yaml --out audit2.json
    
    # Validate result IDs match (byte-identical)
    python -c "
    import json
    r1 = json.load(open('audit1.json'))
    r2 = json.load(open('audit2.json'))
    assert r1['result_id'] == r2['result_id'], 'Non-deterministic!'
    "
```

**Why this works**:
- `result_id` is a hash of canonical JSON
- Canonical JSON uses fixed precision (6 decimals)
- Result IDs match if and only if all metrics match at that precision

---

### Notebook: Tolerance-Based Comparison

For **exploratory analysis** in notebooks:

```python
# Run audit, save baseline
baseline = ga.audit.from_model(..., random_seed=42)
baseline.save("baseline.json")

# Later: Load and compare
current = ga.audit.from_model(..., random_seed=42)
baseline_loaded = ga.audit.AuditResult.load("baseline.json")

# Validate reproducibility
assert current.equals(baseline_loaded, rtol=1e-5, atol=1e-8)
```

---

## Cross-Platform Differences

### Known Issues

**BLAS/LAPACK backends**:
- **macOS Accelerate**: Apple's optimized BLAS
- **OpenBLAS**: Common on Linux
- **MKL**: Intel Math Kernel Library (Conda)

**Impact**:
- Matrix operations may differ in last few digits
- Summation order may differ (parallel operations)

**Mitigation**:
```bash
# Force single-threaded BLAS (deterministic)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

### Testing Strategy

**Local**: Test with default tolerance
```bash
pytest tests/test_reproducibility.py
```

**CI**: Test with stricter tolerance + deterministic BLAS
```yaml
env:
  OMP_NUM_THREADS: 1
  MKL_NUM_THREADS: 1
  OPENBLAS_NUM_THREADS: 1

- run: pytest tests/test_reproducibility.py --strict-tolerance
```

---

## FAQ

### Q: Why not just use exact equality?

**A**: Floating-point arithmetic is not associative. Even with the same seed, different platforms may compute slightly different results.

**Example**:
```python
# Different summation order
(0.1 + 0.2) + 0.3 != 0.1 + (0.2 + 0.3)
# 0.6000000000000001 != 0.6
```

---

### Q: What tolerance should I use in production?

**A**: Use **default tolerance** (`rtol=1e-5, atol=1e-8`) for most cases.

Use **stricter tolerance** (`rtol=1e-6, atol=1e-9`) if:
- Critical compliance audit (SEC, FDA, etc.)
- Need to catch 0.0001% changes

Use **looser tolerance** (`rtol=1e-4, atol=1e-6`) if:
- Cross-platform comparison (dev laptop vs. CI)
- Exploratory analysis (not regulatory)

---

### Q: How do I validate byte-identical reproducibility?

**A**: Use `result_id` hashing:

```python
result1 = ga.audit.from_model(..., random_seed=42)
result2 = ga.audit.from_model(..., random_seed=42)

# If result IDs match, outputs are byte-identical (at JSON precision)
assert result1.result_id == result2.result_id
```

**Why this works**: `result_id` is SHA-256 hash of canonical JSON. Any metric difference → different ID.

---

### Q: Can I disable tolerance checking?

**A**: Yes, but not recommended:

```python
# ❌ Exact equality (will fail due to floating-point)
assert result1.performance.accuracy == result2.performance.accuracy

# ✅ Tolerance-based (recommended)
assert result1.equals(result2, rtol=1e-5, atol=1e-8)

# ⚠️ Zero tolerance (only for count metrics)
assert result1.equals(result2, rtol=0.0, atol=0.0)
```

---

### Q: How do I debug tolerance failures?

**A**: Compare metric-by-metric:

```python
result1 = ga.audit.from_model(..., random_seed=42)
result2 = ga.audit.from_model(..., random_seed=42)

# Find which metric differs
for key, val1 in result1.performance.items():
    val2 = result2.performance[key]
    diff = abs(val1 - val2)
    rel_diff = diff / abs(val2) if val2 != 0 else float('inf')
    
    if diff > 1e-8 + 1e-5 * abs(val2):
        print(f"{key}: {val1} vs {val2} (diff={diff:.2e}, rel={rel_diff:.2e})")
```

---

## Related Documentation

- [API Reference: AuditResult.equals()](audit-entry-points.md#auditresultequals)
- [Reproducibility Guide](../../guides/reproducibility.md) - Ensuring deterministic audits
- [Stability Index](stability-index.md) - API stability levels

---

**Last Updated**: 2025-10-07  
**Applies to**: v0.2.0+

