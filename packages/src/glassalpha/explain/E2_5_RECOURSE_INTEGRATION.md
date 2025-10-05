# E2.5 (Recourse) Integration Guide

This document describes how E2 (Reason Codes) provides the foundation for E2.5 (Recourse) implementation.

## Overview

**E2 (Reason Codes)** extracts features that negatively impacted a decision.
**E2.5 (Recourse)** generates actionable recommendations to flip that decision.

They share common infrastructure:

- Feature contribution extraction (SHAP values)
- Policy constraint validation (immutables, monotonic constraints)
- Protected attribute handling
- Deterministic ranking logic

## Shared Components

### 1. Feature Contribution Extraction

**E2 Implementation:**

```python
# packages/src/glassalpha/explain/reason_codes.py
def extract_reason_codes(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: pd.Series,
    ...
) -> ReasonCodeResult:
    """Extract top-N negative contributions."""
    # Filters negative SHAP values
    # Excludes protected attributes
    # Ranks by magnitude
```

**E2.5 Reuse:**

```python
# Future: packages/src/glassalpha/explain/recourse.py
from glassalpha.explain import extract_reason_codes

def generate_recourse(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: pd.Series,
    policy_constraints: PolicyConstraints,
    ...
) -> RecourseResult:
    """Generate actionable counterfactuals."""

    # Step 1: Identify negative contributors (reuse E2)
    reason_result = extract_reason_codes(
        shap_values=shap_values,
        feature_names=feature_names,
        feature_values=feature_values,
        protected_attributes=policy_constraints.immutable_features,
        ...
    )

    # Step 2: Generate counterfactuals
    # For each reason code, propose feasible changes
    for code in reason_result.reason_codes:
        if code.feature in policy_constraints.immutable_features:
            continue  # Skip immutable

        # Generate counterfactual...
```

### 2. Policy Constraint Validation

**Shared Module:**

```python
# packages/src/glassalpha/explain/policy.py

@dataclass(frozen=True)
class PolicyConstraints:
    """Shared policy constraints."""
    immutable_features: list[str]
    monotonic_constraints: dict[str, Literal["increase_only", "decrease_only"]]
    feature_costs: dict[str, float]
    feature_bounds: dict[str, tuple[float, float]]

# Validation functions used by both E2 and E2.5
def validate_immutables(features, immutable_features) -> list[str]
def validate_monotonic_constraints(feature, old_val, new_val, constraints) -> bool
def compute_feature_cost(feature, old_val, new_val, costs) -> float
def validate_feature_bounds(feature, value, bounds) -> bool
```

**E2 Usage:**

```python
# Reason codes use immutable validation
excluded = []
for feature in feature_names:
    if feature.lower() in immutable_set:
        excluded.append(feature)
```

**E2.5 Usage:**

```python
# Recourse enforces full policy
from glassalpha.explain.policy import (
    validate_monotonic_constraints,
    compute_feature_cost,
    validate_feature_bounds,
)

# Validate proposed change
if not validate_monotonic_constraints(
    feature, old_value, new_value, constraints
):
    continue  # Invalid change

# Compute cost
cost = compute_feature_cost(
    feature, old_value, new_value, feature_costs
)
```

### 3. Configuration Schema

**Already Defined:**

```python
# packages/src/glassalpha/config/schema.py

class RecourseConfig(BaseModel):
    """Recourse configuration (E2.5)."""
    enabled: bool = False
    immutable_features: list[str]
    monotonic_constraints: dict[str, str]
    cost_function: str = "weighted_l1"
    max_iterations: int = 100
```

**Integration:**

```yaml
# Config file
data:
  protected_attributes: # Used by E2
    - age
    - gender

recourse: # Used by E2.5
  enabled: true
  immutable_features: # Merged with protected_attributes
    - age
    - gender
    - nationality
  monotonic_constraints:
    income: "increase_only"
    debt: "decrease_only"
  feature_costs:
    credit_history: 0.9 # Hard to change
    telephone: 0.1 # Easy to change
```

## Implementation Roadmap for E2.5

### Step 1: Core Recourse Module

Create `packages/src/glassalpha/explain/recourse.py`:

```python
from dataclasses import dataclass
from glassalpha.explain import ReasonCodeResult, extract_reason_codes
from glassalpha.explain.policy import PolicyConstraints

@dataclass(frozen=True)
class RecourseRecommendation:
    """Single recourse recommendation."""
    feature: str
    current_value: float | str
    recommended_value: float | str
    cost: float
    rank: int

@dataclass(frozen=True)
class RecourseResult:
    """Complete recourse generation result."""
    instance_id: str | int
    current_prediction: float
    target_prediction: float
    recommendations: list[RecourseRecommendation]
    total_cost: float
    feasible: bool
    reason_codes_used: ReasonCodeResult  # Link back to E2

def generate_recourse(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: pd.Series,
    model: Any,
    instance_id: str | int,
    current_prediction: float,
    target_prediction: float,
    policy_constraints: PolicyConstraints,
    seed: int = 42,
) -> RecourseResult:
    """Generate actionable counterfactual recommendations.

    This function builds on E2 (Reason Codes) by:
    1. Identifying negative contributors (reuse extract_reason_codes)
    2. Proposing feasible changes (respecting policy constraints)
    3. Optimizing for minimum cost
    4. Validating counterfactual achieves target
    """
```

### Step 2: CLI Command

Add to `packages/src/glassalpha/cli/commands.py`:

```python
def recourse(
    model: Path,
    data: Path,
    instance: int,
    config: Path | None = None,
    output: Path | None = None,
    target_prediction: float = 0.5,
    max_cost: float = 10.0,
):
    """Generate actionable recourse recommendations.

    Examples:
        glassalpha recourse \
            --model model.pkl \
            --data test.csv \
            --instance 42 \
            --target 0.6 \
            --max-cost 5.0
    """
```

### Step 3: Integration Points

**With E2 (Reason Codes):**

- Reuse `extract_reason_codes()` to identify features to change
- Reuse `ReasonCodeResult` in `RecourseResult` for audit trail
- Share protected attribute exclusion logic

**With Policy Module:**

- Use `PolicyConstraints` for all validation
- Use `validate_monotonic_constraints()` to filter proposals
- Use `compute_feature_cost()` for optimization
- Use `validate_feature_bounds()` for feasibility

**With Configuration:**

- Merge `data.protected_attributes` and `recourse.immutable_features`
- Use `recourse.monotonic_constraints` for directional constraints
- Use `recourse.feature_costs` for optimization weights

### Step 4: Testing Strategy

**Contract Tests:**

```python
# tests/unit/test_recourse_contract.py

def test_recourse_respects_immutables():
    """Immutable features never in recommendations."""

def test_recourse_respects_monotonic_constraints():
    """Recommendations follow directional constraints."""

def test_recourse_minimizes_cost():
    """Lower cost recommendations ranked first."""

def test_recourse_achieves_target():
    """Proposed changes flip decision to target."""

def test_recourse_is_deterministic():
    """Same seed → same recommendations."""
```

**Integration Tests:**

```python
# tests/integration/test_recourse_e2e.py

def test_recourse_uses_reason_codes():
    """Recourse builds on reason code extraction."""
    # Generate reason codes
    reason_result = extract_reason_codes(...)

    # Generate recourse using same instance
    recourse_result = generate_recourse(...)

    # Verify consistency
    assert recourse_result.reason_codes_used == reason_result
```

## API Design (E2.5)

### Python API

```python
from glassalpha.explain import generate_recourse
from glassalpha.explain.policy import PolicyConstraints

# Define policy constraints
policy = PolicyConstraints(
    immutable_features=["age", "gender"],
    monotonic_constraints={
        "income": "increase_only",
        "debt": "decrease_only",
    },
    feature_costs={
        "credit_history": 0.9,
        "telephone": 0.1,
    },
    feature_bounds={
        "age": (18, 100),
        "income": (0, 1000000),
    },
)

# Generate recourse
result = generate_recourse(
    shap_values=shap_values,
    feature_names=feature_names,
    feature_values=instance_values,
    model=model,
    instance_id=42,
    current_prediction=0.35,
    target_prediction=0.6,
    policy_constraints=policy,
    seed=42,
)

# Access recommendations
for rec in result.recommendations:
    print(f"{rec.rank}. {rec.feature}: {rec.current_value} → {rec.recommended_value}")
    print(f"   Cost: {rec.cost:.2f}")
```

### CLI API

```bash
# Generate recourse recommendations
glassalpha recourse \
  --model models/credit.pkl \
  --data data/test.csv \
  --instance 42 \
  --config configs/recourse.yaml \
  --output recourse/instance_42.txt

# With custom target and cost limit
glassalpha recourse \
  --model model.pkl \
  --data test.csv \
  --instance 10 \
  --target 0.7 \
  --max-cost 5.0 \
  --format json
```

## File Structure (E2.5)

```
packages/
  src/glassalpha/
    explain/
      reason_codes.py        # E2 (existing)
      recourse.py            # E2.5 (new)
      policy.py              # Shared (existing)
      __init__.py            # Export both

  configs/
    recourse_german_credit.yaml  # E2.5 example config

  tests/
    unit/
      test_reason_codes_contract.py    # E2 (existing)
      test_recourse_contract.py        # E2.5 (new)
      test_policy_constraints.py       # Shared (existing)
    integration/
      test_recourse_e2e.py             # E2.5 integration

  site/docs/guides/
    reason-codes.md          # E2 (existing)
    recourse.md              # E2.5 (new)
```

## Key Design Decisions

### 1. Shared Policy Module

**Why:** Avoid duplication between E2 and E2.5.
**How:** Create `policy.py` with pure validation functions.
**Benefit:** Single source of truth for constraints.

### 2. Reuse Reason Codes Extraction

**Why:** Recourse needs to know what to change (E2 answers this).
**How:** Call `extract_reason_codes()` inside `generate_recourse()`.
**Benefit:** Consistent feature ranking, audit trail linkage.

### 3. Config Schema Already Exists

**Why:** `RecourseConfig` already defined in schema.
**How:** Use existing schema, merge with protected attributes.
**Benefit:** No schema changes needed, immediate integration.

### 4. Separate Modules, Shared Exports

**Why:** Keep concerns separated but easy to use together.
**How:** `reason_codes.py` and `recourse.py` as separate modules, both exported from `explain/__init__.py`.
**Benefit:** Can use independently or together.

## Testing E2.5 with E2

### Consistency Test

```python
def test_recourse_uses_same_features_as_reason_codes():
    """Recourse recommendations target features from reason codes."""
    # Generate reason codes
    reason_result = extract_reason_codes(
        shap_values=shap_values,
        feature_names=feature_names,
        feature_values=instance_values,
        instance_id=42,
        prediction=0.35,
        seed=42,
    )

    # Generate recourse
    recourse_result = generate_recourse(
        shap_values=shap_values,
        feature_names=feature_names,
        feature_values=instance_values,
        model=model,
        instance_id=42,
        current_prediction=0.35,
        target_prediction=0.6,
        policy_constraints=policy,
        seed=42,
    )

    # Verify recourse targets features from reason codes
    reason_features = {code.feature for code in reason_result.reason_codes}
    recourse_features = {rec.feature for rec in recourse_result.recommendations}

    # Recourse should target subset of reason code features
    # (after excluding immutables and applying constraints)
    assert recourse_features.issubset(reason_features)
```

## Summary

**What's Ready:**

- ✅ `extract_reason_codes()` - Feature contribution extraction
- ✅ `policy.py` - Shared constraint validation
- ✅ `RecourseConfig` - Config schema
- ✅ `DEFAULT_PROTECTED_ATTRIBUTES` - Protected attribute list
- ✅ Deterministic ranking logic (seeded)
- ✅ Contract tests for shared components

**What's Needed for E2.5:**

- `recourse.py` - Core recourse generation module
- `generate_recourse()` - Main API function
- `RecourseResult` - Result dataclass
- CLI command: `glassalpha recourse`
- User guide: `site/docs/guides/recourse.md`
- Contract tests: `test_recourse_contract.py`
- Integration tests: `test_recourse_e2e.py`

**Estimated Effort:** ~150k tokens | Band: M

The foundation is solid. E2.5 implementation can proceed immediately with minimal coupling.
