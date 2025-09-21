# Quick Start Guide

Get up and running with Glass Alpha in under 5 minutes.

## The 60-Second Hello World

```python
# 1. Import Glass Alpha
from glassalpha import explain, audit
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb

# 2. Create sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Train a model
model = xgb.XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Generate explanations
explanations = explain(model, X_test)
print(f"Top feature: {explanations.feature_importance[0]}")

# 5. Create audit report
report = audit(model, X_test, y_test, output="my_first_audit.pdf")
print("✅ Audit report generated!")
```

## Core Concepts

### Explainability

Glass Alpha uses TreeSHAP for efficient and accurate explanations:

```python
from glassalpha import Explainer

# Initialize explainer
explainer = Explainer(model, X_train)

# Get SHAP values
shap_values = explainer.explain(X_test)

# Visualize
explainer.plot_summary()  # Feature importance summary
explainer.plot_waterfall(X_test[0])  # Individual explanation
```

### Fairness Analysis

Detect and measure bias across protected attributes:

```python
from glassalpha import FairnessAnalyzer

# Define protected attributes
analyzer = FairnessAnalyzer(
    protected_attributes=['gender', 'race']
)

# Run analysis
results = analyzer.analyze(model, X_test, y_test)

# Check metrics
print(f"Disparate Impact: {results.disparate_impact}")
print(f"Equal Opportunity Difference: {results.equal_opportunity_diff}")
```

### Counterfactual Explanations

Find minimal changes needed to flip predictions:

```python
from glassalpha import CounterfactualExplainer

# Initialize with constraints
cf_explainer = CounterfactualExplainer(
    model,
    immutable_features=['age', 'gender'],
    feature_ranges={'income': (20000, 200000)}
)

# Generate counterfactual
counterfactual = cf_explainer.explain(
    X_test[0], 
    desired_outcome=1
)

print(f"Changes needed: {counterfactual.changes}")
```

### Audit Reports

Generate comprehensive, reproducible audit reports:

```python
from glassalpha import AuditReport

# Configure report
config = {
    'include_shap': True,
    'include_fairness': True,
    'include_drift': False,
    'confidence_level': 0.95
}

# Generate report
report = AuditReport(model, config=config)
report.fit(X_train, y_train)
report.generate(X_test, y_test, output="audit_report.pdf")

# Access programmatically
metrics = report.get_metrics()
print(f"Model accuracy: {metrics['accuracy']}")
```

## Real-World Example: Credit Scoring

```python
import pandas as pd
from glassalpha import GlassAlpha

# Load data
data = pd.read_csv("german_credit.csv")
X = data.drop(['default'], axis=1)
y = data['default']

# Initialize Glass Alpha with policy
glass = GlassAlpha(
    config="configs/policy/lending.yaml",
    random_seed=42
)

# Train model with Glass Alpha wrapper
model = glass.train(X_train, y_train, algorithm='xgboost')

# Full compliance workflow
results = glass.analyze(model, X_test, y_test)

# Check compliance
if results.is_compliant:
    print("✅ Model passes all compliance checks")
    glass.deploy(model, "production/model.pkl")
else:
    print("❌ Compliance issues found:")
    print(results.violations)
```

## Configuration with YAML

Glass Alpha supports configuration-driven workflows:

```yaml
# config.yaml
model:
  type: xgboost
  params:
    max_depth: 5
    learning_rate: 0.1

explainability:
  method: treeshap
  background_samples: 100

fairness:
  protected_attributes:
    - gender
    - race
  metrics:
    - disparate_impact
    - equal_opportunity
  thresholds:
    disparate_impact: 0.8

audit:
  output_format: pdf
  include_sections:
    - executive_summary
    - model_performance
    - fairness_analysis
    - explainability
  reproducibility:
    save_manifest: true
    track_seeds: true
```

Load and use configuration:

```python
from glassalpha import GlassAlpha

glass = GlassAlpha.from_config("config.yaml")
results = glass.run_full_pipeline(X, y)
```

## Best Practices

### 1. Always Set Seeds
```python
from glassalpha.utils import set_global_seed

set_global_seed(42)  # Ensures reproducibility
```

### 2. Use Immutable Manifests
```python
report = AuditReport(track_manifest=True)
# Automatically saves: config hash, data hash, git commit, seeds
```

### 3. Define Policies Upfront
```python
from glassalpha import Policy

policy = Policy(
    immutable_features=['ssn', 'date_of_birth'],
    monotonic_constraints={'age': 'positive'},
    acceptable_ranges={'income': (0, 1000000)}
)

glass = GlassAlpha(policy=policy)
```

## What's Next?

- 📊 [Explore Example Notebooks](examples.md)
- 🔧 [Deep Dive into API](../api/overview.md)
- 🎯 [Advanced Features](../features/explainability.md)
- 💡 [Real-world Use Cases](../examples/german-credit.md)
