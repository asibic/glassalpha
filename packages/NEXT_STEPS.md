# Glass Alpha - Next Steps Quick Reference

## 🚀 Quick Start for New Conversation

```bash
# Navigate to project
cd /Users/gabe/Sites/glassalpha/packages

# Install dependencies (if not done)
pip install pandas numpy scikit-learn xgboost lightgbm shap pydantic typer pyyaml

# Test the architecture is working
python3 demo_foundation_minimal.py

# See current component status
python3 -c "
import sys
sys.path.insert(0, 'src')
from glassalpha.core import list_components
print('Registered components:', list_components())
"
```

## 📂 Where to Add New Components

### Model Wrappers
```
src/glassalpha/models/
└── tabular/
    ├── __init__.py
    ├── xgboost.py      # Create this
    ├── lightgbm.py     # Create this
    └── sklearn.py      # Create this
```

### Explainers
```
src/glassalpha/explain/
└── shap/
    ├── __init__.py
    ├── tree.py         # Create this (TreeSHAP)
    └── kernel.py       # Create this (KernelSHAP)
```

### Metrics
```
src/glassalpha/metrics/
├── performance/
│   ├── __init__.py
│   └── classification.py  # Create this
├── fairness/
│   ├── __init__.py
│   └── group_fairness.py  # Create this
└── drift/
    ├── __init__.py
    └── statistical.py      # Create this
```

## 📝 Component Templates

### Model Template
```python
# src/glassalpha/models/tabular/xgboost.py
from ...core.registry import ModelRegistry

@ModelRegistry.register("xgboost")
class XGBoostWrapper:
    capabilities = {
        "supports_shap": True,
        "data_modality": "tabular"
    }
    version = "1.0.0"

    def predict(self, X):
        # Implementation
        pass
```

### Explainer Template
```python
# src/glassalpha/explain/shap/tree.py
from ...core.registry import ExplainerRegistry

@ExplainerRegistry.register("treeshap", priority=100)
class TreeSHAPExplainer:
    capabilities = {
        "supported_models": ["xgboost", "lightgbm"]
    }
    priority = 100
    version = "1.0.0"

    def explain(self, model, X, y=None):
        # Implementation
        pass
```

### Metric Template
```python
# src/glassalpha/metrics/performance/classification.py
from ...core.registry import MetricRegistry

@MetricRegistry.register("accuracy")
class AccuracyMetric:
    metric_type = "performance"
    version = "1.0.0"

    def compute(self, y_true, y_pred, sensitive_features=None):
        # Implementation
        pass
```

## ✅ Testing Your Components

After creating each component:

```python
# Test it registers
from glassalpha.core import ModelRegistry, list_components

# Check registration
print(ModelRegistry.get("xgboost"))  # Should return your class

# Check in component list
components = list_components()
print("Models:", components['models'])

# Test basic functionality
from glassalpha.models.tabular.xgboost import XGBoostWrapper
model = XGBoostWrapper()
print(model.get_capabilities())
```

## 🎯 Order of Implementation

1. **XGBoostWrapper** - Easiest to test, most common model
2. **TreeSHAPExplainer** - Core value proposition
3. **AccuracyMetric** - Simplest metric to implement
4. **Simple Pipeline** - Connect model → explainer → metric
5. **Basic Report** - Generate first PDF

## 📋 Key Commands

```bash
# Run tests
pytest tests/

# Check component registration
python3 -c "from glassalpha.core import list_components; print(list_components())"

# Test CLI (once components exist)
glassalpha list
glassalpha validate --config configs/example_audit.yaml
glassalpha audit --config configs/example_audit.yaml --out test.pdf --dry-run
```

## 🔗 Important References

- **Architecture**: `.cursor/rules/architecture.mdc` - Design patterns
- **Priorities**: `.cursor/rules/phase1_priorities.mdc` - Current status
- **Handoff**: `HANDOFF.md` - Detailed next steps
- **Config Example**: `configs/example_audit.yaml` - Configuration structure

## 💡 Remember

1. **The architecture is proven** - NoOp components demonstrate all patterns work
2. **Use the registry** - Components auto-register with `@register` decorator
3. **Follow the patterns** - Copy NoOp component structure
4. **Test incrementally** - Verify each component registers before moving on
5. **Keep it deterministic** - Use seeds, avoid randomness

---

**You have a solid foundation. Time to build on it!**
