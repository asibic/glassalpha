# GlassAlpha - Phase 2 Quick Reference

## 🚀 Quick Start for Phase 2

```bash
# Navigate to project
cd /path/to/glassalpha/packages

# Activate virtual environment
source venv/bin/activate

# Verify Phase 1 components are working
python3 -c "
import sys
sys.path.insert(0, 'src')
from glassalpha.core import list_components
c = list_components()
print(f'✅ Models: {len(c[\"models\"])} registered')
print(f'✅ Explainers: {len(c[\"explainers\"])} registered')
print(f'✅ Metrics: {len(c[\"metrics\"])} registered')
print(f'Total: {len(c[\"models\"]) + len(c[\"explainers\"]) + len(c[\"metrics\"])} components')
"

# Expected output:
# ✅ Models: 5 registered
# ✅ Explainers: 3 registered
# ✅ Metrics: 17 registered
# Total: 25 components
```

## 📂 Where to Add Phase 2 Components

### Data Module
```
src/glassalpha/data/
├── __init__.py
├── base.py           # DataInterface protocol (exists)
└── tabular.py        # Create this - CSV/Parquet loading

src/glassalpha/datasets/
├── __init__.py
├── german_credit.py  # Create this
└── adult_income.py   # Create this
```

### Utilities
```
src/glassalpha/utils/
├── __init__.py
├── seeds.py          # Create this - Centralized seed management
├── hashing.py        # Create this - Deterministic hashing
└── manifest.py       # Create this - Audit manifest generation
```

### Pipeline
```
src/glassalpha/pipeline/
├── __init__.py
└── audit.py          # Create this - Main audit orchestrator
```

### Report Generation
```
src/glassalpha/report/
├── __init__.py
├── templates/
│   └── standard_audit.html  # Create this - Jinja2 template
├── renderers/
│   └── pdf.py               # Create this - WeasyPrint renderer
└── plots.py                 # Create this - Deterministic plotting
```

## 📝 Phase 2 Component Templates

### Data Loader Template
```python
# src/glassalpha/data/tabular.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from pydantic import BaseModel

class DataSchema(BaseModel):
    """Schema for tabular data validation."""
    features: list[str]
    target: str
    sensitive_features: Optional[list[str]] = None

class TabularDataLoader:
    """Load and validate tabular datasets."""

    def load_csv(self, path: Path, schema: DataSchema) -> pd.DataFrame:
        """Load CSV with schema validation."""
        df = pd.read_csv(path)
        self._validate_schema(df, schema)
        return df

    def split_features_target(
        self, df: pd.DataFrame, schema: DataSchema
    ) -> Tuple[pd.DataFrame, np.ndarray, Optional[pd.DataFrame]]:
        """Split into X, y, and sensitive features."""
        X = df[schema.features]
        y = df[schema.target].values
        sensitive = df[schema.sensitive_features] if schema.sensitive_features else None
        return X, y, sensitive
```

### Pipeline Template
```python
# src/glassalpha/pipeline/audit.py
from typing import Dict, Any
from ..core import ModelRegistry, ExplainerRegistry, MetricRegistry

class AuditPipeline:
    """Orchestrate the complete audit process."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """Execute the audit pipeline."""
        # 1. Load data
        data = self._load_data()

        # 2. Load/train model
        model = self._get_model()

        # 3. Generate explanations
        explanations = self._generate_explanations(model, data)

        # 4. Compute metrics
        metrics = self._compute_metrics(model, data)

        # 5. Generate report
        report = self._generate_report(explanations, metrics)

        return report
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

## 🎯 Phase 2 Order of Implementation

1. **Data Module** (`data/tabular.py`) - Foundation for everything
2. **Utilities** (`utils/seeds.py`, `utils/hashing.py`) - Enable reproducibility
3. **Simple Pipeline** (`pipeline/audit.py`) - Connect existing components
4. **Basic HTML Report** (`report/templates/standard_audit.html`) - Visualize results
5. **PDF Generation** (`report/renderers/pdf.py`) - Final deliverable
6. **German Credit Example** - First complete end-to-end audit

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

## 📊 Phase 1 Completed Components Summary

All ML components are fully implemented and tested:
- **Models**: 5 wrappers (XGBoost, LightGBM, LogisticRegression, SklearnGeneric, PassThrough)
- **Explainers**: 3 implementations (TreeSHAP, KernelSHAP, NoOp)
- **Metrics**: 17 total (6 Performance, 4 Fairness, 5 Drift, 2 Testing)
- **Registry**: Working with priority-based selection
- **Config**: Pydantic schemas ready
- **CLI**: Basic structure with Typer

## 🧪 Known Issues to Address in Phase 2

### Config Loader Fix Needed
- **Issue**: `load_config()` has bug in `apply_profile_defaults()` call
- **Location**: `src/glassalpha/config/loader.py:160`
- **Priority**: HIGH - Need this for pipeline to work
- **Fix**: Update function call with correct arguments

### Testing Strategy for Phase 2
- **Current Coverage**: ~30% (Phase 1 complete)
- **Phase 2 Goal**: Add integration tests for complete pipeline
- **Focus**: End-to-end audit generation with German Credit dataset

---

**You have a solid foundation. Time to build on it!**
