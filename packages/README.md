# GlassAlpha - AI Compliance Toolkit

GlassAlpha is an **extensible framework** for AI compliance and interpretability, supporting tabular ML models with planned expansion to LLMs and multimodal systems.

### Core Design Principles

1. **Plugin Architecture**: All components (models, explainers, metrics) use dynamic registration
2. **Deterministic Behavior**: Configuration-driven with reproducible results
3. **Clear OSS/Enterprise Separation**: Core functionality is open-source, advanced features are commercial
4. **Modality-Agnostic Interfaces**: Designed to support tabular, text, and vision models

For detailed architecture information, see the [Architecture Guide](../site/docs/architecture.md).

## Key Features

### Current Features
- ✅ XGBoost, LightGBM, LogisticRegression support
- ✅ TreeSHAP explainability
- ✅ Fairness metrics (demographic parity, equal opportunity)
- ✅ Basic recourse (immutables, monotonicity)
- ✅ Deterministic PDF audit reports
- ✅ Full reproducibility (seeds, hashes, manifests)

### Planned Extensions
- LLM support with gradient-based explainability
- Vision model support
- Continuous monitoring dashboards
- Regulator-specific templates
- Cloud integrations

### Project Structure

```
packages/
├── src/
│   └── glassalpha/          # OSS core (Apache 2.0)
│       ├── core/            # Interfaces & protocols
│       ├── models/          # Model wrappers & registry
│       ├── explain/         # Explainability plugins
│       ├── metrics/         # Performance & fairness metrics
│       ├── data/            # Data handling & validation
│       ├── report/          # Report generation
│       ├── profiles/        # Audit profiles
│       ├── cli/             # CLI interface (Typer)
│       └── utils/           # Utilities (seeds, hashing, etc.)
├── configs/                 # Example configurations
├── tests/                   # Test suite
└── pyproject.toml          # Package configuration
```

## Installation

### Basic Installation (OSS)
```bash
pip install glassalpha
```

### Development Installation
```bash
cd packages
pip install -e ".[dev]"
```

## Quick Start

### Generate Audit PDF
```bash
# Basic audit
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf

# Strict mode for regulatory compliance
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf --strict
```

### Configuration Example
```yaml
audit_profile: german_credit_default

data:
  path: data/german_credit_processed.csv
  target_column: credit_risk
  protected_attributes:
    - gender
    - age_group

model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100
    max_depth: 5

explainers:
  strategy: first_compatible
  priority:
    - treeshap
    - kernelshap

reproducibility:
  random_seed: 42
```

## Architecture Highlights

### 1. Protocol-Based Interfaces
All components implement protocols for maximum flexibility:
```python
from typing import Protocol

class ModelInterface(Protocol):
    def predict(self, X): ...
    def get_capabilities(self) -> dict: ...
```

### 2. Registry Pattern
Components self-register for dynamic loading:
```python
@ModelRegistry.register("xgboost")
class XGBoostWrapper(ModelInterface):
    capabilities = {"supports_shap": True, "data_modality": "tabular"}
```

### 3. Deterministic Plugin Selection
Configuration drives component selection with deterministic priority:
```yaml
explainers:
  strategy: "first_compatible"
  priority: ["treeshap", "kernelshap"]
```

### 4. Audit Profiles
Different audit types use different component sets:
```python
class TabularComplianceProfile(AuditProfile):
    compatible_models = ["xgboost", "lightgbm", "logistic_regression"]
    required_metrics = ["accuracy", "fairness", "drift"]
```

## Testing

Run the test suite with coverage:
```bash
pytest  # Comprehensive test coverage
```

Test audit generation:
```bash
# Generate test audit to verify functionality
glassalpha audit --config configs/german_credit_simple.yaml --output test_audit.pdf
```

## Contributing

See [CONTRIBUTING.md](../site/docs/contributing.md) for development guidelines.

## License & Dependencies

### Licensing Structure

- **Core Library**: Apache 2.0 License - Full open source access to core functionality

### Dependency Compatibility

All dependencies are chosen for enterprise compatibility and regulatory compliance:

| Component | License | Purpose | Enterprise Ready |
|-----------|---------|---------|------------------|
| **SHAP** | MIT | TreeSHAP explanations | ✅ No GPL contamination |
| **XGBoost** | Apache 2.0 | Tree models | ✅ Compatible license family |
| **LightGBM** | MIT | Alternative models | ✅ Microsoft-backed |
| **scikit-learn** | BSD | Baseline models | ✅ Academic standard |
| **NumPy/Pandas** | BSD | Data processing | ✅ Core scientific stack |

**Key Licensing Note**: GlassAlpha uses the MIT-licensed Python [SHAP](https://github.com/shap/shap) library, not the GPL-licensed R `treeshap` package, ensuring maximum enterprise compatibility.

### Reproducible Builds

Dependencies are locked for deterministic builds:
```bash
pip install -c constraints.txt -e .
```

## Support

- OSS: GitHub Issues
