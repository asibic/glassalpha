# GlassAlpha - AI Compliance Toolkit

> ⚠️ **Pre-Alpha Status**: GlassAlpha is under active development. Core functionality is being built to deliver deterministic, regulator-ready PDF audits.

## Architecture Overview

GlassAlpha is built as an **extensible framework** for AI compliance and interpretability, starting with tabular ML models in Phase 1 and designed to expand to LLMs and multimodal systems.

### Core Design Principles

1. **Plugin Architecture**: All components (models, explainers, metrics) use dynamic registration
2. **Deterministic Behavior**: Configuration-driven with reproducible results  
3. **Clear OSS/Enterprise Separation**: Core functionality is open-source, advanced features are commercial
4. **Modality-Agnostic Interfaces**: Designed to support tabular, text, and vision models

## Key Features

### Phase 1 (Current Focus)
- ✅ XGBoost, LightGBM, LogisticRegression support
- ✅ TreeSHAP explainability  
- ✅ Fairness metrics (demographic parity, equal opportunity)
- ✅ Basic recourse (immutables, monotonicity)
- ✅ Deterministic PDF audit reports
- ✅ Full reproducibility (seeds, hashes, manifests)

### Future Phases
- 🚧 LLM support with gradient-based explainability
- 🚧 Vision model support
- 🚧 Continuous monitoring dashboards
- 🚧 Regulator-specific templates
- 🚧 Cloud integrations

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

Future enterprise package (separate repository):
```
glassalpha-enterprise/       # Commercial license
├── advanced_explain/        # Deep SHAP, gradient methods
├── monitoring/              # Dashboards & drift tracking
├── templates/               # Regulator-specific templates
├── integrations/            # SageMaker, Databricks, etc.
└── policy_packs/           # Compliance policy bundles
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

### Enterprise Installation (Future)
```bash
# Requires license key
pip install glassalpha-enterprise
export GLASSALPHA_LICENSE_KEY="your-key-here"
```

## Quick Start

### Phase 1: Generate Audit PDF
```bash
# Basic audit
glassalpha audit --config configs/example_audit.yaml --out audit.pdf

# Strict mode for regulatory compliance
glassalpha audit --config configs/example_audit.yaml --out audit.pdf --strict
```

### Configuration Example
```yaml
audit_profile: "tabular_compliance"

model:
  type: "xgboost"
  path: "model.pkl"

explainers:
  priority:
    - treeshap
    - kernelshap

strict_mode: true  # Enforce regulatory requirements
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
pytest  # Configured to require 90% coverage
```

Test determinism:
```bash
make audit-test  # Should produce byte-identical PDFs
```

## Contributing

See [CONTRIBUTING.md](../contributing.md) for development guidelines.

## License

- Core library: Apache 2.0 License
- Enterprise extensions: Commercial License (contact for details)

## Support

- OSS: GitHub Issues
- Enterprise: Priority support with SLA