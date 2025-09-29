# GlassAlpha

!!! info "What is GlassAlpha?"
GlassAlpha delivers **deterministic, regulator-ready PDF audit reports** for tabular ML models. An open-source toolkit for teams who need reproducible, audit-ready model documentation.

## The Goal: One Command Audit Generation

Our target is simple, powerful audit generation:

!!! example "CLI Interface"

````bash # Generate audit PDF
glassalpha audit --config configs/german_credit_simple.yaml --output my_audit.pdf

    # Produces byte-identical PDF audits with complete lineage tracking
    ```
    **Capability:** Generate deterministic audit PDFs for XGBoost, LightGBM, and Logistic Regression models.

## Why We're Building GlassAlpha

### Designed for Regulatory Compliance

- **Deterministic outputs** - Identical PDFs on same seed/data/model
- **Complete lineage** - Git SHA, config hash, data hash, seeds recorded
- **Professional formatting** - Publication-quality reports with visualizations

### On-Premise First Design

- **No external dependencies** - Runs completely offline
- **File-based approach** - No databases or complex infrastructure needed
- **Full reproducibility** - Immutable run manifests for audit trails

### Simplicity as a Core Principle

- **Single command** - `glassalpha audit` handles everything
- **YAML configuration** - Policy-as-code for compliance requirements
- **Fast execution** - Under 3 seconds from model to PDF

## Supported Models

| Model Type          | Status     | Notes                           |
| ------------------- | ---------- | ------------------------------- |
| XGBoost             | Production | TreeSHAP integration optimized  |
| LightGBM            | Production | Native integration available    |
| Logistic Regression | Production | Full scikit-learn compatibility |

_Additional model types available through extension framework_

## Audit Report Contents

Audit reports include:

### 1. Model performance metrics

- Accuracy, precision, recall, F1, AUC-ROC
- Confusion matrices and performance curves
- Cross-validation results

### 2. TreeSHAP explanations

- Feature importance rankings
- Individual prediction explanations
- Waterfall plots for key decisions

### 3. Basic Fairness Analysis

- Protected attribute analysis
- Disparate impact calculations
- Group parity metrics

### 4. Reproducibility Manifest

- Complete configuration hash
- Dataset fingerprint
- Git commit SHA and timestamp
- All random seeds used

## Installation

```bash
# Clone and setup
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages

# Python 3.11 or 3.12 recommended
python3 --version   # should show 3.11.x or 3.12.x

# (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install in editable mode
python -m pip install --upgrade pip
pip install -e ".[dev]"

# Verify installation
glassalpha --help
```

## Contributing

We welcome contributions to enhance GlassAlpha's capabilities:

### Enhancement Areas

1. **Additional Models** - Neural networks, time series, custom integrations
2. **Advanced Explanations** - Counterfactuals, gradient methods, interactive visuals
3. **Extended Compliance** - Additional frameworks, custom templates, industry metrics
4. **Performance** - Large dataset optimization, parallel processing
5. **Documentation** - Examples, tutorials, best practices

### Example: Configuration Format

```yaml
# Working configuration structure
audit_profile: german_credit_default

data:
  path: data/german_credit_processed.csv
  target_column: credit_risk
  protected_attributes:
    - gender
    - age_group
    - foreign_worker

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

metrics:
  performance:
    metrics:
      - accuracy
      - precision
      - recall
      - f1
      - auc_roc
  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity

reproducibility:
  random_seed: 42
```

This configuration format supports deterministic, reproducible audits.

## Use Cases

- [Working Examples](examples/quick-start-audit.md) - Step-by-step tutorials for different use cases
- [Healthcare Bias Detection](examples/healthcare-bias-detection.md) - Medical AI compliance example
- [Configuration Comparison](examples/configuration-comparison.md) - Choosing the right audit approach

_Comprehensive examples with real datasets and regulatory interpretations_

## Documentation

- [Quick Start Guide](getting-started/quickstart.md) - Installation and first audit
- [Configuration Guide](getting-started/configuration.md) - YAML configuration reference
- [Regulatory Compliance](reference/compliance.md) - Compliance frameworks
- [Contributing Guidelines](reference/contributing.md) - Enhancement opportunities

## License & Support

- **License & Guidelines**: Apache 2.0 - See [LICENSE](https://github.com/GlassAlpha/glassalpha/blob/main/LICENSE) and [Trademark Guidelines](https://github.com/GlassAlpha/glassalpha/blob/main/TRADEMARK.md)
- **Issues**: [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)

---

_Built for teams who need reproducible, regulator-ready ML audit reports._
