# GlassAlpha

GlassAlpha makes **deterministic, regulator-ready PDF audit reports** for tabular ML models. It's an open-source ([Apache 2.0](reference/trust-deployment/#licensing-dependencies)) toolkit for reproducible, audit-ready model documentation.

## Quick Links

- [**Quick start guide**](getting-started/quickstart.md): Run an audit in 60 seconds.
- [**Examples**](examples/german-credit-audit.md): Walkthrough ML audits on credit, healthcare bias and fraud detection.
- [**Trust & deployment**](examples/german-credit-audit.md): Misc. info on architecture, licensing, security, compliance, and deployment.

### Run first audit in 60 seconds

#### Clone and install

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages
pip install -e .
```

#### Generate an audit PDF (uses included German Credit example)

```bash
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf
```

That's it. You now have a complete audit report with model performance, SHAP explanations, and fairness metrics.

See [**more setup documentation here**](getting-started/quickstart.md).

## Why GlassAlpha?

### Designed for regulatory compliance

- **Deterministic outputs** - Identical PDFs on same seed/data/model
- **Complete lineage** - Git SHA, config hash, data hash, seeds recorded
- **Professional formatting** - Publication-quality reports with visualizations

### On-premise first design

- **No external dependencies** - Runs completely offline
- **File-based approach** - No databases or complex infrastructure needed
- **Full reproducibility** - Immutable run manifests for audit trails

### Simplicity as a core principle

- **Single command** - `glassalpha audit` handles everything
- **YAML configuration** - Policy-as-code for compliance requirements
- **Fast execution** - Under 3 seconds from model to PDF

## Supported models

| Model Type          | Status     | Notes                           |
| ------------------- | ---------- | ------------------------------- |
| XGBoost             | Production | TreeSHAP integration optimized  |
| LightGBM            | Production | Native integration available    |
| Logistic Regression | Production | Full scikit-learn compatibility |

_Additional model types available through extension framework._

## Audit report contents

Audit reports include:

### 1. Model performance metrics

- Accuracy, precision, recall, F1, AUC-ROC
- Confusion matrices and performance curves
- Cross-validation results

### 2. TreeSHAP explanations

- Feature importance rankings
- Individual prediction explanations
- Waterfall plots for key decisions

### 3. Basic fairness analysis

- Protected attribute analysis
- Disparate impact calculations
- Group parity metrics

### 4. Reproducibility manifest

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

### Enhancement areas

1. **Additional models** - Neural networks, time series, custom integrations
2. **Advanced explanations** - Counterfactuals, gradient methods, interactive visuals
3. **Extended compliance** - Additional frameworks, custom templates, industry metrics
4. **Performance** - Large dataset optimization, parallel processing
5. **Documentation** - Examples, tutorials, best practices

### Example: configuration format

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

## Use cases

- [German credit audit](examples/german-credit-audit.md) - Complete audit walkthrough with German Credit dataset
- [Healthcare bias detection](examples/healthcare-bias-detection.md) - Medical AI compliance example
- [Fraud detection audit](examples/fraud-detection-audit.md) - Financial services compliance example

_Comprehensive examples with real datasets and regulatory interpretations._

## Documentation

- [Quick start guide](getting-started/quickstart.md) - Installation and first audit
- [Configuration guide](getting-started/configuration.md) - YAML configuration reference
- [Trust & deployment](reference/trust-deployment.md) - Architecture, licensing, security, and compliance
- [Contribution guidelines](reference/contributing.md) - Enhancement opportunities

## License & trademark

- **License:** Apache 2.0 - See [LICENSE](https://github.com/GlassAlpha/glassalpha/blob/main/LICENSE)
- **Trademark:** While GlassAlpha's code is open source, the brand is not. We respectfully request that our name and logo not be used in confusing or misleading ways. See [TRADEMARK](reference/TRADEMARK.md).

## Support

- **Issues**: [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)

---

_Built for teams who need reproducible, regulator-ready ML audit reports._
