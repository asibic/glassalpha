# GlassAlpha

GlassAlpha makes **deterministic, regulator-ready PDF audit reports** for tabular ML models. It's an open-source ([Apache 2.0](reference/trust-deployment.md#licensing-dependencies)) toolkit for reproducible, audit-ready model documentation.

_Note: GlassAlpha is currently pre-alpha while I’m still making significant changes. I’ll cut the first official release and publish it on PyPI once things stabilize. The audits do run and the package works, so feel free to try it out, feedback welcome!_

## Quick Links

- [**Quick start guide**](getting-started/quickstart.md): Run an audit in 60 seconds.
- [**Using your own data**](getting-started/custom-data.md): Audit your models with custom CSV files.
- [**Preprocessing verification**](guides/preprocessing.md): Verify production preprocessing artifacts for compliance.
- [**Reason codes**](guides/reason-codes.md): Generate ECOA-compliant adverse action notices.
- [**Public datasets**](getting-started/data-sources.md): Test with 10+ curated benchmark datasets.
- [**Examples**](examples/german-credit-audit.md): Walkthrough ML audits on credit, healthcare bias and fraud detection.
- [**Trust & deployment**](reference/trust-deployment.md): Architecture, licensing, security, and compliance.

### Run first audit in 60 seconds

## Feature Overview

Quick reference for GlassAlpha capabilities and where to learn more:

| Feature | Guide | Reference | Compliance |
|---------|-------|-----------|------------|
| **Group Fairness** | [Configuration](getting-started/configuration.md#fairness-analysis-with-statistical-confidence) | [Fairness Metrics](reference/fairness-metrics.md) | [SR 11-7 §V](compliance/sr-11-7-mapping.md) |
| **Intersectional Fairness** | [Configuration](getting-started/configuration.md#fairness-analysis-with-statistical-confidence) | [Fairness Metrics](reference/fairness-metrics.md#intersectional-fairness-e51) | [SR 11-7 §V](compliance/sr-11-7-mapping.md) |
| **Individual Fairness** | [Configuration](getting-started/configuration.md#fairness-analysis-with-statistical-confidence) | [Fairness Metrics](reference/fairness-metrics.md#individual-fairness-e11) | [SR 11-7 §V](compliance/sr-11-7-mapping.md) |
| **Dataset Bias** | [Detection Guide](guides/dataset-bias.md) | - | [SR 11-7 §III.C.2](compliance/sr-11-7-mapping.md) |
| **Calibration** | [Configuration](getting-started/configuration.md#calibration-analysis-with-confidence-intervals) | [Calibration Reference](reference/calibration.md) | [SR 11-7 §III.B.2](compliance/sr-11-7-mapping.md) |
| **Robustness** | [Configuration](getting-started/configuration.md#robustness-testing-adversarial-perturbations) | [Robustness Reference](reference/robustness.md) | EU AI Act Art. 15 |
| **Shift Testing** | [Shift Testing Guide](guides/shift-testing.md) | - | [SR 11-7 §III.A.3](compliance/sr-11-7-mapping.md) |
| **Reason Codes** | [Reason Codes Guide](guides/reason-codes.md) | - | ECOA §701(d) |
| **Preprocessing** | [Preprocessing Guide](guides/preprocessing.md) | - | [SR 11-7 §III.C.1](compliance/sr-11-7-mapping.md) |


#### Clone and install

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages
pip install -e .
```

#### Generate an audit report (uses included German Credit example)

```bash
glassalpha audit --config configs/german_credit_simple.yaml --output audit.html
```

That's it. You now have a complete audit report with model performance, explanations, and fairness metrics.

**Note:** To generate PDF reports instead of HTML, install with `pip install 'glassalpha[docs]'`.

See [**more setup documentation here**](getting-started/quickstart.md).

## See It in Action (5-Minute Demo)

Want to see what you get? Generate a professional audit PDF in 5 minutes:

```bash
# 1. Clone and install (90 seconds)
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages && pip install -e .

# 2. Generate audit (30 seconds)
glassalpha audit --config configs/german_credit_simple.yaml --output audit.html

# 3. Done! Open your professional report
open audit.html  # macOS
# xdg-open audit.html  # Linux
# start audit.html  # Windows
```

**Result**: A professional audit report with:

- ✅ Model performance metrics (accuracy, precision, recall, F1, AUC)
- ✅ Fairness analysis (bias detection across demographic groups)
- ✅ Feature importance (coefficient-based explanations for linear models, SHAP for tree models)
- ✅ Individual explanations (why specific decisions were made)
- ✅ Reason codes (ECOA-compliant adverse action notices for credit decisions)
- ✅ Preprocessing verification (production artifact validation)
- ✅ Complete audit trail (reproducibility manifest with all seeds and hashes)

**Note:** Base install includes LogisticRegression with zero-dependency coefficient explanations. For tree models with SHAP, install with `pip install -e ".[explain]"`.

[See example output](examples/german-credit-audit.md)

## Why Choose GlassAlpha?

### How GlassAlpha Compares

| Feature                   | GlassAlpha                      | Fairlearn        | AIF360               | Commercial Tools     |
| ------------------------- | ------------------------------- | ---------------- | -------------------- | -------------------- |
| **Audit PDFs**            | ✅ Professional, byte-identical | ❌ No reports    | ❌ No reports        | ✅ $$$               |
| **Custom Data in 5 min**  | ✅ Yes                          | ⚠️ Complex setup | ⚠️ Complex setup     | ⚠️ Support needed    |
| **Built-in Datasets**     | ✅ 10+ ready to use             | ❌ None          | ⚠️ Few               | ✅ Limited           |
| **Model Support**         | ✅ XGBoost, LightGBM, sklearn   | ⚠️ sklearn only  | ⚠️ Limited           | ✅ Varies            |
| **Deterministic Results** | ✅ Byte-identical PDFs          | ⚠️ Partial       | ❌ No                | ⚠️ Varies            |
| **Offline/Air-gapped**    | ✅ 100% offline                 | ✅ Yes           | ✅ Yes               | ❌ Requires internet |
| **Cost**                  | ✅ Free (Apache 2.0)            | ✅ Free (MIT)    | ✅ Free (Apache 2.0) | 💰 $5K-$50K+         |
| **Regulatory Ready**      | ✅ Audit trails + manifests     | ❌ No trails     | ❌ No trails         | ✅ $$$               |
| **Learning Curve**        | ✅ 60-second start              | ⚠️ Steep         | ⚠️ Steep             | ⚠️ Training needed   |

**Bottom line**: GlassAlpha is the only OSS tool that combines professional audit PDFs, easy custom data support, and complete regulatory compliance—all in a 60-second setup.

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
- **Fast execution** - Under 60 seconds from model to PDF

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

### 2. Model explanations

- Feature importance rankings (coefficients for linear, SHAP for tree models)
- Individual prediction explanations
- Visual breakdown of key decisions

### 3. Basic fairness analysis

- Protected attribute analysis
- Disparate impact calculations
- Group parity metrics

### 4. Reason codes (ECOA compliance)

- Top-N negative feature contributions
- ECOA-compliant adverse action notices
- Protected attribute exclusion
- Deterministic ranking

### 5. Reproducibility manifest

- Complete configuration hash
- Dataset fingerprint
- Git commit SHA and timestamp
- All random seeds used

## Installation

Clone and setup

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages
```

Python 3.11, 3.12, or 3.13 supported

```bash
python3 --version   # should show 3.11.x, 3.12.x, or 3.13.x
```

(Recommended) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip and install in editable mode

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Verify installation

```bash
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

Working configuration structure:

```yaml
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
- [Using custom data](getting-started/custom-data.md) - Audit your own models
- [Preprocessing verification](guides/preprocessing.md) - Verify production preprocessing artifacts
- [Reason codes](guides/reason-codes.md) - ECOA-compliant adverse action notices
- [Public datasets](getting-started/data-sources.md) - 10+ curated benchmark datasets
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
