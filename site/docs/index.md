# Glass Alpha

!!! info "Audit-First Focus"
    Glass Alpha delivers **deterministic, regulator-ready PDF audit reports** for tabular ML models. Built for teams who need reproducible, audit-ready model documentation.

## One Command, Regulator-Ready PDF Audit

Generate comprehensive audit reports for your ML models in under 60 seconds:

!!! success "Hello Audit - 60 Second Demo"
    ```bash
    # 1. Install Glass Alpha
    pip install glassalpha
    
    # 2. Generate your first audit
    glassalpha audit --config configs/german_credit.yaml --out my_audit.pdf
    
    # 3. Done! You have a deterministic, reproducible audit report
    ```
    **That's it!** Byte-identical PDF audits with complete lineage tracking.

## Why Glass Alpha Audit?

### 🏛️ Regulator-Ready
- **Deterministic outputs**: Identical PDFs on same seed/data/model
- **Complete lineage**: Git SHA, config hash, data hash, seeds recorded
- **Professional formatting**: Publication-quality reports with visualizations

### 🔒 On-Premise First  
- **Zero external calls**: Runs completely offline
- **File-based**: No databases or complex infrastructure
- **Reproducible**: Immutable run manifests for audit trails

### ⚡ CLI Simplicity
- **Single command**: `glassalpha audit` does everything
- **YAML configuration**: Policy-as-code for compliance requirements
- **60-second runtime**: From model to PDF in under a minute

## Supported Models

| Model Type | Status | Notes |
|-----------|--------|-------|
| XGBoost | ✅ Full support | TreeSHAP optimized |
| LightGBM | ✅ Full support | Native integration |
| Logistic Regression | ✅ Full support | scikit-learn compatible |

*Additional model types planned for future releases*

## What's in an Audit Report?

Every Glass Alpha audit includes:

1. **Model Performance Metrics**
   - Accuracy, precision, recall, F1, AUC-ROC
   - Confusion matrices and performance curves
   - Cross-validation results

2. **TreeSHAP Explanations** 
   - Feature importance rankings
   - Individual prediction explanations
   - Waterfall plots for key decisions

3. **Basic Fairness Analysis**
   - Protected attribute analysis
   - Disparate impact calculations
   - Group parity metrics

4. **Reproducibility Manifest**
   - Complete configuration hash
   - Dataset fingerprint
   - Git commit SHA and timestamp
   - All random seeds used

## Installation

```bash
# Standard installation
pip install glassalpha

# Verify installation
glassalpha --version
```

## Hello Audit Tutorial

### 1. Download Sample Data
```bash
# Get sample dataset (regulatory benchmark)
wget https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
```

### 2. Create Audit Configuration
```yaml
# sample_audit.yaml
model:
  type: xgboost
  target_column: default
  
data:
  train_path: german_credit_train.csv
  test_path: german_credit_test.csv
  
audit:
  protected_attributes:
    - gender
    - age_group
  confidence_level: 0.95
  
reproducibility:
  random_seed: 42
```

### 3. Generate Audit
```bash
glassalpha audit --config sample_audit.yaml --out german_credit_audit.pdf
```

**Result**: A professional PDF audit report with complete model documentation.

## Examples

- 📊 [Financial Lending Audit](examples/german-credit-audit.md) - Credit scoring compliance
- 💰 [Fair Hiring Audit](examples/adult-income-audit.md) - Employment screening analysis

*Additional examples planned for future releases*

## Next Steps

- 📚 [Quick Start Guide](getting-started/quickstart.md) - Step-by-step tutorial
- ⚙️ [Configuration Reference](getting-started/configuration.md) - YAML options
- 🏛️ [Regulatory Compliance](compliance/overview.md) - Legal considerations
- 👥 [Contributing](contributing.md) - Join the project

## Future Development

Future improvements may include additional features through ongoing development.

## License & Support

- **License**: Apache 2.0 - See [LICENSE](https://github.com/GlassAlpha/glassalpha/blob/main/LICENSE)
- **Issues**: [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)

---

*Built for teams who need reproducible, regulator-ready ML audit reports.*
