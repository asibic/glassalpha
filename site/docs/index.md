# Glass Alpha

!!! warning "Pre-Alpha Development Status"
    Glass Alpha is under active development. The features described here represent our current development goals and are not yet implemented. Current version: 0.1.0 (initial structure only)

!!! info "Project Vision"
    Glass Alpha will deliver **deterministic, regulator-ready PDF audit reports** for tabular ML models. We're building an open-source toolkit for teams who need reproducible, audit-ready model documentation.

## The Goal: One Command Audit Generation

Our target is simple, powerful audit generation:

!!! example "Planned Interface (Coming Soon)"
    ```bash
    # Future CLI interface (not yet available)
    glassalpha audit --config configs/german_credit.yaml --out my_audit.pdf
    
    # Will produce byte-identical PDF audits with complete lineage tracking
    ```
    **Goal:** Generate deterministic audit PDFs for XGBoost, LightGBM, and Logistic Regression models.

## Why We're Building Glass Alpha

### Designed for Regulatory Compliance
- **Planned: Deterministic outputs** - Identical PDFs on same seed/data/model
- **Planned: Complete lineage** - Git SHA, config hash, data hash, seeds will be recorded
- **Planned: Professional formatting** - Publication-quality reports with visualizations

### On-Premise First Design 
- **No external dependencies** - Will run completely offline
- **File-based approach** - No databases or complex infrastructure needed
- **Full reproducibility** - Immutable run manifests for audit trails

### Simplicity as a Core Principle
- **Single command goal** - `glassalpha audit` will handle everything
- **YAML configuration** - Policy-as-code for compliance requirements
- **Fast execution target** - Under 60 seconds from model to PDF

## Planned Model Support

| Model Type | Target Status | Notes |
|-----------|--------------|-------|
| XGBoost | Planned | TreeSHAP optimization planned |
| LightGBM | Planned | Native integration planned |
| Logistic Regression | Planned | scikit-learn compatibility planned |

*Additional model types may be considered based on community needs*

## Planned Audit Report Contents

Audit reports will include:

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

## Development Setup (Current)

```bash
# Clone the repository
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha

# Set up development environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e packages/[dev]

# Run tests to verify setup
pytest
```

## Contribute to Development

We're actively building the audit generation system. Here's how you can help:

### Priority Areas
1. **Core Audit Engine** - PDF generation pipeline
2. **TreeSHAP Integration** - Model explanation system  
3. **Fairness Metrics** - Bias detection implementation
4. **CLI Interface** - Command-line tool development
5. **Test Coverage** - Example datasets and validation

### Example: Planned Configuration Format

```yaml
# Future configuration structure (design phase)
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

This configuration format is being designed to support deterministic, reproducible audits.

## Target Use Cases

- [Planned: Financial Lending Audit](examples/german-credit-audit.md) - Credit scoring compliance example
- [Planned: Fair Hiring Audit](examples/adult-income-audit.md) - Employment screening analysis example

*These examples demonstrate our target capabilities*

## Documentation

- [Development Guide](getting-started/quickstart.md) - Set up your development environment
- [Design: Configuration Schema](getting-started/configuration.md) - Planned YAML structure
- [Vision: Regulatory Compliance](compliance/overview.md) - Target compliance frameworks  
- [Contributing](contributing.md) - Help build Glass Alpha

## Development Status

### Current Focus
- [ ] PDF generation pipeline
- [ ] TreeSHAP integration for XGBoost/LightGBM
- [ ] Basic fairness metrics
- [ ] Deterministic execution
- [ ] CLI interface
- [ ] German Credit & Adult Income examples

## License & Support

- **License**: Apache 2.0 - See [LICENSE](https://github.com/GlassAlpha/glassalpha/blob/main/LICENSE)
- **Issues**: [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)

---

*Built for teams who need reproducible, regulator-ready ML audit reports.*
