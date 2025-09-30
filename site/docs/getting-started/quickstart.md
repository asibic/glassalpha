# Quick Start Guide

Get up and running with GlassAlpha in less than 10 minutes. This guide will take you from installation to generating your first professional audit PDF.

## Prerequisites

- Python 3.11 or higher
- Git
- 2GB available disk space
- Command line access

## Step 1: Installation

### Clone and Install

Clone and setup:

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages
```

Python 3.11 or 3.12 recommended:

```bash
python3 --version   # should show 3.11.x or 3.12.x
```

Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip and install in editable mode:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Verify installation:

```bash
glassalpha --help
```

You should see the CLI help message with available commands.

## Step 2: Generate Your First Audit

GlassAlpha comes with a ready-to-use German Credit dataset example that demonstrates all core capabilities.

### Run the Audit Command

Generate audit PDF (takes ~3 seconds):

```bash
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --output my_first_audit.pdf
```

### What Happens

1. **Automatic Dataset Resolution**: Uses built-in German Credit dataset from registry
2. **Model Training**: Trains XGBoost classifier with optimal parameters
3. **Explanations**: Generates TreeSHAP feature importance analysis
4. **Fairness Analysis**: Computes bias metrics for protected attributes (gender, age)
5. **PDF Generation**: Creates professional audit report with visualizations

### Expected Output

```
Loading data and initializing components...
✓ Audit pipeline completed in 2.34s

📊 Audit Summary:
  ✅ Performance metrics: 6 computed
     ✅ accuracy: 73.5%
  ⚖️ Fairness metrics: 8/8 computed
     ✅ No bias detected
  🔍 Explanations: ✅ Global feature importance
     Most important: duration_months (+0.127)
  📋 Dataset: 1,000 samples, 21 features
  🔧 Components: 3 selected
     Model: xgboost

Generating PDF report: my_first_audit.pdf
✓ Saved plot to /tmp/plots/shap_importance.png
✓ Saved plot to /tmp/plots/performance_summary.png
✓ Saved plot to /tmp/plots/fairness_analysis.png

🎉 Audit Report Generated Successfully!
==================================================
📁 Output: /path/to/my_first_audit.pdf
📊 Size: 847,329 bytes (827.5 KB)
⏱️ Total time: 3.12s
   • Pipeline: 2.34s
   • PDF generation: 0.78s

The audit report is ready for review and regulatory submission.
```

## Step 3: Review Your Audit Report

Open `my_first_audit.pdf` to see your comprehensive audit report containing:

### Executive Summary

- Key findings and compliance status
- Model performance overview
- Bias detection results
- Regulatory assessment

### Model Performance Analysis

- Accuracy, precision, recall, F1 score, AUC-ROC
- Confusion matrix
- Performance visualizations

### SHAP Explanations

- Global feature importance rankings
- Individual prediction explanations
- Waterfall plots showing decision factors

### Fairness Analysis

- Demographic parity assessment
- Equal opportunity analysis
- Bias detection across protected attributes
- Statistical significance testing

### Reproducibility Manifest

- Complete audit trail with timestamps
- Dataset fingerprints and model parameters
- Random seeds and component versions
- Git commit information

## Step 4: Understanding the Configuration

The `configs/german_credit_simple.yaml` file contains all audit settings:

Audit profile determines component selection:

```yaml
audit_profile: german_credit_default
```

Reproducibility settings:

```yaml
reproducibility:
  random_seed: 42
```

Data configuration:

```yaml
data:
  dataset: german_credit # Uses built-in German Credit dataset
  fetch: if_missing # Automatically download if needed
  target_column: credit_risk
  protected_attributes:
    - gender
    - age_group
    - foreign_worker
```

Model configuration:

```yaml
model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100
    max_depth: 5
```

Explainer selection:

```yaml
explainers:
  strategy: first_compatible
  priority:
    - treeshap # Primary choice for tree models
    - kernelshap # Fallback for any model type
```

Metrics to compute:

```yaml
metrics:
  performance:
    metrics: [accuracy, precision, recall, f1, auc_roc]
  fairness:
    metrics: [demographic_parity, equal_opportunity]
```

## Next Steps

### Try Advanced Features

Enable strict mode for regulatory compliance:

```bash
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --output regulatory_audit.pdf \
  --strict
```

Use a different model (edit config file: model.type: lightgbm):

```bash
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --output lightgbm_audit.pdf
```

### Explore More Options

See all available CLI options:

```bash
glassalpha audit --help
```

List available components:

```bash
glassalpha list
```

Validate configuration without running audit:

```bash
glassalpha validate --config configs/german_credit_simple.yaml
```

Manage datasets:

```bash
glassalpha datasets list        # See available datasets
glassalpha datasets info german_credit  # Show dataset details
glassalpha datasets cache-dir   # Show where datasets are cached
```

### Work with Your Own Data

1. **Prepare your data**: CSV format with target column and features
2. **Create configuration**: Copy and modify `german_credit_simple.yaml`
3. **Run audit**: Use your configuration file

See the [Configuration Guide](configuration.md) for detailed customization options.

## Common Use Cases

### Financial Services Compliance

- Credit scoring model validation
- Fair lending assessments
- Regulatory reporting (ECOA, FCRA)
- Model risk management

### HR and Employment

- Hiring algorithm audits
- Promotion decision analysis
- Salary equity assessments
- EEO compliance verification

### Healthcare and Insurance

- Risk assessment model validation
- Treatment recommendation audits
- Coverage decision analysis
- Health equity evaluations

## Getting Help

- **Documentation**: [Complete Guide](../index.md)
- **Configuration Reference**: [Configuration Guide](configuration.md)
- **Examples**:
  - [5-minute Quick Start](../examples/quick-start-audit.md)
  - [German Credit Deep Dive](../examples/german-credit-audit.md)
  - [Healthcare Bias Detection](../examples/healthcare-bias-detection.md)
  - [Configuration Comparison](../examples/configuration-comparison.md)
- **Issues**: [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)

## Summary

You now have GlassAlpha installed and have generated your first audit report. The system provides:

- **Production-ready audit generation** in seconds
- **Professional PDF reports** suitable for regulatory review
- **Comprehensive analysis** covering performance, fairness, and explainability
- **Full reproducibility** with complete audit trails
- **Flexible configuration** for different use cases and models

GlassAlpha transforms complex ML audit requirements into a simple, reliable workflow that meets the highest professional and regulatory standards.
