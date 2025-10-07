# Quick start guide

## The 5-minute version

Get your first professional audit PDF in 5 minutes:

### Using QuickStart Generator (Easiest)

```bash
# 1. Install (30 seconds)
pip install glassalpha

# 2. Generate project (interactive wizard)
glassalpha quickstart

# 3. Run audit (20 seconds)
cd my-audit-project && python run_audit.py

# 4. Done! Open your professional report
open reports/audit_report.html  # macOS
```

### Using Repository Example

```bash
# 1. Clone and install (90 seconds)
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages && pip install -e .

# 2. Generate audit (30 seconds)
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf

# 3. Done! Open your professional PDF
open audit.pdf  # macOS
# xdg-open audit.pdf  # Linux
# start audit.pdf  # Windows
```

**What you get**: A 10-page professional audit PDF with:

- ✅ Model performance metrics (accuracy, precision, recall, F1, AUC)
- ✅ Fairness analysis (bias detection across demographic groups)
- ✅ Feature importance (coefficient-based explanations showing what drives predictions)
- ✅ Individual explanations (why specific decisions were made)
- ✅ Preprocessing verification (optional, for production artifact validation)
- ✅ Complete audit trail (reproducibility manifest with all seeds and hashes)

**Note:** This quickstart uses LogisticRegression with coefficient-based explanations (zero dependencies). For tree-based models with SHAP explanations, install with `pip install -e ".[explain]"`.

**Next steps**:

- [Use your own data](custom-data.md)
- [Verify preprocessing artifacts](../guides/preprocessing.md) (for production audits)
- [Try other datasets](data-sources.md)
- [Understand the configuration](configuration.md)

## The 10-minute version

Get up and running with GlassAlpha in less than 10 minutes. This guide will take you from installation to generating your first professional audit PDF.

## Prerequisites

- Python 3.11 or higher
- Git (optional, only needed for manual setup)
- 2GB available disk space
- Command line access

## Step 1: Installation

Choose between the quickstart generator (recommended) or manual setup:

### Option A: QuickStart Generator (Recommended)

The fastest way to get started. Creates a complete audit project in <60 seconds:

```bash
# Install GlassAlpha (if not already installed)
pip install glassalpha

# Generate a ready-to-run audit project
glassalpha quickstart
```

The interactive wizard will ask you:

1. **Project name**: Where to create the project directory
2. **Dataset**: German Credit (1K samples) or Adult Income (48K samples)
3. **Model type**: XGBoost (recommended), LightGBM (fast), or Logistic Regression (simple)

**What you get:**

- Complete project directory structure (data/, models/, reports/, configs/)
- Pre-configured audit configuration file (`audit_config.yaml`)
- Example run script (`run_audit.py`) demonstrating programmatic API
- Project README with next steps and advanced usage
- `.gitignore` tailored for GlassAlpha projects

**Run your first audit:**

```bash
cd my-audit-project
python run_audit.py  # Generates audit report in <5 seconds
```

**Non-interactive mode** (for scripts/CI):

```bash
glassalpha quickstart \
  --dataset german_credit \
  --model xgboost \
  --output my-project \
  --no-interactive
```

**Skip to:** [Step 3: Review your audit report](#step-3-review-your-audit-report) once your report is generated.

### Option B: Manual Setup

If you prefer manual setup or want to work from the repository:

### Clone and install

Clone and setup:

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages
```

Python 3.11, 3.12, or 3.13 supported:

```bash
python3 --version   # should show 3.11.x, 3.12.x, or 3.13.x
```

Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install GlassAlpha:

```bash
python -m pip install --upgrade pip

# Option 1: Base install (LogisticRegression only, recommended for getting started)
pip install -e .

# Option 2: With advanced ML libraries (if you need XGBoost/LightGBM)
pip install -e ".[explain]"      # SHAP + XGBoost + LightGBM
pip install -e ".[all]"          # All features

# Option 3: Development install (includes testing tools)
pip install -e ".[dev]"
```

Verify installation:

```bash
glassalpha --help

# Check what models are available
glassalpha models
```

You should see the CLI help message with available commands.

## Step 2: Generate your first audit

GlassAlpha comes with a ready-to-use German Credit dataset example that demonstrates all core capabilities.

### Run the audit command

Generate audit report (takes ~3 seconds):

```bash
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --output my_first_audit.html
```

**Note:** The simple configuration uses `logistic_regression` model (always available). For advanced models like XGBoost or LightGBM, install with `pip install 'glassalpha[explain]'`.

### What happens

1. **Automatic Dataset Resolution**: Uses built-in German Credit dataset from registry
2. **Model Training**: Trains LogisticRegression classifier (baseline model)
3. **Explanations**: Generates coefficient-based feature importance
4. **Fairness Analysis**: Computes bias metrics for protected attributes (gender, age)
5. **Report Generation**: Creates professional HTML audit report with visualizations

### Expected output

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

## Step 3: Review your audit report

Open `my_first_audit.pdf` to see your comprehensive audit report containing:

### Executive summary

- Key findings and compliance status
- Model performance overview
- Bias detection results
- Regulatory assessment

### Model performance analysis

- Accuracy, precision, recall, F1 score, AUC-ROC
- Confusion matrix
- Performance visualizations

### Model explanations

- Global feature importance rankings (coefficient-based for linear models, SHAP for tree models)
- Individual prediction explanations
- Clear visualization of what drives predictions

### Fairness analysis

- Demographic parity assessment
- Equal opportunity analysis
- Bias detection across protected attributes
- Statistical significance testing

### Reproducibility manifest

- Complete audit trail with timestamps
- Dataset fingerprints and model parameters
- Random seeds and component versions
- Git commit information

## Step 4: Understanding the configuration

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
  type: logistic_regression # Baseline model (always available)
  params:
    random_state: 42
    max_iter: 1000
# For advanced models (requires pip install 'glassalpha[explain]'):
# type: xgboost
# params:
#   objective: binary:logistic
#   n_estimators: 100
#   max_depth: 5
```

Explainer selection:

```yaml
explainers:
  strategy: first_compatible
  priority:
    - coefficients # Zero-dependency explainer for linear models
  config:
    coefficients:
      normalize: true
# For tree models with SHAP (requires pip install 'glassalpha[explain]'):
# priority:
#   - treeshap # Best for XGBoost, LightGBM, RandomForest
#   - kernelshap # Model-agnostic SHAP fallback
```

Metrics to compute:

```yaml
metrics:
  performance:
    metrics: [accuracy, precision, recall, f1, auc_roc]
  fairness:
    metrics: [demographic_parity, equal_opportunity]
```

## Next steps

### Try advanced features

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

### Explore more options

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

### Work with your own data

Ready to audit your own models? We've made it easy:

1. **Follow the tutorial**: See [Using Custom Data](custom-data.md) for step-by-step guidance
2. **Use our template**: The fully-commented configuration template is in `packages/configs/custom_template.yaml`
3. **Try public datasets**: Browse [freely available data sources](data-sources.md) for testing

**Need to choose a model?** The [Model Selection Guide](../reference/model-selection.md) helps you pick between LogisticRegression, XGBoost, and LightGBM with performance benchmarks.

For detailed customization options, see the [Configuration Guide](configuration.md).

## Common use cases

### Financial services compliance

- Credit scoring model validation
- Fair lending assessments
- Regulatory reporting (ECOA, FCRA)
- Model risk management

### HR and employment

- Hiring algorithm audits
- Promotion decision analysis
- Salary equity assessments
- EEO compliance verification

### Healthcare and insurance

- Risk assessment model validation
- Treatment recommendation audits
- Coverage decision analysis
- Health equity evaluations

## Getting help

- **Documentation**: [Complete Guide](../index.md)
- **Guides**:
  - [Using Custom Data](custom-data.md) - Audit your own models
  - [Preprocessing Verification](../guides/preprocessing.md) - Verify production artifacts
  - [Freely Available Data Sources](data-sources.md) - Public datasets for testing
  - [Configuration Reference](configuration.md) - All configuration options
  - [Model Selection Guide](../reference/model-selection.md) - Choose the right model
  - [Explainer Deep Dive](../reference/explainers.md) - Understanding explanations
- **Examples**:
  - [German Credit Deep Dive](../examples/german-credit-audit.md) - Complete audit walkthrough
  - [Healthcare Bias Detection](../examples/healthcare-bias-detection.md) - Medical AI compliance example
  - [Fraud Detection Audit](../examples/fraud-detection-audit.md) - Financial services example
- **Support**:
  - [FAQ](../reference/faq.md) - Frequently asked questions
  - [Troubleshooting Guide](../reference/troubleshooting.md) - Common issues and solutions
  - [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues) - Report bugs or request features

## Summary

You now have GlassAlpha installed and have generated your first audit report. The system provides:

- **Production-ready audit generation** in seconds
- **Professional PDF reports** suitable for regulatory review
- **Comprehensive analysis** covering performance, fairness, and explainability
- **Full reproducibility** with complete audit trails
- **Flexible configuration** for different use cases and models

GlassAlpha transforms complex ML audit requirements into a simple, reliable workflow that meets the highest professional and regulatory standards.

## Next Steps

### Dive Deeper into Core Concepts

- **[Understanding Fairness Metrics](../reference/fairness-metrics.md)** - Learn about group, intersectional, and individual fairness with statistical confidence intervals
- **[Detecting Dataset Bias](../guides/dataset-bias.md)** - Catch proxy correlations, drift, and sampling bias before model training
- **[Calibration Analysis](../reference/calibration.md)** - Ensure predicted probabilities match observed outcomes

### Advanced Features

- **[Testing Demographic Shifts](../guides/shift-testing.md)** - Validate model robustness under population changes with CI/CD gates
- **[Robustness Testing](../reference/robustness.md)** - Test stability under adversarial perturbations (ε-perturbation sweeps)
- **[Generating Reason Codes](../guides/reason-codes.md)** - ECOA-compliant adverse action notices

### Regulatory Compliance

- **[SR 11-7 Compliance Mapping](../compliance/sr-11-7-mapping.md)** - Federal Reserve guidance for banking models (complete clause-to-artifact mapping)
- **[Trust & Deployment](../reference/trust-deployment.md)** - Reproducibility, determinism, and evidence pack export
