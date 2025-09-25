# Quick Start: Your First Audit in 5 Minutes

This example shows the fastest way to generate an ML audit report with GlassAlpha using the built-in German Credit dataset and minimal configuration.

## Overview

Perfect for:
- **First-time users** learning GlassAlpha basics
- **Quick demonstrations** of audit capabilities
- **Testing installations** and verifying functionality
- **Understanding core concepts** before advanced usage

## Prerequisites

- GlassAlpha installed ([Installation Guide](../getting-started/installation.md))
- 5 minutes of time
- No additional data required (uses built-in dataset)

## Step 1: Verify Installation

```bash
# Confirm GlassAlpha is working
glassalpha --version
glassalpha list
```

Expected output shows available models, explainers, and metrics.

## Step 2: Run Your First Audit

Use the minimal configuration that comes with GlassAlpha:

```bash
# Generate audit with minimal configuration
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --output my_first_audit.pdf
```

**What happens:**
- **Data loading**: Automatically downloads German Credit dataset
- **Model training**: Trains XGBoost classifier with default parameters
- **Explanation generation**: Creates SHAP explanations for model decisions
- **Metrics computation**: Calculates performance and fairness metrics
- **Report creation**: Generates professional PDF report

**Execution time**: 10-30 seconds on typical hardware.

## Step 3: Review Your Report

Open `my_first_audit.pdf` to see:

1. **Executive Summary**
   - Model performance overview
   - Key fairness findings
   - Regulatory compliance status

2. **Model Performance**
   - Accuracy: ~77% (typical for German Credit dataset)
   - Precision, recall, F1 scores
   - ROC curve and confusion matrix

3. **SHAP Explanations**
   - Feature importance rankings
   - Sample individual explanations
   - Waterfall plots showing decision factors

4. **Fairness Analysis**
   - Demographic parity across gender and age groups
   - Statistical significance testing
   - Bias detection results

5. **Reproducibility**
   - Complete audit manifest
   - Configuration hash and random seeds
   - Data integrity verification

## Understanding the Configuration

The minimal configuration (`configs/german_credit_simple.yaml`) contains:

```yaml
# Minimal audit configuration
audit_profile: german_credit_default
reproducibility:
  random_seed: 42

# Model configuration
model:
  type: xgboost
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

# Data configuration (uses built-in dataset)
data:
  target_column: credit_risk
  protected_attributes:
    - gender
    - age_group

# Explainer configuration
explainers:
  priority: [treeshap, kernelshap]

# Metrics to compute
metrics:
  performance: [accuracy, precision, recall, f1, auc_roc]
  fairness: [demographic_parity, equal_opportunity]
```

**Key concepts:**
- **audit_profile**: Determines which components are used
- **random_seed**: Ensures reproducible results
- **protected_attributes**: Enable fairness analysis
- **priority**: Determines explainer selection order

## What You've Accomplished

In 5 minutes, you've:
- ✅ **Generated a professional audit report** suitable for compliance review
- ✅ **Performed bias detection** across demographic groups
- ✅ **Created model explanations** with SHAP
- ✅ **Established reproducibility** with complete audit trails
- ✅ **Learned core concepts** for advanced usage

## Common First-Time Questions

**Q: Why did it use XGBoost?**
A: The configuration specifies `model.type: xgboost`. GlassAlpha supports XGBoost, LightGBM, and Logistic Regression.

**Q: Can I use my own data?**
A: Yes! Change `data.path` to your CSV file and update `target_column` and `feature_columns`. See [Configuration Guide](../getting-started/configuration.md).

**Q: What if I get different results?**
A: Results should be identical with the same `random_seed`. Different results suggest configuration or data changes.

**Q: Is this suitable for production?**
A: This minimal example is for learning. Production usage requires additional validation, testing, and security considerations. See [Production Deployment Guide](../deployment.md).

## Next Steps

### Try Different Models

```bash
# Use LightGBM instead of XGBoost
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --output lightgbm_audit.pdf \
  --override '{"model": {"type": "lightgbm"}}'
```

### Add Strict Mode

```bash
# Enable regulatory compliance mode
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --output strict_audit.pdf \
  --strict
```

### Explore Advanced Features

1. **Custom Configuration** - [Configuration Guide](../getting-started/configuration.md)
2. **Detailed Example** - [German Credit Deep Dive](german-credit-audit.md)
3. **Production Setup** - [Deployment Guide](../deployment.md)
4. **API Usage** - [API Reference](../reference/api.md)

### Use Your Own Data

```yaml
# Create custom configuration
audit_profile: tabular_compliance
reproducibility:
  random_seed: 42

data:
  path: your_data.csv
  target_column: your_target
  protected_attributes:
    - your_sensitive_attribute

model:
  type: xgboost

# Save as my_config.yaml and run:
# glassalpha audit --config my_config.yaml --output my_audit.pdf
```

## Troubleshooting

**Installation issues?** → [Installation Guide](../getting-started/installation.md)
**Configuration errors?** → [Configuration Guide](../getting-started/configuration.md)
**Command problems?** → [CLI Reference](../reference/cli.md)
**General questions?** → [FAQ](../faq.md)

## Summary

You've successfully generated your first ML audit report with GlassAlpha! The minimal configuration demonstrates core capabilities:

- **Model training and evaluation**
- **Bias detection and fairness analysis**
- **Explainable AI with SHAP**
- **Professional report generation**
- **Complete reproducibility**

This foundation prepares you for advanced usage scenarios and production deployment in regulated industries.

Ready for more? Explore the [comprehensive German Credit example](german-credit-audit.md) for detailed regulatory analysis and interpretation.
