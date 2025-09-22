# Financial Lending Audit - German Credit Dataset

This example demonstrates a complete audit workflow for financial lending compliance using the German Credit dataset.

!!! success "Phase 1 Priority"
    This is one of the two required example audits for Phase 1 completion. Must produce publication-ready PDF reports.

## Overview

The German Credit dataset is a regulatory compliance benchmark containing 1,000 loan applications with 20 features including demographics, financial status, and loan details. This example shows how to generate a regulator-ready PDF audit for credit scoring models.

## Dataset Details

- **Size**: 1,000 instances, 20 attributes
- **Target**: Binary (good/bad credit risk)
- **Protected Attributes**: Age, gender, foreign worker status
- **Use Case**: Financial lending compliance
- **Regulatory Context**: EU GDPR, Equal Credit Opportunity Act

## Quick Start

```bash
# 1. Download the dataset
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data

# 2. Generate audit with basic config
glassalpha audit --data german.data --target credit_risk --out german_credit_audit.pdf
```

## Full Configuration Example

```yaml
# german_credit_audit.yaml
model:
  type: xgboost
  target_column: credit_risk
  params:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100

data:
  train_path: german_credit_train.csv
  test_path: german_credit_test.csv
  feature_columns:
    - duration_months
    - credit_amount
    - installment_rate
    - present_residence_since
    - age
    - number_existing_credits
    - number_people_liable

audit:
  protected_attributes:
    - age_group:  # Age < 25, 25-40, > 40
        column: age
        groups: [0, 25, 40, 100]
    - gender:
        column: personal_status_sex
        groups: ["male", "female"]
    - foreign_worker:
        column: foreign_worker
        groups: [0, 1]
        
  fairness_metrics:
    - demographic_parity
    - equalized_odds  
    - equal_opportunity
    
  confidence_level: 0.95
  
explainability:
  shap_values: true
  feature_importance: true
  waterfall_plots: 10  # Top 10 most important predictions
  
reproducibility:
  random_seed: 42
  track_git: true
  track_data_hash: true
```

## Generate Complete Audit

```bash
glassalpha audit --config german_credit_audit.yaml --out german_credit_complete_audit.pdf
```

## What's in the Audit Report

1. **Executive Summary**
   - Model performance overview
   - Key fairness findings
   - Regulatory compliance status

2. **Data Analysis** 
   - Dataset statistics and distributions
   - Protected attribute analysis
   - Feature correlation matrix

3. **Model Performance**
   - Confusion matrix and classification metrics
   - ROC and Precision-Recall curves
   - Cross-validation results

4. **TreeSHAP Explanations**
   - Global feature importance rankings
   - Individual prediction explanations
   - Waterfall plots for key decisions

5. **Fairness Analysis**
   - Demographic parity calculations
   - Equalized odds analysis
   - Disparate impact metrics

6. **Reproducibility Manifest**
   - Complete configuration hash: `sha256:a1b2c3...`
   - Dataset fingerprint: `sha256:d4e5f6...`
   - Git commit: `7890abc...`
   - Random seed: `42`
   - Generation timestamp

## Regulatory Context

This audit addresses common compliance requirements:

- **Fair Credit Reporting Act (FCRA)**
- **Equal Credit Opportunity Act (ECOA)** 
- **EU GDPR Article 22** (Automated Decision-Making)
- **Basel III** (Model Risk Management)

## Expected Results

With seed `42`, expect these approximate performance metrics:
- **Accuracy**: 72.5% ± 2.1%
- **AUC-ROC**: 0.78 ± 0.03
- **Demographic Parity Ratio**: 0.85 (age), 0.92 (gender)

!!! warning "Deterministic Output"
    Results should be byte-identical across runs with the same seed and data. If not, file a bug report.

## Next Steps

- [Adult Income Audit](adult-income-audit.md) - Employment screening example
- [Configuration Reference](../getting-started/configuration.md) - Full YAML options
- [Regulatory Compliance](../compliance/overview.md) - Legal considerations
