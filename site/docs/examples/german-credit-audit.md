# Target Example: Financial Lending Audit - German Credit Dataset

!!! warning "Planned Feature - Not Yet Implemented"
    This example describes the target functionality. The audit generation system is currently under development.

!!! success "Development Priority"
    This is one of the two required example audits for initial release. When complete, it will produce publication-ready PDF reports.

## Overview

The German Credit dataset is our primary benchmark for developing the financial lending compliance audit. It contains 1,000 loan applications with 20 features including demographics, financial status, and loan details. This example will demonstrate how Glass Alpha generates regulator-ready PDF audits for credit scoring models.

## Why This Example Matters

- **Regulatory relevance**: Directly addresses ECOA/FCRA requirements
- **Real-world complexity**: Mixed numerical and categorical features
- **Fairness challenges**: Contains protected attributes requiring bias analysis
- **Industry standard**: Widely used compliance benchmark

## Dataset Details

- **Size**: 1,000 instances, 20 attributes
- **Target**: Binary (good/bad credit risk)
- **Protected Attributes**: Age, gender, foreign worker status
- **Use Case**: Financial lending compliance
- **Regulatory Context**: EU GDPR, Equal Credit Opportunity Act

## Planned Implementation

### Goal: Basic Audit

```bash
# Future CLI interface (not yet available)
glassalpha audit --data german.data --target credit_risk --out german_credit_audit.pdf
```

### Planned Configuration Design

```yaml
# Design specification for german_credit_audit.yaml
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

## Planned Audit Generation

Once implemented, the audit will be generated with:

```bash
# Future command (not yet available)
glassalpha audit --config german_credit_audit.yaml --out german_credit_complete_audit.pdf
```

## Target Audit Report Contents

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

## Design Goals for This Example

### Target Metrics (with seed `42`)
- **Accuracy**: ~72.5%
- **AUC-ROC**: ~0.78
- **Demographic Parity Ratio**: Target 0.80+ for all protected groups

### Determinism Requirements
- Byte-identical PDF outputs with same configuration
- Complete reproducibility manifest in every report
- All randomness controlled through explicit seeds

## Next Steps

- [Adult Income Audit](adult-income-audit.md) - Employment screening example
- [Configuration Reference](../getting-started/configuration.md) - Full YAML options
- [Regulatory Compliance](../compliance/overview.md) - Legal considerations
