# Fair Hiring Audit - Adult Income Dataset

This example demonstrates bias detection and fairness analysis for employment screening using the Adult Income (Census) dataset.

!!! success "Phase 1 Priority"
    This is one of the two required example audits for Phase 1 completion. Must produce publication-ready PDF reports.

## Overview

The Adult Income dataset contains 48,842 census records with demographic and employment information. This example shows how to audit ML models used in hiring decisions for protected class bias.

## Dataset Details

- **Size**: 48,842 instances, 14 attributes  
- **Target**: Binary (income >$50K or ≤$50K)
- **Protected Attributes**: Race, gender, age, native country
- **Use Case**: Employment screening, hiring bias detection
- **Regulatory Context**: EEOC Guidelines, Title VII, GDPR

## Quick Start

```bash
# 1. Download the dataset
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test

# 2. Generate audit
glassalpha audit --data adult.data --target income --out adult_income_audit.pdf
```

## Full Configuration Example

```yaml
# adult_income_audit.yaml
model:
  type: lightgbm
  target_column: income
  params:
    num_leaves: 31
    learning_rate: 0.1
    feature_fraction: 0.9

data:
  train_path: adult_train.csv
  test_path: adult_test.csv
  categorical_features:
    - workclass
    - education
    - marital_status
    - occupation
    - relationship
    - race
    - gender
    - native_country

audit:
  protected_attributes:
    - race:
        column: race
        groups: ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
    - gender:
        column: gender
        groups: ["Male", "Female"]  
    - age_group:
        column: age
        groups: [0, 25, 35, 45, 55, 100]
    - country_origin:
        column: native_country
        groups: ["United-States", "Other"]
        
  fairness_metrics:
    - statistical_parity
    - equal_opportunity
    - equalized_odds
    - predictive_parity
    
  disparate_impact_threshold: 0.8  # 80% rule
  confidence_level: 0.95

explainability:
  shap_values: true
  feature_importance: true
  waterfall_plots: 20
  cohort_analysis:
    - protected_attribute: race
      top_features: 5
    - protected_attribute: gender
      top_features: 5

reproducibility:
  random_seed: 2024
  track_git: true
  track_data_hash: true
```

## Generate Complete Audit

```bash
glassalpha audit --config adult_income_audit.yaml --out adult_income_hiring_audit.pdf
```

## What's in the Audit Report

1. **Executive Summary**
   - Hiring bias risk assessment
   - EEOC compliance status
   - Recommended actions

2. **Dataset Demographics**
   - Protected class distributions
   - Outcome disparities by group
   - Intersectional analysis

3. **Model Performance by Group**
   - Accuracy across protected attributes
   - False positive/negative rates
   - Precision/recall by demographic

4. **Bias Detection Analysis**
   - Statistical parity ratios
   - Equal opportunity differences  
   - 80% rule compliance check
   - Disparate impact calculations

5. **Feature Impact Analysis**
   - SHAP values by protected group
   - Proxy feature detection
   - Correlation with protected attributes

6. **Regulatory Assessment**
   - EEOC Four-Fifths Rule compliance
   - Title VII risk indicators
   - Recommended mitigation strategies

## Regulatory Context

This audit addresses employment law requirements:

- **Title VII** (Civil Rights Act 1964)
- **EEOC Uniform Guidelines** (Four-Fifths Rule)
- **ADA** (Americans with Disabilities Act)  
- **ADEA** (Age Discrimination in Employment Act)
- **EU GDPR** (Automated Decision-Making)

## Expected Results

With seed `2024`, expect these approximate metrics:

**Overall Performance**:
- Accuracy: 85.2% ± 1.1%
- AUC-ROC: 0.90 ± 0.02

**Fairness Metrics**:
- Gender parity ratio: 0.83 (⚠️ Below 0.8 threshold)
- Race parity ratio: 0.72 (❌ Fails 80% rule)
- Age discrimination index: 0.91 (✅ Acceptable)

!!! danger "Bias Alert"
    Default model shows significant bias against protected groups. Audit report includes mitigation strategies.

## Mitigation Strategies

The audit report will include actionable recommendations:

1. **Feature Engineering**: Remove proxy variables
2. **Algorithmic Debiasing**: Fairness constraints during training
3. **Threshold Adjustment**: Group-specific cutoffs
4. **Process Changes**: Human-in-the-loop review for borderline cases

## Next Steps

- [German Credit Audit](german-credit-audit.md) - Financial lending example
- [Regulatory Compliance](../compliance/overview.md) - Compliance requirements  
- [Legal Compliance](../compliance/overview.md) - Regulatory requirements
