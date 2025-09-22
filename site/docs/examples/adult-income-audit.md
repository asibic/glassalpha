# Target Example: Fair Hiring Audit - Adult Income Dataset

!!! warning "Planned Feature - Not Yet Implemented"
    This example describes the target functionality. The audit generation system is currently under development.

!!! success "Development Priority"
    This is one of the two required example audits for initial release. When complete, it will produce publication-ready PDF reports demonstrating bias detection.

## Overview

The Adult Income dataset contains 48,842 census records with demographic and employment information. This example will demonstrate how Glass Alpha audits ML models for protected class bias in hiring decisions.

## Why This Example Matters

- **EEOC compliance**: Directly addresses Four-Fifths Rule requirements
- **Scale testing**: Large dataset tests performance at scale
- **Multiple protected classes**: Race, gender, age intersectionality
- **Real bias patterns**: Dataset exhibits known fairness challenges

## Dataset Details

- **Size**: 48,842 instances, 14 attributes  
- **Target**: Binary (income >$50K or ≤$50K)
- **Protected Attributes**: Race, gender, age, native country
- **Use Case**: Employment screening, hiring bias detection
- **Regulatory Context**: EEOC Guidelines, Title VII, GDPR

## Planned Implementation

### Goal: Fairness Audit

```bash
# Future CLI interface (not yet available)
glassalpha audit --data adult.data --target income --out adult_income_audit.pdf
```

### Planned Configuration Design

```yaml
# Design specification for adult_income_audit.yaml
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

## Planned Audit Generation

Once implemented, the audit will be generated with:

```bash
# Future command (not yet available)
glassalpha audit --config adult_income_audit.yaml --out adult_income_hiring_audit.pdf
```

## Target Audit Report Contents

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

## Design Goals for This Example

### Target Performance Metrics (with seed `2024`)

**Model Performance Goals**:
- Accuracy: ~85%
- AUC-ROC: ~0.90

**Fairness Testing Goals**:
- Demonstrate detection of bias violations
- Show gender parity ratio below 80% threshold
- Illustrate race-based disparate impact
- Provide clear visualization of fairness gaps

### Key Demonstration Points

This example will showcase Glass Alpha's ability to:
1. **Detect bias** - Identify when models fail the Four-Fifths Rule
2. **Explain disparities** - Show which features drive unfair outcomes  
3. **Suggest mitigations** - Provide actionable recommendations

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
