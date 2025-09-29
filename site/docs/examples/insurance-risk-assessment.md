# Insurance Risk Assessment Audit Tutorial

Complete walkthrough of performing a comprehensive ML audit on an insurance risk assessment model using GlassAlpha. This tutorial demonstrates how to audit automated underwriting systems for compliance with insurance regulations and fair lending laws.

## Overview

Insurance companies use machine learning models to assess risk and set premiums for auto insurance policies. These models must comply with:

- **Rate Regulation**: Premiums must be based on actuarial risk factors, not discriminatory criteria
- **Fair Lending Laws**: Equal treatment across protected demographic groups
- **Transparency Requirements**: Ability to explain automated decisions to consumers
- **Actuarial Standards**: Sound statistical practices and documentation

### What You'll Learn

- How to configure GlassAlpha for insurance risk models
- Interpreting performance metrics for insurance underwriting
- Understanding SHAP explanations for premium decisions
- Identifying bias in risk assessment across demographic groups
- Generating regulatory-ready audit reports for insurance compliance

### Regulatory Context

Insurance risk assessment models must comply with:

- **Insurance Rate Regulation**: State insurance departments require actuarial justification
- **Unfair Discrimination Laws**: Prohibit discrimination based on protected characteristics
- **Consumer Protection Laws**: Right to explanation for automated decisions
- **Data Privacy Regulations**: GDPR, CCPA requirements for personal data use

## Prerequisites

- GlassAlpha installed ([Installation Guide](../getting-started/installation.md))
- Basic understanding of insurance underwriting and risk assessment
- Familiarity with bias and fairness concepts in algorithmic decision-making

## Step 1: Understanding the Insurance Dataset

### Dataset Characteristics

The insurance risk dataset contains:

- **10,000 synthetic auto insurance policies**
- **17 features** covering demographics, vehicle information, and driving history
- **Binary target**: Filed insurance claim (1) vs No claim (0)
- **Protected attributes**: Gender, age groups, socioeconomic indicators

### Key Features

**Demographic & Personal Information:**

- `age` - Policyholder age (18-80 years)
- `gender` - Policyholder gender (protected attribute)
- `age_group` - Age categorization (protected attribute)
- `marital_status` - Marital status indicator
- `occupation_risk` - Occupation-based risk category

**Vehicle & Coverage Information:**

- `vehicle_value` - Vehicle replacement cost ($5,000-$100,000)
- `annual_mileage` - Miles driven per year (0-50,000)
- `vehicle_age` - Vehicle age in years (0-15)
- `policy_type` - Coverage level (Basic/Standard/Premium)
- `deductible_amount` - Policy deductible ($250-$2,500)
- `coverage_limit` - Maximum coverage amount ($50,000-$500,000)

**Risk Factors:**

- `years_insured` - Insurance history (0-50 years)
- `previous_claims` - Number of prior claims (0-5)
- `credit_score` - Credit rating (300-850)
- `location_risk_score` - Geographic risk factor (0-100)
- `safe_driver_discount` - Safe driving program participation
- `discount_eligibility` - Eligibility for discounts

## Step 2: Configuration Setup

Create a configuration file for the insurance risk audit:

```yaml
# insurance_risk_audit.yaml
audit_profile: insurance_risk_assessment

# Reproducibility for regulatory compliance
reproducibility:
  random_seed: 42

# Data configuration
data:
  path: ~/.glassalpha/data/insurance_risk.csv
  target_column: claim_outcome
  protected_attributes:
    - gender
    - age_group

# LightGBM model for insurance risk assessment
model:
  type: lightgbm
  params:
    objective: binary
    metric: binary_logloss
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8

# Explanation configuration
explainers:
  strategy: first_compatible
  priority:
    - treeshap # LightGBM supports TreeSHAP
    - kernelshap # Universal fallback

# Insurance-specific metrics
metrics:
  performance:
    metrics:
      - accuracy
      - precision # Minimize false positives (unnecessary rate increases)
      - recall # Minimize false negatives (missed high-risk cases)
      - f1
      - auc_roc

  fairness:
    metrics:
      - demographic_parity # Equal approval rates across groups
      - equal_opportunity # Equal true positive rates
      - equalized_odds # Equal TPR and FPR

# Professional audit report
report:
  template: standard_audit
  styling:
    color_scheme: professional
    compliance_statement: true
```

## Step 3: Running the Audit

Execute the insurance risk assessment audit:

```bash
# Generate comprehensive audit
glassalpha audit \
  --config insurance_risk_audit.yaml \
  --output insurance_risk_audit.pdf \
  --strict
```

### Expected Execution

```
GlassAlpha Audit Generation
========================================
Loading configuration from: insurance_risk_audit.yaml
Audit profile: insurance_risk_assessment
Strict mode: ENABLED
⚠️ Strict mode enabled - enforcing regulatory compliance

Running audit pipeline...
  Loading data and initializing components...
✓ Audit pipeline completed in 3.87s

📊 Audit Summary:
  ✅ Performance metrics: 6 computed
     ✅ accuracy: 78.4%
  ⚖️ Fairness metrics: 12/12 computed
     ⚠️ Bias detected in: age_group.demographic_parity
  🔍 Explanations: ✅ Global feature importance
     Most important: previous_claims (+0.287)
  📋 Dataset: 10,000 samples, 17 features
  🔧 Components: 3 selected
     Model: lightgbm

Generating PDF report: insurance_risk_audit.pdf
✓ Saved plot to /tmp/plots/shap_importance.png
✓ Saved plot to /tmp/plots/performance_summary.png
✓ Saved plot to /tmp/plots/fairness_analysis.png
✓ Saved plot to /tmp/plots/confusion_matrix.png

🎉 Audit Report Generated Successfully!
==================================================
📁 Output: /path/to/insurance_risk_audit.pdf
📊 Size: 1,456,231 bytes (1.4 MB)
⏱️ Total time: 4.92s
   • Pipeline: 3.87s
   • PDF generation: 1.05s

🛡️ Strict mode: Report meets regulatory compliance requirements
```

## Step 4: Interpreting the Results

### Model Performance Analysis

**Overall Performance:**

- **Accuracy: 78.4%** - Model correctly classifies 78% of insurance policies
- **AUC-ROC: 0.847** - Strong ability to distinguish high-risk from low-risk policies
- **Precision: 76.2%** - Of policies predicted as high-risk, 76% actually filed claims
- **Recall: 72.1%** - Model identifies 72% of all actual high-risk policies

**Insurance Business Interpretation:**

- **Conservative Risk Assessment**: Higher precision than recall (safer underwriting)
- **False Positive Rate**: 24% of low-risk policies incorrectly flagged as high-risk
- **False Negative Rate**: 28% of high-risk policies not identified
- **Business Impact**: Balance between over-charging safe drivers vs. under-charging risky drivers

### SHAP Explanations

**Global Feature Importance (Top 5):**

1. **`previous_claims` (+0.287)**

   - Most predictive factor in risk assessment
   - Each additional claim increases risk score significantly
   - Aligns with actuarial principles and insurance industry standards

2. **`age` (+0.234)**

   - Younger drivers show higher risk patterns
   - Statistical correlation with accident frequency
   - Must be monitored for age discrimination concerns

3. **`annual_mileage` (+0.198)**

   - Higher mileage correlates with increased accident probability
   - Direct relationship to exposure (time on road)
   - Standard actuarial risk factor

4. **`vehicle_value` (-0.156)**

   - Higher value vehicles associated with lower risk
   - May indicate socioeconomic factors or careful ownership
   - Requires fairness analysis across income groups

5. **`location_risk_score` (+0.142)**
   - Geographic risk factors significantly impact predictions
   - Must ensure not proxying for protected demographic characteristics
   - Urban vs. rural differences in accident rates

**Individual Policy Example:**
For a 28-year-old male with 2 previous claims, driving 15,000 miles annually:

- **Base risk score:** 0.35 (population average)
- **Previous claims (2):** +0.18 risk increase
- **Age (28):** +0.12 risk increase
- **Annual mileage (15,000):** +0.08 risk increase
- **Vehicle value (high):** -0.06 risk decrease
- **Final risk score:** 0.67 (high-risk classification)

### Fairness Analysis Results

**Demographic Parity Analysis:**

**Age Group Bias (DETECTED):**

- **Young (18-24):** 34.2% predicted high-risk rate
- **Young Adult (25-34):** 28.7% predicted high-risk rate
- **Middle Age (35-49):** 22.1% predicted high-risk rate
- **Senior (50-64):** 18.9% predicted high-risk rate
- **Elderly (65+):** 15.3% predicted high-risk rate
- **Maximum difference:** 18.9% (Young vs. Elderly)
- **Conclusion:** Significant age-based disparities detected

**Gender Analysis:**

- **Male:** 26.4% predicted high-risk rate
- **Female:** 25.8% predicted high-risk rate
- **Difference:** 0.6% (within acceptable range)
- **Conclusion:** No significant gender bias detected

**Equal Opportunity Analysis (True Positive Rate Parity):**

- **Young:** 68.4% of actual high-risk young drivers correctly identified
- **Middle Age:** 74.2% of actual high-risk middle-aged drivers correctly identified
- **Difference:** 5.8% (borderline concern)
- **Conclusion:** Slight disparity in identifying high-risk drivers across age groups

### Risk Assessment

**High Risk Findings:**

1. **Age-Based Discrimination Risk**

   - 18.9% difference in predicted risk rates across age groups
   - Exceeds typical insurance regulatory thresholds
   - Could result in age discrimination claims
   - Requires immediate model adjustment or feature engineering

2. **Socioeconomic Proxy Risk**
   - Vehicle value and location factors may proxy for protected characteristics
   - Geographic risk scores could reflect demographic patterns
   - Requires careful feature analysis and validation

**Medium Risk Findings:**

**Model Interpretability**

- Complex interactions between age, mileage, and claims history
- Requires domain expertise to validate business logic
- Consider simpler models for higher interpretability

**Compliance Assessment:**

- **Rate Regulation:** ⚠️ REVIEW - Age-based patterns may violate actuarial standards
- **Fair Lending:** ❌ FAIL - Age discrimination detected
- **Transparency:** ✅ PASS - SHAP explanations provide clear rationale
- **Documentation:** ✅ PASS - Complete audit trail and methodology

## Step 5: Regulatory Recommendations

### Immediate Actions Required

1. **Address Age Discrimination**

   ```python
   # Consider preprocessing approaches:
   # - Age binning to reduce granularity
   # - Remove age-correlated features
   # - Apply fairness constraints during training
   ```

2. **Feature Engineering Review**

   - Audit features correlated with protected age characteristics
   - Consider removing or transforming biased location factors
   - Implement fairness-aware feature selection

3. **Model Adjustment Options**
   - Retrain with fairness constraints
   - Consider ensemble methods with bias reduction
   - Validate improvements with new audit

### Long-term Compliance Strategy

1. **Ongoing Monitoring**

   - Regular bias audits on new policy data
   - Statistical tests for demographic parity
   - Performance monitoring across protected groups

2. **Documentation Requirements**

   - Maintain complete actuarial justification
   - Document bias mitigation efforts
   - Prepare regulatory submission packages

3. **Process Improvements**
   - Establish fairness review boards
   - Implement bias testing in model development
   - Create remediation procedures for biased decisions

## Step 6: Business Impact Analysis

### Financial Impact

**Current Model:**

- **Risk assessment accuracy:** 78.4%
- **False positive rate:** 24% (safe drivers over-charged)
- **False negative rate:** 28% (risky drivers under-charged)
- **Premium impact:** Moderate (balanced risk management)

**With Bias Correction:**

- **May reduce premiums for younger drivers**
- **Could slightly increase premiums for some groups**
- **Compliance benefits outweigh small financial trade-offs**

### Legal Risk Mitigation

**Before Correction:**

- High risk of age discrimination violations
- Potential for class-action lawsuits
- Regulatory enforcement actions
- Reputational damage

**After Correction:**

- Compliance with fair insurance practices
- Reduced legal exposure
- Improved consumer trust and market position

## Step 7: Next Steps and Recommendations

### Technical Remediation

1. **Implement Fairness Constraints**

   ```python
   # Example: Add fairness penalty to LightGBM training
   # Consider libraries like fairlearn or aif360 for insurance-specific fairness
   ```

2. **Alternative Modeling Approaches**

   - Pre-processing: Age group binning or feature removal
   - In-processing: Fairness-constrained optimization
   - Post-processing: Adjust predictions to achieve parity

3. **Validation Strategy**
   - Cross-validation with fairness metrics
   - Holdout testing on diverse demographic groups
   - A/B testing for production deployment

### Operational Changes

1. **Model Governance**

   - Establish bias testing requirements for all models
   - Create fairness review processes
   - Implement continuous monitoring systems

2. **Regulatory Compliance**

   - Engage with state insurance departments
   - Prepare actuarial memorandums for rate filings
   - Document all model changes and validations

3. **Stakeholder Engagement**
   - Train underwriting staff on fair practices
   - Engage with compliance and legal teams
   - Communicate changes to policyholders

## Conclusion

This insurance risk assessment audit revealed a technically sound model with significant age-based bias that requires immediate attention before production deployment. The audit demonstrated:

**Strengths:**

- Strong predictive performance (78% accuracy, 0.85 AUC)
- Interpretable feature importance aligned with actuarial principles
- Comprehensive bias detection and measurement

**Critical Issues:**

- Age discrimination exceeding regulatory thresholds
- Potential socioeconomic proxy discrimination
- Non-compliance with fair insurance practices

**Action Plan:**

1. Implement bias mitigation techniques
2. Retrain model with fairness constraints
3. Re-audit improved model
4. Deploy with ongoing monitoring

This tutorial demonstrates how GlassAlpha enables thorough, regulatory-ready ML auditing that identifies both technical performance and compliance risks, providing the detailed analysis necessary for responsible AI deployment in regulated insurance markets.

## Additional Resources

- [Configuration Guide](../getting-started/configuration.md) - Detailed configuration options
- [CLI Reference](../reference/cli.md) - Complete command documentation
- [Compliance Overview](../reference/compliance.md) - Insurance regulatory framework guidance
- [Troubleshooting Guide](../reference/troubleshooting.md) - Common issues and solutions

For questions or support, please visit our [GitHub repository](https://github.com/GlassAlpha/glassalpha) or contact our team.
