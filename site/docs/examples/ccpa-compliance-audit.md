# CCPA Compliance Audit Tutorial

Complete walkthrough of performing a comprehensive ML audit on an automated decision-making system using GlassAlpha. This tutorial demonstrates how to audit consumer consent prediction models for compliance with the California Consumer Privacy Act and consumer rights protection.

## Overview

Businesses use machine learning models to predict consumer consent for automated decision-making and data processing activities. These models must comply with:

- **CCPA Consumer Rights**: Right to know, delete, opt-out of data sales, and non-discrimination
- **Automated Decision-Making Transparency**: Clear explanations of algorithmic decisions
- **Data Minimization**: Collection and use of only necessary personal information
- **Non-Discrimination**: Equal treatment regardless of privacy choices

### What You'll Learn

- How to configure GlassAlpha for CCPA compliance auditing
- Interpreting performance metrics for automated decision-making systems
- Understanding SHAP explanations for consumer consent transparency
- Identifying bias in automated decisions across demographic groups
- Generating regulatory-ready audit reports for California privacy compliance

### Regulatory Context

CCPA compliance requires:

- **Consumer Privacy Rights**: Right to know, access, delete, and opt-out of data sales
- **Automated Decision-Making**: Transparency and human intervention capabilities
- **Non-Discrimination**: Equal treatment regardless of privacy exercise
- **Data Processing Transparency**: Clear information about data collection and use

## Prerequisites

- GlassAlpha installed ([Installation Guide](../getting-started/installation.md))
- Basic understanding of CCPA and California consumer privacy law
- Familiarity with automated decision-making and consumer rights systems

## Step 1: Understanding the CCPA Compliance Dataset

### Dataset Characteristics

The CCPA compliance dataset contains:

- **8,000 synthetic California consumer records**
- **14 features** covering demographics, privacy preferences, and behavior
- **Binary target**: Consent for automated decisions (1) vs Opt-out (0)
- **Protected attributes**: Gender, age groups, income brackets
- **CCPA-specific fields**: Consumer rights exercised, profiling categories

### Consumer Demographics

**Age Distribution:**
- Young (18-24): 12% of consumers
- Young Adult (25-34): 25% of consumers
- Middle Age (35-49): 35% of consumers
- Senior (50-64): 20% of consumers
- Elderly (65+): 8% of consumers

**California Regions:**
- Los Angeles: 35%
- San Francisco: 25%
- San Diego: 15%
- Sacramento: 10%
- Other California: 15%

**Consumer Rights Exercised:**
- Do Not Sell Requests: 10%
- Data Deletion Requests: 2%
- Data Portability Requests: 5%

### Key Features

**Demographic Information:**
- `age` - Consumer age (18-85)
- `gender` - Consumer gender (protected attribute)
- `age_group` - Age categorization (protected attribute)
- `income_bracket` - Income categorization (socioeconomic analysis)
- `california_region` - Geographic region within California

**Privacy Preferences:**
- `marketing_consent_given` - Explicit marketing consent status
- `data_sharing_opt_out` - Data sharing opt-out status
- `tracking_cookies_accepted` - Cookie consent status
- `location_services_enabled` - Location data consent

**Behavioral Indicators:**
- `online_purchase_frequency` - Online shopping frequency (0-60 purchases/year)
- `subscription_services` - Subscription count (0-12 services)
- `social_media_activity` - Social media engagement (0-100)
- `mobile_app_usage` - App interaction frequency (0-200 sessions)

**CCPA Compliance Fields:**
- `automated_decision_consent` - Consent for automated decision-making
- `data_portability_requested` - Data export requests
- `data_deletion_requested` - Data deletion requests
- `do_not_sell_requested` - Data sales opt-out requests
- `profiling_category` - Type of automated profiling (marketing, credit, employment, insurance)

## Step 2: Configuration Setup

Create a configuration file for the CCPA compliance audit:

```yaml
# ccpa_compliance_audit.yaml
audit_profile: ccpa_compliance

# Reproducibility for regulatory compliance
reproducibility:
  random_seed: 42

# Data configuration
data:
  path: ~/.glassalpha/data/ccpa_compliance.csv
  target_column: automated_decision_consent
  protected_attributes:
    - gender
    - age_group
    - income_bracket

# XGBoost model for automated decision consent prediction
model:
  type: xgboost
  params:
    objective: binary:logistic
    eval_metric: logloss
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42

# Explanation configuration
explainers:
  strategy: first_compatible
  priority:
    - treeshap    # XGBoost supports TreeSHAP for CCPA transparency
    - kernelshap  # Fallback for any model type

# CCPA-specific metrics
metrics:
  performance:
    metrics:
      - accuracy
      - precision    # Important for consent accuracy
      - recall       # Important for consumer rights
      - f1
      - auc_roc     # Overall discriminative ability

  fairness:
    metrics:
      - demographic_parity    # Equal consent rates across groups
      - equal_opportunity     # Equal TPR for consent prediction
      - predictive_parity      # Equal precision across demographic groups
    config:
      # CCPA-compliant thresholds
      demographic_parity:
        threshold: 0.05  # Maximum 5% difference
      equal_opportunity:
        threshold: 0.05
      predictive_parity:
        threshold: 0.05

# Professional CCPA audit report
report:
  template: standard_audit
  styling:
    color_scheme: professional
    compliance_statement: true
```

## Step 3: Running the Audit

Execute the CCPA compliance audit:

```bash
# Generate comprehensive audit
glassalpha audit \
  --config ccpa_compliance_audit.yaml \
  --output ccpa_compliance_audit.pdf \
  --strict
```

### Expected Execution

```
GlassAlpha Audit Generation
========================================
Loading configuration from: ccpa_compliance_audit.yaml
Audit profile: ccpa_compliance
Strict mode: ENABLED
⚠️ Strict mode enabled - enforcing regulatory compliance

Running audit pipeline...
  Loading data and initializing components...
✓ Audit pipeline completed in 3.45s

📊 Audit Summary:
  ✅ Performance metrics: 5 computed
     ✅ accuracy: 81.2%
  ⚖️ Fairness metrics: 9/9 computed
     ✅ demographic_parity: 0.018 (within 5% threshold)
  🔍 Explanations: ✅ Global feature importance
     Most important: marketing_consent_given (+0.298)
  📋 Dataset: 8,000 samples, 14 features
  🔧 Components: 3 selected
     Model: xgboost

Generating PDF report: ccpa_compliance_audit.pdf
✓ Saved plot to /tmp/plots/shap_importance.png
✓ Saved plot to /tmp/plots/performance_summary.png
✓ Saved plot to /tmp/plots/fairness_analysis.png
✓ Saved plot to /tmp/plots/confusion_matrix.png

🎉 Audit Report Generated Successfully!
==================================================
📁 Output: /path/to/ccpa_compliance_audit.pdf
📊 Size: 1,234,567 bytes (1.2 MB)
⏱️ Total time: 4.67s
   • Pipeline: 3.45s
   • PDF generation: 1.22s

🛡️ Strict mode: Report meets regulatory compliance requirements
```

## Step 4: Interpreting the Results

### Model Performance Analysis

**Overall Performance:**
- **Accuracy: 81.2%** - Model correctly predicts 81% of automated decision consent
- **AUC-ROC: 0.873** - Strong discriminative ability for consent prediction
- **Precision: 79.4%** - Of predicted consents, 79% are actually granted
- **Recall: 74.8%** - Model identifies 75% of all actual consent grants

**CCPA Compliance Interpretation:**
- **Consumer Rights Accuracy**: 81% accuracy supports consumer choice respect
- **False Positive Rate**: 21% of predicted consents are actually opt-outs
- **False Negative Rate**: 25% of actual consents are missed
- **Non-Discrimination**: Model respects consumer privacy choices

### SHAP Explanations

**Global Feature Importance (Top 5):**

1. **`marketing_consent_given` (+0.298)**
   - Most important CCPA compliance factor
   - Explicit marketing consent strongly predicts automated decision consent
   - Demonstrates respect for consumer privacy preferences

2. **`online_purchase_frequency` (+0.267)**
   - Purchase behavior strongly correlates with consent likelihood
   - Engaged customers more willing to consent to automated decisions
   - Indicates legitimate interest basis for processing

3. **`subscription_services` (+0.234)**
   - Subscription engagement predicts consent willingness
   - Ongoing service relationships increase consent probability
   - Supports relationship-based consent justification

4. **`tracking_cookies_accepted` (+0.198)**
   - Cookie consent status influences automated decision consent
   - Privacy-aware consumers show consistent behavior patterns
   - Important for understanding consumer privacy preferences

5. **`california_region` (+0.156)**
   - Geographic location within California affects consent patterns
   - May reflect regional privacy awareness differences
   - Requires monitoring for geographic discrimination

**Individual Consumer Example:**
For a 32-year-old consumer in Los Angeles with high engagement:

- **Base consent probability:** 0.68 (68% population average)
- **Marketing consent (given):** +0.16 probability increase
- **Online purchases (25/year):** +0.14 probability increase
- **Subscriptions (4 services):** +0.12 probability increase
- **Cookies accepted (yes):** +0.08 probability increase
- **California region (LA):** +0.04 probability increase
- **Final consent probability:** 0.92 (92% - high confidence in consent)

### Fairness Analysis Results

**Demographic Parity Analysis:**

**Age Group Consent Rates:**
- **Young (18-24):** 64.5% predicted consent rate
- **Young Adult (25-34):** 72.1% predicted consent rate
- **Middle Age (35-49):** 78.3% predicted consent rate
- **Senior (50-64):** 82.7% predicted consent rate
- **Elderly (65+):** 85.2% predicted consent rate
- **Maximum difference:** 20.7% (Young vs. Elderly)
- **Conclusion:** ⚠️ Significant age-based disparities detected

**Income Bracket Analysis:**
- **Low Income:** 71.8% predicted consent rate
- **Middle Income:** 76.4% predicted consent rate
- **Upper Middle:** 79.2% predicted consent rate
- **High Income:** 81.6% predicted consent rate
- **Maximum difference:** 9.8% (exceeds 5% threshold)
- **Conclusion:** ⚠️ Income-based disparities detected

**Predictive Parity Analysis:**
- **Young:** 71.2% precision (of flagged young consumers, 71% actually consent)
- **High Income:** 83.4% precision
- **Difference:** 12.2% (exceeds 5% threshold)
- **Conclusion:** ⚠️ Socioeconomic accuracy disparity in consent prediction

### Risk Assessment

**High Risk Findings:**

1. **Age and Income-Based Disparities**
   - 20.7% difference in consent rates across age groups
   - 9.8% difference across income brackets
   - May indicate discrimination in automated decision-making
   - Could result in CCPA non-discrimination violations

2. **Consumer Rights Impact**
   - 10% of consumers have exercised "Do Not Sell" rights
   - Model must respect these privacy choices
   - Requires enhanced transparency for automated decisions

**Medium Risk Findings:**

1. **Consent Prediction Accuracy**
   - 19% false positive rate may lead to unwanted automated decisions
   - 25% false negative rate misses consent opportunities
   - Balance needed between automation efficiency and consumer rights

**Compliance Assessment:**

- **CCPA Consumer Rights:** ⚠️ REVIEW - Disparities require attention
- **Automated Decision-Making:** ✅ PASS - Transparency mechanisms in place
- **Non-Discrimination:** ❌ FAIL - Age and income bias detected
- **Data Processing Transparency:** ✅ PASS - Clear data usage documentation

## Step 5: Regulatory Recommendations

### Immediate Actions Required

1. **Address Demographic Disparities**
   ```python
   # Consider CCPA-compliant approaches:
   # - Demographic-stratified consent prediction models
   # - Enhanced consumer rights verification
   # - Human oversight for high-risk automated decisions
   ```

2. **Enhance Consumer Rights Mechanisms**
   - Implement clear opt-out mechanisms for automated decisions
   - Provide meaningful explanations for algorithmic decisions
   - Ensure non-discrimination regardless of privacy choices

3. **Transparency Enhancement**
   - Develop consumer-friendly automated decision explanations
   - Implement right to human intervention processes
   - Document legitimate interest basis for processing

### Long-term Compliance Strategy

1. **Ongoing CCPA Monitoring**
   - Regular bias audits on automated decision-making models
   - Consumer rights exercise tracking and analysis
   - Model performance monitoring across demographic groups

2. **Consumer Rights Management**
   - Automated consumer rights request processing
   - Regular privacy preference verification
   - Enhanced transparency for data processing activities

3. **Regulatory Documentation**
   - Maintain comprehensive consumer rights documentation
   - Document non-discrimination measures
   - Prepare regulatory examination materials

## Step 6: Business Impact Analysis

### CCPA Compliance Impact

**Current Model:**
- **Automated decision consent prediction:** 81% accuracy
- **Consumer rights support:** Enables privacy choice respect
- **Business efficiency:** Automated consent management at scale
- **Regulatory compliance:** Supports CCPA requirements

**Consumer Rights Benefits:**
- **Privacy protection:** Respects consumer data choices
- **Transparency:** Clear automated decision-making explanations
- **Non-discrimination:** Equal treatment regardless of privacy preferences
- **Trust building:** Demonstrates commitment to consumer rights

### Regulatory Risk Mitigation

**Before Audit:**
- Potential CCPA violations if bias undetected
- Consumer rights non-compliance without transparency
- Regulatory scrutiny without proper consumer protections

**After Audit:**
- Documented compliance with CCPA consumer rights requirements
- Demonstrated fairness across demographic groups
- Clear audit trail for California Attorney General review

## Step 7: Next Steps and Recommendations

### Technical Improvements

1. **Model Enhancement**
   - Demographic-aware consent prediction modeling
   - Integration with consumer rights management systems
   - Enhanced transparency mechanisms for automated decisions

2. **CCPA Compliance Features**
   - Automated consumer rights request processing
   - Consent withdrawal tracking and implementation
   - Data processing necessity and proportionality assessments

3. **Advanced Analytics**
   - Consumer rights exercise pattern analysis
   - Automated decision impact assessment
   - Privacy preference trend monitoring

### Operational Changes

1. **Consumer Rights Integration**
   - Real-time consumer rights dashboard
   - Automated privacy preference processing
   - Consumer education on automated decision rights

2. **CCPA Program Management**
   - California Attorney General reporting
   - Consumer rights exercise tracking
   - Privacy impact assessment maintenance

3. **Transparency Enhancement**
   - Consumer-friendly automated decision explanations
   - Right to human intervention implementation
   - Clear privacy policy communications

## Conclusion

This CCPA compliance audit revealed an effective automated decision consent model with important demographic disparities that require immediate attention for full California consumer privacy compliance. The audit demonstrated:

**Strengths:**
- Good overall performance (81% accuracy) for consent management
- Strong correlation between privacy preferences and consent behavior
- Comprehensive bias detection across multiple demographic dimensions
- CCPA-compliant transparency through SHAP explanations

**Critical Issues:**
- Age and income discrimination exceeding regulatory thresholds
- Need for enhanced consumer rights mechanisms
- Consumer non-discrimination requirements

**Action Plan:**
1. Address demographic disparities through model refinement
2. Implement enhanced consumer rights transparency
3. Conduct comprehensive CCPA compliance review
4. Deploy with ongoing consumer rights monitoring

This tutorial demonstrates how GlassAlpha enables thorough, regulatory-ready ML auditing for automated decision-making systems, providing the detailed analysis necessary for responsible AI deployment under California consumer privacy law.

## Additional Resources

- [Configuration Guide](../getting-started/configuration.md) - Detailed configuration options
- [CLI Reference](../reference/cli.md) - Complete command documentation
- [Compliance Overview](../compliance/overview.md) - CCPA regulatory framework guidance
- [Troubleshooting Guide](../reference/troubleshooting.md) - Common issues and solutions

For questions or support, please visit our [GitHub repository](https://github.com/GlassAlpha/glassalpha) or contact our team.
