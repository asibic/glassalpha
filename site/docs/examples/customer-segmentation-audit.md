# Customer Segmentation Audit Tutorial

Complete walkthrough of performing a comprehensive ML audit on a multi-class customer segmentation model using GlassAlpha. This tutorial demonstrates how to audit automated customer classification systems for compliance with marketing regulations and consumer privacy laws.

## Overview

Companies use machine learning models to segment customers for targeted marketing campaigns. These models must balance:

- **Marketing Effectiveness**: Accurate customer grouping for relevant messaging
- **Privacy Compliance**: Respect for consumer data protection rights
- **Fairness**: Avoid discriminatory segmentation practices
- **Transparency**: Ability to explain segmentation decisions to consumers

### What You'll Learn

- How to configure GlassAlpha for multi-class classification problems
- Interpreting performance metrics for customer segmentation
- Understanding SHAP explanations for customer grouping decisions
- Identifying bias in marketing segmentation across demographic groups
- Generating regulatory-ready audit reports for marketing compliance

### Regulatory Context

Customer segmentation models must comply with:

- **Consumer Privacy Laws**: GDPR, CCPA data processing requirements
- **Marketing Regulations**: CAN-SPAM, TCPA opt-out requirements
- **Anti-Discrimination Laws**: Fair treatment across protected groups
- **Transparency Requirements**: Right to explanation for automated decisions

## Prerequisites

- GlassAlpha installed ([Installation Guide](../getting-started/installation.md))
- Basic understanding of customer segmentation and marketing analytics
- Familiarity with privacy and fairness concepts in consumer data usage

## Step 1: Understanding the Customer Segmentation Dataset

### Dataset Characteristics

The customer segmentation dataset contains:

- **20,000 synthetic customers**
- **18 features** covering demographics, behavior, and financial indicators
- **4 customer segments**: High-Value, Regular, Occasional, At-Risk
- **Protected attributes**: Gender, age groups for fairness analysis

### Customer Segments

**Segment 0: High-Value Customers**

- High income, frequent purchases, loyalty program members
- Premium product preferences, high engagement
- Most profitable customer group

**Segment 1: Regular Customers**

- Moderate income, consistent purchasing patterns
- Standard product preferences, moderate engagement
- Steady, reliable customer base

**Segment 2: Occasional Customers**

- Variable income, sporadic purchasing behavior
- Discount-driven purchases, low engagement
- Growth opportunity segment

**Segment 3: At-Risk Customers**

- Declining engagement, potential churn risk
- Price-sensitive, low loyalty program participation
- Retention focus required

### Key Features

**Demographic Information:**

- `age` - Customer age (18-80)
- `gender` - Customer gender (protected attribute)
- `age_group` - Age categorization (protected attribute)
- `marital_status` - Relationship status indicator
- `household_size` - Number of people in household
- `children_count` - Number of children

**Financial Indicators:**

- `income` - Annual household income ($20,000-$200,000)
- `income_bracket` - Income categorization
- `education_years` - Years of formal education
- `education_level` - Education attainment level
- `home_ownership` - Housing situation
- `credit_score` - Financial health indicator

**Behavioral Metrics:**

- `purchase_frequency` - Annual purchase count (0-100)
- `avg_order_value` - Average transaction amount ($10-$1,000)
- `category_preferences` - Product category interests (0-10)
- `loyalty_program_member` - Program participation status
- `social_media_engagement` - Social platform activity (0-100)
- `mobile_app_usage` - App interaction level (0-100)

## Step 2: Configuration Setup

Create a configuration file for the customer segmentation audit:

```yaml
# customer_segmentation_audit.yaml
audit_profile: customer_segmentation

# Reproducibility for regulatory compliance
reproducibility:
  random_seed: 42

# Data configuration
data:
  path: ~/.glassalpha/data/customer_segmentation.csv
  target_column: customer_segment
  protected_attributes:
    - gender
    - age_group

# Random Forest model for multi-class segmentation
model:
  type: sklearn_generic
  params:
    model_type: RandomForestClassifier
    n_estimators: 100
    max_depth: 10
    random_state: 42

# Explanation configuration (Random Forest needs KernelSHAP)
explainers:
  strategy: first_compatible
  priority:
    - kernelshap # Random Forest requires model-agnostic explanations
  config:
    kernelshap:
      n_samples: 1000
      background_size: 100

# Multi-class specific metrics
metrics:
  performance:
    metrics:
      - accuracy
      - precision_macro # Average precision across all segments
      - recall_macro # Average recall across all segments
      - f1_macro # Macro-averaged F1 score

  fairness:
    metrics:
      - demographic_parity # Equal representation in each segment
      - equal_opportunity # Equal classification accuracy across groups
    config:
      demographic_parity:
        threshold: 0.05 # Maximum 5% difference in segment distribution
      equal_opportunity:
        threshold: 0.05

# Professional audit report
report:
  template: standard_audit
  styling:
    color_scheme: professional
    compliance_statement: true
```

## Step 3: Running the Audit

Execute the customer segmentation audit:

```bash
# Generate comprehensive audit
glassalpha audit \
  --config customer_segmentation_audit.yaml \
  --output customer_segmentation_audit.pdf \
  --strict
```

### Expected Execution

```
GlassAlpha Audit Generation
========================================
Loading configuration from: customer_segmentation_audit.yaml
Audit profile: customer_segmentation
Strict mode: ENABLED
⚠️ Strict mode enabled - enforcing regulatory compliance

Running audit pipeline...
  Loading data and initializing components...
✓ Audit pipeline completed in 8.45s

📊 Audit Summary:
  ✅ Performance metrics: 4 computed
     ✅ accuracy: 76.2%
  ⚖️ Fairness metrics: 8/8 computed
     ⚠️ Bias detected in: age_group.equal_opportunity
  🔍 Explanations: ✅ Global feature importance
     Most important: purchase_frequency (+0.287)
  📋 Dataset: 20,000 samples, 18 features
  🔧 Components: 3 selected
     Model: sklearn_generic (RandomForestClassifier)

Generating PDF report: customer_segmentation_audit.pdf
✓ Saved plot to /tmp/plots/shap_importance.png
✓ Saved plot to /tmp/plots/performance_summary.png
✓ Saved plot to /tmp/plots/fairness_analysis.png
✓ Saved plot to /tmp/plots/confusion_matrix.png

🎉 Audit Report Generated Successfully!
==================================================
📁 Output: /path/to/customer_segmentation_audit.pdf
📊 Size: 1,523,847 bytes (1.5 MB)
⏱️ Total time: 9.87s
   • Pipeline: 8.45s
   • PDF generation: 1.42s

🛡️ Strict mode: Report meets regulatory compliance requirements
```

## Step 4: Interpreting the Results

### Model Performance Analysis

**Overall Performance:**

- **Accuracy: 76.2%** - Model correctly classifies 76% of customers
- **Macro Precision: 74.8%** - Average precision across all segments
- **Macro Recall: 75.1%** - Average recall across all segments
- **Macro F1: 74.9%** - Balanced performance across segments

**Multi-Class Business Interpretation:**

- **Segment Balance**: Model performs consistently across all customer segments
- **Marketing Impact**: 76% accuracy enables reasonably targeted campaigns
- **Error Analysis**: 24% misclassification rate requires business consideration
- **Segment Distribution**: High-Value (28%), Regular (32%), Occasional (25%), At-Risk (15%)

### SHAP Explanations

**Global Feature Importance (Top 5):**

**1. `purchase_frequency` (+0.287)**

- Most important segmentation factor
- Frequent purchasers clearly separated into high-value segments
- Strong behavioral indicator of customer value

**2. `income` (+0.234)**

- Income level strongly influences segment assignment
- Higher income customers more likely in high-value segments
- Requires monitoring for socioeconomic bias

**3. `loyalty_program_member` (+0.198)**

- Program participation strongly correlated with high-value segments
- Self-selection bias in loyalty program membership
- Important for understanding customer engagement

**4. `avg_order_value` (+0.156)**

- Transaction size indicates customer value orientation
- Higher-value customers make larger purchases
- Correlates with willingness to pay premium prices

**5. `social_media_engagement` (+0.142)**

- Digital engagement level influences segmentation
- Higher engagement associated with active customer segments
- May correlate with younger demographics

**Individual Customer Example:**
For a 35-year-old customer with $85,000 income, purchasing 45 times per year:

- **Base segment probability:** High-Value (0.35), Regular (0.30), Occasional (0.25), At-Risk (0.10)
- **Purchase frequency (45):** +0.12 shift toward High-Value
- **Income ($85,000):** +0.08 shift toward High-Value
- **Loyalty program member:** +0.06 shift toward High-Value
- **Average order value ($150):** +0.04 shift toward High-Value
- **Final segment:** High-Value (0.65 probability)

### Fairness Analysis Results

**Demographic Parity Analysis:**

**Age Group Representation:**

- **Young (18-24):** 12% High-Value, 28% Regular, 35% Occasional, 25% At-Risk
- **Young Adult (25-34):** 22% High-Value, 32% Regular, 28% Occasional, 18% At-Risk
- **Middle Age (35-49):** 35% High-Value, 30% Regular, 22% Occasional, 13% At-Risk
- **Senior (50-64):** 28% High-Value, 35% Regular, 25% Occasional, 12% At-Risk
- **Elderly (65+):** 18% High-Value, 38% Regular, 28% Occasional, 16% At-Risk
- **Conclusion:** Age groups represented proportionally in segments

**Gender Analysis:**

- **Male:** 26% High-Value, 32% Regular, 26% Occasional, 16% At-Risk
- **Female:** 24% High-Value, 31% Regular, 27% Occasional, 18% At-Risk
- **Difference:** 2% maximum (within acceptable range)
- **Conclusion:** ✅ No significant gender bias in segment distribution

**Equal Opportunity Analysis:**

- **Young:** 71.2% accuracy in segment classification
- **Middle Age:** 78.4% accuracy in segment classification
- **Difference:** 7.2% (exceeds 5% threshold)
- **Conclusion:** ⚠️ Slight accuracy disparity across age groups

### Risk Assessment

**Medium Risk Findings:**

**1. Age-Based Accuracy Disparity**

- 7.2% difference in classification accuracy across age groups
- May indicate model difficulty with younger customer patterns
- Consider age-specific model tuning or feature adjustments

**2. Income Correlation**

- Income strongly influences segment assignment
- May create socioeconomic segmentation bias
- Monitor for equitable treatment across income levels

**Low Risk Findings:**

1. **Overall Model Performance**: 76% accuracy is reasonable for marketing segmentation
2. **Demographic Parity**: Balanced representation across protected groups
3. **Interpretability**: Clear feature importance provides business insights

**Compliance Assessment:**

- **Privacy Compliance:** ✅ PASS - Data processing respects consumer rights
- **Marketing Regulations:** ✅ PASS - No discriminatory segmentation detected
- **Transparency:** ✅ PASS - SHAP explanations provide clear rationale
- **Fairness:** ⚠️ REVIEW - Minor age-based accuracy differences noted

## Step 5: Regulatory Recommendations

### Immediate Actions Required

**1. Address Age Accuracy Disparity**

- Investigate why younger customers are harder to classify accurately
- Consider age-specific model parameters or additional features
- Validate with diverse age group test data

**2. Income Feature Review**

- Ensure income-based segmentation doesn't create unfair outcomes
- Consider income normalization or alternative value indicators
- Monitor for socioeconomic bias in marketing campaigns

### Long-term Compliance Strategy

**1. Ongoing Monitoring**

- Regular bias audits on new customer data
- Performance tracking across demographic groups
- Model drift detection for changing customer behaviors

**2. Privacy Compliance**

- Regular privacy impact assessments
- Consumer opt-out mechanism validation
- Data minimization principle adherence

**3. Marketing Governance**

- Segmentation model review board
- Campaign fairness impact assessments
- Consumer complaint monitoring and response

## Step 6: Business Impact Analysis

### Marketing Impact

**Current Model:**

- **Segmentation accuracy:** 76% enables targeted marketing
- **Customer insights:** Clear behavioral patterns identified
- **Campaign optimization:** Data-driven customer group targeting
- **ROI improvement:** Better matching of offers to customer preferences

**Optimization Opportunities:**

- **Accuracy improvement:** 5-10% accuracy gain through model tuning
- **Segment refinement:** Better distinction between Regular and Occasional customers
- **Personalization:** More precise individual customer recommendations

### Privacy Risk Mitigation

**Before Audit:**

- Potential privacy violations if segmentation criteria unclear
- Regulatory scrutiny without proper documentation
- Consumer trust issues without transparency

**After Audit:**

- Documented compliance with privacy regulations
- Clear data processing justification
- Consumer explanation capabilities

## Step 7: Next Steps and Recommendations

### Technical Improvements

**1. Model Enhancement**

- Feature engineering for better age group representation
- Alternative model architectures for multi-class problems
- Ensemble methods combining multiple segmentation approaches

**2. Data Quality**

- Additional behavioral data sources
- Customer feedback integration
- External data validation

**3. Advanced Analytics**

- Customer lifetime value modeling
- Churn prediction within segments
- Dynamic segmentation based on behavior changes

### Operational Changes

**1. Marketing Integration**

- Real-time segmentation updates
- Campaign performance tracking by segment
- A/B testing for segmentation strategies

**2. Consumer Experience**

- Transparent segmentation explanations
- Opt-out mechanisms for targeted marketing
- Personalized privacy controls

**3. Regulatory Compliance**

- Regular privacy impact assessments
- Consumer data rights implementation
- Regulatory reporting and documentation

## Conclusion

This customer segmentation audit revealed a solid multi-class classification model that effectively groups customers while maintaining reasonable fairness across demographic groups. The audit demonstrated:

**Strengths:**

- Good overall performance (76% accuracy) for marketing applications
- Balanced segment representation across protected groups
- Clear interpretability of segmentation factors
- Privacy-compliant data processing

**Areas for Enhancement:**

- Minor age-based accuracy differences requiring attention
- Income correlation monitoring for socioeconomic fairness
- Enhanced model tuning for better segment distinction

**Compliance Status:**

- ✅ Privacy regulation compliance
- ✅ Marketing regulation adherence
- ✅ Transparency requirements met
- ⚠️ Minor fairness considerations noted

This tutorial demonstrates how GlassAlpha enables thorough, regulatory-ready ML auditing for multi-class problems, providing the detailed analysis necessary for responsible AI deployment in consumer marketing and segmentation systems.

## Additional Resources

- [Configuration Guide](../getting-started/configuration.md) - Detailed configuration options
- [CLI Reference](../reference/cli.md) - Complete command documentation
- [Compliance Overview](../reference/compliance.md) - Consumer privacy regulatory framework guidance
- [Troubleshooting Guide](../reference/troubleshooting.md) - Common issues and solutions

For questions or support, please visit our [GitHub repository](https://github.com/GlassAlpha/glassalpha) or contact our team.
