# Compliance Framework Overview

GlassAlpha addresses key regulatory frameworks governing AI and algorithmic decision-making. This guide explains how specific features map to regulatory requirements and provides configuration guidance for compliance.

## Supported Regulatory Frameworks

### EU General Data Protection Regulation (GDPR)
**Jurisdiction:** European Union
**Effective:** May 25, 2018
**Key Requirements:** Data protection, consent, right to explanation

### Equal Credit Opportunity Act (ECOA)
**Jurisdiction:** United States
**Effective:** 1974 (amended multiple times)
**Key Requirements:** Fair lending, non-discrimination in credit decisions

### Fair Credit Reporting Act (FCRA)
**Jurisdiction:** United States
**Effective:** 1970 (amended multiple times)
**Key Requirements:** Accuracy, fairness, privacy in credit reporting

### EU AI Act
**Jurisdiction:** European Union
**Status:** Adopted April 2024, phased implementation through 2027
**Key Requirements:** Risk classification, transparency, human oversight, quality management

**Implementation Timeline:**
- **August 2024**: Prohibited practices banned
- **February 2025**: General purpose AI model obligations
- **August 2026**: High-risk AI system requirements
- **August 2027**: Full implementation for all provisions

### Fair Housing Act (FHA)
**Jurisdiction:** United States
**Effective:** 1968
**Key Requirements:** Non-discrimination in housing decisions

### Employment Standards
**Jurisdiction:** Various (EEOC, state laws)
**Key Requirements:** Non-discrimination in hiring and promotion

## GDPR Compliance

### Article 22: Automated Decision-Making

**Requirement:** Individuals have the right not to be subject to decisions based solely on automated processing that produce legal or similarly significant effects.

**GlassAlpha Implementation:**

```yaml
# GDPR-compliant configuration
audit_profile: gdpr_compliance

# Enable comprehensive explanation generation
explainers:
  strategy: first_compatible
  priority:
    - treeshap      # Provides detailed feature attributions
    - kernelshap    # Model-agnostic explanations
  config:
    treeshap:
      max_samples: 1000
      include_individual_explanations: true

# Generate human-readable explanations
report:
  include_sections:
    - individual_explanations  # Article 22 requirement
    - processing_logic         # Algorithm transparency
    - data_sources             # Data provenance
    - decision_factors         # Key decision factors
```

**Key Features:**
- **Individual Explanations**: SHAP waterfall plots show how each feature contributed to specific decisions
- **Transparent Logic**: Reports document the decision-making process in human-readable language
- **Data Provenance**: Complete audit trail of data sources and transformations
- **Contestation Support**: Detailed explanations enable individuals to challenge decisions

### Article 13-14: Information to Data Subjects

**Requirement:** Provide meaningful information about automated decision-making logic.

**GlassAlpha Implementation:**
- **Model Documentation**: Comprehensive model cards with algorithm details
- **Feature Documentation**: Clear explanations of input variables and their business meaning
- **Decision Boundaries**: Visualizations of how decisions are made
- **Statistical Metrics**: Performance statistics in accessible format

## ECOA Compliance (Fair Lending)

### Regulatory Context

The Equal Credit Opportunity Act prohibits credit discrimination and establishes specific requirements for credit decision-making. Regulation B (12 CFR 1002) implements ECOA with detailed compliance obligations.

**Prohibited Bases (15 USC 1691(a)):**
Creditors may not discriminate on the basis of:
- Race or color
- Religion
- National origin
- Sex or gender identity
- Marital status
- Age (with exceptions for legal capacity)
- Receipt of public assistance
- Good faith exercise of Consumer Credit Protection Act rights

**Disparate Impact Standard:**
Under ECOA, practices that have a disparate impact on protected classes may be unlawful even without discriminatory intent, unless the practice serves legitimate business needs that cannot reasonably be achieved by less discriminatory means.

### Prohibition of Discrimination

**Requirement:** Creditors may not discriminate against applicants based on race, color, religion, national origin, sex, marital status, age, or public assistance status.

**GlassAlpha Implementation:**

```yaml
# ECOA compliance configuration
audit_profile: ecoa_compliance

data:
  protected_attributes:
    - race
    - gender
    - age_group
    - marital_status
    - national_origin

# Strict bias detection thresholds
metrics:
  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity
      - equalized_odds
    config:
      demographic_parity:
        threshold: 0.02          # 2% maximum disparity
      equal_opportunity:
        threshold: 0.02

compliance:
  frameworks:
    - ecoa
  fairness_thresholds:
    demographic_parity: 0.02   # Stricter than typical research (5%)
    equal_opportunity: 0.02
    statistical_significance: 0.01
```

**Key Compliance Features:**

1. **Disparate Impact Analysis**
   - Automated calculation of approval rate ratios by protected class
   - Statistical significance testing with confidence intervals
   - Visual representations of disparities across demographic groups

2. **Adverse Action Reasons**
   - SHAP explanations provide specific reasons for denials
   - Feature importance rankings identify key decision factors
   - Individual prediction explanations support adverse action notices

3. **Model Monitoring**
   - Ongoing bias detection across protected characteristics
   - Statistical tests for demographic parity and equal opportunity
   - Drift detection to identify changing discriminatory patterns

### Documentation Requirements

**ECOA Section 1002.13:** Maintain records demonstrating compliance.

**GlassAlpha Evidence:**
- **Audit Reports**: Comprehensive bias analysis with statistical testing
- **Model Documentation**: Detailed methodology and validation procedures
- **Decision Records**: Individual prediction explanations and confidence scores
- **Monitoring Reports**: Ongoing performance tracking across demographic groups

## Configuration Patterns by Framework

### Financial Services (ECOA/FCRA)

```yaml
audit_profile: financial_compliance

data:
  protected_attributes:
    - race
    - gender
    - age_group
    - marital_status

model:
  type: xgboost
  validation:
    cross_validation: true
    bootstrap_confidence: true

metrics:
  fairness:
    metrics: [demographic_parity, equal_opportunity, equalized_odds]
    config:
      demographic_parity: { threshold: 0.02 }
      statistical_tests: true
      confidence_intervals: true

compliance:
  frameworks: [ecoa, fcra]
  documentation:
    adverse_action_reasons: true
    accuracy_statements: true
    fairness_analysis: true
```

### Healthcare (GDPR/FDA/Medical Device Regulations)

```yaml
audit_profile: healthcare_compliance

data:
  protected_attributes:
    - age_group
    - gender
    - race
    - disability_status

explainers:
  priority: [treeshap, kernelshap]
  config:
    individual_explanations: true
    clinical_interpretability: true

compliance:
  frameworks: [gdpr, medical_device_regulation]
  risk_assessment: high
  human_oversight: required
```

### Employment (EEOC/GDPR)

```yaml
audit_profile: employment_compliance

data:
  protected_attributes:
    - race
    - gender
    - age_group
    - disability_status
    - national_origin

metrics:
  fairness:
    metrics: [demographic_parity, equal_opportunity]
    config:
      four_fifths_rule: true    # 80% rule for adverse impact
      statistical_significance: true

compliance:
  frameworks: [eeoc_uniform_guidelines, gdpr]
```

## Best Practices

### Development Phase

1. **Early Compliance Integration**
   - Define regulatory requirements before model development
   - Build fairness constraints into model training
   - Establish bias testing procedures from day one

2. **Documentation Standards**
   - Maintain comprehensive model development records
   - Document all design decisions and trade-offs
   - Create audit trails for all data and model changes

### Production Deployment

1. **Ongoing Monitoring**
   - Implement continuous bias detection
   - Monitor model performance across demographic groups
   - Track individual decision outcomes for patterns

2. **Incident Response**
   - Establish procedures for bias detection alerts
   - Maintain capability to explain any historical decision
   - Document remediation actions and their effectiveness

This compliance framework overview provides the foundation for using GlassAlpha in regulated environments. For specific regulatory guidance, consult with legal counsel and compliance experts familiar with your jurisdiction and use case.
