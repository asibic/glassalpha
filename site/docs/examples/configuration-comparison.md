# Configuration Comparison: Audit Profiles and Options

This example demonstrates how different GlassAlpha configurations affect audit outcomes, helping you choose the right settings for your use case and regulatory requirements.

## Overview

Understanding configuration impact is crucial for:

- **Compliance requirements** - Different regulations need different approaches
- **Performance optimization** - Balancing speed vs comprehensiveness
- **Audit quality** - Selecting appropriate metrics and explanations
- **Organizational needs** - Matching audit depth to stakeholder requirements

We'll compare four configuration approaches using the same German Credit dataset:

1. **Minimal Configuration** - Fastest execution, basic audit
2. **Standard Compliance** - Balanced approach for general use
3. **Strict Regulatory** - Comprehensive audit for regulated industries
4. **Performance Optimized** - Speed-focused for large-scale operations

## Configuration Comparison Matrix

| Aspect | Minimal | Standard | Strict | Performance |
|--------|---------|----------|--------|-------------|
| **Execution Time** | ~10 seconds | ~30 seconds | ~60 seconds | ~15 seconds |
| **Explainer Depth** | Basic SHAP | Full SHAP | Multiple methods | Sampled SHAP |
| **Metrics Coverage** | Core only | Comprehensive | All + custom | Performance focus |
| **Validation Level** | Basic | Standard | Comprehensive | Minimal |
| **Reproducibility** | Basic | Full | Auditable | Full |
| **Report Detail** | Summary | Complete | Regulatory | Streamlined |
| **Use Case** | Learning | Production | Compliance | Scale operations |

## 1. Minimal Configuration

**File**: `configs/german_credit_simple.yaml`

```yaml
# Minimal configuration for quick testing
audit_profile: tabular_basic
reproducibility:
  random_seed: 42

data:
  path: german_credit_data.csv
  target_column: credit_risk
  protected_attributes: [gender]

model:
  type: xgboost

explainers:
  priority: [treeshap]
  config:
    treeshap:
      max_samples: 100  # Fast execution

metrics:
  performance: [accuracy, auc_roc]
  fairness: [demographic_parity]

report:
  template: summary
```

**Command**:
```bash
glassalpha audit --config configs/german_credit_simple.yaml --output minimal_audit.pdf
```

**Results**:

- **Execution**: ~10 seconds
- **Report size**: 8 pages
- **Key findings**: Basic performance (77% accuracy) and simple bias check
- **Use case**: Quick validation, learning, proof-of-concept

**Trade-offs**:

- ✅ **Fast execution** for rapid iteration
- ✅ **Simple results** easy to understand
- ❌ **Limited compliance** value for regulations
- ❌ **Shallow analysis** may miss important biases

## 2. Standard Compliance Configuration

**File**: `configs/example_audit.yaml`

```yaml
# Standard configuration for production use
audit_profile: tabular_compliance
strict_mode: false
reproducibility:
  random_seed: 42
  track_git_sha: true

data:
  path: german_credit_data.csv
  target_column: credit_risk
  protected_attributes: [gender, age_group]
  feature_columns: auto  # Auto-detect features

model:
  type: xgboost
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  validation:
    cross_validation: true
    folds: 5

explainers:
  priority: [treeshap, kernelshap]
  config:
    treeshap:
      max_samples: 500
      include_individual_explanations: true
    kernelshap:
      n_samples: 100

metrics:
  performance: [accuracy, precision, recall, f1, auc_roc]
  fairness: [demographic_parity, equal_opportunity, equalized_odds]
  drift: [psi, kl_divergence]

report:
  template: standard_audit
  include_sections:
    - executive_summary
    - model_performance
    - fairness_analysis
    - explanations
    - reproducibility

validation:
  schema_validation: true
  data_quality_checks: true
```

**Command**:
```bash
glassalpha audit --config configs/example_audit.yaml --output standard_audit.pdf
```

**Results**:

- **Execution**: ~30 seconds
- **Report size**: 18 pages
- **Key findings**: Comprehensive analysis with 77% accuracy, gender bias detected
- **Use case**: Production deployments, general compliance

**Analysis**:

- ✅ **Balanced approach** between speed and depth
- ✅ **Good compliance** coverage for most use cases
- ✅ **Comprehensive metrics** without overwhelming detail
- ⚠️ **May not meet** strictest regulatory requirements

## 3. Strict Regulatory Configuration

**File**: `configs/gdpr_compliance.yaml`

```yaml
# Strict configuration for regulatory compliance
audit_profile: financial_compliance
strict_mode: true  # Enforces regulatory requirements
reproducibility:
  random_seed: 42
  track_git_sha: true
  track_environment: true
  require_data_hash: true

data:
  path: german_credit_data.csv
  target_column: credit_risk
  protected_attributes: [gender, age_group, personal_status]
  feature_columns: auto
  schema_validation: strict

model:
  type: xgboost
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    random_state: 42
  validation:
    cross_validation: true
    folds: 10  # More robust validation
    stratified: true
    bootstrap_confidence: true

explainers:
  strategy: first_compatible
  priority: [treeshap, kernelshap]
  config:
    treeshap:
      max_samples: 1000  # Maximum accuracy
      include_individual_explanations: true
      check_additivity: true
    kernelshap:
      n_samples: 500
      background_size: 200

metrics:
  performance: [accuracy, precision, recall, f1, auc_roc, classification_report]
  fairness: [demographic_parity, equal_opportunity, equalized_odds, predictive_parity]
  drift: [psi, kl_divergence, ks_test, js_divergence]
  fairness_thresholds:
    demographic_parity: 0.02  # Strict threshold
    statistical_significance: 0.01

compliance:
  frameworks: [gdpr, ecoa, fcra]
  documentation:
    model_cards: true
    dataset_cards: true
    bias_assessment: true
    audit_trail: complete

report:
  template: regulatory_compliance
  audience: regulators
  include_sections:
    - executive_summary
    - regulatory_assessment
    - model_performance
    - bias_analysis
    - individual_explanations
    - risk_assessment
    - mitigation_recommendations
    - technical_appendix
    - reproducibility
    - manifest

validation:
  schema_validation: strict
  data_quality_checks: comprehensive
  statistical_tests: true
  confidence_intervals: true

manifest:
  include_config_hash: true
  include_data_hash: true
  include_model_hash: true
  include_environment: true
```

**Command**:
```bash
glassalpha audit --config configs/gdpr_compliance.yaml --output regulatory_audit.pdf --strict
```

**Results**:

- **Execution**: ~60 seconds
- **Report size**: 28 pages
- **Key findings**: Comprehensive regulatory analysis, detailed bias assessment, complete audit trail
- **Use case**: Regulatory submissions, legal compliance, auditor review

**Regulatory Features**:

- ✅ **Complete audit trail** with all hashes and environment info
- ✅ **Strict validation** catching configuration errors
- ✅ **Comprehensive bias testing** across multiple protected attributes
- ✅ **Statistical rigor** with confidence intervals and significance tests
- ✅ **Regulatory alignment** with GDPR, ECOA, FCRA requirements
- ❌ **Slower execution** due to comprehensive analysis

## 4. Performance Optimized Configuration

**File**: `configs/fraud_detection.yaml`

```yaml
# Performance-optimized configuration for scale
audit_profile: tabular_performance
reproducibility:
  random_seed: 42

data:
  path: german_credit_data.csv
  target_column: credit_risk
  protected_attributes: [gender]

model:
  type: xgboost
  params:
    n_estimators: 50  # Fewer trees for speed
    max_depth: 4      # Lower complexity
    learning_rate: 0.2 # Faster convergence
  validation:
    cross_validation: false  # Skip CV for speed

explainers:
  priority: [treeshap]
  config:
    treeshap:
      max_samples: 50   # Minimal sampling
      approximate: true # Use approximations

metrics:
  performance: [accuracy, auc_roc]  # Essential metrics only
  fairness: [demographic_parity]

performance:
  n_jobs: -1          # Use all CPU cores
  low_memory_mode: false
  parallel_processing: true

report:
  template: performance_summary
  include_sections:
    - executive_summary
    - key_metrics
    - basic_explanations

validation:
  schema_validation: false  # Skip for speed
  data_quality_checks: false
```

**Command**:
```bash
glassalpha audit --config configs/fraud_detection.yaml --output performance_audit.pdf
```

**Results**:

- **Execution**: ~15 seconds
- **Report size**: 10 pages
- **Key findings**: Core performance metrics, basic bias check
- **Use case**: High-volume processing, monitoring pipelines, quick checks

**Optimization Features**:

- ✅ **Fast execution** suitable for automated pipelines
- ✅ **Resource efficient** for large-scale operations
- ✅ **Core insights** without overwhelming detail
- ❌ **Limited depth** for thorough compliance analysis
- ❌ **Reduced validation** may miss data issues

## Side-by-Side Results Comparison

### Model Performance Consistency

All configurations train on the same data with similar parameters:

| Configuration | Accuracy | AUC-ROC | Precision | Recall | F1 |
|---------------|----------|---------|-----------|---------|-----|
| Minimal | 77.0% | 0.758 | 0.65 | 0.42 | 0.51 |
| Standard | 76.8% | 0.761 | 0.64 | 0.43 | 0.52 |
| Regulatory | 76.9% | 0.760 | 0.64 | 0.43 | 0.51 |
| Performance | 76.2% | 0.752 | 0.63 | 0.41 | 0.50 |

**Key Insight**: Model performance remains consistent across configurations. Differences come from validation depth and reporting detail.

### Bias Detection Results

Gender bias analysis across configurations:

| Configuration | Demographic Parity | Statistical Significance | Confidence Interval |
|---------------|-------------------|-------------------------|-------------------|
| Minimal | 0.089 | Not tested | Not provided |
| Standard | 0.087 | p < 0.05 | [0.042, 0.132] |
| Regulatory | 0.089 | p < 0.001 | [0.051, 0.127] |
| Performance | 0.088 | Not tested | Not provided |

**Key Insight**: All configurations detect the same underlying bias, but regulatory provides the statistical rigor needed for compliance.

### Execution Time Analysis

| Configuration | Data Loading | Model Training | Explanations | Metrics | Report Gen | Total |
|---------------|-------------|----------------|-------------|---------|------------|-------|
| Minimal | 2s | 3s | 3s | 1s | 1s | **10s** |
| Standard | 3s | 5s | 15s | 4s | 3s | **30s** |
| Regulatory | 4s | 8s | 35s | 8s | 5s | **60s** |
| Performance | 2s | 2s | 2s | 1s | 1s | **8s** |

**Key Insight**: Explanation generation is the primary time driver. SHAP sampling levels directly impact execution time.

## Configuration Selection Guide

### Choose **Minimal** when:
- Learning GlassAlpha capabilities
- Rapid prototyping and iteration
- Basic proof-of-concept demonstrations
- Resource-constrained environments
- Quick validation of data/model setup

### Choose **Standard** when:
- Production model deployment
- General compliance requirements
- Balanced speed/comprehensiveness needs
- Regular audit reporting
- Stakeholder presentations

### Choose **Regulatory** when:
- Regulatory submission required
- Legal compliance critical
- External auditor review
- High-risk applications (finance, healthcare)
- Complete documentation needed

### Choose **Performance** when:
- High-volume automated processing
- Monitoring pipelines
- Resource optimization critical
- Quick bias checks needed
- Large-scale model validation

## Advanced Configuration Patterns

### Environment-Specific Overrides

```bash
# Base configuration with environment-specific overrides
glassalpha audit \
  --config configs/example_audit.yaml \
  --override configs/gdpr_compliance.yaml \
  --output production_audit.pdf \
  --strict

# Dynamic configuration for different models
glassalpha audit \
  --config configs/example_audit.yaml \
  --override '{"model": {"type": "lightgbm", "params": {"num_leaves": 31}}}' \
  --output lightgbm_comparison.pdf
```

### Profile Switching

```bash
# Switch between audit profiles for same data
glassalpha audit --config config.yaml --profile basic_compliance --output basic.pdf
glassalpha audit --config config.yaml --profile strict_compliance --output strict.pdf
glassalpha audit --config config.yaml --profile performance --output fast.pdf
```

### Configuration Testing

```bash
# Validate configuration before full run
glassalpha validate --config configs/german_credit.yaml --strict

# Dry run to check configuration
glassalpha audit --config configs/gdpr_compliance.yaml --output test.pdf --dry-run
```

## Best Practices

### Development Workflow

1. **Start with minimal** configuration for initial validation
2. **Use standard** for iterative development and testing
3. **Apply regulatory** for final compliance verification
4. **Optimize performance** for production deployment

### Configuration Management

```bash
# Version control configurations
git add configs/
git commit -m "Add regulatory audit configuration v1.2"
git tag audit-config-v1.2

# Environment-specific configs
configs/
├── environments/
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
├── profiles/
│   ├── basic.yaml
│   ├── standard.yaml
│   └── regulatory.yaml
└── overrides/
    ├── lightgbm.yaml
    └── strict_mode.yaml
```

### Testing Strategy

```bash
# Test all configurations for consistency
for config in configs/*.yaml; do
  echo "Testing $config..."
  glassalpha audit --config "$config" --output "test_$(basename $config .yaml).pdf" --dry-run
done

# Compare results across configurations
glassalpha compare \
  --reports minimal_audit.pdf standard_audit.pdf regulatory_audit.pdf \
  --output comparison_report.pdf
```

## Conclusion

Configuration choice significantly impacts audit outcomes:

- **Execution time** varies 6x between performance and regulatory configurations
- **Report depth** ranges from 8 to 28 pages depending on configuration
- **Compliance value** increases substantially with comprehensive configurations
- **Resource usage** scales with analysis depth and validation level

**Recommendation**: Start with standard configuration and adjust based on specific requirements. Use regulatory configuration for compliance-critical applications and performance configuration for high-volume operations.

The key is matching configuration complexity to your actual needs - over-engineering wastes resources while under-engineering risks missing critical biases or failing compliance requirements.

For specific configuration guidance, see the [Configuration Guide](../getting-started/configuration.md) or consult regulatory experts for compliance-critical applications.
