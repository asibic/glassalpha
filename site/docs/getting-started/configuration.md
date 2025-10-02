# Configuration guide

Complete guide to configuring GlassAlpha for different use cases, models, and compliance requirements.

## Overview

GlassAlpha uses YAML configuration files to define every aspect of the audit process. Configuration files are policy-as-code, enabling version control, review processes, and reproducible audits.

### Basic structure

```yaml
# Required: Audit profile determines component selection
audit_profile: tabular_compliance

# Required: Reproducibility settings
reproducibility:
  random_seed: 42

# Required: Data configuration
data:
  path: data/my_dataset.csv
  target_column: outcome

# Required: Model configuration
model:
  type: xgboost

# Optional: Additional sections
explainers: { ... }
metrics: { ... }
report: { ... }
```

## Core configuration sections

### Audit profile

The audit profile determines which components are available and what validations are enforced.

```yaml
# Determines the audit context and available components
audit_profile: tabular_compliance # Currently supported profile
```

**Available Profiles:**

- `tabular_compliance` - Standard tabular ML compliance audit
- `german_credit_default` - German Credit dataset specific profile

### Reproducibility settings

Ensures deterministic, reproducible audit results.

```yaml
reproducibility:
  # Master random seed (required in strict mode)
  random_seed: 42

  # Optional: Advanced reproducibility settings
  deterministic: true # Enforce deterministic behavior
  capture_environment: true # Capture system information
  validate_determinism: true # Verify reproducibility
```

**Best Practices:**

- Always set `random_seed` for reproducible results
- Use the same seed for comparative audits
- Document seed values in audit reports

### Data configuration

Defines the dataset and feature structure for the audit.

```yaml
data:
  # Required: Path to dataset
  path: data/my_dataset.csv

  # Required: Target column name
  target_column: outcome

  # Optional: Explicit feature columns
  feature_columns:
    - feature1
    - feature2
    - feature3

  # Optional: Protected attributes for fairness analysis
  protected_attributes:
    - gender
    - age_group
    - ethnicity
```

**Supported Data Formats:**

- CSV (`.csv`)
- Parquet (`.parquet`)
- Feather (`.feather`)
- Pickle (`.pkl`)

**Feature Selection:**

- If `feature_columns` not specified, uses all columns except target
- Protected attributes should be included in features for bias analysis
- Features are automatically preprocessed based on data type

### Model configuration

Specifies the ML model to audit and its parameters.

```yaml
model:
  # Required: Model type (triggers appropriate wrapper)
  type: logistic_regression # Default baseline model (always available)

  # Optional: Allow fallback to baseline model if requested model unavailable
  allow_fallback: true # Default: true

  # Optional: Pre-trained model path
  path: models/my_model.pkl

  # Optional: Model parameters (for training)
  params:
    random_state: 42
    max_iter: 1000
```

**Supported Model Types:**

- `logistic_regression` - Scikit-learn LogisticRegression (baseline, always available)
- `xgboost` - XGBoost gradient boosting (optional, requires `pip install 'glassalpha[xgboost]'`)
- `lightgbm` - LightGBM gradient boosting (optional, requires `pip install 'glassalpha[lightgbm]'`)
- `sklearn_generic` - Generic scikit-learn models (baseline, always available)

**Optional Dependencies:**

GlassAlpha uses optional dependencies to keep the core installation lightweight. If you request a model that's not installed, GlassAlpha will:

1. **Automatically fall back** to `logistic_regression` if `allow_fallback: true` (default)
2. **Show clear installation instructions** for the requested model
3. **Fail gracefully** if `allow_fallback: false` and model unavailable

Example fallback behavior:

```bash
glassalpha audit --config my_config.yaml --output report.pdf
# If XGBoost requested but not installed:
# "Model 'xgboost' not available. Falling back to 'logistic_regression'.
#  To enable 'xgboost', run: pip install 'glassalpha[xgboost]'"
```

**Model Loading vs Training:**

- If `path` exists: loads pre-trained model
- If `path` missing: trains new model with `params`
- Parameters are passed to the underlying library

## Explainer configuration

Controls how model predictions are explained and interpreted.

```yaml
explainers:
  # Required: Selection strategy
  strategy: first_compatible

  # Required: Priority order (deterministic selection)
  priority:
    - treeshap # First choice for tree models
    - kernelshap # Fallback for any model

  # Optional: Explainer-specific configuration
  config:
    treeshap:
      max_samples: 1000 # Samples for SHAP computation
      check_additivity: true # Verify SHAP properties

    kernelshap:
      n_samples: 500 # Model evaluations
      background_size: 100 # Background dataset size
```

**Available Explainers:**

- `treeshap` - Exact SHAP values for tree models (XGBoost, LightGBM)
- `kernelshap` - Model-agnostic SHAP approximation
- `noop` - No-op placeholder (for testing)

**Selection Strategies:**

- `first_compatible` - Use first explainer compatible with model
- `best_available` - Select highest-priority compatible explainer

## Metrics configuration

Defines which performance and fairness metrics to compute.

```yaml
metrics:
  # Performance evaluation metrics
  performance:
    metrics:
      - accuracy
      - precision
      - recall
      - f1
      - auc_roc
      - classification_report

  # Fairness and bias detection metrics
  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity
      - equalized_odds
      - predictive_parity

    # Optional: Bias tolerance thresholds
    config:
      demographic_parity:
        threshold: 0.05 # 5% maximum group difference

  # Optional: Data drift detection metrics
  drift:
    metrics:
      - population_stability_index
      - kl_divergence
      - kolmogorov_smirnov
```

**Performance Metrics:**

- `accuracy` - Overall classification accuracy
- `precision` - Positive predictive value
- `recall` - True positive rate (sensitivity)
- `f1` - Harmonic mean of precision and recall
- `auc_roc` - Area under ROC curve
- `classification_report` - Comprehensive per-class metrics

**Fairness Metrics:**

- `demographic_parity` - Equal positive prediction rates across groups
- `equal_opportunity` - Equal true positive rates across groups
- `equalized_odds` - Equal TPR and FPR across groups
- `predictive_parity` - Equal precision across groups

## Report configuration

Controls the format and content of generated audit reports.

```yaml
report:
  # Report template (determines structure and styling)
  template: standard_audit

  # Output format
  output_format: pdf

  # Optional: Report sections to include
  include_sections:
    - executive_summary
    - data_overview
    - model_performance
    - global_explanations
    - local_explanations
    - fairness_analysis
    - audit_manifest
    - regulatory_compliance

  # Optional: Report styling
  styling:
    color_scheme: professional
    page_size: A4
    margins: standard
    compliance_statement: true
```

**Available Templates:**

- `standard_audit` - Comprehensive audit report with all sections

**Styling Options:**

- `color_scheme`: professional, minimal, colorful
- `page_size`: A4, Letter, Legal
- `margins`: standard, narrow, wide

## Advanced configuration

### Strict mode

Enforces additional regulatory compliance requirements.

```yaml
# Enable via CLI: --strict
# Or in configuration:
strict_mode: true
```

**Strict Mode Requirements:**

- Explicit random seeds (no defaults)
- Complete data schema specification
- Full manifest generation
- Deterministic component selection
- All optional validations enabled

### Manifest configuration

Controls audit trail generation and completeness.

```yaml
manifest:
  enabled: true # Generate audit manifest
  include_git_sha: true # Include Git commit information
  include_config_hash: true # Include configuration integrity hash
  include_data_hash: true # Include dataset integrity hash
  track_component_selection: true # Track selected components
  include_execution_info: true # Include timing and environment
```

### Preprocessing options

Controls data preprocessing before model training/evaluation.

```yaml
preprocessing:
  handle_missing: true # Handle missing values
  missing_strategy: median # median, mode, drop
  scale_features: false # Feature scaling (not needed for trees)
  scaling_method: standard # standard, minmax, robust
  categorical_encoding: label # label, onehot, target
  feature_selection: false # Enable feature selection
  selection_method: mutual_info # mutual_info, correlation
  max_features: 20 # Maximum features to select
```

### Validation configuration

Controls model evaluation and statistical testing.

```yaml
validation:
  cv_folds: 5 # Cross-validation folds
  cv_scoring: roc_auc # Scoring metric for CV
  test_size: 0.2 # Train/test split ratio
  stratify_split: true # Stratify split by target
  bootstrap_samples: 1000 # Bootstrap samples for confidence intervals
  confidence_level: 0.95 # Statistical confidence level
```

### Performance optimization

Controls computational performance and resource usage.

```yaml
performance:
  n_jobs: -1 # Parallel processing (-1 = all cores)
  low_memory_mode: false # Optimize for memory usage
  verbose: true # Enable progress reporting
  progress_bar: true # Show progress bars
```

## Configuration examples

### Basic German Credit audit

```yaml
audit_profile: german_credit_default

reproducibility:
  random_seed: 42

data:
  path: ~/.glassalpha/data/german_credit_processed.csv
  target_column: credit_risk
  protected_attributes:
    - gender
    - age_group

model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100

explainers:
  strategy: first_compatible
  priority: [treeshap, kernelshap]

metrics:
  performance:
    metrics: [accuracy, precision, recall, f1, auc_roc]
  fairness:
    metrics: [demographic_parity, equal_opportunity]
```

### Enterprise compliance configuration

```yaml
audit_profile: tabular_compliance
strict_mode: true

reproducibility:
  random_seed: 42
  deterministic: true
  capture_environment: true

data:
  path: data/production_dataset.csv
  target_column: decision
  feature_columns:
    - income
    - employment_length
    - debt_to_income
    - credit_score
  protected_attributes:
    - race
    - gender
    - age_group

model:
  type: lightgbm
  params:
    objective: binary
    metric: auc
    num_leaves: 31
    feature_fraction: 0.9
    bagging_fraction: 0.8
    bagging_freq: 5

explainers:
  strategy: first_compatible
  priority: [treeshap, kernelshap]
  config:
    treeshap:
      max_samples: 10000
      check_additivity: true

metrics:
  performance:
    metrics: [accuracy, precision, recall, f1, auc_roc, classification_report]
  fairness:
    metrics:
      [demographic_parity, equal_opportunity, equalized_odds, predictive_parity]
    config:
      demographic_parity:
        threshold: 0.02 # Stricter threshold for production
      equal_opportunity:
        threshold: 0.02

manifest:
  enabled: true
  include_git_sha: true
  include_config_hash: true
  include_data_hash: true
  track_component_selection: true

report:
  template: standard_audit
  output_format: pdf
  styling:
    color_scheme: professional
    compliance_statement: true

compliance:
  frameworks: [gdpr, ecoa, fcra]
  fairness_thresholds:
    demographic_parity: 0.02
    equal_opportunity: 0.02
```

### Custom model configuration

```yaml
audit_profile: tabular_compliance

reproducibility:
  random_seed: 123

data:
  path: data/custom_dataset.csv
  target_column: target
  feature_columns:
    - numerical_feature_1
    - numerical_feature_2
    - categorical_feature_1
    - categorical_feature_2
  protected_attributes:
    - protected_attribute_1

model:
  type: logistic_regression
  params:
    C: 1.0
    penalty: l2
    solver: lbfgs
    max_iter: 1000

explainers:
  strategy: first_compatible
  priority: [kernelshap] # Use KernelSHAP for linear models
  config:
    kernelshap:
      n_samples: 1000
      background_size: 500

metrics:
  performance:
    metrics: [accuracy, precision, recall, f1]
  fairness:
    metrics: [demographic_parity]

preprocessing:
  handle_missing: true
  missing_strategy: median
  scale_features: true # Important for linear models
  scaling_method: standard
  categorical_encoding: onehot

validation:
  cv_folds: 10
  test_size: 0.3
  stratify_split: true
```

## Configuration best practices

### Reproducibility

1. **Always set random seeds** for deterministic results
2. **Use version control** for configuration files
3. **Document configuration changes** in commit messages
4. **Enable manifest generation** for complete audit trails

### Performance

1. **Use appropriate model types** for your data size and complexity
2. **Adjust sample sizes** for explainers based on dataset size
3. **Enable parallel processing** (`n_jobs: -1`) for faster computation
4. **Use appropriate metrics** - don't compute unnecessary evaluations

### Compliance

1. **Enable strict mode** for regulatory submissions
2. **Set appropriate bias thresholds** for your use case and jurisdiction
3. **Include all relevant protected attributes** in fairness analysis
4. **Document configuration rationale** for audit review

### Security

1. **Use relative paths** or environment variables for file locations
2. **Don't embed sensitive data** in configuration files
3. **Review configurations** before committing to version control
4. **Use appropriate access controls** for configuration repositories

## Troubleshooting configuration issues

### Common configuration errors

**Missing Required Fields:**

```yaml
# Error: Missing required field 'data.target_column'
data:
  path: data.csv
  # target_column: missing!
```

**Invalid Model Type:**

```yaml
# Error: Model type 'invalid_model' not found in registry
model:
  type: invalid_model # Should be: xgboost, lightgbm, etc.
```

**Incompatible Components:**

```yaml
# Warning: No compatible explainers for model type
explainers:
  priority: [treeshap] # TreeSHAP only works with tree models
model:
  type: logistic_regression # Linear model - use kernelshap instead
```

### Validation commands

```bash
# Validate configuration before running audit
glassalpha validate --config my_config.yaml

# Check strict mode compliance
glassalpha validate --config my_config.yaml --strict

# List available components
glassalpha list
```

### Configuration schema validation

GlassAlpha uses Pydantic for configuration validation with detailed error messages:

```bash
ValidationError: 2 validation errors for AuditConfig
data.target_column
  field required (type=value_error.missing)
model.type
  ensure this value has at least 1 characters (type=value_error.any_str.min_length; limit_value=1)
```

## Schema reference

For detailed technical information about GlassAlpha's architecture and interfaces, see the [Trust & Deployment Guide](../reference/trust-deployment.md).

This configuration guide provides the foundation for creating effective, compliant audit configurations. Start with the provided examples and customize based on your specific requirements, data characteristics, and regulatory context.
