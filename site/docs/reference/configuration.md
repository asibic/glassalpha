# Configuration Reference

This document provides a complete reference for all GlassAlpha configuration options.

## Configuration File Structure

GlassAlpha uses YAML configuration files with the following top-level sections:

- `audit_profile` - Audit profile selection
- `data` - Data configuration (datasets, paths, features)
- `model` - Model configuration
- `explainers` - Explainer configuration
- `metrics` - Metrics configuration
- `report` - Report generation settings
- `reproducibility` - Reproducibility settings
- `recourse` - Recourse generation settings (advanced)
- `preprocessing` - Data preprocessing options
- `validation` - Validation settings
- `performance` - Performance optimization
- `compliance` - Regulatory compliance settings

## Data Configuration

Configure data sources, datasets, and feature specifications.

### Dataset Specification

**`data.dataset`** - Dataset key from the registry (recommended)

```yaml
data:
  dataset: german_credit # Use built-in German Credit dataset
```

**`data.path`** - Explicit file path (alternative to dataset)

```yaml
data:
  path: "~/.glassalpha/data/german_credit_processed.csv"
```

**`data.fetch`** - Fetch policy for automatic dataset downloading

```yaml
data:
  fetch: if_missing # Options: never | if_missing | always
```

**`data.offline`** - Disable network operations

```yaml
data:
  offline: false # Set to true for air-gapped environments
```

### Feature Configuration

**`data.target_column`** - Name of the target/prediction column

```yaml
data:
  target_column: credit_risk
```

**`data.feature_columns`** - List of feature columns to use

```yaml
data:
  feature_columns:
    - checking_account_status
    - duration_months
    - credit_amount
    - age_years
```

**`data.protected_attributes`** - Sensitive attributes for fairness analysis

```yaml
data:
  protected_attributes:
    - gender
    - age_group
    - foreign_worker
```

### Schema Configuration

**`data.schema_path`** - Path to data schema file (optional)

```yaml
data:
  schema_path: "schemas/german_credit.json"
```

**`data.data_schema`** - Inline schema definition (optional)

```yaml
data:
  data_schema:
    types:
      checking_account_status: categorical
      duration_months: numeric
      credit_amount: numeric
```

## Model Configuration

Configure the machine learning model to audit.

**`model.type`** - Model type identifier

```yaml
model:
  type: xgboost # Options: xgboost, lightgbm, logistic_regression, etc.
```

**`model.path`** - Path to pre-trained model file (optional)

```yaml
model:
  path: "models/german_credit_xgboost.pkl"
```

**`model.params`** - Model hyperparameters

```yaml
model:
  params:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
    objective: "binary:logistic"
```

**`model.calibration`** - Probability calibration settings

```yaml
model:
  calibration:
    method: isotonic # Options: isotonic | sigmoid
    cv: 5
    ensemble: true
```

## Explainer Configuration

Configure explainability methods and their parameters.

**`explainers.strategy`** - Explainer selection strategy

```yaml
explainers:
  strategy: first_compatible # Options: first_compatible | best_score
```

**`explainers.priority`** - Ordered list of explainer preferences

```yaml
explainers:
  priority:
    - treeshap # TreeSHAP (preferred for tree models)
    - kernelshap # KernelSHAP (model-agnostic fallback)
    - noop # No-op fallback
```

**`explainers.config`** - Per-explainer configuration

```yaml
explainers:
  config:
    treeshap:
      max_samples: 1000
      check_additivity: true
    kernelshap:
      n_samples: 500
      background_size: 100
```

## Metrics Configuration

Configure performance and fairness metrics.

**`metrics.performance`** - Performance metrics to compute

```yaml
metrics:
  performance:
    - accuracy
    - precision
    - recall
    - f1
    - auc_roc
```

**`metrics.fairness`** - Fairness metrics to compute

```yaml
metrics:
  fairness:
    - demographic_parity
    - equal_opportunity
    - equalized_odds
    - predictive_parity
```

**`metrics.drift`** - Drift detection metrics (requires reference data)

```yaml
metrics:
  drift:
    - population_stability_index
    - kl_divergence
    - kolmogorov_smirnov
```

## Report Configuration

Configure audit report generation.

**`report.template`** - Report template to use

```yaml
report:
  template: standard_audit
```

**`report.output_format`** - Output format

```yaml
report:
  output_format: pdf # Options: pdf | html | json
```

**`report.include_sections`** - Report sections to include

```yaml
report:
  include_sections:
    - executive_summary
    - data_overview
    - model_performance
    - global_explanations
    - local_explanations
    - fairness_analysis
    - drift_detection
    - audit_manifest
    - regulatory_compliance
```

## Reproducibility Configuration

Configure settings for reproducible audits.

**`reproducibility.random_seed`** - Master random seed

```yaml
reproducibility:
  random_seed: 42
```

**`reproducibility.deterministic`** - Enforce deterministic behavior

```yaml
reproducibility:
  deterministic: true
```

**`reproducibility.capture_environment`** - Capture execution environment

```yaml
reproducibility:
  capture_environment: true
```

**`reproducibility.validate_determinism`** - Validate deterministic execution

```yaml
reproducibility:
  validate_determinism: true
```

## Recourse Configuration (Advanced)

Configure counterfactual explanation generation.

**`recourse.enabled`** - Enable recourse generation

```yaml
recourse:
  enabled: true
```

**`recourse.immutable_features`** - Features that cannot be changed

```yaml
recourse:
  immutable_features:
    - age_years
    - gender
    - personal_status_sex
```

**`recourse.monotonic_constraints`** - Directional constraints

```yaml
recourse:
  monotonic_constraints:
    credit_amount: "decrease_preferred"
    duration_months: "decrease_preferred"
    savings_account: "increase_only"
```

## Preprocessing Configuration

Configure data preprocessing options.

**`preprocessing.handle_missing`** - Handle missing values

```yaml
preprocessing:
  handle_missing: true
  missing_strategy: "median"
```

**`preprocessing.scale_features`** - Feature scaling

```yaml
preprocessing:
  scale_features: false
  scaling_method: "standard"
```

**`preprocessing.categorical_encoding`** - Categorical encoding

```yaml
preprocessing:
  categorical_encoding: "label"
```

## Validation Configuration

Configure cross-validation and testing settings.

**`validation.cv_folds`** - Cross-validation folds

```yaml
validation:
  cv_folds: 5
  cv_scoring: "roc_auc"
```

**`validation.test_size`** - Train/test split ratio

```yaml
validation:
  test_size: 0.2
  stratify_split: true
```

## Performance Configuration

Configure performance optimization settings.

**`performance.n_jobs`** - Parallel processing

```yaml
performance:
  n_jobs: -1 # Use all available cores
```

**`performance.low_memory_mode`** - Memory optimization

```yaml
performance:
  low_memory_mode: false
```

## Compliance Configuration

Configure regulatory compliance settings.

**`compliance.frameworks`** - Regulatory frameworks to check

```yaml
compliance:
  frameworks:
    - gdpr
    - ecoa
    - fcra
    - eu_ai_act
```

**`compliance.fairness_thresholds`** - Fairness tolerance thresholds

```yaml
compliance:
  fairness_thresholds:
    demographic_parity: 0.05
    equal_opportunity: 0.05
    equalized_odds: 0.05
```

## Environment-Specific Overrides

Configure different settings for different environments.

**`development`** - Development environment settings

```yaml
development:
  sample_size: 500
  cv_folds: 3
  fairness_thresholds:
    demographic_parity: 0.10
```

**`production`** - Production environment settings

```yaml
production:
  sample_size: null # Use full dataset
  cv_folds: 10
  fairness_thresholds:
    demographic_parity: 0.02
  approval_required: true
```

## Complete Example

Here's a complete configuration example:

```yaml
# Audit profile
audit_profile: tabular_compliance

# Data configuration
data:
  dataset: german_credit
  fetch: if_missing
  offline: false
  target_column: credit_risk
  feature_columns:
    - checking_account_status
    - duration_months
    - credit_amount
    - savings_account
    - employment_duration
    - age_years
    - gender
  protected_attributes:
    - gender

# Model configuration
model:
  type: xgboost
  params:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
    objective: "binary:logistic"

# Explainer configuration
explainers:
  strategy: first_compatible
  priority:
    - treeshap
    - kernelshap
  config:
    treeshap:
      max_samples: 1000
    kernelshap:
      n_samples: 500

# Metrics configuration
metrics:
  performance:
    - accuracy
    - precision
    - recall
    - f1
    - auc_roc
  fairness:
    - demographic_parity
    - equal_opportunity

# Report configuration
report:
  template: standard_audit
  output_format: pdf
  include_sections:
    - executive_summary
    - data_overview
    - model_performance
    - fairness_analysis

# Reproducibility
reproducibility:
  random_seed: 42
  deterministic: true
  capture_environment: true
  validate_determinism: true

# Validation
validation:
  cv_folds: 5
  test_size: 0.2
  stratify_split: true

# Performance
performance:
  n_jobs: -1
  verbose: true
```

## Configuration Validation

GlassAlpha validates configurations for:

- **Type Safety**: All fields have correct types
- **Required Fields**: Essential configuration is present
- **Valid Values**: Enum values are within allowed ranges
- **Consistency**: Related fields are compatible
- **Security**: Paths are safe and permissions are appropriate

Invalid configurations will produce clear error messages indicating what needs to be fixed.
