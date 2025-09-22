# Configuration Design Reference

!!! warning "Design Documentation"
    This document describes the **planned** configuration schema for GlassAlpha. These features are not yet implemented. This serves as a design specification for development.

## Planned Configuration Structure

The configuration system will use YAML files with four main sections:

```yaml
model:      # Model type and training parameters
data:       # Dataset paths and preprocessing  
audit:      # Audit scope and fairness metrics
reproducibility:  # Determinism and tracking
```

## Design Goals

This configuration schema is designed to:

1. **Enable reproducibility** - All parameters affecting output must be configurable
2. **Support compliance** - Include settings required for regulatory documentation
3. **Maintain simplicity** - Common use cases should work with minimal configuration
4. **Allow extensibility** - Future features can be added without breaking changes

## Model Configuration (Planned)

### Target Model Support

```yaml
model:
  type: xgboost | lightgbm | logistic_regression
  target_column: string  # Required
  params: {}  # Model-specific parameters
```

### XGBoost Parameters

```yaml
model:
  type: xgboost
  params:
    max_depth: 6                # Tree depth (default: 6)
    learning_rate: 0.1          # Boosting rate (default: 0.3) 
    n_estimators: 100           # Number of trees (default: 100)
    subsample: 1.0              # Sample fraction (default: 1.0)
    colsample_bytree: 1.0       # Feature fraction (default: 1.0)
    reg_alpha: 0.0              # L1 regularization (default: 0)
    reg_lambda: 1.0             # L2 regularization (default: 1)
```

### LightGBM Parameters  

```yaml
model:
  type: lightgbm
  params:
    num_leaves: 31              # Max leaves per tree (default: 31)
    learning_rate: 0.1          # Boosting rate (default: 0.1)
    feature_fraction: 0.9       # Feature sampling (default: 1.0)
    bagging_fraction: 1.0       # Row sampling (default: 1.0)
    bagging_freq: 0             # Bagging frequency (default: 0)
    min_data_in_leaf: 20        # Min samples per leaf (default: 20)
```

### Logistic Regression Parameters

```yaml
model:
  type: logistic_regression  
  params:
    C: 1.0                      # Inverse regularization (default: 1.0)
    penalty: l2                 # l1, l2, or elasticnet (default: l2)
    solver: lbfgs               # Optimization algorithm (default: lbfgs)
    max_iter: 100               # Max iterations (default: 100)
```

## Data Configuration

```yaml
data:
  train_path: path/to/train.csv       # Required
  test_path: path/to/test.csv         # Optional (splits train if missing)
  target_column: string               # Required  
  feature_columns: [list]             # Optional (uses all if missing)
  categorical_features: [list]        # Optional
  drop_columns: [list]                # Optional
  
  # Data preprocessing
  handle_missing: drop | impute       # Default: drop
  imputation_strategy: mean | median | mode  # Default: mean
  scaling: standard | minmax | robust | none  # Default: none
  
  # Train/test split (if test_path not provided)
  test_size: 0.2                      # Default: 0.2
  stratify: true                      # Default: true
```

## Audit Configuration

```yaml
audit:
  # Protected attributes for fairness analysis
  protected_attributes:
    - attribute_name:
        column: column_name
        groups: [group1, group2, ...]   # Explicit groups
        # OR
        bins: [0, 25, 50, 100]         # For continuous variables
        
  # Fairness metrics to compute
  fairness_metrics:
    - demographic_parity              # P(Y=1|A=0) = P(Y=1|A=1) 
    - equalized_odds                  # TPR and FPR equal across groups
    - equal_opportunity               # TPR equal across groups
    - predictive_parity               # PPV equal across groups
    - statistical_parity             # Same as demographic_parity
    
  # Thresholds
  disparate_impact_threshold: 0.8     # 80% rule (default: 0.8)
  confidence_level: 0.95              # Statistical confidence (default: 0.95)
  
  # Report sections to include
  include_sections:
    - executive_summary               # Always included
    - data_analysis                   # Dataset statistics
    - model_performance               # Accuracy metrics
    - fairness_analysis               # Bias detection
    - explanations                    # SHAP values
    - reproducibility_manifest        # Tracking info
```

### Protected Attributes Examples

**Categorical attributes:**
```yaml
protected_attributes:
  - gender:
      column: sex
      groups: ["M", "F"]
  - race:
      column: race
      groups: ["White", "Black", "Hispanic", "Asian", "Other"]
```

**Continuous attributes (age binning):**
```yaml
protected_attributes:
  - age_group:
      column: age  
      bins: [0, 25, 35, 50, 65, 100]
      labels: ["<25", "25-35", "35-50", "50-65", "65+"]
```

**Multi-level attributes:**
```yaml
protected_attributes:
  - education_level:
      column: education
      groups: 
        low: ["Some-school", "HS-grad"]
        medium: ["Some-college", "Assoc-voc", "Assoc-acdm"]  
        high: ["Bachelors", "Masters", "Doctorate"]
```

## Explainability Configuration

```yaml
explainability:
  # SHAP analysis
  shap_values: true                   # Compute SHAP values (default: true)
  shap_sample_size: 1000              # Sample for SHAP computation
  
  # Feature importance
  feature_importance: true            # Global importance (default: true)
  importance_method: gain | split     # XGBoost only
  
  # Individual explanations  
  waterfall_plots: 10                 # Number of examples (default: 0)
  force_plots: 5                      # Interactive plots (default: 0)
  
  # Cohort analysis
  cohort_analysis:
    - protected_attribute: gender
      top_features: 5                 # Features to highlight
      sample_size: 100                # Sample per group
```

## Reproducibility Configuration

```yaml
reproducibility:
  # Randomness control
  random_seed: 42                     # Master seed (default: 42)
  numpy_seed: 42                      # NumPy seed (optional)
  
  # Tracking and versioning
  track_git: true                     # Include git commit (default: true)
  track_data_hash: true               # Hash input data (default: true)
  track_config_hash: true             # Hash configuration (default: true)
  
  # Output manifest
  save_manifest: true                 # Save run metadata (default: true)
  manifest_path: audit_manifest.json  # Manifest location (optional)
```

## Output Configuration

```yaml
output:
  # PDF generation
  pdf_template: default | minimal | detailed  # Report template
  include_plots: true                 # Embed visualizations  
  plot_dpi: 300                       # Image resolution
  
  # Additional outputs
  save_model: true                    # Save trained model
  save_predictions: true              # Save test predictions
  save_shap_values: true              # Save SHAP explanations
  
  # File paths (optional)
  model_path: models/audit_model.pkl
  predictions_path: outputs/predictions.csv
  shap_path: outputs/shap_values.csv
```

## Complete Example

```yaml
# complete_audit_config.yaml
model:
  type: xgboost
  target_column: target
  params:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100

data:
  train_path: data/train.csv
  test_path: data/test.csv
  categorical_features: [category1, category2]
  handle_missing: impute
  scaling: standard

audit:
  protected_attributes:
    - gender:
        column: sex
        groups: ["M", "F"]
    - age_group:
        column: age
        bins: [0, 30, 50, 100]
        
  fairness_metrics:
    - demographic_parity
    - equalized_odds
    
  disparate_impact_threshold: 0.8
  confidence_level: 0.95

explainability:
  shap_values: true
  waterfall_plots: 10
  
reproducibility:
  random_seed: 42
  track_git: true
  track_data_hash: true
```

## Usage

```bash
glassalpha audit --config complete_audit_config.yaml --out audit_report.pdf
```

## Validation

GlassAlpha validates configuration files at runtime:

- **Required fields**: Missing required fields cause immediate failure
- **Type checking**: Values must match expected types  
- **Value ranges**: Numeric parameters validated against allowed ranges
- **Model compatibility**: Parameters checked against model requirements

Common validation errors:
```
❌ ConfigurationError: 'target_column' is required
❌ ValidationError: learning_rate must be between 0 and 1  
❌ ModelError: 'penalty=l1' not supported with solver='lbfgs'
```

## Next Steps

- [Hello Audit Tutorial](quickstart.md) - Step-by-step walkthrough
- [German Credit Example](../examples/german-credit-audit.md) - Complete example
- [Adult Income Example](../examples/adult-income-audit.md) - Bias detection case study
