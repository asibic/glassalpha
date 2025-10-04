# CLI reference

Complete reference for all GlassAlpha command-line interface commands and options.

!!! tip "Quick Links" - **New to GlassAlpha?** → Start with [Quick Start Guide](../getting-started/quickstart.md) - **Need config help?** → See [Configuration Guide](../getting-started/configuration.md) - **Choosing models/explainers?** → Check [Model Selection](model-selection.md) and [Explainer Selection](explainers.md)

## Global options

Available with all commands:

```bash
glassalpha [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]
```

### Global flags

| Flag        | Short | Description               |
| ----------- | ----- | ------------------------- |
| `--version` | `-V`  | Show version and exit     |
| `--verbose` | `-v`  | Enable verbose logging    |
| `--quiet`   | `-q`  | Suppress non-error output |
| `--help`    | `-h`  | Show help message         |

### Examples

```bash
# Show version
glassalpha --version

# Enable verbose logging for any command
glassalpha --verbose audit --config config.yaml --output report.pdf

# Suppress output (errors only)
glassalpha --quiet list
```

## Commands

### models

Show available models and installation options.

```bash
glassalpha models
```

### audit

Generate comprehensive audit reports from ML models.

```bash
glassalpha audit --config CONFIG --output OUTPUT [OPTIONS]
```

#### Required arguments

| Argument         | Type | Description                           |
| ---------------- | ---- | ------------------------------------- |
| `--config`, `-c` | Path | Path to audit configuration YAML file |
| `--output`, `-o` | Path | Path for output PDF report            |

#### Optional arguments

| Argument          | Type   | Default | Description                                                       |
| ----------------- | ------ | ------- | ----------------------------------------------------------------- |
| `--strict`, `-s`  | Flag   | False   | Enable strict mode for regulatory compliance                      |
| `--repro`         | Flag   | False   | Enable deterministic reproduction mode for byte-identical results |
| `--profile`, `-p` | String | None    | Override audit profile from config                                |
| `--override`      | Path   | None    | Additional config file to override settings                       |
| `--dry-run`       | Flag   | False   | Validate configuration without generating report                  |

#### Examples

```bash
# Basic audit generation
glassalpha audit \
  --config configs/german_credit.yaml \
  --output audit_report.pdf

# Regulatory compliance mode
glassalpha audit \
  --config configs/german_credit.yaml \
  --output regulatory_audit.pdf \
  --strict

# Override audit profile
glassalpha audit \
  --config configs/german_credit.yaml \
  --output custom_audit.pdf \
  --profile custom_compliance

# Apply configuration overrides
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --override configs/example_audit.yaml \
  --output modified_audit.pdf

# Validate configuration without running audit
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --output test.pdf \
  --dry-run

# Enable deterministic reproduction for byte-identical results
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --output audit.pdf \
  --repro

# Combine strict and repro modes for maximum compliance
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --output audit.pdf \
  --strict \
  --repro
```

#### Strict mode

Strict mode enforces additional regulatory compliance requirements:

- Explicit random seeds (no defaults allowed)
- Locked data schema (no inference)
- Full manifest generation required
- Deterministic plugin selection verified
- All optional fields become required
- Warnings treated as errors

Use strict mode for regulatory submissions and compliance documentation.

#### Reproduction mode

Reproduction mode (`--repro`) enables deterministic execution for byte-identical results:

- Controls NumPy, pandas, and scikit-learn random states
- Sets thread counts for consistent parallel processing
- Enables strict determinism controls across all libraries
- Automatically uses configuration random seed or defaults to 42

Reproduction mode is essential for regulatory audits that require identical outputs across different execution environments.

#### Output

The audit command produces:

- **PDF Report**: Professional audit document with visualizations
- **Manifest File**: Complete audit trail (same directory as PDF, `.manifest.json` extension)
- **Console Output**: Progress updates and detailed audit summary

**Console Output Format:**
The CLI provides detailed progress feedback and results summary:

```
GlassAlpha Audit Generation
========================================
Loading configuration from: config.yaml
Audit profile: tabular_compliance
Strict mode: ENABLED

Running audit pipeline...
  Loading data and initializing components...
✓ Audit pipeline completed in 4.23s

📊 Audit Summary:
  ✅ Performance metrics: 6 computed
     ✅ accuracy: 75.2%
  ⚖️ Fairness metrics: 12/12 computed
     ⚠️ Bias detected in: gender.demographic_parity
  🔍 Explanations: ✅ Global feature importance
     Most important: checking_account_status (+0.234)
  📋 Dataset: 1,000 samples, 21 features
  🔧 Components: 3 selected
     Model: xgboost

🎉 Audit Report Generated Successfully!
==================================================
📁 Output: /path/to/audit.pdf
📊 Size: 1,247,832 bytes (1.2 MB)
⏱️ Total time: 5.67s
   • Pipeline: 4.23s
   • PDF generation: 1.44s

🛡️ Strict mode: Report meets regulatory compliance requirements

The audit report is ready for review and regulatory submission.
```

### validate

Validate configuration files without running audits.

```bash
glassalpha validate --config CONFIG [OPTIONS]
```

#### Required arguments

| Argument         | Type | Description                            |
| ---------------- | ---- | -------------------------------------- |
| `--config`, `-c` | Path | Path to configuration file to validate |

#### Optional arguments

| Argument          | Type   | Default | Description                         |
| ----------------- | ------ | ------- | ----------------------------------- |
| `--profile`, `-p` | String | None    | Validate against specific profile   |
| `--strict`        | Flag   | False   | Validate for strict mode compliance |

#### Examples

```bash
# Basic validation
glassalpha validate --config audit.yaml

# Validate for specific profile
glassalpha validate \
  --config audit.yaml \
  --profile tabular_compliance

# Check strict mode compliance
glassalpha validate \
  --config audit.yaml \
  --strict

# Validate configuration that overrides profile
glassalpha validate \
  --config audit.yaml \
  --profile eu_ai_act
```

#### Output

Validation provides:

- Configuration parsing results
- Schema compliance verification
- Profile and model type identification
- Warnings for missing optional settings
- Strict mode requirement validation

**Example Output:**

```
Validating configuration: audit.yaml
Profile: tabular_compliance
Model type: xgboost
Strict mode: valid

✓ Configuration is valid

Warning: No random seed specified - results may vary
Warning: No protected attributes - fairness analysis limited
```

### models

Show available models and installation options.

```bash
glassalpha models [OPTIONS]
```

#### Description

Displays which ML models are currently available in your environment and provides installation instructions for missing optional models.

#### Options

None

#### Examples

```bash
# Show available models
glassalpha models

# Output shows:
# ✅ logistic_regression (included in base install)
# ❌ xgboost (install with: pip install 'glassalpha[xgboost]')
# ❌ lightgbm (install with: pip install 'glassalpha[lightgbm]')
```

### list

List available components and system capabilities.

```bash
glassalpha list [COMPONENT_TYPE] [OPTIONS]
```

#### Optional arguments

| Argument                     | Type   | Description                                           |
| ---------------------------- | ------ | ----------------------------------------------------- |
| `COMPONENT_TYPE`             | String | Filter by type: models, explainers, metrics, profiles |
| `--include-enterprise`, `-e` | Flag   | Include enterprise components                         |
| `--verbose`, `-v`            | Flag   | Show component details                                |

#### Examples

```bash
# List all components
glassalpha list

# List only models
glassalpha list models

# List only explainers
glassalpha list explainers

# List only metrics
glassalpha list metrics

# List only audit profiles
glassalpha list profiles

# Include enterprise components
glassalpha list --include-enterprise

# Show detailed information
glassalpha list --verbose

# Combine filters
glassalpha list models --include-enterprise --verbose
```

#### Available Component Types

| Type         | Description         | Examples                                                                        |
| ------------ | ------------------- | ------------------------------------------------------------------------------- |
| `models`     | ML model wrappers   | xgboost, lightgbm, logistic_regression, sklearn_generic, passthrough            |
| `explainers` | Explanation methods | treeshap, kernelshap, noop                                                      |
| `metrics`    | Evaluation metrics  | accuracy, precision, recall, f1, auc_roc, demographic_parity, equal_opportunity |
| `profiles`   | Audit profiles      | tabular_compliance, german_credit_default                                       |

#### Output

The list command shows:

- Registered component names by type
- Component counts and status
- License requirements for enterprise features (when `--include-enterprise` used)

**Example Output:**

```
Available Components
========================================

MODELS:
  - lightgbm
  - logistic_regression
  - passthrough
  - sklearn_generic
  - xgboost

EXPLAINERS:
  - kernelshap
  - noop
  - treeshap

METRICS:
  - accuracy
  - auc_roc
  - demographic_parity
  - equal_opportunity
  - equalized_odds
  - f1
  - precision
  - predictive_parity
  - recall

PROFILES:
  - german_credit_default
  - tabular_compliance
```

## Enterprise commands

### dashboard serve

Start monitoring dashboard (Enterprise only).

```bash
glassalpha dashboard serve [OPTIONS]
```

#### Optional arguments

| Argument       | Type    | Default   | Description      |
| -------------- | ------- | --------- | ---------------- |
| `--port`, `-p` | Integer | 8080      | Port to serve on |
| `--host`, `-h` | String  | localhost | Host to bind to  |

#### Example

```bash
# Start dashboard on default port
glassalpha dashboard serve

# Start on custom port and host
glassalpha dashboard serve --port 9000 --host 0.0.0.0
```

### monitor drift

Monitor model drift over time (Enterprise only).

```bash
glassalpha monitor drift --config CONFIG --baseline BASELINE
```

#### Required arguments

| Argument           | Type | Description                      |
| ------------------ | ---- | -------------------------------- |
| `--config`, `-c`   | Path | Configuration file               |
| `--baseline`, `-b` | Path | Baseline manifest for comparison |

#### Example

```bash
# Monitor drift against baseline
glassalpha monitor drift \
  --config current_config.yaml \
  --baseline baseline_manifest.json
```

## Error handling

GlassAlpha provides clear error messages for common issues:

### Configuration errors

```bash
Configuration error: Missing required field 'data.path'
Validation failed: Invalid audit profile 'nonexistent_profile'
```

### File not found errors

```bash
File 'missing.yaml' does not exist.
Override file 'overrides.yaml' does not exist.
```

### Component errors

```bash
Warning: Model type 'unknown_model' not found in registry
```

### Audit pipeline errors

```bash
❌ Audit pipeline failed: Dataset file 'data/missing.csv' not found
❌ Audit failed: Input contains NaN, infinity or a value too large
```

### Enterprise license errors

```bash
Enterprise feature 'dashboard' requires valid license key
Set GLASSALPHA_LICENSE_KEY environment variable
```

### Reproduction mode warnings

```bash
⚠️ Some determinism controls failed - results may not be fully reproducible
```

## Exit codes

| Code | Meaning                                             |
| ---- | --------------------------------------------------- |
| 0    | Success                                             |
| 1    | General error (configuration, file not found, etc.) |
| 2    | Invalid command line arguments                      |

## Environment variables

| Variable                 | Description                                 | Default            | Usage                                      |
| ------------------------ | ------------------------------------------- | ------------------ | ------------------------------------------ |
| `GLASSALPHA_LICENSE_KEY` | Enterprise license key                      | None               | Required for enterprise features           |
| `GLASSALPHA_LOG_LEVEL`   | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO               | Controls console output verbosity          |
| `GLASSALPHA_CONFIG_DIR`  | Default config directory                    | ~/.glassalpha      | Used for automatic config discovery        |
| `GLASSALPHA_DATA_DIR`    | Default data directory                      | ~/.glassalpha/data | Used for built-in datasets (German Credit) |

## Performance notes

### Typical execution times

- **Small datasets** (< 1,000 rows): 1-3 seconds
- **Medium datasets** (1,000-10,000 rows): 3-10 seconds
- **Large datasets** (10,000+ rows): 10-60 seconds

### Memory requirements

- **Minimum**: 1GB RAM
- **Recommended**: 4GB+ RAM for large datasets
- **PDF Generation**: Additional 500MB temporary space

### Optimization tips

- Use `--dry-run` to validate configurations quickly
- Enable `--quiet` for batch processing scripts
- Use specific component types with `list` for faster startup

## Integration examples

### Batch processing script

```bash
#!/bin/bash
set -e

for config in configs/*.yaml; do
    output="reports/$(basename "$config" .yaml).pdf"
    echo "Processing $config..."

    glassalpha audit \
        --config "$config" \
        --output "$output" \
        --strict \
        --quiet

    echo "Generated $output"
done
```

### CI/CD pipeline

```yaml
# Example GitHub Actions step
- name: Generate Audit Reports
  run: |
    glassalpha validate --config audit.yaml --strict
    glassalpha audit --config audit.yaml --output audit.pdf --strict
```

### Docker usage

```dockerfile
FROM python:3.11-slim
RUN pip install glassalpha
COPY configs/ /app/configs/
WORKDIR /app
CMD ["glassalpha", "audit", "--config", "configs/production.yaml", "--output", "audit.pdf"]
```

This CLI reference covers all available commands and options for GlassAlpha's command-line interface. For configuration file syntax and options, see the [Configuration Guide](../getting-started/configuration.md).
