# CLI reference

Complete reference for all GlassAlpha command-line interface commands and options.

!!! tip "Quick Links"

- **New to GlassAlpha?** → Start with [Quick Start Guide](../getting-started/quickstart.md)
- **Need config help?** → See [Configuration Guide](../getting-started/configuration.md)
- **Choosing models/explainers?** → Check [Model Selection](model-selection.md) and [Explainer Selection](explainers.md)

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

### reasons

Generate ECOA-compliant reason codes for adverse action notices.

```bash
glassalpha reasons --model MODEL --data DATA --instance INSTANCE [OPTIONS]
```

#### Required arguments

| Argument     | Short | Type | Description                                |
| ------------ | ----- | ---- | ------------------------------------------ |
| `--model`    | `-m`  | Path | Path to trained model file (.pkl, .joblib) |
| `--data`     | `-d`  | Path | Path to test data file (CSV)               |
| `--instance` | `-i`  | Int  | Row index of instance to explain (0-based) |

#### Optional arguments

| Argument      | Short | Type   | Default | Description                              |
| ------------- | ----- | ------ | ------- | ---------------------------------------- |
| `--config`    | `-c`  | Path   | None    | Path to reason codes configuration YAML  |
| `--output`    | `-o`  | Path   | stdout  | Path for output notice file              |
| `--threshold` | `-t`  | Float  | 0.5     | Decision threshold for approved/denied   |
| `--top-n`     | `-n`  | Int    | 4       | Number of reason codes (ECOA typical: 4) |
| `--format`    | `-f`  | String | text    | Output format: 'text' (ECOA) or 'json'   |

#### Examples

```bash
# Generate reason codes for instance 42
glassalpha reasons \
  --model models/german_credit.pkl \
  --data data/test.csv \
  --instance 42 \
  --output notices/instance_42.txt

# With custom config
glassalpha reasons \
  -m model.pkl \
  -d test.csv \
  -i 10 \
  -c configs/reason_codes.yaml

# JSON output
glassalpha reasons \
  --model model.pkl \
  --data test.csv \
  --instance 5 \
  --format json

# Custom threshold and top-N
glassalpha reasons \
  --model model.pkl \
  --data test.csv \
  --instance 0 \
  --threshold 0.6 \
  --top-n 3
```

#### Output

The reasons command produces:

- **Text Format (default)**: ECOA-compliant adverse action notice with:

  - Instance ID and decision (APPROVED/DENIED)
  - Predicted score and threshold
  - Top-N ranked reason codes
  - ECOA-required rights disclosure
  - Audit trail (timestamp, model hash, seed)

- **JSON Format** (`--format json`): Structured data with:
  - All reason code details (rank, feature, contribution, value)
  - Decision metadata (prediction, threshold, decision)
  - Excluded protected attributes
  - Complete audit trail

**Example Text Output:**

```
ADVERSE ACTION NOTICE
Equal Credit Opportunity Act (ECOA) Disclosure

Example Financial Institution
Date: 2025-01-15T10:30:00+00:00
Application ID: 42

DECISION: DENIED
Predicted Score: 35.0%

PRINCIPAL REASONS FOR ADVERSE ACTION:

1. Debt: Value of 5000 negatively impacted the decision
2. Duration: Value of 24 negatively impacted the decision
3. Credit History: Value of 2 negatively impacted the decision
4. Savings: Value of 1000 negatively impacted the decision

IMPORTANT RIGHTS UNDER FEDERAL LAW:
[ECOA disclosure text...]
```

#### Notes

- Requires TreeSHAP-compatible model (XGBoost, LightGBM, RandomForest)
- Protected attributes automatically excluded (age, gender, race, etc.)
- Output is deterministic with fixed seed (reproducible)
- See [Reason Codes Guide](../guides/reason-codes.md) for details

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

## Preprocessing commands

The `prep` command group provides tools for managing and validating preprocessing artifacts. See the [Preprocessing Verification Guide](../guides/preprocessing.md) for complete documentation.

### prep hash

Compute verification hashes for a preprocessing artifact.

```bash
glassalpha prep hash ARTIFACT_PATH [OPTIONS]
```

#### Required arguments

| Argument        | Type | Description                         |
| --------------- | ---- | ----------------------------------- |
| `ARTIFACT_PATH` | Path | Path to preprocessing artifact file |

#### Optional arguments

| Argument         | Type | Default | Description                                      |
| ---------------- | ---- | ------- | ------------------------------------------------ |
| `--params`, `-p` | Flag | False   | Also compute params hash and show config snippet |
| `--output`, `-o` | Path | None    | Save hashes to JSON file                         |

#### Examples

```bash
# Compute file hash only
glassalpha prep hash artifacts/preprocessor.joblib

# Compute both hashes with config snippet
glassalpha prep hash artifacts/preprocessor.joblib --params

# Save hashes to file
glassalpha prep hash artifacts/preprocessor.joblib --params --output hashes.json
```

#### Output Example

```
✓ File hash:   sha256:9373ae67dbcd5c1558fa4ba7a727e05575f9421358a8604a1d0eb0da80385a26
✓ Params hash: sha256:cf5e71ee7e1733ca6d857b7f21df94a41f29ebeaaec9c6d46783336757cfcf37

Config snippet (copy to your audit config):
preprocessing:
  mode: artifact
  artifact_path: artifacts/preprocessor.joblib
  expected_file_hash: "sha256:9373ae67..."
  expected_params_hash: "sha256:cf5e71ee..."
  expected_sparse: false
  fail_on_mismatch: true
```

### prep inspect

Inspect a preprocessing artifact and display its manifest.

```bash
glassalpha prep inspect ARTIFACT_PATH [OPTIONS]
```

#### Required arguments

| Argument        | Type | Description                         |
| --------------- | ---- | ----------------------------------- |
| `ARTIFACT_PATH` | Path | Path to preprocessing artifact file |

#### Optional arguments

| Argument          | Type | Default | Description                        |
| ----------------- | ---- | ------- | ---------------------------------- |
| `--verbose`, `-v` | Flag | False   | Show detailed component parameters |
| `--output`, `-o`  | Path | None    | Save manifest to JSON file         |

#### Examples

```bash
# Basic inspection
glassalpha prep inspect artifacts/preprocessor.joblib

# Detailed inspection with all parameters
glassalpha prep inspect artifacts/preprocessor.joblib --verbose

# Save manifest to file
glassalpha prep inspect artifacts/preprocessor.joblib --output manifest.json
```

#### Output Example

```
Preprocessing Artifact Manifest
================================================================================

Artifact Information:
  Path: artifacts/german_credit_preprocessor.joblib
  Type: sklearn.compose._column_transformer.ColumnTransformer
  Components: 2

Component 1: numeric_features
  Class: sklearn.preprocessing._data.StandardScaler
  Columns: 7 numeric features
  Learned Parameters:
    - mean_: [38.2, 2973.5, ...]
    - scale_: [11.1, 2938.6, ...]

Component 2: categorical_features
  Class: sklearn.preprocessing._encoders.OneHotEncoder
  Columns: 13 categorical features
  Settings:
    - handle_unknown: ignore
    - drop: first
    - sparse_output: False
  Categories: 62 unique values across all columns
```

### prep validate

Validate a preprocessing artifact against expected properties.

```bash
glassalpha prep validate ARTIFACT_PATH [OPTIONS]
```

#### Required arguments

| Argument        | Type | Description                         |
| --------------- | ---- | ----------------------------------- |
| `ARTIFACT_PATH` | Path | Path to preprocessing artifact file |

#### Optional arguments

| Argument              | Type    | Default | Description                            |
| --------------------- | ------- | ------- | -------------------------------------- |
| `--file-hash`, `-f`   | String  | None    | Expected file hash for verification    |
| `--params-hash`, `-p` | String  | None    | Expected params hash for verification  |
| `--check-classes`     | Flag    | True    | Validate allowed transformer classes   |
| `--check-versions`    | Flag    | True    | Validate runtime version compatibility |
| `--check-sparsity`    | Flag    | False   | Validate expected output sparsity      |
| `--expected-sparse`   | Boolean | None    | Expected sparsity (True/False)         |

#### Examples

```bash
# Full validation with expected hashes
glassalpha prep validate artifacts/preprocessor.joblib \
  --file-hash sha256:9373ae67... \
  --params-hash sha256:cf5e71ee...

# Quick validation (classes + versions only)
glassalpha prep validate artifacts/preprocessor.joblib

# Skip version checking
glassalpha prep validate artifacts/preprocessor.joblib --no-check-versions

# Validate sparsity
glassalpha prep validate artifacts/preprocessor.joblib \
  --check-sparsity \
  --expected-sparse false
```

#### Output Example

```
Validating artifact: artifacts/preprocessor.joblib

1. Verifying file hash...
   ✓ File hash matches

2. Loading artifact...
   ✓ Artifact loaded successfully

3. Validating transformer classes...
   ✓ All classes are allowed

4. Verifying params hash...
   ✓ Params hash matches

5. Checking runtime version compatibility...
   ✓ Runtime versions compatible

============================================================
✓ VALIDATION PASSED
============================================================
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
