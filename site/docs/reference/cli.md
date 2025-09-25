# CLI Reference

Complete reference for all GlassAlpha command-line interface commands and options.

## Global Options

Available with all commands:

```bash
glassalpha [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]
```

### Global Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--version` | `-V` | Show version and exit |
| `--verbose` | `-v` | Enable verbose logging |
| `--quiet` | `-q` | Suppress non-error output |
| `--help` | `-h` | Show help message |

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

### audit

Generate comprehensive audit reports from ML models.

```bash
glassalpha audit --config CONFIG --output OUTPUT [OPTIONS]
```

#### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--config`, `-c` | Path | Path to audit configuration YAML file |
| `--output`, `-o` | Path | Path for output PDF report |

#### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--strict`, `-s` | Flag | False | Enable strict mode for regulatory compliance |
| `--profile`, `-p` | String | None | Override audit profile from config |
| `--override` | Path | None | Additional config file to override settings |
| `--dry-run` | Flag | False | Validate configuration without generating report |

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
  --config configs/base.yaml \
  --override configs/overrides.yaml \
  --output modified_audit.pdf

# Validate configuration without running audit
glassalpha audit \
  --config configs/test.yaml \
  --output test.pdf \
  --dry-run
```

#### Strict Mode

Strict mode enforces additional regulatory compliance requirements:

- Explicit random seeds (no defaults allowed)
- Locked data schema (no inference)
- Full manifest generation required
- Deterministic plugin selection verified
- All optional fields become required
- Warnings treated as errors

Use strict mode for regulatory submissions and compliance documentation.

#### Output

The audit command produces:
- **PDF Report**: Professional audit document with visualizations
- **Manifest File**: Complete audit trail (same directory as PDF)
- **Console Output**: Progress updates and summary statistics

### validate

Validate configuration files without running audits.

```bash
glassalpha validate --config CONFIG [OPTIONS]
```

#### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--config`, `-c` | Path | Path to configuration file to validate |

#### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--profile`, `-p` | String | None | Validate against specific profile |
| `--strict` | Flag | False | Validate for strict mode compliance |

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
- Component availability checks
- Profile compatibility assessment
- Strict mode requirement validation

### list

List available components and system capabilities.

```bash
glassalpha list [COMPONENT_TYPE] [OPTIONS]
```

#### Optional Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `COMPONENT_TYPE` | String | Filter by type: models, explainers, metrics, profiles |
| `--include-enterprise`, `-e` | Flag | Include enterprise components |
| `--verbose`, `-v` | Flag | Show component details |

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

| Type | Description | Examples |
|------|-------------|----------|
| `models` | ML model wrappers | xgboost, lightgbm, sklearn |
| `explainers` | Explanation methods | treeshap, kernelshap |
| `metrics` | Evaluation metrics | accuracy, demographic_parity |
| `profiles` | Audit profiles | tabular_compliance, german_credit_default |

#### Output

The list command shows:
- Registered component names by type
- Enterprise vs OSS availability
- Component counts and status
- License requirements for enterprise features

## Enterprise Commands

### dashboard serve

Start monitoring dashboard (Enterprise only).

```bash
glassalpha dashboard serve [OPTIONS]
```

#### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--port`, `-p` | Integer | 8080 | Port to serve on |
| `--host`, `-h` | String | localhost | Host to bind to |

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

#### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--config`, `-c` | Path | Configuration file |
| `--baseline`, `-b` | Path | Baseline manifest for comparison |

#### Example

```bash
# Monitor drift against baseline
glassalpha monitor drift \
  --config current_config.yaml \
  --baseline baseline_manifest.json
```

## Error Handling

GlassAlpha provides clear error messages for common issues:

### Configuration Errors
```bash
Configuration error: Missing required field 'data.path'
```

### File Not Found Errors
```bash
Error: Configuration file 'missing.yaml' not found
```

### Component Errors
```bash
Warning: Model type 'unknown_model' not found in registry
```

### Enterprise License Errors
```bash
Enterprise feature 'dashboard' requires valid license key
Set GLASSALPHA_LICENSE_KEY environment variable
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (configuration, file not found, etc.) |
| 2 | Invalid command line arguments |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GLASSALPHA_LICENSE_KEY` | Enterprise license key | None |
| `GLASSALPHA_LOG_LEVEL` | Logging level | INFO |
| `GLASSALPHA_CONFIG_DIR` | Default config directory | ~/.glassalpha |

## Performance Notes

### Typical Execution Times
- **Small datasets** (< 1,000 rows): 1-3 seconds
- **Medium datasets** (1,000-10,000 rows): 3-10 seconds
- **Large datasets** (10,000+ rows): 10-60 seconds

### Memory Requirements
- **Minimum**: 1GB RAM
- **Recommended**: 4GB+ RAM for large datasets
- **PDF Generation**: Additional 500MB temporary space

### Optimization Tips
- Use `--dry-run` to validate configurations quickly
- Enable `--quiet` for batch processing scripts
- Use specific component types with `list` for faster startup

## Integration Examples

### Batch Processing Script
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

### CI/CD Pipeline
```yaml
# Example GitHub Actions step
- name: Generate Audit Reports
  run: |
    glassalpha validate --config audit.yaml --strict
    glassalpha audit --config audit.yaml --output audit.pdf --strict
```

### Docker Usage
```dockerfile
FROM python:3.11-slim
RUN pip install glassalpha
COPY configs/ /app/configs/
WORKDIR /app
CMD ["glassalpha", "audit", "--config", "configs/production.yaml", "--output", "audit.pdf"]
```

This CLI reference covers all available commands and options for GlassAlpha's command-line interface. For configuration file syntax and options, see the [Configuration Guide](../getting-started/configuration.md).
