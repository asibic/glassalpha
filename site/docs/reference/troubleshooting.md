# Troubleshooting guide

Common issues, error messages, and solutions for GlassAlpha. This guide helps diagnose and resolve problems quickly.

## Quick diagnostics

If you're experiencing issues, start with these diagnostic commands:

```bash
# Verify installation (if package is installed)
glassalpha --version

# Alternative: Use module invocation (development/uninstalled)
PYTHONPATH=src python3 -m glassalpha --version

# Check component availability
glassalpha list
# Alternative: PYTHONPATH=src python3 -m glassalpha list

# Validate configuration
glassalpha validate --config configs/german_credit_simple.yaml
# Alternative: PYTHONPATH=src python3 -m glassalpha validate --config configs/german_credit_simple.yaml

# Test with minimal configuration
glassalpha audit --config configs/german_credit_simple.yaml --output test.pdf --dry-run
# Alternative: PYTHONPATH=src python3 -m glassalpha audit --config configs/german_credit_simple.yaml --output test.pdf --dry-run
```

## CLI command issues

### `glassalpha: command not found`

This indicates GlassAlpha isn't properly installed or the CLI entry point isn't available.

**Solutions:**

1. **Install the package** (recommended for users):

   ```bash
   cd glassalpha/packages
   pip install -e .

   # Verify installation
   glassalpha --version
   ```

2. **Use module invocation** (development/troubleshooting):

   ```bash
   cd glassalpha/packages
   PYTHONPATH=src python3 -m glassalpha --version
   PYTHONPATH=src python3 -m glassalpha audit --config configs/german_credit_simple.yaml --output test.pdf --dry-run
   ```

3. **Check your environment**:

   ```bash
   # Check if package is installed
   pip list | grep glassalpha

   # Check Python path
   python3 -c "import sys; print('\n'.join(sys.path))"
   ```

## Installation issues

### Python version compatibility

**Error:**

```
ERROR: Package 'glassalpha' requires a different Python: 3.10.0 not in '>=3.11'
```

**Solution:**

```bash
# Check Python version
python --version

# Upgrade Python (macOS with Homebrew)
brew install python@3.11
python3.11 -m pip install -e .

# Upgrade Python (Linux)
sudo apt update && sudo apt install python3.11

# Create virtual environment with correct Python
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
```

### XGBoost installation issues (macOS)

**Error:**

```
RuntimeError: libomp not found. Install with: brew install libomp
```

**Solution:**

```bash
# Install OpenMP library
brew install libomp

# Reinstall XGBoost
pip uninstall xgboost
pip install xgboost

# Verify installation
python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"
```

### Dependency conflicts

**Error:**

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solution:**

```bash
# Create clean virtual environment
python -m venv clean_env
source clean_env/bin/activate  # On Windows: clean_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install GlassAlpha
pip install -e .

# If conflicts persist, install dependencies individually
pip install pandas numpy scikit-learn
pip install xgboost lightgbm
pip install shap matplotlib seaborn
pip install -e . --no-deps
```

### Missing system dependencies

**Error (Linux):**

```
ImportError: libgomp.so.1: cannot open shared object file
```

**Solution (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install libgomp1 build-essential
```

**Solution (CentOS/RHEL):**

```bash
sudo yum install libgomp gcc gcc-c++
```

## Configuration errors

### Missing required fields

**Error:**

```
ValidationError: 1 validation error for AuditConfig
data.target_column
  field required (type=value_error.missing)
```

**Solution:**

```yaml
# Add missing required fields
data:
  path: data/dataset.csv
  target_column: outcome # This was missing

model:
  type: xgboost # Ensure model type is specified
```

### Invalid file paths

**Error:**

```
FileNotFoundError: Configuration file 'nonexistent.yaml' not found
```

**Solutions:**

```bash
# Use absolute paths
glassalpha audit --config /full/path/to/config.yaml --output report.pdf

# Check current directory
pwd
ls -la configs/

# Use relative paths from correct directory
cd /Users/gabe/Sites/glassalpha/packages
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf
```

### Model type not found

**Error:**

```
Warning: Model type 'unknown_model' not found in registry
```

**Solution:**

```bash
# Check available models
glassalpha list models

# Use correct model type
# Valid options: xgboost, lightgbm, logistic_regression, sklearn_generic
```

```yaml
model:
  type: xgboost # Use registered model type
```

### Schema validation errors

**Error:**

```
ValidationError: 2 validation errors for TabularDataSchema
features
  ensure this value has at least 1 characters
target
  field required
```

**Solution:**

```yaml
data:
  path: data/dataset.csv
  target_column: target # Must specify target
  feature_columns: # Either specify features explicitly
    - feature1
    - feature2
  # OR let GlassAlpha auto-detect (remove feature_columns)
```

## Model and dependency issues

### Model not available errors

**Error messages:**

```
Model 'xgboost' not available. Falling back to 'logistic_regression'.
To enable 'xgboost', run: pip install 'glassalpha[xgboost]'
```

Or:

```
Missing optional dependency 'xgboost' for plugin 'xgboost'.
Try: pip install 'glassalpha[xgboost]'
```

**Cause:** GlassAlpha uses optional dependencies to keep the core installation lightweight. XGBoost and LightGBM are not installed by default.

**Solutions:**

1. **Use the baseline model** (recommended for getting started):

   ```yaml
   # In your config.yaml
   model:
     type: logistic_regression # Always available
     params:
       random_state: 42
   ```

2. **Install the features you need**:

   ```bash
   # For SHAP + tree models (includes XGBoost and LightGBM)
   pip install 'glassalpha[explain]'

   # For all features
   pip install 'glassalpha[all]'
   ```

3. **Check what's currently available**:

   ```bash
   glassalpha doctor
   ```

4. **Disable fallbacks for strict requirements**:
   ```yaml
   model:
     type: xgboost
     allow_fallback: false # Fail if XGBoost not available
   ```

### Model loading failures

**Error messages:**

```
ModuleNotFoundError: No module named 'xgboost'
```

**Cause:** The model library isn't installed in your environment.

**Solutions:** Same as above - install the required optional dependency.

### Fallback model not suitable

**Error messages:**

```
Model 'xgboost' not available. Falling back to 'logistic_regression'.
```

**Cause:** The requested model isn't available, and fallback is enabled.

**Solutions:**

1. **Accept the fallback** (LogisticRegression is a solid baseline)
2. **Install the requested model** using the provided installation command
3. **Disable fallbacks** if you need the specific model:
   ```yaml
   model:
     type: xgboost
     allow_fallback: false
   ```

## Data issues

### Dataset not found

**Error:**

```
FileNotFoundError: Dataset file 'data/missing.csv' not found
```

**Solutions:**

```bash
# Check file existence
ls -la data/missing.csv

# Use German Credit dataset (automatically downloaded)
# Update config to use built-in dataset path:
```

```yaml
data:
  path: ~/.glassalpha/data/german_credit_processed.csv # Auto-downloaded
```

### Column not found

**Error:**

```
KeyError: Column 'target' not found in dataset
```

**Solutions:**

```bash
# Check column names in your data
python -c "import pandas as pd; print(pd.read_csv('data/dataset.csv').columns.tolist())"

# Update configuration with correct column name
```

```yaml
data:
  target_column: correct_column_name # Use actual column name
```

### Data type issues

**Error:**

```
TypeError: Object of type 'datetime64' is not JSON serializable
```

**Solution:**

```python
# Preprocess data to handle datetime columns
import pandas as pd

df = pd.read_csv('data/dataset.csv')
# Convert datetime to timestamp or remove
df['date_column'] = pd.to_datetime(df['date_column']).astype('int64')
df.to_csv('data/processed_dataset.csv', index=False)
```

### Missing values

**Error:**

```
ValueError: Input contains NaN, infinity or a value too large
```

**Solution:**

```yaml
# Add preprocessing configuration
preprocessing:
  handle_missing: true
  missing_strategy: median # or 'mode' for categorical
```

## Runtime errors

### Memory issues

**Error:**

```
MemoryError: Unable to allocate array with shape (10000, 10000)
```

**Solutions:**

```yaml
# Reduce sample sizes for explainers
explainers:
  config:
    treeshap:
      max_samples: 100 # Reduce from default 1000
    kernelshap:
      n_samples: 50 # Reduce from default 500

# Enable low memory mode
performance:
  low_memory_mode: true
  n_jobs: 1 # Reduce parallelism
```

```bash
# Use smaller datasets for testing
head -n 1000 large_dataset.csv > small_dataset.csv
```

### Model training failures

**Error:**

```
XGBoostError: Check failed: labels.Size() == num_row
```

**Solution:**

```python
# Check data consistency
import pandas as pd
df = pd.read_csv('data/dataset.csv')
print(f"Rows: {len(df)}")
print(f"Target values: {df['target'].value_counts()}")
print(f"Missing values: {df.isnull().sum()}")

# Remove rows with missing target values
df = df.dropna(subset=['target'])
df.to_csv('data/cleaned_dataset.csv', index=False)
```

### SHAP computation errors

**Error:**

```
Exception: TreeExplainer only supports the following model types: xgboost.Booster
```

**Solution:**

```yaml
# Use compatible explainer
explainers:
  priority:
    - kernelshap # Model-agnostic alternative
  config:
    kernelshap:
      n_samples: 100
```

### PDF generation issues

**Error:**

```
OSError: cannot load library 'pango-1.0-0'
```

**Solution (Linux):**

```bash
# Install required system libraries
sudo apt install libpango1.0-dev libcairo2-dev libgtk-3-dev
```

**Solution (macOS):**

```bash
# Install with Homebrew
brew install pango cairo gtk+3
```

**Solution (Alternative):**

```bash
# Use HTML output instead
# Modify report configuration:
```

```yaml
report:
  output_format: html # Generate HTML instead of PDF
```

## Performance issues

### Slow execution

**Problem:** Audit takes longer than expected

**Solutions:**

1. **Reduce Sample Sizes:**

```yaml
explainers:
  config:
    treeshap:
      max_samples: 100 # Default: 1000
    kernelshap:
      n_samples: 50 # Default: 500
      background_size: 50 # Default: 100
```

2. **Enable Parallel Processing:**

```yaml
performance:
  n_jobs: -1 # Use all CPU cores
```

3. **Use Simpler Models:**

```yaml
model:
  params:
    n_estimators: 50 # Reduce from 100
    max_depth: 3 # Reduce from 6
```

4. **Skip Expensive Operations:**

```bash
# Use dry run for config validation
glassalpha audit --config config.yaml --output test.pdf --dry-run

# Test with smaller datasets first
head -n 500 large_dataset.csv > test_dataset.csv
```

### High memory usage

**Problem:** System runs out of memory during audit

**Solutions:**

1. **Enable Memory Optimization:**

```yaml
performance:
  low_memory_mode: true
  n_jobs: 1 # Reduce parallelism
```

2. **Process Data in Chunks:**

```python
# Pre-process large datasets
import pandas as pd
df = pd.read_csv('large_dataset.csv')
sample = df.sample(n=1000, random_state=42)  # Use sample for testing
sample.to_csv('sample_dataset.csv', index=False)
```

3. **Optimize Explainer Settings:**

```yaml
explainers:
  config:
    treeshap:
      max_samples: 50 # Very small for memory constraints
    kernelshap:
      n_samples: 20
      background_size: 20
```

### Large PDF files

**Problem:** Generated PDFs are too large

**Solutions:**

1. **Optimize Report Configuration:**

```yaml
report:
  styling:
    optimize_size: true
    compress_images: true
```

2. **Reduce Plot Resolution:**

```yaml
explainers:
  config:
    treeshap:
      plot_dpi: 150 # Default: 300
```

3. **Exclude Optional Sections:**

```yaml
report:
  include_sections:
    - executive_summary
    - model_performance
    # Remove: local_explanations, detailed_plots
```

## Debugging and diagnostics

### Enable verbose logging

```bash
# Enable detailed logging
glassalpha --verbose audit --config config.yaml --output audit.pdf

# Check specific component status
export GLASSALPHA_LOG_LEVEL=DEBUG
glassalpha audit --config config.yaml --output audit.pdf
```

### Component registry debugging

```python
# Debug component registration
python -c "
from glassalpha.core import list_components
import pprint
pprint.pprint(list_components())
"
```

### Configuration debugging

```python
# Debug configuration loading
python -c "
from glassalpha.config import load_config_from_file
config = load_config_from_file('your_config.yaml')
print('Loaded config:', config.model_dump())
"
```

### Data loading debugging

```python
# Debug data issues
python -c "
from glassalpha.data import TabularDataLoader
loader = TabularDataLoader()
data = loader.load('data/dataset.csv')
print('Shape:', data.shape)
print('Columns:', data.columns.tolist())
print('Dtypes:', data.dtypes.to_dict())
print('Missing:', data.isnull().sum().to_dict())
"
```

## Getting help

### Before requesting support

1. **Check this troubleshooting guide**
2. **Run diagnostic commands** shown above
3. **Try with the German Credit example** (known working configuration)
4. **Enable verbose logging** for detailed error information

### Information to include in support requests

```bash
# System information
glassalpha --version
python --version
pip list | grep -E "(glassalpha|xgboost|lightgbm|shap|pandas)"

# Configuration (remove sensitive data)
cat your_config.yaml

# Complete error message
glassalpha --verbose audit --config config.yaml --output audit.pdf 2>&1
```

### Support channels

- **GitHub Issues:** [https://github.com/GlassAlpha/glassalpha/issues](https://github.com/GlassAlpha/glassalpha/issues)
- **Documentation:** [Complete guides](../index.md)
- **Community:** [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)

## Prevention best practices

### Environment setup

```bash
# Always use virtual environments
python -m venv glassalpha-env
source glassalpha-env/bin/activate

# Keep dependencies updated
pip install --upgrade pip
pip install --upgrade glassalpha
```

### Configuration management

```yaml
# Always specify explicit seeds for reproducibility
reproducibility:
  random_seed: 42
# Use version control for configurations
# Git commit: "Update audit configuration for production"

# Validate configurations before use
```

```bash
glassalpha validate --config production.yaml --strict
```

### Testing strategy

```bash
# Test with small datasets first
glassalpha audit --config config.yaml --output test.pdf --dry-run

# Test new configurations incrementally
# Start with minimal config, add features gradually
```

### Monitoring and logging

```bash
# Enable logging for production
export GLASSALPHA_LOG_LEVEL=INFO

# Log all audit runs
glassalpha audit --config config.yaml --output audit.pdf 2>&1 | tee audit.log
```

This troubleshooting guide covers the most common issues encountered with GlassAlpha. If you encounter an issue not covered here, please check the GitHub issues or contact support with the diagnostic information requested above.
