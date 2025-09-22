# Development Quick Start

!!! warning "Pre-Alpha Software"
    Glass Alpha is under active development. The audit generation features described in our documentation are planned but not yet implemented.

## Current Development Setup

### Prerequisites
- Python 3.11+
- Git
- pip/venv

### Step 1: Clone and Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e packages/[dev]
```

### Step 2: Verify Installation

```bash
# Run existing tests
pytest

# Check code quality tools
ruff check packages/
mypy packages/
```

### Step 3: Explore the Project Structure

```bash
# Package source code
ls packages/src/glassalpha/

# Documentation
ls site/docs/

# Tests
ls packages/tests/
```

## What's Currently Implemented

✅ **Available Now:**
- Basic package structure
- Development environment setup
- Documentation site (MkDocs)
- Testing framework (pytest)
- Code quality tools (ruff, mypy)

🚧 **In Development:**

- PDF audit report generation
- TreeSHAP integration for model explanations
- Basic fairness metrics implementation
- Deterministic, reproducible execution
- CLI interface (`glassalpha audit`)

## Contributing to Development

### Areas Needing Implementation

1. **Core Audit Module** (`packages/src/glassalpha/audit/`)
   - PDF generation pipeline
   - Report template system
   - Visualization components

2. **Explainability Module** (`packages/src/glassalpha/explain/`)
   - TreeSHAP wrapper for XGBoost/LightGBM
   - Feature importance calculations
   - Waterfall plot generation

3. **Fairness Module** (`packages/src/glassalpha/fairness/`)
   - Demographic parity metrics
   - Equalized odds calculations
   - Protected attribute analysis

4. **CLI Interface** (`packages/src/glassalpha/cli/`)
   - Command-line argument parsing
   - Configuration file loading
   - Progress reporting

### Example: Future Configuration Design

We're designing the configuration schema to support:

```yaml
# Planned configuration structure
model:
  type: xgboost
  params:
    max_depth: 5
    learning_rate: 0.1

data:
  train_path: german_train.csv  
  test_path: german_test.csv
  target_column: default

audit:
  protected_attributes:
    - gender
    - age_group
  confidence_level: 0.95
  
reproducibility:
  random_seed: 42
  track_git: true
```

## How to Contribute

1. **Check open issues** on GitHub for current priorities
2. **Join discussions** about architecture decisions
3. **Submit PRs** with tests and documentation
4. **Help with examples** using German Credit and Adult Income datasets

## Next Steps

- 📊 [Target Example: German Credit](../examples/german-credit-audit.md)
- ⚙️ [Design Doc: Configuration](configuration.md)  
- 🏛️ [Vision: Regulatory Compliance](../compliance/overview.md)
- 👥 [Contributing Guidelines](../contributing.md)

**Goal:** Transform Glass Alpha from vision to reality - one deterministic, regulator-ready PDF at a time.
