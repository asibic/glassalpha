# GlassAlpha dev docs

This is a technical deep-dive for devs. If you just want to generate your first audit, check the [main README](../README.md) instead.

## What this does

GlassAlpha generates deterministic audit PDFs with complete lineage tracking. Same config, same data, same seed = byte-identical PDF every time. Built for regulated ML deployments where you need to prove your model isn't biased and explain every decision.

Currently supports tabular ML with LogisticRegression baseline (always available) plus optional XGBoost and LightGBM support. LLM and vision model support may be added later based on demand.

**[→ Project Homepage](../README.md)** | **[→ User Documentation](https://glassalpha.com)**

### Design principles

1. **Plugin architecture**: All components (models, explainers, metrics) use dynamic registration - adding new ones shouldn't require touching core code
2. **Deterministic behavior**: Configuration-driven with reproducible results (regulatory requirement)
3. **Clear OSS/enterprise separation**: Core functionality is open-source, advanced features may be commercial if there's demand
4. **Modality-agnostic interfaces**: Designed to support tabular, text, and vision models (currently tabular only)

## Key features

### Current features

- ✅ LogisticRegression baseline (always available)
- ✅ XGBoost, LightGBM support (optional, install with extras)
- ✅ TreeSHAP explainability
- ✅ Fairness metrics (demographic parity, equal opportunity)
- ✅ Basic recourse (immutables, monotonicity)
- ✅ Deterministic PDF audit reports
- ✅ Full reproducibility (seeds, hashes, manifests)

### Future extensions

More may be added depending on support and resourcing. Looking for something? **[Start a discussion](https://github.com/GlassAlpha/glassalpha/discussions)** or [contact me](https://glassalpha.com/contact/).

### Project structure

```
packages/
├── src/
│   └── glassalpha/          # OSS core (Apache 2.0)
│       ├── core/            # Interfaces & protocols
│       ├── models/          # Model wrappers & registry
│       ├── explain/         # Explainability plugins
│       ├── metrics/         # Performance & fairness metrics
│       ├── data/            # Data handling & validation
│       ├── report/          # Report generation
│       ├── profiles/        # Audit profiles
│       ├── cli/             # CLI interface (Typer)
│       └── utils/           # Utilities (seeds, hashing, etc.)
├── configs/                 # Example configurations
├── tests/                   # Test suite
└── pyproject.toml          # Package configuration
```

## Installation

### Requirements

- **Python**: 3.11, 3.12, or 3.13 (all fully supported)
- **Memory**: 2GB minimum (8GB recommended for large datasets)
- **Storage**: 1GB free space for datasets and reports
- **Network**: Required for initial dataset downloads (unless using `--offline` mode)

### Quick installation

Clone the repository

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages
```

Install base framework only (lightweight, includes LogisticRegression)

```bash
pip install -e .

# Or install with advanced ML libraries
pip install -e ".[xgboost]"      # XGBoost + SHAP only
pip install -e ".[lightgbm]"     # LightGBM only
pip install -e ".[tabular]"      # All tabular ML libraries
```

### Development installation

For contributors and advanced users:

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip and install in development mode with all extras
python -m pip install --upgrade pip
pip install -e ".[dev]"         # Development tools only
pip install -e ".[tabular]"     # Add ML libraries if needed

# Verify installation
glassalpha --help
```

### Reproducible installation

For CI/CD or reproducible builds, use the constraints file:

```bash
pip install -c constraints.txt -e ".[dev,tabular]"
```

This locks all dependencies to tested versions.

### Offline installation

For air-gapped environments:

```bash
# Pre-download datasets to cache directory first
glassalpha datasets fetch german_credit
glassalpha datasets fetch adult_income

# Then use with offline mode
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf --offline
```

### Troubleshooting installation

**Common Issues:**

1. **Python version**: Ensure you're using Python 3.11+

   ```bash
   python3 --version  # Should show 3.11.x or higher
   ```

2. **Virtual environment**: Always use a virtual environment for isolation

   ```bash
   python3 -m venv glassalpha-env
   source glassalpha-env/bin/activate
   ```

3. **Build dependencies**: If you encounter build errors:

   ```bash
   # Install system dependencies first
   # On Ubuntu/Debian:
   sudo apt-get install build-essential python3-dev

   # On macOS:
   brew install gcc
   ```

**Verification:**

```bash
# Check installation
glassalpha --help

# List available components
glassalpha list

# Test with German Credit example
glassalpha audit --config configs/german_credit_simple.yaml --output test_audit.pdf
```

## Quick start

### Generate audit PDF in 10 seconds

Basic audit:

```bash
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf
```

Strict mode (enforces regulatory requirements):

```bash
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf --strict
```

### Configuration example

```yaml
audit_profile: tabular_compliance

data:
  dataset: german_credit # Use built-in dataset
  fetch: if_missing # Auto-fetch if not cached
  offline: false # Allow network operations
  target_column: credit_risk
  protected_attributes:
    - gender

model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100
    max_depth: 5

explainers:
  strategy: first_compatible
  priority:
    - treeshap
    - kernelshap

reproducibility:
  random_seed: 42
```

### Data sources

GlassAlpha supports three ways to specify data:

**1. Built-in datasets (recommended)**

Use registered datasets that are automatically fetched and cached:

```yaml
data:
  dataset: german_credit
  fetch: if_missing # Options: never, if_missing, always
  offline: false # Set true for air-gapped environments
```

**2. Custom data files**

Use your own data files:

```yaml
data:
  dataset: custom
  path: /absolute/path/to/your/data.csv
  target_column: outcome
  feature_columns: [feature1, feature2, ...]
```

**3. Schema-only (testing/validation)**

For schema validation utilities:

```yaml
data:
  data_schema:
    type: object
    properties:
      feature1: { type: number }
      feature2: { type: string }
```

### Dataset management CLI

```bash
# List available built-in datasets
glassalpha datasets list

# Show dataset info
glassalpha datasets info german_credit

# Show cache directory
glassalpha datasets cache-dir

# Pre-fetch a dataset
glassalpha datasets fetch german_credit
```

**Cache directory resolution:**

- Default: OS-specific user data directory (e.g., `~/Library/Application Support/glassalpha/data` on macOS)
- Override: Set `GLASSALPHA_DATA_DIR` environment variable
- All paths are canonicalized (symlinks resolved) for consistency
- The system logs both requested and effective paths for transparency

## Architecture highlights

The architecture focuses on making it easy to add new models, explainers, or metrics without touching core code. Everything is plugin-based and deterministic (regulatory requirement). Here's how it works:

### 1. Protocol-based interfaces

All components implement protocols (PEP 544) for maximum flexibility:

```python
from typing import Protocol

class ModelInterface(Protocol):
    def predict(self, X): ...
    def get_capabilities(self) -> dict: ...
```

**Why protocols instead of base classes:**

- Duck typing - focus on behavior, not inheritance
- Wrap existing libraries without modification
- Get type safety with mypy verification

### 2. Registry pattern

Components self-register for dynamic loading:

```python
@ModelRegistry.register("xgboost")
class XGBoostWrapper(ModelInterface):
    capabilities = {"supports_shap": True, "data_modality": "tabular"}
```

**Selection process:**

1. Configuration specifies preferences (e.g., `priority: [treeshap, kernelshap]`)
2. Registry finds first compatible component based on model capabilities
3. Selected component logged in audit manifest for reproducibility

### 3. Deterministic plugin selection

Configuration drives component selection with explicit priority ordering:

```yaml
explainers:
  strategy: "first_compatible"
  priority: ["treeshap", "kernelshap"]
```

**Why this matters:**

- Same config = same components selected (reproducible audits)
- Audit trails are verifiable (regulatory requirement)
- Predictable behavior makes debugging easier

### 4. Audit profiles

Different audit types use different component sets:

```python
class TabularComplianceProfile(AuditProfile):
    compatible_models = ["xgboost", "lightgbm", "logistic_regression"]
    required_metrics = ["accuracy", "demographic_parity"]
    default_explainers = ["treeshap", "kernelshap"]
    report_template = "standard_audit.html"
```

**Benefits:**

- **Validated combinations**: Ensures compatible components
- **Simplified configuration**: Users specify profile, not individual components
- **Regulatory alignment**: Profiles match compliance requirements

### 5. Configuration-driven pipeline

Every aspect of the audit pipeline is configuration-driven:

```yaml
# High-level intent
audit_profile: german_credit_default

# Model selection
model:
  type: xgboost

# Explainer preferences
explainers:
  priority: [treeshap, kernelshap]

# Strict mode for regulatory compliance
strict: true
```

**Configuration benefits:**

- **Human-readable**: YAML format that's version-controllable
- **Validated**: Pydantic models catch errors before execution
- **Flexible**: Support for environment-specific overrides
- **Auditable**: Configuration hash included in audit manifest

### 6. Enterprise feature separation

Clear boundaries between OSS and potential enterprise features:

```python
@check_feature("advanced_explainers")
def deep_shap_explain():
    if not is_enterprise():
        raise FeatureNotAvailable("Enterprise license required")
    # Enterprise implementation
```

**OSS (always free):** Core audit functionality, TreeSHAP, standard metrics
**Enterprise (if there's demand):** Advanced explainers, monitoring dashboards, custom integrations

### 7. Comprehensive audit trails

Complete provenance tracking for regulatory compliance:

```python
class ManifestGenerator:
    def generate(self):
        return {
            "config_hash": hash_config(config),
            "data_hash": hash_dataframe(data),
            "git_sha": get_git_commit(),
            "selected_components": self.components,
            "seeds": self.seeds,
            "timestamp": datetime.utcnow()
        }
```

**Audit trail contents:**

- Configuration fingerprint (hash)
- Dataset fingerprint (hash)
- Git commit SHA and timestamp
- Selected components and their versions
- All random seeds used
- Environment information

## Testing

Run the full test suite:

```bash
pytest  # All tests with coverage
```

Quick smoke test:

```bash
glassalpha audit --config configs/german_credit_simple.yaml --output test_audit.pdf
```

## Development & contributing

### Development setup

1. **Clone & setup:**

   ```bash
   git clone https://github.com/GlassAlpha/glassalpha
   cd glassalpha/packages
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

2. **Run tests:**

   ```bash
   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=src/glassalpha

   # Run specific test file
   pytest tests/test_audit_pipeline.py -v
   ```

3. **Code quality:**

   ```bash
   # Format code
   ruff format src/ tests/

   # Lint code
   ruff check src/ tests/

   # Type check
   mypy src/
   ```

### Contributing guidelines

**Good areas to contribute:**

- Additional model support (neural networks, time series, etc.)
- Advanced explainability methods (counterfactuals, gradient-based)
- Extended compliance frameworks (GDPR, HIPAA, SOX)
- Performance optimizations for large datasets

**Development workflow:**

1. Create feature branch from `main`
2. Write tests first (TDD approach)
3. Implement feature with type hints
4. Add documentation and examples
5. Run full test suite and linting
6. Submit PR with clear description

**Code standards (enforced):**

- Type hints for all public APIs
- Google-style docstrings for all modules/functions
- > 90% test coverage
- Ruff + Black formatting

See [Contributing Guide](https://glassalpha.com/reference/contributing/) for detailed guidelines.

## License & dependencies

### License structure

- **GlassAlpha core**: Apache 2.0 ([LICENSE](../LICENSE)) - use it however you want
- **Enterprise features**: Commercial license if there's demand for advanced/custom features
- **Brand**: "GlassAlpha" trademark - see [TRADEMARK.md](../TRADEMARK.md)

### Technology stack & licenses

All dependencies are permissively licensed (MIT, BSD, Apache 2.0) - no GPL anywhere:

| Component        | License     | Purpose                     | Why Chosen                                     |
| ---------------- | ----------- | --------------------------- | ---------------------------------------------- |
| **Python SHAP**  | MIT License | TreeSHAP explanations       | ✅ Enterprise-compatible, no GPL contamination |
| **XGBoost**      | Apache 2.0  | Gradient boosting models    | ✅ Same license family, proven in production   |
| **LightGBM**     | MIT License | Alternative tree models     | ✅ Microsoft-backed, widely adopted            |
| **scikit-learn** | BSD License | Baseline models & utilities | ✅ Academic standard, fully compatible         |
| **NumPy**        | BSD License | Numerical computing         | ✅ Core scientific Python library              |
| **Pandas**       | BSD License | Data manipulation           | ✅ Industry standard for data science          |
| **WeasyPrint**   | BSD License | PDF generation              | ✅ Pure Python, no system dependencies         |
| **Typer**        | MIT License | CLI framework               | ✅ Modern, type-safe command interface         |
| **Pydantic**     | MIT License | Configuration validation    | ✅ Runtime type checking and validation        |

### Why this matters

**No GPL dependencies**: Deliberately avoided GPL to ensure maximum compatibility. Using MIT-licensed Python [SHAP](https://github.com/shap/shap) instead of GPL-licensed R `treeshap`.

**All Apache 2.0 compatible**: You can:

- Use commercially without restrictions
- Integrate with proprietary systems
- Distribute in closed-source applications
- Get patent protection as a contributor

**Regulatory compliance ready**:

- Complete audit trails
- Reproducible builds with locked versions
- No vendor lock-in
- Full source transparency for regulatory review

### Enterprise integration

The clean licensing structure enables:

- **Container integration**: Deploy in Docker/Kubernetes without license conflicts
- **CI/CD pipelines**: Automated builds with reproducible dependency resolution
- **Cloud deployment**: Compatible with AWS, Azure, GCP licensing requirements
- **On-premise installation**: Full control over software stack and dependencies

### Dependency verification

All dependencies are locked to specific versions in [`constraints.txt`](constraints.txt) for reproducible builds:

```bash
# Reproducible installation
pip install -c constraints.txt -e .
```

## Support

- **Questions**: [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- **Bugs**: [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)
- **Docs**: [FAQ](https://glassalpha.com/reference/faq/) and [full documentation](https://glassalpha.com)

---

_Questions? Start a [discussion](https://github.com/GlassAlpha/glassalpha/discussions) or check the [FAQ](https://glassalpha.com/reference/faq/)._
