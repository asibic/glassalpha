# GlassAlpha - AI Compliance Toolkit

GlassAlpha is an **extensible framework** for AI compliance and interpretability, supporting tabular ML models with planned expansion to LLMs and multimodal systems.

## Overview

GlassAlpha provides **deterministic, regulator-ready audit reports** with complete lineage tracking. Run the same config twice, get byte-identical PDFs. Every decision is explainable, every metric is reproducible, every audit trail is complete.

**Designed for**: Organizations needing transparent, auditable ML systems that meet regulatory requirements (EU AI Act, CFPB guidance, etc.).

**[→ Project Homepage](../README.md)** | **[→ User Documentation](../site/docs/)**

### Core design principles

1. **Plugin architecture**: All components (models, explainers, metrics) use dynamic registration
2. **Deterministic behavior**: Configuration-driven with reproducible results
3. **Clear OSS/enterprise deparation**: Core functionality is open-source, advanced features are commercial
4. **Modality-agnostic interfaces**: Designed to support tabular, text, and vision models

For detailed architecture information, see the [Architecture Guide](../site/docs/architecture.md).

## Key features

### Current features

- ✅ XGBoost, LightGBM, LogisticRegression support
- ✅ TreeSHAP explainability
- ✅ Fairness metrics (demographic parity, equal opportunity)
- ✅ Basic recourse (immutables, monotonicity)
- ✅ Deterministic PDF audit reports
- ✅ Full reproducibility (seeds, hashes, manifests)

### Planned extensions

- LLM support with gradient-based explainability
- Vision model support
- Continuous monitoring dashboards
- Regulator-specific templates
- Cloud integrations

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

- **Python**: 3.11 or higher (3.12 recommended)
- **Memory**: 2GB minimum (8GB recommended for large datasets)
- **Storage**: 1GB free space for datasets and reports
- **Network**: Required for initial dataset downloads (unless using `--offline` mode)

### Quick installation

```bash
# Clone the repository
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages

# Install GlassAlpha (includes all dependencies)
pip install -e .
```

### Development installation

For contributors and advanced users:

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip and install in development mode
python -m pip install --upgrade pip
pip install -e ".[dev]"

# Verify installation
glassalpha --help
```

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

### Generate audit PDF

Basic audit:

```bash
glassalpha audit --config configs/german_credit_simple.yaml --output audit.pdf
```

Strict mode for regulatory compliance:

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

GlassAlpha's architecture is designed for **extensibility**, **determinism**, and **regulatory compliance**. The system uses modern Python patterns to ensure reliable, auditable ML model assessments.

### 1. Protocol-based interfaces

All components implement protocols for maximum flexibility:

```python
from typing import Protocol

class ModelInterface(Protocol):
    def predict(self, X): ...
    def get_capabilities(self) -> dict: ...
```

**Benefits:**

- **Duck typing**: Focus on behavior, not inheritance hierarchies
- **Easy wrapping**: Wrap existing libraries without modification
- **Type safety**: mypy verification of protocol compliance

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

Configuration drives component selection with deterministic priority:

```yaml
explainers:
  strategy: "first_compatible"
  priority: ["treeshap", "kernelshap"]
```

**Why Deterministic:**

- **Reproducible audits**: Same config = same components selected
- **Regulatory compliance**: Audit trails must be verifiable
- **Debugging**: Predictable component selection aids troubleshooting

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

Clear boundaries between OSS and Enterprise features:

```python
@check_feature("advanced_explainers")
def deep_shap_explain():
    if not is_enterprise():
        raise FeatureNotAvailable("Enterprise license required")
    # Enterprise implementation
```

**OSS focus:** Core audit functionality, TreeSHAP, standard metrics
**Enterprise:** Advanced explainers, monitoring dashboards, custom integrations

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

Run the test suite with coverage:

```bash
pytest  # Comprehensive test coverage
```

Test audit generation:

Generate test audit to verify functionality:

```bash
glassalpha audit --config configs/german_credit_simple.yaml --output test_audit.pdf
```

## Contributing

See [CONTRIBUTING.md](../site/docs/contributing.md) for development guidelines.

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

**Areas for enhancement:**

- Additional model support (neural networks, time series)
- Advanced explainability methods (counterfactuals, gradient-based)
- Extended compliance frameworks (GDPR, HIPAA, SOX)
- Performance optimizations for large datasets

**Development workflow:**

1. Create feature branch from `main`
2. Write tests first (TDD approach)
3. Implement feature with type hints
4. Add documentation and examples
5. Run full test suite
6. Submit PR with clear description

**Code standards:**

- **Type hints**: Required for all public APIs
- **Documentation**: Google-style docstrings for all modules/functions
- **Testing**: >90% test coverage required
- **Linting**: Ruff + Black formatting enforced

See [Contributing Guide](../site/docs/contributing.md) for detailed development guidelines.

## License & dependencies

### Core license structure

- **GlassAlpha framework**: Apache License 2.0 ([LICENSE](../LICENSE))
- **Enterprise extensions**: Commercial license for advanced features (separate package)
- **Brand**: "GlassAlpha" trademark - see [TRADEMARK.md](../TRADEMARK.md)

### Technology stack & licenses

GlassAlpha uses industry-standard, enterprise-compatible dependencies with proven licensing compatibility:

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

### Licensing confidence & risk mitigation

**✅ No GPL dependencies**: GlassAlpha deliberately avoids GPL-licensed components to ensure maximum compatibility with enterprise environments. We use the MIT-licensed Python [SHAP](https://github.com/shap/shap) library rather than the GPL-licensed R `treeshap` package.

**✅ Apache 2.0 compatible stack**: All dependencies are compatible with Apache 2.0 licensing, allowing:

- Commercial use without restrictions
- Integration with proprietary systems
- Distribution in closed-source applications
- Patent protection for contributors

**✅ Regulatory compliance ready**: The licensing structure supports:

- Audit trail preservation
- Reproducible builds with locked dependency versions
- No vendor lock-in through open standards
- Full source code transparency for regulatory review

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

- **OSS community**: [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)
- **Enterprise**: Priority support with SLA (contact for availability)
- **Discussions**: [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)

---

_For questions, see the [FAQ](../site/docs/faq.md) or start a [GitHub Discussion](https://github.com/GlassAlpha/glassalpha/discussions)._
