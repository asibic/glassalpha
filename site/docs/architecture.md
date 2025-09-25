# Architecture Overview

GlassAlpha is designed as an **extensible framework** for ML model auditing, built with regulatory compliance and professional use in mind. This guide explains the system architecture, design decisions, and how components work together.

## Design Philosophy

### Audit-First Approach
GlassAlpha prioritizes **regulatory compliance** and **audit quality** over cutting-edge features:

- **Deterministic behavior** - Same input always produces identical output
- **Complete audit trails** - Every decision is tracked and reproducible
- **Professional quality** - Reports suitable for regulatory submission
- **Transparency** - Open source code that can be verified and trusted

### Plugin Architecture
All major components use **dynamic registration** and **interface-based design**:

- **Models**, **explainers**, and **metrics** are plugins that register themselves
- **Configuration drives selection** - specify preferences, system picks best match
- **Easy extensibility** - add new implementations without changing core code
- **Enterprise separation** - clear boundaries between OSS and commercial features

## System Architecture

### High-Level Component Flow

```
User Configuration (YAML)
          ↓
    Config Loading & Validation
          ↓
    Component Selection (Registries)
          ↓
    Data Loading & Processing
          ↓
    Model Training/Loading
          ↓
    Explanation Generation
          ↓
    Metrics Computation
          ↓
    Report Generation (HTML → PDF)
          ↓
    Audit Manifest Creation
```

### Core Components

#### 1. Configuration System
**Purpose**: Declarative, reproducible audit specification

**Key Files:**
- `config/schema.py` - Pydantic models for validation
- `config/loader.py` - YAML loading and environment handling
- `config/strict.py` - Regulatory compliance validation

**Design Features:**
- **YAML-based** - Human-readable and version-controllable
- **Strict mode** - Enforces regulatory requirements (explicit seeds, no defaults)
- **Validation** - Comprehensive error checking before execution
- **Overrides** - Environment-specific configuration management

**Example Flow:**
```yaml
# User specifies high-level intent
audit_profile: tabular_compliance
model:
  type: xgboost
explainers:
  priority: [treeshap, kernelshap]
```
→ System validates, fills defaults, ensures determinism

#### 2. Registry System
**Purpose**: Dynamic component discovery and selection

**Key Files:**
- `core/registry.py` - Registration and selection logic
- `core/interfaces.py` - Protocol definitions

**How It Works:**
```python
# Components register themselves
@ModelRegistry.register("xgboost")
class XGBoostWrapper:
    capabilities = {"supports_shap": True}

# System selects based on config + capabilities
model_cls = ModelRegistry.select("xgboost")
explainer_cls = ExplainerRegistry.select_compatible(model, ["treeshap"])
```

**Benefits:**
- **Deterministic selection** - Same config = same components
- **Capability matching** - Explainers only run on compatible models
- **Enterprise gating** - Features locked behind license checks
- **Extensibility** - Add new types without changing core

#### 3. Data Pipeline
**Purpose**: Load, validate, and prepare data for analysis

**Key Files:**
- `data/tabular.py` - CSV/Parquet loading with pandas
- `data/german_credit.py` - Example dataset with preprocessing

**Processing Steps:**
1. **Format Detection** - Auto-detect CSV, Parquet, Feather
2. **Schema Validation** - Ensure required columns exist
3. **Type Conversion** - Handle categorical/numeric features
4. **Protected Attributes** - Extract demographic data for fairness analysis
5. **Train/Test Split** - Deterministic splitting with fixed seeds
6. **Data Hashing** - Integrity verification for audit trail

#### 4. Model Integration
**Purpose**: Unified interface to different ML libraries

**Key Files:**
- `models/tabular/xgboost.py` - XGBoost wrapper
- `models/tabular/lightgbm.py` - LightGBM wrapper
- `models/tabular/sklearn.py` - Scikit-learn wrapper

**Interface Design:**
```python
class ModelInterface(Protocol):
    capabilities: dict[str, Any]

    def predict(self, X: pd.DataFrame) -> np.ndarray
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray
    def get_model_type(self) -> str
```

**Key Features:**
- **Capability declaration** - Models specify what explainers they support
- **Consistent interface** - Same API regardless of underlying library
- **Flexible loading** - Support pre-trained models or train from scratch
- **Error handling** - Graceful degradation when libraries unavailable

#### 5. Explanation System
**Purpose**: Generate interpretable explanations for model decisions

**Key Files:**
- `explain/shap/tree.py` - TreeSHAP for tree-based models
- `explain/shap/kernel.py` - KernelSHAP for any model type

**Selection Logic:**
```python
# Configuration specifies preferences
explainers:
  priority: [treeshap, kernelshap]

# System finds first compatible explainer
for explainer_name in priority:
    if explainer.supports_model(model):
        selected = explainer
        break
```

**Design Features:**
- **Priority-based selection** - Deterministic fallback chain
- **Capability checking** - Only run explainers that work with the model
- **Configurable sampling** - Balance accuracy vs speed
- **Reproducible results** - Fixed seeds for consistent outputs

#### 6. Metrics System
**Purpose**: Comprehensive model evaluation across multiple dimensions

**Categories:**
- **Performance** (6 metrics): Accuracy, Precision, Recall, F1, AUC-ROC, Classification Report
- **Fairness** (4 metrics): Demographic Parity, Equal Opportunity, Equalized Odds, Predictive Parity
- **Drift** (5 metrics): PSI, KL Divergence, KS Test, JS Divergence, Prediction Drift

**Design Features:**
- **Category-based organization** - Group related metrics
- **Conditional computation** - Only run relevant metrics (e.g., fairness needs protected attributes)
- **Statistical rigor** - Confidence intervals and significance tests
- **Flexible requirements** - Some metrics need probabilities, others just predictions

#### 7. Report Generation
**Purpose**: Professional PDF reports with visualizations

**Key Files:**
- `report/renderer.py` - HTML to PDF conversion
- `report/plots.py` - Matplotlib/Seaborn visualizations

**Pipeline:**
1. **Template Selection** - Choose HTML template based on audit profile
2. **Data Preparation** - Organize results for template rendering
3. **Plot Generation** - Create deterministic visualizations with fixed seeds
4. **HTML Rendering** - Jinja2 templating with embedded CSS
5. **PDF Conversion** - WeasyPrint for publication-quality output
6. **Metadata Addition** - PDF properties and audit trail information

### Audit Profiles
**Purpose**: Pre-configured component sets for different use cases

**Current Profiles:**
- `tabular_compliance` - Standard audit for tabular models
- `german_credit_default` - Optimized for German Credit dataset

**Profile Structure:**
```python
class TabularComplianceProfile:
    compatible_models = ["xgboost", "lightgbm", "logistic_regression"]
    required_metrics = ["accuracy", "demographic_parity"]
    default_explainers = ["treeshap", "kernelshap"]
    report_template = "standard_audit.html"
```

**Benefits:**
- **Validated combinations** - Ensures compatible components
- **Simplified configuration** - Users specify profile, not individual components
- **Regulatory alignment** - Profiles match compliance requirements
- **Quality assurance** - Pre-tested component combinations

## Design Patterns

### 1. Protocol-Based Interfaces
Instead of inheritance, GlassAlpha uses Python's `typing.Protocol`:

```python
@runtime_checkable
class ModelInterface(Protocol):
    capabilities: dict[str, Any]

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
```

**Advantages:**
- **Duck typing** - Focus on behavior, not inheritance
- **Flexibility** - Easy to wrap existing libraries
- **Testing** - Can create mock implementations easily
- **Type safety** - mypy can verify protocol compliance

### 2. Capability-Based Selection
Components declare what they can do, system matches capabilities to needs:

```python
class TreeSHAPExplainer:
    capabilities = {
        "model_types": ["xgboost", "lightgbm"],
        "explanation_type": "feature_attribution",
        "supports_interactions": True
    }

    def supports_model(self, model):
        return model.get_model_type() in self.capabilities["model_types"]
```

### 3. Registry Pattern
Central registration system for dynamic component discovery:

```python
class ExplainerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name, priority=50):
        def decorator(explainer_class):
            cls._registry[name] = {
                "class": explainer_class,
                "priority": priority
            }
            return explainer_class
        return decorator
```

### 4. Deterministic Operations
Every source of randomness is controlled and seeded:

```python
# Central seed management
set_global_seed(42)

# Component-specific seeds
with with_component_seed("explainer"):
    shap_values = explainer.explain(model, X)
```

### 5. Audit Trail Generation
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

## Enterprise Architecture

### Feature Separation
GlassAlpha maintains clear boundaries between OSS and Enterprise features:

**Current Approach (Feature Flags):**
```python
@check_feature("advanced_explainers")
def deep_shap_explain():
    if not is_enterprise():
        raise FeatureNotAvailable()
    # Enterprise implementation
```

**Future Approach (Package Separation):**
```python
# glassalpha (OSS) - Core functionality only
# glassalpha-enterprise - Advanced features in separate package
```

### Extension Points
The architecture provides multiple extension points:

1. **Custom Models** - Implement `ModelInterface`
2. **Custom Explainers** - Implement `ExplainerInterface`
3. **Custom Metrics** - Implement `MetricInterface`
4. **Custom Reports** - Add new Jinja2 templates
5. **Custom Profiles** - Define component combinations

## Performance Considerations

### Scalability Design
- **Streaming-friendly** - Process data in chunks when needed
- **Memory efficient** - Release resources between pipeline stages
- **Parallel processing** - Use `n_jobs` parameter for CPU-bound operations
- **Caching** - Reuse expensive computations within single audit

### Optimization Strategies
- **TreeSHAP over KernelSHAP** - Use exact methods when available
- **Sample size tuning** - Balance accuracy vs speed in explanations
- **Vectorized operations** - Leverage pandas/numpy optimizations
- **Deterministic shortcuts** - Skip redundant computations with same seeds

## Security and Privacy

### Data Handling
- **Local processing only** - No external API calls
- **File-based storage** - No persistent databases required
- **Memory cleanup** - Sensitive data cleared after use
- **Audit logging** - Track data access without storing content

### Enterprise Security
- **License validation** - Environment variable or server-based
- **Feature gating** - Graceful degradation without enterprise features
- **Access controls** - Role-based component access (enterprise)
- **Audit trails** - User activity tracking (enterprise)

## Quality Assurance

### Testing Strategy
- **Unit tests** - Each component tested independently
- **Integration tests** - Full pipeline validation
- **Determinism tests** - Verify reproducible outputs
- **Enterprise tests** - Feature gating and license validation

### Validation Approaches
- **Schema validation** - Pydantic models catch configuration errors
- **Component compatibility** - Registry validates model/explainer combinations
- **Data integrity** - Hash verification detects changes
- **Output verification** - PDF byte-level comparison for reproducibility

## Future Architecture Considerations

### Extensibility Planning
The current architecture supports planned extensions:

- **New Model Types** - LLMs, vision models through same interfaces
- **New Explanation Methods** - Gradient-based, attention analysis
- **New Data Modalities** - Text, image through `DataInterface`
- **Cloud Integration** - Remote model serving, distributed processing

### Maintenance Strategy
- **Interface stability** - Protocol changes are backwards compatible
- **Migration support** - Clear upgrade paths between versions
- **Documentation** - Architecture decisions documented for future maintainers
- **Performance monitoring** - Metrics collection for optimization guidance

This architecture balances immediate needs (professional tabular ML auditing) with future extensibility, ensuring GlassAlpha can evolve while maintaining stability and trust for regulated industry use.

---

*For implementation details, see the [API Reference](reference/api.md). For usage examples, see [Examples](examples/german-credit-audit.md).*
