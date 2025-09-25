# API Reference

Complete reference for GlassAlpha's public API, interfaces, and extension points.

For system design and component interaction details, see the [Architecture Guide](../architecture.md). This document focuses on implementation specifics for developers extending GlassAlpha.

## Core Interfaces

GlassAlpha uses protocol-based interfaces to enable extensibility and plugin architecture.

### ModelInterface

Base protocol for all model implementations.

```python
from typing import Protocol
import pandas as pd
import numpy as np

class ModelInterface(Protocol):
    capabilities: dict[str, Any]
    version: str

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for input data."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions."""

    def get_model_type(self) -> str:
        """Return model type identifier."""

    def get_capabilities(self) -> dict[str, Any]:
        """Return model capabilities for plugin selection."""
```

**Required Attributes:**
- `capabilities`: Dict describing model capabilities (e.g., `{"supports_shap": True}`)
- `version`: String identifying implementation version

**Methods:**
- `predict()`: Generate class predictions
- `predict_proba()`: Generate probability predictions
- `get_model_type()`: Return type identifier (e.g., "xgboost")
- `get_capabilities()`: Return capabilities for component selection

### ExplainerInterface

Base protocol for explanation methods.

```python
class ExplainerInterface(Protocol):
    capabilities: dict[str, Any]
    version: str
    priority: int  # Higher = preferred

    def explain(self, model: ModelInterface, X: pd.DataFrame,
                y: np.ndarray = None) -> dict[str, Any]:
        """Generate model explanations."""

    def supports_model(self, model: ModelInterface) -> bool:
        """Check if explainer supports the model."""

    def get_explanation_type(self) -> str:
        """Return explanation type identifier."""
```

**Required Attributes:**
- `capabilities`: Dict describing explainer capabilities
- `version`: Implementation version
- `priority`: Selection priority (higher = preferred)

**Methods:**
- `explain()`: Generate explanations for model predictions
- `supports_model()`: Check compatibility with specific model
- `get_explanation_type()`: Return type (e.g., "shap", "lime")

### MetricInterface

Base protocol for evaluation metrics.

```python
class MetricInterface(Protocol):
    capabilities: dict[str, Any]
    version: str
    metric_type: str  # "performance", "fairness", "drift"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray,
                **kwargs) -> dict[str, Any]:
        """Compute the metric."""

    def requires_probabilities(self) -> bool:
        """Check if metric requires prediction probabilities."""

    def requires_sensitive_features(self) -> bool:
        """Check if metric requires sensitive features."""
```

**Required Attributes:**
- `capabilities`: Dict describing metric capabilities
- `version`: Implementation version
- `metric_type`: Category ("performance", "fairness", "drift")

**Methods:**
- `compute()`: Calculate metric value
- `requires_probabilities()`: Whether metric needs probability predictions
- `requires_sensitive_features()`: Whether metric needs demographic data

## Registry System

Dynamic component registration and selection system.

### ModelRegistry

Registry for model implementations.

```python
from glassalpha.core import ModelRegistry

# Register a model implementation
@ModelRegistry.register("custom_model")
class CustomModel:
    capabilities = {"supports_shap": True, "data_modality": "tabular"}
    version = "1.0.0"

    def predict(self, X):
        # Implementation
        pass

# Get registered models
available_models = ModelRegistry.get_all()

# Get specific model class
model_cls = ModelRegistry.get("xgboost")
```

**Methods:**
- `register(name)`: Decorator to register model implementations
- `get(name)`: Retrieve model class by name
- `get_all()`: Get all registered models
- `is_registered(name)`: Check if model is registered

### ExplainerRegistry

Registry for explanation methods.

```python
from glassalpha.core import ExplainerRegistry

@ExplainerRegistry.register("custom_explainer", priority=75)
class CustomExplainer:
    capabilities = {"model_types": ["custom_model"]}
    version = "1.0.0"
    priority = 75

    def explain(self, model, X, y=None):
        # Implementation
        pass

# Select best explainer for model
explainer_cls = ExplainerRegistry.select_best(model, priority_list=["shap", "lime"])
```

**Methods:**
- `register(name, priority=50)`: Register explainer with priority
- `select_best(model, priority_list)`: Select best compatible explainer
- `get_compatible(model)`: Get all explainers compatible with model

### MetricRegistry

Registry for evaluation metrics.

```python
from glassalpha.core import MetricRegistry

@MetricRegistry.register("custom_metric")
class CustomMetric:
    metric_type = "performance"
    capabilities = {"binary_classification": True}
    version = "1.0.0"

    def compute(self, y_true, y_pred, **kwargs):
        # Implementation
        pass

# Get metrics by type
performance_metrics = MetricRegistry.get_by_type("performance")
fairness_metrics = MetricRegistry.get_by_type("fairness")
```

**Methods:**
- `register(name)`: Register metric implementation
- `get_by_type(metric_type)`: Get metrics of specific type
- `get_all_types()`: Get available metric types

## Pipeline System

Core audit execution pipeline.

### AuditPipeline

Main orchestrator for audit execution.

```python
from glassalpha.pipeline import AuditPipeline
from glassalpha.config import AuditConfig

# Initialize pipeline
config = AuditConfig.from_yaml("audit_config.yaml")
pipeline = AuditPipeline(config)

# Execute audit
results = pipeline.run(progress_callback=lambda msg, pct: print(f"{msg}: {pct}%"))

print(f"Audit success: {results.success}")
print(f"Performance: {results.model_performance}")
print(f"Fairness: {results.fairness_analysis}")
```

**Constructor:**
- `AuditPipeline(config: AuditConfig)`: Initialize with configuration

**Methods:**
- `run(progress_callback=None)`: Execute complete audit pipeline
- Returns `AuditResults` object with comprehensive results

### AuditResults

Container for audit results and metadata.

```python
from dataclasses import dataclass

@dataclass
class AuditResults:
    # Core results
    model_performance: dict[str, Any]
    fairness_analysis: dict[str, Any]
    drift_analysis: dict[str, Any]
    explanations: dict[str, Any]

    # Data information
    data_summary: dict[str, Any]
    schema_info: dict[str, Any]

    # Model information
    model_info: dict[str, Any]
    selected_components: dict[str, Any]

    # Audit metadata
    execution_info: dict[str, Any]
    manifest: dict[str, Any]

    # Success indicators
    success: bool
    error_message: str | None
```

**Attributes:**
- `model_performance`: Performance metrics and evaluations
- `fairness_analysis`: Bias detection and fairness metrics
- `explanations`: SHAP values and feature importance
- `manifest`: Complete audit trail and reproducibility info
- `success`: Whether audit completed successfully

## Configuration System

Pydantic-based configuration management.

### AuditConfig

Main configuration schema.

```python
from glassalpha.config import AuditConfig

# Load from file
config = AuditConfig.from_yaml("config.yaml")

# Access configuration sections
print(config.audit_profile)  # "tabular_compliance"
print(config.model.type)     # "xgboost"
print(config.reproducibility.random_seed)  # 42

# Validate configuration
config.validate_strict_mode()
```

**Class Methods:**
- `from_yaml(path)`: Load configuration from YAML file
- `from_dict(data)`: Create from dictionary

**Properties:**
- `audit_profile`: Selected audit profile
- `model`: Model configuration
- `data`: Data configuration
- `explainers`: Explainer configuration
- `metrics`: Metrics configuration
- `reproducibility`: Reproducibility settings

### DataConfig

Data source and schema configuration.

```python
from glassalpha.config.schemas import DataConfig

data_config = DataConfig(
    path="data/dataset.csv",
    target_column="outcome",
    feature_columns=["feature1", "feature2"],
    protected_attributes=["gender", "age_group"]
)
```

**Attributes:**
- `path`: Path to dataset file
- `target_column`: Name of target/label column
- `feature_columns`: List of feature column names (optional)
- `protected_attributes`: Sensitive attributes for fairness analysis

### ModelConfig

Model specification and parameters.

```python
from glassalpha.config.schemas import ModelConfig

model_config = ModelConfig(
    type="xgboost",
    params={
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1
    }
)
```

**Attributes:**
- `type`: Model type identifier (must be registered)
- `path`: Optional path to pre-trained model
- `params`: Model hyperparameters

## Data Handling

Data loading and preprocessing utilities.

### TabularDataLoader

Main data loader for tabular datasets.

```python
from glassalpha.data import TabularDataLoader, TabularDataSchema

loader = TabularDataLoader()

# Load data with automatic format detection
data = loader.load("data/dataset.csv")

# Create schema
schema = TabularDataSchema(
    target="outcome",
    features=["feature1", "feature2"],
    sensitive_features=["gender"]
)

# Validate data against schema
loader.validate_schema(data, schema)

# Extract features and target
X, y, sensitive = loader.extract_features_target(data, schema)
```

**Methods:**
- `load(path, schema=None)`: Load data from file
- `validate_schema(data, schema)`: Validate data structure
- `extract_features_target(data, schema)`: Split into X, y, sensitive features
- `hash_data(data)`: Generate deterministic data hash
- `split_data(data, test_size, random_state)`: Train/test split

### TabularDataSchema

Schema definition for tabular data.

```python
from glassalpha.data import TabularDataSchema

schema = TabularDataSchema(
    target="credit_risk",
    features=["income", "debt_ratio", "employment_length"],
    sensitive_features=["gender", "age_group"],
    categorical_features=["employment_type", "housing"],
    numeric_features=["income", "debt_ratio"]
)
```

**Attributes:**
- `target`: Target column name
- `features`: Feature column names
- `sensitive_features`: Protected attributes
- `categorical_features`: Categorical feature names
- `numeric_features`: Numeric feature names

## Utilities

Helper functions and utilities.

### Seed Management

Centralized random seed management.

```python
from glassalpha.utils import set_global_seed, get_component_seed, with_component_seed

# Set master seed
set_global_seed(42)

# Get component-specific seed
explainer_seed = get_component_seed("explainer")

# Use seeded context manager
with with_component_seed("model"):
    # All randomness uses deterministic seed
    model.fit(X, y)
```

**Functions:**
- `set_global_seed(seed)`: Set master random seed
- `get_component_seed(component_name)`: Get deterministic seed for component
- `with_component_seed(component_name)`: Context manager for seeded execution
- `with_seed(seed)`: Context manager for specific seed

### Hashing

Deterministic content hashing.

```python
from glassalpha.utils import hash_config, hash_dataframe, hash_object

# Hash configuration
config_hash = hash_config(audit_config.model_dump())

# Hash dataset
data_hash = hash_dataframe(df)

# Hash arbitrary object
object_hash = hash_object({"key": "value"})
```

**Functions:**
- `hash_config(config_dict)`: Hash configuration dictionary
- `hash_dataframe(df)`: Hash pandas DataFrame
- `hash_object(obj)`: Hash arbitrary Python object
- `hash_file(path)`: Hash file contents

### Manifest Generation

Audit trail and manifest generation.

```python
from glassalpha.utils import AuditManifest, ManifestGenerator

# Generate manifest
generator = ManifestGenerator()
generator.add_config(config)
generator.add_data_info(data_hash, data_summary)
generator.add_model_info(model_info)

manifest = generator.generate()

# Save manifest
manifest_path = output_dir / "audit_manifest.json"
generator.save(manifest_path)
```

**Classes:**
- `ManifestGenerator`: Builder for audit manifests
- `AuditManifest`: Pydantic model for manifest data

**Methods:**
- `add_config()`: Add configuration information
- `add_data_info()`: Add dataset information
- `add_model_info()`: Add model information
- `generate()`: Create complete manifest
- `save()`: Save manifest to file

## CLI Integration

Command-line interface components.

### Main Application

```python
from glassalpha.cli import app
import typer

# Extend CLI with custom commands
@app.command()
def custom_command(
    input_file: str = typer.Option(..., help="Input file path")
):
    """Custom audit command."""
    print(f"Processing {input_file}")
```

### Available Commands

- `glassalpha audit`: Generate audit reports
- `glassalpha validate`: Validate configurations
- `glassalpha list`: List available components

## Extension Examples

### Custom Model Implementation

```python
from glassalpha.core import ModelRegistry, ModelInterface
import pandas as pd
import numpy as np

@ModelRegistry.register("my_model")
class MyCustomModel:
    capabilities = {
        "supports_shap": False,
        "supports_lime": True,
        "data_modality": "tabular"
    }
    version = "1.0.0"

    def __init__(self):
        self._model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        # Training implementation
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Prediction implementation
        return np.array([0, 1, 0])  # Example

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Probability prediction
        return np.array([[0.8, 0.2], [0.3, 0.7]])  # Example

    def get_model_type(self) -> str:
        return "my_model"

    def get_capabilities(self) -> dict:
        return self.capabilities
```

### Custom Explainer Implementation

```python
from glassalpha.core import ExplainerRegistry, ExplainerInterface

@ExplainerRegistry.register("my_explainer", priority=60)
class MyCustomExplainer:
    capabilities = {"model_types": ["my_model"]}
    version = "1.0.0"
    priority = 60

    def explain(self, model, X, y=None):
        # Generate explanations
        return {
            "feature_importance": {"feature1": 0.5, "feature2": 0.3},
            "explanation_type": "custom",
            "metadata": {"samples_explained": len(X)}
        }

    def supports_model(self, model):
        return model.get_model_type() == "my_model"

    def get_explanation_type(self):
        return "custom"
```

### Custom Metric Implementation

```python
from glassalpha.core import MetricRegistry, MetricInterface
import numpy as np

@MetricRegistry.register("my_metric")
class MyCustomMetric:
    metric_type = "performance"
    capabilities = {"binary_classification": True}
    version = "1.0.0"

    def compute(self, y_true, y_pred, **kwargs):
        # Custom metric calculation
        accuracy = np.mean(y_true == y_pred)

        return {
            "value": accuracy,
            "interpretation": "higher_is_better",
            "range": [0, 1],
            "description": "Custom accuracy metric"
        }

    def requires_probabilities(self):
        return False

    def requires_sensitive_features(self):
        return False
```

## Error Handling

### Exception Types

```python
from glassalpha.core import FeatureNotAvailable

try:
    # Enterprise feature usage
    from glassalpha.enterprise import AdvancedExplainer
except FeatureNotAvailable as e:
    print(f"Enterprise feature required: {e}")
    # Fallback to OSS alternative
```

### Common Exceptions

- `FeatureNotAvailable`: Enterprise feature without license
- `ValidationError`: Configuration validation failures
- `ComponentNotFound`: Registry lookup failures
- `ModelNotSupported`: Incompatible model/explainer combinations

## Version Information

```python
from glassalpha import __version__
print(f"GlassAlpha version: {__version__}")

# Check component versions
from glassalpha.core import ModelRegistry
xgb_cls = ModelRegistry.get("xgboost")
print(f"XGBoost wrapper version: {xgb_cls.version}")
```

This API reference provides comprehensive coverage of GlassAlpha's public interfaces, enabling developers to extend functionality, integrate with existing systems, and build custom audit workflows.
