# Glass Alpha Package Structure

## Overview

Glass Alpha uses a dual-package architecture to maintain clear separation between open-source and enterprise features. During Phase 1, we use feature flags for separation. In Phase 2, we'll split into separate packages.

## Current Structure (Phase 1 - Feature Flags)

```
glassalpha/packages/
├── pyproject.toml              # Single package with feature flags
├── src/
│   └── glassalpha/            # All code in one package
│       ├── core/              # OSS: Core interfaces and registries
│       ├── models/            # OSS: Basic model wrappers
│       ├── explain/           # OSS: TreeSHAP, KernelSHAP
│       ├── metrics/           # OSS: Basic metrics
│       ├── report/            # OSS: Basic PDF generation
│       ├── config/            # OSS: Configuration system
│       ├── cli/               # Mixed: CLI with enterprise stubs
│       └── profiles/          # OSS: Audit profiles
└── tests/                     # All tests
```

Enterprise features are gated using `@check_feature` decorator.

## Future Structure (Phase 2+ - Package Split)

```
glassalpha/                    # Main repository (OSS)
├── packages/
│   ├── glassalpha/            # OSS package (Apache 2.0)
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   └── glassalpha/
│   │   │       ├── core/      # Interfaces, registries
│   │   │       ├── models/    # Basic wrappers
│   │   │       ├── explain/   # TreeSHAP, KernelSHAP
│   │   │       ├── metrics/   # Basic metrics
│   │   │       └── report/    # Basic templates
│   │   └── tests/
│   │
│   └── glassalpha-enterprise/ # Enterprise package (Commercial)
│       ├── pyproject.toml     # Depends on glassalpha
│       ├── LICENSE            # Commercial license
│       ├── src/
│       │   └── glassalpha_enterprise/
│       │       ├── explain/   # DeepSHAP, GradientSHAP
│       │       ├── dashboard/ # Monitoring UI
│       │       ├── templates/ # Regulatory templates
│       │       ├── integrations/ # Cloud connectors
│       │       └── policy/    # Policy packs
│       └── tests/
```

## Package Dependencies

### OSS Package (`glassalpha`)
```toml
[project]
name = "glassalpha"
version = "1.0.0"
license = {text = "Apache-2.0"}
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0", 
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "shap>=0.43.0",
    "typer>=0.9.0",
    "pydantic>=2.5.0",
    # ... other OSS deps
]
```

### Enterprise Package (`glassalpha-enterprise`)
```toml
[project]
name = "glassalpha-enterprise"
version = "1.0.0"
license = {text = "Commercial"}
dependencies = [
    "glassalpha>=1.0.0",  # Depends on OSS
    "plotly>=5.0.0",      # Advanced visualizations
    "dash>=2.0.0",        # Dashboard
    "sqlalchemy>=2.0.0",  # State management
    # ... enterprise-specific deps
]
```

## Migration Path

### Phase 1 → Phase 2 Migration Steps

1. **Create separate package directories**
   ```bash
   mkdir -p packages/glassalpha-enterprise/src/glassalpha_enterprise
   ```

2. **Move enterprise features**
   - Identify all `@check_feature` decorated functions
   - Move to enterprise package
   - Update imports to use `glassalpha_enterprise`

3. **Update registrations**
   ```python
   # In enterprise package
   from glassalpha.core import ExplainerRegistry
   
   @ExplainerRegistry.register("deep_shap", enterprise=True)
   class DeepSHAPExplainer:
       ...
   ```

4. **Update CLI**
   ```python
   # Enterprise commands in separate module
   from glassalpha.cli import app
   from glassalpha_enterprise.cli import add_enterprise_commands
   
   if is_enterprise():
       add_enterprise_commands(app)
   ```

## Import Structure

### OSS Users
```python
from glassalpha import audit, explain, report
from glassalpha.models import XGBoostWrapper
from glassalpha.explain import TreeSHAP
```

### Enterprise Users
```python
from glassalpha import audit  # Still use OSS base
from glassalpha_enterprise.explain import DeepSHAP
from glassalpha_enterprise.dashboard import serve_dashboard
from glassalpha_enterprise.templates import EUAIActTemplate
```

## Feature Flag Migration

### Current (Phase 1)
```python
# In OSS package
@check_feature("deep_shap")
def deep_shap_explain():
    raise FeatureNotAvailable("Install glassalpha-enterprise")
```

### Future (Phase 2)
```python
# Not in OSS at all
# Only in glassalpha_enterprise.explain.deep
def deep_shap_explain():
    # Full implementation
    ...
```

## Installation

### OSS Installation
```bash
pip install glassalpha
```

### Enterprise Installation (Phase 1)
```bash
pip install glassalpha
export GLASSALPHA_LICENSE_KEY="your-key"
```

### Enterprise Installation (Phase 2+)
```bash
# Private PyPI or GitHub repo
pip install glassalpha
pip install glassalpha-enterprise --index-url https://pypi.glassalpha.ai
```

## License Verification

### Phase 1 (Environment Variable)
```python
def is_enterprise():
    return bool(os.environ.get("GLASSALPHA_LICENSE_KEY"))
```

### Phase 2+ (License Server)
```python
def is_enterprise():
    license_key = os.environ.get("GLASSALPHA_LICENSE_KEY")
    if not license_key:
        return False
    
    # Validate with license server
    return validate_license(license_key)
```

## Testing Strategy

### OSS Tests
- Run without any enterprise features
- Ensure all core functionality works
- Test that enterprise features fail gracefully

### Enterprise Tests
- Run with license key set
- Test all enterprise features
- Test integration with OSS components

### CI/CD
```yaml
# .github/workflows/test.yml
jobs:
  test-oss:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/ -m "not enterprise"
  
  test-enterprise:
    runs-on: ubuntu-latest
    env:
      GLASSALPHA_LICENSE_KEY: ${{ secrets.LICENSE_KEY }}
    steps:
      - run: pytest tests/ -m "enterprise"
```

## Documentation

### OSS Documentation
- Public GitHub Pages
- Focus on core features
- Links to enterprise features (marketing)

### Enterprise Documentation
- Private documentation site
- Requires authentication
- Includes all features
- Priority support contact

## Release Process

### OSS Releases
1. Version bump in `pyproject.toml`
2. Update CHANGELOG.md
3. Tag release on GitHub
4. Publish to PyPI

### Enterprise Releases
1. Ensure compatibility with latest OSS
2. Version bump (can be independent)
3. Update enterprise CHANGELOG
4. Publish to private repository
5. Notify enterprise customers

## Benefits of This Structure

1. **Clear Boundaries**: OSS users can't accidentally use enterprise features
2. **Independent Releases**: Can update OSS and enterprise separately
3. **License Protection**: Enterprise code never in public repository
4. **Easy Migration**: Users can upgrade from OSS to enterprise easily
5. **Testing Isolation**: Can test OSS and enterprise independently

## Current Status

✅ **Phase 1 Implementation Complete:**
- Feature flags implemented (`core/features.py`)
- Registry supports enterprise metadata
- CLI has enterprise command stubs
- Tests verify feature gating

⏳ **Phase 2 (Future):**
- Physical package separation
- Private PyPI server
- License validation server
- Automated enterprise builds
