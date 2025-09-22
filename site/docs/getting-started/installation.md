# Installation

Glass Alpha can be installed via pip or from source. The library requires Python 3.11 or higher.

## Requirements

- Python 3.11+
- pip or conda
- (Optional) Virtual environment tool (venv, conda, poetry)

## Standard Installation

### Using pip

```bash
pip install glassalpha
```

### Using conda

```bash
conda install -c conda-forge glassalpha
```

## Development Installation

For contributors or those who need the latest development version:

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

## Optional Dependencies

Glass Alpha has optional dependencies for specific features:

### Visualization
```bash
pip install glassalpha[viz]
```

### Deep Learning Support (Future)
```bash
pip install glassalpha[deep]
```

### All Optional Dependencies
```bash
pip install glassalpha[all]
```

## Verifying Installation

After installation, verify everything is working:

```python
import glassalpha
print(glassalpha.__version__)

# Run basic smoke test
from glassalpha import Explainer
print("Glass Alpha successfully installed!")
```

## Platform-Specific Notes

### macOS
- Requires macOS 11+ for M1/M2 chips
- XGBoost may require additional setup for GPU support

### Linux
- Tested on Ubuntu 20.04+ and RHEL 8+
- May require additional system libraries for visualization

### Windows
- Fully supported on Windows 10/11
- Use Anaconda for easiest setup

## Troubleshooting

### Common Issues

**ImportError: No module named 'glassalpha'**
- Ensure you've activated your virtual environment
- Check pip list to verify installation

**Version conflicts**
```bash
pip install --upgrade glassalpha
pip install --upgrade --force-reinstall glassalpha
```

**Permission errors**
```bash
pip install --user glassalpha
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Example Audits](../examples/german-credit-audit.md)
