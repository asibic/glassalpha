# Installation Guide

Complete installation instructions for GlassAlpha on different platforms and environments.

## System Requirements

### Minimum Requirements

- **Python**: 3.11 or higher
- **Memory**: 2GB RAM available
- **Storage**: 1GB disk space for installation and temporary files
- **OS**: macOS 10.15+, Linux (Ubuntu 20.04+), Windows 10+ (WSL2 recommended)

### Recommended Environment

- **Python**: 3.11+
- **Memory**: 8GB+ RAM for large datasets
- **Storage**: SSD for better performance
- **CPU**: Multi-core processor for parallel processing

### Supported Platforms

- **macOS**: Intel and Apple Silicon (M1/M2/M3)
- **Linux**: Ubuntu 20.04+, CentOS 8+, RHEL 8+, and most modern distributions
- **Windows**: 10/11 (native support, WSL2 recommended for best experience)

## Installation Methods

### Standard Installation (Recommended)

GlassAlpha is currently distributed as source code via GitHub. PyPI distribution may be available in potential future releases.

```bash
# Clone the repository
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages

# Create and activate virtual environment (recommended)
python -m venv glassalpha-env
source glassalpha-env/bin/activate  # On Windows: glassalpha-env\Scripts\activate

# Install GlassAlpha and dependencies
pip install -e .
```

### Development Installation

For contributors or those who need development tools:

```bash
# Clone the repository
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages

# Create virtual environment
python -m venv dev-env
source dev-env/bin/activate  # On Windows: dev-env\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

Development dependencies include:

- **Testing**: pytest, pytest-cov
- **Code Quality**: ruff, mypy, black, pre-commit
- **Documentation**: mkdocs, mkdocs-material

## Platform-specific installation

### macOS installation

**Prerequisites:**

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11+ (if not already available)
brew install python@3.11

# Install OpenMP for XGBoost support
brew install libomp
```

**Installation:**

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages

python3.11 -m venv glassalpha-env
source glassalpha-env/bin/activate
pip install --upgrade pip
pip install -e .
```

**Common macOS Issues:**

- **XGBoost libomp error**: Install with `brew install libomp`
- **Python version conflicts**: Use `python3.11` explicitly
- **Permission issues**: Avoid `sudo pip install`, use virtual environments

### Linux installation (Ubuntu/Debian)

**Prerequisites:**

```bash
# Update package list
sudo apt update

# Install Python 3.11 and development tools
sudo apt install python3.11 python3.11-venv python3.11-dev
sudo apt install build-essential git

# Install system libraries for PDF generation
sudo apt install libpango1.0-dev libcairo2-dev libgtk-3-dev
```

**Installation:**

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages

python3.11 -m venv glassalpha-env
source glassalpha-env/bin/activate
pip install --upgrade pip
pip install -e .
```

### Linux installation (CentOS/RHEL)

**Prerequisites:**

```bash
# Install EPEL repository
sudo yum install epel-release

# Install Python 3.11 and development tools
sudo yum install python3.11 python3.11-devel python3.11-pip
sudo yum install gcc gcc-c++ make git

# Install system libraries
sudo yum install pango-devel cairo-devel gtk3-devel
```

**Installation:**

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages

python3.11 -m venv glassalpha-env
source glassalpha-env/bin/activate
pip install --upgrade pip
pip install -e .
```

### Windows installation

**Option 1: Windows Subsystem for Linux (Recommended)**

```bash
# Install WSL2 and Ubuntu
wsl --install Ubuntu

# In WSL2, follow Ubuntu installation instructions
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages
python3.11 -m venv glassalpha-env
source glassalpha-env/bin/activate
pip install -e .
```

**Option 2: Native Windows**

```powershell
# Install Python 3.11+ from python.org
# Install Git from git-scm.com

# Clone and install
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha\packages

python -m venv glassalpha-env
glassalpha-env\Scripts\activate
pip install --upgrade pip
pip install -e .
```

**Windows notes:**

- WSL2 provides better compatibility and performance
- Native Windows may have PDF generation limitations
- Use PowerShell or Command Prompt for native installation

## Verification and Testing

### Basic Verification

After installation, verify GlassAlpha is working correctly:

```bash
# Check version
glassalpha --version

# Verify CLI is working
glassalpha --help

# List available components
glassalpha list
```

Expected output:

```
GlassAlpha version 0.1.0

Available Components
==================
MODELS:
  - xgboost
  - lightgbm
  - logistic_regression
  - sklearn_generic
  - passthrough

EXPLAINERS:
  - treeshap
  - kernelshap
  - noop

METRICS:
  - accuracy
  - precision
  - recall
  - f1
  - auc_roc
  - demographic_parity
  - equal_opportunity
  ...
```

### Python API Verification

Test the Python API:

```python
# Verify imports work
from glassalpha.core import ModelRegistry, ExplainerRegistry
from glassalpha.pipeline import AuditPipeline
from glassalpha.config import AuditConfig

# Check registries
print("Available models:", list(ModelRegistry.get_all().keys()))
print("Available explainers:", list(ExplainerRegistry.get_all().keys()))

# Verify configuration loading
print("Configuration system working!")
```

### End-to-End Test

Run a complete audit to verify all components:

```bash
# Quick smoke test with German Credit dataset
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --output test_audit.pdf \
  --dry-run

# If dry-run passes, run actual audit
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --output test_audit.pdf
```

Successful execution should:

- Complete in under 30 seconds
- Generate a PDF report (~500KB+)
- Show no error messages
- Display audit summary statistics

## Dependencies

### Core Dependencies

GlassAlpha automatically installs these required dependencies:

**Data Processing:**

- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning utilities

**Machine Learning:**

- `xgboost>=1.7.0` - Gradient boosting framework
- `lightgbm>=3.3.0` - Microsoft's gradient boosting
- `shap>=0.42.0` - Model explanations

**Visualization & Reporting:**

- `matplotlib>=3.7.0` - Plotting and visualization
- `seaborn>=0.12.0` - Statistical visualization
- `weasyprint>=59.0` - PDF generation

**Configuration & CLI:**

- `pydantic>=2.0.0` - Configuration validation
- `typer>=0.9.0` - Command-line interface
- `pyyaml>=6.0` - YAML configuration files

### Optional Dependencies

**Development Tools** (with `[dev]` install):

- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Test coverage
- `ruff>=0.0.280` - Code linting and formatting
- `mypy>=1.5.0` - Type checking
- `pre-commit>=3.3.0` - Git hooks

### Version Compatibility

GlassAlpha is tested with:

- **Python**: 3.11, 3.12
- **scikit-learn**: 1.3.x, 1.4.x
- **XGBoost**: 1.7.x, 2.0.x
- **LightGBM**: 3.3.x, 4.0.x
- **SHAP**: 0.42.x, 0.43.x

## Environment Management

### Using Virtual Environments

**Why Virtual Environments?**

- Isolate dependencies from system Python
- Avoid version conflicts
- Enable project-specific configurations
- Facilitate deployment and distribution

**Creating Virtual Environments:**

```bash
# Using venv (built-in)
python -m venv glassalpha-env
source glassalpha-env/bin/activate  # Linux/macOS
# glassalpha-env\Scripts\activate   # Windows

# Using conda
conda create -n glassalpha python=3.11
conda activate glassalpha

# Using poetry
poetry init
poetry add glassalpha
poetry shell
```

### Environment Variables

**Configuration:**

- `GLASSALPHA_CONFIG_DIR` - Default configuration directory
- `GLASSALPHA_DATA_DIR` - Default data directory
- `GLASSALPHA_LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

**Example:**

```bash
export GLASSALPHA_LOG_LEVEL=DEBUG
export GLASSALPHA_CONFIG_DIR=~/.config/glassalpha
glassalpha audit --config my_config.yaml --output audit.pdf
```

## Troubleshooting Installation

### Common Installation Issues

**Python Version Issues:**

```bash
# Check Python version
python --version

# If version is too old, install Python 3.11+
# Use python3.11 explicitly if multiple versions installed
python3.11 -m venv glassalpha-env
```

**Permission Errors:**

```bash
# Use virtual environment instead of system-wide installation
python -m venv glassalpha-env
source glassalpha-env/bin/activate
pip install -e .

# Or install for user only (not recommended)
pip install --user -e .
```

**Dependency Conflicts:**

```bash
# Create clean environment
rm -rf glassalpha-env
python -m venv glassalpha-env
source glassalpha-env/bin/activate
pip install --upgrade pip
pip install -e .
```

**XGBoost Issues (macOS):**

```bash
# Install OpenMP library
brew install libomp

# Reinstall XGBoost
pip uninstall xgboost
pip install xgboost

# Verify installation
python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"
```

**PDF Generation Issues (Linux):**

```bash
# Install system libraries
sudo apt install libpango1.0-dev libcairo2-dev libgtk-3-dev  # Ubuntu/Debian
sudo yum install pango-devel cairo-devel gtk3-devel          # CentOS/RHEL

# Reinstall WeasyPrint
pip uninstall weasyprint
pip install weasyprint
```

### Memory and Performance Issues

**Insufficient Memory:**

```bash
# Monitor memory usage during installation
pip install -e . --verbose

# If installation fails due to memory, increase swap space (Linux)
sudo swapon --show
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Slow Installation:**

```bash
# Use faster package index
pip install -e . --index-url https://pypi.org/simple/

# Install binary wheels when available
pip install -e . --only-binary=all --prefer-binary
```

### Verification Failures

**CLI Command Not Found:**

```bash
# Verify installation
pip list | grep glassalpha

# Check PATH contains virtual environment
which glassalpha
echo $PATH

# Reinstall if necessary
pip install --force-reinstall -e .
```

**Import Errors:**

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Verify installation location
pip show glassalpha

# Test imports individually
python -c "import pandas; print('pandas OK')"
python -c "import xgboost; print('xgboost OK')"
python -c "import glassalpha; print('glassalpha OK')"
```

## Docker Installation (Alternative)

For containerized environments:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libpango1.0-dev \
    libcairo2-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and install GlassAlpha
RUN git clone https://github.com/GlassAlpha/glassalpha.git
WORKDIR /glassalpha/packages
RUN pip install -e .

# Set entry point
ENTRYPOINT ["glassalpha"]
```

**Usage:**

```bash
# Build image
docker build -t glassalpha .

# Run audit
docker run -v $(pwd):/data glassalpha \
  audit --config /data/config.yaml --output /data/audit.pdf
```

## Next Steps

After successful installation:

1. **Try the Quick Start** - [Generate your first audit in 10 minutes](../getting-started/quickstart.md)
2. **Explore Examples** - [German Credit tutorial](../examples/german-credit-audit.md)
3. **Read Configuration Guide** - [Understand all options](../getting-started/configuration.md)
4. **Production Deployment** - [Enterprise deployment guide](../reference/deployment.md) for regulated environments
5. **Join the Community** - [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)

## Getting Help

If you encounter installation issues not covered here:

1. **Check the [Troubleshooting Guide](../reference/troubleshooting.md)**
2. **Search [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)**
3. **Ask in [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)**
4. **Review the [FAQ](../reference/faq.md)**

For enterprise support and custom installation assistance, contact: enterprise@glassalpha.com

This installation guide ensures you have a working GlassAlpha environment ready for professional ML auditing and compliance analysis.
